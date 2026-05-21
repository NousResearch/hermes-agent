from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.codex_worker import (
    build_codex_argv,
    parse_progress_items,
    run_codex_worker,
    _safe_env_for_codex,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.worker_lanes import (
    WorkerLane,
    clear_worker_lanes,
    enable_worker_lane_request,
    get_worker_lane,
    list_worker_lanes,
    register_configured_worker_lanes,
    register_worker_lane,
    resolve_worker_assignee,
    validate_worker_lane_request,
)


@pytest.fixture(autouse=True)
def clean_lanes():
    clear_worker_lanes()
    yield
    clear_worker_lanes()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _lane(name="codex-deep", *, pid=1234, max_concurrency=None):
    def spawn(task, workspace, *, board=None):
        return pid

    return WorkerLane(
        name=name,
        kind="codex_cli",
        description="fake lane",
        spawn_fn=spawn,
        success_policy="block_for_review",
        max_concurrency=max_concurrency,
        source="test",
    )


def test_worker_lane_registry_register_query_and_conflict():
    lane = register_worker_lane(_lane("codex-deep"))
    assert get_worker_lane("CODEX-DEEP") is lane
    assert [x.name for x in list_worker_lanes()] == ["codex-deep"]
    with pytest.raises(ValueError, match="already registered"):
        register_worker_lane(_lane("codex-deep"))


def test_resolve_worker_assignee_prefers_lane_over_profile(monkeypatch):
    from hermes_cli import profiles

    register_worker_lane(_lane("daily"))
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    res = resolve_worker_assignee("daily")
    assert res.kind == "worker_lane"
    assert res.lane is not None


def test_plugin_context_register_worker_lane():
    mgr = PluginManager()
    ctx = PluginContext(PluginManifest(name="worker-plugin"), mgr)

    def spawn(task, workspace, *, board=None):
        return 9

    ctx.register_worker_lane(
        name="plugin-codex",
        kind="plugin",
        description="plugin lane",
        spawn_fn=spawn,
    )
    lane = get_worker_lane("plugin-codex")
    assert lane is not None
    assert lane.source == "plugin:worker-plugin"
    assert "plugin-codex" in mgr._plugin_worker_lane_names


def test_plugin_context_worker_lane_failure_is_logged(caplog):
    mgr = PluginManager()
    ctx = PluginContext(PluginManifest(name="broken-plugin"), mgr)
    ctx.register_worker_lane(name="bad lane", spawn_fn=None)
    assert get_worker_lane("bad lane") is None
    assert any("failed to register worker lane" in r.message for r in caplog.records)


def test_config_registers_multiple_codex_lanes():
    register_configured_worker_lanes({
        "kanban": {
            "worker_lanes": {
                "codex-fast": {
                    "type": "codex_cli",
                    "model": "gpt-5.4-mini",
                    "sandbox": "workspace-write",
                    "approval": "never",
                    "max_concurrency": 2,
                },
                "codex-deep": {
                    "type": "codex_cli",
                    "model": "gpt-5.5",
                    "sandbox": "workspace-write",
                    "approval": "never",
                    "max_concurrency": 1,
                },
            }
        }
    })
    lanes = {lane.name: lane for lane in list_worker_lanes()}
    assert set(lanes) == {"codex-deep", "codex-fast"}
    assert lanes["codex-fast"].max_concurrency == 2
    assert lanes["codex-deep"].config["model"] == "gpt-5.5"


def test_lane_request_validator_rejects_shell_command():
    with pytest.raises(ValueError, match="command"):
        validate_worker_lane_request({
            "name": "codex-unsafe",
            "type": "codex_cli",
            "command": "rm -rf /",
        })


def test_enable_worker_lane_request_registers_sanitized_lane():
    lane = enable_worker_lane_request({
        "name": "codex-long-context",
        "type": "codex_cli",
        "model": "gpt-5.5",
        "sandbox": "workspace-write",
        "approval": "never",
        "max_concurrency": 1,
        "success_policy": "block_for_review",
        "reason": "large refactor",
    })

    assert lane.name == "codex-long-context"
    assert lane.kind == "codex_cli"
    assert lane.source == "lane_request"
    assert get_worker_lane("codex-long-context") is lane
    assert resolve_worker_assignee("codex-long-context", refresh_config=False).kind == "worker_lane"


def test_enable_worker_lane_request_can_persist_sanitized_config(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    lane = enable_worker_lane_request(
        {
            "name": "codex-approved",
            "type": "codex_cli",
            "model": "gpt-5.4-mini",
            "sandbox": "workspace-write",
            "approval": "never",
            "max_concurrency": 2,
            "success_policy": "block_for_review",
            "reason": "operator approved",
        },
        persist=True,
    )

    assert lane.source == "config"
    from hermes_cli.config import read_raw_config

    raw = read_raw_config()
    stored = raw["kanban"]["worker_lanes"]["codex-approved"]
    assert stored["type"] == "codex_cli"
    assert stored["model"] == "gpt-5.4-mini"
    assert stored["max_concurrency"] == 2
    assert "reason" not in stored
    assert "command" not in stored


def test_dispatcher_uses_external_lane_assignee(kanban_home, monkeypatch):
    from hermes_cli import profiles

    calls = []

    def spawn(task, workspace, *, board=None):
        calls.append((task.id, task.assignee, workspace, board))
        return 4321

    register_worker_lane(WorkerLane(
        name="codex-deep",
        kind="codex_cli",
        description="fake",
        spawn_fn=spawn,
        max_concurrency=2,
    ))
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="external", assignee="codex-deep")
        res = kb.dispatch_once(conn)
        task = kb.get_task(conn, tid)
    assert res.spawned == [(tid, "codex-deep", calls[0][2])]
    assert task.status == "running"
    assert task.worker_pid == 4321
    assert calls[0][1] == "codex-deep"


def test_unregistered_assignee_still_skipped_nonspawnable(kanban_home, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="terminal", assignee="orion-cc")
        res = kb.dispatch_once(conn, dry_run=True)
    assert tid in res.skipped_nonspawnable
    assert not res.spawned


def test_scheduled_tasks_are_not_dispatchable_for_external_lane(kanban_home, monkeypatch):
    from hermes_cli import profiles

    register_worker_lane(_lane("codex-deep"))
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="later", assignee="codex-deep")
        assert kb.schedule_task(conn, tid, reason="wait for clock")
        res = kb.dispatch_once(conn, dry_run=True)
        task = kb.get_task(conn, tid)
    assert task.status == "scheduled"
    assert not res.spawned


def test_hermes_profile_lane_behavior_unchanged(kanban_home, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "worker")
    calls = []

    def spawn(task, workspace):
        calls.append((task.id, task.assignee, workspace))
        return 77

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="profile", assignee="worker")
        res = kb.dispatch_once(conn, spawn_fn=spawn)
        task = kb.get_task(conn, tid)
    assert res.spawned[0][0] == tid
    assert task.worker_pid == 77
    assert calls[0][1] == "worker"


def test_lane_max_concurrency_and_instances_are_distinct(kanban_home):
    calls = []

    def spawn(task, workspace, *, board=None):
        calls.append(task.id)
        return 9000 + len(calls)

    register_worker_lane(WorkerLane(
        name="codex-fast",
        kind="codex_cli",
        description="fake",
        spawn_fn=spawn,
        max_concurrency=2,
    ))
    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="a", assignee="codex-fast")
        t2 = kb.create_task(conn, title="b", assignee="codex-fast")
        t3 = kb.create_task(conn, title="c", assignee="codex-fast")
        res = kb.dispatch_once(conn, max_spawn=10)
        task1 = kb.get_task(conn, t1)
        task2 = kb.get_task(conn, t2)
        task3 = kb.get_task(conn, t3)
    assert calls == [t1, t2]
    assert task1.worker_pid != task2.worker_pid
    assert task3.status == "ready"
    assert t3 in res.skipped_concurrency


def test_review_profile_lane_behavior_unchanged(kanban_home, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "reviewer")
    spawned = []

    def spawn(task, workspace):
        spawned.append(task)
        return 88

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review", assignee="reviewer")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        res = kb.dispatch_once(conn, spawn_fn=spawn)
        task = kb.get_task(conn, tid)
    assert res.spawned[0][0] == tid
    assert task.status == "running"
    assert task.worker_pid == 88
    assert spawned[0].skills == ["sdlc-review"]


def test_review_external_lane_dispatches_without_profile_review_skill(kanban_home, monkeypatch):
    from hermes_cli import profiles

    calls = []

    def spawn(task, workspace, *, board=None):
        calls.append((task.id, task.assignee, task.skills))
        return 501

    register_worker_lane(WorkerLane(
        name="codex-review",
        kind="codex_cli",
        description="fake review external lane",
        spawn_fn=spawn,
        max_concurrency=1,
    ))
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review externally", assignee="codex-review")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        res = kb.dispatch_once(conn)
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
    assert res.spawned[0][0] == tid
    assert calls == [(tid, "codex-review", None)]
    assert task.status == "running"
    assert task.worker_pid == 501
    spawned_events = [e for e in events if e.kind == "spawned"]
    assert spawned_events[-1].payload["worker_lane"] == "codex-review"


def _claim_for_codex(conn, title="codex task"):
    tid = kb.create_task(
        conn,
        title=title,
        body="Edit the repository and report progress.",
        assignee="codex-deep",
        workspace_kind="dir",
        workspace_path=os.getcwd(),
    )
    task = kb.claim_task(conn, tid, claimer="host:test")
    assert task is not None
    return tid, kb.get_task(conn, tid)


def _make_fake_codex(tmp_path: Path, body: str, *, exit_code: int = 0) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "codex"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "_ = sys.stdin.read()\n"
        f"sys.stdout.write({body!r})\n"
        "sys.stdout.flush()\n"
        f"sys.exit({int(exit_code)})\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def test_codex_argv_model_parameter():
    argv = build_codex_argv(
        binary="/usr/bin/codex",
        workspace="/tmp/ws",
        sandbox="workspace-write",
        approval="never",
        model="gpt-5.5",
    )
    assert argv == [
        "/usr/bin/codex",
        "--cd",
        "/tmp/ws",
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
        "--model",
        "gpt-5.5",
        "exec",
        "-",
    ]


def test_codex_env_preserves_existing_writable_codex_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)

    env = _safe_env_for_codex(str(workspace))

    assert env["HOME"] == str(home)
    assert env.get("CODEX_HOME") is None


def test_codex_env_uses_workspace_home_when_inherited_home_unwritable(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "blocked-home"))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(
        "hermes_cli.codex_worker._path_is_writable_dir",
        lambda path: False,
    )

    env = _safe_env_for_codex(str(workspace))

    assert env["HOME"] == str(workspace / ".hermes-codex-home")
    assert env["CODEX_HOME"] == str(workspace / ".hermes-codex")


def test_progress_parser_supports_ordinals_and_checkboxes():
    items = parse_progress_items(
        "o (1) 分析入口\nx (2) 修改 dispatcher\n- [ ] 补测试\n- [x] 完成文档\n"
    )
    assert items[:2] == [
        {"index": 1, "status": "done", "text": "分析入口"},
        {"index": 2, "status": "running", "text": "修改 dispatcher"},
    ]
    assert {"index": 1, "status": "pending", "text": "补测试"} in items
    assert {"index": 2, "status": "done", "text": "完成文档"} in items


def test_progress_parser_ignores_template_placeholders():
    assert parse_progress_items(
        "Progress:\n- [x] ...\n- [ ] ...\no (1) ...\nx (2) ...\n"
    ) == []


def test_progress_parser_deduplicates_repeated_items():
    items = parse_progress_items(
        "Progress:\n- [ ] Create smoke_result.txt\n"
        "Progress:\n- [x] Create smoke_result.txt\n"
    )
    assert items == [
        {"index": 1, "status": "done", "text": "Create smoke_result.txt"}
    ]


def test_codex_binary_missing_blocks_with_metadata(kanban_home, monkeypatch):
    monkeypatch.setenv("PATH", "/tmp/definitely-no-codex")
    with kb.connect() as conn:
        tid, task = _claim_for_codex(conn)
        run_id = task.current_run_id
        assert run_id is not None
    rc = run_codex_worker(
        task_id=tid,
        lane="codex-deep",
        workspace=os.getcwd(),
        sandbox="workspace-write",
        approval="never",
        run_id=run_id,
        claim_lock=task.claim_lock,
        heartbeat_interval=0.01,
    )
    assert rc == 0
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        events = kb.list_events(conn, tid)
    assert task.status == "blocked"
    assert run.metadata["worker_lane"]["binary_missing"] is True
    assert run.metadata["review"]["required"] is False
    assert "codex binary not found" in run.summary
    assert any(e.kind == "worker_failed" for e in events)


def test_codex_exit_zero_blocks_for_review_and_records_progress_metadata(
    kanban_home, tmp_path, monkeypatch,
):
    old_path = os.environ.get("PATH", "")
    fake_bin = _make_fake_codex(
        tmp_path,
        "o (1) 分析入口\nx (2) 修改 dispatcher\n"
        "Progress:\n- [x] 分析入口\n- [ ] 补测试\n"
        "Changed files:\n- hermes_cli/kanban_db.py\n"
        "Verification:\n- command: pytest fake\n  result: passed\n",
    )
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + old_path)
    with kb.connect() as conn:
        tid, task = _claim_for_codex(conn)
        run_id = task.current_run_id
    rc = run_codex_worker(
        task_id=tid,
        lane="codex-deep",
        workspace=os.getcwd(),
        sandbox="workspace-write",
        approval="never",
        model="gpt-5.5",
        run_id=run_id,
        claim_lock=task.claim_lock,
        heartbeat_interval=0.01,
    )
    assert rc == 0
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        events = kb.list_events(conn, tid)
        log = kb.read_worker_log(tid)
    assert task.status == "blocked"
    assert run.outcome == "blocked"
    assert run.summary.startswith("review-required:")
    assert run.metadata["worker_instance"]["worker_lane"] == "codex-deep"
    assert run.metadata["worker_instance"]["run_id"] == run_id
    assert run.metadata["worker_lane"]["exit_code"] == 0
    assert run.metadata["review"]["required"] is True
    assert "hermes_cli/kanban_db.py" in run.metadata["worker_lane"]["output_tail"]
    assert run.metadata["verification"]["commands"] == ["pytest fake"]
    assert any(e.kind == "heartbeat" for e in events)
    progress = [e for e in events if e.kind == "worker_progress"]
    assert progress
    assert progress[-1].payload["lane"] == "codex-deep"
    assert any(item["text"] == "修改 dispatcher" for item in progress[-1].payload["items"])
    assert "[codex-worker]" in log


def test_codex_metadata_ignores_prompt_template_verification(
    kanban_home, tmp_path, monkeypatch,
):
    old_path = os.environ.get("PATH", "")
    fake_bin = _make_fake_codex(
        tmp_path,
        "Progress:\n- [x] ...\n- [ ] ...\n"
        "Verification:\n- command: ...\n  result: ...\n\n"
        "Progress:\n- [x] Create smoke_result.txt\n\n"
        "Verification:\n"
        "- command: `cmp -s smoke_result.txt <(printf 'codex worker lane smoke ok\\n') && echo exact`\n"
        "  result: `exact`\n",
    )
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + old_path)
    with kb.connect() as conn:
        tid, task = _claim_for_codex(conn)
        run_id = task.current_run_id
    run_codex_worker(
        task_id=tid,
        lane="codex-smoke",
        workspace=os.getcwd(),
        sandbox="workspace-write",
        approval="never",
        run_id=run_id,
        claim_lock=task.claim_lock,
        heartbeat_interval=0.01,
    )
    with kb.connect() as conn:
        run = kb.latest_run(conn, tid)
        events = kb.list_events(conn, tid)
    assert run.metadata["verification"]["commands"] == [
        "cmp -s smoke_result.txt <(printf 'codex worker lane smoke ok\\n') && echo exact"
    ]
    assert "result: `exact`" in run.metadata["verification"]["summary"]
    progress = [e for e in events if e.kind == "worker_progress"]
    assert progress[-1].payload["items"] == [
        {"index": 1, "status": "done", "text": "Create smoke_result.txt"}
    ]


def test_codex_exit_nonzero_blocks_failed(kanban_home, tmp_path, monkeypatch):
    old_path = os.environ.get("PATH", "")
    fake_bin = _make_fake_codex(tmp_path, "boom\n", exit_code=7)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + old_path)
    with kb.connect() as conn:
        tid, task = _claim_for_codex(conn)
        run_id = task.current_run_id
    run_codex_worker(
        task_id=tid,
        lane="codex-fast",
        workspace=os.getcwd(),
        sandbox="workspace-write",
        approval="never",
        run_id=run_id,
        claim_lock=task.claim_lock,
        heartbeat_interval=0.01,
    )
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
    assert task.status == "blocked"
    assert run.summary == "codex-failed: exit code 7"
    assert run.metadata["worker_lane"]["exit_code"] == 7
    assert run.metadata["review"]["required"] is False


def test_codex_timeout_blocks_and_records_metadata(kanban_home, tmp_path, monkeypatch):
    old_path = os.environ.get("PATH", "")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "codex"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "sys.stdin.read()\n"
        "print('started', flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + old_path)
    with kb.connect() as conn:
        tid, task = _claim_for_codex(conn)
        run_id = task.current_run_id
    run_codex_worker(
        task_id=tid,
        lane="codex-deep",
        workspace=os.getcwd(),
        sandbox="workspace-write",
        approval="never",
        run_id=run_id,
        claim_lock=task.claim_lock,
        timeout_seconds=0.2,
        heartbeat_interval=0.01,
    )
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        events = kb.list_events(conn, tid)
    assert task.status == "blocked"
    assert run.summary.startswith("codex-timeout:")
    assert run.metadata["worker_lane"]["timed_out"] is True
    assert any(e.kind == "worker_timed_out" for e in events)


def test_codex_output_tail_is_truncated(kanban_home, tmp_path, monkeypatch):
    old_path = os.environ.get("PATH", "")
    fake_bin = _make_fake_codex(tmp_path, "A" * 20000)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + old_path)
    with kb.connect() as conn:
        tid, task = _claim_for_codex(conn)
        run_id = task.current_run_id
    run_codex_worker(
        task_id=tid,
        lane="codex-deep",
        workspace=os.getcwd(),
        sandbox="workspace-write",
        approval="never",
        run_id=run_id,
        claim_lock=task.claim_lock,
        heartbeat_interval=0.01,
    )
    with kb.connect() as conn:
        run = kb.latest_run(conn, tid)
    assert len(run.metadata["worker_lane"]["output_tail"].encode("utf-8")) <= 8192
