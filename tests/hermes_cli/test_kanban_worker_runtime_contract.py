"""Regression coverage for the dispatcher-owned Kanban worker runtime contract."""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb._worker_registry.clear()
    kb._worker_exit_context.clear()
    kb._recent_worker_exits.clear()
    yield home
    kb._worker_registry.clear()
    kb._worker_exit_context.clear()
    kb._recent_worker_exits.clear()


def _task(task_id="t_runtime"):
    return kb.Task(
        id=task_id, title="runtime", body=None, assignee="worker", status="running",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="dir", workspace_path=None, claim_lock="host:1",
        claim_expires=None, tenant=None,
    )


def test_registered_worker_nonzero_exit_is_retained_and_redacted(kanban_home, monkeypatch):
    class Proc:
        pid = 42420
        returncode = 7

        def poll(self):
            return self.returncode

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="runtime", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        kb._register_worker_process(
            Proc(), claimed, board="default", log_path="/tmp/worker.log",
            route={"profile": "worker", "provider": "safe", "model": "safe-model"},
        )
        kb._set_worker_pid(conn, task_id, Proc.pid)

        assert kb.reap_worker_zombies() == [Proc.pid]
        assert kb.detect_crashed_workers(conn) == [task_id]

        run = kb.list_runs(conn, task_id)[0]
        assert run.metadata["exit"]["kind"] == "nonzero_exit"
        assert run.metadata["exit"]["code"] == 7
        assert "api_key" not in str(run.metadata).lower()


def test_unknown_dead_pid_is_blocked_not_blindly_retried(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="unknown", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, 98765)

        assert kb.detect_crashed_workers(conn) == [task_id]
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert "unknown" in (task.last_failure_error or "")
        run = kb.list_runs(conn, task_id)[0]
        assert run.outcome == "unknown_exit"
        assert run.metadata["exit"]["kind"] == "unknown"


def test_preflight_rejects_missing_workspace_before_spawn(kanban_home):
    result = kb._preflight_worker_spawn(_task(), "/does/not/exist", board="default")
    assert result["ok"] is False
    assert result["code"] == "workspace_invalid"


def test_reclaim_never_signals_pid_without_live_gateway_ownership(kanban_home, monkeypatch):
    """A host-local lock and PID are not proof that this gateway owns it."""
    calls = []
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")

    result = kb._terminate_reclaimed_worker(
        42420, "host:old-dispatcher", signal_fn=lambda *args: calls.append(args),
    )

    assert calls == []
    assert result["termination_attempted"] is False
    assert result["ownership_verified"] is False
    assert result["ownership_reason"] == "missing_registry_record"


def test_timeout_never_signals_persisted_pid_without_live_gateway_ownership(
    kanban_home, monkeypatch,
):
    """Timeout cleanup cannot kill PID 424242 after a dispatcher restart."""
    calls = []
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb.time, "time", lambda: 10_000)
    monkeypatch.setattr(kb.os, "killpg", lambda *args: calls.append(args))
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="expired", assignee="worker", max_runtime_seconds=1,
        )
        claimed = kb.claim_task(conn, task_id, claimer="host:old-dispatcher")
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, 424242)
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?",
            (1, claimed.current_run_id),
        )

        assert kb.enforce_max_runtime(conn) == []

        assert calls == []
        assert kb.get_task(conn, task_id).status == "running"


def test_reclaim_never_signals_on_process_group_mismatch(kanban_home, monkeypatch):
    class Proc:
        pid = 42420

        def poll(self):
            return None

    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb.os, "getpgid", lambda _pid: 999)
    kb._register_worker_process(Proc(), _task("t_owned"), board="default", log_path="/tmp/worker.log")
    kb._worker_registry[42420].process_group = 111
    calls = []

    result = kb._terminate_reclaimed_worker(
        42420, "host:old-dispatcher", signal_fn=lambda *args: calls.append(args),
    )

    assert calls == []
    assert result["termination_attempted"] is False
    assert result["ownership_verified"] is False
    assert result["ownership_reason"] == "process_group_mismatch"


def test_execution_route_is_persisted_without_runtime_credentials(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="route", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.execution_route = {
            "profile": "worker", "provider": "openai", "model": "gpt-test",
            "credential_ref": "credential_pool", "api_key": "must-not-persist",
        }
        kb._persist_execution_route(conn, task)

        assert kb._latest_execution_route(conn, task_id) == {
            "profile": "worker", "provider": "openai", "model": "gpt-test",
            "credential_ref": "credential_pool",
        }
        assert "must-not-persist" not in str(kb.list_events(conn, task_id))


def test_exit_context_is_bounded_and_releases_exited_popen_references(kanban_home, monkeypatch):
    class Proc:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 1

    monkeypatch.setattr(kb, "_RECENT_WORKER_EXITS_MAX", 8)
    task = _task("t_bounded")
    for pid in range(1000, 1032):
        kb._register_worker_process(Proc(pid), task, board="default", log_path="/tmp/worker.log")
    kb._poll_registered_workers()

    assert len(kb._recent_worker_exits) <= 8
    assert len(kb._worker_exit_context) <= 8


def test_child_inherits_parent_execution_route_and_validated_override(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        claimed = kb.claim_task(conn, parent)
        assert claimed is not None
        claimed.execution_route = {
            "profile": "worker", "provider": "openai", "model": "gpt-parent",
            "credential_ref": "pool:primary",
        }
        kb._persist_execution_route(conn, claimed)
        child = kb.create_task(
            conn, title="child", assignee="reviewer", parents=[parent],
            execution_route_override={"model": "gpt-review"},
        )
        assert kb._latest_execution_route(conn, child) == {
            "profile": "worker", "provider": "openai", "model": "gpt-review",
            "credential_ref": "pool:primary",
        }
        with pytest.raises(ValueError, match="unknown execution route override"):
            kb.create_task(
                conn, title="bad", assignee="worker", execution_route_override={"api_key": "secret"},
            )


def test_exit_evidence_is_atomic_consume_once(kanban_home):
    class Proc:
        pid = 5151

        def poll(self):
            return 3

    task = _task("t_once")
    task.current_run_id = 7
    kb._register_worker_process(Proc(), task, board="default", log_path="/tmp/worker.log")
    kb._poll_registered_workers()
    first = kb._consume_worker_exit(Proc.pid, task.id, task.current_run_id)
    second = kb._consume_worker_exit(Proc.pid, task.id, task.current_run_id)
    assert first[:2] == ("nonzero_exit", 3)
    assert first[2] is not None
    assert second == ("unknown", None, None)


def _prepare_claimed_task(conn, workspace: Path, *, skills=None):
    task_id = kb.create_task(
        conn, title="preflight", assignee="worker", workspace_kind="dir",
        workspace_path=str(workspace), skills=skills,
    )
    task = kb.claim_task(conn, task_id, claimer="host:dispatcher")
    assert task is not None
    return task


def test_canonical_default_route_uses_temp_profile_and_makes_no_model_call(
    kanban_home, tmp_path, monkeypatch,
):
    profile_home = tmp_path / ".hermes" / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: gpt-from-profile\n", encoding="utf-8",
    )
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda home: ["file"])
    calls = []

    def resolve(task, profile, provider, model):
        calls.append((profile, provider, model))
        return {"ok": True, "route": {
            "profile": profile, "provider": provider, "model": model,
            "credential_ref": "auth.json:openai:0",
        }}

    monkeypatch.setattr(kb, "_resolve_worker_route", resolve)
    with kb.connect() as conn:
        task = _prepare_claimed_task(conn, tmp_path)
        result = kb._preflight_worker_spawn(task, str(tmp_path), board="default")
        assert result["ok"] is True
        assert calls == [("worker", "openrouter", "gpt-from-profile")]
        assert result["route"]["credential_ref"] == "auth.json:openai:0"
        assert "api_key" not in json.dumps(result)
        kb._persist_execution_route(conn, task)

        class Proc:
            pid = 5252

            def poll(self):
                return 9

        monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        kb._register_worker_process(
            Proc(), task, board="default", log_path="/tmp/worker.log", route=result["route"],
        )
        kb._set_worker_pid(conn, task.id, Proc.pid)
        kb._poll_registered_workers()
        assert kb.detect_crashed_workers(conn) == [task.id]
        metadata = kb.list_runs(conn, task.id)[0].metadata
        assert metadata is not None
        assert metadata["route"] == result["route"]
        assert "api_key" not in json.dumps(metadata)


def test_unavailable_attached_skill_is_typed_preflight_failure_before_popen(
    kanban_home, tmp_path, monkeypatch,
):
    profile_home = tmp_path / ".hermes" / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: gpt-test\n", encoding="utf-8",
    )
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda home: ["file"])
    monkeypatch.setattr(kb, "_resolve_worker_route", lambda *args: {"ok": True, "route": {
        "profile": "worker", "provider": "openrouter", "model": "gpt-test",
        "credential_ref": "auth.json:openai:0",
    }})
    monkeypatch.setattr("agent.skill_commands.scan_skill_commands", lambda: {})
    popen_calls = []
    monkeypatch.setattr(kb.subprocess, "Popen", lambda *a, **kw: popen_calls.append((a, kw)))
    with kb.connect() as conn:
        task = _prepare_claimed_task(conn, tmp_path, skills=["missing-carrier-skill"])
        result = kb._preflight_worker_spawn(task, str(tmp_path), board="default")
        assert result["code"] == "skills_invalid"
        kb._block_preflight_failure(conn, task.id, result)
        assert popen_calls == []
        stored = kb.get_task(conn, task.id)
        assert stored is not None
        assert stored.status == "blocked"
        assert stored.consecutive_failures == 0
        run = kb.list_runs(conn, task.id)[0]
        assert run.outcome == "preflight_failed"
        assert "pid gone" not in (run.error or "")


def test_invalid_explicit_model_is_rejected_before_popen(kanban_home, tmp_path, monkeypatch):
    profile_home = tmp_path / ".hermes" / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: valid-model\n", encoding="utf-8",
    )
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda home: ["file"])
    monkeypatch.setattr(kb, "_resolve_worker_route", lambda *args: {
        "ok": False, "code": "route_invalid", "detail": "model is unavailable for provider",
    })
    popen_calls = []
    monkeypatch.setattr(kb.subprocess, "Popen", lambda *a, **kw: popen_calls.append((a, kw)))
    with kb.connect() as conn:
        task = _prepare_claimed_task(conn, tmp_path)
        task.model_override = "not-in-catalog"
        result = kb._preflight_worker_spawn(task, str(tmp_path), board="default")
        assert result["code"] == "route_invalid"
        assert result["detail"] == "model is unavailable for provider"
        assert popen_calls == []


@pytest.mark.parametrize(
    ("platform_name", "scenario", "expected_timeout"),
    [
        ("posix", "missing", False),
        ("posix", "pid_mismatch", False),
        ("posix", "group_mismatch", False),
        ("posix", "owned", True),
        ("nt", "missing", False),
        ("nt", "pid_mismatch", False),
        ("nt", "owned", True),
    ],
)
def test_enforce_max_runtime_ownership_matrix(
    kanban_home, monkeypatch, platform_name, scenario, expected_timeout,
):
    class Proc:
        def __init__(self, pid):
            self.pid = pid
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    pid = 6060
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb.time, "time", lambda: 10_000)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    real_os = kb.os

    class PlatformOS:
        name = platform_name

        def getpgid(self, _pid):
            return 7000 if scenario != "group_mismatch" else 7999

        def __getattr__(self, name):
            return getattr(real_os, name)

    monkeypatch.setattr(kb, "os", PlatformOS())
    signals = []
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="timeout", assignee="worker", max_runtime_seconds=1)
        task = kb.claim_task(conn, task_id, claimer="host:dispatcher")
        assert task is not None
        kb._set_worker_pid(conn, task_id, pid)
        conn.execute("UPDATE task_runs SET started_at=1 WHERE id=?", (task.current_run_id,))
        proc = Proc(pid)
        if scenario != "missing":
            record_pid = pid + 1 if scenario == "pid_mismatch" else pid
            with kb._worker_registry_lock:
                kb._worker_registry[pid] = kb._WorkerProcess(
                    pid=record_pid, task_id=task_id, run_id=task.current_run_id,
                    board="default", proc=proc, started_at=1, process_group=7000,
                    log_path="/tmp/worker.log", route={},
                )
        result = kb.enforce_max_runtime(
            conn, signal_fn=(lambda *args: signals.append(args)) if scenario != "missing" else None,
        )
        assert (task_id in result) is expected_timeout
        stored = kb.get_task(conn, task_id)
        assert stored is not None
        if not expected_timeout:
            assert stored.status == "running"
        elif platform_name == "nt":
            assert proc.terminated is True
        else:
            assert signals
