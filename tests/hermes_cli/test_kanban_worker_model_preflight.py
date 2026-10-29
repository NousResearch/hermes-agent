"""A profile without a model must fail before dispatch spawns a worker (#144027)."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(params=[False, True], ids=["standalone", "multiplex"])
def dispatch_profile(tmp_path, monkeypatch, request):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: launch-model\n", encoding="utf-8")
    (profile / ".env").write_text("", encoding="utf-8")
    (profile / "SOUL.md").write_text("Worker", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "normal")
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, command: command)
    spawned = []

    def popen(command, **kwargs):
        spawned.append((command, kwargs))
        return SimpleNamespace(pid=424242)

    monkeypatch.setattr("subprocess.Popen", popen)
    previous = is_multiplex_active()
    set_multiplex_active(request.param)
    try:
        kb.create_board("work")
        yield root, profile, spawned
    finally:
        set_multiplex_active(previous)


@pytest.mark.parametrize("config", [None, {}, {"model": {"default": ""}}, {"model": "  "}])
def test_unconfigured_assignee_refused_before_spawn_with_task_log(dispatch_profile, config):
    from hermes_cli import kanban as cli, kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect_closing

    root, profile, spawned = dispatch_profile
    if config is not None:
        (profile / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    with connect_closing(board="work") as conn:
        task_id = kb.create_task(conn, title="needs a model", assignee="worker", board="work")
        result = dispatch.dispatch_once(conn, board="work")
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
        runs = kb.list_runs(conn, task_id)
    assert not spawned, "dispatch launched an assignee whose owner cannot restore its policy"
    assert not result.spawned and not result.auto_blocked
    assert task.status == "ready" and task.worker_pid is None
    assert "No model configured for this profile" in task.last_failure_error
    assert "hermes -p worker model" in task.last_failure_error
    assert any(event.kind == "spawn_failed" and "No model configured" in event.payload["error"] for event in events)
    assert len(runs) == 1 and runs[0].outcome == "spawn_failed"
    assert "hermes -p worker model" in cli.run_slash(f"--board work log {task_id}")
    # A direct owner call, or a config removed after preflight, must also refuse before
    # create_local_session can persist a receipt that restore_policy will reject.
    from hermes_state_runtime import RuntimeStoreError
    with connect_closing(board="work") as conn:
        direct_id = kb.create_task(conn, title="direct owner", assignee="worker", board="work",
                                   workspace_kind="dir", workspace_path=str(root))
        direct = kb.claim_task(conn, direct_id)
    with pytest.raises(RuntimeStoreError) as caught:
        _owner_policy(profile, direct)
    assert caught.value.reason == "invalid_params" and caught.value.field == "model"
    assert "hermes -p worker model" in caught.value.detail
    if config is None:
        assert not (profile / "config.yaml").exists()


@pytest.mark.parametrize("route", ["configured", "task-override", "welcome-trial"])
def test_resolvable_assignee_keeps_existing_spawn(dispatch_profile, monkeypatch, route):
    from hermes_cli import anon_auth, kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect_closing
    from hermes_constants import get_hermes_home

    root, profile, spawned = dispatch_profile
    if route == "configured":
        (profile / "config.yaml").write_text("model:\n  default: worker-model\n", encoding="utf-8")
    elif route == "welcome-trial":
        # Only the assignee's claimed trial is eligible; the resolver still reads its own scope.
        monkeypatch.setattr(anon_auth, "free_inference_active", lambda: get_hermes_home() == profile)
    with connect_closing(board="work") as conn:
        task_id = kb.create_task(conn, title="configured worker", assignee="worker", board="work",
                                 model_override="task-model" if route == "task-override" else None)
        result = dispatch.dispatch_once(conn, board="work")
        task = kb.get_task(conn, task_id)
    assert len(spawned) == 1 and result.spawned[0][0] == task_id
    assert task.status == "running" and task.worker_pid == 424242
    assert spawned[0][1]["env"]["HERMES_HOME"] == str(profile)
    assert spawned[0][1]["env"]["HERMES_KANBAN_DB"] == str(kb.kanban_db_path("work"))
    from gateway.session_policy import restore_policy
    policy, _ = _owner_policy(profile, task)
    expected = {"configured": "worker-model", "task-override": "task-model", "welcome-trial": "nous/welcome"}
    assert restore_policy(asdict(policy)).model == expected[route]
    assert get_hermes_home() == root


def _owner_policy(profile, task):
    from gateway.run import _load_gateway_config
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli.kanban_db_dispatch import _worker_profile_scope

    actor = Principal("owner", str(profile), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(profile)), actor=actor, native_owner=True)
    params = dict(board="work", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    with _worker_profile_scope(str(profile)):
        return build_kanban_policy(connection, params, _load_gateway_config())
