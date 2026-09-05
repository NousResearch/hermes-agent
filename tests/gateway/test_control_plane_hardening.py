"""Gateway control-plane hardening: origin marker, lock owner, stall, restart handoff.

Tests use fakes and temp homes only. They must never restart a live gateway.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from gateway.control_plane import (
    GATEWAY_ORIGIN_ENV,
    build_restart_handoff,
    claim_restart_recovery,
    clear_dispatcher_lock_owner,
    format_lock_owner,
    is_gateway_originated,
    load_restart_handoff,
    persist_restart_handoff,
    propagate_gateway_origin,
    read_dispatcher_lock_owner,
    scrub_gateway_markers_for_restart_watcher,
    should_refuse_inline_gateway_lifecycle,
    stamp_gateway_origin,
    StalledDispatchTracker,
    write_dispatcher_lock_owner,
)


def test_origin_marker_propagates_after_gateway_env_stripped():
    parent = {"_HERMES_GATEWAY": "1", "PATH": "/usr/bin"}
    child = dict(parent)
    child.pop("_HERMES_GATEWAY")
    propagate_gateway_origin(parent, child)
    assert child.get("_HERMES_GATEWAY") is None
    assert is_gateway_originated(child)
    assert should_refuse_inline_gateway_lifecycle(child, supervised=False)


def test_origin_marker_not_invented_for_external_shell():
    child = {"PATH": "/usr/bin"}
    propagate_gateway_origin({"PATH": "/usr/bin"}, child)
    assert not is_gateway_originated(child)
    assert not should_refuse_inline_gateway_lifecycle(child, supervised=False)


def test_restart_watcher_scrub_drops_both_markers():
    env = {"_HERMES_GATEWAY": "1", GATEWAY_ORIGIN_ENV: "1", "HOME": "/tmp"}
    scrub_gateway_markers_for_restart_watcher(env)
    assert "_HERMES_GATEWAY" not in env
    assert GATEWAY_ORIGIN_ENV not in env
    assert not should_refuse_inline_gateway_lifecycle(env, supervised=False)


def test_kanban_worker_env_refuses_without_supervised_pid():
    env = {"HERMES_KANBAN_TASK": "t_abc", "PATH": "/usr/bin"}
    assert should_refuse_inline_gateway_lifecycle(env, supervised=False)


def test_stamp_origin_is_explicit():
    env = {}
    stamp_gateway_origin(env)
    assert env[GATEWAY_ORIGIN_ENV] == "1"


def test_cli_restart_refuses_gateway_originated_worker(monkeypatch):
    monkeypatch.setenv(GATEWAY_ORIGIN_ENV, "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    from hermes_cli.gateway import gateway_command

    with pytest.raises(SystemExit) as exc_info:
        gateway_command(Namespace(gateway_command="restart", all=False, system=False))
    assert exc_info.value.code == 1


def test_cli_stop_refuses_kanban_worker(monkeypatch):
    monkeypatch.delenv(GATEWAY_ORIGIN_ENV, raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    from hermes_cli.gateway import gateway_command

    with pytest.raises(SystemExit) as exc_info:
        gateway_command(Namespace(gateway_command="stop", all=False, system=False))
    assert exc_info.value.code == 1


def test_cli_stop_still_allowed_from_external_shell(monkeypatch):
    monkeypatch.delenv(GATEWAY_ORIGIN_ENV, raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    import hermes_cli.gateway as gw

    class _Reached(Exception):
        pass

    def _sentinel(*a, **k):
        raise _Reached()

    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", _sentinel)
    monkeypatch.setattr(gw, "_dispatch_all_via_service_manager_if_s6", _sentinel)
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)

    with pytest.raises(_Reached):
        gw.gateway_command(Namespace(gateway_command="stop", all=False, system=False))


class TestTerminalOriginGuard:
    def _block(self, command: str) -> dict:
        from tools.terminal_tool_guards import gateway_lifecycle_block

        class _FakeEnv:
            env = {}

            def execute(self, command, **kwargs):  # pragma: no cover
                raise AssertionError("execute must not be reached")

        blocked = gateway_lifecycle_block(
            command=command,
            env=_FakeEnv(),
            env_type="local",
            cwd="/tmp",
            workdir=None,
            session_key="",
        )
        assert blocked is not None
        return json.loads(blocked)

    def test_blocks_restart_when_origin_set(self, monkeypatch):
        from tools import process_registry

        monkeypatch.setenv(GATEWAY_ORIGIN_ENV, "1")
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
        result = self._block("hermes gateway restart")
        assert result["exit_code"] == 1
        assert "Blocked" in result["error"]

    def test_blocks_stop_for_kanban_worker(self, monkeypatch):
        from tools import process_registry

        monkeypatch.delenv(GATEWAY_ORIGIN_ENV, raising=False)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
        monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
        result = self._block("hermes gateway stop")
        assert result["exit_code"] == 1
        assert "Blocked" in result["error"]


def test_lock_owner_roundtrip(tmp_path):
    lock_path = tmp_path / ".dispatcher.lock"
    lock_path.write_text("", encoding="utf-8")
    write_dispatcher_lock_owner(lock_path, profile="analytics", pid=4242, acquired_at=1_700_000_000)
    owner = read_dispatcher_lock_owner(lock_path)
    assert owner is not None
    assert owner["profile"] == "analytics"
    assert owner["pid"] == 4242
    assert owner["acquired_at"]
    assert "analytics" in format_lock_owner(owner)
    assert "4242" in format_lock_owner(owner)
    clear_dispatcher_lock_owner(lock_path)
    assert read_dispatcher_lock_owner(lock_path) is None


def test_current_profile_name_reads_env_not_a_fixed_roster(monkeypatch):
    from gateway.control_plane import current_profile_name

    monkeypatch.setenv("HERMES_PROFILE", "custom-profile-xyz")
    assert current_profile_name() == "custom-profile-xyz"


def test_stalled_dispatch_warns_after_one_interval_not_before():
    tracker = StalledDispatchTracker()
    owner = {"profile": "default", "pid": 9, "acquired_at": "2026-09-04T00:00:00Z"}
    first = tracker.observe(ready=3, running=0, board="default", owner=owner)
    assert first is None
    second = tracker.observe(ready=3, running=0, board="default", owner=owner)
    assert second is not None
    assert "READY" in second
    assert "running=0" in second
    assert "default" in second
    assert "lock_owner=" in second
    assert "Not auto-running" in second
    recovered = tracker.observe(ready=1, running=1, board="default", owner=owner)
    assert recovered is None


def test_ready_running_snapshot_skips_quarantined_corrupt_board(monkeypatch):
    """Stall probing must not reopen a board already quarantined as corrupt."""
    from types import SimpleNamespace

    from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher
    from hermes_cli import kanban_db_connect as _kbc

    calls = {"connect": 0}

    def _boom(*_args, **_kwargs):
        calls["connect"] += 1
        raise AssertionError("quarantined board must not be reopened")

    monkeypatch.setattr(_kbc, "connect", _boom)

    kb = SimpleNamespace(
        DEFAULT_BOARD="default",
        list_boards=lambda include_archived=False: [{"slug": "default"}],
        read_board_metadata=lambda slug: {"slug": slug},
    )
    settings = _DispatcherSettings(
        interval=60.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )
    dispatcher = _KanbanDispatcher(kb, settings)
    dispatcher.disabled_corrupt_boards["default"] = (("fp", 1, 1), 0.0)

    ready, running, board = dispatcher.ready_running_snapshot()
    assert calls["connect"] == 0
    assert ready == 0
    assert running == 0
    assert board == "default"


def test_restart_handoff_persists_before_recovery_claim(tmp_path):
    path = tmp_path / ".restart_handoff.json"
    handoff = build_restart_handoff(
        session_key="telegram:1:dm",
        platform="telegram",
        chat_id="99",
        message_id="m1",
    )
    persist_restart_handoff(path, handoff)
    loaded = load_restart_handoff(path)
    assert loaded is not None
    assert loaded["acknowledged"] is True
    assert loaded["session_key"] == "telegram:1:dm"
    assert claim_restart_recovery(loaded, launcher_exited=True) is False
    assert claim_restart_recovery(loaded, gateway_online=True, launcher_exited=True) is False
    assert claim_restart_recovery(
        loaded,
        gateway_online=True,
        notify_delivered=True,
        launcher_exited=True,
    )
    assert claim_restart_recovery(
        loaded,
        gateway_online=True,
        sessions_restored=True,
    )


def test_restart_handoff_unacknowledged_cannot_claim():
    assert claim_restart_recovery({"acknowledged": False}, gateway_online=True, notify_delivered=True) is False
    assert claim_restart_recovery(None, gateway_online=True, notify_delivered=True) is False


@pytest.mark.asyncio
async def test_restart_command_persists_handoff_before_request(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from gateway.control_plane import RESTART_HANDOFF_FILENAME, load_restart_handoff
    from gateway.platforms.base import MessageEvent, MessageType
    from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", raising=False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m-handoff",
    )
    result = await runner._handle_restart_command(event)
    assert result
    runner.request_restart.assert_called_once()
    handoff = load_restart_handoff(tmp_path / RESTART_HANDOFF_FILENAME)
    assert handoff is not None
    assert handoff["acknowledged"] is True
    assert handoff["platform"] == "telegram"
    assert handoff["chat_id"] == "123456"
    assert handoff["message_id"] == "m-handoff"


@pytest.mark.asyncio
async def test_restart_command_refuses_when_handoff_persist_fails(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from gateway.platforms.base import MessageEvent, MessageType
    from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", raising=False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr(
        "gateway.control_plane.persist_restart_handoff",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m-handoff-fail",
    )
    result = await runner._handle_restart_command(event)
    assert "Restart aborted" in str(result)
    runner.request_restart.assert_not_called()


class TestExecuteCodeOriginGuard:
    def _run(self, monkeypatch, code: str) -> dict:
        import tools.code_execution_tool as cet
        from tools import process_registry

        monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
        monkeypatch.setattr("tools.terminal_scope.enforce_no_refusal", lambda: None)
        monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
        return json.loads(cet.execute_code(code))

    def test_blocks_restart_when_origin_set(self, monkeypatch):
        monkeypatch.setenv(GATEWAY_ORIGIN_ENV, "1")
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        result = self._run(monkeypatch, 'import os; os.system("hermes gateway restart")')
        assert "Blocked" in result.get("error", "")

    def test_blocks_stop_for_kanban_worker(self, monkeypatch):
        monkeypatch.delenv(GATEWAY_ORIGIN_ENV, raising=False)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
        result = self._run(monkeypatch, 'import subprocess; subprocess.run(["systemctl", "stop", "hermes-gateway"])')
        assert "Blocked" in result.get("error", "")


def test_default_spawn_propagates_origin_when_parent_is_gateway(monkeypatch, tmp_path):
    import subprocess

    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _default_spawn
    from gateway.control_plane import GATEWAY_ORIGIN_ENV

    captured = {}

    class FakeProc:
        pid = 99

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv(GATEWAY_ORIGIN_ENV, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ws = tmp_path / "ws"
    ws.mkdir()
    task = kb.Task(
        id="t_origin",
        title="origin",
        body=None,
        assignee="worker",
        status="ready",
        priority=0,
        created_by="user",
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    _default_spawn(task, str(ws))
    assert captured["env"].get(GATEWAY_ORIGIN_ENV) == "1"
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_origin"
