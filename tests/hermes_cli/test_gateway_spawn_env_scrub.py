"""Regression tests: gateway spawn env must never inherit the Kanban write fence.

Incident: a gateway process launched from an
environment carrying ``HERMES_DELEGATED_CHILD_CONTEXT`` was fenced from its own
board — dispatcher reclaim, notifier subscriptions and every cron script
subprocess (``kb.create_task``) failed with ``PermissionError: delegate_task
child contexts cannot mutate Kanban tasks or boards`` until the gateway was
restarted clean. ``agent.delegation_context.scrub_gateway_spawn_env`` strips
worker identity (``KANBAN_ENV_KEYS``) and the fence marker at every gateway
spawn site while retaining board-location vars, matching ``scrub_kanban_env``.
"""

from types import SimpleNamespace

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    KANBAN_ENV_KEYS,
    scrub_gateway_spawn_env,
)


def _fenced_env() -> dict:
    env = {"PATH": "/usr/bin", "HERMES_KANBAN_DB": "/boards/main.db", "HERMES_KANBAN_BOARD": "default"}
    for key in KANBAN_ENV_KEYS:
        env[key] = "worker-value"
    env[DELEGATED_CHILD_ENV_MARKER] = "/boards"
    return env


def test_scrub_gateway_spawn_env_strips_worker_identity_and_fence():
    cleaned = scrub_gateway_spawn_env(_fenced_env())
    for key in KANBAN_ENV_KEYS:
        assert key not in cleaned
    assert DELEGATED_CHILD_ENV_MARKER not in cleaned
    # Board location and ordinary env survive (retain-location rule).
    assert cleaned["HERMES_KANBAN_DB"] == "/boards/main.db"
    assert cleaned["HERMES_KANBAN_BOARD"] == "default"
    assert cleaned["PATH"] == "/usr/bin"


def test_restart_watcher_spawn_env_is_scrubbed(monkeypatch):
    """``_spawn_gateway_restart_watcher`` respawns the gateway with the watcher's
    own os.environ, so the watcher's env must be scrubbed at spawn."""
    import hermes_cli.gateway as gateway

    captured = {}

    def fake_popen(argv, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=12345)

    for key, value in _fenced_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)

    assert gateway._spawn_gateway_restart_watcher(
        999999, ["python", "-m", "hermes_cli.main", "gateway", "run"]
    )

    env = captured["kwargs"]["env"]
    for key in KANBAN_ENV_KEYS:
        assert key not in env
    assert DELEGATED_CHILD_ENV_MARKER not in env
    assert env["HERMES_KANBAN_DB"] == "/boards/main.db"


def test_launchd_detached_spawn_env_is_scrubbed(monkeypatch, tmp_path):
    import hermes_cli.gateway as gateway
    import hermes_cli.gateway_launchd as gateway_launchd

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=12345)

    for key, value in _fenced_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(gateway, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway, "get_python_path", lambda: "/usr/bin/python3")
    monkeypatch.setattr(
        gateway, "_gateway_run_command",
        lambda: ["/usr/bin/python3", "-m", "hermes_cli.main", "gateway", "run", "--replace"],
    )
    monkeypatch.setattr(gateway_launchd.subprocess, "Popen", fake_popen)

    assert gateway_launchd._spawn_detached_gateway() is True

    env = captured["kwargs"]["env"]
    for key in KANBAN_ENV_KEYS:
        assert key not in env
    assert DELEGATED_CHILD_ENV_MARKER not in env
    assert env["HERMES_KANBAN_DB"] == "/boards/main.db"


@pytest.mark.platforms("windows")
def test_windows_detached_spawn_env_is_scrubbed(monkeypatch, tmp_path):
    import hermes_cli.gateway_windows as gateway_windows

    captured = {}

    def fake_popen(argv, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=12345)

    for key, value in _fenced_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        gateway_windows, "_build_gateway_argv",
        lambda home=None: (["python", "-m", "hermes_cli.main", "gateway", "run"], str(tmp_path), {"HERMES_HOME": str(tmp_path)}),
    )
    monkeypatch.setattr(gateway_windows, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway_windows.subprocess, "Popen", fake_popen)

    assert gateway_windows._spawn_detached() == 12345

    env = captured["kwargs"]["env"]
    for key in KANBAN_ENV_KEYS:
        assert key not in env
    assert DELEGATED_CHILD_ENV_MARKER not in env
    assert env["HERMES_KANBAN_DB"] == "/boards/main.db"
    # The spawn overlay still applies on top of the scrubbed base.
    assert env["HERMES_HOME"] == str(tmp_path)
