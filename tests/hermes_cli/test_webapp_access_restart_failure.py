"""A Web Access restart whose child exits without replacing the gateway says why on ``/status``.

The restart route answers ``restart_started: true`` as soon as ``hermes gateway restart`` is
spawned. When that child exits at once (a missing dependency environment prints ``run `hermes pm
repair```), the old gateway keeps serving the old config, so the pane polling ``/status`` for the
new one needs the failure and its cause there, not a 90-second timeout.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("fastapi")


@pytest.fixture()
def client(monkeypatch):
    from tests.hermes_cli.native_gateway_client import native_desktop_client

    return native_desktop_client(monkeypatch)


@pytest.fixture()
def restart_child(monkeypatch, tmp_path):
    """The real restart spawn (``_spawn_gateway_restart`` → ``_spawn_hermes_action``, log header
    included), with the ``hermes gateway restart`` argv swapped for a stand-in child whose output
    and exit code each test picks: no gateway lifecycle code runs."""
    import hermes_cli._launchers as launchers
    from hermes_cli import web_server
    import hermes_cli.web_server_gateway as gw
    from hermes_cli import webapp_access

    monkeypatch.setattr(gw, "_ACTION_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(gw, "_action_targets_system_gateway", lambda _subcommand: False)
    monkeypatch.setattr(gw, "_profile_action_environment", lambda _subcommand, _overrides=None: {
        "PATH": os.environ.get("PATH", "")})
    monkeypatch.setattr(gw, "live_session_refusal", lambda _profile, _verb: None)
    monkeypatch.setattr(web_server, "_LAST_GATEWAY_RESTART", None)
    monkeypatch.setattr(webapp_access, "lan_address", lambda: "192.168.1.50")
    child: dict[str, str] = {}

    def _runtime_command(_root, _subcommand):
        return [sys.executable, "-c", child["script"]]

    monkeypatch.setattr(launchers, "runtime_command", _runtime_command)
    yield child
    recent = web_server._LAST_GATEWAY_RESTART
    if recent is not None:
        recent[1].wait(timeout=10)


def _status_after_child_exits(client, pid: int) -> dict:
    from hermes_cli import web_server

    web_server._LAST_GATEWAY_RESTART[1].wait(timeout=10)
    assert web_server._LAST_GATEWAY_RESTART[1].pid == pid
    response = client.get("/api/webapp-access/status")
    assert response.status_code == 200, response.text
    return response.json()


def test_a_restart_child_that_exits_at_once_reports_its_last_error_line(client, restart_child):
    restart_child["script"] = (
        "import sys; print('Starting gateway restart...');"
        "print(\"error: Hermes's dependency environment is missing; run `hermes pm repair`\");"
        "sys.exit(1)"
    )

    answer = client.post("/api/webapp-access/restart").json()
    assert answer["restart_started"] is True

    failure = _status_after_child_exits(client, answer["restart_pid"])["restart_failure"]
    assert failure == {
        "pid": answer["restart_pid"],
        "exit_code": 1,
        "reason": "error: Hermes's dependency environment is missing; run `hermes pm repair`",
    }


def test_a_restart_child_that_succeeds_reports_no_failure(client, restart_child):
    restart_child["script"] = "print('Gateway restarted.')"

    answer = client.post("/api/webapp-access/restart").json()
    assert answer["restart_started"] is True

    assert _status_after_child_exits(client, answer["restart_pid"])["restart_failure"] is None


def test_a_silent_failure_names_its_exit_code_not_an_earlier_runs_error(client, restart_child):
    import hermes_cli.web_server_gateway as gw

    gw._ACTION_LOG_DIR.mkdir(parents=True)
    (gw._ACTION_LOG_DIR / "gateway-restart.log").write_text(
        "\n=== gateway-restart started 2026-10-01 09:00:00 ===\nan earlier run's error\n")
    restart_child["script"] = "import sys; sys.exit(3)"

    answer = client.post("/api/webapp-access/restart").json()

    failure = _status_after_child_exits(client, answer["restart_pid"])["restart_failure"]
    assert failure["reason"] == "exit code 3"


def test_a_retry_that_fails_to_spawn_does_not_lend_its_error_to_the_recorded_child(
        client, restart_child, monkeypatch):
    """A retry writes its own log header before it spawns; when that spawn fails, the child still on
    record keeps its own reason."""
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "GATEWAY_RESTART_COOLDOWN_SECONDS", 0.0)
    restart_child["script"] = "import sys; print('first child: run `hermes pm repair`'); sys.exit(1)"
    first = client.post("/api/webapp-access/restart").json()
    web_server._LAST_GATEWAY_RESTART[1].wait(timeout=10)

    import hermes_cli.web_server_gateway as gw

    def _spawn_refused(_subcommand, _overrides=None):
        raise RuntimeError("retry could not start")

    monkeypatch.setattr(gw, "_profile_action_environment", _spawn_refused)
    retry = client.post("/api/webapp-access/restart").json()
    assert retry["restart_started"] is False

    failure = client.get("/api/webapp-access/status").json()["restart_failure"]
    assert failure["pid"] == first["restart_pid"]
    assert failure["reason"] == "first child: run `hermes pm repair`"
