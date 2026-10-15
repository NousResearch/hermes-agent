"""Refused/failed CLI-16 restores never bypass ownership or abandon the CLI viewer."""
from types import SimpleNamespace

import pytest

from hermes_cli.backup import create_quick_snapshot
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_connection import GatewayConnection, SessionState
from hermes_cli.gateway_runtime import GatewayDiscovery, GatewayEndpoint
from hermes_cli.gateway_session_store import GatewaySessionStore


class _Connection:
    def __init__(self, refusal=None):
        self.final = False
        self.states = {}
        self.calls = []
        self.refusal = refusal

    def close(self):
        self.final = True

    def state(self, sid):
        return self.states.setdefault(sid, SessionState(sid))

    def call(self, method, **params):
        self.calls.append(method)
        if self.refusal:
            raise GatewayClientError(self.refusal)
        assert method == "session.resume"
        return {"stored_session_id": "kept", "subscription_id": "fresh-subscription",
                "revision": 4, "execution_generation": 2, "messages": [], "info": {}}


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    home = tmp_path / "state"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    snapshot = create_quick_snapshot(hermes_home=home)
    assert snapshot
    old, fresh = _Connection(), _Connection()
    store = GatewaySessionStore(old)
    closed = []
    store._reader = SimpleNamespace(close=lambda: closed.append(True))
    store.snapshots["kept"] = {"subscription_id": "stale"}
    cli = SimpleNamespace(_session_db=store, _gateway_session_open=True, session_id="kept",
                          agent=object(), conversation_history=[], model="fixture",
                          _goal_manager=object(), _loop_manager=object(), _heartbeat_manager=object())
    endpoint = GatewayEndpoint(str(home), "owner", 5, 1, "http://127.0.0.1:1", "none",
                               frozenset({"session-authority-v1"}), str(home))
    monkeypatch.setattr("hermes_cli.gateway_runtime.discover_gateway_endpoint",
                        lambda home: GatewayDiscovery("ready", endpoint))
    monkeypatch.setattr(GatewayConnection, "shared", classmethod(lambda cls: fresh))
    monkeypatch.setattr("hermes_cli.gateway_tools.bind_session", lambda *args: None)
    monkeypatch.setattr("gateway.control_socket.pause_gateway_for_update", lambda *a, **kw: {"pausing": True})
    monkeypatch.setattr("hermes_cli.profiles_remove.wait_profile_unowned", lambda home: True)
    return cli, snapshot, old, fresh, closed


@pytest.mark.parametrize("result", [False, "raise"])
def test_failed_restore_still_rejoins_the_viewer(viewer, monkeypatch, result):
    cli, snapshot, old, fresh, closed = viewer

    def restore(sid):
        assert old.final and closed == [True]
        if result == "raise":
            raise OSError("fixture restore refused")
        return result

    monkeypatch.setattr("hermes_cli.backup.restore_quick_snapshot", restore)
    CLICommandsMixin._snapshot_restore(cli, ["/snapshot", "restore", snapshot])
    assert cli._session_db.connection is fresh
    assert fresh.calls == ["session.resume"]
    assert cli._session_db.current["subscription_id"] == "fresh-subscription"
    assert cli.agent is None
    assert cli._goal_manager is cli._loop_manager is cli._heartbeat_manager is None


@pytest.mark.parametrize("boundary", ["refused", "still-owned"])
def test_handoff_must_release_the_owner_before_restoring(viewer, monkeypatch, boundary):
    cli, snapshot, old, fresh, closed = viewer
    if boundary == "refused":
        monkeypatch.setattr("gateway.control_socket.pause_gateway_for_update", lambda *a, **kw: None)
    else:
        monkeypatch.setattr("hermes_cli.profiles_remove.wait_profile_unowned", lambda home: False)

    def forbidden_restore(sid):
        raise AssertionError("restore must not run before ownership is released")

    monkeypatch.setattr("hermes_cli.backup.restore_quick_snapshot", forbidden_restore)
    CLICommandsMixin._snapshot_restore(cli, ["/snapshot", "restore", snapshot])
    assert old.final and closed == [True]
    assert cli._session_db.connection is fresh and fresh.calls == ["session.resume"]


def test_resume_authorization_refusal_never_creates_another_session(viewer, monkeypatch):
    cli, snapshot, _old, fresh, _closed = viewer
    fresh.refusal = "permission_denied"
    monkeypatch.setattr("hermes_cli.backup.restore_quick_snapshot", lambda sid: True)
    CLICommandsMixin._snapshot_restore(cli, ["/snapshot", "restore", snapshot])
    assert fresh.calls == ["session.resume"]
    assert cli.session_id == "kept" and cli._session_db.connection is fresh


@pytest.mark.parametrize("target", ["missing", "../outside", "0", "99999"])
def test_invalid_target_does_not_disconnect_the_gateway(viewer, target):
    cli, _snapshot, old, fresh, closed = viewer
    CLICommandsMixin._snapshot_restore(cli, ["/snapshot", "restore", target])
    assert not old.final and closed == [] and fresh.calls == []
