"""CLI-16: the restore's own outcome decides the reply; a supervised owner or bad id never drains."""
from types import SimpleNamespace

import pytest

from hermes_cli.backup import create_quick_snapshot
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_runtime import GatewayDiscovery, GatewayEndpoint
from hermes_cli.gateway_session_store import GatewaySessionStore


class _Connection:
    def __init__(self):
        self.final = False

    def close(self):
        self.final = True

    def call(self, method, **params):
        # The rejoin after maintenance fails: it must not decide the restore's reply.
        raise GatewayClientError("permission_denied")

    def resume(self, call, session_id, *, typed=False):
        return call("session.resume", session_id=session_id)  # GatewayConnection.resume, no launch key


def _restore_error(sid):
    raise OSError("fixture restore refused")


def _corrupt_manifest(sid):
    import json
    return json.loads("{")  # what a truncated manifest.json raises out of the restore


@pytest.mark.parametrize("case, restore, supervisor, expected", [
    ("restored", lambda sid: True, "manual", "Restored state from"),
    ("restore-error", _restore_error, "manual", "fixture restore refused"),
    ("supervised", None, "systemd", "stop its systemd unit"),
    ("satellite-not-reserved", lambda sid: True, "manual", "is not served again yet: host operation is still pending"),
    ("invalid-id", lambda sid: False, "manual", "Snapshot not found: ../outside"),
    ("other-pid-paused", None, "manual", "Gateway refused the snapshot maintenance handoff"),
    ("satellite-unserve-error", None, "manual", "nothing was restored. Park the profile with `hermes -p state"),
    ("corrupt-manifest", _corrupt_manifest, "manual", "was not fully restored"),
    # A remote gateway's chat restores this host's snapshot offline, as on main, and stays connected.
    ("remote", lambda sid: True, None, "Restart recommended"),
])
def test_restore_reply_follows_the_restore_not_the_rejoin(tmp_path, monkeypatch, capsys,
                                                          case, restore, supervisor, expected):
    home = (tmp_path / "state").resolve()
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    if case == "remote":
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "wss://remote.invalid/api/ws")
    else:
        monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    snapshot = create_quick_snapshot(hermes_home=home)
    old = _Connection()
    store = GatewaySessionStore(old)
    cli = SimpleNamespace(_session_db=store, _gateway_session_open=True, session_id="kept",
                          agent=object(), conversation_history=[])
    satellite = case.startswith("satellite-")
    control_home = str(tmp_path) if satellite else str(home)
    endpoint = GatewayEndpoint(str(home), "owner", 5, 1, "http://127.0.0.1:1", "none",
                               frozenset({"session-authority-v1"}), control_home)
    monkeypatch.setattr("hermes_cli.gateway_runtime.discover_gateway_endpoint",
                        lambda home: GatewayDiscovery("ready", endpoint))
    monkeypatch.setattr(GatewayConnection, "shared", classmethod(lambda cls: _Connection()))
    monkeypatch.setattr("gateway.control_socket.identify_gateway", lambda h, **kw: {
        "pid": 4321, "hermes_home": str(home), "launch_supervisor": supervisor})
    paused = []
    monkeypatch.setattr("gateway.control_socket.pause_gateway_for_update",
                        lambda *a, **kw: paused.append(a) or {"pausing": True,
                                                              "pid": 99 if case == "other-pid-paused" else 4321})
    monkeypatch.setattr("gateway.control_socket.request_unserve_profile",
                        lambda *a: {"unserved": home.name, "error": "teardown failed"}
                        if case == "satellite-unserve-error" else {"unserved": home.name})
    monkeypatch.setattr("gateway.control_socket.request_serve_profile_hot", lambda *a: {"pending": True})
    monkeypatch.setattr("hermes_cli.profiles_remove.wait_profile_unowned", lambda home: True)
    monkeypatch.setattr("hermes_cli.backup.restore_quick_snapshot",
                        restore or (lambda sid: pytest.fail("restore ran without the identified owner's handoff")))

    target = "../outside" if case == "invalid-id" else snapshot
    CLICommandsMixin._snapshot_restore(cli, ["/snapshot", "restore", target])

    out = capsys.readouterr().out
    assert expected in out
    if case in {"supervised", "invalid-id", "remote"}:
        assert not paused and not old.final  # refused before tearing down the viewer
        assert store.connection is old  # a remote chat keeps its connection too
    else:
        assert "Reconnecting to the gateway failed: permission_denied" in out
