"""An accepted routing handoff cannot let snapshot restore bypass a retained profile lock."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import runtime_ownership
from hermes_cli.backup import create_quick_snapshot
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_runtime import GatewayDiscovery, GatewayEndpoint
from hermes_cli.gateway_session_store import GatewaySessionStore
from hermes_cli.profiles import parked_marker_path
from hermes_cli import profiles_remove


class _Connection:
    launch_api_key = None
    final = False

    def close(self):
        self.final = True


@pytest.mark.parametrize("ack", [
    {"unserved": "worker"},
    {"unserved": "worker", "writer_settlement": "retained", "reservation": "retained"},
    {"pending": True},
], ids=["legacy-routing-ack", "retained-receipt", "pending"])
@pytest.mark.parametrize("already_parked", [False, True], ids=["restore-marker", "user-marker"])
def test_snapshot_handoff_waits_for_real_profile_lock(tmp_path, monkeypatch, capsys, ack, already_parked):
    launch = tmp_path / ".hermes"
    home = launch / "profiles" / "worker"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    snapshot = create_quick_snapshot(hermes_home=home)
    assert snapshot
    marker = parked_marker_path(home)
    if already_parked:
        marker.touch()
    store = GatewaySessionStore(_Connection())
    cli = SimpleNamespace(_session_db=store, _gateway_session_open=True, session_id="kept")
    endpoint = GatewayEndpoint(str(home), "owner", 5, 1, "http://127.0.0.1:1", "none",
                               frozenset({"session-authority-v1"}), str(launch))
    monkeypatch.setattr("hermes_cli.gateway_runtime.discover_gateway_endpoint",
                        lambda target: GatewayDiscovery("ready", endpoint))
    monkeypatch.setattr(GatewayConnection, "shared", classmethod(lambda cls: _Connection()))
    monkeypatch.setattr("hermes_cli.cli_snapshot._rejoin_cli", lambda cli, store: None)
    owner = runtime_ownership.ProfileOwnership()
    owner.reserve([home])
    inode = (home / "gateway.lock").stat().st_ino
    handoffs, serves, restores, attempts, waited = [], [], [], [], []

    def unserve(control_home, name):
        assert control_home == launch and name == "worker"
        assert marker.exists() and owner.owns(home)
        handoffs.append(name)
        return ack

    def serve(control_home, name):
        assert control_home == launch and name == "worker"
        assert not marker.exists() and owner.owns(home)
        serves.append(name)
        return {"pending": True}

    real_maintenance = runtime_ownership.exclusive_maintenance

    @contextmanager
    def maintenance(homes, **kwargs):
        attempts.append(tuple(homes))
        with real_maintenance(homes, **kwargs):
            yield

    real_wait = profiles_remove.wait_profile_unowned

    def bounded_wait(target):
        assert target == home and marker.exists() and owner.owns(home)
        result = real_wait(target, timeout=0)
        waited.append(result)
        return result

    monkeypatch.setattr("gateway.control_socket.request_unserve_profile", unserve)
    monkeypatch.setattr("gateway.control_socket.request_serve_profile_hot", serve)
    monkeypatch.setattr(runtime_ownership, "exclusive_maintenance", maintenance)
    monkeypatch.setattr(profiles_remove, "wait_profile_unowned", bounded_wait)
    monkeypatch.setattr("hermes_cli.backup.restore_quick_snapshot",
                        lambda sid: restores.append(sid) or True)
    try:
        # The real CLI mixin reaches _restore_gateway_snapshot and _pause_owner.
        # Their accepted ACK still has to pass the actual independent lock attempt.
        CLICommandsMixin._snapshot_restore(cli, ["/snapshot", "restore", snapshot])
        assert handoffs == ["worker"]
        assert attempts == [(home,)] and waited == [False]
        assert restores == []
        assert "Gateway still owns the profile; nothing was restored" in capsys.readouterr().out
        assert owner.owns(home) and (home / "gateway.lock").stat().st_ino == inode
        # Restore removes only the marker it created, then attempts re-serve. A
        # user's existing parked marker survives and suppresses that request.
        assert marker.exists() is already_parked
        assert serves == ([] if already_parked else ["worker"])
    finally:
        owner.release(home)
