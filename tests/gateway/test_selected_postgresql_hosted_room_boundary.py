"""Selected PostgreSQL must refuse hosted rooms before legacy SQLite or transport work."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from gateway.hosted_room_coordination import (
    HostedRoomCoordinationUnavailableError,
    sqlite_hosted_room_coordination,
    static_sqlite_hosted_room_factory_inventory,
)
from state_store_runtime_readiness import trap_state_db_opens

_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


def _selected_pg_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / ".hermes" / "profiles" / "selected-pg"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    return home


def test_selected_postgresql_refuses_hosted_room_construction_before_sqlite_or_transport(tmp_path, monkeypatch):
    from tui_gateway.hosted_room_driver import HostedRoomRuntime
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    from tui_gateway.hosted_room_service import HostedRoomService

    home = _selected_pg_home(tmp_path, monkeypatch)
    db_path = home / "state.db"

    with trap_state_db_opens(home) as opens:
        for construct in (
            lambda: sqlite_hosted_room_coordination(db_path),
            lambda: HostedRoomService(ModuleType("hosted-room-test"), db_path=db_path),
            lambda: HostedRoomRuntime(db_path=db_path, rooms=(), turn_lock=lambda _profile: None),
            lambda: PeerRunsHTTPClient(base_url="https://peer.example.test", api_key="", receipt_db_path=db_path),
        ):
            with pytest.raises(HostedRoomCoordinationUnavailableError) as caught:
                construct()
            assert caught.value.capability == "hosted-room-coordination-full-protocol"
            assert caught.value.report.profile_home == str(home.resolve())

    assert opens == []
    assert not db_path.exists()
    assert not (home / "sessions").exists()


def test_selected_postgresql_groups_rpc_refuses_before_room_or_peer_side_effects(tmp_path, monkeypatch):
    import tui_gateway.server as server
    from tui_gateway import methods_groups

    home = _selected_pg_home(tmp_path, monkeypatch)
    methods_groups.stop_hosted_room_service(timeout=0.01)
    with trap_state_db_opens(home) as opens:
        capability = server._methods["groups.capabilities"](1, {})
        invitation = server._methods["groups.peer.invite"](2, {"room_id": "must-not-exist"})
        creation = server._methods["groups.create"](3, {"room_id": "must-not-exist"})

    for envelope in (capability, invitation, creation):
        assert envelope["error"]["code"] in {5109, 5111, 4120}
        assert "hosted-room-coordination-full-protocol" in envelope["error"]["message"]
    assert opens == []
    assert not (home / "state.db").exists()
    assert not (home / "sessions").exists()


def test_selected_named_profile_cannot_construct_a_root_sqlite_coordination(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    profile = root / "profiles" / "selected-pg"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")

    root_db = root / "state.db"
    with trap_state_db_opens(root, profile) as opens:
        with pytest.raises(HostedRoomCoordinationUnavailableError) as caught:
            sqlite_hosted_room_coordination(root_db, home=profile)

    assert caught.value.report.profile_home == str(profile.resolve())
    assert opens == []
    assert not root_db.exists()
    assert not (profile / "state.db").exists()


def test_sqlite_hosted_room_factory_inventory_has_no_unreviewed_production_bypass():
    assert [(call.path, call.symbol) for call in static_sqlite_hosted_room_factory_inventory()] == [
        ("gateway/hosted_room_links.py", "_link_rows"),
        ("gateway/hosted_room_links.py", "mark_room_link_status"),
        ("gateway/hosted_room_links.py", "save_room_link"),
        ("tui_gateway/hosted_room_driver.py", "__init__"),
        ("tui_gateway/hosted_room_peer_http.py", "_admit_dispatch"),
        ("tui_gateway/hosted_room_peer_http.py", "_receipt"),
        ("tui_gateway/hosted_room_service.py", "__init__"),
        ("tui_gateway/methods_groups.py", "_"),
        ("tui_gateway/methods_groups.py", "_"),
    ]


def test_default_sqlite_hosted_room_coordination_still_constructs(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    coordination = sqlite_hosted_room_coordination(home / "state.db")

    assert coordination.db_path == home / "state.db"
    assert not coordination.db_path.exists()
