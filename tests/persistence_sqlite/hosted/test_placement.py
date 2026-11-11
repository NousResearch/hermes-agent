"""Which file each placement reaches, and the one-time import of pre-isolation rooms into the shared file.

``'profile'`` is the bound profile's ``state.db``, ``'shared'`` the install's root ``shared-state.db`` that every profile of the install shares. The first open of a ``shared-state.db`` imports the pre-isolation rooms of its sibling ``state.db`` once, so a later purge is never undone; a profile's ``state.db`` never imports.
"""

from __future__ import annotations

import pytest

import sqlite3
from pathlib import Path

from persistence import PermanentStoreError, ProfileScope, StoreUnavailable
from persistence.sqlite import SQLiteBackend, _registry
from tests.persistence_sqlite.hosted.conftest import NOW, dump, tables


pytestmark = pytest.mark.usefixtures("clock")

MEMBERS = [{"member_id": "m1", "profile": "p1"}]
USER = {"kind": "user", "id": "u1"}
# The tables the rooms step creates in every file a room verb opens.
SCHEMA = {"hosted_rooms", "hosted_room_events", "hosted_room_retired_ids", "hosted_room_links", "hosted_room_remote_runs",
          "hosted_room_revoked_grants", "hosted_room_peer_reservations"}


def _room_ids(path: Path) -> list[str]:
    with sqlite3.connect(path) as conn:
        return [row[0] for row in conn.execute("SELECT room_id FROM hosted_rooms ORDER BY room_id")]


def _create(store, room_id: str, placement: str, now: float = NOW) -> None:
    store.hosted_rooms.create_room(placement=placement, room_id=room_id, name="Room", members=MEMBERS,
                                   authority_gateway_id="gw", now=now)


def test_each_placement_writes_its_own_file_only(backend, scopes, root):
    store = backend.open(scopes["a"])
    state_db, shared = scopes["a"].profile_dir / "state.db", root / "shared-state.db"
    _create(store, "in-profile", "profile")
    assert not shared.exists()
    _create(store, "in-shared", "shared")
    assert _room_ids(state_db) == ["in-profile"]
    assert _room_ids(shared) == ["in-shared"]
    assert SCHEMA <= tables(state_db) and SCHEMA <= tables(shared)
    # The shared file holds the hosted-room tables and the import marker, never the session store.
    assert not {"sessions", "messages", "state_meta"} & tables(shared)
    assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement="profile")] == ["in-profile"]
    assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement="shared")] == ["in-shared"]


def test_profiles_of_one_install_share_one_shared_file_and_keep_their_own_state_db(backend, scopes, root):
    store_a, store_b, store_root = (backend.open(scopes[name]) for name in ("a", "b", "root"))
    paths = {name: backend.open(scopes[name]).hosted_rooms._files.path("shared") for name in scopes}
    assert set(paths.values()) == {root / "shared-state.db"}
    _create(store_a, "together", "shared")
    _create(store_b, "alone", "profile")
    assert [room["room_id"] for room in store_b.hosted_rooms.list_rooms(placement="shared")] == ["together"]
    assert store_a.hosted_rooms.list_rooms(placement="profile") == []
    store_b.hosted_rooms.append_event(
        placement="shared", room_id="together", event_id="e1", kind="message.user", actor=USER, payload={"text": "hi"},
        authority_gateway_id="gw", authority_epoch=1, now=NOW + 1)
    assert store_root.hosted_rooms.read_events(placement="shared", room_id="together")["latest_seq"] == 1
    assert store_a.hosted_rooms.probe_hosted_room(placement="shared", room_id="together") is True
    assert store_a.hosted_rooms.probe_hosted_room(placement="profile", room_id="alone") is False
    assert _room_ids(scopes["b"].profile_dir / "state.db") == ["alone"]
    assert _room_ids(scopes["a"].profile_dir / "state.db") == []


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_a_room_repository_kept_past_the_backends_close_refuses_and_opens_no_file(backend, scopes, root, placement):
    rooms = backend.open(scopes["a"]).hosted_rooms
    backend.close()
    with pytest.raises(RuntimeError, match="used after close"):
        rooms.create_room(placement=placement, room_id="late", name="Late", members=MEMBERS, authority_gateway_id="gw",
                          now=NOW)
    with pytest.raises(RuntimeError, match="used after close"):
        rooms.list_rooms(placement=placement)
    with pytest.raises(RuntimeError, match="used after close"):
        rooms.probe_hosted_room(placement=placement, room_id="late")
    assert not (root / "shared-state.db").exists()
    assert not (scopes["a"].profile_dir / "state.db").exists()


def _pre_isolation_store(store) -> None:
    """The root profile's ``state.db`` holding rooms from before the shared file existed, written through the room verbs
    of the root profile's ``store``."""
    rooms = store.hosted_rooms
    rooms.create_room(placement="profile", room_id="old", name="Old", members=MEMBERS, authority_gateway_id="gw",
                      now=NOW - 90)
    rooms.append_event(placement="profile", room_id="old", event_id="e1", kind="message.user", actor=USER,
                       payload={"text": "kept"}, authority_gateway_id="gw", authority_epoch=1, now=NOW - 80)
    rooms.create_room(placement="profile", room_id="gone", name="Gone", members=[], authority_gateway_id="gw",
                      now=NOW - 70)
    rooms.disband_room(placement="profile", room_id="gone", expected_gateway_id="gw", expected_epoch=1, now=NOW - 60)


def test_the_import_runs_once_so_a_purge_is_never_undone(scopes, root):
    stores = {scope.hermes_home_key: scope.profile_dir / "state.db" for scope in scopes.values()}
    first = SQLiteBackend(stores)
    try:
        _pre_isolation_store(first.open(scopes["root"]))
        hosted = first.open(scopes["a"]).hosted_rooms
        assert hosted.room_state(placement="shared", room_id="old")["latest_seq"] == 1
        hosted.disband_room(placement="shared", room_id="old", expected_gateway_id="gw", expected_epoch=1, now=NOW)
        assert hosted.prune_disbanded_rooms(placement="shared", now=NOW + 400 * 86400) == 2
    finally:
        first.close()
        _registry.close_all()
    second = SQLiteBackend(stores)
    try:
        hosted = second.open(scopes["b"]).hosted_rooms
        assert hosted.list_rooms(placement="shared", include_disbanded=True) == []
        assert {row[0] for row in dump(root / "shared-state.db")["hosted_room_retired_ids"]} == {"old", "gone"}
    finally:
        second.close()


def test_a_profile_state_db_never_imports(backend, scopes, root):
    _pre_isolation_store(backend.open(scopes["root"]))
    store = backend.open(scopes["a"])
    assert store.hosted_rooms.list_rooms(placement="profile", include_disbanded=True) == []
    assert "hosted_room_legacy_imports" not in tables(scopes["a"].profile_dir / "state.db")


def _read_only(*scopes: ProfileScope) -> SQLiteBackend:
    return SQLiteBackend({scope.hermes_home_key: scope.profile_dir / "state.db" for scope in scopes}, read_only=True)


def test_a_read_only_store_creates_no_missing_shared_file(root):
    elsewhere = ProfileScope(profile_dir=root / "elsewhere" / "profiles" / "p", name="p")
    reader = _read_only(elsewhere)
    try:
        store = reader.open(elsewhere)
        with pytest.raises(StoreUnavailable):
            store.hosted_rooms.list_rooms(placement="shared")
        with pytest.raises(PermanentStoreError):
            _create(store, "refused", "shared")
        with pytest.raises(PermanentStoreError):
            store.hosted_room_attachments.prepare(placement="shared")
        assert not (root / "elsewhere").exists()
    finally:
        reader.close()


def test_a_read_only_store_reads_the_shared_file_and_never_writes_or_migrates_it(backend, scopes, root):
    _create(backend.open(scopes["a"]), "kept", "shared")
    backend.close()
    shared = root / "shared-state.db"
    reader = _read_only(*scopes.values())
    try:
        store = reader.open(scopes["b"])
        assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement="shared")] == ["kept"]
        before = dump(shared)
        with pytest.raises(PermanentStoreError):
            _create(store, "refused", "shared")
        assert dump(shared) == before
        # A file whose schema is behind is read by a writer first: the read-only store refuses and leaves it as it is.
        with sqlite3.connect(shared) as conn:
            conn.execute("DROP INDEX idx_hosted_room_events_cursor")
        with pytest.raises(PermanentStoreError):
            store.hosted_rooms.list_rooms(placement="shared")
        with sqlite3.connect(shared) as conn:
            assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='idx_hosted_room_events_cursor'").fetchone() is None
    finally:
        reader.close()
