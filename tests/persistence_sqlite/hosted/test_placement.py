"""Which file each placement reaches and when each hosted-room module's tables appear in it.

``'profile'`` is the bound profile's ``state.db``, ``'shared'`` the install's root ``shared-state.db`` that every profile of the install shares, and the first open of a ``shared-state.db`` imports the pre-isolation rooms of its sibling ``state.db`` once, as main's first open does. A verb creates only the tables main's operation creates, at the same operation, on both placements: each case runs the backend verb on the placement's file and main's operation on a twin and compares the hosted-room schema after every step.
"""

from __future__ import annotations

import pytest

import sqlite3
from pathlib import Path

from tests.persistence_sqlite._main_callers import hosted_room_attachments as m1_attachments
from persistence.gateway.hosted_rooms import HostedTaskIdentity, ReplicaPage
from persistence.sqlite import SQLiteBackend, _registry
from persistence.sqlite.gateway import hosted_room_attachments as backend_attachments
from session_rules.hosted_rooms import _validate_members
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted.conftest import NOW, copy_file, dump, outcome, tables

__all__ = ["legacy_store"]

pytestmark = pytest.mark.usefixtures("clock")

MEMBERS = [{"member_id": "m1", "profile": "p1"}]
USER = {"kind": "user", "id": "u1"}
# The tables the rooms step creates in every file a room verb opens.
SCHEMA = {"hosted_rooms", "hosted_room_events", "hosted_room_retired_ids", "hosted_room_links", "hosted_room_remote_runs",
          "hosted_room_revoked_grants", "hosted_room_peer_reservations"}
PAYLOAD = {"target_profile": "p1", "target_member_id": "m1", "prompt": "go", "source_event_seq": 1}
DRAFT_DRIVER = (
    "CREATE TABLE hosted_room_driver_leases (room_id TEXT PRIMARY KEY)",
    "CREATE TABLE hosted_room_driver_tasks (room_id TEXT NOT NULL, task_id TEXT NOT NULL,"
    " status TEXT CHECK (status IN ('queued', 'deferred', 'stopping')))",
    "CREATE INDEX idx_hosted_room_driver_tasks_status ON hosted_room_driver_tasks(room_id, status)")


@pytest.fixture
def m1(legacy_store):
    """Main's ``gateway.hosted_rooms`` from the pinned checkout, importable for the whole test."""
    return legacy_import("gateway.hosted_rooms")


def _room_ids(path: Path) -> list[str]:
    with sqlite3.connect(path) as conn:
        return [row[0] for row in conn.execute("SELECT room_id FROM hosted_rooms ORDER BY room_id")]


def _create(store, room_id: str, placement: str, now: float = NOW) -> None:
    store.hosted_rooms.create_room(placement=placement, room_id=room_id, name="Room", members=MEMBERS,
                                   authority_gateway_id="gw", now=now)


def _hosted_schema(path: Path) -> list[tuple[str, str]]:
    """The hosted-room tables, indexes and triggers of ``path``."""
    with sqlite3.connect(path) as conn:
        return sorted(conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name LIKE '%hosted_room%' OR name LIKE 'trg_work_%'"))


def _page() -> dict:
    event = {"seq": 1, "event_id": "e1", "kind": "message.user", "actor": USER, "authority_epoch": 1,
             "payload": {"text": "hi"}, "created_at": NOW}
    return {"events": [event], "authority": {"gateway_id": "gw-other", "epoch": 1}, "latest_seq": 1}


def _twin(backend, scope, placement: str, tmp_path: Path) -> tuple[Path, Path]:
    """The placement's file (the profile's ``state.db`` exists, the shared file does not yet) and main's twin path."""
    store = backend.open(scope)
    path = store.hosted_rooms._files.path(placement)
    legacy = tmp_path / "m1" / path.name
    legacy.parent.mkdir()
    if placement == "profile":
        store.ready()
        backend.reopen()
        _registry.close_all()
        copy_file(path, legacy)
    return path, legacy


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


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_each_verb_creates_the_tables_its_m1_operation_creates(backend, scopes, tmp_path, monkeypatch, placement,
                                                                m1):
    # CONTRACT: a verb runs exactly the schema steps of main's operation: a room verb never creates the replica,
    # driver, checkpoint, attachment or work-record tables, and each appears with the same operation as in main.
    m1_driver, m1_replicas, m1_checkpoint = (legacy_import(f"gateway.{name}") for name in (
        "hosted_room_driver", "hosted_room_replicas", "hosted_room_policy_checkpoint"))
    monkeypatch.setattr(m1_attachments, "_STARTED", set())
    monkeypatch.setattr(backend_attachments, "_STARTED", set())
    path, legacy = _twin(backend, scopes["a"], placement, tmp_path)
    store = backend.open(scopes["a"])
    identity = ("r1", "task-1", "thread-1", "turn-1")
    steps = [
        ("create_room",
         lambda: _create(store, "r1", placement),
         lambda: m1.create_room(legacy, room_id="r1", name="Room", members=MEMBERS, authority_gateway_id="gw", now=NOW)),
        ("append_event",
         lambda: store.hosted_rooms.append_event(
             placement=placement, room_id="r1", event_id="e1", kind="message.user", actor=USER,
             payload={"text": "go", "thread_id": "thread-1"}, authority_gateway_id="gw", authority_epoch=1, now=NOW),
         lambda: m1.append_event(
             legacy, room_id="r1", event_id="e1", kind="message.user", actor=USER,
             payload={"text": "go", "thread_id": "thread-1"}, authority_gateway_id="gw", authority_epoch=1, now=NOW)),
        ("ingest_page",
         lambda: store.hosted_room_replicas.ingest_page(
             placement=placement, room_id="rep", room_name="Rep", members_json=_validate_members(MEMBERS)[1],
             page=ReplicaPage(authority_gateway_id="gw-other", authority_epoch=1, events=tuple(_page()["events"]),
                              latest_seq=1), now=NOW),
         lambda: m1_replicas.ingest_page(legacy, room_id="rep", room_name="Rep", members=MEMBERS, page=_page(), now=NOW)),
        ("acquire_lease",
         lambda: store.hosted_room_driver.acquire_lease(
             placement=placement, room_id="r1", gateway_id="gw", authority_epoch=1, process_generation="p-1",
             ttl_seconds=60, now=NOW),
         lambda: m1_driver.acquire_lease(legacy, room_id="r1", gateway_id="gw", authority_epoch=1,
                                         process_generation="p-1", ttl_seconds=60, clock=lambda: NOW)),
        ("publication_exists",
         lambda: store.hosted_room_policy.publication_exists(
             placement=placement, room_id="r1", task_id="t", status="settled", execution_generation=0),
         lambda: m1_checkpoint.HostedRoomPolicyCheckpoint(legacy).publication_exists(
             room_id="r1", task_id="t", status="settled", execution_generation=0)),
        ("put",
         lambda: store.hosted_room_attachments.put(
             placement=placement, room_id="r1", upload_id="u1", kind="file", name="a.txt", mime="text/plain",
             data=b"x", now=NOW),
         lambda: m1_attachments.HostedRoomAttachmentStore(legacy, clock=lambda: NOW).put(
             room_id="r1", upload_id="u1", kind="file", name="a.txt", mime="text/plain", data=b"x")),
        ("admit_task",
         lambda: store.hosted_room_driver.admit_task(
             HostedTaskIdentity(*identity), placement=placement, payload=PAYLOAD, now=NOW),
         lambda: m1_driver.admit_task(legacy, m1_driver.TaskIdentity(*identity), payload=PAYLOAD, clock=lambda: NOW)),
    ]
    for name, verb, operation in steps:
        verb()
        operation()
        assert _hosted_schema(path) == _hosted_schema(legacy), name
        if name == "create_room":
            assert {table for kind, table in _hosted_schema(path) if kind == "table"} <= SCHEMA | {
                "hosted_room_legacy_imports"}
    assert {"hosted_room_replicas", "hosted_room_driver_tasks", "hosted_room_policy_cursors",
            "hosted_room_attachments", "hosted_room_work_records_source"} <= tables(path)


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_a_bad_driver_draft_fails_only_the_driver_verbs(backend, scopes, tmp_path, monkeypatch, placement, m1):
    # CONTRACT: an incompatible draft of the driver tables refuses the driver verbs with main's DriverStateError and
    # leaves every other hosted-room verb working, as main validates the draft only on its driver connection.
    m1_driver = legacy_import("gateway.hosted_room_driver")
    monkeypatch.setattr(m1_attachments, "_STARTED", set())
    monkeypatch.setattr(backend_attachments, "_STARTED", set())
    path, legacy = _twin(backend, scopes["a"], placement, tmp_path)
    store = backend.open(scopes["a"])
    _create(store, "r1", placement)
    m1.create_room(legacy, room_id="r1", name="Room", members=MEMBERS, authority_gateway_id="gw", now=NOW)
    for target in (path, legacy):
        with sqlite3.connect(target) as conn:
            for statement in DRAFT_DRIVER:
                conn.execute(statement)
    driver = outcome(lambda: store.hosted_room_driver.list_tasks(placement=placement, room_id="r1"))
    assert driver == outcome(lambda: m1_driver.list_tasks(legacy, room_id="r1"))
    assert driver[:2] == ("raised", "DriverStateError")
    assert store.hosted_rooms.append_event(
        placement=placement, room_id="r1", event_id="e1", kind="message.user", actor=USER, payload={"text": "hi"},
        authority_gateway_id="gw", authority_epoch=1, now=NOW)["seq"] == 1
    assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement=placement)] == ["r1"]
    assert store.hosted_room_policy.publication_exists(
        placement=placement, room_id="r1", task_id="t", status="settled", execution_generation=0) is False
    assert store.hosted_room_attachments.put(
        placement=placement, room_id="r1", upload_id="u1", kind="file", name="a.txt", mime="text/plain", data=b"x",
        now=NOW)["idempotent"] is False
    assert store.hosted_rooms.prune_disbanded_rooms(placement=placement, now=NOW) == 0


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


def test_the_first_shared_open_imports_the_pre_isolation_rooms_like_m1(backend, scopes, root, tmp_path, m1):
    _pre_isolation_store(backend.open(scopes["root"]))
    legacy_root = tmp_path / "m1"
    copy_file(root / "state.db", legacy_root / "state.db")
    m1.list_rooms(legacy_root / "shared-state.db", include_disbanded=True)
    rooms = backend.open(scopes["b"]).hosted_rooms.list_rooms(placement="shared", include_disbanded=True)
    assert [room["room_id"] for room in rooms] == ["gone", "old"]
    assert _hosted_schema(root / "shared-state.db") == _hosted_schema(legacy_root / "shared-state.db")
    names = [name for name in sorted(SCHEMA | {"hosted_room_legacy_imports"}) if name in tables(legacy_root / "shared-state.db")]
    assert dump(root / "shared-state.db", names) == dump(legacy_root / "shared-state.db", names)
    assert dump(root / "shared-state.db")["hosted_room_legacy_imports"] == [("state.db", NOW, 2)]


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
