"""The standalone->gateway handoff moves each imported room's standalone-era member
rows out of the ``Group: <room_id>`` title namespace, so the member's first gateway
turn can label its session instead of failing UNIQUE(title) after the row committed
(#149682)."""

import sqlite3
from pathlib import Path

from gateway import hosted_rooms
from gateway.hosted_rooms_gateway_import import handoff_gateway_id, import_standalone_rooms
from hermes_state import SessionDB

MEMBERS = [{"member_id": "a", "profile": "default", "handle": "a"},
           {"member_id": "b", "profile": "default", "handle": "b"}]


def _seed_row(db, sid, title, hidden=1, source="bot_room"):
    db.create_session(sid, source)
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET title=?, hidden=? WHERE id=?", (title, hidden, sid)))


def _standalone_room(source: Path, room_id: str) -> None:
    hosted_rooms.create_room(source, room_id=room_id, name=room_id, members=MEMBERS,
                             authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    hosted_rooms.append_event(
        source, room_id=room_id, event_id=f"user:{room_id}", kind="message.user",
        actor={"kind": "user", "id": "desktop"}, payload={"text": "hello"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)


def _import_store(tmp_path: Path) -> SessionDB:
    """A real gateway session store at ``state.db`` whose sibling ``shared-state.db``
    is the standalone backend's coordination store."""
    target = tmp_path / "state.db"
    db = SessionDB(db_path=target)
    assert target.name == "state.db"
    return db


def test_import_moves_standalone_member_rows_out_of_the_title_namespace(tmp_path):
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    db = _import_store(tmp_path)
    # The standalone-era member row holding the room title (the store's single squatter:
    # UNIQUE(title) means only one row can ever hold it).
    _seed_row(db, "20261031_000426_6a3fd4", "Group: room-desk")
    # A squatter on the suffix title is bumped, not collided with.
    _seed_row(db, "my-notes", "Group: room-desk (standalone)", hidden=0, source="cli")

    assert import_standalone_rooms(tmp_path / "state.db") == 1

    legacy = db.get_session("20261031_000426_6a3fd4")
    user = db.get_session("my-notes")
    assert legacy["title"] == "Group: room-desk (standalone 2)"        # bumped past the squatter
    assert user["title"] == "Group: room-desk (standalone)" and user["hidden"] == 0  # untouched

    # The room title is free: the member's first turn can now label its session.
    db.create_session("gateway-member", "bot_room")
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET title='Group: room-desk', hidden=1 WHERE id='gateway-member'"))
    assert db.get_session("gateway-member")["title"] == "Group: room-desk"

    # Restart is idempotent: no second suffix, no duplicate move.
    assert import_standalone_rooms(tmp_path / "state.db") == 0
    assert db.get_session("20261031_000426_6a3fd4")["title"] == "Group: room-desk (standalone 2)"


def test_a_user_titled_squatter_is_never_touched_and_the_error_surfaces(tmp_path):
    """A user-titled chat holding the room title keeps it: the import moves only
    ``bot_room`` rows, and the member's first turn then fails closed with the real
    storage cause (surfaced by the transport fix) instead of a fake drain."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    db = _import_store(tmp_path)
    _seed_row(db, "my-chat", "Group: room-desk", hidden=0, source="cli")

    import_standalone_rooms(tmp_path / "state.db")
    row = db.get_session("my-chat")
    assert row["title"] == "Group: room-desk" and row["hidden"] == 0   # never touched

    # The title stays squatted: labeling a new session hits UNIQUE — by design, the
    # same fail-closed contract the API room dispatch already uses for title conflicts.
    db.create_session("gateway-member", "bot_room")
    import sqlite3 as _sqlite3
    try:
        db._execute_write(lambda c: c.execute(
            "UPDATE sessions SET title='Group: room-desk', hidden=1 WHERE id='gateway-member'"))
        raised = False
    except _sqlite3.IntegrityError:
        raised = True
    assert raised


def test_import_leaves_bare_coordination_stores_alone(tmp_path):
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    db = _import_store(tmp_path)
    _seed_row(db, "member-row", "Group: room-desk")

    import_standalone_rooms(tmp_path / "state.db")
    assert db.get_session("member-row")["title"] == "Group: room-desk (standalone)"

    bare = tmp_path / "coord-only.db"  # no sessions table: nothing to move, no crash
    with sqlite3.connect(bare) as conn:
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    assert import_standalone_rooms(bare) == 0


def test_the_wire_surfaces_a_deterministic_storage_error(monkeypatch, tmp_path):
    """If a title conflict does survive (a user-titled chat squats the room title), the
    member's first turn fails with the real storage cause instead of runtime_draining."""
    import json as _json
    import gateway.session_hosted_transport as t
    from hermes_state_runtime import RuntimeStoreError

    def canned(raw_error):
        payload = _json.dumps({'ok': False, 'error': raw_error}).encode()
        from types import SimpleNamespace
        monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(
                                state='ready', endpoint=SimpleNamespace(profile_id=str(Path(home)))))
        monkeypatch.setattr('hermes_cli.gateway_runtime.control_home_for', lambda home, endpoint: home)

        class _Peer:
            def sendall(self, _): pass

            def settimeout(self, _): pass

            def __enter__(self): return self

            def __exit__(self, *a): return False
        monkeypatch.setattr('hermes_cli.gateway_runtime_discovery.connect_private',
                            lambda home, timeout: _Peer())
        monkeypatch.setattr('gateway.control_socket._read_response_line',
                            lambda read, deadline: payload)

    home = str(tmp_path)
    canned('IntegrityError: UNIQUE constraint failed: sessions.title')
    import pytest
    with pytest.raises(RuntimeStoreError) as exc:
        t.owner_request(home, 'hosted-producer', {'operation': 'create'})
    assert exc.value.reason == 'sessions.title'          # real cause, not runtime_draining

    canned('')                                            # transport noise still folds
    with pytest.raises(RuntimeStoreError) as exc:
        t.owner_request(home, 'hosted-producer', {'operation': 'create'})
    assert exc.value.reason == 'runtime_draining'


def test_a_user_renamed_bot_room_row_keeps_its_title(tmp_path):
    """A bot_room row the USER renamed (title_source='user') is theirs: the import
    never moves it, and the member's first turn fails closed like any title squatter."""
    from types import SimpleNamespace
    from gateway.hosted_rooms_gateway_import import sweep_standalone_member_rows
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    db = _import_store(tmp_path)
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET title_source='user' WHERE id='20261031_000426_6a3fd4'"
    )) if False else None
    _seed_row(db, "renamed-by-user", "Group: room-desk")
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET title_source='user' WHERE id='renamed-by-user'"))

    import_standalone_rooms(tmp_path / "state.db")
    assert db.get_session("renamed-by-user")["title"] == "Group: room-desk"  # kept


def test_upgrade_repair_sweeps_rooms_imported_before_this_fix(tmp_path):
    """Rooms the registry already knows (imported by an older gateway) get their
    straggler rows moved on the next start, even with nothing newly fenced."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    db = _import_store(tmp_path)
    assert import_standalone_rooms(tmp_path / "state.db") == 1

    # A straggler appears afterwards (late standalone write, or a pre-fix upgrade):
    _seed_row(db, "straggler", "Group: room-desk")
    assert import_standalone_rooms(tmp_path / "state.db") == 0  # nothing newly fenced…
    assert db.get_session("straggler")["title"] == "Group: room-desk (standalone)"  # …but swept


def test_named_profile_stores_are_swept_for_claimed_rooms(tmp_path):
    """Members seated on named profiles keep their titled rows in their own stores,
    which have no import source: the ready sweep moves them for claimed rooms."""
    from types import SimpleNamespace
    from gateway.hosted_rooms_gateway_import import sweep_standalone_member_rows
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    _import_store(tmp_path)
    assert import_standalone_rooms(tmp_path / "state.db") == 1

    named = tmp_path / "profiles" / "ops" / "state.db"
    named.parent.mkdir(parents=True)
    named_db = SessionDB(db_path=named)
    _seed_row(named_db, "named-member", "Group: room-desk")  # UNIQUE(title): one squatter
    named_db.close()

    root_reopened = SessionDB(db_path=tmp_path / "state.db")
    # epoch=None mirrors "no runtime epoch to guard" (production passes the live epoch).
    authorities = [SimpleNamespace(db=root_reopened, epoch=None),
                   SimpleNamespace(db=SessionDB(db_path=named), epoch=None)]
    moved = sweep_standalone_member_rows(authorities)
    assert moved >= 1
    root_reopened.close()

    named_check = SessionDB(db_path=named)
    try:
        assert named_check.get_session("named-member")["title"] == "Group: room-desk (standalone)"
    finally:
        named_check.close()


def test_sweep_skips_rooms_demoted_to_a_peer(tmp_path):
    """A room whose authority moved to a peer keeps its member rows, even though the
    registry still knows the id: only currently-locally-claimed rooms are swept."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    db = _import_store(tmp_path)
    assert import_standalone_rooms(tmp_path / "state.db") == 1

    # Demote: the coordination row moves to the peer (handoff) authority.
    import sqlite3 as _sq
    with _sq.connect(tmp_path / "state.db") as conn:
        conn.execute("UPDATE hosted_rooms SET authority_gateway_id=? WHERE room_id='room-desk'",
                     (handoff_gateway_id(),))
    _seed_row(db, "late-row", "Group: room-desk")
    assert import_standalone_rooms(tmp_path / "state.db") == 0  # idempotence
    assert db.get_session("late-row")["title"] == "Group: room-desk"  # peer room: untouched


def test_stranded_unlabeled_room_session_is_repaired_on_retry(hosted_owner, tmp_path, monkeypatch):
    """A create that failed its label before the sweep ran (title squatted) leaves an
    untitled committed row; the next create repairs it instead of stranding it."""
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True)
    # Claim the room in the AUTHORITY's own registry (the default-profile case: the
    # claiming store is this store; the repair checks it inside the write transaction).
    from gateway import hosted_rooms as _hr
    _hr.create_room(authority.db.db_path, room_id="room", name="room",
                    members=[{"member_id": "one", "profile": "default", "handle": "one"}],
                    authority_gateway_id=_hr.local_authority_gateway_id())
    # Produce the stranded state through the real path: commit lands, label write dies.
    real_write = authority.db._execute_write
    def crash_on_label(fn, *a, **kw):
        if getattr(fn, '__name__', '') == 'label':
            raise RuntimeError('death between commit and label')
        return real_write(fn, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(authority.db, '_execute_write', crash_on_label)
        try:
            rpc.create(profile="default", source="bot_room", title="Group: room")
        except RuntimeError:
            pass
    row = authority.db.get_session(rpc.ref.session_id)
    assert row is not None and row["title"] is None  # committed, stranded untitled

    result = rpc.create(profile="default", source="bot_room", title="Group: room")
    fixed = authority.db.get_session(rpc.ref.session_id)
    assert result["session_id"] == rpc.ref.session_id
    assert fixed["title"] == "Group: room" and fixed["hidden"]


def test_repair_never_touches_a_titled_or_user_renamed_row(hosted_owner, tmp_path, monkeypatch):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True)
    # Claim the room in the AUTHORITY's own registry (the default-profile case: the
    # claiming store is this store; the repair checks it inside the write transaction).
    from gateway import hosted_rooms as _hr
    _hr.create_room(authority.db.db_path, room_id="room", name="room",
                    members=[{"member_id": "one", "profile": "default", "handle": "one"}],
                    authority_gateway_id=_hr.local_authority_gateway_id())
    rpc.create(profile="default", source="bot_room", title="Group: room")  # real create+label
    authority.db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET title='My notes', hidden=0 WHERE id=?", (rpc.ref.session_id,)))
    rpc.create(profile="default", source="bot_room", title="Group: room")  # retry: never overwrite
    row = authority.db.get_session(rpc.ref.session_id)
    assert row["title"] == "My notes" and not row["hidden"]  # compare-and-set: never overwritten


def test_repair_fails_closed_when_the_room_is_not_locally_claimed(hosted_owner, tmp_path, monkeypatch):
    """Cross-file registry without our claim: the untitled row is left for the normal
    path rather than labeled for a room this gateway no longer serves."""
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True)
    from gateway import hosted_rooms as _hr
    _hr.create_room(authority.db.db_path, room_id="room", name="room",
                    members=[{"member_id": "one", "profile": "default", "handle": "one"}],
                    authority_gateway_id=handoff_gateway_id())  # a PEER owns it
    authority.db.create_session(rpc.ref.session_id, "bot_room")  # untitled stranded row
    import pytest as _pytest
    with _pytest.raises(Exception):  # resume of a bare row cannot resolve -- expected
        rpc.create(profile="default", source="bot_room", title="Group: room")  # retry path
    assert authority.db.get_session(rpc.ref.session_id)["title"] is None  # untouched: fail closed


def test_named_profile_repair_uses_the_root_claiming_registry(hosted_owner, tmp_path):
    """A named-profile member's stranded row is repaired using the claim in the root
    authority's state.db — reached through the explicit claim_source (the binding's
    source_home in production), not any home-walk guess."""
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    authority, loop, principal, _ = hosted_owner
    root = tmp_path / "install"
    (root / "profiles" / "ops").mkdir(parents=True)
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True,
                                 claim_source=root)  # the claiming service's home
    from gateway import hosted_rooms as _hr
    _hr.create_room(root / "state.db", room_id="room", name="room",
                    members=[{"member_id": "one", "profile": "default", "handle": "one"}],
                    authority_gateway_id=_hr.local_authority_gateway_id())
    authority.db.create_session(rpc.ref.session_id, "bot_room")  # untitled stranded row
    import pytest as _pytest
    with _pytest.raises(Exception):  # bare fixture row cannot resolve -- expected
        rpc.create(profile="default", source="bot_room", title="Group: room")
    fixed = authority.db.get_session(rpc.ref.session_id)
    assert fixed["title"] == "Group: room" and fixed["hidden"]  # root claim authorized it


def test_cross_authority_repair_uses_the_binding_claim_source(hosted_owner, tmp_path):
    """The claim can live in any served authority's store, not only the root: the
    repair follows the explicit claim_source (the binding's source_home)."""
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    authority, loop, principal, _ = hosted_owner
    claimer = tmp_path / "profiles" / "claim"
    claimer.mkdir(parents=True)
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True,
                                 claim_source=claimer)
    from gateway import hosted_rooms as _hr
    _hr.create_room(claimer / "state.db", room_id="room", name="room",
                    members=[{"member_id": "one", "profile": "default", "handle": "one"}],
                    authority_gateway_id=_hr.local_authority_gateway_id())
    authority.db.create_session(rpc.ref.session_id, "bot_room")  # untitled stranded row
    import pytest as _pytest
    with _pytest.raises(Exception):  # bare fixture row cannot resolve -- expected
        rpc.create(profile="default", source="bot_room", title="Group: room")
    fixed = authority.db.get_session(rpc.ref.session_id)
    assert fixed["title"] == "Group: room" and fixed["hidden"]
