"""The standalone->gateway handoff moves each imported room's standalone-era member
rows out of the ``Group: <room_id>`` title namespace, so the member's first gateway
turn can label its session instead of failing UNIQUE(title) after the row committed
(#149682)."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_rooms_gateway_import import (
    handoff_gateway_id,
    import_standalone_rooms,
    sweep_standalone_member_rows,
)
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from gateway.session_persistence import authority_persistence
from persistence.conversation.sessions import NewSession, TitleConflict
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import open_persistence, profile_scope, store_location

MEMBERS = [
    {"member_id": "a", "profile": "default", "handle": "a"},
    {"member_id": "b", "profile": "default", "handle": "b"},
]


def _seed_row(home: Path, sid: str, title: str, hidden: int = 1, source: str = "bot_room",
              title_source: str | None = None) -> None:
    with open_persistence(profile_scope(home=home)) as store:
        store.sessions.create(NewSession(session_id=sid, source=source))
    db_path = home / "state.db"
    with sqlite3.connect(db_path) as conn:
        if title_source is not None:
            conn.execute(
                "UPDATE sessions SET title=?, hidden=?, title_source=? WHERE id=?",
                (title, hidden, title_source, sid),
            )
        else:
            conn.execute("UPDATE sessions SET title=?, hidden=? WHERE id=?", (title, hidden, sid))


def _get_session(home: Path, sid: str):
    with open_persistence(profile_scope(home=home)) as store:
        return store.sessions.get(sid)


def _standalone_room(source: Path, room_id: str) -> None:
    with open_persistence(profile_scope(home=source.parent)) as store:
        hosted_rooms.create_room(
            store, placement="shared", room_id=room_id, name=room_id, members=MEMBERS,
            authority_gateway_id=hosted_rooms.local_authority_gateway_id())
        hosted_rooms.append_event(
            store, placement="shared", room_id=room_id, event_id=f"user:{room_id}", kind="message.user",
            actor={"kind": "user", "id": "desktop"}, payload={"text": "hello"},
            authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)


def _profile_room(home: Path, room_id: str, *, authority_gateway_id: str | None = None) -> None:
    with open_persistence(profile_scope(home=home)) as store:
        hosted_rooms.create_room(
            store, placement="profile", room_id=room_id, name=room_id,
            members=[{"member_id": "one", "profile": "default", "handle": "one"}],
            authority_gateway_id=authority_gateway_id or hosted_rooms.local_authority_gateway_id())


def _import_store(tmp_path: Path) -> Path:
    """A real gateway session store at ``state.db`` whose sibling ``shared-state.db``
    is the standalone backend's coordination store."""
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.get("__probe__")
    target = tmp_path / "state.db"
    assert target.is_file()
    return target


def test_import_moves_standalone_member_rows_out_of_the_title_namespace(tmp_path):
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    target = _import_store(tmp_path)
    # The standalone-era member row holding the room title (the store's single squatter:
    # UNIQUE(title) means only one row can ever hold it).
    _seed_row(tmp_path, "20261031_000426_6a3fd4", "Group: room-desk")
    # A squatter on the suffix title is bumped, not collided with.
    _seed_row(tmp_path, "my-notes", "Group: room-desk (standalone)", hidden=0, source="cli")

    assert import_standalone_rooms(target) == 1

    legacy = _get_session(tmp_path, "20261031_000426_6a3fd4")
    user = _get_session(tmp_path, "my-notes")
    assert legacy.title == "Group: room-desk (standalone 2)"        # bumped past the squatter
    assert user.title == "Group: room-desk (standalone)" and not user.hidden  # untouched

    # The room title is free: the member's first turn can now label its session.
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(session_id="gateway-member", source="bot_room"))
    with sqlite3.connect(target) as conn:
        conn.execute("UPDATE sessions SET title='Group: room-desk', hidden=1 WHERE id='gateway-member'")
    assert _get_session(tmp_path, "gateway-member").title == "Group: room-desk"

    # Restart is idempotent: no second suffix, no duplicate move.
    assert import_standalone_rooms(target) == 0
    assert _get_session(tmp_path, "20261031_000426_6a3fd4").title == "Group: room-desk (standalone 2)"


def test_a_user_titled_squatter_is_never_touched_and_the_error_surfaces(tmp_path):
    """A user-titled chat holding the room title keeps it: the import moves only
    ``bot_room`` rows, and the member's first turn then fails closed with the real
    storage cause (surfaced by the transport fix) instead of a fake drain."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    target = _import_store(tmp_path)
    _seed_row(tmp_path, "my-chat", "Group: room-desk", hidden=0, source="cli")

    import_standalone_rooms(target)
    row = _get_session(tmp_path, "my-chat")
    assert row.title == "Group: room-desk" and not row.hidden   # never touched

    # The title stays squatted: labeling a new session hits TitleConflict — fail-closed.
    from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.sessions.create(NewSession(session_id="gateway-member", source="bot_room"))
        epoch = begin_runtime_epoch(store._bound_store().holder(), instance_id="test")
        with pytest.raises(TitleConflict):
            store.hosted_rooms.label_room_session(
                epoch=epoch, session_id="gateway-member", title="Group: room-desk")


def test_import_leaves_bare_coordination_stores_alone(tmp_path):
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    target = _import_store(tmp_path)
    _seed_row(tmp_path, "member-row", "Group: room-desk")

    import_standalone_rooms(target)
    assert _get_session(tmp_path, "member-row").title == "Group: room-desk (standalone)"

    bare = tmp_path / "coord-only.db"  # no sessions table: nothing to move, no crash
    with sqlite3.connect(bare) as conn:
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    assert import_standalone_rooms(bare) == 0


def test_the_wire_surfaces_a_deterministic_storage_error(monkeypatch, tmp_path):
    """If a title conflict does survive (a user-titled chat squats the room title), the
    member's first turn fails with the real storage cause instead of runtime_draining."""
    import gateway.session_hosted_transport as t

    def canned(raw_error):
        payload = json.dumps({'ok': False, 'error': raw_error}).encode()
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
    with pytest.raises(AuthorityRefusal) as exc:
        t.owner_request(home, 'hosted-producer', {'operation': 'create'})
    assert exc.value.reason == 'sessions.title'          # real cause, not runtime_draining

    canned('')                                            # transport noise still folds
    with pytest.raises(AuthorityRefusal) as exc:
        t.owner_request(home, 'hosted-producer', {'operation': 'create'})
    assert exc.value.reason == 'runtime_draining'


def test_a_user_renamed_bot_room_row_keeps_its_title(tmp_path):
    """A bot_room row the USER renamed (title_source='user') is theirs: the import
    never moves it, and the member's first turn fails closed like any title squatter."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    target = _import_store(tmp_path)
    _seed_row(tmp_path, "renamed-by-user", "Group: room-desk", title_source="user")

    import_standalone_rooms(target)
    assert _get_session(tmp_path, "renamed-by-user").title == "Group: room-desk"  # kept


def test_upgrade_repair_sweeps_rooms_imported_before_this_fix(tmp_path):
    """Rooms the registry already knows (imported by an older gateway) get their
    straggler rows moved on the next start, even with nothing newly fenced."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    target = _import_store(tmp_path)
    assert import_standalone_rooms(target) == 1

    # A straggler appears afterwards (late standalone write, or a pre-fix upgrade):
    _seed_row(tmp_path, "straggler", "Group: room-desk")
    assert import_standalone_rooms(target) == 0  # nothing newly fenced…
    assert _get_session(tmp_path, "straggler").title == "Group: room-desk (standalone)"  # …but swept


def test_named_profile_stores_are_swept_for_claimed_rooms(tmp_path):
    """Members seated on named profiles keep their titled rows in their own stores,
    which have no import source: the ready sweep moves them for claimed rooms."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    target = _import_store(tmp_path)
    assert import_standalone_rooms(target) == 1

    named_home = tmp_path / "profiles" / "ops"
    named_home.mkdir(parents=True)
    _seed_row(named_home, "named-member", "Group: room-desk")  # UNIQUE(title): one squatter

    with open_persistence(profile_scope(home=tmp_path)) as root_store, \
            open_persistence(profile_scope(home=named_home)) as named_store:
        # epoch=None mirrors "no runtime epoch to guard" (production passes the live epoch).
        authorities = [SimpleNamespace(persistence=root_store, profile_id="default", epoch=None),
                       SimpleNamespace(persistence=named_store, profile_id="ops", epoch=None)]
        moved = sweep_standalone_member_rows(authorities)
        assert moved >= 1

    assert _get_session(named_home, "named-member").title == "Group: room-desk (standalone)"


def test_sweep_skips_rooms_demoted_to_a_peer(tmp_path):
    """A room whose authority moved to a peer keeps its member rows, even though the
    registry still knows the id: only currently-locally-claimed rooms are swept."""
    source = tmp_path / "shared-state.db"
    _standalone_room(source, "room-desk")
    target = _import_store(tmp_path)
    assert import_standalone_rooms(target) == 1

    # Demote: the coordination row moves to the peer (handoff) authority.
    with sqlite3.connect(target) as conn:
        conn.execute("UPDATE hosted_rooms SET authority_gateway_id=? WHERE room_id='room-desk'",
                     (handoff_gateway_id(),))
    _seed_row(tmp_path, "late-row", "Group: room-desk")
    assert import_standalone_rooms(target) == 0  # idempotence
    assert _get_session(tmp_path, "late-row").title == "Group: room-desk"  # peer room: untouched


def test_stranded_unlabeled_room_session_is_repaired_on_retry(hosted_owner, tmp_path, monkeypatch):
    """A create that failed its label before the sweep ran (title squatted) leaves an
    untitled committed row; the next create repairs it instead of stranding it."""
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True)
    store = authority_persistence(authority)
    own_home = store_location(store.scope).parent
    # Claim the room in the AUTHORITY's own registry (the default-profile case: the
    # claiming store is this store; the repair checks it inside the write transaction).
    _profile_room(own_home, "room")
    # Produce the stranded state through the real path: commit lands, label write dies.
    def crash_on_label(*a, **kw):
        raise RuntimeError('death between commit and label')
    with monkeypatch.context() as patch:
        patch.setattr(store.hosted_rooms, 'label_room_session', crash_on_label)
        try:
            rpc.create(profile="default", source="bot_room", title="Group: room")
        except RuntimeError:
            pass
    row = store.sessions.get(rpc.ref.session_id)
    assert row is not None and row.title is None  # committed, stranded untitled

    result = rpc.create(profile="default", source="bot_room", title="Group: room")
    fixed = store.sessions.get(rpc.ref.session_id)
    assert result["session_id"] == rpc.ref.session_id
    assert fixed.title == "Group: room" and fixed.hidden


def test_repair_never_touches_a_titled_or_user_renamed_row(hosted_owner, tmp_path, monkeypatch):
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True)
    store = authority_persistence(authority)
    own_db = store_location(store.scope)
    _profile_room(own_db.parent, "room")
    rpc.create(profile="default", source="bot_room", title="Group: room")  # real create+label
    with sqlite3.connect(own_db) as conn:
        conn.execute("UPDATE sessions SET title='My notes', hidden=0 WHERE id=?", (rpc.ref.session_id,))
    rpc.create(profile="default", source="bot_room", title="Group: room")  # retry: never overwrite
    row = store.sessions.get(rpc.ref.session_id)
    assert row.title == "My notes" and not row.hidden  # compare-and-set: never overwritten


def test_repair_fails_closed_when_the_room_is_not_locally_claimed(hosted_owner, tmp_path, monkeypatch):
    """Cross-file registry without our claim: the untitled row is left for the normal
    path rather than labeled for a room this gateway no longer serves."""
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True)
    store = authority_persistence(authority)
    own_home = store_location(store.scope).parent
    _profile_room(own_home, "room", authority_gateway_id=handoff_gateway_id())  # a PEER owns it
    store.sessions.create(NewSession(session_id=rpc.ref.session_id, source="bot_room"))  # untitled stranded row
    with pytest.raises(Exception):  # resume of a bare row cannot resolve -- expected
        rpc.create(profile="default", source="bot_room", title="Group: room")  # retry path
    assert store.sessions.get(rpc.ref.session_id).title is None  # untouched: fail closed


def test_named_profile_repair_uses_the_root_claiming_registry(hosted_owner, tmp_path):
    """A named-profile member's stranded row is repaired using the claim in the root
    authority's state.db — reached through the explicit claim_source (the binding's
    source_home in production), not any home-walk guess."""
    authority, loop, principal, _ = hosted_owner
    root = tmp_path / "install"
    (root / "profiles" / "ops").mkdir(parents=True)
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True,
                                 claim_source=root)  # the claiming service's home
    _profile_room(root, "room")
    store = authority_persistence(authority)
    store.sessions.create(NewSession(session_id=rpc.ref.session_id, source="bot_room"))  # untitled stranded row
    with pytest.raises(Exception):  # bare fixture row cannot resolve -- expected
        rpc.create(profile="default", source="bot_room", title="Group: room")
    fixed = store.sessions.get(rpc.ref.session_id)
    assert fixed.title == "Group: room" and fixed.hidden  # root claim authorized it


def test_cross_authority_repair_uses_the_binding_claim_source(hosted_owner, tmp_path):
    """The claim can live in any served authority's store, not only the root: the
    repair follows the explicit claim_source (the binding's source_home)."""
    authority, loop, principal, _ = hosted_owner
    claimer = tmp_path / "profiles" / "claim"
    claimer.mkdir(parents=True)
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id="room", member_id="one", profile="default",
                                 principal=principal, authorize=lambda *a: True,
                                 claim_source=claimer)
    _profile_room(claimer, "room")
    store = authority_persistence(authority)
    store.sessions.create(NewSession(session_id=rpc.ref.session_id, source="bot_room"))  # untitled stranded row
    with pytest.raises(Exception):  # bare fixture row cannot resolve -- expected
        rpc.create(profile="default", source="bot_room", title="Group: room")
    fixed = store.sessions.get(rpc.ref.session_id)
    assert fixed.title == "Group: room" and fixed.hidden
