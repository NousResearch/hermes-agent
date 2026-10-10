"""Durable recovery of a gateway route historically switched onto a delegate child."""

from datetime import datetime
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore
from hermes_state import SessionDB


@pytest.fixture
def route(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = SessionSource(
        Platform.DISCORD, chat_id="thread-106742", chat_type="thread",
        user_id="axl", thread_id="thread-106742", scope_id="guild-1",
    )
    key = store._generate_session_key(source)
    peer = dict(
        source="discord", session_key=key, user_id=source.user_id,
        chat_id=source.chat_id, chat_type=source.chat_type,
        thread_id=source.thread_id, origin_json=json.dumps(source.to_dict()),
    )
    yield store, db, source, key, peer
    db.close()


def _stamp(db, session_id, **values):
    columns = ", ".join(f"{column} = ?" for column in values)
    with db._lock:
        db._conn.execute(
            f"UPDATE sessions SET {columns} WHERE id = ?",
            (*values.values(), session_id),
        )
        db._conn.commit()


def _entry(key, session_id, source):
    now = datetime.now()
    return SessionEntry(
        session_key=key, session_id=session_id, created_at=now, updated_at=now,
        origin=source, platform=source.platform, chat_type=source.chat_type,
    )


def _owner(db, peer, *, session_id="owner"):
    db.create_session(session_id, **peer)
    _stamp(db, session_id, started_at=100.0)
    return session_id


def _delegate(db, parent_id, peer, *, session_id="delegate", started_at=200.0,
              stamp_peer=True, mark_delegate=True):
    db.create_session(
        session_id, "subagent", parent_session_id=parent_id,
        model_config={"_delegate_from": parent_id} if mark_delegate else None,
    )
    _stamp(db, session_id, started_at=started_at)
    if stamp_peer:
        # Historical gateway writer stamped the child before the provenance guard existed.
        # Bypass today's guarded writer only to recreate that persisted legacy state.
        _stamp(
            db, session_id, source=peer["source"], session_key=peer["session_key"],
            user_id=peer["user_id"], chat_id=peer["chat_id"],
            chat_type=peer["chat_type"], thread_id=peer["thread_id"],
            origin_json=peer["origin_json"],
        )
    return session_id


def _verdict(route, session_id):
    store, db, source, key, _ = route
    return store._poisoned_delegate_route_verdict(
        session_key=key, entry=_entry(key, session_id, source), source=source, db=db,
    )


def test_switch_stamped_owner_of_direct_delegate_is_recoverable(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer, mark_delegate=False)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")

    assert db.get_session(child)["source"] == "discord"
    assert db.get_session(child)["created_source"] == "subagent"
    verdict = _verdict(route, child)
    assert (verdict.kind, verdict.owner_id) == ("recoverable", owner)


def test_nested_delegate_recovers_nearest_real_ancestor(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    first = _delegate(db, owner, peer, session_id="delegate-1", stamp_peer=False)
    second = _delegate(db, first, peer, session_id="delegate-2", started_at=250.0)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")

    verdict = _verdict(route, second)
    assert (verdict.kind, verdict.owner_id) == ("recoverable", owner)


def test_shared_thread_new_sender_recovers_same_route_owner(route):
    store, db, source, key, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer, mark_delegate=False)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")
    next_sender = SessionSource(
        Platform.DISCORD, chat_id=source.chat_id, chat_type="thread",
        user_id="another-authorized-sender", thread_id=source.thread_id,
        scope_id=source.scope_id,
    )
    assert store._generate_session_key(next_sender) == key

    verdict = store._poisoned_delegate_route_verdict(
        session_key=key, entry=_entry(key, child, next_sender), source=next_sender, db=db,
    )
    assert (verdict.kind, verdict.owner_id) == ("recoverable", owner)


def test_prospective_discord_thread_continuation_recovers_owner(route):
    """The channel initiator and its later thread reply share one canonical route."""
    store, db, _, _, _ = route
    initiating = SessionSource(
        Platform.DISCORD, chat_id="channel-1", chat_type="group", user_id="axl",
        prospective_thread_id="msg-100", scope_id="guild-1",
    )
    follow_up = SessionSource(
        Platform.DISCORD, chat_id="channel-1", chat_type="thread", user_id="axl",
        thread_id="msg-100", scope_id="guild-1",
    )
    key = store._generate_session_key(initiating)
    assert key == store._generate_session_key(follow_up)
    peer = dict(
        source="discord", session_key=key, user_id=initiating.user_id,
        chat_id=initiating.chat_id, chat_type=initiating.chat_type,
        thread_id=initiating.thread_id,
        origin_json=json.dumps(initiating.to_dict()),
    )
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")

    verdict = store._poisoned_delegate_route_verdict(
        session_key=key, entry=_entry(key, child, follow_up), source=follow_up, db=db,
    )
    assert (verdict.kind, verdict.owner_id) == ("recoverable", owner)

    # Another prospective id must never be accepted as this route's peer.
    _stamp(db, owner, origin_json=json.dumps({**initiating.to_dict(),
                                              "prospective_thread_id": "another-thread"}))
    verdict = store._poisoned_delegate_route_verdict(
        session_key=key, entry=_entry(key, child, follow_up), source=follow_up, db=db,
    )
    assert verdict.kind == "invalid"


@pytest.mark.parametrize("tamper", ["no_parent", "different_key", "different_peer"])
def test_unrelated_ancestor_cannot_be_guessed(route, tamper):
    _, db, _, key, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    if tamper == "no_parent":
        _stamp(db, child, parent_session_id=None)
    elif tamper == "different_key":
        _stamp(db, owner, session_key="agent:main:discord:dm:other")
    else:
        _stamp(db, owner, chat_id="other-thread")

    verdict = _verdict(route, child)
    assert verdict.kind == "invalid"
    assert verdict.owner_id is None


def test_later_explicit_boundary_on_another_normal_row_blocks_recovery(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")
    db.create_session("later-normal", **{**peer, "user_id": "another-authorized-sender"})
    _stamp(db, "later-normal", started_at=400.0, ended_at=500.0, end_reason="session_reset")

    verdict = _verdict(route, child)
    assert (verdict.kind, verdict.owner_id) == ("invalid", None)


@pytest.mark.parametrize("boundary_id,boundary_reason", [
    ("owner", "session_reset"),
    ("delegate", "session_reset"),
])
def test_explicit_boundary_on_ancestor_or_delegate_blocks_recovery(route, boundary_id, boundary_reason):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    _stamp(db, boundary_id, ended_at=300.0, end_reason=boundary_reason)

    verdict = _verdict(route, child)
    assert (verdict.kind, verdict.owner_id) == ("invalid", None)


def test_owner_reset_before_child_birth_is_still_a_boundary(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    _stamp(db, owner, ended_at=150.0, end_reason="session_reset")

    verdict = _verdict(route, child)
    assert (verdict.kind, verdict.owner_id) == ("invalid", None)


def test_newer_normal_route_for_same_peer_blocks_old_owner(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")
    db.create_session("new-owner", **peer)
    _stamp(db, "new-owner", started_at=400.0)

    verdict = _verdict(route, child)
    assert (verdict.kind, verdict.owner_id) == ("invalid", None)


def test_another_active_normal_route_blocks_old_owner_even_if_older(route):
    _, db, _, _, peer = route
    db.create_session("older-active", **peer)
    _stamp(db, "older-active", started_at=50.0)
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")

    verdict = _verdict(route, child)
    assert (verdict.kind, verdict.owner_id) == ("invalid", None)


def test_delegate_marker_recovers_old_row_without_created_source(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    _stamp(db, child, created_source=None)
    _stamp(db, owner, ended_at=300.0, end_reason="session_switch")

    verdict = _verdict(route, child)
    assert (verdict.kind, verdict.owner_id) == ("recoverable", owner)


def test_missing_or_unreadable_database_never_returns_an_owner(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    child = _delegate(db, owner, peer)
    store, _, source, key, _ = route
    entry = _entry(key, child, source)

    assert store._poisoned_delegate_route_verdict(
        session_key=key, entry=entry, source=source, db=None,
    ).kind == "unverified"

    class BrokenDB:
        def _read_all(self, *_args):
            raise OSError("database unavailable")

    verdict = store._poisoned_delegate_route_verdict(
        session_key=key, entry=entry, source=source, db=BrokenDB(),
    )
    assert (verdict.kind, verdict.owner_id) == ("unverified", None)

    missing_row = store._poisoned_delegate_route_verdict(
        session_key=key, entry=_entry(key, "missing-row", source), source=source, db=db,
    )
    assert (missing_row.kind, missing_row.owner_id) == ("unverified", None)


def test_ordinary_gateway_row_is_not_a_poisoned_delegate(route):
    _, db, _, _, peer = route
    owner = _owner(db, peer)
    verdict = _verdict(route, owner)
    assert (verdict.kind, verdict.owner_id) == ("not_delegate", None)
