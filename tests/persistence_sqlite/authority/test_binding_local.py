"""``SessionBindingRepository`` on SQLite: the local reservation, the untracked-compression adoption, the local ``/new``
reset, the idle end and the boot hand-off of receipts a messaging lane took over each open exactly one write
transaction, which a failure right after its first mutating statement leaves unchanged. The reservation's ``created``
is True only when its own write inserted the receipt.
"""

from __future__ import annotations

import json
from typing import Any

from persistence.authority.bindings import NewSessionBinding, local_session_id
from persistence.core.identity import SessionId
from tests.persistence_sqlite.authority._admission_support import begin_epoch
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_write, build_seed

PROFILE = "default"
PRINCIPAL = "user-1"
OWNER = "owner-a"
CREATED_AT = "2026-09-16T10:00:00+00:00"
SID = str(local_session_id(PROFILE, PRINCIPAL, "req-1"))


def make_binding(session_id: str = SID, *, request_id: str = "req-1", route: str | None = None,
                 request_json: str = '{"kind":"launch","n":1}', policy: dict | None = None, source: str = "cli") -> NewSessionBinding:
    from gateway.session_policy import created_surface

    route = route or f"agent:main:local:dm:{session_id}"
    return NewSessionBinding(
        profile_id=PROFILE, principal_id=PRINCIPAL, request_id=request_id, session_id=SessionId(session_id), route=route,
        entry={"session_key": route, "session_id": session_id, "created_at": CREATED_AT, "updated_at": CREATED_AT,
               "origin": {"chat_id": session_id, "chat_type": "dm", "user_id": PRINCIPAL, "platform": "local"}},
        policy={"source": source, "model": "model-a", "cwd": "/workspace", "request_json": request_json, **(policy or {})},
        created_source=created_surface(source) or source)


def seed(store: Any, *steps: Any) -> None:
    """The epoch, then each ``step(db)``, through the backend."""
    def run(db: Any) -> None:
        begin_epoch(db, OWNER)
        for step in steps:
            step(db)
    build_seed(store, run)


def reserved(binding: NewSessionBinding):
    """A seed step reserving ``binding`` through the verb."""
    def step(db: Any) -> None:
        db.store.bindings.reserve(epoch=1, binding=binding, owner=None)
    return step


def meta(key: str, value: Any):
    text = value if isinstance(value, str) else json.dumps(value)
    return lambda db: db.store.holder().set_meta(key, text)


# ── reserve ───────────────────────────────────────────────────────────────────


def test_reserve_created_is_false_on_the_stored_receipt_replay(store):
    seed(store)
    first = store.bindings.reserve(epoch=1, binding=make_binding(), owner=None)
    again = store.bindings.reserve(epoch=1, binding=make_binding(), owner=None)
    assert (first.created, again.created) == (True, False)
    assert again.binding == first.binding


def test_reserve_is_one_write(store):
    seed(store)
    assert_one_write(store, lambda: store.bindings.reserve(epoch=1, binding=make_binding(), owner=None))


# ── a reserved lineage ────────────────────────────────────────────────────────


def _lineage_seed(store: Any) -> None:
    """A reserved local session advanced once by a ``/new`` reset, plus a second session without lineage."""
    from persistence.sqlite.conversation._store.local import POLICY_PREFIX

    def reset(db: Any) -> None:
        db.store.bindings.reset_target(epoch=1, parent_session_id=SessionId(SID), entry=_child_entry("child-1"))

    second = make_binding(str(local_session_id(PROFILE, PRINCIPAL, "req-2")), request_id="req-2")
    seed(store, reserved(make_binding()), reserved(second), reset, meta(POLICY_PREFIX + "broken", "nope"))


def _child_entry(child: str) -> dict[str, Any]:
    route = f"agent:main:local:dm:{SID}"
    return {"session_key": route, "session_id": child, "created_at": CREATED_AT, "updated_at": CREATED_AT,
            "origin": {"chat_id": SID, "chat_type": "dm", "user_id": PRINCIPAL, "platform": "local"}}


# ── untracked compression adoption ────────────────────────────────────────────


def _untracked_step(db: Any, *, parent: str = SID, child: str = "cmp-1", identity: tuple | None = None) -> None:
    """An older writer's compression step past the receipt's tip: the tip ends ``compression`` and the child copies its
    route, owner and principal, but the receipt never advances."""
    route, chat_id, user_id = identity or (f"agent:main:local:dm:{SID}", SID, PRINCIPAL)
    db.create_session(child, source="cli", parent_session_id=parent)

    def write(conn: Any) -> None:
        conn.execute("UPDATE sessions SET ended_at=1.0, end_reason='compression' WHERE id=?", (parent,))
        conn.execute("UPDATE sessions SET session_key=?, chat_id=?, user_id=? WHERE id=?", (route, chat_id, user_id, child))
    db._execute_write(write)


def test_adopt_untracked_compression_is_one_write(store):
    seed(store, reserved(make_binding()), _untracked_step)
    expected = store.bindings.receipt(SID)
    assert_one_write(store, lambda: store.bindings.adopt_untracked_compression(
        epoch=1, session_id=SessionId(SID), expected=expected))


# ── reset_target ──────────────────────────────────────────────────────────────


def test_reset_target_is_one_write(store):
    seed(store, reserved(make_binding()))
    assert_one_write(store, lambda: store.bindings.reset_target(epoch=1, parent_session_id=SessionId(SID), entry=_child_entry("child-2")))


# ── end_idle ──────────────────────────────────────────────────────────────────


def test_end_idle_is_one_write(store):
    seed(store, reserved(make_binding()))
    assert_one_write(store, lambda: store.bindings.end_idle(epoch=1, session_id=SessionId(SID), target_id=SessionId(SID), reason="idle"))


# ── release_handed_off_receipts ───────────────────────────────────────────────


def _handed_off(db: Any) -> None:
    """A ``/handoff`` by an older build: the receipt's current target row now carries a Telegram lane's route."""
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET session_key='agent:main:telegram:dm:42', source='telegram', chat_id='42' WHERE id='child-1'"))


def test_release_handed_off_receipts_is_one_write(store):
    _lineage_seed(store)
    build_seed(store, _handed_off)
    assert_one_write(store, lambda: store.bindings.release_handed_off_receipts(epoch=1, lanes=frozenset({"telegram"})))
