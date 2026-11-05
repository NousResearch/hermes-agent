"""``SessionBindingRepository`` on SQLite against main: the API route binding, its stored receipt, the declared key
registry and the history-control authorization.

``bind_api`` runs the write main's ``bind_api_steps`` yields on one copy of a seed and the verb, built from the
values main's steps compute, on the other. The admission gate is the owner's verdict passed as data and raised right
after the epoch check, where main's write re-runs ``_require_admission_open``; ``walked_from`` is walked over
``api_generation_id(walked_from, n)`` inside the same write. ``authorize_history`` runs main's ``require_history_claim``
and ``authorize_history`` in one read, with the native transport registry's answer as data.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from persistence import FenceStale
from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import NewApiBinding, api_generation_id
from persistence.core.identity import SessionId
from tests.persistence_sqlite.authority._admission_support import begin_epoch
from tests.persistence_sqlite.authority._support import assert_same, run_both
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_read, assert_one_write, build_seed

OWNER = "owner-a"
API_SID = "api-conv-1"


def seed(store: Any, *steps: Any, legacy: bool = True) -> None:
    def run(db: Any) -> None:
        begin_epoch(db, OWNER)
        for step in steps:
            step(db)
    build_seed(store, run, legacy=legacy)


def meta(key: str, value: Any):
    return lambda db: db.set_meta(key, value if isinstance(value, str) else json.dumps(value))


class _Authority:
    """The owner attributes main's ``bind_api_steps`` reads: a non-multiplexed runner and an admission gate that closes
    after ``open_checks`` checks."""

    def __init__(self, *, open_checks: Optional[int] = None) -> None:
        self.db = SimpleNamespace(_execute_write=print)  # the steps name the writer before any store is open
        self.epoch = 1
        self.profile_id = "default"
        self.sessions: dict[str, Any] = {}
        self.runner = SimpleNamespace(
            config=SimpleNamespace(multiplex_profiles=False), _primary_profile_name=None,
            session_store=SimpleNamespace(_generate_session_key=lambda source: f"agent:main:api_server:dm:{source.chat_id}"))
        self._open_checks = open_checks

    def _require_admission_open(self) -> None:
        from persistence.core.errors import AuthorityRefusal

        if self._open_checks is not None:
            if self._open_checks == 0:
                raise AuthorityRefusal("runtime_draining")
            self._open_checks -= 1


def m1_bind(session_id: str, *, declared_key: Optional[str] = None, caller_key: Optional[str] = None,
            walked_from: Optional[str] = None, open_checks: Optional[int] = None, epoch: int = 1):
    """``(legacy(db), NewApiBinding)``: main's binding write for ``session_id`` under ``epoch`` and the typed binding of
    the same values."""
    from tests.persistence_sqlite._main_callers.session_api import bind_api_steps

    authority = _Authority(open_checks=open_checks)
    authority.epoch = epoch
    step = next(bind_api_steps(authority, session_id, declared_key=declared_key, caller_key=caller_key, walked_from=walked_from))
    write = step.args[0]
    captured = dict(zip(write.__code__.co_freevars, (cell.cell_contents for cell in write.__closure__)))
    binding = NewApiBinding(
        session_id=SessionId(session_id), route=captured["route"], storage_source=captured["storage_source"],
        title=captured["title"], declared_key=declared_key, caller_key=caller_key,
        origin=captured["source"].to_dict(), entry=captured["entry"].to_dict(), receipt=dict(captured["receipt"]),
        started_at=captured["now"].timestamp())

    def legacy(db: Any) -> Any:
        authority.db = db
        return db._execute_write(write)
    return legacy, binding


def _retired(sid: str, *, expires: bool = False):
    from hermes_state_mutation_retirement import RETIRED_PREFIX

    return meta(RETIRED_PREFIX + sid, {"retired_at": 1.0, **({"expires": True} if expires else {})})


def _legacy_row(db: Any) -> None:
    """An earlier release's API row: the caller-declared key stamped directly on ``session_key``."""
    db.create_session(API_SID, source="api_server")
    with db._lock:
        db._conn.execute("UPDATE sessions SET session_key='declared-1' WHERE id=?", (API_SID,))
        db._conn.commit()


def _bind_cases():
    from hermes_state_local import API_BINDING_PREFIX, API_DECLARED_PREFIX

    return {
        "fresh": dict(steps=[], kwargs={}),
        "declared": dict(steps=[], kwargs={"declared_key": "declared-1"}),
        "declared_held": dict(steps=[meta(API_DECLARED_PREFIX + "declared-1", "api-other")], kwargs={"declared_key": "declared-1"}),
        "bound_again": dict(steps=["bind"], kwargs={}),
        "bound_other_key": dict(steps=["bind"], kwargs={"declared_key": "declared-2"}),
        "bound_other_source": dict(steps=[meta(API_BINDING_PREFIX + API_SID, {"storage_source": "bot_room"})], kwargs={}),
        "row_other_source": dict(steps=[lambda db: db.create_session(API_SID, source="cli")], kwargs={}),
        "legacy_key_adopted": dict(steps=[_legacy_row], kwargs={"caller_key": "declared-1"}),
        "legacy_key_foreign": dict(steps=[_legacy_row], kwargs={}),
        "retired": dict(steps=[_retired(API_SID)], kwargs={}),
        "route_held_unbound": dict(steps=["route_held"], kwargs={}),
        "route_held_bound": dict(steps=["route_held", meta(API_BINDING_PREFIX + "api-heal", {"storage_source": "api_server"})], kwargs={}),
    }


def _route_held(db: Any) -> None:
    """The stale-route heal's replacement holds the API route."""
    route = f"agent:main:api_server:dm:{API_SID}"
    with db._lock:
        db._conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,1.0)",
                         (route, json.dumps({"session_id": "api-heal"})))
        db._conn.commit()


@pytest.mark.parametrize("case", ["bound_again", "bound_other_key", "bound_other_source", "declared", "declared_held", "fresh",
                                  "legacy_key_adopted", "legacy_key_foreign", "retired", "route_held_bound",
                                  "route_held_unbound", "row_other_source"])
def test_bind_api_matches_m1(store, legacy_store, case):
    spec = _bind_cases()[case]
    steps = []
    for step in spec["steps"]:
        if step == "bind":
            first, _ = m1_bind(API_SID)
            steps.append(lambda db, first=first: first(db))
        elif step == "route_held":
            steps.append(_route_held)
        else:
            steps.append(step)
    seed(store, *steps)
    legacy, binding = m1_bind(API_SID, **spec["kwargs"])
    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.bind_api(epoch=1, binding=binding))
    assert_same(legacy_out, ours)


def test_bind_api_refuses_a_stale_epoch_like_m1(store, legacy_store):
    seed(store)
    legacy, binding = m1_bind(API_SID, epoch=2)
    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.bind_api(epoch=2, binding=binding))
    assert_same(legacy_out, ours)
    assert ours[1].reason == "stale_epoch"


def test_bind_api_raises_the_gate_right_after_the_epoch_check(store, legacy_store):
    # Main re-runs the admission gate inside the write, right after _epoch: a retired id
    # behind a closed gate still answers the gate's refusal, and a stale epoch still answers stale_epoch.
    seed(store, _retired(API_SID))
    legacy, binding = m1_bind(API_SID, open_checks=1)
    gate = GateRefusal("runtime_draining")
    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.bind_api(epoch=1, binding=binding, gate=gate))
    assert_same(legacy_out, ours)
    assert ours[1].reason == "runtime_draining"
    with pytest.raises(FenceStale):
        store.bindings.bind_api(epoch=2, binding=binding, gate=gate)


@pytest.mark.parametrize("passed", ["kept", "lost"])
def test_bind_api_walks_passed_fences_inside_the_write(store, legacy_store, passed):
    root = "api-root"
    successor = str(api_generation_id(root, 2))
    steps = [_retired(str(api_generation_id(root, 0)), expires=True)]
    if passed == "kept":
        steps.append(_retired(str(api_generation_id(root, 1)), expires=True))
    seed(store, *steps)
    legacy, binding = m1_bind(successor, walked_from=root)
    legacy_out, ours = run_both(store.store_path, legacy,
                                lambda: store.bindings.bind_api(epoch=1, binding=binding, walked_from=SessionId(root)))
    assert_same(legacy_out, ours)
    if passed == "lost":
        assert ours[1].reason == "chain_moved"


def test_bind_api_is_one_write(store):
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now

    seed(store, legacy=False)
    # The binding m1_bind builds for ``API_SID`` with a declared key, from the gateway's own records.
    route, now = f"agent:main:api_server:dm:{API_SID}", _now()
    source = SessionSource(platform=Platform.API_SERVER, chat_id=API_SID, user_id="api", chat_type="dm")
    entry = SessionEntry(route, API_SID, now, now, origin=source, platform=Platform.API_SERVER).to_dict()
    binding = NewApiBinding(
        session_id=SessionId(API_SID), route=route, storage_source="api_server", title=None, declared_key="declared-1",
        caller_key=None, origin=source.to_dict(), entry=entry,
        receipt={"profile_id": "default", "session_id": API_SID, "route": route, "entry": entry, "declared_key": "declared-1"},
        started_at=now.timestamp())
    assert_one_write(store, lambda: store.bindings.bind_api(epoch=1, binding=binding))


# ── api_binding / declared_session ────────────────────────────────────────────


@pytest.mark.parametrize("bound", [True, False])
def test_api_binding_matches_m1(store, legacy_store, bound):
    from tests.persistence_sqlite._main_callers.session_api import api_storage_source

    first, _ = m1_bind(API_SID, declared_key="declared-1")
    seed(store, *([first] if bound else []))
    legacy, ours = run_both(store.store_path, lambda db: api_storage_source(db, API_SID, "api_server"),
                            lambda: store.bindings.api_binding(SessionId(API_SID)))
    assert_same(legacy, ours, lambda found: "api_server" if found is None else found.storage_source)
    if bound:
        assert (ours[1].session_id, ours[1].declared_key, ours[1].room_generation) == (API_SID, "declared-1", 0)
    else:
        assert ours[1] is None
    assert_one_read(store, lambda: store.bindings.api_binding(SessionId(API_SID)))


@pytest.mark.parametrize("key", ["declared-1", "declared-none"])
def test_declared_session_matches_the_registry_read(store, legacy_store, key):
    from tests.persistence_sqlite._main_callers.session_api import declared_api_session

    first, _ = m1_bind(API_SID, declared_key="declared-1")
    seed(store, first)
    legacy, ours = run_both(store.store_path, lambda db: declared_api_session(db, key),
                            lambda: store.bindings.declared_session(key))
    assert_same(legacy, ours)
    assert_one_read(store, lambda: store.bindings.declared_session(key))


# ── authorize_history ─────────────────────────────────────────────────────────


SUBJECT = 'auth:v1:["github","","alice"]'


def _row(sid: str, *, source: str = "cli", user_id: Optional[str] = None, routed: bool = False):
    def step(db: Any) -> None:
        db.create_session(sid, source=source, user_id=user_id)
        if routed:
            with db._lock:
                db._conn.execute("UPDATE sessions SET session_key='agent:main:cli:dm:x' WHERE id=?", (sid,))
                db._conn.commit()
    return step


def _history_cases():
    from hermes_state_mutation_binding import BINDING_PREFIX

    return {
        "missing": [],
        "routed": [_row("h-1", routed=True)],
        "unbound_own": [_row("h-1")],
        "record_match": [_row("h-1"), meta(BINDING_PREFIX + "h-1", '["default",' + json.dumps(SUBJECT) + ']')],
        "record_other": [_row("h-1"), meta(BINDING_PREFIX + "h-1", '["default","someone"]')],
        "foreign_source": [_row("h-1", source="telegram")],
        "foreign_owner": [_row("h-1", user_id="bob")],
        "login_label_owner": [_row("h-1", source="desktop", user_id="github:alice")],
        "nous_label": [_row("h-1", source="desktop", user_id="nous:42")],
    }


@pytest.mark.parametrize("native", [None, True, False])
@pytest.mark.parametrize("may_create", [True, False])
@pytest.mark.parametrize("case", ["foreign_owner", "foreign_source", "login_label_owner", "missing", "nous_label",
                                  "record_match", "record_other", "routed", "unbound_own"])
def test_authorize_history_matches_m1(store, legacy_store, case, may_create, native):
    from tests.persistence_sqlite._main_callers.session_local_migration import require_history_claim
    from hermes_state_mutation_binding import authorize_history

    seed(store, *_history_cases()[case])
    actor = SimpleNamespace(profile_id="default", subject=SUBJECT, transport_id="t-1",
                            capabilities=("session:create",) if may_create else ())

    def legacy(db: Any) -> Any:
        authority = SimpleNamespace(db=db, _native_legacy_transports={"t-1": actor} if native else {})
        with db._read_ctx() as conn:
            if native is not None:
                require_history_claim(authority, conn, actor, "h-1")
            return authorize_history(conn, actor, "h-1")

    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.authorize_history(
        session_id=SessionId("h-1"), profile_id="default", subject=SUBJECT, may_create=may_create, native_transport=native, owner=None))
    assert_same(legacy_out, ours)
    assert_one_read(store, lambda: store.bindings.authorize_history(
        session_id=SessionId("h-1"), profile_id="default", subject=SUBJECT, may_create=may_create, native_transport=native, owner=None))
