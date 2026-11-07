"""``SessionBindingRepository`` on SQLite against main: the local reservation, the receipt and lineage reads, the
untracked-compression adoption, the local ``/new`` reset, the idle end and the boot hand-off of receipts a messaging
lane took over.

Each verb runs main's function and the backend verb on copies of one seed (same outcome, same rows) and opens exactly
one transaction; a write that fails right after its first mutating statement leaves every table unchanged. The
reservation's ``created`` is True only when its own write inserted the receipt. The hand-off takes the messaging lanes
as data: main tests each source with ``_messaging_platform``, the verb tests membership in ``lanes``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from persistence.authority.bindings import NewSessionBinding, local_session_id
from persistence.core.identity import SessionId
from tests.persistence_sqlite.authority._admission_support import BackendSeed, begin_epoch
from tests.persistence_sqlite.authority._support import assert_same, run_both, seed_legacy
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_read, assert_one_write, build_seed

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


def legacy_receipt(binding: NewSessionBinding) -> dict[str, Any]:
    """The receipt dict main's ``commit_local_session`` takes for ``binding``."""
    receipt = {"profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
               "session_id": binding.session_id, "route": binding.route, "entry": dict(binding.entry),
               "policy": dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage is not None:
        receipt["lineage"] = list(binding.lineage)
    return receipt


def as_record(receipt: dict[str, Any]) -> Any:
    from persistence.sqlite.authority._local import binding_record

    return binding_record(receipt)


@pytest.fixture
def frozen_clock(monkeypatch):
    """Both runs stamp the same ``time.time()``, so the rows they end compare exactly."""
    import time

    monkeypatch.setattr(time, "time", lambda: 1_790_000_000.0)


def seed(store: Any, *steps: Any, legacy: bool = True) -> None:
    """The epoch, then each ``step(db)``: on the legacy store with ``legacy``, else through the backend."""
    def run(db: Any) -> None:
        begin_epoch(db, OWNER)
        for step in steps:
            step(db)
    build_seed(store, run, legacy=legacy)


def reserved(binding: NewSessionBinding):
    """A seed step reserving ``binding``: ``commit_local_session`` on the legacy store, the verb on the backend."""
    def step(db: Any) -> None:
        if isinstance(db, BackendSeed):
            db.store.bindings.reserve(epoch=1, binding=binding, owner=None)
            return
        from hermes_state_local import commit_local_session

        commit_local_session(db, epoch=1, receipt=legacy_receipt(binding))
    return step


def meta(key: str, value: Any):
    text = value if isinstance(value, str) else json.dumps(value)
    return lambda db: (db.store.holder() if isinstance(db, BackendSeed) else db).set_meta(key, text)


# ── reserve ───────────────────────────────────────────────────────────────────


def _reserve_cases():
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_mutation_retirement import RETIRED_PREFIX

    other_route = make_binding(route="agent:main:local:dm:taken", request_id="req-2",
                               session_id=str(local_session_id(PROFILE, PRINCIPAL, "req-2")))
    return {
        "fresh": ([], make_binding(), 1),
        "exact_replay": ([reserved(make_binding())], make_binding(), 1),
        "replay_other_request": ([reserved(make_binding())], make_binding(request_json='{"kind":"launch","n":2}'), 1),
        "replay_undecodable": ([meta(POLICY_PREFIX + SID, "not json")], make_binding(), 1),
        "retired": ([meta(RETIRED_PREFIX + SID, {"retired_at": 1.0})], make_binding(), 1),
        "row_exists": ([lambda db: db.create_session(SID, source="cli")], make_binding(), 1),
        "route_held": ([reserved(other_route)], make_binding(route="agent:main:local:dm:taken"), 1),
        "stale_epoch": ([], make_binding(), 2),
        "client_env_without_request": ([], make_binding(policy={"client_env_json": "{}"}), 1),
    }


@pytest.mark.parametrize("case", ["fresh", "exact_replay", "replay_other_request", "replay_undecodable", "retired",
                                  "row_exists", "route_held", "stale_epoch", "client_env_without_request"])
def test_reserve_matches_m1(store, legacy_store, case):
    from hermes_state_local import commit_local_session

    steps, binding, epoch = _reserve_cases()[case]
    seed(store, *steps)
    legacy, ours = run_both(
        store.store_path, lambda db: as_record(commit_local_session(db, epoch=epoch, receipt=legacy_receipt(binding))),
        lambda: store.bindings.reserve(epoch=epoch, binding=binding, owner=None))
    assert_same(legacy, ours, lambda reservation: reservation.binding)
    if ours[0] == "ok":
        # CONTRACT: created only when this transaction inserted the receipt.
        assert ours[1].created is (case == "fresh")


def test_reserve_created_is_false_on_the_stored_receipt_replay(store):
    seed(store, legacy=False)
    first = store.bindings.reserve(epoch=1, binding=make_binding(), owner=None)
    again = store.bindings.reserve(epoch=1, binding=make_binding(), owner=None)
    assert (first.created, again.created) == (True, False)
    assert again.binding == first.binding


def test_reserve_is_one_write(store):
    seed(store, legacy=False)
    assert_one_write(store, lambda: store.bindings.reserve(epoch=1, binding=make_binding(), owner=None))


# ── receipt and the lineage reads ─────────────────────────────────────────────


@pytest.mark.parametrize("stored", ["reserved", "missing", "undecodable", "not_object"])
def test_receipt_matches_m1(store, legacy_store, stored):
    from hermes_state_local import POLICY_PREFIX, local_receipt

    steps = {"reserved": [reserved(make_binding())], "missing": [],
             "undecodable": [meta(POLICY_PREFIX + SID, "{")], "not_object": [meta(POLICY_PREFIX + SID, "[1]")]}[stored]
    seed(store, *steps)
    legacy, ours = run_both(store.store_path, lambda db: as_record(local_receipt(db, SID)), lambda: store.bindings.receipt(SID))
    assert_same(legacy, ours)
    assert_one_read(store, lambda: store.bindings.receipt(SID))


def _lineage_seed(store: Any, *, legacy: bool = True) -> None:
    """A reserved local session advanced once by a ``/new`` reset, plus a second session without lineage."""
    from persistence.sqlite.conversation._store.local import POLICY_PREFIX

    def reset(db: Any) -> None:
        if isinstance(db, BackendSeed):
            db.store.bindings.reset_target(epoch=1, parent_session_id=SessionId(SID), entry=_child_entry("child-1"))
            return
        from hermes_state_local_lineage import reset_local_target

        reset_local_target(db, epoch=1, parent_session_id=SID, entry=_child_entry("child-1"))

    second = make_binding(str(local_session_id(PROFILE, PRINCIPAL, "req-2")), request_id="req-2")
    seed(store, reserved(make_binding()), reserved(second), reset, meta(POLICY_PREFIX + "broken", "nope"), legacy=legacy)


def _child_entry(child: str) -> dict[str, Any]:
    route = f"agent:main:local:dm:{SID}"
    return {"session_key": route, "session_id": child, "created_at": CREATED_AT, "updated_at": CREATED_AT,
            "origin": {"chat_id": SID, "chat_type": "dm", "user_id": PRINCIPAL, "platform": "local"}}


def test_local_session_ids_matches_m1(store, legacy_store):
    from hermes_state_local import POLICY_PREFIX

    _lineage_seed(store)

    def legacy(db: Any) -> Any:
        # main's read in gateway/session_acp_lifecycle.py
        with db._read_ctx() as conn:
            return [row[0][len(POLICY_PREFIX):] for row in conn.execute(
                'SELECT key FROM state_meta WHERE key LIKE ?', (POLICY_PREFIX + '%',))]

    legacy_out, ours = run_both(store.store_path, legacy, store.bindings.local_session_ids)
    assert_same(legacy_out, ours)
    assert_one_read(store, store.bindings.local_session_ids)


def test_list_lineages_matches_the_catalog(store, legacy_store):
    from hermes_state_local import POLICY_PREFIX

    def _local_owners(db: Any) -> dict[str, Any]:
        # main's read in acp_adapter/catalog.py
        owners = {}
        with db._read_ctx() as conn:
            rows = conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)).fetchall()
        for (raw,) in rows:
            try:
                receipt = json.loads(raw)
            except (TypeError, ValueError):
                continue
            owner = receipt.get('session_id') if isinstance(receipt, dict) else None
            lineage = receipt.get('lineage') if isinstance(owner, str) else None
            for physical in lineage if isinstance(lineage, list) else ():
                if isinstance(physical, str):
                    owners[physical] = owner
        return owners

    _lineage_seed(store)
    legacy, ours = run_both(store.store_path, _local_owners, store.bindings.list_lineages)
    assert_same(legacy, ours, lambda found: {physical: item.owner for item in found for physical in item.lineage})
    assert [(item.owner, item.lineage) for item in ours[1]] == [(SID, (SID, "child-1"))]
    assert_one_read(store, store.bindings.list_lineages)


@pytest.mark.parametrize("session_id, lineage_of", [(SID, None), (SID, "child-1"), ("child-1", "child-1"), ("nobody", "child-1")])
def test_binding_marks_match_m1(store, legacy_store, session_id, lineage_of):
    from hermes_state_local import POLICY_PREFIX, local_lineage_owner
    from hermes_state_local_migration import LEGACY_PREFIX

    _lineage_seed(store)

    def legacy(db: Any) -> Any:
        # main's checks: the receipt (gateway/session_bot.py), the legacy binding (gateway/session_a2a.py) and
        # the owner of a bound lineage (gateway/session_local_migration.py).
        with db._read_ctx() as conn:
            policy = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (POLICY_PREFIX + session_id,)).fetchone()
            legacy_mark = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (LEGACY_PREFIX + session_id,)).fetchone()
            owner = local_lineage_owner(conn, lineage_of) if lineage_of and (policy or legacy_mark) else None
        return policy is not None, legacy_mark is not None, owner

    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.binding_marks(session_id, lineage_of=lineage_of))
    assert_same(legacy_out, ours, lambda marks: (marks.policy, marks.legacy, marks.lineage_owner))
    assert_one_read(store, lambda: store.bindings.binding_marks(session_id, lineage_of=lineage_of))


@pytest.mark.parametrize("session_id", [SID, "child-1", "nobody"])
def test_lineage_owner_and_conversation_ids_match_m1(store, legacy_store, session_id):
    from hermes_state_local import local_conversation_ids, local_lineage_owner

    _lineage_seed(store)

    def read(fn):
        def legacy(db: Any) -> Any:
            with db._read_ctx() as conn:
                return fn(conn, session_id)
        return legacy

    legacy, ours = run_both(store.store_path, read(local_lineage_owner), lambda: store.bindings.lineage_owner(session_id))
    assert_same(legacy, ours)
    legacy, ours = run_both(store.store_path, read(local_conversation_ids), lambda: store.bindings.conversation_ids(session_id))
    assert_same(legacy, ours)
    assert_one_read(store, lambda: store.bindings.lineage_owner(session_id))
    assert_one_read(store, lambda: store.bindings.conversation_ids(session_id))


@pytest.mark.parametrize("tamper", ["none", "wrong_identity", "lineage_root", "lineage_tip"])
def test_validate_lineage_matches_m1(store, legacy_store, tamper):
    from hermes_state_local import local_receipt
    from hermes_state_local_lineage import validate_local_lineage

    _lineage_seed(store)
    receipt: dict[str, Any] = {}
    seed_legacy(store.store_path, lambda db: receipt.update(local_receipt(db, SID)))
    if tamper == "wrong_identity":
        receipt["request_id"] = "req-x"
    elif tamper == "lineage_root":
        receipt["lineage"] = ["child-1"]
    elif tamper == "lineage_tip":
        receipt["lineage"] = [SID]

    def legacy(db: Any) -> Any:
        with db._read_ctx() as conn:
            return validate_local_lineage(conn, receipt)

    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.validate_lineage(as_record(receipt)))
    assert_same(legacy_out, ours)
    assert_one_read(store, lambda: store.bindings.validate_lineage(as_record(receipt)))


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


@pytest.mark.parametrize("case", ["adoptable", "nothing_past_tip", "foreign_step", "receipt_moved", "stale_epoch"])
def test_adopt_untracked_compression_matches_m1(store, legacy_store, case):
    from hermes_state_local import local_receipt
    from hermes_state_local_lineage import adopt_untracked_compression

    steps = [reserved(make_binding())]
    if case != "nothing_past_tip":
        steps.append(lambda db: _untracked_step(db, identity=("other-route", SID, PRINCIPAL) if case == "foreign_step" else None))
    seed(store, *steps)
    expected: dict[str, Any] = {}
    seed_legacy(store.store_path, lambda db: expected.update(local_receipt(db, SID)))
    if case == "receipt_moved":
        expected = {**expected, "request_id": "req-older"}
    epoch = 2 if case == "stale_epoch" else 1
    legacy, ours = run_both(
        store.store_path, lambda db: adopt_untracked_compression(db, epoch=epoch, session_id=SID, expected=expected),
        lambda: store.bindings.adopt_untracked_compression(epoch=epoch, session_id=SessionId(SID), expected=as_record(expected)))
    assert_same(legacy, ours)


def test_adopt_untracked_compression_is_one_write(store):
    seed(store, reserved(make_binding()), _untracked_step, legacy=False)
    expected = store.bindings.receipt(SID)
    assert_one_write(store, lambda: store.bindings.adopt_untracked_compression(
        epoch=1, session_id=SessionId(SID), expected=expected))


@pytest.mark.parametrize("case", ["adoptable", "nothing_past_tip", "foreign_step"])
def test_compression_adoptable_matches_m1(store, legacy_store, case):
    from hermes_state_local import local_receipt
    from hermes_state_local_lineage import adoptable_compression

    steps = [reserved(make_binding())]
    if case != "nothing_past_tip":
        steps.append(lambda db: _untracked_step(db, identity=("other-route", SID, PRINCIPAL) if case == "foreign_step" else None))
    seed(store, *steps)
    receipt: dict[str, Any] = {}
    seed_legacy(store.store_path, lambda db: receipt.update(local_receipt(db, SID)))

    def legacy(db: Any) -> Any:
        with db._read_ctx() as conn:
            return adoptable_compression(conn, receipt) is not None

    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.bindings.compression_adoptable(as_record(receipt)))
    assert_same(legacy_out, ours)
    assert ours[1] is (case == "adoptable")
    assert_one_read(store, lambda: store.bindings.compression_adoptable(as_record(receipt)))


# ── reset_target ──────────────────────────────────────────────────────────────


def _queued(db: Any, session_id: str = SID) -> None:
    from hermes_state_runtime import admit_session_input

    admit_session_input(db, epoch=1, principal_id=PRINCIPAL, session_id=session_id, request_id="in-1",
                        payload={"text": "hello"})


def _started(db: Any) -> None:
    from hermes_state_runtime import claim_session_input

    _queued(db)
    claim_session_input(db, epoch=1, session_id=SID)


@pytest.mark.parametrize("case", ["reset", "queued_follower", "running", "missing_parent", "no_receipt", "not_tip", "stale_epoch"])
def test_reset_target_matches_m1(store, legacy_store, frozen_clock, case):
    from hermes_state_local_lineage import reset_local_target

    steps = [] if case == "no_receipt" else [reserved(make_binding())]
    if case == "no_receipt":
        steps.append(lambda db: db.create_session(SID, source="cli", chat_id=SID))
    if case == "queued_follower":
        steps.append(_queued)
    if case == "running":
        steps.append(_started)
    if case == "not_tip":
        steps.append(lambda db: reset_local_target(db, epoch=1, parent_session_id=SID, entry=_child_entry("child-1")))
    parent = "nobody" if case == "missing_parent" else SID
    epoch = 2 if case == "stale_epoch" else 1
    seed(store, *steps)
    legacy, ours = run_both(
        store.store_path, lambda db: reset_local_target(db, epoch=epoch, parent_session_id=parent, entry=_child_entry("child-2")),
        lambda: store.bindings.reset_target(epoch=epoch, parent_session_id=SessionId(parent), entry=_child_entry("child-2")))
    assert_same(legacy, ours)


def test_reset_target_is_one_write(store):
    seed(store, reserved(make_binding()), legacy=False)
    assert_one_write(store, lambda: store.bindings.reset_target(epoch=1, parent_session_id=SessionId(SID), entry=_child_entry("child-2")))


# ── end_idle ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["idle", "queued", "already_ended", "stale_epoch"])
def test_end_idle_matches_m1(store, legacy_store, frozen_clock, case):
    from hermes_state_local import end_idle_local_session

    steps = [reserved(make_binding())]
    if case == "queued":
        steps.append(_queued)
    if case == "already_ended":
        steps.append(lambda db: db.end_session(SID, "user_exit"))
    seed(store, *steps)
    epoch = 2 if case == "stale_epoch" else 1
    legacy, ours = run_both(
        store.store_path, lambda db: end_idle_local_session(db, epoch=epoch, session_id=SID, target_id=SID, reason="idle"),
        lambda: store.bindings.end_idle(epoch=epoch, session_id=SessionId(SID), target_id=SessionId(SID), reason="idle"))
    assert_same(legacy, ours)


def test_end_idle_is_one_write(store):
    seed(store, reserved(make_binding()), legacy=False)
    assert_one_write(store, lambda: store.bindings.end_idle(epoch=1, session_id=SessionId(SID), target_id=SessionId(SID), reason="idle"))


# ── release_handed_off_receipts ───────────────────────────────────────────────


def _handed_off(db: Any) -> None:
    """A ``/handoff`` by an older build: the receipt's current target row now carries a Telegram lane's route."""
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET session_key='agent:main:telegram:dm:42', source='telegram', chat_id='42' WHERE id='child-1'"))


@pytest.mark.parametrize("lanes", [frozenset({"telegram", "discord"}), frozenset()])
def test_release_handed_off_receipts_matches_m1_with_lanes_as_data(store, legacy_store, lanes):
    from tests.persistence_sqlite._main_callers.session_local_recovery import release_handed_off_receipts

    _lineage_seed(store)
    seed_legacy(store.store_path, _handed_off)
    if lanes:
        legacy, ours = run_both(store.store_path, lambda db: release_handed_off_receipts(db, epoch=1),
                                lambda: store.bindings.release_handed_off_receipts(epoch=1, lanes=lanes))
        assert legacy[0] == "ok"
        assert ours == ("ok", {SID: ("child-1", "telegram")})
    else:
        # No source is a lane: nothing is handed off, as main leaves a receipt whose target's source names no platform.
        legacy, ours = run_both(store.store_path, lambda db: None,
                                lambda: store.bindings.release_handed_off_receipts(epoch=1, lanes=lanes))
        assert ours == ("ok", {})


def test_release_handed_off_receipts_refuses_a_stale_epoch_like_m1(store, legacy_store):
    from tests.persistence_sqlite._main_callers.session_local_recovery import release_handed_off_receipts

    _lineage_seed(store)
    seed_legacy(store.store_path, _handed_off)
    legacy, ours = run_both(store.store_path, lambda db: release_handed_off_receipts(db, epoch=2),
                            lambda: store.bindings.release_handed_off_receipts(epoch=2, lanes=frozenset({"telegram"})))
    assert_same(legacy, ours)


def test_release_handed_off_receipts_is_one_write(store):
    _lineage_seed(store, legacy=False)
    build_seed(store, _handed_off, legacy=False)
    assert_one_write(store, lambda: store.bindings.release_handed_off_receipts(epoch=1, lanes=frozenset({"telegram"})))
