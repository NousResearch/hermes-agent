"""Contract cases for the authority operations ``legacy_adoption_scan``, ``pending_ledger_status``, ``mark_delegation_delivered``, ``replace_deferred_pick``, ``history_binding``, ``handoff_head`` and ``boot_session_ids``.

Each case fetches its operation first, so a backend that has not built it skips as staged. Malformed receipts are seeded through the reference meta space and gate on failpoints, as the lineage suite seeds them.
"""

from __future__ import annotations

import json

import pytest

from persistence import DelegationId, SessionId
from persistence.authority import runtime as rt
from persistence.authority.bindings import LegacyAdoptionScan, local_session_id
from persistence.authority.coordination import DeliveryState
from persistence.conversation.sessions import NewSession
from tests.contracts._reference._authority import authority_state
from tests.contracts._staged import operation
from tests.contracts.authority.test_bindings_contract import _legacy, begin, make_binding
from tests.contracts.authority.test_delegation_ledger_contract import NOW, complete, dispatch
from tests.contracts.authority.test_owner_rule_contract import RULE
from tests.contracts.conftest import require_failpoints


def _local(store, epoch: int, request_id: str) -> SessionId:
    """A reserved local session with its derived id."""
    sid = local_session_id("default", "user-1", request_id)
    store.bindings.reserve(epoch=epoch, binding=make_binding(sid, route="route-" + request_id, request_id=request_id), owner=None)
    return SessionId(sid)


def _seed_receipt_text(backend, scope, session_id: str, text: str) -> None:
    with authority_state(backend, scope).transaction() as staged:
        staged.authority["meta"]["binding:" + session_id] = text


# ── legacy_adoption_scan ──────────────────────────────────────────────────


def test_legacy_adoption_scan_orders_legacy_sources_by_start_then_id(backend, scope_a):
    store = backend.open(scope_a)
    scan = operation(store, "bindings", "legacy_adoption_scan")
    store.maintenance.import_sessions([{"id": sid, "source": source, "started_at": started, "messages": []} for sid, source, started in (
        ("z-old", "cli", 100.0), ("b-mid", "tui", 200.0), ("a-mid", "desktop", 200.0), ("c-api", "api_server", 50.0), ("d-acp", "acp", 150.0))])
    # CONTRACT: only legacy local sources, ordered by start time with the id breaking ties; nothing bound yet.
    assert scan(RULE) == LegacyAdoptionScan(session_ids=("z-old", "d-acp", "a-mid", "b-mid"), marked=frozenset())


def test_legacy_adoption_scan_marks_receipts_and_legacy_bindings(backend, scope_a):
    store = backend.open(scope_a)
    scan = operation(store, "bindings", "legacy_adoption_scan")
    epoch = begin(store)
    local = _local(store, epoch, "req-R2")
    store.sessions.create(NewSession(session_id=SessionId("old-1"), source="tui"))
    store.bindings.reserve(epoch=epoch, binding=_legacy("old-1"), owner=None)
    store.sessions.create(NewSession(session_id=SessionId("p-1"), source="cli"))
    result = scan(RULE)
    # CONTRACT: a creation receipt and a legacy binding each mark their id, prefix stripped; the rows stay candidates (the caller skips marked tips).
    assert result.marked == frozenset({local, "old-1"})
    assert set(result.session_ids) == {local, "old-1", "p-1"}


def test_legacy_adoption_scan_lists_every_member_of_a_compressed_chain(backend, scope_a):
    store = backend.open(scope_a)
    scan = operation(store, "bindings", "legacy_adoption_scan")
    store.sessions.create(NewSession(session_id=SessionId("r-root"), source="cli"))
    store.sessions.create(NewSession(session_id=SessionId("r-tip"), source="cli", parent_session_id=SessionId("r-root")))
    store.sessions.end(SessionId("r-root"), "compression")
    # CONTRACT: each member is listed by its own id, root first; the caller dedupes by tip.
    assert scan(RULE).session_ids == ("r-root", "r-tip")


# ── pending_ledger_status ─────────────────────────────────────────────────


def _admit(store, epoch, session_id: str, request_id: str):
    record, _ = store.admissions.admit(epoch=epoch, principal_id="p", session_id=session_id, request_id=request_id, payload={"text": request_id})
    return record


def _worker(store, epoch, execution_id: str, session_id: str, *, adopt: bool = False) -> None:
    store.worker_executions.register(epoch=epoch, execution_id=execution_id, session_id=SessionId(session_id), generation=0, kind="child", adoption_secret="sec")
    if adopt:
        store.worker_executions.adopt(epoch=epoch, execution_id=execution_id, session_id=SessionId(session_id), generation=0, adoption_secret="sec")


def test_pending_ledger_status_reads_admissions_then_workers(backend, scope_a):
    store = backend.open(scope_a)
    status = operation(store, "runtime", "pending_ledger_status")
    for sid in ("s-none", "s-q", "s-s", "s-u", "s-t", "s-w", "s-r", "s-both"):
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
    epoch = begin(store)
    _admit(store, epoch, "s-u", "u")
    store.admissions.claim(epoch=epoch, session_id="s-u")
    epoch = begin(store, "instance-b")
    store.admissions.recover(epoch=epoch)
    _admit(store, epoch, "s-s", "s")
    store.admissions.claim(epoch=epoch, session_id="s-s")
    _admit(store, epoch, "s-q", "q")
    store.admissions.cancel(epoch=epoch, admission_id=_admit(store, epoch, "s-t", "t").admission_id)
    _worker(store, epoch, "w-1", "s-w")
    _worker(store, epoch, "w-2", "s-r", adopt=True)
    _worker(store, epoch, "w-3", "s-both")
    _admit(store, epoch, "s-both", "b")
    # CONTRACT: a non-terminal admission's status, else a live worker's, else None; a terminal admission is ignored.
    assert {key: status(key) for key in ("s-none", "s-q", "s-s", "s-u", "s-t", "s-w", "s-r", "nobody")} == {
        "s-none": None, "s-q": "queued", "s-s": "started", "s-u": "unknown", "s-t": None, "s-w": "registered", "s-r": "running", "nobody": None}
    # CONTRACT: the admission answers before the worker (the refusal text depends on it).
    assert status("s-both") == "queued"


def test_pending_ledger_status_with_several_admissions_answers_one_of_them(backend, scope_a):
    store = backend.open(scope_a)
    status = operation(store, "runtime", "pending_ledger_status")
    store.sessions.create(NewSession(session_id=SessionId("s-two"), source="cli"))
    epoch = begin(store)
    _admit(store, epoch, "s-two", "a")
    _admit(store, epoch, "s-two", "b")
    store.admissions.claim(epoch=epoch, session_id="s-two")
    # QUIRK(no-order): the read defines no order, so which of two pending admissions answers is unspecified.
    assert status("s-two") in {"queued", "started"}


# ── mark_delegation_delivered ─────────────────────────────────────────────


@pytest.fixture
def clock(monkeypatch):
    now = [NOW]
    monkeypatch.setattr("time.time", lambda: now[0])
    return now


def test_mark_delegation_delivered_stamps_once(backend, scope_a, clock):
    store = backend.open(scope_a)
    mark = operation(store, "coordination", "mark_delegation_delivered")
    coord = store.coordination
    did = dispatch(coord, "d-run")
    clock[0] = NOW + 10
    # CONTRACT: a pending row is delivered and stamped; a second call changes nothing; a missing row is False.
    assert mark(delegation_id=did) is True
    inflight = {row.delegation_id: row for row in coord.list_inflight_delegations()}[did]
    assert inflight.updated_at == NOW + 10
    assert coord.get_delegation(delegation_id=did).delivery_state is DeliveryState.DELIVERED
    clock[0] = NOW + 20
    assert mark(delegation_id=did) is False
    assert {row.delegation_id: row for row in coord.list_inflight_delegations()}[did].updated_at == NOW + 10
    assert mark(delegation_id=DelegationId("d-ghost")) is False


def test_mark_delegation_delivered_ignores_a_held_claim(backend, scope_a, clock):
    store = backend.open(scope_a)
    mark = operation(store, "coordination", "mark_delegation_delivered")
    coord = store.coordination
    did = dispatch(coord, "d-claimed")
    complete(coord, did)
    assert coord.claim_delivery(delegation_id=did, claim_id="c1") is True
    # CONTRACT: no claim check: a claimed completion is marked, and the claim can no longer complete it.
    assert mark(delegation_id=did) is True
    assert coord.get_delegation(delegation_id=did).delivery_state is DeliveryState.DELIVERED
    assert coord.complete_delivery(delegation_id=did, claim_id="c1") is False


# ── replace_deferred_pick ─────────────────────────────────────────────────


def _defer(store, epoch, request_id: str, operation_name: str, payload: dict) -> None:
    key, digest = rt.mutation_receipt_key("p", "s-pick", request_id, operation=operation_name, payload=payload, expected_revision=0)
    store.runtime.defer_model_pick(epoch, key=key, digest=digest, session_id="s-pick", pick=rt.DeferredPick(operation_name, "p", request_id, payload))


def test_replace_deferred_pick_compares_on_the_stored_text(backend, scope_a):
    store = backend.open(scope_a)
    replace = operation(store, "runtime", "replace_deferred_pick")
    store.sessions.create(NewSession(session_id=SessionId("s-pick"), source="cli"))
    epoch = begin(store)
    _defer(store, epoch, "m1", "model", {"model": "gpt"})
    _defer(store, epoch, "r1", "reasoning", {"effort": "high"})
    slot = store.runtime.deferred_pick("s-pick")
    rest = {key: value for key, value in slot.pick.items() if key != "reasoning"}
    encoded = '{"application_id":"deferred-pick:m1","payload":{"model":"gpt"},"principal_id":"p","request_id":"m1"}'
    # CONTRACT: the canonical text is returned whether or not the slot still held ``stored``; a mismatch writes nothing.
    assert replace("s-pick", stored='{"stale":1}', pick=rest) == encoded
    assert store.runtime.deferred_pick("s-pick") == slot
    assert replace("s-pick", stored=slot.stored, pick=rest) == encoded
    assert store.runtime.deferred_pick("s-pick").stored == encoded
    assert replace("s-none", stored=slot.stored, pick=rest) == encoded
    assert store.runtime.deferred_pick("s-none") is None


def test_replace_deferred_pick_encodes_canonically(backend, scope_a):
    store = backend.open(scope_a)
    replace = operation(store, "runtime", "replace_deferred_pick")
    # CONTRACT: sorted keys at every depth, compact separators, non-ASCII kept.
    assert replace("s-none", stored="x", pick={"b": 1, "a": {"z": "é", "y": [2, 1]}}) == '{"a":{"y":[2,1],"z":"é"},"b":1}'


# ── history_binding ───────────────────────────────────────────────────────


def test_history_binding_returns_the_stored_text(backend, scope_a):
    store = backend.open(scope_a)
    read = operation(store, "bindings", "history_binding")
    for sid in ("s-cold", "s-plain"):
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
    epoch = begin(store)
    claim = rt.ClaimHistory(rt.HistoryActor("default", "alice-é", None, True))
    store.runtime.mutate(epoch, principal_id="p", session_id="s-cold", request_id="claim", expected_revision=0, command=rt.Rename("t"), history=claim, owner=None)
    # CONTRACT: the exact stored bytes (compact JSON, ASCII-escaped), which the caller compares byte for byte; None when unbound or missing.
    assert read(SessionId("s-cold")) == '["default","alice-\\u00e9"]'
    assert read(SessionId("s-cold")) == json.dumps(["default", "alice-é"], separators=(",", ":"))
    assert read(SessionId("s-plain")) is None
    assert read(SessionId("s-missing")) is None


# ── handoff_head ──────────────────────────────────────────────────────────


def test_handoff_head_of_rows_without_a_local_receipt(backend, scope_a):
    store = backend.open(scope_a)
    head = operation(store, "bindings", "handoff_head")
    store.sessions.create(NewSession(session_id=SessionId("g-plain"), source="cli"))
    store.sessions.create(NewSession(session_id=SessionId("g-root"), source="cli"))
    store.sessions.create(NewSession(session_id=SessionId("g-tip"), source="cli", parent_session_id=SessionId("g-root")))
    store.sessions.end(SessionId("g-root"), "compression")
    # CONTRACT: no owner, the row's compression tip; a missing row answers itself.
    assert head(SessionId("g-plain")) == (None, "g-plain")
    assert head(SessionId("g-root")) == (None, "g-tip")
    assert head(SessionId("g-missing")) == (None, "g-missing")


def test_handoff_head_of_a_local_conversation_follows_its_receipt(backend, scope_a):
    store = backend.open(scope_a)
    head = operation(store, "bindings", "handoff_head")
    epoch = begin(store)
    sid = _local(store, epoch, "req-G")
    route = "route-req-G"
    assert head(sid) == (sid, sid)
    store.bindings.reset_target(epoch=epoch, parent_session_id=sid, entry={**dict(make_binding(sid, route=route, request_id="req-G").entry), "session_id": "g-seg-2"})
    # CONTRACT: the owner and its receipt's current target, from the owner or any segment.
    assert head(sid) == (sid, "g-seg-2")
    assert head(SessionId("g-seg-2")) == (sid, "g-seg-2")
    store.sessions.create(NewSession(session_id=SessionId("g-seg-3"), source="cli", parent_session_id=SessionId("g-seg-2"), session_key=route, chat_id=sid, user_id="user-1"))
    store.sessions.end(SessionId("g-seg-2"), "compression")
    # CONTRACT: then that target's compression tip, though the receipt has not adopted the step.
    assert head(sid) == (sid, "g-seg-3")


def test_handoff_head_answers_a_malformed_receipt_without_an_owner(backend, scope_a):
    store = backend.open(scope_a)
    head = operation(store, "bindings", "handoff_head")
    require_failpoints(backend)
    for sid, text in (("g-text", "not json"), ("g-array", "[1, 2]"), ("g-noentry", json.dumps({"entry": "x"}))):
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
        _seed_receipt_text(backend, scope_a, sid, text)
        # CONTRACT: a receipt naming no target answers (None, the row's own tip) rather than refusing.
        assert head(SessionId(sid)) == (None, sid)


# ── boot_session_ids ──────────────────────────────────────────────────────


def test_boot_session_ids_keeps_restorable_receipts(backend, scope_a):
    store = backend.open(scope_a)
    boot = operation(store, "bindings", "boot_session_ids")
    epoch = begin(store)
    ids = {name: _local(store, epoch, "req-H-" + name) for name in ("open", "ended", "comp", "queued", "term", "close")}
    _admit(store, epoch, ids["queued"], "hq")
    store.admissions.cancel(epoch=epoch, admission_id=_admit(store, epoch, ids["term"], "ht").admission_id)
    store.runtime.store_viewer_close(epoch, session_id=ids["close"], record="{}")
    for name in ("ended", "comp", "queued", "term", "close"):
        store.sessions.end(ids[name], "compression" if name == "comp" else "user_exit")
    # CONTRACT: open, compression-ended, unsettled input on the receipt id, or a stored viewer close; an ended target with only settled input is left for first touch.
    assert set(boot()) == {ids[name] for name in ("open", "comp", "queued", "close")}


def test_boot_session_ids_keeps_a_receipt_that_names_no_row(backend, scope_a):
    store = backend.open(scope_a)
    boot = operation(store, "bindings", "boot_session_ids")
    require_failpoints(backend)
    _seed_receipt_text(backend, scope_a, "h-mal", "not json")
    # CONTRACT: an undecodable receipt joins no row, so it reads as open and boot restores it (its own reads then refuse it).
    assert boot() == ["h-mal"]
