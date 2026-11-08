"""``AdmissionRepository`` verbs that reach the conversation family's bodies (the retired-session prefix, the runtime
lineage walk, the transcript writes of the discarded-turn closer and the ``/retry`` rewrite, the media retirement
prefixes): parity with the legacy store, one transaction per verb, and the owner gate raised at the legacy guard's
point with exact replays bypassing it."""

from __future__ import annotations

import json
import time
from functools import partial

import pytest

from persistence.authority.admissions import (
    ANY_HEAD, ActiveReplace, ApiTargetClaim, CapturedResult, CarrierRewind, GateRefusal, QueuedReplacement,
    RetryRewriteRefused, TurnBoundary,
)
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite.authority._support import assert_same, counted_transactions, dump, run_both
from tests.persistence_sqlite.authority._admission_support import (
    EPOCH, admit, assert_rolls_back, begin_epoch, claim, record_dict, recover, settle,
)

DRAINING = GateRefusal("runtime_draining")


def _refuse(conn=None):
    """The legacy owner gate refusing inside the write (``_require_admission_open``)."""
    raise AuthorityRefusal("runtime_draining")


def _pair(found):
    return record_dict(found[0]), found[1]


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)


# ── seeds ──────────────────────────────────────────────────────────────────────


def _ledger(db, epoch):
    db.create_session("s1", source="cli")
    db.create_session("s2", source="cli")
    first = admit(db, "s1", "r1")
    second = admit(db, "s1", "r2")
    db.create_session("gone", source="cli")
    retired = admit(db, "gone", "rg", principal_id="api", payload={"text": "gone"})
    settle(db, claim(db, "gone"))
    db.delete_session("gone")
    return {"first": first, "second": second, "retired": retired}


def _full(db, epoch):
    values = _ledger(db, epoch)
    for i in range(32):
        admit(db, "s2", f"q{i}")
    return values


# ── admit ──────────────────────────────────────────────────────────────────────


_ADMITS = {
    "new": dict(session_id="s1", request_id="r3", payload={"text": "r3"}),
    "live_replay": dict(session_id="s1", request_id="r1", payload={"text": "r1"}),
    "conflict": dict(session_id="s1", request_id="r1", payload={"text": "changed"}),
    "missing_session": dict(session_id="absent", request_id="r9", payload={"text": "x"}),
    "tombstone_replay": dict(session_id="gone", request_id="rg", payload={"text": "gone"}, principal_id="api"),
    "tombstone_unseen": dict(session_id="gone", request_id="new", payload={"text": "gone"}, principal_id="api"),
    "tombstone_conflict": dict(session_id="gone", request_id="rg", payload={"text": "other"}, principal_id="api"),
    "steer_intent": dict(session_id="s1", request_id="r4", payload={"text": "r4"}, intent="steer"),
}


@pytest.mark.parametrize("case", sorted(_ADMITS))
@pytest.mark.parametrize("gated", [False, True])
def test_admit_matches_legacy(seeded, legacy_store, fixed_ids, case, gated):
    from hermes_state_runtime import admit_session_input
    path, store, _ = seeded(_ledger)
    kwargs = dict({"principal_id": "p", "epoch": EPOCH}, **_ADMITS[case])

    def legacy(db):
        fixed_ids()
        return admit_session_input(db, **kwargs, _authorize_write=_refuse if gated else None)

    def backend():
        fixed_ids()
        return store.admissions.admit(**kwargs, gate=DRAINING if gated else None)

    before, after = run_both(path, legacy, backend)
    assert_same(before, after, lambda found: record_dict(found[0]))
    if after[0] == "ok":
        assert after[1][1] is (case in ("new", "steer_intent"))


def test_admit_queue_cap_comes_before_the_gate(seeded, legacy_store, fixed_ids):
    from hermes_state_runtime import admit_session_input
    path, store, _ = seeded(_full)
    kwargs = dict(epoch=EPOCH, principal_id="p", session_id="s2", request_id="one-more", payload={"text": "x"})
    before, after = run_both(path, lambda db: admit_session_input(db, **kwargs, _authorize_write=_refuse),
                             lambda: store.admissions.admit(**kwargs, gate=DRAINING))
    assert_same(before, after)
    assert after[1].reason == "queue_full"


@pytest.mark.parametrize("extra", ["replaces_started", "supersedes_missing", "api_target_conflict"])
def test_admit_gate_comes_before_every_extra(seeded, extra):
    def seed(db, epoch):
        values = _ledger(db, epoch)
        values["started"] = claim(db, "s1")
        admit(db, "s2", "api-req", principal_id="api", payload={"text": "a"})
        return values

    path, store, v = seeded(seed)
    extras = {"replaces_started": {"replaces": QueuedReplacement(v["started"]["admission_id"])},
              "supersedes_missing": {"supersedes": "absent"},
              "api_target_conflict": {"api_target": ApiTargetClaim(settings={}, client_key=True)}}
    request_id = "api-req" if extra == "api_target_conflict" else "fresh"
    before = dump(path)
    for gate, reason in ((DRAINING, "runtime_draining"),
                         (None, {"replaces_started": "admission_started", "supersedes_missing": "stale_generation",
                                 "api_target_conflict": "admission_conflict"}[extra])):
        with pytest.raises(AuthorityRefusal) as refused:
            store.admissions.admit(epoch=EPOCH, principal_id="api", session_id="s1", request_id=request_id,
                                   payload={"text": "fresh"}, gate=gate, **extras[extra])
        assert refused.value.reason == reason
        assert dump(path) == before


def test_admit_reads_the_tombstones_then_writes_once(seeded):
    path, store, _ = seeded(_ledger)
    call = lambda: store.admissions.admit(epoch=EPOCH, principal_id="p", session_id="s1", request_id="r5",
                                          payload={"text": "r5"})
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        call()
    assert counts == {"write": 1, "read": 1}


# ── admit's typed extras ───────────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["queued", "started", "replay", "replay_other"])
def test_admit_replaces_matches_legacy(seeded, legacy_store, fixed_ids, case):
    from hermes_state_runtime import admit_session_input, check_replaced_input, replacement_binding, retire_replaced_input

    def seed(db, epoch):
        values = _ledger(db, epoch)
        if case == "started":
            claim(db, "s1")
        if case.startswith("replay"):
            binding = replacement_binding("p", "s1", "rep")
            admit_session_input(db, epoch=EPOCH, principal_id="p", session_id="s1", request_id="rep",
                                payload={"text": "rep"}, _authorize_write=partial(
                                    retire_replaced_input, admission_id=values["second"]["admission_id"], binding=binding))
        return values

    path, store, v = seeded(seed)
    old = v["first" if case in ("started", "replay_other") else "second"]["admission_id"]
    binding = replacement_binding("p", "s1", "rep")
    kwargs = dict(epoch=EPOCH, principal_id="p", session_id="s1", request_id="rep", payload={"text": "rep"})

    def legacy(db):
        fixed_ids()
        return admit_session_input(db, **kwargs, _authorize_write=partial(retire_replaced_input, admission_id=old, binding=binding),
                                   _replay_guard=partial(check_replaced_input, admission_id=old, binding=binding))

    def backend():
        fixed_ids()
        return store.admissions.admit(**kwargs, replaces=QueuedReplacement(old))

    before, after = run_both(path, legacy, backend)
    assert_same(before, after, lambda found: record_dict(found[0]))


@pytest.mark.parametrize("which", ["second", "absent"])
def test_admit_supersedes_matches_legacy(seeded, legacy_store, fixed_ids, which):
    from tests.persistence_sqlite._main_callers.session_automation import _replace_queued
    from hermes_state_runtime import admit_session_input
    path, store, v = seeded(_ledger)
    old = v[which]["admission_id"] if which != "absent" else "absent"
    kwargs = dict(epoch=EPOCH, principal_id="p", session_id="s1", request_id="tick", payload={"text": "tick"})

    def legacy(db):
        fixed_ids()
        return admit_session_input(db, **kwargs, _authorize_write=_replace_queued(old, "s1", []))

    def backend():
        fixed_ids()
        return store.admissions.admit(**kwargs, supersedes=old)

    before, after = run_both(path, legacy, backend)
    assert_same(before, after, lambda found: record_dict(found[0]))


@pytest.mark.parametrize("request_id,client_key", [("fresh", True), ("rg", True), ("rg", False)])
def test_admit_api_target_matches_legacy(seeded, legacy_store, fixed_ids, request_id, client_key):
    from tests.persistence_sqlite._main_callers.session_api_turn import _admit_target
    from hermes_state_runtime import admit_session_input
    path, store, _ = seeded(_ledger)
    settings = {"model": "m", "instructions": None}
    kwargs = dict(epoch=EPOCH, principal_id="api", session_id="s1", request_id=request_id, payload={"text": "x"})

    def legacy(db):
        fixed_ids()
        return admit_session_input(db, **kwargs, _authorize_write=partial(_admit_target, "s1", request_id, settings,
                                                                         client_key=client_key))

    def backend():
        fixed_ids()
        return store.admissions.admit(**kwargs, api_target=ApiTargetClaim(settings=settings, client_key=client_key))

    before, after = run_both(path, legacy, backend)
    assert_same(before, after, lambda found: record_dict(found[0]))


def _transcript(db, epoch):
    values = _ledger(db, epoch)
    for i, (role, content) in enumerate((("user", "first"), ("assistant", "reply"), ("user", "again"),
                                         ("assistant", "reply 2"))):
        db.append_message("s2", role=role, content=content, timestamp=1_700_000_000.0 + i)
    values["active"] = db.get_active_message_ids("s2")
    return values


def _retry(db, write):
    from tests.persistence_sqlite._main_callers.session_retry_admission import RetryRewrite
    return RetryRewrite(db=db, session_id="s2", source=None, write=write, on_commit=lambda: None).guard


@pytest.mark.parametrize("pinned", ["current", "stale"])
def test_admit_active_replace_matches_legacy(seeded, legacy_store, fixed_ids, frozen_clock, pinned):
    from hermes_state_runtime import admit_session_input
    path, store, v = seeded(_transcript)
    pins = list(v["active"]) if pinned == "current" else list(v["active"][:-1])
    truncated = [{"role": "user", "content": "first", "timestamp": 1_700_000_000.0},
                 {"role": "assistant", "content": "reply", "timestamp": 1_700_000_001.0}]
    kwargs = dict(epoch=EPOCH, principal_id="p", session_id="s2", request_id="retry", payload={"text": "again"})

    def legacy(db):
        fixed_ids()
        return admit_session_input(db, **kwargs, _authorize_write=_retry(db, lambda conn: db.replace_messages_in_transaction(
            conn, "s2", [dict(m) for m in truncated], active_only=True, reject_active_turn_lease=True,
            expected_active_ids=pins)))

    def backend():
        fixed_ids()
        return store.admissions.admit(**kwargs, rewrite=ActiveReplace(session_id="s2", messages=tuple(truncated),
                                                                      expected_active_ids=tuple(pins)))

    before, after = run_both(path, legacy, backend)
    if pinned == "current":
        assert_same(before, after, lambda found: record_dict(found[0]))
    else:
        assert before[0] == after[0] == "error"
        assert isinstance(after[1], RetryRewriteRefused) and str(after[1]) == str(before[1])


@pytest.mark.parametrize("pinned", ["current", "stale"])
def test_admit_carrier_rewind_matches_legacy(seeded, legacy_store, fixed_ids, frozen_clock, pinned):
    from hermes_state_mutation_transcript import rewind_in_transaction
    from hermes_state_runtime import admit_session_input
    path, store, v = seeded(_transcript)
    pins = list(v["active"]) if pinned == "current" else list(v["active"][:-1])
    target = v["active"][2]
    kwargs = dict(epoch=EPOCH, principal_id="p", session_id="s2", request_id="retry", payload={"text": "again"})

    def legacy(db):
        fixed_ids()
        return admit_session_input(db, **kwargs, _authorize_write=_retry(db, lambda conn: rewind_in_transaction(
            db, conn, "s2", target, preserve_compaction_handoff=True, expected_active_ids=pins,
            expected_target_content="again")))

    def backend():
        fixed_ids()
        return store.admissions.admit(**kwargs, rewrite=CarrierRewind(
            session_id="s2", target_row_id=target, expected_active_ids=tuple(pins), expected_target_content="again"))

    before, after = run_both(path, legacy, backend)
    if before[0] == "ok":
        assert_same(before, after, lambda found: record_dict(found[0]))
    else:
        assert after[0] == "error" and isinstance(after[1], RetryRewriteRefused) and str(after[1]) == str(before[1])


# ── retired_replay ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["live", "tombstone_replay", "tombstone_unseen", "tombstone_conflict"])
def test_retired_replay_matches_legacy(seeded, legacy_store, case):
    from hermes_state_terminal import retry_terminal_admission
    path, store, _ = seeded(_ledger)
    kwargs = dict({"principal_id": "p", "epoch": EPOCH},
                  **(_ADMITS[case] if case != "live" else _ADMITS["live_replay"]))
    before, after = run_both(path, lambda db: retry_terminal_admission(db, **kwargs),
                             lambda: store.admissions.retired_replay(**kwargs))
    assert_same(before, after, record_dict)
    with counted_transactions(store.holder()) as counts:
        try:
            store.admissions.retired_replay(**kwargs)
        except AuthorityRefusal:
            pass
    assert counts == {"write": 0, "read": 1}


# ── claim ──────────────────────────────────────────────────────────────────────


def _claim_seed(db, epoch):
    values = _ledger(db, epoch)
    db.create_session("s3", source="cli")
    admit(db, "s3", "lost")
    claim(db, "s3")
    db.create_session("empty", source="cli")
    values["epoch"] = begin_epoch(db, "owner-2")
    recover(db, values["epoch"])
    return values


@pytest.mark.parametrize("session_id,head", [("s1", ANY_HEAD), ("s1", "first"), ("s1", "second"), ("s1", None),
                                              ("empty", ANY_HEAD), ("empty", None), ("s3", ANY_HEAD),
                                              ("absent", ANY_HEAD)])
@pytest.mark.parametrize("gated", [False, True])
def test_claim_matches_legacy(seeded, legacy_store, session_id, head, gated):
    from hermes_state_runtime import _ANY_HEAD, claim_session_input
    path, store, v = seeded(_claim_seed)
    wanted = head if head is ANY_HEAD or head is None else v[head]["admission_id"]
    epoch = v["epoch"]
    before, after = run_both(
        path, lambda db: claim_session_input(db, epoch=epoch, session_id=session_id, _guard=_refuse if gated else None,
                                             head=_ANY_HEAD if wanted is ANY_HEAD else wanted),
        lambda: store.admissions.claim(epoch=epoch, session_id=session_id, gate=DRAINING if gated else None, head=wanted))
    assert_same(before, after, record_dict)
    if after[0] == "ok" and after[1]:
        assert after[1].queued_owner_epoch == EPOCH


def test_claim_gate_comes_right_after_the_epoch_fence(seeded):
    path, store, v = seeded(_claim_seed)
    before = dump(path)
    with pytest.raises(AuthorityRefusal) as refused:
        store.admissions.claim(epoch=EPOCH, session_id="s1", gate=DRAINING)
    assert refused.value.reason == "stale_epoch"
    with pytest.raises(AuthorityRefusal) as refused:  # the session is never read: not not_found
        store.admissions.claim(epoch=v["epoch"], session_id="absent", gate=DRAINING)
    assert refused.value.reason == "runtime_draining"
    with pytest.raises(AuthorityRefusal) as refused:  # nor the FIFO: not unknown_execution
        store.admissions.claim(epoch=v["epoch"], session_id="s3", gate=DRAINING)
    assert refused.value.reason == "runtime_draining"
    assert dump(path) == before


def test_claim_is_one_write_and_rolls_back(seeded):
    path, store, v = seeded(_claim_seed)
    call = lambda: store.admissions.claim(epoch=v["epoch"], session_id="s1")
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        assert call().admission_id == v["first"]["admission_id"]
    assert counts == {"write": 1, "read": 0}


# ── settle with the discarded-turn closer, resolve_unknown ─────────────────────


def _open_turn(db, epoch):
    values = _ledger(db, epoch)
    db.append_message("s1", role="user", content="r1", timestamp=1_700_000_000.0)
    values["claimed"] = claim(db, "s1")
    db.append_message("s1", role="assistant", content="Send it again", display_kind="failed_turn",
                      timestamp=1_700_000_001.0)
    return values


@pytest.mark.parametrize("replace_after", ["mark", "none", "above"])
def test_settle_with_a_boundary_matches_legacy(seeded, legacy_store, frozen_clock, fixed_ids, replace_after):
    from tests.persistence_sqlite._main_callers.session_results import close_discarded_turn
    from hermes_state_runtime import settle_session_input
    path, store, v = seeded(_open_turn)
    row = v["claimed"]
    mark = {"mark": row["transcript_mark"], "none": None, "above": 10_000}[replace_after]
    metadata = {"error": "stopped", "error_surface": "gateway"}

    def legacy(db):
        fixed_ids()
        return settle_session_input(
            db, epoch=EPOCH, admission_id=row["admission_id"], generation=row["generation"], outcome="interrupted",
            _terminal_write=lambda conn, _current: close_discarded_turn(db, conn, row, "Stopped by restart",
                                                                        replace_after=mark, display_metadata=metadata))

    def backend():
        fixed_ids()
        return store.admissions.settle(epoch=EPOCH, admission_id=row["admission_id"], generation=row["generation"],
                                       outcome="interrupted", boundary=TurnBoundary(
                                           notice="Stopped by restart", replace_after=mark, display_metadata=metadata))

    before, after = run_both(path, legacy, backend)
    assert_same(before, after, record_dict)


def _unknown(db, epoch):
    values = _open_turn(db, epoch)
    values["epoch"] = begin_epoch(db, "owner-2")
    recover(db, values["epoch"])
    return values


def _own_unknown(db, epoch):
    from tests.persistence_sqlite._main_callers.session_settlement_recovery import _mark_unknown
    from types import SimpleNamespace
    values = _open_turn(db, epoch)
    _mark_unknown(SimpleNamespace(db=db, epoch=EPOCH, sessions={}), values["claimed"])
    values["epoch"] = EPOCH
    return values


_CAPTURED = ("completed", {"result": {"final_response": "kept", "messages": []}, "usage": {}})


@pytest.mark.parametrize("seed_name,notice,captured,offset", [
    ("_unknown", None, False, 0), ("_unknown", "An automated delivery was interrupted.", False, 0),
    ("_unknown", "n", True, 0), ("_own_unknown", "n", True, 0), ("_own_unknown", None, False, 1)])
def test_resolve_unknown_matches_legacy(seeded, legacy_store, frozen_clock, fixed_ids, seed_name, notice, captured, offset):
    from tests.persistence_sqlite._main_callers.session_results import close_discarded_turn
    from hermes_state_runtime import resolve_unknown_session_input
    path, store, v = seeded(globals()[seed_name])
    row, epoch = v["claimed"], v["epoch"]
    generation = row["generation"] + offset
    prepared = _CAPTURED if captured else None

    def legacy(db):
        fixed_ids()
        return resolve_unknown_session_input(
            db, epoch=epoch, admission_id=row["admission_id"], generation=generation, _captured=prepared,
            _terminal_write=lambda conn, lost: close_discarded_turn(db, conn, lost, None if (
                prepared is not None and lost["owner_epoch"] == epoch) else notice))

    def backend():
        fixed_ids()
        return store.admissions.resolve_unknown(
            epoch=epoch, admission_id=row["admission_id"], generation=generation, notice=notice,
            captured=CapturedResult(*prepared) if prepared else None)

    before, after = run_both(path, legacy, backend)
    assert_same(before, after, record_dict)


def test_resolve_unknown_replays_a_resolved_row_and_is_one_write(seeded, frozen_clock):
    path, store, v = seeded(_unknown)
    row, epoch = v["claimed"], v["epoch"]
    call = lambda: store.admissions.resolve_unknown(epoch=epoch, admission_id=row["admission_id"],
                                                    generation=row["generation"], notice="n")
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        resolved = call()
    assert counts == {"write": 1, "read": 0}
    after = dump(path)
    assert call() == resolved
    assert dump(path) == after


def test_settle_with_a_boundary_is_one_write_and_rolls_back(seeded, frozen_clock):
    path, store, v = seeded(_open_turn)
    row = v["claimed"]
    call = lambda: store.admissions.settle(
        epoch=EPOCH, admission_id=row["admission_id"], generation=row["generation"], outcome="interrupted",
        boundary=TurnBoundary(notice="Stopped"))
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        call()
    assert counts == {"write": 1, "read": 0}


# ── media retirement candidates ────────────────────────────────────────────────


_MEDIA = [{"path": "/media/" + "a" * 64 + "/one.png", "sha256": "a" * 64},
          {"path": "/media/" + "b" * 64 + "/two.png", "sha256": "b" * 64}]


def test_record_retired_media_matches_legacy(seeded, legacy_store):
    from hermes_state_media import retire_media

    def seed(db, epoch):  # the first candidate is already recorded: first write wins
        db._execute_write(lambda conn: retire_media(conn, {"attachments_v1": {"media": _MEDIA[:1]}}))

    path, store, _ = seeded(seed)
    before, after = run_both(
        path, lambda db: db._execute_write(lambda conn: retire_media(conn, {"attachments_v1": {"media": _MEDIA}})),
        lambda: store.admissions.record_retired_media(_MEDIA))
    assert_same(before, after)
    assert_rolls_back(path, store.holder(), lambda: store.admissions.record_retired_media(_MEDIA))
    with counted_transactions(store.holder()) as counts:
        store.admissions.record_retired_media(_MEDIA)
    assert counts == {"write": 1, "read": 0}


@pytest.mark.parametrize("kind", ["media", "spillover"])
def test_retired_candidates_match_legacy(seeded, legacy_store, kind):
    from hermes_state_keys import prefix_range
    from hermes_state_media import PREFIX, SPILLOVER_PREFIX, retire_media

    def seed(db, epoch):
        db._execute_write(lambda conn: retire_media(conn, {"attachments_v1": {"media": _MEDIA}}))
        db._execute_write(lambda conn: conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)",
                                                    (SPILLOVER_PREFIX + "x.txt", json.dumps("x.txt"))))

    path, store, _ = seeded(seed)

    def legacy(db):  # main's reads in hermes_state_media.py
        with db._read_ctx() as conn:
            rows = conn.execute('SELECT key,value FROM state_meta WHERE key>=? AND key<?',
                                prefix_range(PREFIX if kind == "media" else SPILLOVER_PREFIX)).fetchall()
        return [(key, json.loads(raw)) for key, raw in rows]

    before, after = run_both(path, legacy, lambda: store.admissions.retired_candidates(kind))
    assert_same(before, after)
    assert after[1]
    with counted_transactions(store.holder()) as counts:
        store.admissions.retired_candidates(kind)
    assert counts == {"write": 0, "read": 1}


# ── input cases that reach the conversation bodies ─────────────


def test_admit_accepts_a_textual_id_and_dict_payload(seeded):
    path, store, _ = seeded(lambda db, epoch: db.create_session("42", source="test"))
    before = dump(path)
    admitted, created = store.admissions.admit(epoch=EPOCH, principal_id="owner", session_id="42",
                                               request_id="request", payload={"text": "hello"})
    assert created and admitted.status == "queued" and admitted.target_session_id == "42"
    assert admitted.payload == {"text": "hello"}
    assert store.admissions.list(session_id="42", pending_only=False) == [admitted]
    assert dump(path) != before


@pytest.mark.parametrize("intent", ["queue", "steer", "redirect"])
def test_admit_takes_the_stored_intent_strings(seeded, intent):
    # The legacy store stores the intent string.
    _path, store, _ = seeded(lambda db, epoch: db.create_session("1", source="cli"))
    admitted, _ = store.admissions.admit(epoch=EPOCH, principal_id="p", session_id="1", request_id="new", payload={},
                                         intent=intent)
    assert admitted.intent == intent


def test_numeric_session_claim_matches_legacy(seeded, legacy_store):
    # The legacy store fails building the local-policy key for a non-string id and claims nothing.
    def seed(db, epoch):
        for sid in ("1", "True"):
            db.create_session(sid, source="cli")
            admit(db, sid, sid, payload={"sid": sid})

    path, store, _ = seeded(seed)
    before = dump(path)
    legacy, backend = run_both(path, lambda db: claim(db, True),
                               lambda: store.admissions.claim(epoch=EPOCH, session_id=True))
    assert legacy[0] == "error" and isinstance(legacy[1], TypeError)
    assert_same(legacy, backend)
    assert dump(path) == before
