"""Invariants of the draft work ledger (agent/work_ledger.py): admission idempotency, CAS claim,
generation fencing, settlement replay, dead-holder reclaim policy, and a complete event log."""

from __future__ import annotations

import threading

import pytest

from agent.work_ledger import (
    AdmissionConflictError,
    KindPolicy,
    SettlementConflictError,
    StaleLeaseError,
    WorkLedger,
    payload_digest_of,
)

T0 = 1_000.0
def DEAD(holder_id, fingerprint):
    return False


@pytest.fixture
def ledger(tmp_path):
    return WorkLedger(tmp_path / "ledger.db", policies={"retry": KindPolicy(retryable=True, max_attempts=2)})


def _claim(ledger, holder="h1", unit_id="u1", now=T0, ttl=30.0):
    return ledger.claim(holder, ttl, unit_id=unit_id, holder_fingerprint=f"fp-{holder}", now=now)


def test_admission_is_idempotent_and_conflicts_on_digest_drift(ledger):
    row, created = ledger.admit("u1", "plain", {"prompt": "a"}, now=T0)
    assert created and row["status"] == "queued" and row["payload_digest"] == payload_digest_of({"prompt": "a"})
    again, created = ledger.admit("u1", "plain", {"prompt": "a"}, now=T0 + 5)
    assert not created and again == row
    with pytest.raises(AdmissionConflictError):
        ledger.admit("u1", "plain", {"prompt": "b"}, now=T0 + 6)
    with pytest.raises(AdmissionConflictError):
        ledger.admit("u1", "other-kind", {"prompt": "a"}, now=T0 + 6)
    assert [e.kind for e in ledger.events("u1")] == ["admitted"]


def test_concurrent_claims_have_exactly_one_winner(ledger):
    ledger.admit("u1", "plain", {}, now=T0)
    barrier, leases = threading.Barrier(2), {}

    def worker(holder):
        barrier.wait()
        leases[holder] = _claim(ledger, holder)

    threads = [threading.Thread(target=worker, args=(h,)) for h in ("h1", "h2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    winners = [lease for lease in leases.values() if lease is not None]
    assert len(leases) == 2 and len(winners) == 1
    assert winners[0].generation == 1
    assert ledger.get("u1")["holder_id"] == winners[0].holder_id
    assert _claim(ledger, "h3") is None


def test_renew_with_stale_generation_is_rejected(ledger):
    ledger.admit("u1", "plain", {}, now=T0)
    first = _claim(ledger, "h1")
    ledger.release(first, now=T0 + 1)
    second = _claim(ledger, "h1", now=T0 + 2)
    assert second.generation == first.generation + 1
    with pytest.raises(StaleLeaseError):
        ledger.renew(first, 30.0, now=T0 + 3)
    renewed = ledger.renew(second, 60.0, now=T0 + 3)
    assert renewed.generation == second.generation and renewed.expires_at == T0 + 63
    with pytest.raises(StaleLeaseError):  # an expired lease fails closed even at the right generation
        ledger.renew(renewed, 30.0, now=T0 + 100)


def test_settle_replays_identical_and_conflicts_on_divergence(ledger):
    ledger.admit("u1", "plain", {}, now=T0)
    lease = _claim(ledger)
    first = ledger.settle(lease, "s1", "settled", {"answer": 42}, now=T0 + 1)
    assert not first.idempotent
    replay = ledger.settle(lease, "s1", "settled", {"answer": 42}, now=T0 + 2)
    assert replay.idempotent and replay.result == {"answer": 42}
    for args in (("s2", "settled", {"answer": 42}), ("s1", "failed", {"answer": 42}), ("s1", "settled", {"x": 1})):
        with pytest.raises(SettlementConflictError):
            ledger.settle(lease, *args, now=T0 + 3)
    assert [e.kind for e in ledger.events("u1")] == ["admitted", "claimed", "settled"]


def test_settle_after_generation_moved_is_stale(ledger):
    ledger.admit("u1", "plain", {}, now=T0)
    old = _claim(ledger, "h1")
    ledger.release(old, now=T0 + 1)
    _claim(ledger, "h2", now=T0 + 2)
    with pytest.raises(StaleLeaseError):
        ledger.settle(old, "s1", "settled", None, now=T0 + 3)


def test_expired_dead_holder_goes_indeterminate_and_live_holder_is_kept(ledger):
    ledger.admit("u1", "plain", {}, now=T0)
    ledger.admit("u2", "plain", {}, now=T0)
    _claim(ledger, "dead", "u1")
    _claim(ledger, "live", "u2")
    probed = []

    def liveness(holder_id, fingerprint):
        probed.append((holder_id, fingerprint))
        return holder_id == "live"

    assert ledger.reclaim_expired(liveness, now=T0 + 10) == []  # nothing expired yet
    reclaimed = ledger.reclaim_expired(liveness, now=T0 + 31)
    assert [(r.unit_id, r.status, r.attempts) for r in reclaimed] == [("u1", "indeterminate", 1)]
    assert sorted(probed) == [("dead", "fp-dead"), ("live", "fp-live")]
    assert ledger.get("u1")["status"] == "indeterminate" and ledger.get("u1")["holder_id"] is None
    assert ledger.get("u2")["status"] == "claimed"
    assert _claim(ledger, "h3", "u1", now=T0 + 32) is None  # never silently re-run


def test_retryable_kind_requeues_then_fails_at_attempt_cap(ledger):
    ledger.admit("r1", "retry", {}, now=T0)
    first = _claim(ledger, "h1", "r1")
    [r] = ledger.reclaim_expired(DEAD, now=T0 + 31)
    assert (r.status, r.attempts) == ("queued", 1)
    second = _claim(ledger, "h2", "r1", now=T0 + 32)
    assert second.generation == first.generation + 1
    [r] = ledger.reclaim_expired(DEAD, now=T0 + 70)
    assert (r.status, r.attempts) == ("failed", 2)
    row = ledger.get("r1")
    assert row["status"] == "failed" and row["attempts"] == row["max_attempts"] == 2
    assert _claim(ledger, "h3", "r1", now=T0 + 71) is None


def test_event_log_has_one_event_per_transition_with_contiguous_seq(ledger):
    ledger.admit("r1", "retry", {}, now=T0)
    lease = _claim(ledger, "h1", "r1")
    ledger.renew(lease, 30.0, now=T0 + 1)
    ledger.defer(lease, T0 + 50, now=T0 + 2)
    assert _claim(ledger, "h1", "r1", now=T0 + 10) is None  # deferred: not claimable yet
    lease = _claim(ledger, "h1", "r1", now=T0 + 50)
    ledger.reclaim_expired(DEAD, now=T0 + 90)
    lease = _claim(ledger, "h2", "r1", now=T0 + 91)
    ledger.settle(lease, "s1", "settled", {"ok": True}, now=T0 + 92)
    ledger.settle(lease, "s1", "settled", {"ok": True}, now=T0 + 93)  # replay: no new event
    events = ledger.events("r1")
    assert [e.kind for e in events] == [
        "admitted", "claimed", "renewed", "deferred", "claimed", "reclaimed", "claimed", "settled"]
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert [e.data.get("generation") for e in events if e.kind == "claimed"] == [1, 2, 3]
    assert events[-1].data == {"from": "claimed", "to": "settled", "settlement_id": "s1", "generation": 3}
