"""Tests for the deferred restart request protocol (gateway/defer_restart.py).

Protocol invariants covered (see openspec/changes/gateway-deferred-restart):
I1  acceptance is persisted per request_id before the gateway may final-exit,
    and a final outcome is NEVER overwritten (first decision wins);
I2  the cooldown budget is keyed by stable service identity (NOT pid),
    a replay never spends a second restart, and the window is recoverable
    from persisted records;
I3  no deferred request -> legacy SIGUSR1 behaviour; an INVALID deferred
    request must never fall through to the unconditional legacy restart;
    once protocol traffic is engaged, empty-dir wakes take no action.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from gateway import defer_restart as dr


FAKE_GENERATION = {"pid": 4242, "start_time": 1_760_000_000}
FAKE_IDENTITY = {
    "kind": "launchd",
    "service_id": "hermes-gateway",
    "hermes_home": "/tmp/hermes-test",
}


@pytest.fixture(autouse=True)
def _reset_engaged():
    dr._deferred_engaged = False
    yield
    dr._deferred_engaged = False


@pytest.fixture()
def sandbox(monkeypatch, tmp_path):
    """Point the protocol at an isolated requests dir with fixed identity."""
    requests_dir = tmp_path / "restart_requests"
    requests_dir.mkdir(parents=True, exist_ok=True)  # like the real requests_dir()
    monkeypatch.setattr(dr, "requests_dir", lambda: requests_dir)
    monkeypatch.setattr(dr, "process_generation", lambda: dict(FAKE_GENERATION))
    monkeypatch.setattr(dr, "compute_service_identity", lambda: dict(FAKE_IDENTITY))
    return requests_dir


def _write_pending(requests_dir, *, request_id="req-1", **overrides) -> None:
    requests_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": dr.PROTOCOL_SCHEMA_VERSION,
        "request_id": request_id,
        "service_identity": dict(FAKE_IDENTITY),
        "target_generation": dict(FAKE_GENERATION),
        "requested_at": time.time(),
    }
    payload.update(overrides)
    path = requests_dir / f"pending-{request_id}.json"
    path.write_text(json.dumps(payload))
    path.chmod(0o600)


def _read_result(requests_dir, request_id):
    return json.loads((requests_dir / f"result-{request_id}.json").read_text())


class Sched:
    """Recording schedule_restart/cancel_restart pair for the 3-way contract."""

    def __init__(self, result="scheduled"):
        self.result = result
        self.calls = []
        self.cancels = 0

    def schedule(self):
        self.calls.append(True)
        return self.result

    def cancel(self):
        self.cancels += 1


# --- I3: legacy compatibility and fail-closed validation -----------------------


def test_no_pending_request_keeps_legacy_restart(sandbox):
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert len(s.calls) == 1


def test_unlistable_dir_fails_closed_to_no_action(sandbox, monkeypatch):
    """Listing failure is "unknown", never "confirmed empty" — no legacy restart."""
    monkeypatch.setattr(dr, "list_pending_request_files", lambda: None)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []


def test_invalid_request_never_falls_through_to_legacy_restart(sandbox):
    """A pending-but-invalid request must be rejected, not restart the gateway."""
    _write_pending(sandbox, service_identity={"kind": "launchd", "service_id": "evil", "hermes_home": "/x"})
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_REJECTED_INVALID
    # Consumed/rejected requests are removed so later legacy signals are not blocked.
    assert not (sandbox / "pending-req-1.json").exists()


def test_stale_generation_request_is_not_consumed(sandbox):
    _write_pending(sandbox, target_generation={"pid": 4242, "start_time": FAKE_GENERATION["start_time"] + 999})
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    # Left in place: it is addressed to some other real generation (expiry cleans it).
    assert (sandbox / "pending-req-1.json").exists()


def test_generation_start_time_drift_is_tolerated(sandbox):
    """Same-host fingerprint readings drift ~1 s (#117505) — same generation still matches.

    The 2 s window is a bounded engineering tolerance, NOT a uniqueness proof:
    pid equality is the primary gate (review gate 1, ruling A).
    """
    _write_pending(sandbox, target_generation={"pid": 4242, "start_time": FAKE_GENERATION["start_time"] + 100})
    payload, outcome, detail = dr.consume_pending_request()
    assert payload is not None
    assert outcome == dr.OUTCOME_ACCEPTED


def test_generation_tolerance_boundary(sandbox):
    """Exactly +200 units (the comparator tolerance) still matches; +201 does not."""
    _write_pending(sandbox, request_id="edge-in",
                   target_generation={"pid": 4242, "start_time": FAKE_GENERATION["start_time"] + 200})
    payload, outcome, _ = dr.consume_pending_request()
    assert payload is not None and outcome == dr.OUTCOME_ACCEPTED

    dr._deferred_engaged = False
    _write_pending(sandbox, request_id="edge-out",
                   target_generation={"pid": 4242, "start_time": FAKE_GENERATION["start_time"] + 201})
    payload, outcome, _ = dr.consume_pending_request()
    assert payload is None and outcome == dr.OUTCOME_REJECTED_TARGET_MISMATCH


def test_different_pid_with_close_start_time_is_rejected(sandbox):
    _write_pending(sandbox, target_generation={"pid": 4243, "start_time": FAKE_GENERATION["start_time"]})
    payload, outcome, detail = dr.consume_pending_request()
    assert payload is None
    assert outcome == dr.OUTCOME_REJECTED_TARGET_MISMATCH


@pytest.mark.parametrize("bad_start", [None, True, False, "1760000000", 0, -5])
def test_illegal_fingerprint_is_rejected_by_schema(sandbox, bad_start):
    """Unreadable/illegal fingerprints never degrade to a pid-only check (ruling A)."""
    _write_pending(sandbox, target_generation={"pid": 4242, "start_time": bad_start})
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_REJECTED_INVALID


@pytest.mark.parametrize("overrides", [
    {"schema_version": True},
    {"schema_version": "1"},
    {"request_id": "bad id with spaces"},
    {"request_id": "has.dots"},
    {"request_id": "x" * 129},
])
def test_schema_type_confusion_is_rejected(sandbox, overrides):
    """bool/int confusion and hostile ids are refused at the schema layer (ruling A)."""
    _write_pending(sandbox, **overrides)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []


def test_symlink_request_file_is_rejected(sandbox, tmp_path):
    sandbox.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({
        "schema_version": dr.PROTOCOL_SCHEMA_VERSION,
        "request_id": "req-1",
        "service_identity": dict(FAKE_IDENTITY),
        "target_generation": dict(FAKE_GENERATION),
        "requested_at": time.time(),
    }))
    (sandbox / "pending-req-1.json").symlink_to(target)
    payload, outcome, detail = dr.consume_pending_request()
    assert payload is None
    assert outcome == dr.OUTCOME_REJECTED_INVALID


def test_world_writable_request_file_is_rejected(sandbox):
    _write_pending(sandbox)
    (sandbox / "pending-req-1.json").chmod(0o666)
    payload, outcome, detail = dr.consume_pending_request()
    assert payload is None
    assert outcome == dr.OUTCOME_REJECTED_INVALID


# --- I1: acceptance persistence, first decision wins ---------------------------


def test_acceptance_persisted_after_successful_schedule(sandbox):
    _write_pending(sandbox)
    s = Sched("scheduled")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert len(s.calls) == 1
    result = _read_result(sandbox, "req-1")
    assert result["outcome"] == dr.OUTCOME_ACCEPTED
    assert result["generation"] == FAKE_GENERATION
    assert not (sandbox / "pending-req-1.json").exists()


def test_schedule_already_records_superseded(sandbox):
    _write_pending(sandbox)
    s = Sched("already")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_SUPERSEDED


def test_stopping_is_never_a_fake_success(sandbox):
    """A plain stop without a restart must not masquerade as superseded (ruling)."""
    _write_pending(sandbox)
    s = Sched("stopping")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_REJECTED_STOPPING


def test_acceptance_write_failure_cancels_restart(sandbox, monkeypatch):
    """No controlled exit without a persisted decision (I1): the restart is cancelled."""
    _write_pending(sandbox)
    monkeypatch.setattr(dr, "write_result_with_retry", lambda *a, **k: False)
    monkeypatch.setattr(dr, "record_cooldown", lambda *a, **k: True)
    s = Sched("scheduled")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.cancels == 1


def test_acceptance_write_retries_then_succeeds(sandbox, monkeypatch):
    _write_pending(sandbox)
    real_write = dr.write_result
    attempts = {"n": 0}

    def flaky(request_id, outcome, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return False
        return real_write(request_id, outcome, **kwargs)

    monkeypatch.setattr(dr, "write_result", flaky)
    s = Sched("scheduled")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.cancels == 0
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_ACCEPTED
    assert attempts["n"] >= 2


def test_existing_decision_is_never_overwritten(sandbox):
    """G1 accepted X; G2 replaying X must leave the original result byte-identical."""
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: "scheduled")
    result_path = sandbox / "result-req-1.json"
    before = result_path.read_bytes()

    _write_pending(sandbox)  # replay, e.g. consumed by another generation
    payload, outcome, detail = dr.consume_pending_request()
    assert payload is not None
    assert outcome == dr.OUTCOME_DUPLICATE
    assert result_path.read_bytes() == before


def test_rejected_decision_is_never_overwritten(sandbox):
    _write_pending(sandbox, service_identity={"kind": "launchd", "service_id": "evil", "hermes_home": "/x"})
    dr.process_sigusr1_request(lambda: "scheduled")
    result_path = sandbox / "result-req-1.json"
    before = result_path.read_bytes()
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_REJECTED_INVALID

    _write_pending(sandbox)  # a well-formed retry of the same request id
    dr.process_sigusr1_request(lambda: "scheduled")
    assert result_path.read_bytes() == before


def test_expiry_never_overwrites_existing_decision(sandbox):
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: "scheduled")
    result_path = sandbox / "result-req-1.json"
    before = result_path.read_bytes()

    _write_pending(sandbox)  # duplicate pending for the decided request
    path = sandbox / "pending-req-1.json"
    stale = time.time() - dr.PENDING_TTL_S - 60
    os.utime(path, (stale, stale))
    dr.cleanup_expired()

    assert result_path.read_bytes() == before
    assert not path.exists()


# --- I2: cooldown keyed by stable service identity -----------------------------


def test_second_distinct_request_cooldown_rejected(sandbox):
    _write_pending(sandbox, request_id="req-a")
    dr.process_sigusr1_request(lambda: "scheduled")

    _write_pending(sandbox, request_id="req-b")
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-b")["outcome"] == dr.OUTCOME_REJECTED_COOLDOWN
    entry = dr.cooldown_active_for(dr.service_identity_key(FAKE_IDENTITY))
    assert entry is not None and entry["request_id"] == "req-a"


def test_replay_repeats_existing_decision_not_new_budget(sandbox):
    """Same request_id re-submitted: existing decision stands, no second restart."""
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: "scheduled")

    _write_pending(sandbox)  # replay after result exists
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    cooldown = dr.cooldown_active_for(dr.service_identity_key(FAKE_IDENTITY))
    assert cooldown is not None and cooldown["request_id"] == "req-1"


def test_cooldown_is_not_keyed_by_pid(monkeypatch, sandbox):
    """A restarted gateway (new pid) must not bypass the cooldown (I2)."""
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: "scheduled")

    new_generation = {"pid": 9999, "start_time": FAKE_GENERATION["start_time"] + 5}
    monkeypatch.setattr(dr, "process_generation", lambda: dict(new_generation))
    _write_pending(sandbox, request_id="req-b", target_generation=new_generation)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-b")["outcome"] == dr.OUTCOME_REJECTED_COOLDOWN


def test_cooldown_recovered_from_records_when_store_lost(sandbox):
    """I2 durability: losing cooldown.json must not reopen the budget window."""
    _write_pending(sandbox, request_id="req-a")
    dr.process_sigusr1_request(lambda: "scheduled")
    (sandbox / "cooldown.json").unlink()  # simulated loss

    entry = dr.cooldown_active_for(dr.service_identity_key(FAKE_IDENTITY))
    assert entry is not None and entry["request_id"] == "req-a"

    _write_pending(sandbox, request_id="req-b")
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-b")["outcome"] == dr.OUTCOME_REJECTED_COOLDOWN


def test_unreadable_cooldown_store_fails_closed(sandbox, monkeypatch):
    """A read error is treated as ACTIVE, never as a miss (ruling, item 3)."""
    monkeypatch.setattr(dr, "read_cooldown", lambda: ({}, False))
    entry = dr.cooldown_active_for("any-key")
    assert entry is not None


# --- signal interleavings (review item 1) --------------------------------------


def test_double_wake_after_consumption_never_restarts(sandbox):
    """Two wakes, one (expired) request: the second wake must not fall into legacy."""
    _write_pending(sandbox)
    path = sandbox / "pending-req-1.json"
    stale = time.time() - dr.PENDING_TTL_S - 60
    os.utime(path, (stale, stale))

    first, second = Sched(), Sched()
    dr.process_sigusr1_request(first.schedule, cancel_restart=first.cancel)   # consumes → EXPIRED
    dr.process_sigusr1_request(second.schedule, cancel_restart=second.cancel)  # empty + engaged
    assert first.calls == []
    assert second.calls == []
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_EXPIRED


def test_late_wake_after_other_consumer_takes_no_action(sandbox):
    """CLI B's request is consumed by an earlier wake; B's own signal arrives late."""
    _write_pending(sandbox, request_id="req-b")
    first = Sched("scheduled")
    dr.process_sigusr1_request(first.schedule, cancel_restart=first.cancel)  # consumes req-b
    assert len(first.calls) == 1

    second = Sched()  # B's own SIGUSR1 lands afterwards: empty dir, traffic engaged
    dr.process_sigusr1_request(second.schedule, cancel_restart=second.cancel)
    assert second.calls == []
    # B's CLI still reads its attributable decision.
    assert _read_result(sandbox, "req-b")["outcome"] == dr.OUTCOME_ACCEPTED


def test_multiple_pending_one_wake_all_get_explicit_outcomes(sandbox):
    """One wake with several pending requests: none is left decided-less."""
    _write_pending(sandbox, request_id="req-a")
    _write_pending(sandbox, request_id="req-b")
    s = Sched("scheduled")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)

    outcomes = {rid: _read_result(sandbox, rid)["outcome"] for rid in ("req-a", "req-b")}
    assert len(s.calls) == 1
    accepted = [rid for rid, o in outcomes.items() if o == dr.OUTCOME_ACCEPTED]
    assert len(accepted) == 1
    other = next(rid for rid in outcomes if rid not in accepted)
    assert outcomes[other] in (dr.OUTCOME_REJECTED_COOLDOWN, dr.OUTCOME_SUPERSEDED)


def test_claim_race_lost_by_second_consumer(sandbox):
    """Only the consumer whose unlink() succeeds may schedule (review item 5)."""
    _write_pending(sandbox)
    payload, outcome, detail = dr.consume_pending_request()
    assert payload is not None and outcome == dr.OUTCOME_ACCEPTED

    # A second consumer (or late wake) sees nothing claimable.
    payload2, outcome2, detail2 = dr.consume_pending_request()
    assert payload2 is None and outcome2 == ""


# --- expiry cleanup preserves decisions ----------------------------------------


def test_expired_pending_is_cleaned_with_result(sandbox):
    _write_pending(sandbox)
    path = sandbox / "pending-req-1.json"
    stale = time.time() - dr.PENDING_TTL_S - 60
    os.utime(path, (stale, stale))

    dr.cleanup_expired()
    assert not path.exists()
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_EXPIRED


def test_accepted_evidence_survives_cleanup_within_retention(sandbox):
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: "scheduled")

    for f in sandbox.glob("result-*.json"):
        old = time.time() - dr.RESULT_RETENTION_S / 2
        os.utime(f, (old, old))
    dr.cleanup_expired()

    # Within the cross-generation query window the acceptance stays readable (I1).
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_ACCEPTED


def test_results_are_pruned_after_retention(sandbox):
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: "scheduled")

    for f in sandbox.glob("result-*.json"):
        stale = time.time() - dr.RESULT_RETENTION_S - 60
        os.utime(f, (stale, stale))
    dr.cleanup_expired()

    assert not (sandbox / "result-req-1.json").exists()


# --- review gate 1 round 2: hardening additions --------------------------------


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan"), 1e309])
def test_non_finite_numbers_are_rejected_by_schema(sandbox, bad_value):
    """JSON's default decoder accepts Infinity/NaN and 1e309 overflows to inf."""
    _write_pending(sandbox, target_generation={"pid": 4242, "start_time": bad_value})
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_REJECTED_INVALID


def test_stale_valid_cooldown_file_does_not_bypass_recovery(sandbox):
    """A stale-but-valid cooldown.json must not short-circuit the record scan
    (the window is determined by the NEWER of file entry and recovered record)."""
    # First generation spends the budget and records a cooldown.
    _write_pending(sandbox, request_id="req-a")
    dr.process_sigusr1_request(lambda: "scheduled")
    # Simulate record_cooldown failing after acceptance: result-req-a.json exists
    # (recovery source), while cooldown.json is rewound to an expired entry.
    old = time.time() - dr.COOLDOWN_WINDOW_S - 3600
    (sandbox / "cooldown.json").write_text(json.dumps({
        dr.service_identity_key(FAKE_IDENTITY): {
            "request_id": "ancient", "accepted_at": old,
        },
    }))

    _write_pending(sandbox, request_id="req-b")
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-b")["outcome"] == dr.OUTCOME_REJECTED_COOLDOWN


def test_orphaned_claim_gets_explicit_decision_and_no_second_schedule(sandbox):
    """A claim without a result (failed acceptance write / crash) must never let
    a re-published id schedule a second restart."""
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "claim-req-1.json").write_text(json.dumps({
        "schema_version": dr.PROTOCOL_SCHEMA_VERSION,
        "request_id": "req-1",
        "generation": {"pid": 1, "start_time": 1},
        "claimed_at": time.time() - 60,
        "service_key": dr.service_identity_key(FAKE_IDENTITY),
    }))
    _write_pending(sandbox)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_REJECTED_INVALID
    assert not (sandbox / "pending-req-1.json").exists()


def test_republished_id_after_acceptance_is_replay_only(sandbox):
    """Same id re-published after an accepted decision: no second schedule, and
    the claim/result evidence remains untouched."""
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: "scheduled")
    claim_before = (sandbox / "claim-req-1.json").read_bytes()
    result_before = (sandbox / "result-req-1.json").read_bytes()

    _write_pending(sandbox)  # CLI retries the very same request id
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert (sandbox / "claim-req-1.json").read_bytes() == claim_before
    assert (sandbox / "result-req-1.json").read_bytes() == result_before


def test_stopping_round_trip_through_real_result_files(sandbox):
    """rejected_stopping is a REAL persisted outcome the CLI can read back
    (not just a mock-level contract)."""
    _write_pending(sandbox)
    s = Sched("stopping")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)

    payload = dr.read_result("req-1")
    assert payload is not None
    assert payload["outcome"] == dr.OUTCOME_REJECTED_STOPPING
    assert payload["generation"] == FAKE_GENERATION


# --- review gate 1 round 3: result-record validity + integer overflow ----------


@pytest.mark.parametrize("bad_content", [
    "not json at all",
    '{"request_id": "someone-else", "outcome": "accepted"}',
    '{"request_id": "req-1", "outcome": "not-a-real-outcome"}',
    '[]',
])
def test_corrupt_result_file_fails_closed_without_scheduling(sandbox, bad_content):
    """A result file that EXISTS but is not a valid terminal record must never
    be treated as 'no decision' — fail-closed, zero schedules (R1)."""
    (sandbox / "result-req-1.json").write_text(bad_content)
    _write_pending(sandbox)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    # Evidence preserved: nothing valid was produced over the corrupt file.
    assert dr.read_result("req-1") is None
    assert (sandbox / "result-req-1.json").read_text() == bad_content
    assert not (sandbox / "pending-req-1.json").exists()


def test_retry_recovery_trusts_content_not_path_existence(sandbox, monkeypatch):
    """R1: write_result_with_retry must not treat 'path exists' as persisted."""
    monkeypatch.setattr(dr, "write_result", lambda *a, **k: False)  # all writes fail

    # Corrupt file at the path: NOT success — the caller must cancel (I1).
    (sandbox / "result-req-1.json").write_text("garbage")
    assert dr.write_result_with_retry("req-1", dr.OUTCOME_ACCEPTED) is False

    # A valid record with a DIFFERENT outcome: not proof of THIS acceptance.
    path = sandbox / "result-req-1.json"
    path.write_text(json.dumps({
        "schema_version": dr.PROTOCOL_SCHEMA_VERSION,
        "request_id": "req-1",
        "outcome": dr.OUTCOME_REJECTED_INVALID,
        "generation": FAKE_GENERATION,
        "decided_at": time.time(),
    }))
    path.chmod(0o600)  # valid permissions: proves the CONTENT check, not the perms
    assert dr.write_result_with_retry("req-1", dr.OUTCOME_ACCEPTED) is False

    # A valid record matching the attempted decision IS the persistence proof.
    path.write_text(json.dumps({
        "schema_version": dr.PROTOCOL_SCHEMA_VERSION,
        "request_id": "req-1",
        "outcome": dr.OUTCOME_ACCEPTED,
        "generation": FAKE_GENERATION,
        "decided_at": time.time(),
    }))
    path.chmod(0o600)
    assert dr.write_result_with_retry("req-1", dr.OUTCOME_ACCEPTED) is True


def test_huge_integer_fingerprint_is_rejected_without_crashing_the_batch(sandbox):
    """math.isfinite() overflows on huge JSON ints (R2) — refused as an illegal
    number, and the rest of the batch still gets decided."""
    _write_pending(sandbox, request_id="req-huge",
                   target_generation={"pid": 4242, "start_time": 10 ** 400})
    _write_pending(sandbox, request_id="req-good")
    s = Sched("scheduled")
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)

    assert _read_result(sandbox, "req-huge")["outcome"] == dr.OUTCOME_REJECTED_INVALID
    assert _read_result(sandbox, "req-good")["outcome"] == dr.OUTCOME_ACCEPTED
    assert len(s.calls) == 1
