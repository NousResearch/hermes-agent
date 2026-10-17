"""Tests for the deferred restart request protocol (gateway/defer_restart.py).

Protocol invariants covered (see openspec/changes/gateway-deferred-restart):
I1  acceptance is persisted per request_id before the gateway may final-exit;
I2  the cooldown budget is keyed by stable service identity (NOT pid),
    and a replayed request may never consume a second restart;
I3  no deferred request -> legacy SIGUSR1 behaviour; an INVALID deferred
    request must never fall through to the unconditional legacy restart.
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


@pytest.fixture()
def sandbox(monkeypatch, tmp_path):
    """Point the protocol at an isolated requests dir with fixed identity."""
    requests_dir = tmp_path / "restart_requests"
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


# --- I3: legacy compatibility and fail-closed validation -----------------------


def test_no_pending_request_keeps_legacy_restart(sandbox):
    calls = []
    dr.process_sigusr1_request(lambda: calls.append(True) or True)
    assert calls == [True]


def test_invalid_request_never_falls_through_to_legacy_restart(sandbox):
    """A pending-but-invalid request must be rejected, not restart the gateway."""
    _write_pending(sandbox, service_identity={"kind": "launchd", "service_id": "evil", "hermes_home": "/x"})
    calls = []
    dr.process_sigusr1_request(lambda: calls.append(True) or True)
    assert calls == []
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_REJECTED_INVALID
    # Consumed/rejected requests are removed so later legacy signals are not blocked.
    assert not (sandbox / "pending-req-1.json").exists()


def test_stale_generation_request_is_not_consumed(sandbox):
    _write_pending(sandbox, target_generation={"pid": 4242, "start_time": FAKE_GENERATION["start_time"] + 999})
    calls = []
    dr.process_sigusr1_request(lambda: calls.append(True) or True)
    assert calls == []
    # Left in place: it is addressed to some other real generation (expiry cleans it).
    assert (sandbox / "pending-req-1.json").exists()


def test_generation_start_time_drift_is_tolerated(sandbox):
    """Same-host fingerprint readings drift ~1 s (#117505) — same generation still matches."""
    _write_pending(sandbox, target_generation={"pid": 4242, "start_time": FAKE_GENERATION["start_time"] + 100})
    payload, outcome, detail = dr.consume_pending_request()
    assert payload is not None
    assert outcome == dr.OUTCOME_ACCEPTED


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


# --- I1: acceptance persistence ------------------------------------------------


def test_acceptance_persisted_after_successful_schedule(sandbox):
    _write_pending(sandbox)

    order = []

    def schedule():
        order.append("schedule")
        return True

    dr.process_sigusr1_request(schedule)
    assert order == ["schedule"]
    result = _read_result(sandbox, "req-1")
    assert result["outcome"] == dr.OUTCOME_ACCEPTED
    assert result["generation"] == FAKE_GENERATION
    assert not (sandbox / "pending-req-1.json").exists()


def test_schedule_rejection_records_superseded(sandbox):
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: False)
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_SUPERSEDED


# --- I2: cooldown keyed by stable service identity -----------------------------


def test_second_distinct_request_cooldown_rejected(sandbox):
    _write_pending(sandbox, request_id="req-a")

    dr.process_sigusr1_request(lambda: True)

    _write_pending(sandbox, request_id="req-b")
    calls = []
    dr.process_sigusr1_request(lambda: calls.append(True) or True)
    assert calls == []
    assert _read_result(sandbox, "req-b")["outcome"] == dr.OUTCOME_REJECTED_COOLDOWN
    entry = dr.cooldown_active_for(dr.service_identity_key(FAKE_IDENTITY))
    assert entry is not None and entry["request_id"] == "req-a"


def test_replay_repeats_existing_decision_not_new_budget(sandbox):
    """Same request_id re-submitted: existing decision is repeated, no second restart."""
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: True)

    _write_pending(sandbox)  # replay after result exists
    calls = []
    dr.process_sigusr1_request(lambda: calls.append(True) or True)
    assert calls == []
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_ACCEPTED
    # Only one budget entry — the original acceptance time.
    cooldown = dr.cooldown_active_for(dr.service_identity_key(FAKE_IDENTITY))
    assert cooldown is not None and cooldown["request_id"] == "req-1"


def test_cooldown_is_not_keyed_by_pid(monkeypatch, sandbox):
    """A restarted gateway (new pid) must not bypass the cooldown (I2)."""
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: True)

    new_generation = {"pid": 9999, "start_time": FAKE_GENERATION["start_time"] + 5}
    monkeypatch.setattr(dr, "process_generation", lambda: dict(new_generation))
    _write_pending(sandbox, request_id="req-b", target_generation=new_generation)
    calls = []
    dr.process_sigusr1_request(lambda: calls.append(True) or True)
    assert calls == []
    assert _read_result(sandbox, "req-b")["outcome"] == dr.OUTCOME_REJECTED_COOLDOWN


# --- expiry cleanup preserves acceptance evidence ------------------------------


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
    dr.process_sigusr1_request(lambda: True)

    for f in sandbox.glob("result-*.json"):
        old = time.time() - dr.RESULT_RETENTION_S / 2
        os.utime(f, (old, old))
    dr.cleanup_expired()

    # Within the cross-generation query window the acceptance stays readable (I1).
    assert _read_result(sandbox, "req-1")["outcome"] == dr.OUTCOME_ACCEPTED


def test_results_are_pruned_after_retention(sandbox):
    _write_pending(sandbox)
    dr.process_sigusr1_request(lambda: True)

    for f in sandbox.glob("result-*.json"):
        stale = time.time() - dr.RESULT_RETENTION_S - 60
        os.utime(f, (stale, stale))
    dr.cleanup_expired()

    assert not (sandbox / "result-req-1.json").exists()
