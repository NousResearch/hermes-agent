"""Audit trail for the deferred restart protocol (spec: every decision that passes
resolution appends an atomic record — request id, actor reference, timestamp, target
generation, service identity key, outcome). Audit is evidence, never a gate: a failed
append must not affect the decision."""

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


def _read_audit(requests_dir):
    path = requests_dir / "audit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


class Sched:
    def __init__(self, result="scheduled"):
        self.result = result
        self.calls = []
        self.cancels = 0

    def schedule(self):
        self.calls.append(True)
        return self.result

    def cancel(self):
        self.cancels += 1


def test_accepted_request_is_audited_with_full_provenance(sandbox):
    _write_pending(sandbox, actor_ref="cli:1000:2000")
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert len(s.calls) == 1
    audit = _read_audit(sandbox)
    assert len(audit) == 1
    rec = audit[0]
    assert rec["request_id"] == "req-1"
    assert rec["outcome"] == dr.OUTCOME_ACCEPTED
    assert rec["actor_ref"] == "cli:1000:2000"
    assert rec["generation"] == FAKE_GENERATION
    assert isinstance(rec["service_key"], str) and rec["service_key"]
    assert rec["recorded_at"] > 0


def test_audit_file_is_owner_only(sandbox):
    _write_pending(sandbox)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    mode = os.stat(sandbox / "audit.jsonl").st_mode & 0o777
    assert mode == 0o600


def test_cooldown_rejection_is_audited(sandbox):
    _write_pending(sandbox, request_id="req-1", actor_ref="cli:1:2")
    dr.process_sigusr1_request(Sched().schedule, cancel_restart=lambda: None)
    _write_pending(sandbox, request_id="req-2", actor_ref="cli:3:4")
    s2 = Sched()
    dr.process_sigusr1_request(s2.schedule, cancel_restart=s2.cancel)
    assert s2.calls == []
    audit = _read_audit(sandbox)
    assert [r["outcome"] for r in audit] == [dr.OUTCOME_ACCEPTED, dr.OUTCOME_REJECTED_COOLDOWN]
    assert audit[1]["request_id"] == "req-2"
    assert audit[1]["actor_ref"] == "cli:3:4"


def test_duplicate_replay_is_audited(sandbox):
    _write_pending(sandbox, actor_ref="cli:5:6")
    dr.process_sigusr1_request(Sched().schedule, cancel_restart=lambda: None)
    _write_pending(sandbox, actor_ref="cli:5:6")  # replay of the accepted request id
    s2 = Sched()
    dr.process_sigusr1_request(s2.schedule, cancel_restart=s2.cancel)
    assert s2.calls == []
    audit = _read_audit(sandbox)
    assert audit[-1]["outcome"] == dr.OUTCOME_DUPLICATE
    assert audit[-1]["request_id"] == "req-1"


def test_schema_rejection_is_audited(sandbox):
    """A malformed actor_ref never blocks the decision — it is dropped from the audit."""
    _write_pending(sandbox, actor_ref={"pid": 1})
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    audit = _read_audit(sandbox)
    assert audit[-1]["outcome"] == dr.OUTCOME_REJECTED_INVALID
    assert audit[-1]["actor_ref"] == ""


def test_oversized_actor_ref_is_malformed_request(sandbox):
    """Schema contract: actor_ref, when present, is a bounded string — else malformed."""
    _write_pending(sandbox, actor_ref="x" * 300)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert s.calls == []
    assert _read_audit(sandbox)[-1]["outcome"] == dr.OUTCOME_REJECTED_INVALID


def test_audit_append_failure_never_affects_the_decision(sandbox, monkeypatch):
    def _explode():
        raise OSError("audit disk full")

    monkeypatch.setattr(dr, "_audit_path", _explode)
    _write_pending(sandbox)
    s = Sched()
    dr.process_sigusr1_request(s.schedule, cancel_restart=s.cancel)
    assert len(s.calls) == 1
    assert json.loads((sandbox / "result-req-1.json").read_text())["outcome"] == dr.OUTCOME_ACCEPTED
