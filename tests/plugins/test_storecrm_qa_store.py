from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from plugins.storecrm_qa.models import (
    CaseStatus,
    Evidence,
    JobStatus,
    PolicyMetadata,
    RunnerOutcome,
    RunnerResult,
    utc_now,
)
from plugins.storecrm_qa.redaction import redact_value
from plugins.storecrm_qa.store import StoreCRMQAStore, default_db_path


def _store(tmp_path: Path) -> StoreCRMQAStore:
    return StoreCRMQAStore(tmp_path / "qa.sqlite3")


def _metadata() -> PolicyMetadata:
    return PolicyMetadata(
        tenant_id="tenant-a",
        store_id="store-a",
        risk="medium",
        allowed_operations=("read", "assert"),
    )


def _enqueue(store: StoreCRMQAStore, *, max_attempts: int = 2):
    return store.enqueue_job(
        name="smoke",
        metadata=_metadata(),
        max_attempts=max_attempts,
        cases=[
            {"name": "case-a", "input": {"token": "sk-test-secret", "visible": "ok"}},
            {"name": "case-b", "input": {"visible": "ok"}},
        ],
    )


def test_enqueue_list_and_complete_lifecycle(tmp_path):
    store = _store(tmp_path)
    job = _enqueue(store)

    assert job.status == JobStatus.QUEUED
    cases = store.list_cases(job_id=job.id)
    assert [case.status for case in cases] == [CaseStatus.QUEUED, CaseStatus.QUEUED]
    assert cases[0].input_payload["token"] == "<redacted>"

    leased = store.claim_next_case(owner="worker-a")
    assert leased.status == CaseStatus.LEASED
    assert leased.attempts == 1

    done = store.complete_case(
        case_id=leased.id,
        owner="worker-a",
        result=RunnerResult(outcome=RunnerOutcome.PASS, summary="all good"),
    )
    assert done.status == CaseStatus.PASSED
    assert done.lease_owner is None
    assert store.get_job(job.id).status == JobStatus.QUEUED


def test_lease_exclusivity(tmp_path):
    store = _store(tmp_path)
    _enqueue(store)

    first = store.claim_next_case(owner="worker-a")
    second = store.claim_next_case(owner="worker-b")

    assert first.id != second.id
    assert first.lease_owner == "worker-a"
    assert second.lease_owner == "worker-b"
    assert store.claim_next_case(owner="worker-c") is None


def test_stale_lease_recovery_requeues_when_attempts_remain(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, max_attempts=2)
    now = utc_now()

    leased = store.claim_next_case(owner="worker-a", lease_seconds=10, now=now)
    recovered = store.recover_stale_leases(now=now + timedelta(seconds=11))

    assert recovered == 1
    case = store.get_case(leased.id)
    assert case.status == CaseStatus.QUEUED
    assert case.attempts == 1
    assert case.lease_owner is None


def test_retry_limit_exhausts_case(tmp_path):
    store = _store(tmp_path)
    _enqueue(store, max_attempts=1)

    leased = store.claim_next_case(owner="worker-a")
    case = store.fail_for_retry(
        case_id=leased.id,
        owner="worker-a",
        summary="transient StoreCRM adapter error",
    )

    assert case.status == CaseStatus.EXHAUSTED
    assert case.attempts == 1
    assert store.get_job(case.job_id).status == JobStatus.QUEUED


def test_redaction_covers_nested_keys_and_token_like_text():
    payload = {
        "ok": "visible",
        "credentials": {"password": "secret", "nested": "ok"},
        "log": "Authorization: Bearer abcdefghijklmnop",
        "key": "api_key=sk-test-abcdefghijk",
    }

    redacted = redact_value(payload)

    assert redacted["ok"] == "visible"
    assert redacted["credentials"] == "<redacted>"
    assert "abcdefghijklmnop" not in redacted["log"]
    assert "sk-test" not in redacted["key"]


def test_reports_redact_evidence_before_writing(tmp_path):
    store = _store(tmp_path)
    job = _enqueue(store)
    leased = store.claim_next_case(owner="worker-a")
    store.complete_case(
        case_id=leased.id,
        owner="worker-a",
        result=RunnerResult(
            outcome=RunnerOutcome.FAIL,
            summary="api_key=sk-test-abcdefghijk",
            evidence=(Evidence(kind="log", summary="Bearer abcdefghijklmnop"),),
        ),
    )

    report = store.write_report(job.id, tmp_path / "report.json")
    text = report.read_text()
    payload = json.loads(text)

    assert "sk-test" not in text
    assert "abcdefghijklmnop" not in text
    assert payload["failed"] == 1
    assert payload["attempts"][0]["summary"] == "api_key=<redacted>"
    assert payload["attempts"][0]["evidence"][0]["summary"] == "Bearer <redacted>"


def test_default_path_uses_active_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert default_db_path() == home / "storecrm_qa" / "qa_jobs.sqlite3"

    store = StoreCRMQAStore()
    assert store.db_path == default_db_path()
    assert store.db_path.exists()


def test_cli_dry_run_enqueue_lease_complete_report(tmp_path):
    db = tmp_path / "qa.sqlite3"
    report = tmp_path / "report.json"
    base = [sys.executable, "-m", "plugins.storecrm_qa.cli", "--db", str(db)]

    enqueue = subprocess.run(
        [
            *base,
            "enqueue",
            "--name",
            "dry-run",
            "--tenant",
            "tenant-a",
            "--store",
            "store-a",
            "--case",
            "smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = json.loads(enqueue.stdout)["job_id"]
    lease = subprocess.run(
        [*base, "lease", "--owner", "cli-worker"],
        check=True,
        capture_output=True,
        text=True,
    )
    case_id = json.loads(lease.stdout)["id"]
    subprocess.run(
        [
            *base,
            "complete",
            "--case-id",
            str(case_id),
            "--owner",
            "cli-worker",
            "--outcome",
            "pass",
            "--summary",
            "ok",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*base, "report", "--job-id", str(job_id), "--output", str(report)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(report.read_text())["passed"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
