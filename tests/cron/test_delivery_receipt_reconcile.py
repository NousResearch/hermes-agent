"""`last_delivery_queued` reconciles against the receipts Bot Chat already settled (#134092)."""

import json

from cron import jobs
from cron.scheduler_delivery import reconcile_delivery_receipts


def _write_receipt(home, delivery_id: str, status: str, **extra) -> None:
    record = dict(
        delivery_id=delivery_id,
        id=delivery_id,
        owner={
            "profile_home": str(home),
            "session_id": "s1",
            "lease_id": "l1",
            "live_session_id": "v1",
        },
        message="payload",
        status=status,
        created_at=1,
        sequence=1,
        **extra,
    )
    path = home / "runtime" / "bot_live_delivery" / f"{delivery_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _queued_job(delivery_id: str) -> dict:
    job = jobs.create_job("canary", "every 1h")
    jobs.update_job(
        job["id"],
        {
            "last_status": "delivery_queued",
            "last_delivery_queued": {
                "bot-chat:(own)": {"status": "queued", "delivery_id": delivery_id}
            },
        },
    )
    return jobs.get_job(job["id"])


def test_settled_receipt_graduates_delivery_queued_to_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    key = "a" * 32
    _write_receipt(tmp_path, key, "settled")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = _queued_job(key)
        assert reconcile_delivery_receipts(job) is True
        assert job["last_status"] == "ok"
        assert job["last_delivery_queued"] is None
        assert jobs.get_job(job["id"])["last_status"] == "ok"


def test_failed_receipt_marks_delivery_failed_with_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    key = "b" * 32
    _write_receipt(tmp_path, key, "failed", error="owner crashed")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = _queued_job(key)
        assert reconcile_delivery_receipts(job) is True
        assert job["last_status"] == "delivery_failed"
        assert "bot-chat:(own) failed" in job["last_delivery_error"]
        assert "owner crashed" in job["last_delivery_error"]
        assert job["last_delivery_queued"] is None


def test_claimed_receipt_keeps_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    key = "c" * 32
    _write_receipt(tmp_path, key, "claimed")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = _queued_job(key)
        assert reconcile_delivery_receipts(job) is False
        assert job["last_status"] == "delivery_queued"
        assert job["last_delivery_queued"] is not None


def test_missing_receipt_keeps_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path / "cron"):
        # No receipt file at all: the deferred lane's records live elsewhere — never a settle.
        job = _queued_job("d" * 32)
        assert reconcile_delivery_receipts(job) is False
        assert job["last_status"] == "delivery_queued"


def test_partial_settle_keeps_only_pending_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    settled, claimed = "e" * 32, "f" * 32
    _write_receipt(tmp_path, settled, "settled")
    _write_receipt(tmp_path, claimed, "claimed")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("canary", "every 1h")
        jobs.update_job(
            job["id"],
            {
                "last_status": "delivery_queued",
                "last_delivery_queued": {
                    "bot-chat:(own)": {"status": "queued", "delivery_id": settled},
                    "bot-chat:other": {"status": "queued", "delivery_id": claimed},
                },
            },
        )
        job = jobs.get_job(job["id"])
        assert reconcile_delivery_receipts(job) is True
        # The cross-profile entry has no receipt in this home and must survive verbatim.
        assert job["last_delivery_queued"] == {
            "bot-chat:other": {"status": "queued", "delivery_id": claimed}
        }
        assert job["last_status"] == "delivery_queued"


def test_newer_run_status_survives_a_late_settle(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    key = "1" * 32
    _write_receipt(tmp_path, key, "settled")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("canary", "every 1h")
        jobs.update_job(
            job["id"],
            {
                "last_status": "error",  # a later run already spoke; the marker is stale bookkeeping
                "last_delivery_queued": {
                    "bot-chat:(own)": {"status": "queued", "delivery_id": key}
                },
            },
        )
        job = jobs.get_job(job["id"])
        assert reconcile_delivery_receipts(job) is True
        assert job["last_status"] == "error"
        assert job["last_delivery_queued"] is None
