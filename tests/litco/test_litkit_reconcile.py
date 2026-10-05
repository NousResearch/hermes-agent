"""Production-row reconciliation before Ana reads it.

Regression for LKP-1007/LKP-1008: a production whose ingest finished on 2026-08-05 still read
``status: ingesting`` with a stale archive-password failure and a blocking QC verdict, and Ana
told a reviewer the ingest was still running.
"""

from __future__ import annotations

import copy

import pytest

from litco.litkit.reconcile import reconcile_production_status, reconcile_productions


def _lkp1007() -> dict:
    """The production row Ana received on 2026-10-05."""
    return {
        "id": "30c84324-0000-4000-8000-000000000001",
        "name": "Adobe Foret Vol. 1",
        "status": "ingesting",
        "fileCount": 52483,
        "documentCount": 52483,
        "latestIngestJob": {"status": "done", "totalDocs": 52483, "doneDocs": 52483,
                            "finishedAt": "2026-08-05T19:42:01Z"},
        "orchestration": {"status": "completed", "phase": "completed", "route": "concordance",
                          "qcDecision": {"action": "proceed"}},
        "failureSummary": {"failureCode": "archive_password_required", "source": "import_event",
                           "traceId": "event:abc", "headline": "The archive needs a password"},
        "qcGate": {"verdict": "blocking", "headline": "3 blocking findings", "blockingCount": 3},
    }


def _finished_with_omissions() -> dict:
    row = _lkp1007()
    row["failureSummary"] = {"failureCode": "ocr_or_render_failed", "source": "process_job", "traceId": "process:1"}
    return row


def test_a_finished_ingest_that_still_reads_ingesting_is_reported_finished():
    raw = _lkp1007()
    out = reconcile_production_status(raw)
    assert out["rawStatus"] == "ingesting"
    assert out["status"] in ("ingested", "ingested_partial")
    assert "2026-08-05" in out["statusNote"] and "not running" in out["statusNote"]
    # the archive was evidently opened: its pre-run password failure is not a current failure
    assert "failureSummary" not in out
    assert out["qcGate"]["decision"] == "proceed"
    assert out["qcGate"]["note"] == "QC findings were overridden by the reviewer; ingest proceeded"
    assert raw == _lkp1007(), "the caller's dict is not mutated"


def test_per_document_omissions_make_it_partial_and_are_kept():
    out = reconcile_production_status(_finished_with_omissions())
    assert (out["status"], out["rawStatus"]) == ("ingested_partial", "ingesting")
    assert out["failureSummary"]["failureCode"] == "ocr_or_render_failed"

    short = _lkp1007()
    short["latestIngestJob"]["doneDocs"] = 52000
    assert reconcile_production_status(short)["status"] == "ingested_partial"


def test_the_servers_own_status_note_is_preferred():
    row = _finished_with_omissions()
    row["statusNote"] = "finished on Aug 5, 2026; the ingest is not running. 112 documents are missing viewer PDFs"
    assert reconcile_production_status(row)["statusNote"] == row["statusNote"]


def test_a_dated_failure_survives_only_when_it_postdates_the_run():
    row = _lkp1007()
    row["orchestration"]["completedAt"] = "2026-08-05T19:42:01Z"
    row["failureSummary"]["at"] = "2026-08-04T10:00:00Z"
    assert "failureSummary" not in reconcile_production_status(row)
    row["failureSummary"]["at"] = "2026-09-01T10:00:00Z"
    assert reconcile_production_status(row)["failureSummary"]["failureCode"] == "archive_password_required"


def test_a_running_production_is_untouched():
    row = _lkp1007()
    row["latestIngestJob"].update(status="running", doneDocs=1000, finishedAt=None)
    row["orchestration"].update(status="running", phase="ingesting")
    assert reconcile_production_status(copy.deepcopy(row)) == row


@pytest.mark.parametrize("bad", [
    None, "ingesting", 7, [],
    {"status": "ingesting", "latestIngestJob": "done", "orchestration": {"status": "completed"}},
    {"status": "ingesting", "latestIngestJob": {"status": "done", "finishedAt": object()},
     "orchestration": ["completed"]},
])
def test_a_malformed_payload_is_returned_unchanged(bad):
    assert reconcile_production_status(bad) is bad


def test_every_payload_shape_is_reconciled():
    assert reconcile_productions([_lkp1007()])[0]["rawStatus"] == "ingesting"
    assert reconcile_productions({"production": _lkp1007()})["production"]["rawStatus"] == "ingesting"
    assert reconcile_productions({"productions": [_lkp1007()]})["productions"][0]["rawStatus"] == "ingesting"
    assert reconcile_productions({"error": "not_found"}) == {"error": "not_found"}
