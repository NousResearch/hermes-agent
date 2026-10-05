"""Reconcile a LitKit production row before Ana reads it.

LitKit's ``status`` column can lag the work it describes: a production whose ingest job is
``done`` and whose orchestration is ``completed`` may still read ``ingesting``, and carry a
failure summary and QC verdict from before the run that finished it (LKP-1007). Read as-is,
that row tells Ana an ingest is still running. :func:`reconcile_production_status` rewrites
only what the row's own evidence contradicts and keeps the original status in ``rawStatus``.
Pure and total: a row it cannot read comes back unchanged.
"""

from __future__ import annotations

import copy
import datetime as _dt
from typing import Any, Dict, Optional

# Failure codes that describe a finished ingest with per-document omissions, not a stopped one.
INGEST_PARTIAL_CODES = frozenset({"partial_ingest", "count_mismatch", "ocr_or_render_failed"})
# Sources whose failure predates the ingest run: the upload and the archive import.
PRE_RUN_SOURCES = frozenset({"import_event", "upload_session", "production_import"})
QC_OVERRIDE_NOTE = "QC findings were overridden by the reviewer; ingest proceeded"


def _when(value: Any) -> Optional[_dt.datetime]:
    """An ISO string or epoch-ms number as an aware datetime; None when unreadable."""
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return _dt.datetime.fromtimestamp(value / 1000, tz=_dt.timezone.utc)
        if isinstance(value, str) and value:
            parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return None


def _finished(job: Dict[str, Any], orch: Dict[str, Any]) -> bool:
    return job.get("status") == "done" and orch.get("status") == "completed"


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _stale_failure(p: Dict[str, Any], job: Dict[str, Any], orch: Dict[str, Any]) -> bool:
    summary = p.get("failureSummary")
    if not isinstance(summary, dict) or summary.get("failureCode") in INGEST_PARTIAL_CODES:
        return False
    run_at = _when(orch.get("completedAt")) or _when(orch.get("claimedAt"))
    failed_at = _when(summary.get("at")) or _when(summary.get("createdAt"))
    if run_at and failed_at:
        return summary.get("source") in PRE_RUN_SOURCES and failed_at < run_at
    # No dates to compare: a password-protected archive that yielded documents was opened.
    opened = (_int(p.get("documentCount")) or 0) > 0 or (_int(p.get("fileCount")) or 0) > 0
    return summary.get("failureCode") == "archive_password_required" and opened


def _status_note(job: Dict[str, Any]) -> str:
    finished = _when(job.get("finishedAt"))
    on = f"finished on {finished.date().isoformat()}" if finished else "finished"
    return f"{on}; the ingest is not running. Some documents may be missing viewer PDFs — check exceptions"


def reconcile_production_status(p: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of production row ``p`` whose status, failure summary and QC gate agree with its
    finished ingest job and completed orchestration. Any other row is returned unchanged."""
    try:
        if not isinstance(p, dict):
            return p
        job = p.get("latestIngestJob") or {}
        orch = p.get("orchestration") or {}
        if not isinstance(job, dict) or not isinstance(orch, dict) or not _finished(job, orch):
            return p
        out = copy.deepcopy(p)
        summary = out.get("failureSummary")
        if _stale_failure(out, job, orch):
            out.pop("failureSummary", None)
            summary = None
        if out.get("status") == "ingesting":
            total, done = _int(job.get("totalDocs")), _int(job.get("doneDocs"))
            partial = (isinstance(summary, dict) and summary.get("failureCode") in INGEST_PARTIAL_CODES) \
                or (total is not None and done is not None and done < total)
            out["rawStatus"] = out["status"]
            out["status"] = "ingested_partial" if partial else "ingested"
            if not out.get("statusNote"):
                out["statusNote"] = _status_note(job)
        gate = out.get("qcGate")
        if isinstance(gate, dict):
            decision = orch.get("qcDecision") if isinstance(orch.get("qcDecision"), dict) else {}
            if gate.get("decision") or decision.get("action") == "proceed":
                gate["decision"] = gate.get("decision") or "proceed"
                gate["note"] = QC_OVERRIDE_NOTE
        return out
    except Exception:
        return p


def reconcile_productions(payload: Any) -> Any:
    """Apply :func:`reconcile_production_status` to every row in a productions payload: a list
    of rows, ``{"production": row}``, ``{"productions": [rows]}``, or a bare row."""
    try:
        if isinstance(payload, list):
            return [reconcile_production_status(row) for row in payload]
        if isinstance(payload, dict):
            if isinstance(payload.get("production"), dict):
                return {**payload, "production": reconcile_production_status(payload["production"])}
            if isinstance(payload.get("productions"), list):
                return {**payload, "productions": [reconcile_production_status(r) for r in payload["productions"]]}
            return reconcile_production_status(payload)
    except Exception:
        return payload
    return payload
