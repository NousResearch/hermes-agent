from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .admission import evaluate_candidate
from .extractor import extract_candidates
from .ledger import LedgerStore
from .projection import rebuild_current_view
from .reconciliation import reconcile_candidate
from .spool import TruthSpool

_MAX_PROCESS_LIMIT = 3
_DEFAULT_PROCESS_LIMIT = 1


def _ledger_history(root: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    ledger_dir = root / "ledger"
    for ledger_file in sorted(ledger_dir.glob("*.jsonl")):
        for raw_line in ledger_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(event, dict):
                history.append(event)
    return history


def _observation(envelope: Mapping[str, Any]) -> dict[str, Any]:
    raw_origin = envelope.get("origin")
    origin: Mapping[str, Any] = raw_origin if isinstance(raw_origin, Mapping) else {}
    return {
        "profile": envelope.get("profile"),
        "platform": origin.get("platform"),
        "session_id": envelope.get("session_id"),
        "turn_id": envelope.get("turn_id"),
        "task_id": None,
        "speaker_id": origin.get("speaker_id"),
        "conversation_id": origin.get("conversation_id"),
        "thread_id": origin.get("thread_id"),
    }


def _admission_metadata(envelope: Mapping[str, Any]) -> dict[str, Any]:
    raw_origin = envelope.get("origin")
    origin: Mapping[str, Any] = raw_origin if isinstance(raw_origin, Mapping) else {}
    raw_input = envelope.get("input")
    input_payload: Mapping[str, Any] = raw_input if isinstance(raw_input, Mapping) else {}
    return {
        "source_channel": origin.get("source_channel") or "",
        "speaker_id": origin.get("speaker_id"),
        "source_text": input_payload.get("user_message") or "",
    }


def _candidate_for_reconciliation(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scope": candidate.get("scope"),
        "kind": candidate.get("kind"),
        "subject": candidate.get("subject"),
        "key": candidate.get("key"),
        "value": candidate.get("value"),
        "proposed_operation": candidate.get("operation") or "assert",
        "evidence_type": candidate.get("evidence_type") or "user_stated",
    }


async def process_pending(
    *,
    root: Path | str,
    ctx: Any,
    limit: int = _DEFAULT_PROCESS_LIMIT,
    apply: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    try:
        bounded_limit = int(limit)
    except (TypeError, ValueError):
        bounded_limit = 0
    if bounded_limit < 1 or bounded_limit > _MAX_PROCESS_LIMIT:
        return {
            "ok": False,
            "action": "process",
            "reason": "invalid_limit",
            "minimum": 1,
            "maximum": _MAX_PROCESS_LIMIT,
            "dry_run": not apply,
        }

    pending_dir = root / "spool" / "pending"
    pending_before = len(list(pending_dir.glob("*.json"))) if pending_dir.exists() else 0
    base: dict[str, Any] = {
        "ok": True,
        "action": "process",
        "dry_run": not apply,
        "limit": bounded_limit,
        "pending_before": pending_before,
    }
    if not apply:
        base.update({"would_process": min(bounded_limit, pending_before)})
        return base
    if ctx is None or not hasattr(ctx, "llm"):
        base.update({"ok": False, "reason": "runtime_context_required"})
        return base

    spool = TruthSpool(root)
    pending_snapshot = spool.snapshot_pending(bounded_limit)

    counters = {
        "claimed": 0,
        "acked": 0,
        "appended": 0,
        "duplicates": 0,
        "rejected": 0,
        "none": 0,
        "retried": 0,
        "dead_lettered": 0,
    }
    ledger = LedgerStore(root)
    history = _ledger_history(root)
    projection_dirty = False

    for pending_path in pending_snapshot:
        claim = spool.claim_path(pending_path, owner=f"operator:{os.getpid()}")
        if claim is None:
            continue
        processing_path = Path(claim["path"])
        raw_record = claim.get("record")
        record: dict[str, Any] = raw_record if isinstance(raw_record, dict) else {}
        counters["claimed"] += 1

        envelope = claim["envelope"]
        extracted = await extract_candidates(ctx=ctx, envelope=envelope)
        status = str(extracted.get("status") or "")
        if status == "retry":
            spool.retry_processing(
                processing_path,
                error_code=str(extracted.get("reason") or "retry"),
                delay_ms=int(extracted.get("retry_delay_ms") or 0),
            )
            counters["retried"] += 1
            continue
        if status == "dead_letter":
            spool.dead_letter(processing_path, reason=str(extracted.get("reason") or "extraction_failed"))
            counters["dead_lettered"] += 1
            continue
        if status == "none":
            spool.ack_processing(processing_path)
            counters["none"] += 1
            counters["acked"] += 1
            continue
        if status != "ok":
            spool.dead_letter(processing_path, reason="invalid_extractor_status")
            counters["dead_lettered"] += 1
            continue

        observation = _observation(envelope)
        metadata = _admission_metadata(envelope)
        record_failed = False
        for raw_candidate in extracted.get("facts") or []:
            if not isinstance(raw_candidate, Mapping):
                counters["rejected"] += 1
                continue
            admission = evaluate_candidate(raw_candidate, metadata)
            if not bool(admission.get("admit", False)):
                counters["rejected"] += 1
                continue

            decision = reconcile_candidate(
                history=history,
                observation=observation,
                candidate=_candidate_for_reconciliation(raw_candidate),
                occurred_at=str(envelope.get("captured_at") or ""),
                extraction=extracted.get("extraction") if isinstance(extracted.get("extraction"), Mapping) else None,
            )
            decision_name = str(decision.get("decision") or "")
            if decision_name == "duplicate":
                counters["duplicates"] += 1
                continue
            if decision_name == "none":
                counters["rejected"] += 1
                continue
            event = decision.get("event")
            if decision_name != "append" or not isinstance(event, dict):
                spool.dead_letter(processing_path, reason="invalid_reconciliation_decision")
                counters["dead_lettered"] += 1
                record_failed = True
                break

            append_out = ledger.append_event(event=event, event_key=str(event.get("event_id") or ""))
            append_status = str(append_out.get("status") or "")
            if append_status == "indexed":
                counters["appended"] += 1
                history.append(event)
                projection_dirty = True
                continue
            if append_status == "duplicate":
                counters["duplicates"] += 1
                continue
            if append_status == "retry":
                spool.retry_processing(processing_path, error_code=str(append_out.get("reason") or "ledger_retry"))
                counters["retried"] += 1
                record_failed = True
                break
            spool.dead_letter(processing_path, reason=str(append_out.get("reason") or "ledger_append_failed"))
            counters["dead_lettered"] += 1
            record_failed = True
            break

        if record_failed:
            continue
        spool.ack_processing(processing_path)
        counters["acked"] += 1

    active_facts = 0
    if projection_dirty:
        projection = rebuild_current_view(root)
        active_facts = int(projection.get("active", 0))
    else:
        current = root / "views" / "current.jsonl"
        if current.exists():
            active_facts = sum(1 for line in current.read_text(encoding="utf-8").splitlines() if line.strip())

    base.update(counters)
    base.update(
        {
            "active_facts": active_facts,
            "pending_after": len(list(spool.pending_dir.glob("*.json"))),
            "processing_after": len(list(spool.processing_dir.glob("*.json"))),
        }
    )
    return base
