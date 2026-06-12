"""JSONL reporting for completion-auditor."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .classifier import classify_response
from .config import SCHEMA, AuditorConfig
from .evidence import TurnEvidence, redact_text
from .verdict import evaluate_claim


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _profile_name() -> str:
    return os.getenv("HERMES_PROFILE") or os.getenv("HERMES_ACTIVE_PROFILE") or "default"


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except Exception:
        pass


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    _private_dir(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        path.chmod(0o600)
    except Exception:
        pass


def _daily_log_path(log_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return log_dir / f"completion-audit-{stamp}.jsonl"


def _prune_old_logs(settings: AuditorConfig) -> None:
    if settings.log_retention_days <= 0 or not settings.log_dir.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.log_retention_days)
    for path in settings.log_dir.glob("completion-audit-*.jsonl*"):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        except Exception:
            continue
        if modified < cutoff:
            try:
                path.unlink()
            except Exception:
                pass


def _rotated_log_path(settings: AuditorConfig, base_path: Path) -> Path:
    if settings.max_log_size_mb <= 0 or not base_path.exists():
        return base_path
    max_bytes = settings.max_log_size_mb * 1024 * 1024
    try:
        if base_path.stat().st_size < max_bytes:
            return base_path
    except Exception:
        return base_path
    index = 1
    while True:
        candidate = base_path.with_name(f"{base_path.stem}.{index}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        try:
            if candidate.stat().st_size < max_bytes:
                return candidate
        except Exception:
            return candidate
        index += 1


def _base_record(
    *,
    settings: AuditorConfig,
    session_id: str | None,
    turn_id: str | None,
    task_id: str | None,
    assistant_response: str | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "timestamp": _utc_now(),
        "profile": _profile_name(),
        "session_id": session_id,
        "turn_id": turn_id,
        "task_id": task_id,
        "plugin_enabled": True,
        "audit_executed": True,
        "mode": settings.mode,
        "claim_text": None,
        "claim_type": "other",
        "claim_scope": None,
        "assistant_response_chars": len(assistant_response or ""),
        "evidence_refs": [],
        "evidence_tier": "tier_4",
        "verdict": "not_applicable",
        "semantic_correctness_guaranteed": False,
        "degraded": False,
        "degrade_reason": None,
        "contradictions": [],
        "final_response_mutated": False,
        "tool_result_excerpt_included": settings.include_tool_result_excerpt,
    }


def build_record(
    *,
    settings: AuditorConfig,
    session_id: str | None,
    turn_id: str | None,
    task_id: str | None = None,
    assistant_response: str | None = None,
    evidence: TurnEvidence | None = None,
) -> dict[str, Any]:
    """Build a schema-v1 audit record for a completed turn.

    This slice classifies explicit final-response completion claims and runs a
    conservative deterministic evidence-alignment evaluator.
    """
    record = _base_record(
        settings=settings,
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        assistant_response=assistant_response,
    )
    if not session_id or not turn_id:
        record.update(
            {
                "verdict": "audit_error",
                "degraded": True,
                "degrade_reason": "missing session_id or turn_id",
            }
        )
        return record

    claim = classify_response(assistant_response)
    if claim is not None:
        record.update(
            {
                "claim_text": redact_text(claim.claim_text),
                "claim_type": claim.claim_type,
                "claim_scope": claim.claim_scope,
            }
        )

    if evidence is not None:
        record["evidence_refs"] = evidence.to_refs()
    verdict = evaluate_claim(claim, evidence)
    record.update(
        {
            "verdict": verdict.verdict,
            "evidence_tier": verdict.evidence_tier,
            "contradictions": verdict.contradictions,
            "degraded": verdict.degraded,
            "degrade_reason": verdict.degrade_reason,
        }
    )
    return record


def write_record(settings: AuditorConfig, record: dict[str, Any]) -> Path:
    _prune_old_logs(settings)
    path = _rotated_log_path(settings, _daily_log_path(settings.log_dir))
    _append_jsonl(path, record)
    return path
