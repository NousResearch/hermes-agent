"""Result Consolidator for Task Runtime.

Builds the final TaskResult and writes an append-only trace.jsonl.

NO mutations outside the configured output dir. NO secrets in the trace.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SECRET_PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
)

def _redact_secrets(text: str) -> str:
    out = text
    for pat in SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


@dataclass
class TaskResult:
    intent_id: str
    contract_id: str
    contract_fingerprint: str
    execution_mode: str
    pipeline_result: dict
    final_answer: str
    trace_path: str | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp_utc: str = ""  # set by build()


def build(
    resolved_intent,
    contract: dict[str, Any],
    pipeline_result,
    trace_dir: Path | None = None,
) -> TaskResult:
    """Build TaskResult and write append-only trace.jsonl.

    Args:
        resolved_intent: from IntentResolver.
        contract: from TaskContractBuilder.
        pipeline_result: a PipelineResult (or dict-like) from ExecutionPipeline.
        trace_dir: directory to write trace.jsonl. If None, defaults to
                   ~/.hermes/traces/. Created if missing.

    Returns:
        TaskResult.
    """
    import datetime as _dt

    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    final_answer = _build_final_answer(resolved_intent, contract, pipeline_result)

    if trace_dir is None:
        trace_dir = Path.home() / ".hermes" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace.jsonl"

    pr_dict = (
        asdict(pipeline_result)
        if hasattr(pipeline_result, "__dataclass_fields__")
        else dict(pipeline_result)
    )

    trace_record = {
        "timestamp_utc": ts,
        "intent_id": resolved_intent.intent_id,
        "contract_id": contract.get("contract_id"),
        "contract_fingerprint": contract.get("contract_fingerprint"),
        "execution_mode": contract.get("execution_mode"),
        "task_type": resolved_intent.task_type,
        "pipeline_result": pr_dict,
        "final_answer": _redact_secrets(final_answer),
    }

    # Append-only: open in append mode, write one line, flush.
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace_record, ensure_ascii=False, sort_keys=True) + "\n")

    return TaskResult(
        intent_id=resolved_intent.intent_id,
        contract_id=str(contract.get("contract_id") or ""),
        contract_fingerprint=str(contract.get("contract_fingerprint") or ""),
        execution_mode=str(contract.get("execution_mode") or "dry-run"),
        pipeline_result=pr_dict,
        final_answer=final_answer,
        trace_path=str(trace_path),
        timestamp_utc=ts,
    )


def _build_final_answer(resolved_intent, contract, pipeline_result) -> str:
    """Format a concise final answer for the user."""
    pr_dict = (
        asdict(pipeline_result)
        if hasattr(pipeline_result, "__dataclass_fields__")
        else dict(pipeline_result)
    )
    lines = [
        f"[task] intent_id: {resolved_intent.intent_id}",
        f"[task] task_type: {resolved_intent.task_type}",
        f"[task] execution_mode: {contract.get('execution_mode')}",
        f"[task] skills_suggested: {','.join(resolved_intent.suggested_skills) or '(none)'}",
        f"[task] normalizer_verdict: {pr_dict.get('normalizer_verdict', '?')}",
        f"[task] reviewer_called: {pr_dict.get('reviewer_called', False)}",
        f"[task] reviewer_verdict: {pr_dict.get('reviewer_verdict') or '(skipped)'}",
        f"[task] engine_status: {pr_dict.get('engine_status', '?')}",
    ]
    if pr_dict.get("errors"):
        lines.append(f"[task] errors: {'; '.join(pr_dict['errors'])}")
    return "\n".join(lines)