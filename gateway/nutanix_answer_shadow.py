"""Shadow-mode post-answer verification for Nutanix RAG answers.

This module is intentionally small and optional: failures must never block or
mutate delivery. It only extracts Hermes/NX-Shield RAG tool output from the
current turn, runs the external Nutanix answer verifier, and writes a local JSON
report for later review.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from hermes_constants import get_hermes_home

RAG_TOOL_NAMES = {
    "hermes_master_search",
    "mcp_nutanix_rag_search_hermes_master_search",
    "mcp_nutanix_rag_search_canary_hermes_master_search",
}
DEFAULT_RAG_ROOT = get_hermes_home() / "rag" / "nutanix"
PASS_VERDICTS = {"PASS", "PASS_WITH_WARNINGS"}
REVISION_VERDICTS = {"REWRITE_REQUIRED"}
FAIL_CLOSED_VERDICTS = {"FAIL_CLOSED"}
DeliveryAction = Literal["send_original", "request_revision", "send_evidence_fallback", "unchanged"]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _tool_name_and_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function") if isinstance(call, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or ""), _parse_tool_arguments(fn.get("arguments"))
    return str(call.get("name") or ""), _parse_tool_arguments(call.get("arguments"))


def _iter_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    return [call for call in raw if isinstance(call, dict)]


def _strip_tool_output_query_echo(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        if line.strip().lower().startswith("query:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_rag_evidence(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract current-turn RAG MCP tool outputs as verifier evidence rows."""
    calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    evidence: list[dict[str, str]] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for call in _iter_tool_calls(msg.get("tool_calls")):
                call_id = str(call.get("id") or "")
                tool_name, args = _tool_name_and_args(call)
                if call_id and tool_name in RAG_TOOL_NAMES:
                    calls_by_id[call_id] = (tool_name, args)
            continue

        if msg.get("role") not in {"tool", "function"}:
            continue

        tool_name = str(msg.get("name") or msg.get("tool_name") or "")
        args: dict[str, Any] = {}
        call_id = str(msg.get("tool_call_id") or "")
        if call_id in calls_by_id:
            tool_name, args = calls_by_id[call_id]

        if tool_name not in RAG_TOOL_NAMES:
            continue

        text = _strip_tool_output_query_echo(_safe_str(msg.get("content"))).strip()
        if not text:
            continue
        query = _safe_str(args.get("query")).strip()
        row = {
            "source": f"mcp_tool:{tool_name}",
            "tool_name": tool_name,
            "query": query,
            "text": text,
        }
        if "[OFFICIAL_PORTAL]" in text or "official_nutanix_portal" in text.lower():
            row["source_authority"] = "official_nutanix_portal"
        evidence.append(row)

    return evidence


def default_shadow_audit_dir() -> Path:
    configured = os.environ.get("NUTANIX_ANSWER_VERIFIER_AUDIT_DIR", "").strip()
    if configured:
        return Path(configured)
    rag_root = Path(os.environ.get("HERMES_NUTANIX_RAG_ROOT", str(DEFAULT_RAG_ROOT)))
    day = datetime.now().strftime("%Y%m%d")
    return rag_root / "audits" / "answer_verifier_shadow" / day


def _load_default_verifier() -> Callable[..., dict[str, Any]]:
    rag_root = Path(os.environ.get("HERMES_NUTANIX_RAG_ROOT", str(DEFAULT_RAG_ROOT)))
    src = rag_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    module = importlib.import_module("nx_rag.answer_verifier")
    return module.verify_answer


def _report_name(session_id: str, query: str) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S%z")
    digest = hashlib.sha256(f"{session_id}\n{query}".encode("utf-8")).hexdigest()[:12]
    safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)[:80]
    return f"{ts}_{safe_session or 'session'}_{digest}.json"


def enforcement_enabled_from_config(config: dict[str, Any] | None = None) -> bool:
    """Resolve the disabled-by-default enforcement gate.

    Env ``NUTANIX_ANSWER_VERIFIER_ENFORCE`` wins. Config path is
    ``rag.answer_verification.enforce_enabled``. This helper is intentionally
    separate from shadow mode so shadow reporting can stay on while delivery
    enforcement remains off.
    """
    raw_env = os.environ.get("NUTANIX_ANSWER_VERIFIER_ENFORCE")
    if raw_env is not None:
        return raw_env.strip().lower() in {"1", "true", "yes", "on"}
    cfg = config or {}
    try:
        rag = cfg.get("rag") if isinstance(cfg, dict) else None
        av = rag.get("answer_verification") if isinstance(rag, dict) else None
        return bool(av.get("enforce_enabled")) if isinstance(av, dict) else False
    except Exception:
        return False


def build_evidence_fallback(query: str, evidence: list[dict[str, Any]], verification: dict[str, Any] | None = None) -> str:
    """Build a conservative fallback answer from evidence metadata/text snippets.

    This avoids asserting unsupported claims. It is only a candidate fallback for
    future enforcement mode; the current gateway path does not send it.
    """
    lines = [
        "I could not verify the drafted answer strongly enough against the retrieved Nutanix evidence.",
        "Here is the safest evidence-bound summary instead:",
    ]
    if query:
        lines.append(f"- Query: {query}")
    for idx, row in enumerate(evidence[:3], start=1):
        source = _safe_str(row.get("source") or row.get("doc_id") or row.get("tool_name") or "Nutanix RAG evidence")
        text = re.sub(r"\s+", " ", _safe_str(row.get("text"))).strip()
        snippet = text[:300] + ("…" if len(text) > 300 else "")
        lines.append(f"- Evidence {idx}: {source}")
        if snippet:
            lines.append(f"  - {snippet}")
    issues = []
    if isinstance(verification, dict):
        issues = [str(item) for item in verification.get("issues") or [] if str(item)]
    if issues:
        lines.append("- Verification issues:")
        for issue in issues[:5]:
            lines.append(f"  - {issue}")
    return "\n".join(lines)


def build_revision_request_prompt(
    *,
    query: str,
    draft_answer: str,
    evidence: list[dict[str, Any]],
    verification: dict[str, Any] | None,
) -> str:
    """Build a bounded prompt for a single verifier-driven answer revision.

    The prompt is pure data: callers may use it to run exactly one additional
    model turn, then must verify the revised answer again before delivery. Keep
    evidence snippets short to avoid leaking unrelated retrieval text or causing
    the revision step to drift beyond current-turn evidence.
    """
    lines = [
        "Revise the Nutanix answer using only the retrieved evidence and verifier feedback below.",
        "Do not introduce new claims, assumptions, or external sources.",
        "If the evidence does not support a claim, remove it or state the limitation.",
        "Return only the revised user-facing answer.",
        "",
    ]
    if query:
        lines.extend(["User query:", query.strip(), ""])
    if draft_answer:
        lines.extend(["Draft answer to revise:", draft_answer.strip(), ""])
    issues: list[str] = []
    warnings: list[str] = []
    verdict = "UNKNOWN"
    if isinstance(verification, dict):
        verdict = _safe_str(verification.get("verdict") or "UNKNOWN") or "UNKNOWN"
        issues = [_safe_str(item).strip() for item in verification.get("issues") or [] if _safe_str(item).strip()]
        warnings = [_safe_str(item).strip() for item in verification.get("warnings") or [] if _safe_str(item).strip()]
    lines.extend(["Verifier result:", f"- Verdict: {verdict}"])
    if issues:
        lines.append("- Issues:")
        for issue in issues[:5]:
            lines.append(f"  - {issue}")
    if warnings:
        lines.append("- Warnings:")
        for warning in warnings[:5]:
            lines.append(f"  - {warning}")
    lines.append("")
    lines.append("Retrieved evidence, limited to top 3 rows:")
    for idx, row in enumerate(evidence[:3], start=1):
        source = _safe_str(row.get("source") or row.get("doc_id") or row.get("tool_name") or "Nutanix RAG evidence")
        text = re.sub(r"\s+", " ", _safe_str(row.get("text"))).strip()
        snippet = text[:500] + ("…" if len(text) > 500 else "")
        lines.append(f"- Evidence {idx}: {source}")
        if snippet:
            lines.append(f"  - {snippet}")
    return "\n".join(lines).strip()


def decide_delivery_action(
    *,
    enforce_enabled: bool,
    verification: dict[str, Any] | None,
    query: str,
    evidence: list[dict[str, Any]],
    revision_attempted: bool = False,
    draft_answer: str = "",
) -> dict[str, Any]:
    """Return a delivery decision for future verifier enforcement.

    This is pure decision logic and has no side effects. The current production
    gateway still uses shadow mode only unless explicit wiring enables it later.
    """
    if not enforce_enabled:
        return {"action": "unchanged", "reason": "enforcement_disabled"}
    if not isinstance(verification, dict):
        return {
            "action": "send_evidence_fallback",
            "reason": "missing_verification_report",
            "fallback_answer": build_evidence_fallback(query, evidence, verification),
        }
    verdict = str(verification.get("verdict") or "UNKNOWN")
    if verdict in PASS_VERDICTS:
        return {"action": "send_original", "reason": f"verdict_{verdict.lower()}", "verdict": verdict}
    if verdict in REVISION_VERDICTS and not revision_attempted:
        return {
            "action": "request_revision",
            "reason": "rewrite_required_before_delivery",
            "verdict": verdict,
            "issues": verification.get("issues", []),
            "warnings": verification.get("warnings", []),
            "revision_prompt": build_revision_request_prompt(
                query=query,
                draft_answer=draft_answer,
                evidence=evidence,
                verification=verification,
            ),
        }
    return {
        "action": "send_evidence_fallback",
        "reason": "fail_closed_or_revision_exhausted",
        "verdict": verdict,
        "fallback_answer": build_evidence_fallback(query, evidence, verification),
    }


def maybe_run_shadow_verification(
    *,
    enabled: bool,
    query: str,
    answer: str,
    messages: list[dict[str, Any]],
    identity: str,
    audit_dir: str | Path | None = None,
    session_id: str = "",
    platform: str = "",
    chat_id: str = "",
    verifier: Callable[..., dict[str, Any]] | None = None,
) -> str | None:
    """Run verifier in shadow mode and write a report.

    Returns the report path when a report is written. Returns ``None`` when the
    feature is disabled, there is no answer, or no RAG evidence exists. Raises no
    deliberate delivery-affecting exceptions; callers should still wrap this in a
    best-effort try/except at the delivery boundary.
    """
    if not enabled or not (answer or "").strip():
        return None

    evidence = extract_rag_evidence(messages)
    if not evidence:
        return None

    verifier = verifier or _load_default_verifier()
    verification = verifier(
        query=query or (evidence[-1].get("query") or ""),
        answer=answer,
        evidence=evidence,
        identity=identity or "hermes",
    )
    verdict = str(verification.get("verdict") or "UNKNOWN") if isinstance(verification, dict) else "UNKNOWN"

    out_dir = Path(audit_dir) if audit_dir is not None else default_shadow_audit_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / _report_name(session_id, query or evidence[-1].get("query", ""))

    report = {
        "schema_version": 1,
        "mode": "shadow",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "platform": platform,
        "chat_id": chat_id,
        "identity": identity or "hermes",
        "query": query,
        "answer": answer,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "verdict": verdict,
        "verification": verification,
        "delivery_action": "unchanged",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(report_path)


def shadow_enabled_from_config(config: dict[str, Any] | None = None) -> bool:
    """Resolve the shadow-mode gate from env or config.

    Env ``NUTANIX_ANSWER_VERIFIER_SHADOW`` wins. Config path is
    ``rag.answer_verification.shadow_enabled``.
    """
    raw_env = os.environ.get("NUTANIX_ANSWER_VERIFIER_SHADOW")
    if raw_env is not None:
        return raw_env.strip().lower() in {"1", "true", "yes", "on"}
    cfg = config or {}
    try:
        rag = cfg.get("rag") if isinstance(cfg, dict) else None
        av = rag.get("answer_verification") if isinstance(rag, dict) else None
        return bool(av.get("shadow_enabled")) if isinstance(av, dict) else False
    except Exception:
        return False
