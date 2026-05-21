"""Deterministic first-pass router for the Jeeves swarm operator."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agent.swarm_state import EvidenceRequirement, PermissionGrant, RoutingPlan, RoutingTelemetry

_PERMISSION_WORDS = (
    "send", "email", "message", "post", "publish", "deploy", "restart", "delete", "remove",
    "drop", "destroy", "overwrite", "merge", "push", "charge", "buy", "purchase", "commit",
)
_PIPE_WORDS = ("n8n", "docker", "container", "compose", "workflow", "webhook")
_SWARM_WORDS = (
    "research", "review", "audit", "compare", "investigate", "lookup", "look up", "analyze",
    "code", "implement", "fix", "test", "summarize", "parallel", "subagent", "agents", "council",
    "critique", "failure mode", "architecture", "strategy",
)
_PROCEDURAL_WORDS = (
    "then", "after that", "next", "finally", "validate", "verify", "check", "source of truth",
    "ground truth", "eval", "run", "collect", "parse", "extract",
)
_CONTEXT_REFERENCE_WORDS = (
    "other session", "earlier", "previous", "above", "prior", "thread", "that file", "those files",
    "last night", "we built", "context", "council",
)
_CURRENT_FACT_WORDS = ("current", "latest", "today", "news", "research", "docs", "available", "native")
_CODE_WORDS = ("code", "implement", "fix", "patch", "config", "edit", "test", "tests", "lint", "typecheck", "function", "executor")


def _count_steps(text: str) -> int:
    lowered = text.lower()
    numbered = len(re.findall(r"(?:^|[\n;])\s*(?:\d+\.|[-*])\s+", text))
    connectors = sum(lowered.count(word) for word in _PROCEDURAL_WORDS)
    # Explicit commas/semicolons with procedural verbs are a decent v1 signal.
    return max(numbered, connectors)


def _permission_requests(text: str) -> List[PermissionGrant]:
    lowered = text.lower()
    requests: List[PermissionGrant] = []
    for word in _PERMISSION_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            requests.append(
                PermissionGrant(
                    permission_id=f"perm_{word.replace(' ', '_')}",
                    description=f"Permission required before external/destructive action: {word}",
                    scope={"matched_word": word},
                )
            )
            break
    if any(word in lowered for word in _PIPE_WORDS):
        requests.append(
            PermissionGrant(
                permission_id="perm_pipe_execution",
                description="Permission required before invoking n8n/docker execution pipes",
                scope={"matched_pipe": True},
            )
        )
    return requests


def _suggested_tasks(text: str, mode: str) -> List[Dict[str, Any]]:
    if mode == "direct":
        return []
    chunks = [part.strip(" .") for part in re.split(r"\b(?:and|then|;|\n)\b", text, flags=re.IGNORECASE) if part.strip()]
    if len(chunks) < 2:
        chunks = [text.strip()]
    return [
        {"title": chunk[:80], "description": chunk, "mode": mode}
        for chunk in chunks[:6]
    ]


def _bounded_score(value: int) -> int:
    return max(0, min(3, value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _contains_word(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def score_context_pressure(text: str, platform_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score context-cliff risk with mechanical signals only."""
    ctx = platform_context or {}
    lowered = (text or "").lower()
    signals: Dict[str, Any] = {}
    score = 0

    if len(text or "") > 1200:
        score += 1
        signals["long_prompt"] = True
    if len(text or "") > 3000:
        score += 1
        signals["very_long_prompt"] = True
    cross_session = any(word in lowered for word in _CONTEXT_REFERENCE_WORDS)
    if cross_session:
        score += 1
        signals["cross_session_reference"] = True
    thread_count = _safe_int(ctx.get("thread_message_count") or ctx.get("message_count") or 0)
    if thread_count >= 10:
        score += 1
        signals["long_thread"] = thread_count
    if thread_count >= 20:
        score += 1
        signals["very_long_thread"] = thread_count
    attachment_count = _safe_int(ctx.get("attachment_count") or ctx.get("file_count") or 0)
    if attachment_count:
        score += 1
        signals["attachment_count"] = attachment_count
    if ctx.get("compressed_context"):
        score += 1
        signals["compressed_context"] = True
    source_count = _safe_int(ctx.get("source_count") or 0)
    if source_count >= 3:
        score += 1
        signals["many_source_systems"] = source_count

    return {"score": _bounded_score(score), "signals": signals}


def score_complexity_risk(text: str, *, step_count: int = 0, permissions: Optional[List[PermissionGrant]] = None) -> Dict[str, Any]:
    lowered = (text or "").lower()
    signals: Dict[str, Any] = {}
    score = 0
    if step_count > 2:
        score += 1
        signals["multi_step"] = step_count
    if step_count > 4:
        score += 1
        signals["long_dependency_chain"] = step_count
    if _contains_word(lowered, _CODE_WORDS):
        score += 1
        signals["code_or_config"] = True
    if permissions:
        score += 2
        signals["external_or_destructive"] = True
    if any(word in lowered for word in ("source of truth", "ground truth", "eval", "audit", "verify", "validate")):
        score += 1
        signals["verification_burden"] = True
    if any(word in lowered for word in ("architecture", "strategy", "council", "failure mode", "critique")):
        score += 1
        signals["ambiguous_judgment"] = True
    return {"score": _bounded_score(score), "signals": signals}


def score_illusion_risk(text: str, complexity_score: int, context_score: int, mode: str) -> Dict[str, Any]:
    lowered = (text or "").lower()
    signals: Dict[str, Any] = {}
    score = 0
    if complexity_score >= 2:
        score += 1
    if context_score >= 2:
        score += 1
    if mode == "swarm":
        score += 1
        signals["child_summary_risk"] = True
    if any(word in lowered for word in _CURRENT_FACT_WORDS):
        score += 1
        signals["current_fact_claims_need_sources"] = True
    if _contains_word(lowered, _CODE_WORDS) and "test" not in lowered:
        score += 1
        signals["code_claims_need_tests"] = True
    if any(word in lowered for word in ("research", "review", "audit", "analyze", "summarize")):
        signals["broad_analysis"] = True
    return {"score": _bounded_score(score), "signals": signals}


def _required_scaffold(mode: str, permissions: List[PermissionGrant], complexity: int, context: int) -> str:
    if permissions:
        return "approval"
    if mode in {"script", "pipe"}:
        return "script" if mode == "script" else "dry_run"
    if context >= 3:
        return "externalize_state"
    if mode == "swarm" or complexity >= 2:
        return "parallel_review"
    if complexity or context:
        return "checklist"
    return "none"


def _evidence_requirements(text: str, mode: str, permissions: List[PermissionGrant], telemetry: RoutingTelemetry) -> List[EvidenceRequirement]:
    lowered = (text or "").lower()
    requirements: List[EvidenceRequirement] = []
    code_like = _contains_word(lowered, _CODE_WORDS)
    research_like = any(word in lowered for word in ("research", "audit", "current", "latest", "docs", "available", "native")) or (
        any(word in lowered for word in ("review", "analyze")) and not code_like
    )
    if research_like:
        requirements.append(EvidenceRequirement("citation", "Provide source links or named source-of-truth evidence for research/current factual claims", source="router"))
    if mode == "script" or telemetry.required_scaffold == "script":
        requirements.append(EvidenceRequirement("command_output", "Provide command/script output or a skipped-verification reason", source="router"))
    if mode == "pipe" or any(word in lowered for word in _PIPE_WORDS):
        requirements.append(EvidenceRequirement("dry_run", "Provide dry-run output or payload contract before live pipe execution", source="router"))
    if _contains_word(lowered, _CODE_WORDS):
        requirements.append(EvidenceRequirement("test", "Provide test/lint/typecheck evidence or explicit skipped-verification reason", source="router"))
    if permissions:
        requirements.append(EvidenceRequirement("human_approval", "Get explicit human approval before external, destructive, or client-facing side effects", source="router"))
    if telemetry.required_scaffold == "externalize_state":
        requirements.append(EvidenceRequirement("artifact", "Externalize state into a plan/checklist/artifact before synthesis", source="router"))

    deduped: List[EvidenceRequirement] = []
    seen: set[str] = set()
    for req in requirements:
        if req.kind not in seen:
            deduped.append(req)
            seen.add(req.kind)
    return deduped


def _panel_policy(text: str, mode: str, complexity: int, context: int, illusion: int) -> Dict[str, Any]:
    lowered = (text or "").lower()
    considered = mode != "direct" or complexity > 0 or context > 0
    explicit_council = any(word in lowered for word in ("council", "panel", "skeptic", "critique"))
    required = bool(explicit_council or mode == "swarm" or complexity >= 2 or illusion >= 3)
    roles: List[str] = []
    if required:
        if any(word in lowered for word in ("architecture", "strategy", "failure mode", "council")):
            roles = ["scout", "skeptic", "synthesizer"]
        elif _contains_word(lowered, _CODE_WORDS):
            roles = ["builder", "reviewer", "verifier"]
        else:
            roles = ["researcher", "skeptic", "synthesizer"]
    return {
        "considered": considered,
        "required": required,
        "roles": roles,
        "reason": "Panel required by complexity/context/illusion risk" if required else "Panel considered but not required",
    }


def route_request(
    text: str,
    platform_context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> RoutingPlan:
    """Classify a request without LLM calls.

    Modes are conservative candidates: ``direct``, ``swarm``, ``script``, or
    ``pipe``. Permission requests are advisory and default-deny downstream work.
    """

    del config  # reserved for later live policy inputs
    raw_text = text or ""
    lowered = raw_text.lower()
    permissions = _permission_requests(raw_text)
    step_count = _count_steps(raw_text)

    if any(word in lowered for word in _PIPE_WORDS):
        mode = "pipe"
        reason = "Request mentions n8n/docker-style execution pipes."
    elif step_count > 3:
        mode = "script"
        reason = "Request appears to have more than three procedural/validation steps."
    elif (
        sum(1 for word in _SWARM_WORDS if word in lowered) >= 2
        or (" and " in lowered and any(word in lowered for word in _SWARM_WORDS))
        or re.search(r"\b(a|b|c)\b.*\b(a|b|c)\b", lowered)
    ):
        mode = "swarm"
        reason = "Request contains multiple independent research/code/review signals."
    else:
        mode = "direct"
        reason = "Simple prompt suitable for direct handling."

    context_result = score_context_pressure(raw_text, platform_context)
    complexity_result = score_complexity_risk(raw_text, step_count=step_count, permissions=permissions)
    illusion_result = score_illusion_risk(raw_text, complexity_result["score"], context_result["score"], mode)
    scaffold = _required_scaffold(mode, permissions, complexity_result["score"], context_result["score"])
    telemetry = RoutingTelemetry(
        complexity_risk=complexity_result["score"],
        context_pressure=context_result["score"],
        illusion_risk=illusion_result["score"],
        required_scaffold=scaffold,
        weak_output_risk=illusion_result["score"] >= 2,
        signals={**context_result["signals"], **complexity_result["signals"], **illusion_result["signals"]},
    )
    evidence_requirements = _evidence_requirements(raw_text, mode, permissions, telemetry)
    panel_policy = _panel_policy(raw_text, mode, telemetry.complexity_risk, telemetry.context_pressure, telemetry.illusion_risk)

    verification_required = mode in {"swarm", "script", "pipe"} or bool(permissions) or bool(evidence_requirements) or any(
        word in lowered for word in ("verify", "validate", "test", "check")
    )
    return RoutingPlan(
        mode=mode,
        reason=reason,
        suggested_tasks=_suggested_tasks(raw_text, mode),
        permission_requests=permissions,
        verification_required=verification_required,
        metadata={
            "step_count": step_count,
            "permission_required": bool(permissions),
            "complexity_risk": telemetry.complexity_risk,
            "context_pressure": telemetry.context_pressure,
            "illusion_risk": telemetry.illusion_risk,
            "required_scaffold": telemetry.required_scaffold,
        },
        evidence_requirements=evidence_requirements,
        routing_telemetry=telemetry,
        panel_policy=panel_policy,
    )


__all__ = [
    "route_request",
    "score_complexity_risk",
    "score_context_pressure",
    "score_illusion_risk",
]
