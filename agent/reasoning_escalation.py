"""Profile-gated per-turn reasoning escalation for approved frontier routes.

The configured/session reasoning level remains authoritative. This module adds
an ephemeral override only when the policy is enabled, the active
provider/model match the allowlist, the current effort equals the configured
baseline, and several independent prompt-complexity dimensions cross a strict
gate. No prompt text is logged or persisted by this module.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_EXPLICIT_OPT_OUT = re.compile(
    r"\b(?:do\s+not|don't|never|without)\s+(?:use\s+|enable\s+|bump\s+to\s+|escalate\s+to\s+)?high\b"
    r"|\b(?:stay|keep|remain)\s+(?:at\s+|on\s+)?medium\b"
    r"|\bmedium\s+only\b",
    re.IGNORECASE,
)

_EXPLICIT_HIGH_REQUEST = re.compile(
    r"\b(?:use|enable|switch|bump|escalate)\s+(?:to\s+)?high\b"
    r"|\bextremely\s+complex\b"
    r"|\bleave\s+no\s+stone\s+unturned\b"
    r"|\bdeepest\s+possible\b"
    r"|\bmaximum\s+(?:possible\s+)?quality\b",
    re.IGNORECASE,
)

_ACTION_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("architecture", re.compile(r"\b(?:architect|architecture|design|spec(?:ification)?|plan)\w*\b", re.I)),
    ("implementation", re.compile(r"\b(?:implement|build|code|refactor|migrate|integrate|deploy)\w*\b", re.I)),
    ("diagnosis", re.compile(r"\b(?:debug|diagnos|root[- ]cause|recover|repair|troubleshoot)\w*\b", re.I)),
    ("research", re.compile(r"\b(?:research|compare|evaluate|source|cite|document)\w*\b", re.I)),
    ("audit", re.compile(r"\b(?:audit|review|threat[- ]model|risk[- ]assess|harden)\w*\b", re.I)),
    ("verification", re.compile(r"\b(?:test|verify|validat|benchmark|measure|prove)\w*\b", re.I)),
)

_BOUNDARY_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("routing_config", re.compile(r"\b(?:config|routing|profile|provider|model|fallback)\w*\b", re.I)),
    ("runtime", re.compile(r"\b(?:runtime|gateway|service|daemon|desktop|browser|process)\w*\b", re.I)),
    ("code_repo", re.compile(r"\b(?:code|source|repo|file|module|package|database|schema)\w*\b", re.I)),
    ("quality", re.compile(r"\b(?:test|ci|build|lint|benchmark|log|state\s*db)\w*\b", re.I)),
    ("knowledge", re.compile(r"\b(?:docs?|documentation|citation|research|source\s+map)\w*\b", re.I)),
    ("safety", re.compile(r"\b(?:security|privacy|auth|credential|secret|permission|production|live)\w*\b", re.I)),
)

_HIGH_STAKES = re.compile(
    r"\b(?:production|live\s+system|deploy|release|security|privacy|credential|secret|"
    r"payment|financial|legal|medical|irreversible|data\s+loss)\w*\b",
    re.I,
)

_ORCHESTRATION = re.compile(
    r"\b(?:end[- ]to[- ]end|entire\s+(?:stack|system|architecture)|multi[- ](?:agent|file|step|profile)|"
    r"parallel|orchestrat|coordinate|cross[- ](?:profile|service|platform|boundary))\w*\b",
    re.I,
)

_CONSTRAINT = re.compile(
    r"\b(?:preserve|unchanged|only|must|mustn't|do\s+not|don't|without|rollback|back\s*up|"
    r"approval|exact(?:ly)?|forbid|never|scope|boundary)\w*\b",
    re.I,
)

_OUTPUT_FAMILY = re.compile(
    r"\b(?:report|artifact|dashboard|table|json|schema|diagram|documentation|runbook|playbook|"
    r"migration\s+plan|launch\s+plan)\w*\b",
    re.I,
)


@dataclass(frozen=True)
class ReasoningEscalationDecision:
    """Sanitized decision record; contains no user prompt text."""

    escalate: bool
    selected_effort: str
    score: int = 0
    threshold: int = 0
    dimensions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


def _matched_names(
    text: str,
    families: Sequence[tuple[str, re.Pattern[str]]],
) -> tuple[str, ...]:
    return tuple(name for name, pattern in families if pattern.search(text))


def decide_reasoning_escalation(
    prompt: Any,
    *,
    threshold: int = 9,
    min_dimensions: int = 3,
    base_effort: str = "medium",
    escalated_effort: str = "high",
) -> ReasoningEscalationDecision:
    """Classify whether a prompt warrants the configured higher effort.

    The gate is deliberately multi-dimensional. Prompt length alone can add one
    dimension but can never trigger escalation by itself.
    """

    text = prompt if isinstance(prompt, str) else str(prompt or "")
    threshold = max(1, int(threshold))
    min_dimensions = max(1, int(min_dimensions))

    if _EXPLICIT_OPT_OUT.search(text):
        return ReasoningEscalationDecision(
            False,
            base_effort,
            threshold=threshold,
            reasons=("explicit_opt_out",),
        )

    score = 0
    dimensions: list[str] = []
    reasons: list[str] = []

    def add(dimension: str, points: int, reason: str) -> None:
        nonlocal score
        score += points
        dimensions.append(dimension)
        reasons.append(reason)

    if _EXPLICIT_HIGH_REQUEST.search(text):
        add("explicit_high", 4, "explicit_high_request")

    words = re.findall(r"\b\w+\b", text)
    if len(words) >= 300:
        add("large_prompt", 3, "prompt_words>=300")
    elif len(words) >= 120:
        add("large_prompt", 2, "prompt_words>=120")

    requirement_lines = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)
    requirement_terms = re.findall(
        r"\b(?:must|should|required|also|then|after|before|while|ensure|include)\b",
        text,
        re.I,
    )
    requirement_count = len(requirement_lines) + len(requirement_terms)
    if requirement_count >= 6:
        add("many_requirements", 3, "requirements>=6")
    elif requirement_count >= 4:
        add("many_requirements", 2, "requirements>=4")

    action_families = _matched_names(text, _ACTION_FAMILIES)
    if len(action_families) >= 3:
        add("multi_phase_work", 3, "action_families>=3")
    elif len(action_families) == 2:
        add("multi_phase_work", 2, "action_families=2")

    boundary_families = _matched_names(text, _BOUNDARY_FAMILIES)
    if len(boundary_families) >= 3:
        add("multi_boundary", 3, "boundaries>=3")
    elif len(boundary_families) == 2:
        add("multi_boundary", 2, "boundaries=2")

    if _HIGH_STAKES.search(text):
        add("high_stakes", 2, "high_stakes")

    if _ORCHESTRATION.search(text):
        add("orchestration", 2, "orchestration")

    constraint_count = len(_CONSTRAINT.findall(text))
    if constraint_count >= 4:
        add("constraint_density", 2, "constraints>=4")
    elif constraint_count >= 2:
        add("constraint_density", 1, "constraints>=2")

    output_count = len(_OUTPUT_FAMILY.findall(text))
    if output_count >= 3:
        add("multi_artifact", 1, "artifacts>=3")

    unique_dimensions = tuple(dict.fromkeys(dimensions))
    escalate = score >= threshold and len(unique_dimensions) >= min_dimensions
    return ReasoningEscalationDecision(
        escalate,
        escalated_effort if escalate else base_effort,
        score=score,
        threshold=threshold,
        dimensions=unique_dimensions,
        reasons=tuple(reasons),
    )


def load_reasoning_escalation_policy() -> dict[str, Any]:
    """Load the active profile policy; malformed/missing config fails closed."""

    try:
        path = Path(get_hermes_home()) / "config.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        agent_cfg = payload.get("agent") if isinstance(payload, dict) else None
        policy = (
            agent_cfg.get("reasoning_auto_escalation")
            if isinstance(agent_cfg, dict)
            else None
        )
        return dict(policy) if isinstance(policy, dict) else {}
    except Exception as exc:
        logger.warning(
            "Reasoning auto-escalation policy load failed; keeping baseline: %s",
            exc,
        )
        return {}


def _normalized_allowlist(value: Any) -> set[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _baseline_effort(agent: Any) -> str:
    cfg = getattr(agent, "reasoning_config", None)
    if not isinstance(cfg, dict) or cfg.get("enabled") is False:
        return ""
    return str(cfg.get("effort") or "").strip().lower()


def _ineligible_decision(
    base_effort: str,
    reason: str,
) -> ReasoningEscalationDecision:
    return ReasoningEscalationDecision(False, base_effort, reasons=(reason,))


def apply_turn_reasoning_escalation(
    agent: Any,
    prompt: Any,
    policy: Mapping[str, Any] | None = None,
) -> ReasoningEscalationDecision:
    """Reset, then optionally install a provider/model-scoped override."""

    agent._turn_reasoning_config_override = None
    agent._turn_reasoning_escalation_target = None
    agent._turn_reasoning_escalation_decision = None

    resolved = dict(policy) if policy is not None else load_reasoning_escalation_policy()
    base_effort = str(resolved.get("base_effort") or "medium").strip().lower()
    escalated_effort = str(resolved.get("escalated_effort") or "high").strip().lower()

    if resolved.get("enabled") is not True:
        decision = _ineligible_decision(base_effort, "policy_disabled")
        agent._turn_reasoning_escalation_decision = decision
        return decision

    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    model = str(getattr(agent, "model", "") or "").strip().lower()
    providers = _normalized_allowlist(resolved.get("providers"))
    models = _normalized_allowlist(resolved.get("models"))
    if provider not in providers or model not in models:
        decision = _ineligible_decision(base_effort, "target_mismatch")
        agent._turn_reasoning_escalation_decision = decision
        return decision

    if _baseline_effort(agent) != base_effort:
        decision = _ineligible_decision(base_effort, "baseline_mismatch")
        agent._turn_reasoning_escalation_decision = decision
        return decision

    decision = decide_reasoning_escalation(
        prompt,
        threshold=int(resolved.get("threshold", 9)),
        min_dimensions=int(resolved.get("min_dimensions", 3)),
        base_effort=base_effort,
        escalated_effort=escalated_effort,
    )
    agent._turn_reasoning_escalation_decision = decision

    if decision.escalate:
        agent._turn_reasoning_config_override = {
            "enabled": True,
            "effort": escalated_effort,
        }
        agent._turn_reasoning_escalation_target = (provider, model)
        if resolved.get("announce") is True:
            emit = getattr(agent, "_emit_status", None)
            if callable(emit):
                emit(
                    f"🧠 Reasoning auto-escalation: {base_effort} → "
                    f"{escalated_effort} (complexity "
                    f"{decision.score}/{decision.threshold})"
                )

    if resolved.get("log_decisions") is True:
        logger.info(
            "Reasoning auto-escalation decision: session=%s provider=%s model=%s "
            "baseline=%s selected=%s score=%s threshold=%s dimensions=%s "
            "signals=%s",
            getattr(agent, "session_id", None) or "none",
            provider,
            model,
            base_effort,
            decision.selected_effort,
            decision.score,
            decision.threshold,
            ",".join(decision.dimensions) or "none",
            ",".join(decision.reasons) or "none",
        )

    return decision


def effective_reasoning_config(agent: Any) -> Any:
    """Return the effective wire effort for the active turn."""

    override = getattr(agent, "_turn_reasoning_config_override", None)
    target = getattr(agent, "_turn_reasoning_escalation_target", None)
    active = (
        str(getattr(agent, "provider", "") or "").strip().lower(),
        str(getattr(agent, "model", "") or "").strip().lower(),
    )
    if isinstance(override, dict) and isinstance(target, tuple) and target == active:
        return override
    return getattr(agent, "reasoning_config", None)
