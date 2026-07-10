"""AMDP config resolution + validation.

Config lives under the top-level ``amdp:`` block in ``config.yaml`` and is
default-OFF. Shape (mirrors the ``moa:`` preset style)::

    amdp:
      enabled: false
      planner:   {provider: onyx-6000,  model: qwen3.6-27b-nvfp4}
      reviewer:  {provider: ref-gptoss, model: gpt-oss-20b}
      n_coas: 3
      gate: {min_estimated_steps: 3}
      staleness_max_s: 120
      audit_log: amdp_audit.jsonl
      hitl_gate_irreversible: true

Validation is deliberately strict when ``enabled: true`` and hard-fails rather
than silently degrading — an empty reviewer slot with AMDP on would otherwise
fall through to a cloud model (the same class of trap MoA hit with
``reference_models: []`` + ``enabled: true`` → silent cloud fallback). A
disabled or absent block returns ``None`` and the caller treats AMDP as off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class AmdpConfigError(ValueError):
    """Raised when an *enabled* AMDP config is invalid. Callers surface this at
    startup rather than letting a misconfigured planner run."""


@dataclass
class AmdpConfig:
    enabled: bool
    planner: dict[str, str]
    reviewer: dict[str, str]
    n_coas: int = 3
    min_estimated_steps: int = 3
    staleness_max_s: float = 120.0
    audit_log: str = "amdp_audit.jsonl"
    hitl_gate_irreversible: bool = True
    decision_profile: str = "v1_balanced"
    reviewer_max_tokens: int | None = 1800
    call_timeout_s: float = 90.0
    episode_deadline_s: float = 240.0
    intake_timeout_s: float = 4.0
    state_feed: str = "auto"          # auto | gateway | proprioception
    exclude_background: bool = False
    review_only_on_risk_or_disagreement: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def _valid_slot(slot: Any) -> bool:
    """A slot is valid only if provider AND model are non-blank STRINGS. A YAML
    list/int/dict for provider must be rejected, not stringified — otherwise a
    garbage provider ('[a, b]') sails through and fails opaquely downstream."""
    if not isinstance(slot, dict):
        return False
    provider, model = slot.get("provider"), slot.get("model")
    return (
        isinstance(provider, str) and bool(provider.strip())
        and isinstance(model, str) and bool(model.strip())
    )


def resolve_amdp_config(config: dict[str, Any] | None) -> AmdpConfig | None:
    """Parse + validate the ``amdp:`` block. Returns None when AMDP is absent or
    disabled. Raises ``AmdpConfigError`` when enabled but misconfigured."""
    block = (config or {}).get("amdp")
    if not isinstance(block, dict):
        return None
    enabled = bool(block.get("enabled", False))
    if not enabled:
        return None

    planner = block.get("planner") or {}
    reviewer = block.get("reviewer") or {}
    if not _valid_slot(planner):
        raise AmdpConfigError(
            "amdp.enabled is true but amdp.planner is missing a provider/model. "
            "Refusing to run AMDP with an unresolved commander (would fall back to a default model)."
        )
    if not _valid_slot(reviewer):
        raise AmdpConfigError(
            "amdp.enabled is true but amdp.reviewer is missing a provider/model. "
            "Refusing to run AMDP with an empty reviewer (would silently skip war-gaming or "
            "fall back to a cloud model)."
        )

    gate = block.get("gate") or {}
    try:
        n_coas = max(1, int(block.get("n_coas", 3)))
    except (TypeError, ValueError):
        n_coas = 3
    try:
        min_steps = max(1, int(gate.get("min_estimated_steps", 3)))
    except (TypeError, ValueError):
        min_steps = 3
    try:
        staleness = float(block.get("staleness_max_s", 120))
    except (TypeError, ValueError):
        staleness = 120.0
    try:
        call_timeout_s = float(block.get("call_timeout_s", 90))
    except (TypeError, ValueError):
        call_timeout_s = 90.0
    try:
        episode_deadline_s = float(block.get("episode_deadline_s", 240))
    except (TypeError, ValueError):
        episode_deadline_s = 240.0
    try:
        intake_timeout_s = float(block.get("intake_timeout_s", 4))
    except (TypeError, ValueError):
        intake_timeout_s = 4.0

    # decision_profile must name a real scoring profile, else _decide would
    # KeyError on every turn (a typo like 'balanced' would silently disable
    # planning after paying for all the model calls). Fall back with a warning.
    from agent.amdp import scoring

    profile = str(block.get("decision_profile") or "v1_balanced")
    if profile not in scoring.PROFILES:
        logger.warning(
            "amdp.decision_profile %r is not a known profile %s; using v1_balanced",
            profile, sorted(scoring.PROFILES),
        )
        profile = "v1_balanced"

    rmt = block.get("reviewer_max_tokens", 1800)
    try:
        reviewer_max_tokens = None if rmt in (None, 0, "", "none") else int(rmt)
    except (TypeError, ValueError):
        logger.warning("amdp.reviewer_max_tokens %r invalid; using 1800", rmt)
        reviewer_max_tokens = 1800

    state_feed = str(block.get("state_feed") or "auto").strip().lower()
    if state_feed not in ("auto", "gateway", "proprioception", "telemetry"):
        logger.warning("amdp.state_feed %r unknown; using auto", state_feed)
        state_feed = "auto"

    return AmdpConfig(
        enabled=True,
        planner={"provider": str(planner["provider"]).strip(), "model": str(planner["model"]).strip()},
        reviewer={"provider": str(reviewer["provider"]).strip(), "model": str(reviewer["model"]).strip()},
        n_coas=n_coas,
        min_estimated_steps=min_steps,
        staleness_max_s=staleness,
        audit_log=str(block.get("audit_log") or "amdp_audit.jsonl"),
        hitl_gate_irreversible=bool(block.get("hitl_gate_irreversible", True)),
        decision_profile=profile,
        reviewer_max_tokens=reviewer_max_tokens,
        call_timeout_s=call_timeout_s,
        episode_deadline_s=episode_deadline_s,
        intake_timeout_s=intake_timeout_s,
        state_feed=state_feed,
        exclude_background=bool(block.get("exclude_background", False)),
        review_only_on_risk_or_disagreement=bool(
            block.get("review_only_on_risk_or_disagreement", False)
        ),
        raw=block,
    )
