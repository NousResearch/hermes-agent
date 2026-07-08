"""Hermes-level adaptive model routing for main agent turns.

This module is deliberately small and inert by default.  It chooses a configured
model tier for a *main* user turn before an ``AIAgent`` is constructed, reusing
``resolve_runtime_provider`` for provider/auth/base-url resolution instead of
adding a parallel provider map.

It is distinct from:
- ``provider_routing``: OpenRouter's within-model provider-selection knobs.
- ``auxiliary.*``: side-task LLM routing (compression, vision, titles, ...).
- ``fallback_providers``: failure recovery after a model/provider errors.
- ``delegation.*`` / cron job model overrides: explicit task/job routing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

RouteRuntimeResolver = Callable[..., Mapping[str, Any]]

_OVERRIDE_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\b(use|route\s+to|switch\s+to)\s+(the\s+)?(cheapest|cheap|fast|fastest|small|smallest)\s+model\b", re.I), "cheap", "explicit override"),
    (re.compile(r"\b(quick|fast)\s+(answer|reply|response)\b", re.I), "cheap", "explicit quick-answer override"),
    (re.compile(r"\b(use|route\s+to|switch\s+to)\s+(the\s+)?(balanced|normal|default)\s+model\b", re.I), "balanced", "explicit override"),
    (re.compile(r"\b(use|route\s+to|switch\s+to)\s+(the\s+)?(best|strongest|smartest|premium|deepest)\s+model\b", re.I), "best", "explicit override"),
    (re.compile(r"\b(use|route\s+to|switch\s+to)\s+(fable|opus|o3|gpt-5\.5|gpt5\.5)\b", re.I), "best", "explicit premium-model override"),
    (re.compile(r"\b(think\s+deeply|deep\s+research|deep\s+reasoning|be\s+very\s+careful)\b", re.I), "best", "explicit deep-reasoning override"),
)

_BEST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\b(architecture|architectural|design\s+the\s+system|system\s+design|routing\s+engine)\b",
        r"\b(hermes\s+internals|provider\s+selection|model\s+routing|fallback\s+chain|prompt\s+cach(?:e|ing)|role\s+alternation)\b",
        r"\b(difficult|hard|tricky|unknown)\s+(bug|debug|failure|regression)\b",
        r"\b(large|big|major)\s+(refactor|migration|rewrite|codebase\s+change)\b",
        r"\b(security|privacy|medical|legal|financial|high[-\s]?stakes)\b",
        r"\b(audit|review)\s+(the\s+)?(architecture|security|whole\s+repo|codebase)\b",
    )
)

_BALANCED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\b(debug|troubleshoot|fix|implement|add\s+tests?|write\s+code|refactor|build|install|configure)\b",
        r"\b(use\s+tools?|run\s+tests?|inspect\s+the\s+repo|open\s+a\s+PR|create\s+an\s+issue)\b",
        r"\b(research|compare|investigate|analy[sz]e)\b",
    )
)

_CHEAP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\b(summarize|summarise|rewrite|rephrase|translate|proofread)\b",
        r"\b(what\s+is|who\s+is|when\s+is|where\s+is|define|explain\s+briefly)\b",
    )
)


@dataclass(frozen=True)
class RoutingDecision:
    tier: str
    provider: str
    model: str
    runtime: dict[str, Any]
    reason: str
    confidence: float
    source: str

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.model,
            self.provider,
            self.runtime.get("base_url"),
            self.runtime.get("api_mode"),
            self.runtime.get("command"),
            tuple(self.runtime.get("args") or ()),
        )


def routing_enabled(config: Mapping[str, Any]) -> bool:
    cfg = config.get("model_routing") if isinstance(config, Mapping) else None
    return isinstance(cfg, Mapping) and bool(cfg.get("enabled", False))


def choose_tier(user_message: str, config: Mapping[str, Any]) -> tuple[str, str, float, str]:
    """Return ``(tier, reason, confidence, source)`` for ``user_message``.

    Deterministic by design: no extra classifier LLM call, no hidden spend.
    """
    cfg = config.get("model_routing") if isinstance(config, Mapping) else {}
    cfg = cfg if isinstance(cfg, Mapping) else {}
    default_tier = str(cfg.get("default_tier") or "balanced").strip() or "balanced"
    text = user_message or ""
    stripped = text.strip()

    for pattern, tier, reason in _OVERRIDE_PATTERNS:
        if pattern.search(stripped):
            return tier, reason, 0.99, "override"

    word_count = len(re.findall(r"\w+", stripped))
    char_count = len(stripped)
    if char_count <= 80 and word_count <= 14 and "\n" not in stripped:
        # Keep common tiny acknowledgements and factual one-liners cheap unless
        # they explicitly ask for a hard/sensitive task.
        if not any(p.search(stripped) for p in _BEST_PATTERNS):
            return "cheap", "short/simple user turn", 0.82, "heuristic"

    if any(p.search(stripped) for p in _BEST_PATTERNS):
        return "best", "architecture/debugging/security/high-stakes task", 0.86, "heuristic"
    if any(p.search(stripped) for p in _BALANCED_PATTERNS):
        return "balanced", "normal tool/coding/research task", 0.74, "heuristic"
    if any(p.search(stripped) for p in _CHEAP_PATTERNS):
        return "cheap", "simple language/factual task", 0.72, "heuristic"
    return default_tier, "default routing tier", 0.50, "policy"


def _tier_entry(config: Mapping[str, Any], tier: str) -> Optional[Mapping[str, Any]]:
    cfg = config.get("model_routing") if isinstance(config, Mapping) else None
    tiers = cfg.get("tiers") if isinstance(cfg, Mapping) else None
    entry = tiers.get(tier) if isinstance(tiers, Mapping) else None
    return entry if isinstance(entry, Mapping) else None


def resolve_model_route(
    *,
    user_message: str,
    current_model: str,
    current_runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    explicit_override: bool = False,
    resolver: Optional[RouteRuntimeResolver] = None,
) -> Optional[RoutingDecision]:
    """Resolve the effective route for one main-agent turn.

    Returns ``None`` when routing is disabled, an explicit caller/session/job
    override must win, the chosen tier is incomplete, or provider resolution
    fails.  Callers should then continue with the already-resolved primary
    runtime.
    """
    if explicit_override or not routing_enabled(config):
        return None
    tier, reason, confidence, source = choose_tier(user_message, config)
    entry = _tier_entry(config, tier)
    if entry is None:
        logger.warning("model_routing: tier %r is not configured; using current model", tier)
        return None

    target_model = str(entry.get("model") or "").strip()
    target_provider = str(entry.get("provider") or "").strip().lower()
    target_base_url = str(entry.get("base_url") or "").strip() or None
    if not target_model or not target_provider:
        logger.warning("model_routing: tier %r missing provider/model; using current model", tier)
        return None

    if resolver is None:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        resolver = resolve_runtime_provider

    try:
        runtime = dict(
            resolver(
                requested=target_provider,
                target_model=target_model,
                explicit_base_url=target_base_url,
            )
        )
    except Exception as exc:  # noqa: BLE001 - fail soft; user turn must continue.
        logger.warning(
            "model_routing: provider resolution failed for tier=%s provider=%s model=%s: %s; using current model",
            tier,
            target_provider,
            target_model,
            exc,
        )
        return None

    if entry.get("api_mode"):
        runtime["api_mode"] = entry.get("api_mode")
    if entry.get("api_key"):
        runtime["api_key"] = entry.get("api_key")

    # Preserve max_tokens from the already resolved session/runtime unless the
    # tier explicitly overrides it.
    if "max_tokens" in current_runtime and "max_tokens" not in runtime:
        runtime["max_tokens"] = current_runtime.get("max_tokens")
    if entry.get("max_tokens") is not None:
        runtime["max_tokens"] = entry.get("max_tokens")

    provider = str(runtime.get("provider") or target_provider).strip().lower()
    resolved_model = str(runtime.get("model") or target_model).strip() or target_model
    runtime.pop("model", None)
    decision = RoutingDecision(
        tier=tier,
        provider=provider,
        model=resolved_model,
        runtime=runtime,
        reason=reason,
        confidence=confidence,
        source=source,
    )
    logger.info(
        "model_routing: routed to %s tier: %s/%s (source=%s confidence=%.2f reason=%s)",
        decision.tier,
        decision.provider,
        decision.model,
        decision.source,
        decision.confidence,
        decision.reason,
    )
    return decision


def apply_route_to_turn(
    *,
    route: MutableMapping[str, Any],
    user_message: str,
    config: Mapping[str, Any],
    explicit_override: bool = False,
    resolver: Optional[RouteRuntimeResolver] = None,
) -> MutableMapping[str, Any]:
    """Mutate and return a CLI/gateway turn-route dict when routing applies."""
    decision = resolve_model_route(
        user_message=user_message,
        current_model=str(route.get("model") or ""),
        current_runtime=route.get("runtime") or {},
        config=config,
        explicit_override=explicit_override,
        resolver=resolver,
    )
    if decision is None:
        return route
    route["model"] = decision.model
    route["runtime"] = decision.runtime
    route["signature"] = decision.signature
    route["routing_decision"] = {
        "tier": decision.tier,
        "provider": decision.provider,
        "model": decision.model,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "source": decision.source,
    }
    return route
