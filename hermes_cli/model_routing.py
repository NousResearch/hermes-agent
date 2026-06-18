"""Provider/model tier selection for Hermes declarative routing.

This module turns the existing provider-neutral routing config into a concrete
model/provider choice for a turn.  It is intentionally pure and conservative:
no selection happens unless ``routing.model_selection.enabled`` is true.
"""

from __future__ import annotations

from typing import Any
import re


DEFAULT_REASONING_TIER_MAP = {
    "none": "balanced",
    "think": "balanced",
    "megathink": "deep",
    "ultrathink": "deep",
}

CUSTODIAL_SCORE_HITS = {
    "citadel-trading",
    "destructive-production",
    "financial-sensitive",
    "secrets-security",
    "email-approval-flow",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def load_config() -> dict[str, Any]:
    """Load Hermes config lazily so tests can monkeypatch this module seam."""
    try:
        from hermes_cli.config import load_config as _load_config

        loaded = _load_config() or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _pattern_matches(pattern: str, prompt: str) -> bool:
    try:
        return re.search(pattern, prompt, re.IGNORECASE) is not None
    except re.error:
        return False


def _any_match(patterns: list[Any], prompt: str) -> bool:
    return any(isinstance(pattern, str) and _pattern_matches(pattern, prompt) for pattern in patterns)


def _first_reasoning_match(rules: list[Any], prompt: str, default_tier: str) -> tuple[str, str]:
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        patterns = _as_list(rule.get("patterns"))
        if _any_match(patterns, prompt):
            tier = _text(rule.get("tier") or default_tier).strip().lower()
            reason = _text(rule.get("reason") or rule.get("name") or "matched")
            return tier or default_tier, reason
    return default_tier, "default"


def _score_delegation_rules(rules: list[Any], prompt: str) -> tuple[int, list[str]]:
    score = 0
    hits: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        patterns = _as_list(rule.get("patterns"))
        if _any_match(patterns, prompt):
            try:
                score += int(rule.get("score", 0) or 0)
            except Exception:
                pass
            hits.append(_text(rule.get("name") or "matched"))
    return score, hits


def _complexity_for_rules(delegation: dict[str, Any], prompt: str) -> tuple[int, list[str]]:
    try:
        complexity = int(delegation.get("default_complexity", 4) or 4)
    except Exception:
        complexity = 4
    hits: list[str] = []
    for rule in _as_list(delegation.get("complexity_rules")):
        if not isinstance(rule, dict):
            continue
        patterns = _as_list(rule.get("patterns"))
        if _any_match(patterns, prompt):
            try:
                complexity = int(rule.get("complexity", complexity) or complexity)
            except Exception:
                pass
            hits.append(_text(rule.get("name") or "matched"))
    return complexity, hits


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def select_model_route(user_message: str, config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the concrete model route for a user turn, or ``None``.

    The expected config shape is:

    ```yaml
    routing:
      enabled: true
      model_selection:
        enabled: true
        default_tier: balanced
        low_complexity_tier: fast
        custodial_tier: custodial_direct
        tier_map:
          none: balanced
          think: balanced
          megathink: deep
          ultrathink: deep
      tiers:
        balanced: {provider: opencode-go, model: deepseek-v4-pro}
    ```
    """
    cfg = config if isinstance(config, dict) else {}
    routing = _as_dict(cfg.get("routing"))
    if not _is_enabled(routing.get("enabled", True)):
        return None

    model_selection = _as_dict(routing.get("model_selection"))
    if not _is_enabled(model_selection.get("enabled", False)):
        return None

    tiers = _as_dict(routing.get("tiers"))
    if not tiers:
        return None

    prompt = _text(user_message)
    delegation = _as_dict(routing.get("delegation"))
    reasoning = _as_dict(routing.get("reasoning"))

    reasoning_default = _text(reasoning.get("default_tier") or "none").strip().lower() or "none"
    reasoning_tier, reasoning_reason = _first_reasoning_match(
        _as_list(reasoning.get("rules")),
        prompt,
        reasoning_default,
    )

    score, score_hits = _score_delegation_rules(_as_list(delegation.get("score_rules")), prompt)
    complexity, complexity_hits = _complexity_for_rules(delegation, prompt)

    try:
        critical_threshold = int(delegation.get("critical_threshold", -5) or -5)
    except Exception:
        critical_threshold = -5

    custodial_tier = _text(model_selection.get("custodial_tier") or "custodial_direct")
    low_complexity_tier = _text(model_selection.get("low_complexity_tier") or "fast")
    default_tier = _text(model_selection.get("default_tier") or "balanced")

    tier_map = dict(DEFAULT_REASONING_TIER_MAP)
    for key, value in _as_dict(model_selection.get("tier_map")).items():
        if isinstance(key, str) and isinstance(value, str):
            tier_map[key.strip().lower()] = value.strip()

    custodial_hits = sorted(set(score_hits).intersection(CUSTODIAL_SCORE_HITS))
    if score <= critical_threshold or custodial_hits:
        selected_tier = custodial_tier
        selected_reason = "custodial-score" if score <= critical_threshold else "custodial-hit"
    elif reasoning_reason == "default" and complexity <= 2 and low_complexity_tier in tiers:
        selected_tier = low_complexity_tier
        selected_reason = "low-complexity"
    else:
        selected_tier = tier_map.get(reasoning_tier, default_tier) or default_tier
        selected_reason = f"reasoning:{reasoning_tier}"

    tier_cfg = tiers.get(selected_tier) if isinstance(tiers.get(selected_tier), dict) else None
    if not tier_cfg:
        tier_cfg = tiers.get(default_tier) if isinstance(tiers.get(default_tier), dict) else None
        selected_tier = default_tier
        selected_reason = "fallback-default-tier"
    if not tier_cfg:
        return None

    provider = _text(tier_cfg.get("provider")).strip()
    model = _text(tier_cfg.get("model")).strip()
    if not provider or not model:
        return None

    return {
        "tier": selected_tier,
        "provider": provider,
        "model": model,
        "reason": selected_reason,
        "reasoning_tier": reasoning_tier,
        "reasoning_reason": reasoning_reason,
        "score": score,
        "score_hits": score_hits,
        "custodial_hits": custodial_hits,
        "complexity": complexity,
        "complexity_hits": complexity_hits,
        "allow_fallback": tier_cfg.get("allow_fallback", True) is not False,
    }


def resolve_turn_model_route(
    user_message: str,
    current_model: str,
    current_runtime: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Resolve a per-turn model route and runtime credentials.

    Returns ``(model, runtime, route_metadata)``. If routing is disabled, no
    tier matches, or a non-custodial routed provider cannot be resolved, the
    current model/runtime are returned with ``route_metadata`` set to ``None``.
    A custodial tier with ``allow_fallback: false`` raises provider-resolution
    errors so the caller fails closed instead of silently downgrading.
    """
    runtime = dict(current_runtime or {})
    route = select_model_route(user_message, config if config is not None else load_config())
    if route is None:
        return current_model, runtime, None

    provider = route["provider"]
    model = route["model"]
    if provider == runtime.get("provider"):
        return model, runtime, route

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        resolved = resolve_runtime_provider(requested=provider)
    except Exception:
        if route.get("allow_fallback") is False:
            raise
        return current_model, runtime, None

    routed_runtime: dict[str, Any] = {
        "api_key": resolved.get("api_key"),
        "base_url": resolved.get("base_url"),
        "provider": resolved.get("provider", provider),
        "api_mode": resolved.get("api_mode"),
        "command": resolved.get("command"),
        "args": list(resolved.get("args") or []),
        "credential_pool": resolved.get("credential_pool"),
    }
    if "max_tokens" in runtime:
        routed_runtime["max_tokens"] = runtime.get("max_tokens")
    provider_overrides = resolved.get("request_overrides")
    if isinstance(provider_overrides, dict) and provider_overrides:
        route = {**route, "request_overrides": dict(provider_overrides)}
    return model, routed_runtime, route
