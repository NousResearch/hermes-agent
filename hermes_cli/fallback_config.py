"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any


_GPT_55_MEDIUM = {"provider": "openai-codex", "model": "gpt-5.5", "reasoning_effort": "medium"}
_GPT_55_XHIGH = {"provider": "openai-codex", "model": "gpt-5.5", "reasoning_effort": "xhigh"}
_MINIMAX_M27_MEDIUM = {"provider": "minimax", "model": "MiniMax-M2.7", "reasoning_effort": "medium"}
_OPUS_48_XHIGH = {"provider": "anthropic", "model": "claude-opus-4.8", "reasoning_effort": "xhigh"}

_ROLE_ALIASES = {
    "goal": "orchestrator",
    "planner": "orchestrator",
    "architect": "orchestrator",
    "architecture": "orchestrator",
    "designer": "orchestrator",
    "design": "orchestrator",
    "orchestrator": "orchestrator",
    "builder": "builder",
    "implementer": "builder",
    "worker": "builder",
    "leaf": "builder",
    "review": "review",
    "reviewer": "review",
    "review_pass": "review",
    "optimization": "review",
    "optimization_pass": "review",
    "optimizer": "review",
    "refactor": "review",
    "refactoring": "review",
    "refactoring_pass": "review",
    "hardening": "review",
    "hardening_pass": "review",
    "hardener": "review",
    "final": "final_approval",
    "final_approval": "final_approval",
    "approval": "final_approval",
    "synthesis": "final_approval",
    "final_synthesis": "final_approval",
    "adversarial": "adversarial_review",
    "adversarial_review": "adversarial_review",
    "adverserial": "adversarial_review",
    "adverserial_review": "adversarial_review",
    "default": "default",
    "general": "default",
}

_DEFAULT_ROLE_ROUTES = {
    "default": {"primary": _GPT_55_MEDIUM, "fallbacks": []},
    "orchestrator": {"primary": _OPUS_48_XHIGH, "fallbacks": [_GPT_55_XHIGH]},
    "builder": {"primary": _MINIMAX_M27_MEDIUM, "fallbacks": [_GPT_55_MEDIUM]},
    "review": {"primary": _GPT_55_MEDIUM, "fallbacks": [_MINIMAX_M27_MEDIUM]},
    "final_approval": {"primary": _OPUS_48_XHIGH, "fallbacks": [_GPT_55_XHIGH]},
    "adversarial_review": {"primary": _GPT_55_XHIGH, "fallbacks": []},
}


_OPUS_48_MARKERS = (
    "opus-4.8",
    "opus4.8",
    "opus_4_8",
    "opus-4-8",
    "opus 4.8",
)


def is_opus_48_model(model: Any) -> bool:
    """Return True when a model slug/name identifies the disallowed Opus 4.8."""

    normalized = str(model or "").strip().lower().replace("_", "-")
    compact = normalized.replace("-", "").replace(" ", "")
    return any(marker in normalized for marker in _OPUS_48_MARKERS) or "opus48" in compact


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
        str(entry.get("reasoning_effort") or "").strip().lower(),
    )


def sanitize_fallback_chain(raw: Any) -> list[dict[str, Any]]:
    """Normalize, de-duplicate, and policy-filter fallback entries.

    Opus 4.8 is not allowed as an active default, runtime fallback, cron
    fallback, or delegated-child inherited fallback. Keeping the filter here
    makes stale config safe even when callers pass the legacy single-dict shape.
    """

    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for entry in _iter_fallback_entries(raw):
        if is_opus_48_model(entry.get("model")):
            continue
        identity = _entry_identity(entry)
        if identity in seen:
            continue
        seen.add(identity)
        chain.append(entry)
    return chain


def _role_key(role: str) -> str:
    return _ROLE_ALIASES.get(str(role or "default").strip().lower(), str(role or "default").strip().lower())


def role_has_route(role: str, config: dict[str, Any] | None = None) -> bool:
    """Return True when a role has an explicit/default primary or fallback route."""

    config = config or {}
    raw_key = str(role or "default").strip().lower()
    key = _role_key(role)
    routes = config.get("role_routes") or {}
    if isinstance(routes, dict) and (key in routes or raw_key in routes):
        return True
    role_overrides = config.get("role_fallbacks") or {}
    if isinstance(role_overrides, dict) and (key in role_overrides or raw_key in role_overrides):
        return True
    # Unknown roles still receive the built-in default route rather than
    # mixing an implicit default primary with legacy credential inheritance.
    return True


def role_primary_route(role: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the configured primary model/provider/reasoning route for a role.

    Defaults encode John's approved ladder:
    general=GPT-5.5 medium; goal/orchestrator/architect/designer=Opus 4.8;
    builders=MiniMax M2.7; review/optimization/hardening=GPT-5.5 medium;
    final approval/synthesis=Opus 4.8; adversarial review=GPT-5.5 xhigh.
    ``role_routes.<role>.primary`` in config may override these defaults.
    """

    config = config or {}
    key = _role_key(role)
    routes = config.get("role_routes") or {}
    if isinstance(routes, dict):
        override = routes.get(key) or routes.get(str(role or "").strip().lower())
        if isinstance(override, dict) and isinstance(override.get("primary"), dict):
            entries = _iter_fallback_entries(override.get("primary"))
            if entries:
                return dict(entries[0])
    default = (_DEFAULT_ROLE_ROUTES.get(key) or _DEFAULT_ROLE_ROUTES["default"])["primary"]
    return dict(default)


def role_fallback_chain(role: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the effective fallback chain for a role.

    ``role_routes.<role>.fallbacks`` wins, then legacy ``role_fallbacks.<role>``.
    Built-in defaults deliberately exclude the primary route: builders start on
    MiniMax and fail only to GPT-5.5 medium; review/optimization/hardening start
    on GPT-5.5 medium and fail to MiniMax; Opus roles fail only to GPT-5.5 xhigh.
    Adversarial review has no non-xhigh fallback by default.
    """

    config = config or {}
    key = _role_key(role)
    routes = config.get("role_routes") or {}
    if isinstance(routes, dict):
        override = routes.get(key) or routes.get(str(role or "").strip().lower())
        if isinstance(override, dict) and "fallbacks" in override:
            return sanitize_fallback_chain(override.get("fallbacks"))

    raw_key = str(role or "").strip().lower()
    role_overrides = config.get("role_fallbacks") or {}
    if isinstance(role_overrides, dict):
        if key in role_overrides:
            return sanitize_fallback_chain(role_overrides.get(key))
        if raw_key in role_overrides:
            return sanitize_fallback_chain(role_overrides.get(raw_key))

    route = _DEFAULT_ROLE_ROUTES.get(key)
    if route is not None:
        return sanitize_fallback_chain(route.get("fallbacks"))

    return get_fallback_chain(config)


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies and excludes disallowed
    Opus 4.8 routes.
    """

    config = config or {}
    return sanitize_fallback_chain([
        *(_iter_fallback_entries(config.get("fallback_providers"))),
        *(_iter_fallback_entries(config.get("fallback_model"))),
    ])
