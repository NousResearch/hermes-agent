"""Deterministic, session-stable model routing for Hermes.

The router intentionally stays heuristic-only: it does not call an LLM to pick
an LLM. It selects a model for a new session, then keeps that session pinned so
prompt-cache and frozen-system-prompt invariants remain intact.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple


DEFAULT_QUICK_KEYWORDS = (
    "quick",
    "quickly",
    "fast answer",
    "short answer",
    "quick response",
    "recon",
    "reconnaissance",
    "scan",
    "inspect",
    "look up",
    "lookup",
    "find",
    "read",
    "summarize",
    "summarise",
    "rename",
    "format",
    "small change",
    "tiny change",
    "low level",
    "low-level",
    "simple",
)

DEFAULT_COMPLEX_KEYWORDS = (
    "architecture",
    "architectural",
    "system design",
    "design doc",
    "planning",
    "plan",
    "strategy",
    "roadmap",
    "advisor",
    "advise",
    "tradeoff",
    "trade-off",
    "deep dive",
    "deep research",
    "complex",
    "high stakes",
    "migration plan",
    "security review",
    "audit",
    "final qc",
    "synthesis",
    "synthesize",
)

EXPLICIT_TIER_ALIASES = {
    "luna": "quick",
    "quick": "quick",
    "fast": "quick",
    "terra": "standard",
    "standard": "standard",
    "normal": "standard",
    "sol": "complex",
    "complex": "complex",
    "architect": "complex",
    "advisor": "complex",
}

_STANDARD_ACTION_WORDS = (
    "implement",
    "build",
    "fix",
    "debug",
    "refactor",
    "change",
    "edit",
    "update",
    "write",
    "create",
    "test",
    "verify",
)


def _extract_text(value: Any) -> str:
    """Best-effort text extraction from chat strings or OpenAI content parts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in ("text", "content", "message", "prompt"):
            if key in value:
                parts.append(_extract_text(value.get(key)))
        return "\n".join(p for p in parts if p)
    if isinstance(value, (list, tuple)):
        return "\n".join(_extract_text(item) for item in value if item is not None)
    return str(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _router_config(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    router = config.get("model_router") or {}
    return router if isinstance(router, Mapping) else {}


def _as_keywords(router_cfg: Mapping[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = router_cfg.get(key)
    if isinstance(raw, str):
        value = raw.strip().lower()
        return (value,) if value else default
    if isinstance(raw, (list, tuple)):
        values = tuple(str(item).lower() for item in raw if str(item).strip())
        return values or default
    return default


def _explicit_tier(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    # Copyable overrides:
    #   /sol design this
    #   route:sol design this
    #   model:luna answer fast
    match = re.search(r"(?:^|\s)(?:route|model)\s*:\s*(luna|terra|sol|quick|standard|complex|advisor)\b", lowered)
    if not match:
        match = re.search(r"^/(luna|terra|sol|quick|standard|complex)\b", lowered)
    if match:
        return EXPLICIT_TIER_ALIASES.get(match.group(1))
    return None


def classify_model_router_tier(user_message: Any, router_cfg: Optional[Mapping[str, Any]] = None) -> str:
    """Classify a user turn into quick, standard, or complex."""
    cfg = router_cfg if isinstance(router_cfg, Mapping) else {}
    default_tier = str(cfg.get("default") or "standard").strip().lower()
    default_tier = EXPLICIT_TIER_ALIASES.get(default_tier, default_tier)
    if default_tier not in {"quick", "standard", "complex"}:
        default_tier = "standard"

    text = _extract_text(user_message)
    lowered = text.lower()
    stripped = lowered.strip()
    if not stripped:
        return default_tier

    explicit = _explicit_tier(stripped)
    if explicit:
        return explicit

    complex_keywords = _as_keywords(cfg, "complex_keywords", DEFAULT_COMPLEX_KEYWORDS)
    quick_keywords = _as_keywords(cfg, "quick_keywords", DEFAULT_QUICK_KEYWORDS)

    if any(keyword and keyword in lowered for keyword in complex_keywords):
        return "complex"

    has_code_block = "```" in text
    has_standard_action = any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in _STANDARD_ACTION_WORDS)

    if any(keyword and keyword in lowered for keyword in quick_keywords):
        # A quick keyword in a long coding task should not demote real work.
        if len(stripped) < 800 and not has_code_block:
            return "quick"

    # Short factual questions and one-line lookups belong on the cheap/fast tier.
    if len(stripped) <= 180 and "?" in stripped and not has_standard_action:
        return "quick"

    return default_tier


def _route_model(router_cfg: Mapping[str, Any], tier: str) -> str:
    routes = router_cfg.get("routes") or {}
    if not isinstance(routes, Mapping):
        return ""
    entry = routes.get(tier) or routes.get(EXPLICIT_TIER_ALIASES.get(tier, tier))
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, Mapping):
        return str(entry.get("model") or "").strip()
    return ""


def select_model_for_turn(
    user_message: Any,
    base_model: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Return (effective_model, router_tier) for a turn.

    If ``model_router.enabled`` is absent/false or the selected route has no
    model, the base model is returned and tier is ``None``.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}

    router_cfg = _router_config(config)
    if not _truthy(router_cfg.get("enabled")):
        return base_model, None

    tier = classify_model_router_tier(user_message, router_cfg)
    model = _route_model(router_cfg, tier) or base_model
    return model, tier


def select_model_for_session_turn(
    user_message: Any,
    base_model: str,
    pinned_model: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Select a model once, then preserve it for a session's later turns.

    A previous effective session model wins while routing is enabled. Keeping a
    model stable avoids rebuilding an agent (and its frozen system prompt/tool
    schemas) merely because a follow-up message contains a different keyword.
    ``pinned_model`` may be a configured route or an explicit session model;
    in the latter case the tier is intentionally unknown (``None``).
    """
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}

    router_cfg = _router_config(config)
    if not _truthy(router_cfg.get("enabled")):
        return base_model, None

    if isinstance(pinned_model, str) and pinned_model.strip():
        pinned = pinned_model.strip()
        for tier in ("quick", "standard", "complex"):
            if _route_model(router_cfg, tier) == pinned:
                return pinned, tier
        return pinned, None

    return select_model_for_turn(user_message, base_model, config)
