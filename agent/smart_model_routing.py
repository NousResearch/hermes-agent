"""Conservative per-turn smart model routing.

This module is intentionally pure/small: it decides whether a user turn is
safe to route to a configured cheaper model and, when requested, resolves the
runtime for that model through the existing provider-resolution stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class SmartRouteDecision:
    """Decision returned by :func:`decide_smart_route`."""

    should_route: bool
    reason: str
    model_config: dict[str, Any]


_CODE_OR_CONTEXT_MARKERS = (
    "```",
    "`",
    "@file",
    "@diff",
    "@code",
    "@repo",
    "@image",
    "@photo",
    "MEDIA:",
    "<image",
    "![",
)

_WORK_INTENT_PATTERNS = [
    # English
    r"\b(debug|fix|repair|implement|code|refactor|test|pytest|mypy|lint|build|deploy|docker|kubernetes|terraform|ssh|git|commit|pull request|pr|review|audit|research|search|browse|scrape|download|upload|install|configure|schedule|cron|database|sql|migration|delete|remove|drop|truncate|send|email|post|publish)\b",
    # Russian stems/verbs. Keep broad and conservative: false negatives are
    # cheap; false positives could send real work to a weak model.
    r"\b(почин\w*|исправ\w*|рефактор\w*|код\w*|тест\w*|запусти\w*|собери\w*|задепло\w*|депло\w*|проверь\w*|найди\w*|исслед\w*|скача\w*|загру\w*|установ\w*|настро\w*|удали\w*|отправ\w*|опубли\w*|коммит\w*|пулл\w*|реквест\w*|аудит\w*)\b",
]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _routing_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    cfg = _as_mapping(config)
    routing = cfg.get("smart_model_routing")
    return _as_mapping(routing)


def _cheap_model_config(routing: Mapping[str, Any]) -> dict[str, Any]:
    raw = routing.get("cheap_model") or routing.get("model")
    if isinstance(raw, str):
        return {"model": raw}
    if isinstance(raw, Mapping):
        return dict(raw)
    # Also accept flat keys for quick hand-edits:
    # smart_model_routing: {provider: openai-codex, model: gpt-...}
    if routing.get("model") or routing.get("provider"):
        return {
            "model": routing.get("model") or "",
            "provider": routing.get("provider") or "",
            "base_url": routing.get("base_url") or "",
            "api_key": routing.get("api_key") or "",
        }
    return {}


def _word_count(text: str) -> int:
    return len(re.findall(r"[\wА-Яа-яЁё]+", text, flags=re.UNICODE))


def _history_has_context(history: Sequence[Mapping[str, Any]] | None) -> bool:
    if not history:
        return False
    return any(bool(_as_mapping(item).get("content")) for item in history)


def decide_smart_route(
    user_message: str,
    config: Mapping[str, Any] | None,
    *,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> SmartRouteDecision:
    """Decide whether ``user_message`` may use the cheap route.

    The classifier is deliberately conservative and deterministic.  It routes
    only short standalone plain-text turns with no obvious need for tools,
    repository context, current web facts, code edits, or external side effects.
    """

    routing = _routing_config(config)
    if not bool(routing.get("enabled", False)):
        return SmartRouteDecision(False, "disabled", {})

    model_cfg = _cheap_model_config(routing)
    if not str(model_cfg.get("model") or "").strip():
        return SmartRouteDecision(False, "missing_cheap_model", {})

    message = str(user_message or "").strip()
    if not message:
        return SmartRouteDecision(False, "empty_message", model_cfg)

    require_empty_history = bool(routing.get("require_empty_history", True))
    if require_empty_history and _history_has_context(history):
        return SmartRouteDecision(False, "has_history", model_cfg)

    if message.startswith("/"):
        return SmartRouteDecision(False, "slash_command", model_cfg)

    for marker in _CODE_OR_CONTEXT_MARKERS:
        if marker.lower() in message.lower():
            return SmartRouteDecision(False, "context_marker", model_cfg)

    try:
        max_chars = int(routing.get("max_simple_chars", 400))
    except (TypeError, ValueError):
        max_chars = 400
    try:
        max_words = int(routing.get("max_simple_words", 80))
    except (TypeError, ValueError):
        max_words = 80

    if len(message) > max_chars:
        return SmartRouteDecision(False, "too_long_chars", model_cfg)
    if _word_count(message) > max_words:
        return SmartRouteDecision(False, "too_many_words", model_cfg)

    lowered = message.lower()
    for pattern in _WORK_INTENT_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE | re.UNICODE):
            return SmartRouteDecision(False, "work_intent", model_cfg)

    return SmartRouteDecision(True, "simple_turn", model_cfg)


def _clean_runtime(runtime: Mapping[str, Any] | None) -> dict[str, Any]:
    runtime = _as_mapping(runtime)
    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }


def resolve_smart_model_route(
    *,
    primary_model: str,
    primary_runtime: Mapping[str, Any],
    user_message: str,
    config: Mapping[str, Any] | None,
    history: Sequence[Mapping[str, Any]] | None = None,
    runtime_resolver: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return a cheap turn route or ``None`` when primary route should be used."""

    decision = decide_smart_route(user_message, config, history=history)
    if not decision.should_route:
        return None

    cheap_cfg = decision.model_config
    cheap_model = str(cheap_cfg.get("model") or "").strip()
    if not cheap_model:
        return None

    resolver = runtime_resolver
    if resolver is None:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        resolver = resolve_runtime_provider

    requested = str(cheap_cfg.get("provider") or "").strip() or None
    explicit_base_url = str(cheap_cfg.get("base_url") or "").strip() or None
    explicit_api_key = str(cheap_cfg.get("api_key") or "").strip() or None

    try:
        resolved_runtime = resolver(
            requested=requested,
            explicit_base_url=explicit_base_url,
            explicit_api_key=explicit_api_key,
            target_model=cheap_model,
        )
    except Exception:
        return None

    runtime = _clean_runtime(resolved_runtime)
    # Some provider resolvers expose requested_provider/source rather than a
    # final provider.  Keep the final provider explicit for route signatures.
    if not runtime.get("provider"):
        runtime["provider"] = requested or _as_mapping(primary_runtime).get("provider")
    if runtime.get("base_url") is None and explicit_base_url:
        runtime["base_url"] = explicit_base_url

    return {
        "model": cheap_model,
        "runtime": runtime,
        "reason": decision.reason,
        "primary_model": primary_model,
    }
