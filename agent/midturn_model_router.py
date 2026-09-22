"""Request-local routing after a successfully completed tool round.

The helper deliberately supports only model/effort changes within the active runtime.
``AIAgent.switch_model()`` is intentionally persistent and invalidates the cached system
prompt, so it is not safe inside a prompt-cached tool loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from hermes_cli.model_router import TurnRoute, resolve_midturn_route


def _completed_tool_outcome(messages: Any) -> str:
    """Return only the contiguous completed tool-result tail in chronological order."""
    if not isinstance(messages, list):
        return ""
    outcomes: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            break
        content = message.get("content")
        if isinstance(content, str) and content:
            outcomes.append(content)
    return "\n".join(reversed(outcomes))


def _active_runtime(agent: Any) -> dict[str, Any]:
    return {
        "provider": getattr(agent, "provider", ""),
        "requested_provider": getattr(agent, "requested_provider", ""),
        "base_url": getattr(agent, "base_url", ""),
        "api_mode": getattr(agent, "api_mode", ""),
        "api_key": getattr(agent, "api_key", ""),
    }


def _same_runtime(agent: Any, route: TurnRoute) -> bool:
    target = route.runtime
    active = _active_runtime(agent)
    return (
        str(target.get("provider") or "").strip().lower()
        == str(active["provider"] or "").strip().lower()
        and str(target.get("base_url") or "").rstrip("/")
        == str(active["base_url"] or "").rstrip("/")
        and str(target.get("api_mode") or "") == str(active["api_mode"] or "")
        # A model-only switch must use the already-live credential identity.  We never
        # transplant a route's credential or pool into the cached agent; accepting a
        # different one would quietly send the selected model on an unknown runtime.
        and target.get("api_key") == active["api_key"]
        and target.get("credential_pool") == getattr(agent, "_credential_pool", None)
    )


def _default_config_loader() -> Mapping[str, Any]:
    from hermes_cli.config import load_config_readonly

    return load_config_readonly() or {}


def _default_runtime_resolver(provider: str, model: str) -> Mapping[str, Any]:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(requested=provider, target_model=model)


def maybe_apply_midturn_route(
    agent: Any,
    *,
    messages: Any,
    original_user_message: Any,
    config_loader: Callable[[], Mapping[str, Any]] = _default_config_loader,
    runtime_resolver: Callable[[str, str], Mapping[str, Any]] = _default_runtime_resolver,
    controller: Callable[..., Mapping[str, Any]] | None = None,
) -> bool:
    """Apply one compatible post-tool route; uncertain or incompatible paths are no-ops."""
    outcome = _completed_tool_outcome(messages)
    if not outcome:
        return False
    try:
        kwargs: dict[str, Any] = {
            "config": config_loader(),
            "user_message": original_user_message,
            "tool_outcome": outcome,
            "base_model": str(getattr(agent, "model", "") or ""),
            "base_runtime": _active_runtime(agent),
            "runtime_resolver": runtime_resolver,
        }
        if controller is not None:
            kwargs["controller"] = controller
        route = resolve_midturn_route(**kwargs)
    except Exception:
        return False
    if route.decision in {"disabled", "unchanged"} or not _same_runtime(agent, route):
        return False

    if not hasattr(agent, "_midturn_route_restore"):
        agent._midturn_route_restore = {
            "model": getattr(agent, "model", ""),
            "reasoning_config": (
                dict(agent.reasoning_config)
                if isinstance(getattr(agent, "reasoning_config", None), dict)
                else getattr(agent, "reasoning_config", None)
            ),
            "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        }
    agent.model = route.model
    agent.reasoning_config = dict(route.reasoning_config) if route.reasoning_config else None
    agent.request_overrides = dict(route.request_overrides or {})
    return True


def restore_midturn_route(agent: Any) -> None:
    """Restore the request-local fields before a cached agent serves another turn."""
    restore = getattr(agent, "_midturn_route_restore", None)
    if not isinstance(restore, Mapping):
        return
    agent.model = restore["model"]
    reasoning = restore["reasoning_config"]
    agent.reasoning_config = dict(reasoning) if isinstance(reasoning, dict) else reasoning
    agent.request_overrides = dict(restore["request_overrides"] or {})
    delattr(agent, "_midturn_route_restore")
