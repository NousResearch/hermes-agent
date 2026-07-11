"""Provider-neutral runtime identity for primary and fallback targets.

``api_mode`` describes an HTTP/wire protocol. ``runtime`` describes who owns
the agent loop. Keeping them separate lets a target use Hermes' native loop,
Codex app-server, or the Claude Agent SDK without overloading provider names.
"""

from __future__ import annotations

from typing import Any, Mapping


HERMES_RUNTIME = "hermes"
CODEX_APP_SERVER_RUNTIME = "codex_app_server"
CLAUDE_AGENT_SDK_RUNTIME = "claude_agent_sdk"

VALID_AGENT_RUNTIMES = frozenset(
    {HERMES_RUNTIME, CODEX_APP_SERVER_RUNTIME, CLAUDE_AGENT_SDK_RUNTIME}
)


def attach_runtime_identity(
    resolved: Mapping[str, Any],
    *,
    route_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of a resolved provider target with ``runtime`` attached."""

    target = dict(resolved)
    target["runtime"] = resolve_runtime_identity(
        provider=str(target.get("provider") or ""),
        api_mode=str(target.get("api_mode") or ""),
        route_config=route_config,
    )
    return target


def resolve_runtime_identity(
    *,
    provider: str,
    api_mode: str,
    route_config: Mapping[str, Any] | None = None,
) -> str:
    """Return the agent-loop runtime for a resolved route.

    An explicit ``runtime`` is authoritative. Unknown explicit values fail
    closed rather than silently selecting a different execution boundary.
    """

    config = route_config or {}
    configured = str(config.get("runtime") or "").strip().lower()
    if configured in VALID_AGENT_RUNTIMES:
        return configured
    if configured:
        raise ValueError(f"Unknown agent runtime: {configured}")
    if api_mode == CODEX_APP_SERVER_RUNTIME:
        return CODEX_APP_SERVER_RUNTIME
    openai_runtime = str(config.get("openai_runtime") or "").strip().lower()
    if (
        provider.strip().lower() in {"openai", "openai-codex"}
        and openai_runtime == CODEX_APP_SERVER_RUNTIME
    ):
        return CODEX_APP_SERVER_RUNTIME
    return HERMES_RUNTIME


__all__ = [
    "CLAUDE_AGENT_SDK_RUNTIME",
    "CODEX_APP_SERVER_RUNTIME",
    "HERMES_RUNTIME",
    "VALID_AGENT_RUNTIMES",
    "attach_runtime_identity",
    "resolve_runtime_identity",
]
