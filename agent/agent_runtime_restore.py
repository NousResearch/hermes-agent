"""Restore a ``_primary_runtime`` snapshot onto an agent (facade size cap).

Split from ``agent.agent_runtime_helpers``. ``_apply_primary_runtime_fields`` copies the
identity/transport fields and ``_build_anthropic_client_from_runtime`` rebuilds the native
Anthropic client; credential-pool and fallback bookkeeping stay with the callers.
"""

from __future__ import annotations

from typing import Any, Dict

from hermes_cli.timeouts import get_provider_request_timeout


def _apply_primary_runtime_fields(agent, rt: Dict[str, Any]) -> None:
    """Copy the identity/transport fields of a ``_primary_runtime`` snapshot onto ``agent``
    (shared by transport recovery and turn-start restore; the caller rebuilds the client)."""
    agent.model = rt["model"]
    agent.provider = rt["provider"]
    agent.requested_provider = rt.get("requested_provider", agent.provider)
    agent.base_url = rt["base_url"]           # setter updates _base_url_lower
    from hermes_cli.providers import is_actual_route
    agent.api_mode = "chat_completions" if is_actual_route(agent.provider, agent.base_url) else rt["api_mode"]
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    agent.api_key = rt["api_key"]
    agent._reasoning_echo_flag = rt.get("reasoning_echo_flag", False)
    agent.request_overrides = dict(rt.get("request_overrides") or {})
    agent.capabilities = dict(rt.get("capabilities") or {})
    agent._client_kwargs = dict(rt["client_kwargs"])


def _build_anthropic_client_from_runtime(agent, rt: Dict[str, Any]) -> None:
    """Rebuild the native Anthropic client from a ``_primary_runtime`` snapshot."""
    from agent.anthropic_adapter import build_anthropic_client
    agent._anthropic_api_key = rt["anthropic_api_key"]
    agent._anthropic_base_url = rt["anthropic_base_url"]
    agent._anthropic_client = build_anthropic_client(
        rt["anthropic_api_key"], rt["anthropic_base_url"],
        timeout=get_provider_request_timeout(agent.provider, agent.model),
        force_oauth=bool(rt["is_anthropic_oauth"]),
    )
    agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
    agent.client = None
