"""Exact current-turn routing for opt-in plugin LLM calls."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from typing import Any, Dict, NamedTuple, Optional

import agent.auxiliary_client as auxiliary_client


class TurnBoundInvocationError(RuntimeError):
    """A plugin asked for an active turn invocation that cannot be used exactly."""


class _TurnInvocationRoute(NamedTuple):
    provider: str
    model: str
    base_url: str
    api_key: Any
    api_mode: str
    client: Any


# Keep the live agent out of auxiliary-client cache keys. Capturing its route lazily
# lets a post-call hook inherit a fallback activated after turn setup.
_TURN_INVOCATION_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("plugin_turn_invocation", default=None)
)


@contextlib.contextmanager
def scoped_turn_invocation(agent: Any = None):
    """Open a reset-safe plugin invocation scope for one conversation turn.

    The scope starts fail-closed when *agent* is omitted; turn setup fills it only
    after primary-runtime restoration has settled the active route.
    """
    lease = {"active": True}
    binding = {"lease": lease, "agent": agent}
    token = _TURN_INVOCATION_CONTEXT.set(binding)
    try:
        yield
    finally:
        # Context copies used by bounded hook workers share this dict. Revoke it
        # before resetting the parent so a timed-out worker cannot outlive the turn.
        lease["active"] = False
        _TURN_INVOCATION_CONTEXT.reset(token)


def set_turn_invocation_agent(agent: Any) -> bool:
    """Bind *agent* inside an open turn scope; return ``False`` outside one."""
    binding = _TURN_INVOCATION_CONTEXT.get()
    lease = binding.get("lease") if isinstance(binding, dict) else None
    if not isinstance(lease, dict) or not lease.get("active"):
        return False
    # Replace the context-local binding instead of mutating it: a worker that
    # copied the empty scope before publication must remain unbound. Both copies
    # still share the revocable lease, so exit invalidates post-publication workers.
    _TURN_INVOCATION_CONTEXT.set({"lease": lease, "agent": agent})
    return True


_SUPPORTED_API_MODES = frozenset({
    "chat_completions", "anthropic_messages", "codex_responses",
})


def _capture_turn_invocation() -> _TurnInvocationRoute:
    """Snapshot the live turn route at call time, including its exact client."""
    binding = _TURN_INVOCATION_CONTEXT.get()
    lease = binding.get("lease") if isinstance(binding, dict) else None
    agent = (
        binding.get("agent")
        if isinstance(binding, dict) and isinstance(lease, dict) and lease.get("active")
        else None
    )
    if agent is None:
        raise TurnBoundInvocationError("no active turn invocation")
    api_mode = str(getattr(agent, "api_mode", "") or "").strip()
    client = (
        getattr(agent, "_anthropic_client", None)
        if api_mode == "anthropic_messages"
        else getattr(agent, "client", None)
    )
    route = _TurnInvocationRoute(
        provider=str(getattr(agent, "provider", "") or "").strip(),
        model=str(getattr(agent, "model", "") or "").strip(),
        base_url=str(getattr(agent, "base_url", "") or "").strip(),
        api_key=getattr(agent, "api_key", ""),
        api_mode=api_mode,
        client=client,
    )
    if (
        route.provider.lower() in {"", "auto", "moa"}
        or not route.model
        or not route.base_url
        or route.client is None
    ):
        raise TurnBoundInvocationError("active turn route is incomplete")
    if route.api_mode not in _SUPPORTED_API_MODES:
        raise TurnBoundInvocationError(
            f"active turn API mode {route.api_mode or '<unset>'!r} "
            "does not support inherited calls"
        )
    return route


def _request_client(route: _TurnInvocationRoute) -> Any:
    """Apply only the protocol adapter required by the captured route."""
    if route.api_mode == "anthropic_messages":
        is_oauth = isinstance(route.api_key, str) and route.api_key.startswith("sk-ant-oat")
        return auxiliary_client.AnthropicAuxiliaryClient(
            route.client, route.model, route.api_key, route.base_url, is_oauth=is_oauth
        )
    if route.api_mode == "codex_responses":
        return auxiliary_client.CodexAuxiliaryClient(route.client, route.model)
    return route.client


def _call_turn_bound_llm(
    route: _TurnInvocationRoute, *, messages: list, temperature: Optional[float],
    max_tokens: Optional[int], timeout: Optional[float], extra_body: Optional[dict],
    route_info: Optional[Dict[str, str]],
) -> Any:
    """Issue one request on *route*, with no auxiliary retry or fallback ladder."""
    request_client = _request_client(route)
    kwargs = auxiliary_client._build_call_kwargs(
        route.provider, route.model, messages, temperature=temperature,
        max_tokens=max_tokens,
        timeout=(auxiliary_client._DEFAULT_AUX_TIMEOUT if timeout is None else timeout),
        extra_body=dict(extra_body or {}), base_url=route.base_url, task=None,
    )
    auxiliary_client._record_route_info(route_info, route.provider, route.model)
    auxiliary_client._set_relay_auxiliary_route(
        route.provider, route.model, route.api_mode
    )
    force_stream = auxiliary_client._provider_requires_stream(
        route.provider, route.base_url
    )

    def _create(request: Dict[str, Any]) -> Any:
        if force_stream:
            return auxiliary_client._create_with_progress_once(
                request_client, request, force_stream=True
            )
        return request_client.chat.completions.create(**request)

    response = auxiliary_client._relay_sync_completion(
        request_client, kwargs, provider=route.provider, api_mode=route.api_mode,
        create=_create,
    )
    return auxiliary_client._validate_llm_response(
        response, provider=route.provider, base_url=route.base_url
    )


@auxiliary_client._relay_auxiliary_call
def call_turn_bound_llm(
    *, messages: list, temperature: Optional[float] = None,
    max_tokens: Optional[int] = None, timeout: Optional[float] = None,
    extra_body: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Use the current turn's exact provider/model/client once, or fail closed."""
    return _call_turn_bound_llm(
        _capture_turn_invocation(), messages=messages, temperature=temperature,
        max_tokens=max_tokens, timeout=timeout, extra_body=extra_body,
        route_info=route_info,
    )


@auxiliary_client._relay_auxiliary_call_async
async def async_call_turn_bound_llm(
    *, messages: list, temperature: Optional[float] = None,
    max_tokens: Optional[int] = None, timeout: Optional[float] = None,
    extra_body: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Async sibling of :func:`call_turn_bound_llm` with the same single-call contract."""
    route = _capture_turn_invocation()
    return await asyncio.to_thread(
        _call_turn_bound_llm, route, messages=messages, temperature=temperature,
        max_tokens=max_tokens, timeout=timeout, extra_body=extra_body,
        route_info=route_info,
    )
