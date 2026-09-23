"""Execution-bound managed MoA context across auxiliary send threads."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from agent.model_selection_store import get_receipt
from agent.model_selection_types import RoutingBlocked


@dataclass(frozen=True)
class _WireContext:
    _managed_routing_home: object
    _managed_routing_receipt_id: str
    requested_provider: str
    base_url: str


_current: ContextVar[_WireContext | None] = ContextVar("managed_aux_wire", default=None)


@contextmanager
def managed_aux_wire_scope(home, receipt_id):
    context = None
    if home and receipt_id:
        receipt = get_receipt(home, receipt_id)
        if receipt is None:
            raise RoutingBlocked("stale_or_revoked_decision", "auxiliary receipt is missing")
        route = receipt["selected"]
        context = _WireContext(home, receipt_id, route["provider"], route.get("endpoint", ""))
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)


def enforce_aux_chat_wire(client, kwargs):
    context = _current.get()
    if context is not None:
        from agent.auxiliary_client import AnthropicAuxiliaryClient, CodexAuxiliaryClient
        from agent.managed_route_wire import enforce_chat_wire
        # Native shims have a later final-wire check after their conversion.
        wire_format = None if isinstance(client, (AnthropicAuxiliaryClient, CodexAuxiliaryClient)) else "chat_completions"
        enforce_chat_wire(context, client, kwargs, wire_format=wire_format)


def enforce_aux_native_wire(client, kwargs, api_mode):
    context = _current.get()
    if context is not None:
        from agent.managed_route_wire import enforce_anthropic_wire, enforce_responses_wire
        validators = {"anthropic_messages": enforce_anthropic_wire, "codex_responses": enforce_responses_wire}
        if api_mode not in validators:
            raise RoutingBlocked("unsupported_executor", "managed auxiliary transport has no final-wire validator")
        validators[api_mode](context, client, kwargs)


@contextmanager
def observe_moa_request(home, receipt_id):
    from agent.managed_route_health import observe_request
    with observe_request(home, receipt_id), managed_aux_wire_scope(home, receipt_id):
        yield


def observe_moa_stream(home, receipt_id, stream):
    from agent.managed_route_health import observe_stream
    with managed_aux_wire_scope(home, receipt_id):
        yield from observe_stream(home, receipt_id, stream)
