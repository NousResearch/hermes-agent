"""Validate managed payloads at the final protocol-specific SDK send boundary."""
from __future__ import annotations

import json

from agent.managed_route_budget import enforce_input_budget
from agent.managed_route_runtime import enforce_worker_route
from agent.model_selection_types import RoutingBlocked
from agent.model_selection_store import append_outcome, content_hash


def enforce_supported_wire(agent) -> None:
    if not getattr(agent, "_managed_routing_receipt_id", None):
        return
    if agent.api_mode == "bedrock_converse" or agent.provider == "moa":
        raise RoutingBlocked("unsupported_executor", "managed routes require a concrete validated wire transport")


def enforce_chat_wire(agent, client, kwargs: dict, *, wire_format: str | None = "chat_completions") -> None:
    receipt_id = getattr(agent, "_managed_routing_receipt_id", None)
    if not receipt_id:
        return
    home = getattr(agent, "_managed_routing_home", None)
    if not home:
        raise RoutingBlocked("stale_or_revoked_decision", "missing routing origin home")
    # The SDK merges extra_body over its serialized arguments.
    extra = kwargs.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise RoutingBlocked("schema_invalid", "managed extra_body must be an object")
    payload = {**kwargs, **extra}
    reasoning = payload.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        raise RoutingBlocked("schema_invalid", "managed reasoning must be an object")
    effort = payload.get("reasoning_effort", reasoning.get("effort"))
    if reasoning.get("enabled") is False:
        effort = "none"
    endpoint = str(getattr(client, "base_url", agent.base_url))
    # OpenAI's SDK appends a slash to its base URL before joining resource paths.
    if endpoint == str(agent.base_url).rstrip("/") + "/":
        endpoint = agent.base_url
    enforce_worker_route(
        home, receipt_id,
        actual_provider=getattr(agent, "requested_provider", None) or agent.provider,
        actual_model=payload.get("model", ""),
        actual_endpoint=endpoint,
        actual_reasoning=effort, record_outcome=False,
    )
    enforce_input_budget(
        home, receipt_id, payload.get("messages"), payload.get("tools"),
        max_tokens=payload.get("max_completion_tokens", payload.get("max_tokens")),
    )
    if wire_format is not None:
        append_outcome(home, receipt_id, "routing_wire_validated", {
            "format": wire_format, "model": payload.get("model"), "reasoning": effort,
            "endpoint_hash": content_hash({"endpoint": endpoint}),
        })


def enforce_anthropic_wire(agent, client, kwargs: dict) -> None:
    if not getattr(agent, "_managed_routing_receipt_id", None):
        return
    from agent.anthropic_adapter import THINKING_BUDGET

    payload = {**kwargs, **(kwargs.get("extra_body") or {})}
    thinking = payload.get("thinking") or {}
    effort = None
    if thinking.get("type") == "adaptive":
        effort = (payload.get("output_config") or {}).get("effort")
    elif thinking.get("type") == "enabled":
        effort = next((name for name, budget in THINKING_BUDGET.items()
                       if budget == thinking.get("budget_tokens")), None)
    elif thinking.get("type") == "disabled":
        effort = "none"
    messages = list(payload.get("messages") or [])
    if payload.get("system"):
        messages.insert(0, {"role": "system", "content": payload["system"]})
    messages = [_anthropic_budget_message(message) for message in messages]
    enforce_chat_wire(agent, client, {**payload, "extra_body": {},
                                     "messages": messages, "reasoning_effort": effort}, wire_format="anthropic_messages")


def _anthropic_budget_message(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    pending = list(content)
    while pending:
        part = pending.pop()
        if not isinstance(part, dict) or part.get("type") not in {
            "text", "thinking", "redacted_thinking", "tool_use", "tool_result",
        }:
            raise RoutingBlocked("missing_input_estimate", "native multimodal input needs verified accounting")
        if part.get("type") == "tool_result" and isinstance(part.get("content"), list):
            pending.extend(part["content"])
    # Native tool/thinking blocks are serialized text, not image/audio tokens.
    # Count their complete framing and fields instead of dropping non-text blocks.
    return {**message, "content": json.dumps(content, ensure_ascii=False, allow_nan=False)}


def enforce_responses_wire(agent, client, kwargs: dict) -> None:
    if not getattr(agent, "_managed_routing_receipt_id", None):
        return
    payload = {**kwargs, **(kwargs.get("extra_body") or {})}
    messages = payload.get("input")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, list):
        messages = list(messages)
    else:
        raise RoutingBlocked("missing_input_estimate", "final Responses input is unavailable")
    if payload.get("instructions"):
        messages.insert(0, {"role": "system", "content": payload["instructions"]})
    enforce_chat_wire(agent, client, {**payload, "extra_body": {}, "messages": messages,
                                     "max_tokens": payload.get("max_output_tokens")}, wire_format="codex_responses")
