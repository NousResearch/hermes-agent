"""Resolve the primary auxiliary request before provider recovery is planned."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.auxiliary_client import _PreparedAuxRequest


def _codex_replay_messages(
    messages: list[dict[str, Any]], renamed: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    """Separate system instructions and alias tool history for this request's schemas."""
    instructions = "You are a helpful assistant."
    replay_messages: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content") or ""
        if msg.get("role", "user") == "system":
            instructions = content if isinstance(content, str) else str(content)
            continue
        if renamed and msg.get("tool_calls"):
            msg = {**msg, "tool_calls": [
                {**tc, "function": {**tc["function"], "name": renamed[tc["function"]["name"]]}}
                if isinstance(tc, dict) and (tc.get("function") or {}).get("name") in renamed else tc
                for tc in msg["tool_calls"]
            ]}
        replay_messages.append(msg)
    return instructions, replay_messages


def _prepare_aux_request(
    task: Optional[str], *, provider: Optional[str], model: Optional[str], base_url: Optional[str],
    api_key: Optional[str], main_runtime: dict[str, Any], messages: list,
    temperature: Optional[float], max_tokens: Optional[int], tools: Optional[list],
    timeout: Optional[float], extra_body: Optional[dict], reasoning_config: Optional[dict],
    extra_headers: Optional[dict[str, str]], api_mode: Optional[str],
    route_info: Optional[dict[str, str]], async_mode: bool,
) -> _PreparedAuxRequest:
    """Resolve route/client once, publish it and build the primary request kwargs.

    Recovery keeps the caller's original bounded messages in `_plan_aux_call`;
    replacing this primary request never expands a different provider's input.
    """
    from agent import auxiliary_client as aux
    from agent.context_compressor_cache import select_summary_input

    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = aux._resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    if api_mode:
        resolved_api_mode = api_mode
    effective_extra_body = aux._get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    client, final_model, resolved_provider, effective_provider = aux._resolve_call_client(
        task, provider=provider, model=model, base_url=base_url, api_key=api_key,
        resolved_provider=resolved_provider, resolved_model=resolved_model,
        resolved_base_url=resolved_base_url, resolved_api_key=resolved_api_key,
        resolved_api_mode=resolved_api_mode, main_runtime=main_runtime, async_mode=async_mode,
    )
    effective_timeout = aux._effective_aux_timeout(task, timeout)
    responses_client = isinstance(client, (aux.CodexAuxiliaryClient, aux.AsyncCodexAuxiliaryClient))
    no_progress_timeout = aux._get_task_no_progress_timeout(task) if responses_client else None
    request_provider = effective_provider or resolved_provider
    if not async_mode:
        compression_config = aux._get_auxiliary_task_config(task) if task == "compression" else {}
        _, effective_extra_body = aux._compression_fast_lane_controls(
            task, actual_provider=request_provider, actual_model=final_model,
            requested_provider=provider, requested_model=model, route_config=compression_config,
            leak_guard_config=compression_config, max_tokens=max_tokens,
            extra_body=effective_extra_body,
        )
    aux._set_relay_auxiliary_route(request_provider, final_model, resolved_api_mode)
    aux._record_route_info(route_info, aux._fallback_provider_from_label(request_provider), final_model)
    if async_mode:
        base_info = str(getattr(client, "base_url", "") or "")
    else:
        base_info = str(getattr(client, "base_url", resolved_base_url) or "")
        if task:
            aux.logger.info("Auxiliary %s: using %s (%s)%s",
                            task, request_provider or "auto", final_model or "default",
                            f" at {base_info}" if base_info and "openrouter" not in base_info else "")
    messages, tools, reasoning_config = select_summary_input(
        task, request_provider, final_model, base_info or resolved_base_url or "",
        "codex_responses" if responses_client else (resolved_api_mode or "chat_completions"),
        messages, tools, reasoning_config,
    )
    kwargs = aux._build_call_kwargs(
        request_provider, final_model, messages, temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        reasoning_config=reasoning_config, base_url=base_info or resolved_base_url, task=task,
        no_progress_timeout=no_progress_timeout)
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)
    client_base = str(getattr(client, "base_url", "") or "")
    if aux._is_anthropic_compat_endpoint(request_provider, client_base):
        kwargs["messages"] = aux._convert_openai_images_to_anthropic(kwargs["messages"])
    return aux._PreparedAuxRequest(
        client, final_model, kwargs, resolved_provider, request_provider, resolved_model,
        resolved_base_url, resolved_api_key, resolved_api_mode, effective_timeout,
        effective_extra_body, base_info)
