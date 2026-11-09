"""Per-attempt request assembly for the conversation turn's retry loop: re-apply the
reasoning echo pad and prompt-cache decoration for the CURRENT provider (a fallback may
differ from the primary), build ``api_kwargs``, run the surrogate/ASCII chokepoints, Codex
preflight, OpenRouter cache bypass, provider per-request headers, the LLM request middleware,
the ``pre_api_request`` hook and the debug dump. Nothing here imports
``agent.conversation_loop`` at module level.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from agent.message_sanitization import sanitize_outbound_kwargs, strip_images_for_rejecting_model
from hermes_cli.observability.shared_metrics_efficiency import observe_request_tools
from utils import env_var_enabled

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ApiRequestBuild:
    """Always ``action == "fallthrough"``; the fields are the request-local values the caller
    rebinds for the attempt."""

    action: str
    api_messages: Any
    _moa_prepared_request: Any
    tools_for_api: Any
    api_kwargs: Any
    _original_api_kwargs: Any
    _llm_middleware_trace: Any


def _set_extra_header(api_kwargs: Any, key: str, value: str) -> None:
    """Copy-on-write header set (the dict may be shared with the transport's defaults)."""
    _xh = dict(api_kwargs.get("extra_headers") or {})
    _xh[key] = value
    api_kwargs["extra_headers"] = _xh


def _apply_user_turn_request_headers(agent: Any, api_kwargs: Any) -> None:
    """Provider per-request headers (turn attribution). Only the first payload built in a user turn (the
    first attempt of its first API call) is user-initiated, and it consumes the turn flag exactly once,
    whatever the route answers (or if its hook fails). Every later build of the turn (a 429/5xx retry, a
    fallback activation, a tool-loop follow-up) gets the route's default: the historical billing
    behaviour, kept on purpose."""
    if not getattr(agent, "_is_user_initiated_turn", False):
        return
    agent._is_user_initiated_turn = False
    from agent.provider_route_hooks import route_answer
    _request_headers = route_answer(
        "request_headers", provider=agent.provider or "", base_url=agent.base_url or "",
        model=agent.model or "", user_initiated=True, scope=agent)
    for _key, _value in (_request_headers or {}).items():
        _set_extra_header(api_kwargs, _key, _value)


def _fire_pre_api_request_hook(
    agent: Any, api_kwargs: Any, api_messages: Any, _llm_middleware_trace: Any, *, messages: Any,
    original_user_message: Any, approx_tokens: Any, total_chars: Any, retry_count: Any,
    api_call_count: Any, api_request_id: Any, api_start_time: Any, effective_task_id: Any,
    turn_id: Any,
) -> None:
    from agent.conversation_loop import _system_prompt_for_hooks

    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook as _invoke_hook
        if has_hook("pre_api_request"):
            request_messages = api_kwargs.get("messages")
            if not isinstance(request_messages, list):
                request_messages = api_kwargs.get("input")
            if not isinstance(request_messages, list):
                request_messages = api_messages
            # Shallow copies: plugins may retain the lists; deepcopy is costly.
            # ``request_messages``/``conversation_history`` are raw langfuse passthroughs.
            # Anthropic (``system``) and Responses/Codex (``instructions``) move the system
            # prompt out of messages; pass it for observability.
            _invoke_hook(
                "pre_api_request",
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                session_id=agent.session_id or "",
                user_message=original_user_message,
                conversation_history=list(messages),
                platform=agent.platform or "",
                model=agent.model,
                provider=agent.provider,
                base_url=agent.base_url,
                api_mode=agent.api_mode,
                api_call_count=api_call_count,
                retry_count=retry_count,
                request_messages=list(request_messages) if isinstance(request_messages, list) else [],
                system_prompt=_system_prompt_for_hooks(api_kwargs, request_messages),
                message_count=len(api_messages),
                tool_count=len(agent.tools or []),
                approx_input_tokens=approx_tokens,
                request_char_count=total_chars,
                max_tokens=agent.max_tokens,
                started_at=api_start_time,
                middleware_trace=list(_llm_middleware_trace),
                request=agent._api_request_payload_for_hook(api_kwargs),
            )
    except Exception:
        pass


def build_request_payload(
    agent: Any, *, api_messages: Any, _moa_prepared_request: Any, tools_for_api: Any,
    system_message: Any, api_call_count: Any, api_request_id: Any, effective_task_id: Any,
    turn_id: Any,
) -> ApiRequestBuild:
    """The provider payload through the LLM request middleware, with no turn hook fired: the
    turn loop's :func:`build_api_request` and off-turn side requests (``ctx.side_question``,
    which reports through ``pre/post_auxiliary_call``) share it."""
    from agent.conversation_loop import _redecorate_prompt_cache_for_provider

    agent._reset_stream_delivery_tracking()
    # Per-attempt first-chunk timestamp so a stale value never leaks into post_api_request.
    agent._last_api_first_chunk_at = None
    # api_messages was built for the primary; a fallback (DeepSeek / Kimi / MiMo) may
    # require reasoning_content — re-apply the echo-back pad (idempotent) and re-render
    # the prompt-cache decoration for the current provider.
    agent._reapply_reasoning_echo_for_provider(api_messages)
    api_messages, _moa_prepared_request, tools_for_api = (
        _redecorate_prompt_cache_for_provider(
            agent, api_messages, system_message=system_message, moa_prepared=_moa_prepared_request,
            tools_for_api=tools_for_api,
        )
    )
    # A model that rejected image content gets text only; history keeps the images.
    strip_images_for_rejecting_model(agent, api_messages)
    observe_request_tools(agent, tools_for_api)
    if tools_for_api == agent.tools:
        api_kwargs = agent._build_api_kwargs(api_messages)
    else:
        api_kwargs = agent._build_api_kwargs(api_messages, tools_for_api=tools_for_api)
    # Messages were scrubbed above; this walk covers the rest of the payload (tool descriptions,
    # extra_body, kwargs strings) — see sanitize_outbound_kwargs for the #50959 rationale.
    sanitize_outbound_kwargs(agent, api_kwargs)
    if agent.api_mode == "codex_responses":
        from agent.codex_responses_adapter import drops_replay_item_ids

        api_kwargs = agent._get_transport().preflight_kwargs(
            api_kwargs, allow_stream=False, drop_replay_item_ids=drops_replay_item_ids(agent.base_url),
            sanitize_harmony_tokens=agent._is_codex_backend(),
        )
    # OpenRouter caching replays identical responses, even empty ones; an empty-response
    # retry must bypass the cache.
    if agent._empty_content_retries > 0 and agent._is_openrouter_url():
        _set_extra_header(api_kwargs, "X-OpenRouter-Cache", "false")
    _apply_user_turn_request_headers(agent, api_kwargs)
    try:
        from hermes_cli.middleware import apply_llm_request_middleware

        _llm_request_mw = apply_llm_request_middleware(
            api_kwargs, task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
            session_id=agent.session_id or "", platform=agent.platform or "", model=agent.model,
            provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode,
            api_call_count=api_call_count,
        )
        api_kwargs = _llm_request_mw.payload
        _original_api_kwargs = _llm_request_mw.original_payload
        _llm_middleware_trace = _llm_request_mw.trace
    except Exception:
        logger.warning("LLM request middleware failed; sending the request unmodified", exc_info=True)
        _original_api_kwargs = dict(api_kwargs)
        _llm_middleware_trace = []
    return ApiRequestBuild(
        "fallthrough", api_messages, _moa_prepared_request, tools_for_api, api_kwargs,
        _original_api_kwargs, _llm_middleware_trace,
    )


def build_api_request(
    agent: Any, *, api_messages: Any, _moa_prepared_request: Any, tools_for_api: Any,
    system_message: Any, messages: Any, original_user_message: Any, approx_tokens: Any,
    total_chars: Any, retry_count: Any, api_call_count: Any, api_request_id: Any,
    api_start_time: Any, effective_task_id: Any, turn_id: Any,
) -> ApiRequestBuild:
    """Assemble the attempt's request in the original order (every mutation happens BEFORE
    middleware/hooks/debug dumps observe the payload)."""
    from agent.conversation_loop import _moa_client_consumes_prepared_request

    built = build_request_payload(
        agent, api_messages=api_messages, _moa_prepared_request=_moa_prepared_request,
        tools_for_api=tools_for_api, system_message=system_message, api_call_count=api_call_count,
        api_request_id=api_request_id, effective_task_id=effective_task_id, turn_id=turn_id,
    )
    api_kwargs, api_messages, _moa_prepared_request = built.api_kwargs, built.api_messages, built._moa_prepared_request
    _fire_pre_api_request_hook(
        agent, api_kwargs, api_messages, built._llm_middleware_trace, messages=messages,
        original_user_message=original_user_message, approx_tokens=approx_tokens,
        total_chars=total_chars, retry_count=retry_count, api_call_count=api_call_count,
        api_request_id=api_request_id, api_start_time=api_start_time,
        effective_task_id=effective_task_id, turn_id=turn_id,
    )

    if env_var_enabled("HERMES_DUMP_REQUESTS"):
        agent._dump_api_request_debug(api_kwargs, reason="preflight")

    # Private to the in-process MoA facade; added after middleware/hooks/debug dumps so
    # none serializes it into the provider payload. Re-read the live client:
    # rotation/fallback/cleanup rebuild agent.client between attempts; a native OpenAI
    # client rejects this key (TypeError).
    if _moa_prepared_request is not None and agent.provider == "moa":
        if _moa_client_consumes_prepared_request(agent.client):
            api_kwargs["_moa_prepared_request"] = _moa_prepared_request
        else:
            logger.warning(
                "MoA client replaced mid-turn (client=%s); sending the "
                "prepared prompt without the MoA handshake",
                type(agent.client).__name__,
            )
    return built
