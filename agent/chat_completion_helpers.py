"""Helper functions for the chat-completions code path.

Extracted from :class:`AIAgent` for cleanliness — bodies of the
non-streaming API call, request kwargs builder, assistant-message
materializer, provider-fallback activator, max-iterations handler,
and per-turn resource cleanup.

Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  :class:`AIAgent` keeps thin forwarder methods so call
sites unchanged.  Symbols that tests patch on ``run_agent`` (e.g.
``cleanup_vm`` / ``cleanup_browser`` in
``test_zombie_process_cleanup.py``) are resolved through
:func:`_ra` so the patch contract is preserved.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import copy
import json
import logging
import os
import random
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlunparse

from hermes_cli.timeouts import get_provider_request_timeout
from agent.error_classifier import classify_api_error, FailoverReason
from agent.model_metadata import is_local_endpoint
from agent.message_sanitization import (
    _sanitize_surrogates,
    _sanitize_messages_surrogates,
    _sanitize_structure_surrogates,
    _sanitize_messages_non_ascii,
    _sanitize_tools_non_ascii,
    _sanitize_structure_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
    _repair_tool_call_arguments,
    _escape_invalid_chars_in_json_strings,
)
from agent.tool_dispatch_helpers import (
    _is_multimodal_tool_result,
    _multimodal_text_summary,
)
from agent.retry_utils import jittered_backoff
from agent.tool_guardrails import (
    ToolGuardrailDecision,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)
from tools.terminal_tool import is_persistent_env
from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)


def _ra():
    """Lazy ``run_agent`` reference.

    Used to honor test patches like
    ``patch("run_agent.cleanup_vm")`` / ``patch("run_agent.cleanup_browser")``
    that target symbols imported into ``run_agent``'s namespace.
    """
    import run_agent
    return run_agent



def interruptible_api_call(agent, api_kwargs: dict):
    """
    Run the API call in a background thread so the main conversation loop
    can detect interrupts without waiting for the full HTTP round-trip.

    Each worker thread gets its own OpenAI client instance. Interrupts only
    close that worker-local client, so retries and other requests never
    inherit a closed transport.

    Includes a stale-call detector: if no response arrives within the
    configured timeout, the connection is killed and an error raised so
    the main retry loop can try again with backoff / credential rotation /
    provider fallback.
    """
    result = {"response": None, "error": None}
    request_client_holder = {"client": None}

    def _call():
        try:
            if agent.api_mode == "codex_responses":
                request_client_holder["client"] = agent._create_request_openai_client(
                    reason="codex_stream_request",
                    api_kwargs=api_kwargs,
                )
                result["response"] = agent._run_codex_stream(
                    api_kwargs,
                    client=request_client_holder["client"],
                    on_first_delta=getattr(agent, "_codex_on_first_delta", None),
                )
            elif agent.api_mode == "anthropic_messages":
                result["response"] = agent._anthropic_messages_create(api_kwargs)
            elif agent.api_mode == "bedrock_converse":
                # Bedrock uses boto3 directly — no OpenAI client needed.
                # normalize_converse_response produces an OpenAI-compatible
                # SimpleNamespace so the rest of the agent loop can treat
                # bedrock responses like chat_completions responses.
                from agent.bedrock_adapter import (
                    _get_bedrock_runtime_client,
                    invalidate_runtime_client,
                    is_stale_connection_error,
                    normalize_converse_response,
                )
                region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                api_kwargs.pop("__bedrock_converse__", None)
                client = _get_bedrock_runtime_client(region)
                try:
                    raw_response = client.converse(**api_kwargs)
                except Exception as _bedrock_exc:
                    # Evict the cached client on stale-connection failures
                    # so the outer retry loop builds a fresh client/pool.
                    if is_stale_connection_error(_bedrock_exc):
                        invalidate_runtime_client(region)
                    raise
                result["response"] = normalize_converse_response(raw_response)
            else:
                request_client_holder["client"] = agent._create_request_openai_client(
                    reason="chat_completion_request",
                    api_kwargs=api_kwargs,
                )
                result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
        except Exception as e:
            result["error"] = e
        finally:
            request_client = request_client_holder.get("client")
            if request_client is not None:
                agent._close_request_openai_client(request_client, reason="request_complete")

    # ── Stale-call timeout (mirrors streaming stale detector) ────────
    # Non-streaming calls return nothing until the full response is
    # ready.  Without this, a hung provider can block for the full
    # httpx timeout (default 1800s) with zero feedback.  The stale
    # detector kills the connection early so the main retry loop can
    # apply richer recovery (credential rotation, provider fallback).
    _stale_timeout = agent._compute_non_stream_stale_timeout(
        api_kwargs.get("messages", [])
    )

    _call_start = time.time()
    agent._touch_activity("waiting for non-streaming API response")

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    _poll_count = 0
    while t.is_alive():
        t.join(timeout=0.3)
        _poll_count += 1

        # Touch activity every ~30s so the gateway's inactivity
        # monitor knows we're alive while waiting for the response.
        if _poll_count % 100 == 0:  # 100 × 0.3s = 30s
            _elapsed = time.time() - _call_start
            agent._touch_activity(
                f"waiting for non-streaming response ({int(_elapsed)}s elapsed)"
            )

        # Stale-call detector: kill the connection if no response
        # arrives within the configured timeout.
        _elapsed = time.time() - _call_start
        if _elapsed > _stale_timeout:
            _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            logger.warning(
                "Non-streaming API call stale for %.0fs (threshold %.0fs). "
                "model=%s context=~%s tokens. Killing connection.",
                _elapsed, _stale_timeout,
                api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
            )
            agent._emit_status(
                f"⚠️ No response from provider for {int(_elapsed)}s "
                f"(non-streaming, model: {api_kwargs.get('model', 'unknown')}). "
                f"Aborting call."
            )
            try:
                if agent.api_mode == "anthropic_messages":
                    agent._anthropic_client.close()
                    agent._rebuild_anthropic_client()
                else:
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        agent._close_request_openai_client(rc, reason="stale_call_kill")
            except Exception:
                pass
            agent._touch_activity(
                f"stale non-streaming call killed after {int(_elapsed)}s"
            )
            # Wait briefly for the thread to notice the closed connection.
            t.join(timeout=2.0)
            if result["error"] is None and result["response"] is None:
                result["error"] = TimeoutError(
                    f"Non-streaming API call timed out after {int(_elapsed)}s "
                    f"with no response (threshold: {int(_stale_timeout)}s)"
                )
            break

        if agent._interrupt_requested:
            # Force-close the in-flight worker-local HTTP connection to stop
            # token generation without poisoning the shared client used to
            # seed future retries.
            try:
                if agent.api_mode == "anthropic_messages":
                    agent._anthropic_client.close()
                    agent._rebuild_anthropic_client()
                else:
                    request_client = request_client_holder.get("client")
                    if request_client is not None:
                        agent._close_request_openai_client(request_client, reason="interrupt_abort")
            except Exception:
                pass
            raise InterruptedError("Agent interrupted during API call")
    if result["error"] is not None:
        raise result["error"]
    return result["response"]



def build_api_kwargs(agent, api_messages: list) -> dict:
    """Build the keyword arguments dict for the active API mode."""
    tools_for_api = agent.tools

    if agent.api_mode == "anthropic_messages":
        _transport = agent._get_transport()
        anthropic_messages = agent._prepare_anthropic_messages_for_api(api_messages)
        ctx_len = getattr(agent, "context_compressor", None)
        ctx_len = ctx_len.context_length if ctx_len else None
        ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
        if ephemeral_out is not None:
            agent._ephemeral_max_output_tokens = None  # consume immediately
        return _transport.build_kwargs(
            model=agent.model,
            messages=anthropic_messages,
            tools=tools_for_api,
            max_tokens=ephemeral_out if ephemeral_out is not None else agent.max_tokens,
            reasoning_config=agent.reasoning_config,
            is_oauth=agent._is_anthropic_oauth,
            preserve_dots=agent._anthropic_preserve_dots(),
            context_length=ctx_len,
            base_url=getattr(agent, "_anthropic_base_url", None),
            fast_mode=(agent.request_overrides or {}).get("speed") == "fast",
            drop_context_1m_beta=bool(getattr(agent, "_oauth_1m_beta_disabled", False)),
        )

    # AWS Bedrock native Converse API — bypasses the OpenAI client entirely.
    # The adapter handles message/tool conversion and boto3 calls directly.
    if agent.api_mode == "bedrock_converse":
        _bt = agent._get_transport()
        region = getattr(agent, "_bedrock_region", None) or "us-east-1"
        guardrail = getattr(agent, "_bedrock_guardrail_config", None)
        return _bt.build_kwargs(
            model=agent.model,
            messages=api_messages,
            tools=tools_for_api,
            max_tokens=agent.max_tokens or 4096,
            region=region,
            guardrail_config=guardrail,
        )

    if agent.api_mode == "codex_responses":
        _ct = agent._get_transport()
        is_github_responses = (
            base_url_host_matches(agent.base_url, "models.github.ai")
            or base_url_host_matches(agent.base_url, "api.githubcopilot.com")
        )
        is_codex_backend = (
            agent.provider == "openai-codex"
            or (
                agent._base_url_hostname == "chatgpt.com"
                and "/backend-api/codex" in agent._base_url_lower
            )
        )
        is_xai_responses = agent.provider in {"xai", "xai-oauth"} or agent._base_url_hostname == "api.x.ai"
        _msgs_for_codex = agent._prepare_messages_for_non_vision_model(api_messages)
        return _ct.build_kwargs(
            model=agent.model,
            messages=_msgs_for_codex,
            tools=tools_for_api,
            reasoning_config=agent.reasoning_config,
            session_id=getattr(agent, "session_id", None),
            max_tokens=agent.max_tokens,
            request_overrides=agent.request_overrides,
            is_github_responses=is_github_responses,
            is_codex_backend=is_codex_backend,
            is_xai_responses=is_xai_responses,
            github_reasoning_extra=agent._github_models_reasoning_extra_body() if is_github_responses else None,
        )

    # ── chat_completions (default) ─────────────────────────────────────
    _ct = agent._get_transport()

    # Provider detection flags
    _is_qwen = agent._is_qwen_portal()
    _is_or = agent._is_openrouter_url()
    _is_gh = (
        base_url_host_matches(agent._base_url_lower, "models.github.ai")
        or base_url_host_matches(agent._base_url_lower, "api.githubcopilot.com")
    )
    _is_nous = "nousresearch" in agent._base_url_lower
    _is_nvidia = "integrate.api.nvidia.com" in agent._base_url_lower
    _is_kimi = (
        base_url_host_matches(agent.base_url, "api.kimi.com")
        or base_url_host_matches(agent.base_url, "moonshot.ai")
        or base_url_host_matches(agent.base_url, "moonshot.cn")
    )
    _is_tokenhub = base_url_host_matches(agent._base_url_lower, "tokenhub.tencentmaas.com")
    _is_lmstudio = (agent.provider or "").strip().lower() == "lmstudio"

    # Temperature: _fixed_temperature_for_model may return OMIT_TEMPERATURE
    # sentinel (temperature omitted entirely), a numeric override, or None.
    try:
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
        _ft = _fixed_temperature_for_model(agent.model, agent.base_url)
        _omit_temp = _ft is OMIT_TEMPERATURE
        _fixed_temp = _ft if not _omit_temp else None
    except Exception:
        _omit_temp = False
        _fixed_temp = None

    # Provider preferences (OpenRouter-style)
    _prefs: Dict[str, Any] = {}
    if agent.providers_allowed:
        _prefs["only"] = agent.providers_allowed
    if agent.providers_ignored:
        _prefs["ignore"] = agent.providers_ignored
    if agent.providers_order:
        _prefs["order"] = agent.providers_order
    if agent.provider_sort:
        _prefs["sort"] = agent.provider_sort
    if agent.provider_require_parameters:
        _prefs["require_parameters"] = True
    if agent.provider_data_collection:
        _prefs["data_collection"] = agent.provider_data_collection

    # Claude max-output override on aggregators
    _ant_max = None
    if (_is_or or _is_nous) and "claude" in (agent.model or "").lower():
        try:
            from agent.anthropic_adapter import _get_anthropic_max_output
            _ant_max = _get_anthropic_max_output(agent.model)
        except Exception:
            pass

    # Qwen session metadata
    _qwen_meta = None
    if _is_qwen:
        _qwen_meta = {
            "sessionId": agent.session_id or "hermes",
            "promptId": str(uuid.uuid4()),
        }

    # ── Provider profile path (registered providers) ───────────────────
    # Profiles handle per-provider quirks via hooks. When a profile is
    # found, delegate fully; otherwise fall through to the legacy flag path.
    try:
        from providers import get_provider_profile
        _profile = get_provider_profile(agent.provider)
    except Exception:
        _profile = None

    if _profile:
        _ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
        if _ephemeral_out is not None:
            agent._ephemeral_max_output_tokens = None

        # Strip image parts for non-vision models that have provider profiles
        # (e.g. DeepSeek, Kimi). The legacy path below already does this, but
        # registered providers with profiles were bypassing the strip.
        api_messages = agent._prepare_messages_for_non_vision_model(api_messages)

        return _ct.build_kwargs(
            model=agent.model,
            messages=api_messages,
            tools=tools_for_api,
            base_url=agent.base_url,
            timeout=agent._resolved_api_call_timeout(),
            max_tokens=agent.max_tokens,
            ephemeral_max_output_tokens=_ephemeral_out,
            max_tokens_param_fn=agent._max_tokens_param,
            reasoning_config=agent.reasoning_config,
            request_overrides=agent.request_overrides,
            session_id=getattr(agent, "session_id", None),
            provider_profile=_profile,
            ollama_num_ctx=agent._ollama_num_ctx,
            # Context forwarded to profile hooks:
            provider_preferences=_prefs or None,
            openrouter_min_coding_score=agent.openrouter_min_coding_score,
            anthropic_max_output=_ant_max,
            supports_reasoning=agent._supports_reasoning_extra_body(),
            qwen_session_metadata=_qwen_meta,
        )

    # ── Legacy flag path ────────────────────────────────────────────
    # Reached only when get_provider_profile() returns None — i.e. a
    # completely unknown provider not in providers/ registry.
    _ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
    if _ephemeral_out is not None:
        agent._ephemeral_max_output_tokens = None

    # Strip image parts for non-vision models (no-op when vision-capable).
    _msgs_for_chat = agent._prepare_messages_for_non_vision_model(api_messages)

    return _ct.build_kwargs(
        model=agent.model,
        messages=_msgs_for_chat,
        tools=tools_for_api,
        base_url=agent.base_url,
        timeout=agent._resolved_api_call_timeout(),
        max_tokens=agent.max_tokens,
        ephemeral_max_output_tokens=_ephemeral_out,
        max_tokens_param_fn=agent._max_tokens_param,
        reasoning_config=agent.reasoning_config,
        request_overrides=agent.request_overrides,
        session_id=getattr(agent, "session_id", None),
        model_lower=(agent.model or "").lower(),
        is_openrouter=_is_or,
        is_nous=_is_nous,
        is_qwen_portal=_is_qwen,
        is_github_models=_is_gh,
        is_nvidia_nim=_is_nvidia,
        is_kimi=_is_kimi,
        is_tokenhub=_is_tokenhub,
        is_lmstudio=_is_lmstudio,
        is_custom_provider=agent.provider == "custom",
        ollama_num_ctx=agent._ollama_num_ctx,
        provider_preferences=_prefs or None,
        openrouter_min_coding_score=agent.openrouter_min_coding_score,
        qwen_prepare_fn=agent._qwen_prepare_chat_messages if _is_qwen else None,
        qwen_prepare_inplace_fn=agent._qwen_prepare_chat_messages_inplace if _is_qwen else None,
        qwen_session_metadata=_qwen_meta,
        fixed_temperature=_fixed_temp,
        omit_temperature=_omit_temp,
        supports_reasoning=agent._supports_reasoning_extra_body(),
        github_reasoning_extra=agent._github_models_reasoning_extra_body() if _is_gh else None,
        lmstudio_reasoning_options=agent._lmstudio_reasoning_options_cached() if _is_lmstudio else None,
        anthropic_max_output=_ant_max,
        provider_name=agent.provider,
    )



def build_assistant_message(agent, assistant_message, finish_reason: str) -> dict:
    """Build a normalized assistant message dict from an API response message.

    Handles reasoning extraction, reasoning_details, and optional tool_calls
    so both the tool-call path and the final-response path share one builder.
    """
    assistant_tool_calls = getattr(assistant_message, "tool_calls", None)
    reasoning_text = agent._extract_reasoning(assistant_message)
    _from_structured = bool(reasoning_text)

    # Fallback: extract inline <think> blocks from content when no structured
    # reasoning fields are present (some models/providers embed thinking
    # directly in the content rather than returning separate API fields).
    if not reasoning_text:
        content = assistant_message.content or ""
        think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
        if think_blocks:
            combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
            reasoning_text = combined or None

    if reasoning_text and agent.verbose_logging:
        logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

    if reasoning_text and agent.reasoning_callback:
        # Skip callback when streaming is active — reasoning was already
        # displayed during the stream via one of two paths:
        #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
        #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
        # When streaming is NOT active, always fire so non-streaming modes
        # (gateway, batch, quiet) still get reasoning.
        # Any reasoning that wasn't shown during streaming is caught by the
        # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
        if not agent.stream_delta_callback and not agent._stream_callback:
            try:
                agent.reasoning_callback(reasoning_text)
            except Exception:
                pass

    # Sanitize surrogates from API response — some models (e.g. Kimi/GLM via Ollama)
    # can return invalid surrogate code points that crash json.dumps() on persist.
    _raw_content = assistant_message.content or ""
    _san_content = _sanitize_surrogates(_raw_content)
    if reasoning_text:
        reasoning_text = _sanitize_surrogates(reasoning_text)

    # Strip inline reasoning tags (<think>…</think> etc.) from the stored
    # assistant content.  Reasoning was already captured into
    # ``reasoning_text`` above (either from structured fields or the
    # inline-block fallback), so the raw tags in content are redundant.
    # Leaving them in place caused reasoning to leak to messaging
    # platforms (#8878, #9568), inflate context on subsequent turns
    # (#9306 observed 16% content-size reduction on a real MiniMax
    # session), and pollute generated session titles.  One strip at the
    # storage boundary cleans content for every downstream consumer:
    # API replay, session transcript, gateway delivery, CLI display,
    # compression, title generation.
    if isinstance(_san_content, str) and _san_content:
        _san_content = agent._strip_think_blocks(_san_content).strip()

    msg = {
        "role": "assistant",
        "content": _san_content,
        "reasoning": reasoning_text,
        "finish_reason": finish_reason,
    }

    raw_reasoning_content = getattr(assistant_message, "reasoning_content", None)
    if raw_reasoning_content is None and hasattr(assistant_message, "model_extra"):
        model_extra = getattr(assistant_message, "model_extra", None) or {}
        if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
            raw_reasoning_content = model_extra["reasoning_content"]
    if raw_reasoning_content is not None:
        msg["reasoning_content"] = _sanitize_surrogates(raw_reasoning_content)
    elif assistant_tool_calls and agent._needs_thinking_reasoning_pad():
        # DeepSeek v4 thinking mode and Kimi / Moonshot thinking mode
        # both require reasoning_content on every assistant tool-call
        # message. Without it, replaying the persisted message causes
        # HTTP 400 ("The reasoning_content in the thinking mode must
        # be passed back to the API"). Include streamed reasoning
        # text when captured; otherwise pad with a single space —
        # DeepSeek V4 Pro tightened validation and rejects empty
        # string ("The reasoning content in the thinking mode must
        # be passed back to the API"). A space satisfies non-empty
        # checks everywhere without leaking fabricated reasoning.
        # Refs #15250, #17400, #17341.
        msg["reasoning_content"] = reasoning_text or " "

    # Additive fallback (refs #16844, #16884). Streaming-only providers
    # (glm, MiniMax, gpt-5.x via aigw, Anthropic via openai-compat shims)
    # accumulate reasoning through ``delta.reasoning_content`` chunks
    # but never land it on the message object as a top-level attribute,
    # so neither branch above fires and the chain-of-thought is stored
    # only under the internal ``reasoning`` key. When the user later
    # replays that history through a DeepSeek-v4 / Kimi thinking model,
    # the missing ``reasoning_content`` causes HTTP 400 ("The
    # reasoning_content in the thinking mode must be passed back to the
    # API.").
    #
    # Promote the already-sanitized streamed ``reasoning_text`` to
    # ``reasoning_content`` at write time, but ONLY when no prior branch
    # already set it AND we actually captured reasoning text. This
    # preserves every existing behavior:
    #   - SDK-exposed ``reasoning_content`` (OpenAI/Moonshot/DeepSeek SDK)
    #     still wins.
    #   - DeepSeek tool-call ""-pad (#15250) still fires.
    #   - Non-thinking turns with no reasoning leave the field absent,
    #     so ``_copy_reasoning_content_for_api``'s cross-provider leak
    #     guard (#15748) and ``reasoning``→``reasoning_content``
    #     promotion tiers still apply at replay time.
    if "reasoning_content" not in msg and reasoning_text:
        msg["reasoning_content"] = reasoning_text

    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        # Pass reasoning_details back unmodified so providers (OpenRouter,
        # Anthropic, OpenAI) can maintain reasoning continuity across turns.
        # Each provider may include opaque fields (signature, encrypted_content)
        # that must be preserved exactly.
        raw_details = assistant_message.reasoning_details
        preserved = []
        for d in raw_details:
            if isinstance(d, dict):
                preserved.append(d)
            elif hasattr(d, "__dict__"):
                preserved.append(d.__dict__)
            elif hasattr(d, "model_dump"):
                preserved.append(d.model_dump())
        if preserved:
            msg["reasoning_details"] = preserved

    # Codex Responses API: preserve encrypted reasoning items for
    # multi-turn continuity. These get replayed as input on the next turn.
    codex_items = getattr(assistant_message, "codex_reasoning_items", None)
    if codex_items:
        msg["codex_reasoning_items"] = codex_items

    # Codex Responses API: preserve exact assistant message items (with
    # id/phase) so follow-up turns can replay structured items instead of
    # flattening to plain text. This is required for prefix cache hits.
    codex_message_items = getattr(assistant_message, "codex_message_items", None)
    if codex_message_items:
        msg["codex_message_items"] = codex_message_items

    if assistant_tool_calls:
        tool_calls = []
        for tool_call in assistant_tool_calls:
            raw_id = getattr(tool_call, "id", None)
            call_id = getattr(tool_call, "call_id", None)
            if not isinstance(call_id, str) or not call_id.strip():
                embedded_call_id, _ = agent._split_responses_tool_id(raw_id)
                call_id = embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                if isinstance(raw_id, str) and raw_id.strip():
                    call_id = raw_id.strip()
                else:
                    _fn = getattr(tool_call, "function", None)
                    _fn_name = getattr(_fn, "name", "") if _fn else ""
                    _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                    call_id = agent._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
            call_id = call_id.strip()

            response_item_id = getattr(tool_call, "response_item_id", None)
            if not isinstance(response_item_id, str) or not response_item_id.strip():
                _, embedded_response_item_id = agent._split_responses_tool_id(raw_id)
                response_item_id = embedded_response_item_id

            response_item_id = agent._derive_responses_function_call_id(
                call_id,
                response_item_id if isinstance(response_item_id, str) else None,
            )

            tc_dict = {
                "id": call_id,
                "call_id": call_id,
                "response_item_id": response_item_id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                },
            }
            # Preserve extra_content (e.g. Gemini thought_signature) so it
            # is sent back on subsequent API calls.  Without this, Gemini 3
            # thinking models reject the request with a 400 error.
            extra = getattr(tool_call, "extra_content", None)
            if extra is not None:
                if hasattr(extra, "model_dump"):
                    extra = extra.model_dump()
                tc_dict["extra_content"] = extra
            tool_calls.append(tc_dict)
        msg["tool_calls"] = tool_calls

    return msg



def try_activate_fallback(agent, reason: "FailoverReason | None" = None) -> bool:
    """Switch to the next fallback model/provider in the chain.

    Called when the current model is failing after retries.  Swaps the
    OpenAI client, model slug, and provider in-place so the retry loop
    can continue with the new backend.  Advances through the chain on
    each call; returns False when exhausted.

    Uses the centralized provider router (resolve_provider_client) for
    auth resolution and client construction — no duplicated provider→key
    mappings.
    """
    if reason in {FailoverReason.rate_limit, FailoverReason.billing}:
        # Only start cooldown when leaving the primary provider.  If we're
        # already on a fallback and chain-switching, the primary wasn't the
        # source of the 429 so the cooldown should not be reset/extended.
        fallback_already_active = bool(getattr(agent, "_fallback_activated", False))
        current_provider = (getattr(agent, "provider", "") or "").strip().lower()
        primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
        if (not fallback_already_active) or (primary_provider and current_provider == primary_provider):
            agent._rate_limited_until = time.monotonic() + 60
    if agent._fallback_index >= len(agent._fallback_chain):
        return False

    fb = agent._fallback_chain[agent._fallback_index]
    agent._fallback_index += 1
    fb_provider = (fb.get("provider") or "").strip().lower()
    fb_model = (fb.get("model") or "").strip()
    if not fb_provider or not fb_model:
        return agent._try_activate_fallback()  # skip invalid, try next

    # Skip entries that resolve to the current (provider, model) — falling
    # back to the same backend that just failed loops the failure. Compare
    # base_url too so two distinct custom_providers entries pointing at the
    # same shim/proxy URL also dedup. See issue #22548.
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    current_model = (getattr(agent, "model", "") or "").strip()
    current_base_url = str(getattr(agent, "base_url", "") or "").rstrip("/").lower()
    fb_base_url_for_dedup = (fb.get("base_url") or "").strip().rstrip("/").lower()
    if fb_provider == current_provider and fb_model == current_model:
        logging.warning(
            "Fallback skip: chain entry %s/%s matches current provider/model",
            fb_provider, fb_model,
        )
        return agent._try_activate_fallback()
    if (
        fb_base_url_for_dedup
        and current_base_url
        and fb_base_url_for_dedup == current_base_url
        and fb_model == current_model
    ):
        logging.warning(
            "Fallback skip: chain entry base_url %s matches current backend",
            fb_base_url_for_dedup,
        )
        return agent._try_activate_fallback()

    # Use centralized router for client construction.
    # raw_codex=True because the main agent needs direct responses.stream()
    # access for Codex providers.
    try:
        from agent.auxiliary_client import resolve_provider_client
        # Pass base_url and api_key from fallback config so custom
        # endpoints (e.g. Ollama Cloud) resolve correctly instead of
        # falling through to OpenRouter defaults.
        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = (fb.get("api_key") or "").strip() or None
        if not fb_api_key_hint:
            # key_env and api_key_env are both documented aliases (see
            # _normalize_custom_provider_entry in hermes_cli/config.py).
            fb_key_env = (fb.get("key_env") or fb.get("api_key_env") or "").strip()
            if fb_key_env:
                fb_api_key_hint = os.getenv(fb_key_env, "").strip() or None
        # For Ollama Cloud endpoints, pull OLLAMA_API_KEY from env
        # when no explicit key is in the fallback config. Host match
        # (not substring) — see GHSA-76xc-57q6-vm5m.
        if fb_base_url_hint and base_url_host_matches(fb_base_url_hint, "ollama.com") and not fb_api_key_hint:
            fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
        fb_client, _resolved_fb_model = resolve_provider_client(
            fb_provider, model=fb_model, raw_codex=True,
            explicit_base_url=fb_base_url_hint,
            explicit_api_key=fb_api_key_hint)
        if fb_client is None:
            logging.warning(
                "Fallback to %s failed: provider not configured",
                fb_provider)
            return agent._try_activate_fallback()  # try next in chain
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider

            fb_model = normalize_model_for_provider(fb_model, fb_provider)
        except Exception:
            pass

        # Determine api_mode from provider / base URL / model
        fb_api_mode = "chat_completions"
        fb_base_url = str(fb_client.base_url)
        _fb_is_azure = agent._is_azure_openai_url(fb_base_url)
        if fb_provider == "openai-codex":
            fb_api_mode = "codex_responses"
        elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith("/anthropic"):
            fb_api_mode = "anthropic_messages"
        elif _fb_is_azure:
            # Azure OpenAI serves gpt-5.x on /chat/completions — does NOT
            # support the Responses API. Stay on chat_completions.
            fb_api_mode = "chat_completions"
        elif agent._is_direct_openai_url(fb_base_url):
            fb_api_mode = "codex_responses"
        elif agent._provider_model_requires_responses_api(
            fb_model,
            provider=fb_provider,
        ):
            # GPT-5.x models usually need Responses API, but keep
            # provider-specific exceptions like Copilot gpt-5-mini on
            # chat completions.
            fb_api_mode = "codex_responses"
        elif fb_provider == "bedrock" or (
            base_url_hostname(fb_base_url).startswith("bedrock-runtime.")
            and base_url_host_matches(fb_base_url, "amazonaws.com")
        ):
            fb_api_mode = "bedrock_converse"

        old_model = agent.model

        # Clear the per-config context_length override so the fallback
        # model's actual context window is resolved instead of inheriting
        # the stale value from the previous model.  See #22387.
        agent._config_context_length = None
        agent.model = fb_model
        agent.provider = fb_provider
        agent.base_url = fb_base_url
        agent.api_mode = fb_api_mode
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent._fallback_activated = True

        # Honor per-provider / per-model request_timeout_seconds for the
        # fallback target (same knob the primary client uses).  None = use
        # SDK default.
        _fb_timeout = get_provider_request_timeout(fb_provider, fb_model)

        if fb_api_mode == "anthropic_messages":
            # Build native Anthropic client instead of using OpenAI client
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
            effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = fb_base_url
            agent._anthropic_client = build_anthropic_client(
                effective_key, agent._anthropic_base_url, timeout=_fb_timeout,
            )
            agent._is_anthropic_oauth = _is_oauth_token(effective_key) if fb_provider == "anthropic" else False
            agent.client = None
            agent._client_kwargs = {}
        else:
            # Swap OpenAI client and config in-place
            agent.api_key = fb_client.api_key
            agent.client = fb_client
            # Preserve provider-specific headers that
            # resolve_provider_client() may have baked into
            # fb_client via the default_headers kwarg.  The OpenAI
            # SDK stores these in _custom_headers.  Without this,
            # subsequent request-client rebuilds (via
            # _create_request_openai_client) drop the headers,
            # causing 403s from providers like Kimi Coding that
            # require a User-Agent sentinel.
            fb_headers = getattr(fb_client, "_custom_headers", None)
            if not fb_headers:
                fb_headers = getattr(fb_client, "default_headers", None)
            agent._client_kwargs = {
                "api_key": fb_client.api_key,
                "base_url": fb_base_url,
                **({"default_headers": dict(fb_headers)} if fb_headers else {}),
            }
            if _fb_timeout is not None:
                agent._client_kwargs["timeout"] = _fb_timeout
                # Rebuild the shared OpenAI client so the configured
                # timeout takes effect on the very next fallback request,
                # not only after a later credential-rotation rebuild.
                agent._replace_primary_openai_client(reason="fallback_timeout_apply")

        # Re-evaluate prompt caching for the new provider/model
        agent._use_prompt_caching, agent._use_native_cache_layout = (
            agent._anthropic_prompt_cache_policy(
                provider=fb_provider,
                base_url=fb_base_url,
                api_mode=fb_api_mode,
                model=fb_model,
            )
        )

        # LM Studio: preload before probing the fallback's context length.
        agent._ensure_lmstudio_runtime_loaded()

        # Update context compressor limits for the fallback model.
        # Without this, compression decisions use the primary model's
        # context window (e.g. 200K) instead of the fallback's (e.g. 32K),
        # causing oversized sessions to overflow the fallback.
        # Also pass _config_context_length so the explicit config override
        # (model.context_length in config.yaml) is respected — without this,
        # the fallback activation drops to 128K even when config says 204800.
        if hasattr(agent, 'context_compressor') and agent.context_compressor:
            from agent.model_metadata import get_model_context_length
            fb_context_length = get_model_context_length(
                agent.model, base_url=agent.base_url,
                api_key=agent.api_key, provider=agent.provider,
                config_context_length=getattr(agent, "_config_context_length", None),
                custom_providers=getattr(agent, "_custom_providers", None),
            )
            agent.context_compressor.update_model(
                model=agent.model,
                context_length=fb_context_length,
                base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),
                provider=agent.provider,
            )

        agent._emit_status(
            f"🔄 Primary model failed — switching to fallback: "
            f"{fb_model} via {fb_provider}"
        )
        logging.info(
            "Fallback activated: %s → %s (%s)",
            old_model, fb_model, fb_provider,
        )
        return True
    except Exception as e:
        logging.error("Failed to activate fallback %s: %s", fb_model, e)
        return agent._try_activate_fallback()  # try next in chain



def handle_max_iterations(agent, messages: list, api_call_count: int) -> str:
    """Request a summary when max iterations are reached. Returns the final response text."""
    print(f"⚠️  Reached maximum iterations ({agent.max_iterations}). Requesting summary...")

    summary_request = (
        "You've reached the maximum number of tool-calling iterations allowed. "
        "Please provide a final response summarizing what you've found and accomplished so far, "
        "without calling any more tools."
    )
    messages.append({"role": "user", "content": summary_request})

    try:
        # Build API messages, stripping internal-only fields
        # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
        _needs_sanitize = agent._should_sanitize_tool_calls()
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            agent._copy_reasoning_content_for_api(msg, api_msg)
            for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                api_msg.pop(internal_field, None)
            if _needs_sanitize:
                agent._sanitize_tool_calls_for_strict_api(api_msg)
            api_messages.append(api_msg)

        effective_system = agent._cached_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages
        if agent.prefill_messages:
            sys_offset = 1 if effective_system else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Same safety net as the main loop: repair tool-call/result
        # pairing before asking for a final summary.  Compression and
        # session resume can leave a tool result whose parent assistant
        # tool_call was summarized away; Responses API rejects that as
        # "No tool call found for function call output".
        api_messages = agent._sanitize_api_messages(api_messages)

        # Same safety net as the main loop: drop thinking-only assistant
        # turns so Anthropic-family providers don't 400 the summary call.
        api_messages = agent._drop_thinking_only_and_merge_users(api_messages)

        summary_extra_body = {}
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
        except Exception:
            _fixed_temperature_for_model = None
            _OMIT_TEMP = None
        _raw_summary_temp = (
            _fixed_temperature_for_model(agent.model, agent.base_url)
            if _fixed_temperature_for_model is not None
            else None
        )
        _omit_summary_temperature = _raw_summary_temp is _OMIT_TEMP
        _summary_temperature = None if _omit_summary_temperature else _raw_summary_temp
        _is_nous = "nousresearch" in agent._base_url_lower
        # LM Studio uses top-level `reasoning_effort` (not extra_body.reasoning).
        # Mirror ChatCompletionsTransport.build_kwargs() so the summary path
        # — which calls chat.completions.create() directly without going
        # through the transport — sends the same shape the transport does.
        _is_lmstudio_summary = (
            (agent.provider or "").strip().lower() == "lmstudio"
            and agent._supports_reasoning_extra_body()
        )
        _lm_reasoning_effort: str | None = (
            agent._resolve_lmstudio_summary_reasoning_effort()
            if _is_lmstudio_summary else None
        )
        if not _is_lmstudio_summary and agent._supports_reasoning_extra_body():
            if agent.reasoning_config is not None:
                summary_extra_body["reasoning"] = agent.reasoning_config
            else:
                summary_extra_body["reasoning"] = {
                    "enabled": True,
                    "effort": "medium"
                }
        if _is_nous:
            from agent.portal_tags import nous_portal_tags as _portal_tags
            summary_extra_body["tags"] = _portal_tags()

        if agent.api_mode == "codex_responses":
            codex_kwargs = agent._build_api_kwargs(api_messages)
            codex_kwargs.pop("tools", None)
            summary_response = agent._run_codex_stream(codex_kwargs)
            _ct_sum = agent._get_transport()
            _cnr_sum = _ct_sum.normalize_response(summary_response)
            final_response = (_cnr_sum.content or "").strip()
        else:
            summary_kwargs = {
                "model": agent.model,
                "messages": api_messages,
            }
            if _summary_temperature is not None:
                summary_kwargs["temperature"] = _summary_temperature
            if agent.max_tokens is not None:
                summary_kwargs.update(agent._max_tokens_param(agent.max_tokens))
            if _lm_reasoning_effort is not None:
                summary_kwargs["reasoning_effort"] = _lm_reasoning_effort

            # Include provider routing preferences
            provider_preferences = {}
            if agent.providers_allowed:
                provider_preferences["only"] = agent.providers_allowed
            if agent.providers_ignored:
                provider_preferences["ignore"] = agent.providers_ignored
            if agent.providers_order:
                provider_preferences["order"] = agent.providers_order
            if agent.provider_sort:
                provider_preferences["sort"] = agent.provider_sort
            if provider_preferences and (
                (agent.provider or "").strip().lower() == "openrouter"
                or agent._is_openrouter_url()
            ):
                summary_extra_body["provider"] = provider_preferences

            # Pareto Code router plugin — model-gated. Same shape as
            # the main-loop emission so summary calls on
            # openrouter/pareto-code respect the user's coding-score floor.
            if (
                agent.model == "openrouter/pareto-code"
                and (
                    (agent.provider or "").strip().lower() == "openrouter"
                    or agent._is_openrouter_url()
                )
                and agent.openrouter_min_coding_score is not None
                and agent.openrouter_min_coding_score != ""
            ):
                try:
                    _ps = float(agent.openrouter_min_coding_score)
                except (TypeError, ValueError):
                    _ps = None
                if _ps is not None and 0.0 <= _ps <= 1.0:
                    summary_extra_body["plugins"] = [
                        {"id": "pareto-router", "min_coding_score": _ps}
                    ]

            if summary_extra_body:
                summary_kwargs["extra_body"] = summary_extra_body

            if agent.api_mode == "anthropic_messages":
                _tsum = agent._get_transport()
                _ant_kw = _tsum.build_kwargs(model=agent.model, messages=api_messages, tools=None,
                               max_tokens=agent.max_tokens, reasoning_config=agent.reasoning_config,
                               is_oauth=agent._is_anthropic_oauth,
                               preserve_dots=agent._anthropic_preserve_dots())
                summary_response = agent._anthropic_messages_create(_ant_kw)
                _summary_result = _tsum.normalize_response(summary_response, strip_tool_prefix=agent._is_anthropic_oauth)
                final_response = (_summary_result.content or "").strip()
            else:
                summary_response = agent._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)
                _summary_result = agent._get_transport().normalize_response(summary_response)
                final_response = (_summary_result.content or "").strip()

        if final_response:
            if "<think>" in final_response:
                final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
            if final_response:
                messages.append({"role": "assistant", "content": final_response})
            else:
                final_response = "I reached the iteration limit and couldn't generate a summary."
        else:
            # Retry summary generation
            if agent.api_mode == "codex_responses":
                codex_kwargs = agent._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                retry_response = agent._run_codex_stream(codex_kwargs)
                _ct_retry = agent._get_transport()
                _cnr_retry = _ct_retry.normalize_response(retry_response)
                final_response = (_cnr_retry.content or "").strip()
            elif agent.api_mode == "anthropic_messages":
                _tretry = agent._get_transport()
                _ant_kw2 = _tretry.build_kwargs(model=agent.model, messages=api_messages, tools=None,
                                is_oauth=agent._is_anthropic_oauth,
                                max_tokens=agent.max_tokens, reasoning_config=agent.reasoning_config,
                                preserve_dots=agent._anthropic_preserve_dots())
                retry_response = agent._anthropic_messages_create(_ant_kw2)
                _retry_result = _tretry.normalize_response(retry_response, strip_tool_prefix=agent._is_anthropic_oauth)
                final_response = (_retry_result.content or "").strip()
            else:
                summary_kwargs = {
                    "model": agent.model,
                    "messages": api_messages,
                }
                if _summary_temperature is not None:
                    summary_kwargs["temperature"] = _summary_temperature
                if agent.max_tokens is not None:
                    summary_kwargs.update(agent._max_tokens_param(agent.max_tokens))
                if _lm_reasoning_effort is not None:
                    summary_kwargs["reasoning_effort"] = _lm_reasoning_effort
                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                summary_response = agent._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)
                _retry_result = agent._get_transport().normalize_response(summary_response)
                final_response = (_retry_result.content or "").strip()

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                final_response = "I reached the iteration limit and couldn't generate a summary."

    except Exception as e:
        logging.warning(f"Failed to get summary response: {e}")
        final_response = f"I reached the maximum iterations ({agent.max_iterations}) but couldn't summarize. Error: {str(e)}"

    return final_response



def cleanup_task_resources(agent, task_id: str) -> None:
    """Clean up VM and browser resources for a given task.

    Skips ``cleanup_vm`` when the active terminal environment is marked
    persistent (``persistent_filesystem=True``) so that long-lived sandbox
    containers survive between turns. The idle reaper in
    ``terminal_tool._cleanup_inactive_envs`` still tears them down once
    ``terminal.lifetime_seconds`` is exceeded. Non-persistent backends are
    torn down per-turn as before to prevent resource leakage (the original
    intent of this hook for the Morph backend, see commit fbd3a2fd).
    """
    try:
        if is_persistent_env(task_id):
            if agent.verbose_logging:
                logging.debug(
                    f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                    f"idle reaper will handle it."
                )
        else:
            _ra().cleanup_vm(task_id)
    except Exception as e:
        if agent.verbose_logging:
            logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
    try:
        _ra().cleanup_browser(task_id)
    except Exception as e:
        if agent.verbose_logging:
            logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")




def interruptible_streaming_api_call(agent, api_kwargs: dict, *, on_first_delta=None):
    """Streaming variant of _interruptible_api_call for real-time token delivery.

    Handles all three api_modes:
    - chat_completions: stream=True on OpenAI-compatible endpoints
    - anthropic_messages: client.messages.stream() via Anthropic SDK
    - codex_responses: delegates to _run_codex_stream (already streaming)

    Fires stream_delta_callback and _stream_callback for each text token.
    Tool-call turns suppress the callback — only text-only final responses
    stream to the consumer.  Returns a SimpleNamespace that mimics the
    non-streaming response shape so the rest of the agent loop is unchanged.

    Falls back to _interruptible_api_call on provider errors indicating
    streaming is not supported.
    """
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted before streaming API call")

    if agent.api_mode == "codex_responses":
        # Codex streams internally via _run_codex_stream. The main dispatch
        # in _interruptible_api_call already calls it; we just need to
        # ensure on_first_delta reaches it. Store it on the instance
        # temporarily so _run_codex_stream can pick it up.
        agent._codex_on_first_delta = on_first_delta
        try:
            return agent._interruptible_api_call(api_kwargs)
        finally:
            agent._codex_on_first_delta = None

    # Bedrock Converse uses boto3's converse_stream() with real-time delta
    # callbacks — same UX as Anthropic and chat_completions streaming.
    if agent.api_mode == "bedrock_converse":
        result = {"response": None, "error": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}

        def _fire_first():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _bedrock_call():
            try:
                from agent.bedrock_adapter import (
                    _get_bedrock_runtime_client,
                    invalidate_runtime_client,
                    is_stale_connection_error,
                    stream_converse_with_callbacks,
                )
                region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                api_kwargs.pop("__bedrock_converse__", None)
                client = _get_bedrock_runtime_client(region)
                try:
                    raw_response = client.converse_stream(**api_kwargs)
                except Exception as _bedrock_exc:
                    # Evict the cached client on stale-connection failures
                    # so the outer retry loop builds a fresh client/pool.
                    if is_stale_connection_error(_bedrock_exc):
                        invalidate_runtime_client(region)
                    raise

                def _on_text(text):
                    _fire_first()
                    agent._fire_stream_delta(text)
                    deltas_were_sent["yes"] = True

                def _on_tool(name):
                    _fire_first()
                    agent._fire_tool_gen_started(name)

                def _on_reasoning(text):
                    _fire_first()
                    agent._fire_reasoning_delta(text)

                result["response"] = stream_converse_with_callbacks(
                    raw_response,
                    on_text_delta=_on_text if agent._has_stream_consumers() else None,
                    on_tool_start=_on_tool,
                    on_reasoning_delta=_on_reasoning if agent.reasoning_callback or agent.stream_delta_callback else None,
                    on_interrupt_check=lambda: agent._interrupt_requested,
                )
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_bedrock_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if agent._interrupt_requested:
                raise InterruptedError("Agent interrupted during Bedrock API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    result = {"response": None, "error": None, "partial_tool_names": []}
    request_client_holder = {"client": None, "diag": None}
    first_delta_fired = {"done": False}
    deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)
    # Wall-clock timestamp of the last real streaming chunk.  The outer
    # poll loop uses this to detect stale connections that keep receiving
    # SSE keep-alive pings but no actual data.
    last_chunk_time = {"t": time.time()}

    def _fire_first_delta():
        if not first_delta_fired["done"] and on_first_delta:
            first_delta_fired["done"] = True
            try:
                on_first_delta()
            except Exception:
                pass

    def _call_chat_completions():
        """Stream a chat completions response."""
        import httpx as _httpx
        # Per-provider / per-model request_timeout_seconds (from config.yaml)
        # wins over the HERMES_API_TIMEOUT env default if the user set it.
        _provider_timeout_cfg = get_provider_request_timeout(agent.provider, agent.model)
        _base_timeout = (
            _provider_timeout_cfg
            if _provider_timeout_cfg is not None
            else float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
        )
        # Read timeout: config wins here too.  Otherwise use
        # HERMES_STREAM_READ_TIMEOUT (default 120s) for cloud providers.
        if _provider_timeout_cfg is not None:
            _stream_read_timeout = _provider_timeout_cfg
        else:
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            # Local providers (Ollama, llama.cpp, vLLM) can take minutes for
            # prefill on large contexts before producing the first token.
            # Auto-increase the httpx read timeout unless the user explicitly
            # overrode HERMES_STREAM_READ_TIMEOUT.
            if _stream_read_timeout == 120.0 and agent.base_url and is_local_endpoint(agent.base_url):
                _stream_read_timeout = _base_timeout
                logger.debug(
                    "Local provider detected (%s) — stream read timeout raised to %.0fs",
                    agent.base_url, _stream_read_timeout,
                )
        stream_kwargs = {
            **api_kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": _httpx.Timeout(
                connect=30.0,
                read=_stream_read_timeout,
                write=_base_timeout,
                pool=30.0,
            ),
        }
        request_client_holder["client"] = agent._create_request_openai_client(
            reason="chat_completion_stream_request",
            api_kwargs=stream_kwargs,
        )
        # Reset stale-stream timer so the detector measures from this
        # attempt's start, not a previous attempt's last chunk.
        last_chunk_time["t"] = time.time()
        agent._touch_activity("waiting for provider response (streaming)")
        # Initialize per-attempt stream diagnostics so the retry block can
        # reach for them after the stream dies.  Lives on
        # ``request_client_holder["diag"]`` for closure access.
        _diag = agent._stream_diag_init()
        request_client_holder["diag"] = _diag
        stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

        # Capture rate limit headers from the initial HTTP response.
        # The OpenAI SDK Stream object exposes the underlying httpx
        # response via .response before any chunks are consumed.
        agent._capture_rate_limits(getattr(stream, "response", None))
        # Snapshot diagnostic headers (cf-ray, x-openrouter-provider, etc.)
        # so they survive even when the stream dies before any chunk
        # arrives.  Best-effort; never raises.
        agent._stream_diag_capture_response(_diag, getattr(stream, "response", None))

        # Log OpenRouter response cache status when present.
        agent._check_openrouter_cache_status(getattr(stream, "response", None))

        content_parts: list = []
        tool_calls_acc: dict = {}
        tool_gen_notified: set = set()
        # Ollama-compatible endpoints reuse index 0 for every tool call
        # in a parallel batch, distinguishing them only by id.  Track
        # the last seen id per raw index so we can detect a new tool
        # call starting at the same index and redirect it to a fresh slot.
        _last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
        _active_slot_by_idx: dict = {}  # raw_index -> current slot in tool_calls_acc
        finish_reason = None
        model_name = None
        role = "assistant"
        reasoning_parts: list = []
        usage_obj = None
        for chunk in stream:
            last_chunk_time["t"] = time.time()
            agent._touch_activity("receiving stream response")

            # Update per-attempt diagnostic counters.  Best-effort —
            # failures are swallowed so the streaming hot path is never
            # interrupted by diagnostic accounting.
            try:
                _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                if _diag.get("first_chunk_at") is None:
                    _diag["first_chunk_at"] = last_chunk_time["t"]
                # Approximate byte size from the chunk's repr — exact wire
                # bytes aren't exposed by the SDK, but len(repr(chunk)) is
                # a stable proxy for "how much content arrived" that
                # survives stub provider differences.
                try:
                    _diag["bytes"] = int(_diag.get("bytes", 0)) + len(repr(chunk))
                except Exception:
                    pass
            except Exception:
                pass

            if agent._interrupt_requested:
                break

            if not chunk.choices:
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model
                # Usage comes in the final chunk with empty choices
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage
                continue

            delta = chunk.choices[0].delta
            if hasattr(chunk, "model") and chunk.model:
                model_name = chunk.model

            # Accumulate reasoning content
            reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
                _fire_first_delta()
                agent._fire_reasoning_delta(reasoning_text)

            # Accumulate text content — fire callback only when no tool calls
            if delta and delta.content:
                content_parts.append(delta.content)
                if not tool_calls_acc:
                    _fire_first_delta()
                    agent._fire_stream_delta(delta.content)
                    deltas_were_sent["yes"] = True
                # Tool calls suppress regular content streaming (avoids
                # displaying chatty "I'll use the tool..." text alongside
                # tool calls).  But reasoning tags embedded in suppressed
                # content should still reach the display — otherwise the
                # reasoning box only appears as a post-response fallback,
                # rendering it confusingly after the already-streamed
                # response.  Route suppressed content through the stream
                # delta callback so its tag extraction can fire the
                # reasoning display.  Non-reasoning text is harmlessly
                # suppressed by the CLI's _stream_delta when the stream
                # box is already closed (tool boundary flush).
                elif agent.stream_delta_callback:
                    try:
                        agent.stream_delta_callback(delta.content)
                        agent._record_streamed_assistant_text(delta.content)
                    except Exception:
                        pass

            # Accumulate tool call deltas — notify display on first name
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    raw_idx = tc_delta.index if tc_delta.index is not None else 0
                    delta_id = tc_delta.id or ""

                    # Ollama fix: detect a new tool call reusing the same
                    # raw index (different id) and redirect to a fresh slot.
                    if raw_idx not in _active_slot_by_idx:
                        _active_slot_by_idx[raw_idx] = raw_idx
                    if (
                        delta_id
                        and raw_idx in _last_id_at_idx
                        and delta_id != _last_id_at_idx[raw_idx]
                    ):
                        new_slot = max(tool_calls_acc, default=-1) + 1
                        _active_slot_by_idx[raw_idx] = new_slot
                    if delta_id:
                        _last_id_at_idx[raw_idx] = delta_id
                    idx = _active_slot_by_idx[raw_idx]

                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                            "extra_content": None,
                        }
                    entry = tool_calls_acc[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            # Use assignment, not +=.  Function names are
                            # atomic identifiers delivered complete in the
                            # first chunk (OpenAI spec).  Some providers
                            # (MiniMax M2.7 via NVIDIA NIM) resend the full
                            # name in every chunk; concatenation would
                            # produce "read_fileread_file".  Assignment
                            # (matching the OpenAI Node SDK / LiteLLM /
                            # Vercel AI patterns) is immune to this.
                            entry["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["function"]["arguments"] += tc_delta.function.arguments
                    extra = getattr(tc_delta, "extra_content", None)
                    if extra is None and hasattr(tc_delta, "model_extra"):
                        extra = (tc_delta.model_extra or {}).get("extra_content")
                    if extra is not None:
                        if hasattr(extra, "model_dump"):
                            extra = extra.model_dump()
                        entry["extra_content"] = extra
                    # Fire once per tool when the full name is available
                    name = entry["function"]["name"]
                    if name and idx not in tool_gen_notified:
                        tool_gen_notified.add(idx)
                        _fire_first_delta()
                        agent._fire_tool_gen_started(name)
                        # Record the partial tool-call name so the outer
                        # stub-builder can surface a user-visible warning
                        # if streaming dies before this tool's arguments
                        # are fully delivered.  Without this, a stall
                        # during tool-call JSON generation lets the stub
                        # at line ~6107 return `tool_calls=None`, silently
                        # discarding the attempted action.
                        result["partial_tool_names"].append(name)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # Usage in the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage_obj = chunk.usage

        # Build mock response matching non-streaming shape
        full_content = "".join(content_parts) or None
        mock_tool_calls = None
        has_truncated_tool_args = False
        if tool_calls_acc:
            mock_tool_calls = []
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                arguments = tc["function"]["arguments"]
                tool_name = tc["function"]["name"] or "?"
                if arguments and arguments.strip():
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        # Attempt repair before flagging as truncated.
                        # Models like GLM-5.1 via Ollama produce trailing
                        # commas, unclosed brackets, Python None, etc.
                        # Without repair, these hit the truncation handler
                        # and kill the session.  _repair_tool_call_arguments
                        # returns "{}" for unrepairable args, which is far
                        # better than a crashed session.
                        repaired = _repair_tool_call_arguments(arguments, tool_name)
                        if repaired != "{}":
                            # Successfully repaired — use the fixed args
                            arguments = repaired
                        else:
                            # Unrepairable — flag for truncation handling
                            has_truncated_tool_args = True
                mock_tool_calls.append(SimpleNamespace(
                    id=tc["id"],
                    type=tc["type"],
                    extra_content=tc.get("extra_content"),
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=arguments,
                    ),
                ))

        effective_finish_reason = finish_reason or "stop"
        if has_truncated_tool_args:
            effective_finish_reason = "length"

        full_reasoning = "".join(reasoning_parts) or None
        mock_message = SimpleNamespace(
            role=role,
            content=full_content,
            tool_calls=mock_tool_calls,
            reasoning_content=full_reasoning,
        )
        mock_choice = SimpleNamespace(
            index=0,
            message=mock_message,
            finish_reason=effective_finish_reason,
        )
        return SimpleNamespace(
            id="stream-" + str(uuid.uuid4()),
            model=model_name,
            choices=[mock_choice],
            usage=usage_obj,
        )

    def _call_anthropic():
        """Stream an Anthropic Messages API response.

        Fires delta callbacks for real-time token delivery, but returns
        the native Anthropic Message object from get_final_message() so
        the rest of the agent loop (validation, tool extraction, etc.)
        works unchanged.
        """
        has_tool_use = False

        # Reset stale-stream timer for this attempt
        last_chunk_time["t"] = time.time()
        # Per-attempt diagnostic dict for the retry block to consume.
        _diag = agent._stream_diag_init()
        request_client_holder["diag"] = _diag
        # Use the Anthropic SDK's streaming context manager
        with agent._anthropic_client.messages.stream(**api_kwargs) as stream:
            # The Anthropic SDK exposes the raw httpx response on
            # ``stream.response``.  Snapshot diagnostic headers
            # immediately so they survive a stream that dies before the
            # first event.
            try:
                agent._stream_diag_capture_response(
                    _diag, getattr(stream, "response", None)
                )
            except Exception:
                pass
            for event in stream:
                # Update stale-stream timer on every event so the
                # outer poll loop knows data is flowing.  Without
                # this, the detector kills healthy long-running
                # Opus streams after 180 s even when events are
                # actively arriving (the chat_completions path
                # already does this at the top of its chunk loop).
                last_chunk_time["t"] = time.time()
                agent._touch_activity("receiving stream response")

                # Update per-attempt diagnostic counters (best-effort).
                try:
                    _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                    if _diag.get("first_chunk_at") is None:
                        _diag["first_chunk_at"] = last_chunk_time["t"]
                    try:
                        _diag["bytes"] = int(_diag.get("bytes", 0)) + len(repr(event))
                    except Exception:
                        pass
                except Exception:
                    pass

                if agent._interrupt_requested:
                    break

                event_type = getattr(event, "type", None)

                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        has_tool_use = True
                        tool_name = getattr(block, "name", None)
                        if tool_name:
                            _fire_first_delta()
                            agent._fire_tool_gen_started(tool_name)

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text and not has_tool_use:
                                _fire_first_delta()
                                agent._fire_stream_delta(text)
                                deltas_were_sent["yes"] = True
                        elif delta_type == "thinking_delta":
                            thinking_text = getattr(delta, "thinking", "")
                            if thinking_text:
                                _fire_first_delta()
                                agent._fire_reasoning_delta(thinking_text)

            # Return the native Anthropic Message for downstream processing
            return stream.get_final_message()

    def _call():
        import httpx as _httpx

        _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

        try:
            for _stream_attempt in range(_max_stream_retries + 1):
                # Check for interrupt before each retry attempt.  Without
                # this, /stop closes the HTTP connection (outer poll loop),
                # but the retry loop opens a FRESH connection — negating the
                # interrupt entirely.  On slow providers (ollama-cloud) each
                # retry can block for the full stream-read timeout (120s+),
                # causing multi-minute delays between /stop and response.
                if agent._interrupt_requested:
                    raise InterruptedError("Agent interrupted before stream retry")
                try:
                    if agent.api_mode == "anthropic_messages":
                        agent._try_refresh_anthropic_client_credentials()
                        result["response"] = _call_anthropic()
                    else:
                        result["response"] = _call_chat_completions()
                    return  # success
                except Exception as e:
                    _is_timeout = isinstance(
                        e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                    )
                    _is_conn_err = isinstance(
                        e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                    )
                    _is_stream_parse_err = agent._is_provider_stream_parse_error(e)

                    # If the stream died AFTER some tokens were delivered:
                    # normally we don't retry (the user already saw text,
                    # retrying would duplicate it).  BUT: if a tool call
                    # was in-flight when the stream died, silently aborting
                    # discards the tool call entirely.  In that case we
                    # prefer to retry — the user sees a brief
                    # "reconnecting" marker + duplicated preamble text,
                    # which is strictly better than a failed action with
                    # a "retry manually" message.  Limit this to transient
                    # connection errors (Clawdbot-style narrow gate): no
                    # tool has executed yet within this API call, so
                    # silent retry is safe wrt side-effects.
                    if deltas_were_sent["yes"]:
                        _partial_tool_in_flight = bool(
                            result.get("partial_tool_names")
                        )
                        _is_sse_conn_err_preview = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                _err_lower_preview = str(e).lower()
                                _SSE_PREVIEW_PHRASES = (
                                    "connection lost",
                                    "connection reset",
                                    "connection closed",
                                    "connection terminated",
                                    "network error",
                                    "network connection",
                                    "terminated",
                                    "peer closed",
                                    "broken pipe",
                                    "upstream connect error",
                                )
                                _is_sse_conn_err_preview = any(
                                    phrase in _err_lower_preview
                                    for phrase in _SSE_PREVIEW_PHRASES
                                )
                        _is_transient = (
                            _is_timeout
                            or _is_conn_err
                            or _is_sse_conn_err_preview
                            or _is_stream_parse_err
                        )
                        _can_silent_retry = (
                            _partial_tool_in_flight
                            and _is_transient
                            and _stream_attempt < _max_stream_retries
                        )
                        if not _can_silent_retry:
                            # Either no tool call was in-flight (so the
                            # turn was a pure text response — current
                            # stub-with-recovered-text behaviour is
                            # correct), or retries are exhausted, or the
                            # error isn't transient.  Fall through to the
                            # stub path.
                            logger.warning(
                                "Streaming failed after partial delivery, not retrying: %s", e
                            )
                            result["error"] = e
                            return
                        # Tool call was in-flight AND error is transient:
                        # retry silently.  Clear per-attempt state so the
                        # next stream starts clean.  Fire a "reconnecting"
                        # marker so the user sees why the preamble is
                        # about to be re-streamed.  Structured WARNING is
                        # emitted by ``_emit_stream_drop`` below; no
                        # additional INFO line needed.
                        try:
                            agent._fire_stream_delta(
                                "\n\n⚠ Connection dropped mid tool-call; "
                                "reconnecting…\n\n"
                            )
                        except Exception:
                            pass
                        # Reset the streamed-text buffer so the retry's
                        # fresh preamble doesn't get double-recorded in
                        # _current_streamed_assistant_text (which would
                        # pollute the interim-visible-text comparison).
                        try:
                            agent._reset_stream_delivery_tracking()
                        except Exception:
                            pass
                        # Reset in-memory accumulators so the next
                        # attempt's chunks don't concat onto the dead
                        # stream's partial JSON.
                        result["partial_tool_names"] = []
                        deltas_were_sent["yes"] = False
                        first_delta_fired["done"] = False
                        agent._emit_stream_drop(
                            error=e,
                            attempt=_stream_attempt + 2,
                            max_attempts=_max_stream_retries + 1,
                            mid_tool_call=True,
                            diag=request_client_holder.get("diag"),
                        )
                        stale = request_client_holder.get("client")
                        if stale is not None:
                            agent._close_request_openai_client(
                                stale, reason="stream_mid_tool_retry_cleanup"
                            )
                            request_client_holder["client"] = None
                        try:
                            agent._replace_primary_openai_client(
                                reason="stream_mid_tool_retry_pool_cleanup"
                            )
                        except Exception:
                            pass
                        continue

                    # SSE error events from proxies (e.g. OpenRouter sends
                    # {"error":{"message":"Network connection lost."}}) are
                    # raised as APIError by the OpenAI SDK.  These are
                    # semantically identical to httpx connection drops —
                    # the upstream stream died — and should be retried with
                    # a fresh connection.  Distinguish from HTTP errors:
                    # APIError from SSE has no status_code, while
                    # APIStatusError (4xx/5xx) always has one.
                    _is_sse_conn_err = False
                    if not _is_timeout and not _is_conn_err:
                        from openai import APIError as _APIError
                        if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                            _err_lower_sse = str(e).lower()
                            _SSE_CONN_PHRASES = (
                                "connection lost",
                                "connection reset",
                                "connection closed",
                                "connection terminated",
                                "network error",
                                "network connection",
                                "terminated",
                                "peer closed",
                                "broken pipe",
                                "upstream connect error",
                            )
                            _is_sse_conn_err = any(
                                phrase in _err_lower_sse
                                for phrase in _SSE_CONN_PHRASES
                            )

                    if _is_timeout or _is_conn_err or _is_sse_conn_err or _is_stream_parse_err:
                        # Transient network / timeout error. Retry the
                        # streaming request with a fresh connection first.
                        if _stream_attempt < _max_stream_retries:
                            agent._emit_stream_drop(
                                error=e,
                                attempt=_stream_attempt + 2,
                                max_attempts=_max_stream_retries + 1,
                                mid_tool_call=False,
                                diag=request_client_holder.get("diag"),
                            )
                            # Close the stale request client before retry
                            stale = request_client_holder.get("client")
                            if stale is not None:
                                agent._close_request_openai_client(
                                    stale, reason="stream_retry_cleanup"
                                )
                                request_client_holder["client"] = None
                            # Also rebuild the primary client to purge
                            # any dead connections from the pool.
                            try:
                                agent._replace_primary_openai_client(
                                    reason="stream_retry_pool_cleanup"
                                )
                            except Exception:
                                pass
                            continue
                        # Retries exhausted. Log the final failure with
                        # full diagnostic detail (chain, headers,
                        # bytes/elapsed) via the same helper used for
                        # mid-flight retries — subagent lines get the
                        # ``[subagent-N]`` log_prefix so the parent can
                        # attribute them.
                        agent._log_stream_retry(
                            kind="exhausted",
                            error=e,
                            attempt=_max_stream_retries + 1,
                            max_attempts=_max_stream_retries + 1,
                            mid_tool_call=False,
                            diag=request_client_holder.get("diag"),
                        )
                        agent._emit_status(
                            "❌ Provider returned malformed streaming data after "
                            f"{_max_stream_retries + 1} attempts. "
                            "The provider may be experiencing issues — "
                            "try again in a moment."
                            if _is_stream_parse_err else
                            "❌ Connection to provider failed after "
                            f"{_max_stream_retries + 1} attempts. "
                            "The provider may be experiencing issues — "
                            "try again in a moment."
                        )
                    else:
                        _err_lower = str(e).lower()
                        _is_stream_unsupported = (
                            "stream" in _err_lower
                            and "not supported" in _err_lower
                        )
                        if _is_stream_unsupported:
                            agent._disable_streaming = True
                            agent._safe_print(
                                "\n⚠  Streaming is not supported for this "
                                "model/provider. Switching to non-streaming.\n"
                                "   To avoid this delay, set display.streaming: false "
                                "in config.yaml\n"
                            )
                        logger.info(
                            "Streaming failed before delivery: %s",
                            e,
                        )

                    # Propagate the error to the main retry loop instead of
                    # falling back to non-streaming inline.  The main loop has
                    # richer recovery: credential rotation, provider fallback,
                    # backoff, and — for "stream not supported" — will switch
                    # to non-streaming on the next attempt via _disable_streaming.
                    result["error"] = e
                    return
        except InterruptedError as e:
            # The interrupt may be noticed inside the worker thread before
            # the polling loop sees it. Surface it through the normal result
            # channel so callers never miss a fast pre-retry interrupt.
            result["error"] = e
            return
        finally:
            request_client = request_client_holder.get("client")
            if request_client is not None:
                agent._close_request_openai_client(request_client, reason="stream_request_complete")

    _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
    # Local providers (Ollama, oMLX, llama-cpp) can take 300+ seconds
    # for prefill on large contexts.  Disable the stale detector unless
    # the user explicitly set HERMES_STREAM_STALE_TIMEOUT.
    if _stream_stale_timeout_base == 180.0 and agent.base_url and is_local_endpoint(agent.base_url):
        _stream_stale_timeout = float("inf")
        logger.debug("Local provider detected (%s) — stale stream timeout disabled", agent.base_url)
    else:
        # Scale the stale timeout for large contexts: slow models (like Opus)
        # can legitimately think for minutes before producing the first token
        # when the context is large.  Without this, the stale detector kills
        # healthy connections during the model's thinking phase, producing
        # spurious RemoteProtocolError ("peer closed connection").
        _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
        if _est_tokens > 100_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
        elif _est_tokens > 50_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
        else:
            _stream_stale_timeout = _stream_stale_timeout_base

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    _last_heartbeat = time.time()
    _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
    while t.is_alive():
        t.join(timeout=0.3)

        # Periodic heartbeat: touch the agent's activity tracker so the
        # gateway's inactivity monitor knows we're alive while waiting
        # for stream chunks.  Without this, long thinking pauses (e.g.
        # reasoning models) or slow prefill on local providers (Ollama)
        # trigger false inactivity timeouts.  The _call thread touches
        # activity on each chunk, but the gap between API call start
        # and first chunk can exceed the gateway timeout — especially
        # when the stale-stream timeout is disabled (local providers).
        _hb_now = time.time()
        if _hb_now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
            _last_heartbeat = _hb_now
            _waiting_secs = int(_hb_now - last_chunk_time["t"])
            agent._touch_activity(
                f"waiting for stream response ({_waiting_secs}s, no chunks yet)"
            )

        # Detect stale streams: connections kept alive by SSE pings
        # but delivering no real chunks.  Kill the client so the
        # inner retry loop can start a fresh connection.
        _stale_elapsed = time.time() - last_chunk_time["t"]
        if _stale_elapsed > _stream_stale_timeout:
            _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            logger.warning(
                "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                "model=%s context=~%s tokens. Killing connection.",
                _stale_elapsed, _stream_stale_timeout,
                api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
            )
            agent._emit_status(
                f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                f"(model: {api_kwargs.get('model', 'unknown')}, "
                f"context: ~{_est_ctx:,} tokens). "
                f"Reconnecting..."
            )
            try:
                rc = request_client_holder.get("client")
                if rc is not None:
                    agent._close_request_openai_client(rc, reason="stale_stream_kill")
            except Exception:
                pass
            # Rebuild the primary client too — its connection pool
            # may hold dead sockets from the same provider outage.
            try:
                agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
            except Exception:
                pass
            # Reset the timer so we don't kill repeatedly while
            # the inner thread processes the closure.
            last_chunk_time["t"] = time.time()
            agent._touch_activity(
                f"stale stream detected after {int(_stale_elapsed)}s, reconnecting"
            )

        if agent._interrupt_requested:
            try:
                if agent.api_mode == "anthropic_messages":
                    agent._anthropic_client.close()
                    agent._rebuild_anthropic_client()
                else:
                    request_client = request_client_holder.get("client")
                    if request_client is not None:
                        agent._close_request_openai_client(request_client, reason="stream_interrupt_abort")
            except Exception:
                pass
            raise InterruptedError("Agent interrupted during streaming API call")
    if result["error"] is not None:
        if deltas_were_sent["yes"]:
            # Streaming failed AFTER some tokens were already delivered to
            # the platform.  Re-raising would let the outer retry loop make
            # a new API call, creating a duplicate message.  Return a
            # partial "stop" response instead so the outer loop treats this
            # turn as complete (no retry, no fallback).
            # Recover whatever content was already streamed to the user.
            # _current_streamed_assistant_text accumulates text fired
            # through _fire_stream_delta, so it has exactly what the
            # user saw before the connection died.
            _partial_text = (
                getattr(agent, "_current_streamed_assistant_text", "") or ""
            ).strip() or None

            # If the stream died while the model was emitting a tool call,
            # the stub below will silently set `tool_calls=None` and the
            # agent loop will treat the turn as complete — the attempted
            # action is lost with no user-facing signal.  Append a
            # human-visible warning to the stub content so (a) the user
            # knows something failed, and (b) the next turn's model sees
            # in conversation history what was attempted and can retry.
            _partial_names = list(result.get("partial_tool_names") or [])
            if _partial_names:
                _name_str = ", ".join(_partial_names[:3])
                if len(_partial_names) > 3:
                    _name_str += f", +{len(_partial_names) - 3} more"
                _warn = (
                    f"\n\n⚠ Stream stalled mid tool-call "
                    f"({_name_str}); the action was not executed. "
                    f"Ask me to retry if you want to continue."
                )
                _partial_text = (_partial_text or "") + _warn
                # Also fire as a streaming delta so the user sees it now
                # instead of only in the persisted transcript.
                try:
                    agent._fire_stream_delta(_warn)
                except Exception:
                    pass
                logger.warning(
                    "Partial stream dropped tool call(s) %s after %s chars "
                    "of text; surfaced warning to user: %s",
                    _partial_names, len(_partial_text or ""), result["error"],
                )
            else:
                logger.warning(
                    "Partial stream delivered before error; returning stub "
                    "response with %s chars of recovered content to prevent "
                    "duplicate messages: %s",
                    len(_partial_text or ""),
                    result["error"],
                )
            _stub_msg = SimpleNamespace(
                role="assistant", content=_partial_text, tool_calls=None,
                reasoning_content=None,
            )
            return SimpleNamespace(
                id="partial-stream-stub",
                model=getattr(agent, "model", "unknown"),
                choices=[SimpleNamespace(
                    index=0, message=_stub_msg, finish_reason="stop",
                )],
                usage=None,
            )
        raise result["error"]
    return result["response"]

# ── Provider fallback ──────────────────────────────────────────────────



__all__ = [
    "interruptible_api_call",
    "build_api_kwargs",
    "build_assistant_message",
    "try_activate_fallback",
    "handle_max_iterations",
    "cleanup_task_resources",
    "interruptible_streaming_api_call",
]
