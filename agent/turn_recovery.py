"""Recovery-branch handlers for the conversation turn's inner retry loop.

When the model call raises, one-shot recovery chains run before the generic retry/backoff
path. Handlers return ``True`` (request repaired in place; loop ``continue``s with the same
``retry_count``) or ``False`` (fall through). Guards live on ``TurnRetryState``; handlers
mutate ``agent`` / ``messages`` / ``api_messages`` in place. Logger name stays
``agent.conversation_loop`` (caplog pins); that module is only imported lazily (cycle + patch sites).
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.conversation_compression import COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE
from agent.model_metadata import is_output_cap_error, parse_available_output_tokens_from_error
from agent.retry_utils import is_zai_coding_overload_error, zai_coding_overload_retry_ceiling
from agent.error_classifier import FailoverReason
from agent.message_sanitization import (
    _looks_like_image_content_rejection, _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates, _sanitize_structure_non_ascii, _sanitize_structure_surrogates,
    _sanitize_tools_non_ascii, _strip_images_from_messages, _strip_non_ascii,
    close_interrupted_tool_sequence,
)
from agent.thinking_timeout_guidance import build_thinking_timeout_guidance, is_thinking_timeout
from agent.turn_failure_copy import (
    CONTENT_POLICY_NEXT_STEPS, content_policy_copy, exhausted_copy, nonretryable_copy, provider_label_for,
    site_copy, stamp_failure,
)
from agent.turn_retry_state import TurnRetryState
from hermes_constants import display_hermes_home
from utils import base_url_host_matches

logger = logging.getLogger("agent.conversation_loop")


def _vlines(agent: Any, *lines: str) -> None:
    """Force-``_vprint`` each line prefixed with ``agent.log_prefix``."""
    for line in lines:
        agent._vprint(f"{agent.log_prefix}{line}", force=True, diagnostic=True)


def _plines(agent: Any, *lines: str) -> None:
    """``print`` each line prefixed with ``agent.log_prefix``."""
    from gateway.warning_notifications import render_notification
    render_notification(
        lambda: [print(f"{agent.log_prefix}{line}") for line in lines],
        platform=getattr(agent, "_notification_platform", getattr(agent, "platform", "cli")),
        user_config=getattr(agent, "_notification_config", None))


def _blines(agent: Any, *lines: str) -> None:
    """``_buffer_vprint`` each line (surfaces only if every retry+fallback exhausts)."""
    for line in lines:
        agent._buffer_vprint(line)


def _image_error_max_dimension(error: Exception) -> Optional[int]:
    """Extract a provider-reported image dimension ceiling, if present."""
    parts = []
    for value in (error, getattr(error, "message", None), getattr(error, "body", None)):
        if value:
            try:
                parts.append(str(value))
            except Exception:
                pass
    text = " ".join(parts).lower()
    # OpenAI Codex Responses reports a tile-patch budget (ceil(w/32)×ceil(h/32))
    # instead of a pixel ceiling. A square image is the worst case for the budget,
    # so a per-side cap of isqrt(limit)*32 px keeps isqrt(limit)² ≤ limit — for the
    # 30000-patch ceiling that is 5536 px. Without this the caller falls back to
    # 8000 px and a 6000 px image that already exceeds the budget is skipped (#106337).
    if "patches after processing" in text:
        match = re.search(r"exceeding the limit of\s*(\d{2,7})", text)
        if not match:
            return None
        max_dimension = math.isqrt(int(match.group(1))) * 32
        return max_dimension if 512 <= max_dimension <= 8000 else None
    if "image" not in text or "dimension" not in text or "max allowed size" not in text:
        return None
    match = re.search(r"max allowed size(?:\s+for [^:]+)?:\s*(\d{3,5})\s*pixels?", text)
    if not match:
        return None
    try:
        max_dimension = int(match.group(1))
    except ValueError:
        return None
    return max_dimension if 512 <= max_dimension <= 8000 else None


def _try_refresh_nous_paid_entitlement_credentials(agent) -> bool:
    """Refresh Nous runtime credentials after a fresh paid-entitlement check."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        if get_nous_portal_account_info(force_fresh=True).paid_service_access is not True:
            return False
        return agent._try_refresh_nous_client_credentials(force=True)
    except Exception:
        return False


def _recover_unicode_encode_error(
    agent: Any, api_error: Exception, messages: List[Dict[str, Any]], api_messages: Any,
    api_kwargs: Any, active_system_prompt: Any,
) -> Tuple[bool, Any]:
    """UnicodeEncodeError recovery: lone surrogates (clipboard paste) first, then an ASCII
    codec under a non-UTF-8 locale. Sanitizes in place; bounded by the caller's
    ``_unicode_sanitization_passes < 2`` guard (surrogate strip, then ASCII-only)."""
    _err_str = str(api_error).lower()
    _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
    # utf-8 refusing U+D800..U+DFFF ("surrogates not allowed").
    _is_surrogate_error = "surrogate" in _err_str or ("'utf-8'" in _err_str and not _is_ascii_codec)
    # Sanitize `messages` AND `api_messages` (may carry reasoning_content/reasoning_details),
    # plus `api_kwargs` and `prefill_messages`. Every sanitizer runs (no short-circuit).
    _prefill = getattr(agent, "prefill_messages", None)
    _surrogates_found = _sanitize_messages_surrogates(messages)
    _surrogates_found |= isinstance(api_messages, list) and _sanitize_messages_surrogates(api_messages)
    _surrogates_found |= isinstance(api_kwargs, dict) and _sanitize_structure_surrogates(api_kwargs)
    _surrogates_found |= isinstance(_prefill, list) and _sanitize_messages_surrogates(_prefill)
    # Gate the retry on the error type, not on whether anything was found — a new
    # transformed field could slip through.
    if _surrogates_found or _is_surrogate_error:
        agent._unicode_sanitization_passes += 1
        agent._buffer_vprint(
            "⚠️  Stripped invalid surrogate characters from messages. Retrying..."
            if _surrogates_found else
            "⚠️  Surrogate encoding error — retrying after full-payload sanitization..."
        )
        return True, active_system_prompt
    if not _is_ascii_codec:
        return False, active_system_prompt

    agent._force_ascii_payload = True
    # Strip all non-ASCII from messages/tool schemas and retry; api_kwargs too so a
    # non-ASCII transformed field doesn't survive via _build_api_kwargs cache paths.
    _messages_sanitized = _sanitize_messages_non_ascii(messages)
    if isinstance(api_messages, list):
        _sanitize_messages_non_ascii(api_messages)
    if isinstance(api_kwargs, dict):
        _sanitize_structure_non_ascii(api_kwargs)
    _prefill_sanitized = isinstance(_prefill, list) and _sanitize_messages_non_ascii(_prefill)
    _tools = getattr(agent, "tools", None)
    _tools_sanitized = isinstance(_tools, list) and _sanitize_tools_non_ascii(_tools)

    _system_sanitized = False
    if isinstance(active_system_prompt, str):
        _sanitized_system = _strip_non_ascii(active_system_prompt)
        if _sanitized_system != active_system_prompt:
            active_system_prompt = agent._cached_system_prompt = _sanitized_system
            _system_sanitized = True
    _ephemeral = getattr(agent, "ephemeral_system_prompt", None)
    if isinstance(_ephemeral, str) and _strip_non_ascii(_ephemeral) != _ephemeral:
        agent.ephemeral_system_prompt = _strip_non_ascii(_ephemeral)
        _system_sanitized = True

    _client_kwargs = getattr(agent, "_client_kwargs", None)
    _default_headers = _client_kwargs.get("default_headers") if isinstance(_client_kwargs, dict) else None
    _headers_sanitized = isinstance(_default_headers, dict) and _sanitize_structure_non_ascii(_default_headers)

    # Non-ASCII in the API key makes httpx fail encoding the Authorization header — the
    # usual persistent cause after message/tool sanitization. Entra ID bearer providers
    # are callables minting ASCII JWTs; skip them (``_strip_non_ascii`` would crash).
    # Sanitize the API key — non-ASCII characters in credentials (e.g. ʋ instead of v from a bad copy-paste)
    # cause httpx to fail when encoding the Authorization header as ASCII. This is the most common cause of
    # persistent UnicodeEncodeError that survives message/tool sanitization (#6843).
    _credential_sanitized = False
    _raw_key = getattr(agent, "api_key", None) or ""
    if _raw_key and isinstance(_raw_key, str):
        _clean_key = _strip_non_ascii(_raw_key)
        if _clean_key != _raw_key:
            agent.api_key = _clean_key
            if isinstance(getattr(agent, "_client_kwargs", None), dict):
                agent._client_kwargs["api_key"] = _clean_key
            # The live client reads its own api_key copy on every request.
            if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
                agent.client.api_key = _clean_key
            _credential_sanitized = True
            _vlines(
                agent,
                "⚠️  API key contained non-ASCII characters (bad copy-paste?) — stripped them. "
                "If auth fails, re-copy the key from your provider's dashboard.",
            )

    # Always retry on ASCII codec detection: _force_ascii_payload sanitizes the full
    # api_kwargs next iteration even when the checks above find nothing.
    agent._unicode_sanitization_passes += 1
    _vlines(
        agent,
        "⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying..."
        if (_messages_sanitized or _prefill_sanitized or _tools_sanitized or _system_sanitized
            or _headers_sanitized or _credential_sanitized) else
        "⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
    )
    return True, active_system_prompt


def recover_before_classification(
    agent: Any, api_error: Exception, *, messages: List[Dict[str, Any]], api_messages: Any,
    api_kwargs: Any, active_system_prompt: Any,
) -> Tuple[bool, Any]:
    """Recovery branches that run BEFORE ``classify_api_error``: UnicodeEncodeError
    sanitization, provider image-content rejection (switch session to text-only), and the
    Bedrock AnthropicBedrock SDK streaming fallback. Returns ``(retry_now,
    active_system_prompt)``; the prompt may be ASCII-sanitized in place."""
    if isinstance(api_error, UnicodeEncodeError) and getattr(agent, '_unicode_sanitization_passes', 0) < 2:
        _recovered, active_system_prompt = _recover_unicode_encode_error(
            agent, api_error, messages, api_messages, api_kwargs, active_system_prompt
        )
        if _recovered:
            return True, active_system_prompt

    # Some providers 4xx on image_url content: strip images, mark session
    # vision-unsupported, retry text-only. English phrase match; extend it.
    _err_body = ""
    try:
        _err_body = str(getattr(api_error, "body", None) or getattr(api_error, "message", None) or str(api_error))
    except Exception:
        pass
    _err_status = getattr(api_error, "status_code", None)
    # 4xx-only gate: 5xx/timeouts are transient and take the retry path.
    _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
    if getattr(agent, "_vision_supported", True) and _looks_like_image_content_rejection(_err_body) and _status_ok:
        agent._vision_supported = False
        _imgs_removed = _strip_images_from_messages(messages)
        if isinstance(api_messages, list):
            _strip_images_from_messages(api_messages)
        _vlines(
            agent,
            "⚠️  Server rejected image content — switching to text-only mode for this session"
            + (". Stripped images from history and retrying." if _imgs_removed else "."),
        )
        return True, active_system_prompt

    # AnthropicBedrock SDK raises "Unexpected event order" when Bedrock errors before
    # message_start; fall back to native Converse for this session.
    if (
        isinstance(api_error, RuntimeError)
        and "unexpected event order" in str(api_error).lower()
        and getattr(agent, "provider", "") == "bedrock"
        and agent.api_mode == "anthropic_messages"
        and not getattr(agent, "_bedrock_converse_fallback_attempted", False)
    ):
        agent._bedrock_converse_fallback_attempted = True
        agent.api_mode = "bedrock_converse"
        agent._bedrock_region = getattr(agent, "_bedrock_region", None) or "us-east-1"
        agent.client = None  # Drop the AnthropicBedrock client
        agent._client_kwargs = {}
        _vlines(agent, "⚠️  AnthropicBedrock SDK streaming failed — falling back to native Converse API for this session.")
        return True, active_system_prompt
    return False, active_system_prompt


def _print_nous_401_diagnostics(agent: Any, api_error: Exception) -> None:
    """Nous 401 that survived a credential refresh: likely Portal OAuth expired/revoked,
    no credits, or agent key blocked."""
    from agent.conversation_loop import _print_nous_entitlement_guidance
    from hermes_constants import display_hermes_home
    _body_text = ""
    try:
        _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
        if _body is not None:
            _body_text = str(_body)[:200]
    except Exception:
        pass
    _plines(agent, "🔐 Nous 401 — Portal authentication failed.")
    if _body_text:
        _plines(agent, f"   Response: {_body_text}")
    try:
        from hermes_cli.anon_auth import is_anonymous_agent
        if is_anonymous_agent(agent):
            # The free tier has no credits, no agent key and no auth.json to inspect: its session
            # ended and could not be replaced. The two doors are a sign-in or another provider.
            _plines(agent, "   Your session ended and Hermes couldn't start a new one.",
                    "   Sign in with a Nous account (it's free), or switch providers with /model.")
            return
    except Exception:
        pass
    if not _print_nous_entitlement_guidance(agent, "Nous model access"):
        _plines(agent, "   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
    _plines(
        agent,
        "   Troubleshooting:",
        "     • Re-authenticate: hermes auth add nous",
        "     • Check credits / billing: https://portal.nousresearch.com",
        f"     • Verify stored credentials: {display_hermes_home()}/auth.json",
        "     • Switch providers temporarily: /model <model> --provider openrouter",
    )


def _print_anthropic_401_diagnostics(agent: Any, key: Any) -> None:
    """Anthropic 401 that survived a credential refresh: show auth method + fixes."""
    from agent.anthropic_credentials import _is_oauth_token
    from agent.azure_identity_adapter import is_token_provider
    from hermes_constants import display_hermes_home
    _plines(agent, "🔐 Anthropic 401 — authentication failed.")
    if is_token_provider(key):
        # Azure Foundry Entra ID: JWT minted per-request by an httpx hook; 401 = Azure
        # rejected it (RBAC, az login, IMDS).
        _plines(
            agent,
            "   Auth method: Microsoft Entra ID (httpx event hook)",
            "   Run `hermes doctor` for credential-chain diagnostics, or",
            "   `az login` if your developer session expired.",
        )
    else:
        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
        _plines(
            agent,
            f"   Auth method: {auth_method}",
            f"   Token prefix: {key[:12]}..." if isinstance(key, str) and len(key) > 12 else "   Token: (empty or short)",
        )
    _dhh = display_hermes_home()
    _plines(
        agent,
        "   Troubleshooting:",
        f"     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens",
        f"     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values",
        "     • For API keys: verify at https://platform.claude.com/settings/keys",
        "     • Hermes login (OAuth): run 'hermes auth add anthropic' to sign in again, then retry",
        "     • Inspect what Hermes holds: hermes auth list anthropic",
        "     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"",
        "     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"",
    )


def _refresh_credentials_after_401(
    agent: Any, api_error: Exception, _retry: TurnRetryState, status_code: Optional[int]
) -> bool:
    """Per-provider one-shot credential refresh on 401 (codex/xai, vertex, nous, copilot,
    anthropic), printing user-facing diagnostics when the nous/anthropic refresh fails.
    Returns True when a refresh succeeded and the call should be retried."""
    from agent.conversation_loop import _is_copilot_provider

    if status_code != 401:
        return False
    if (
        agent.api_mode == "codex_responses"
        and agent.provider in {"openai-codex", "xai-oauth"}
        and not _retry.codex_auth_retry_attempted
    ):
        _retry.codex_auth_retry_attempted = True
        if agent._try_refresh_codex_client_credentials(force=True):
            _label = "xAI OAuth" if agent.provider == "xai-oauth" else "Codex"
            agent._buffer_vprint(f"🔐 {_label} auth refreshed after 401. Retrying request...")
            return True
    if agent.api_mode == "chat_completions" and agent.provider == "vertex" and not _retry.vertex_auth_retry_attempted:
        _retry.vertex_auth_retry_attempted = True
        if agent._try_refresh_vertex_client_credentials():
            agent._buffer_vprint("🔐 Vertex AI token refreshed after 401. Retrying request...")
            return True
    if (
        agent.api_mode in ("chat_completions", "anthropic_messages")
        and agent.provider == "nous"
        and not _retry.nous_auth_retry_attempted
    ):
        _retry.nous_auth_retry_attempted = True
        if agent._try_refresh_nous_client_credentials(force=True):
            agent._buffer_vprint("🔐 Nous agent key refreshed after 401. Retrying request...")
            return True
        _print_nous_401_diagnostics(agent, api_error)
    if _is_copilot_provider(agent) and not _retry.copilot_auth_retry_attempted:
        _retry.copilot_auth_retry_attempted = True
        if agent._try_refresh_copilot_client_credentials():
            agent._buffer_vprint("🔐 Copilot credentials refreshed after 401. Retrying request...")
            return True
    if (
        agent.api_mode == "anthropic_messages"
        and hasattr(agent, '_anthropic_api_key')
        and not _retry.anthropic_auth_retry_attempted
    ):
        _retry.anthropic_auth_retry_attempted = True
        if agent._try_refresh_anthropic_client_credentials():
            _plines(agent, "🔐 Anthropic credentials refreshed after 401. Retrying request...")
            return True
        _print_anthropic_401_diagnostics(agent, agent._anthropic_api_key)
    return False

def _recover_format_errors(
    agent: Any, api_error: Exception, classified: Any, _retry: TurnRetryState,
    messages: List[Dict[str, Any]], api_messages: Any,
) -> bool:
    """One-shot format-recovery strips: thinking-signature → invalid-encrypted-content
    replay disable → native-compaction reject → llama.cpp grammar strip. Returns True when
    the request was repaired and should be retried."""
    # Upstream mutation invalidates Anthropic's thinking-block signature (400). Strip
    # ``reasoning_details`` from ``api_messages`` only, never ``messages`` (state.db).
    if classified.reason == FailoverReason.thinking_signature and not _retry.thinking_sig_retry_attempted:
        _retry.thinking_sig_retry_attempted = True
        _api_stripped = 0
        for _m in api_messages:
            if isinstance(_m, dict) and "reasoning_details" in _m:
                _m.pop("reasoning_details", None)
                _api_stripped += 1
        _vlines(agent, "⚠️  Thinking block signature invalid, stripped reasoning_details from api_messages for retry...")
        logger.warning(
            "%sThinking block signature recovery: stripped "
            "reasoning_details from %d api_messages "
            "(canonical messages unchanged)",
            agent.log_prefix, _api_stripped,
        )
        return True

    # 400 ``invalid_encrypted_content`` on a stale ``codex_reasoning_items`` blob:
    # disable replay for the session, strip cached items, retry once.
    if (
        classified.reason == FailoverReason.invalid_encrypted_content
        and not _retry.invalid_encrypted_content_retry_attempted
        and agent.api_mode == "codex_responses"
        and bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
        and any(
            isinstance(_m, dict)
            and _m.get("role") == "assistant"
            and isinstance(_m.get("codex_reasoning_items"), list)
            and _m.get("codex_reasoning_items")
            for _m in messages
        )
    ):
        _retry.invalid_encrypted_content_retry_attempted = True
        replay_stats = agent._disable_codex_reasoning_replay(messages)
        _vlines(
            agent,
            f"⚠️  Encrypted reasoning replay was rejected by the provider — "
            f"disabled replay and stripped {replay_stats['items']} item(s) from "
            f"{replay_stats['messages']} message(s), retrying...",
        )
        logger.warning(
            "%sInvalid encrypted reasoning recovery: disabled replay and stripped %d items from %d messages",
            agent.log_prefix, replay_stats["items"], replay_stats["messages"],
        )
        return True

    # Structured 400 naming ``context_management``: disable native compaction for the
    # session, retry once; local compression takes over.
    if (
        agent.api_mode == "codex_responses"
        and not _retry.native_compaction_reject_retry_attempted
        and bool(getattr(agent, "codex_responses_native_compaction", False))
    ):
        from agent.native_compaction import is_native_compaction_rejection
        if is_native_compaction_rejection(api_error, getattr(api_error, "status_code", None)):
            _retry.native_compaction_reject_retry_attempted = True
            agent.codex_responses_native_compaction = False
            _vlines(
                agent,
                "⚠️  Provider rejected native compaction (context_management) — disabled for this session, "
                "local compression stays active. Retrying...",
            )
            logger.warning(
                "%sNative compaction rejection recovery: disabled "
                "codex_responses_native for this session and retrying",
                agent.log_prefix,
            )
            return True

    # llama.cpp ``json-schema-to-grammar`` rejects regex escapes and most ``format``
    # values: strip ``pattern``/``format`` from ``agent.tools``, retry once.
    if classified.reason == FailoverReason.llama_cpp_grammar_pattern and not _retry.llama_cpp_grammar_retry_attempted:
        _retry.llama_cpp_grammar_retry_attempted = True
        try:
            from tools.schema_sanitizer import strip_pattern_and_format
            _, _stripped = strip_pattern_and_format(agent.tools)
        except Exception as _strip_exc:  # pragma: no cover — defensive
            logger.warning("%sllama.cpp grammar recovery: strip helper failed: %s", agent.log_prefix, _strip_exc)
            _stripped = 0
        if _stripped:
            _vlines(agent, f"⚠️  llama.cpp rejected tool schema grammar — stripped {_stripped} pattern/format keyword(s), retrying...")
            logger.warning(
                "%sllama.cpp grammar recovery: stripped %d "
                "pattern/format keyword(s) from tool schemas",
                agent.log_prefix, _stripped,
            )
            return True
        # Nothing to strip — fall through to normal retry rather than loop on the same error.
        logger.warning(
            "%sllama.cpp grammar error but no pattern/format "
            "keywords to strip — falling through to normal retry",
            agent.log_prefix,
        )
    return False


_WELCOME_ROUTE_HEAL_COPY = {
    "anon_on_paid_host": "Reconnected to the free model's own route.",
    "named_on_welcome_host": "Reconnected to your Nous account's own route.",
}


def _recover_welcome_tier(agent: Any, classified: Any, _retry: TurnRetryState) -> bool:
    """Two one-shot repairs for the Nous free tier, both silent on the wire and named once in chat.

    ``model_not_free``: the session asked the welcome host for a model it does not serve; move
    to the first alternate the gateway named (its own model) and retry, instead of failing the
    turn. ``anon_on_paid_host`` / ``named_on_welcome_host``: this process is pointed at the other
    identity's host (a stale route); re-read the credentials, which heals the URL, and retry. The
    refresh reports False when the store yields the same route, so a user-set
    ``NOUS_INFERENCE_BASE_URL`` falls straight through to the terminal copy.

    Reads the CLASSIFIER's context (``classified.error_context``): that is where
    ``_nous_welcome_tier`` parks ``welcome_refusal`` / ``welcome_route``. The turn's other context
    (``extract_api_error_context``) never carries them."""
    ctx = getattr(classified, "error_context", None) or {}
    refusal = ctx.get("welcome_refusal") if isinstance(ctx, dict) else None
    if isinstance(refusal, dict) and refusal.get("reason") == "model_not_free" and not _retry.welcome_model_switch_attempted:
        _retry.welcome_model_switch_attempted = True
        alternates = [a for a in (refusal.get("alternates") or []) if isinstance(a, str) and a]
        requested = str(getattr(agent, "model", "") or "")
        target = alternates[0] if alternates else None
        if target and target != requested:
            try:
                agent.model = target
                agent._nous_model_switch = (requested, target)
            except Exception:
                return False
            _vlines(agent, f"↪️  {requested} isn't available without signing in; using {target} for now. Retrying...")
            logger.info("%sNous free tier: moved %s -> %s after model_not_free", agent.log_prefix, requested, target)
            return True
    route = ctx.get("welcome_route") if isinstance(ctx, dict) else None
    if route in _WELCOME_ROUTE_HEAL_COPY and not _retry.welcome_route_heal_attempted:
        _retry.welcome_route_heal_attempted = True
        try:
            healed = bool(agent._try_refresh_nous_client_credentials(force=True))
        except Exception:
            healed = False
        if healed:
            _vlines(agent, f"🔐 {_WELCOME_ROUTE_HEAL_COPY[route]} Retrying request...")
            return True
    return False


def recover_after_classification(
    agent: Any, api_error: Exception, classified: Any, _retry: TurnRetryState, *,
    status_code: Optional[int], error_context: Any, messages: List[Dict[str, Any]],
    api_messages: Any,
) -> Tuple[bool, bool]:
    """One-shot recovery chain that runs AFTER ``classify_api_error`` and before the
    generic retry path. Order is load-bearing (each branch may ``return`` early):
    Nous paid-entitlement refresh → credential-pool rotation → image shrink →
    multimodal-tool-content strip → corrupt-image strip → Anthropic OAuth 1M-beta
    disable → per-provider 401 credential refresh → format-recovery strips.
    Returns ``(retry_now, recovered_with_pool)``; the latter feeds the Nous rate-limit guard."""
    from agent.conversation_loop import _is_nous_inference_route

    if _recover_welcome_tier(agent, classified, _retry):
        return True, False

    if (
        classified.reason == FailoverReason.billing
        and _is_nous_inference_route(
            getattr(agent, "provider", "") or "", getattr(agent, "base_url", "") or ""
        )
        and not _retry.nous_paid_entitlement_refresh_attempted
    ):
        _retry.nous_paid_entitlement_refresh_attempted = True
        if _try_refresh_nous_paid_entitlement_credentials(agent):
            _vlines(agent, "🔐 Nous paid access verified — refreshed runtime credentials and retrying request...")
            return True, False

    recovered_with_pool, _retry.has_retried_429 = agent._recover_with_credential_pool(
        status_code=status_code, has_retried_429=_retry.has_retried_429,
        classified_reason=classified.reason, error_context=error_context,
        billing_unverified=classified.billing_unverified,
    )
    if recovered_with_pool:
        return True, recovered_with_pool

    # Shrink oversized native image parts in-place and retry once.
    if classified.reason == FailoverReason.image_too_large and not _retry.image_shrink_retry_attempted:
        _retry.image_shrink_retry_attempted = True
        if agent._try_shrink_image_parts_in_messages(
            api_messages, max_dimension=_image_error_max_dimension(api_error) or 8000
        ):
            _vlines(agent, "📐 Image(s) exceeded provider size limit — shrank and retrying...")
            return True, recovered_with_pool
        logger.info(
            "image-shrink recovery: no data-URL image parts found "
            "or shrink didn't reduce size; surfacing original error."
        )

    # Strict OpenAI-spec providers 400 on list-type tool content: strip images, mark
    # (provider, model) no-list-tool-content for the session, retry once.
    if (
        classified.reason == FailoverReason.multimodal_tool_content_unsupported
        and not _retry.multimodal_tool_content_retry_attempted
    ):
        _retry.multimodal_tool_content_retry_attempted = True
        if agent._try_strip_image_parts_from_tool_messages(api_messages):
            _vlines(agent, "📐 Provider rejected list-type tool content — downgraded screenshots to text and retrying...")
            return True, recovered_with_pool
        logger.info(
            "multimodal-tool-content recovery: no list-type tool "
            "messages with image parts found; surfacing original error."
        )

    # Reasoning-mandatory route (Nous Portal / OpenRouter, e.g. GLM-5.3) 400s on
    # ``reasoning: {enabled: false}``. The catalog guard in the provider profile normally swallows
    # the disable, but a process that warmed its caps cache before the route flipped keeps sending
    # it. One-shot: never send a disable again this session (the wire builder omits it → upstream
    # default thinking), queue a catalog refresh so the guard is right next time, retry.
    if (
        classified.reason == FailoverReason.reasoning_mandatory
        and not _retry.reasoning_mandatory_retry_attempted
    ):
        _retry.reasoning_mandatory_retry_attempted = True
        agent._reasoning_disable_rejected = True
        try:
            from hermes_cli.models_reasoning_caps import refresh_reasoning_caps_async
            refresh_reasoning_caps_async(agent.provider)
        except Exception:
            pass
        _vlines(agent, f"⚠️  {agent.model} requires reasoning — thinking stays on for this session, retrying...")
        logger.warning("%sReasoning-mandatory recovery: dropping reasoning disable for %s", agent.log_prefix, agent.model)
        return True, recovered_with_pool

    # Provider rejected the image bytes; shrinking can't help, so strip image parts.
    # Strip ONLY the per-call copy: replacing msg["content"] on the shallow api_messages
    # rows keeps canonical history's images (transient rejection must not erase history).
    if classified.reason == FailoverReason.image_corrupt:
        if isinstance(api_messages, list) and _strip_images_from_messages(api_messages):
            _vlines(agent, "⚠️  Provider rejected a corrupted image — stripped images from the retry payload and retrying...")
            return True, recovered_with_pool
        logger.info("image-corrupt recovery: no image parts found to strip; surfacing original error.")

    # Anthropic OAuth subscription rejected the 1M-context beta: disable it for this
    # session, rebuild the client, retry once. Reactive so capable subscriptions keep 1M.
    if (
        # See PR #17680 for the original report (we chose reactive recovery over the proposed unconditional
        # omit so capable subscriptions don't silently lose the capability).
        classified.reason == FailoverReason.oauth_long_context_beta_forbidden
        and agent.api_mode == "anthropic_messages"
        and agent._is_anthropic_oauth
        and not _retry.oauth_1m_beta_retry_attempted
    ):
        _retry.oauth_1m_beta_retry_attempted = True
        if not getattr(agent, "_oauth_1m_beta_disabled", False):
            agent._oauth_1m_beta_disabled = True
            try:
                agent._anthropic_client.close()
            except Exception:
                pass
            agent._rebuild_anthropic_client()
            _vlines(agent, "🔕 OAuth subscription doesn't support the 1M-context beta — disabled for this session and retrying...")
            return True, recovered_with_pool

    if _refresh_credentials_after_401(agent, api_error, _retry, status_code):
        return True, recovered_with_pool

    if _recover_format_errors(agent, api_error, classified, _retry, messages, api_messages):
        return True, recovered_with_pool
    return False, recovered_with_pool


def _failed_turn_result(final_response: str, messages: Any, api_call_count: int, error: str) -> Dict[str, Any]:
    """Base failed-turn result dict shared by the two terminal paths."""
    return {
        "final_response": final_response, "messages": messages, "api_calls": api_call_count,
        "completed": False, "failed": True, "error": error,
    }


def _print_nonretryable_auth_guidance(
    agent: Any, classified: Any, *, status_code: Optional[int], provider: Any, base_url: Any, model: Any,
) -> None:
    """Actionable guidance for a terminal auth / billing error."""
    from agent.conversation_loop import _print_billing_or_entitlement_guidance, _print_nous_entitlement_guidance

    if classified.reason == FailoverReason.billing and _print_billing_or_entitlement_guidance(
        agent, capability="model access", provider=provider, base_url=str(base_url),
        model=model, unverified=classified.billing_unverified,
    ):
        return
    if provider == "nous" and _print_nous_entitlement_guidance(agent, "Nous model access"):
        return
    if provider in {"openai-codex", "xai-oauth", "nous"} and status_code == 401:
        if provider == "openai-codex":
            from agent.turn_failure_copy import oauth_relogin_command

            _vlines(
                agent,
                "   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been",
                "      refreshed by another client (Codex CLI, VS Code) or another Hermes profile.",
                f"      Sign this profile in again: `{oauth_relogin_command(provider)}`",
            )
        elif provider == "xai-oauth":
            _vlines(
                agent,
                "   💡 xAI OAuth token was rejected (HTTP 401). To fix:",
                "      re-authenticate with xAI Grok OAuth (SuperGrok / Premium+) from `hermes model`.",
            )
        else:  # nous
            _vlines(
                agent,
                "   💡 Nous Portal OAuth token was rejected (HTTP 401). Your token may be",
                "      expired, revoked, or your account may be out of credits. To fix:",
                "      1. Re-authenticate: hermes portal",
                "      2. Check your portal account: https://portal.nousresearch.com",
            )
            # ``:free`` is OpenRouter slug syntax; Nous Portal will reject the model
            # name even after a successful re-auth.
            if isinstance(model, str) and model.endswith(":free"):
                _vlines(
                    agent,
                    f"      ⚠️  Note: `{model}` looks like an OpenRouter slug (`:free` suffix).",
                    "         Nous Portal won't recognize that model name. Either switch to a",
                    f"         Nous catalog model, or run `/model openrouter:{model}` to use OpenRouter.",
                )
        return
    _vlines(
        agent,
        "   💡 Your API key was rejected by the provider. Check:",
        "      • Is the key valid? Run: hermes setup",
        f"      • Does your account have access to {model}?",
    )
    if base_url_host_matches(str(base_url), "openrouter.ai"):
        _vlines(agent, "      • Check credits: https://openrouter.ai/settings/credits")


def _welcome_tier_guidance(classified: Any, *, model: Any, in_chat: bool, door: bool = True) -> str:
    """Copy for a Nous free-tier refusal the classifier parsed (``welcome_refusal`` /
    ``welcome_route`` in ``error_context``); empty for every other error."""
    ctx = getattr(classified, "error_context", None) or {}
    refusal, route = ctx.get("welcome_refusal"), ctx.get("welcome_route")
    if not refusal and not route:
        return ""
    from hermes_cli.anon_auth import welcome_refusal_copy, welcome_route_refusal_copy
    if refusal:
        return welcome_refusal_copy(refusal, model=str(model or ""), in_chat=in_chat, door=door)
    return welcome_route_refusal_copy(str(route), in_chat=in_chat, door=door)


# Closed table: every card kind the desktop has copy for. An unknown gateway reason lands on
# "refused" (generic card, sentence kept) rather than a code the desktop cannot key on.
_WELCOME_SURFACE_KINDS = {
    "rate_limited": "rate_limited", "at_capacity": "at_capacity", "admission_closed": "at_capacity",
    "model_not_free": "model_not_free", "feature_not_free": "model_not_free",
}


def _welcome_surface_kind(classified: Any) -> str:
    """The free-tier failure kind a client renders its card from (``error_surface`` code
    ``free_tier_<kind>``): the welcome refusal's reason, or the route refusal; "" otherwise."""
    ctx = getattr(classified, "error_context", None) or {}
    refusal = ctx.get("welcome_refusal") if isinstance(ctx, dict) else None
    if isinstance(refusal, dict):
        return _WELCOME_SURFACE_KINDS.get(str(refusal.get("reason") or ""), "refused")
    route = ctx.get("welcome_route") if isinstance(ctx, dict) else None
    if route == "tier_disabled":
        return "disabled"
    # A named account on the welcome host has already signed in: no sign-in card, copy only.
    if route == "named_on_welcome_host":
        return ""
    return "route" if route else ""


def _stamp_free_tier(result: Dict[str, Any], kind: str, message: str) -> Dict[str, Any]:
    """Structured free-tier failure block: ``error_surface`` keys its code on ``kind`` and a client
    shows ``message`` (the chat sentence) as the card body instead of its own generic copy."""
    result["free_tier"] = {"kind": kind or "refused", "message": message}
    return result


def _welcome_outage_copy(base_url: Any, classified: Any, *, anonymous: bool = False) -> str:
    """On the Nous free tier, a transport / server failure that outlived every retry reads as one
    plain sentence (the free model is having trouble) rather than the technical summary. Empty
    for every other route and for rate limits / billing, which have their own copy."""
    try:
        from hermes_cli.anon_auth import FREE_TIER_OUTAGE_COPY, route_is_welcome_host
        # Both: an anonymous JWT sent to a user-overridden paid host never reached the free model.
        if not anonymous or not route_is_welcome_host(base_url):
            return ""
        # Not ``unknown``: that is the classifier's catch-all for status-less local failures, which
        # are not the free model's trouble.
        if classified.reason in (FailoverReason.timeout, FailoverReason.overloaded, FailoverReason.server_error):
            return FREE_TIER_OUTAGE_COPY
    except Exception:
        pass
    return ""


# Terminal status label per non-retryable reason (default names the HTTP status).
_NONRETRYABLE_LABELS = {
    FailoverReason.content_policy_blocked: "The provider's safety filter refused this request",
    FailoverReason.ssl_cert_verification: "The provider's security certificate could not be verified",
    # Only reached after the one-shot image shrink ran (recover_after_classification sets the flag first).
    FailoverReason.image_too_large: "Request still exceeded the provider's size limit after shrinking images",
}


def _missing_vendor_prefix_suggestion(api_error: Exception, provider: Any, model: Any) -> Optional[str]:
    """Prefixed catalogue id when a bare 404 most likely means ``vendor/model`` lost its prefix."""
    if getattr(api_error, "status_code", None) != 404:
        return None
    try:
        from hermes_cli.model_normalize import suggest_prefixed_model_id

        return suggest_prefixed_model_id(str(provider or ""), str(model or ""))
    except Exception:
        return None


def nonretryable_client_error_result(
    agent: Any, api_error: Exception, classified: Any, *, status_code: Optional[int],
    api_kwargs: Any, api_messages: Any, messages: List[Dict[str, Any]], conversation_history: Any,
    api_call_count: int, approx_tokens: int, provider: Any, base_url: Any, model: Any,
) -> Dict[str, Any]:
    """Terminal path for a non-retryable 4xx once fallback is exhausted: debug dump, flush
    the retry trace, print auth / billing / content-policy / TLS guidance, persist (skipped
    for likely context-overflow 400s so the failure does not grow the session), build result."""
    # Result/guidance helpers stay in the loop module (tests import + patch them there).
    from agent.conversation_loop import _billing_failure_result, _content_policy_blocked_result

    if api_kwargs is not None:
        agent._dump_api_request_debug(api_kwargs, reason="non_retryable_client_error", error=api_error)
    # Terminal — flush buffered context so the user sees what was tried before the abort.
    agent._flush_status_buffer()
    # Summarize once: Cloudflare/proxy HTML pages and raw provider bodies must be
    # collapsed here or they leak verbatim via the ``error`` field.
    _nonretryable_summary = agent._summarize_api_error(api_error)
    _plabel = provider_label_for(provider)
    _label = _NONRETRYABLE_LABELS.get(classified.reason, f"{_plabel} rejected the request and retrying won't help")
    agent._emit_diagnostic_status(f"❌ {_label}: {_nonretryable_summary}")
    # The endpoint/status trace is developer detail: verbose only (the log has it always).
    if getattr(agent, "verbose_logging", False):
        _vlines(
            agent,
            f"   🔌 Provider: {provider}  Model: {model}  (HTTP {status_code})",
            f"   🌐 Endpoint: {base_url}",
        )
    _welcome_hint = _welcome_tier_guidance(classified, model=model, in_chat=False)
    _prefix_suggestion = _missing_vendor_prefix_suggestion(api_error, provider, model)
    if _welcome_hint:
        # A free-tier gate or a wrong-host refusal: the way forward is a sign-in or another
        # provider, never the key/credits advice below.
        _vlines(agent, f"   💡 {_welcome_hint}")
    elif classified.is_auth or classified.reason == FailoverReason.billing:
        _print_nonretryable_auth_guidance(
            agent, classified, status_code=status_code, provider=provider, base_url=base_url, model=model
        )
    elif classified.reason == FailoverReason.model_not_found:
        _vlines(agent, f"   💡 Model '{model}' isn't available on {_plabel}. Pick another with /model.")
        if _prefix_suggestion:
            _vlines(agent, f"      Did you mean '{_prefix_suggestion}'? It looks like the vendor prefix is missing.")
    elif classified.reason not in _NONRETRYABLE_LABELS:
        _vlines(agent, f"   💡 Fix: pick another model (/model), or check `{display_hermes_home()}/logs/agent.log`.")
    # Content-policy blocks: the provider refused this prompt, so recovery is a rephrase
    # or another model, not key/retry advice.
    if classified.reason == FailoverReason.content_policy_blocked:
        _vlines(
            agent,
            f"   💡 {CONTENT_POLICY_NEXT_STEPS}",
            "      To route future blocks to another provider automatically: hermes fallback add",
        )
    # TLS certificate failures are environment problems — name the knobs for each cause.
    if classified.reason == FailoverReason.ssl_cert_verification:
        _vlines(
            agent,
            "   💡 Hermes couldn't verify the provider's security certificate. This fails the same",
            "      way on every retry — fix the environment, then try again:",
            "      • Corporate TLS-inspecting proxy? Point Python at its CA bundle:",
            "        export SSL_CERT_FILE=/path/to/corp-ca.pem  (also REQUESTS_CA_BUNDLE)",
            "      • Missing/stale system CA store? Refresh it (in Hermes's venv: `uv pip install",
            "        --upgrade certifi`; macOS: run 'Install Certificates.command').",
            "      • Self-signed local endpoint (llama.cpp, LM Studio, vLLM)? Use http://",
            "        for localhost, or add the server's cert to your trust store.",
        )
    logger.error("%sNon-retryable client error: %s", agent.log_prefix, api_error)
    # Skip persistence on likely context-overflow (400 + large session): persisting the
    # failed message grows the session and repeats the failure.
    # Persisting the failed user message would make the session even larger, causing the same failure on the
    # next attempt. (#1630)
    if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
        _vlines(agent, "⚠️  Skipping session persistence for large failed session to prevent growth loop.")
    else:
        agent._persist_session(messages, conversation_history)
    if classified.reason == FailoverReason.content_policy_blocked:
        return _content_policy_blocked_result(
            messages, api_call_count,
            final_response="⚠️ " + content_policy_copy(label=_plabel, summary=_nonretryable_summary),
            error_detail=_nonretryable_summary,
        )
    # Billing walls get the same structured recovery descriptor as the max-retries path
    # so every surface renders one consistent signal.
    if classified.reason == FailoverReason.billing:
        return _billing_failure_result(
            classified=classified, summary=_nonretryable_summary, messages=messages,
            api_call_count=api_call_count, provider=provider, base_url=base_url, model=model,
        )
    if _welcome_hint:
        # A free-tier refusal is fully explained by its own sentence; the raw provider summary
        # (status codes, JSON) is for the log, not for a first-time user's chat.
        _final_response = _welcome_tier_guidance(classified, model=model, in_chat=True)
    else:
        # Every surface reads final_response; the CLI hint lines above never reach chat.
        _final_response = nonretryable_copy(
            classified, provider=provider, model=model, summary=_nonretryable_summary,
            prefix_suggestion=_prefix_suggestion,
        )
    result = _failed_turn_result(_final_response, messages, api_call_count, _nonretryable_summary)
    # Same verdict fields as the max-retries path: without them the UI descriptor
    # (agent/error_surface.py) reads a rejected OAuth token as a retryable
    # "Provider error" and offers Retry instead of a re-login.
    result.update({
        "failure_reason": classified.reason.value,
        "failure_retryable": bool(classified.retryable),
    })
    if _welcome_hint and (_kind := _welcome_surface_kind(classified)):
        # The card form: the desktop renders the sign-in as a button, so no "To sign in" tail.
        _stamp_free_tier(result, _kind,
                         _welcome_tier_guidance(classified, model=model, in_chat=True, door=False))
    return result


_STREAM_DROP_MARKERS = (
    "connection lost", "connection reset", "connection closed", "network connection",
    "network error", "terminated",
)


def max_retries_exhausted_result(
    agent: Any, api_error: Exception, classified: Any, *, max_retries: int, is_rate_limited: bool,
    error_msg: str, api_kwargs: Any, api_messages: Any, messages: List[Dict[str, Any]],
    conversation_history: Any, api_call_count: int, approx_tokens: int, provider: Any,
    base_url: Any, model: Any,
) -> Dict[str, Any]:
    """Terminal path once retries, transport recovery and fallback all failed: flush the
    trace, emit the billing / rate-limit / generic status, print stream-drop or thinking-timeout
    guidance (the latter wins), persist, build the result with ``failure_reason`` /
    ``failure_retryable`` / ``billing_block``."""
    # Result/guidance helpers stay in the loop module (tests import + patch them there).
    from hermes_cli.anon_auth import is_anonymous_agent
    from agent.conversation_loop import (
        _billing_block_dict, _billing_or_entitlement_message, _billing_terminal_label,
        _print_billing_or_entitlement_guidance,
    )

    agent._flush_status_buffer()
    _final_summary = agent._summarize_api_error(api_error)
    _billing_guidance = ""
    _is_billing = classified.reason == FailoverReason.billing
    if _is_billing:
        if classified.billing_unverified:
            # Ambiguous body — hedge the terminal line.
            agent._emit_diagnostic_status(
                "❌ Provider reported usage/credit exhaustion "
                f"(unverified — may be a content-filter rejection) — {_final_summary}"
            )
        else:
            agent._emit_diagnostic_status(f"❌ Billing or credits exhausted — {_final_summary}")
        _billing_kw = dict(
            capability="model access", provider=provider, base_url=str(base_url), model=model,
            unverified=classified.billing_unverified,
        )
        _billing_guidance = _billing_or_entitlement_message(**_billing_kw)
        _print_billing_or_entitlement_guidance(agent, **_billing_kw)
    elif is_rate_limited:
        agent._emit_diagnostic_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
    else:
        agent._emit_diagnostic_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
    _vlines(agent, f"   💀 Final error: {_final_summary}")
    _welcome_hint = _welcome_tier_guidance(classified, model=model, in_chat=False)
    if _welcome_hint:
        _vlines(agent, f"   💡 {_welcome_hint}")

    # SSE stream-drop (e.g. "Network connection lost"): usually a proxy/CDN cutting a very
    # large tool call mid-response.
    _is_stream_drop = (
        not getattr(api_error, "status_code", None)
        and any(p in error_msg for p in _STREAM_DROP_MARKERS)
    )
    if _is_stream_drop:
        _vlines(
            agent,
            "   💡 The provider's stream connection keeps dropping. This often happens "
            "when the model tries to write a very large file in a single tool call.",
            "      Try asking the model to use execute_code with Python's open() for "
            "large files, or to write the file in smaller sections.",
        )

    # A known reasoning model hit a transport error before the first content token.
    # Distinct from _is_stream_drop; detection lives in agent.thinking_timeout_guidance.
    _is_thinking_timeout = is_thinking_timeout(classified, model, error_msg)
    if _is_thinking_timeout:
        _vlines(agent, f"   💡 {build_thinking_timeout_guidance(provider=provider, model=model).strip()}")

    logger.error(
        "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
        agent.log_prefix, max_retries, _final_summary,
        provider, model, len(api_messages), f"{approx_tokens:,}",
    )
    if api_kwargs is not None:
        agent._dump_api_request_debug(api_kwargs, reason="max_retries_exhausted", error=api_error)
    agent._persist_session(messages, conversation_history)
    _billing_block = None
    _billing_unverified = False
    _free_tier_kind = ""
    if _is_billing:
        _billing_unverified = classified.billing_unverified
        _final_response = _billing_terminal_label(_final_summary, _billing_unverified)
        if _billing_guidance:
            _final_response += f"\n\n{_billing_guidance}"
        # Structured recovery descriptor so every surface renders the same link + label.
        _billing_block = _billing_block_dict(
            provider, base_url, model, _billing_guidance, unverified=_billing_unverified
        )
    else:
        # Every surface reads final_response (the 💡 lines above are CLI-only), so the chat
        # text carries the plain what-happened + next step itself.
        _final_response = exhausted_copy(
            classified.reason.value, label=provider_label_for(provider), attempts=max_retries,
            summary=_final_summary,
        )
        if _welcome_hint:
            _final_response = _welcome_tier_guidance(classified, model=model, in_chat=True)
            _free_tier_kind = _welcome_surface_kind(classified)
        elif _outage := _welcome_outage_copy(base_url, classified, anonymous=is_anonymous_agent(agent)):
            _final_response, _free_tier_kind = _outage, "outage"
    if _is_thinking_timeout:
        # Thinking-timeout guidance overrides stream-drop guidance, which would wrongly
        # suggest splitting large file writes.
        _final_response += "\n\n" + build_thinking_timeout_guidance(provider=provider, model=model)
    elif _is_stream_drop:
        _final_response += (
            "\n\nThe connection kept dropping while the model was writing — this often "
            "happens when it writes a very large file in one go. Ask me to write the file in "
            "smaller sections (or via execute_code with Python's open())."
        )
    result = _failed_turn_result(_final_response, messages, api_call_count, _final_summary)
    result.update({
        # Classified reason so callers (kanban worker in cli.py) can tell a quota wall
        # (``rate_limit`` / ``billing``) from a task failure.
        "failure_reason": classified.reason.value,
        # The classifier's own retry verdict — UI surfaces use this, not the reason string.
        "failure_retryable": bool(classified.retryable),
        # True when the billing verdict rests on an ambiguous body.
        "billing_unverified": _billing_unverified,
        # Present only for billing walls: (provider, billing_url, is_nous, message).
        "billing_block": _billing_block,
    })
    if _free_tier_kind:
        _stamp_free_tier(result, _free_tier_kind, (
            _welcome_tier_guidance(classified, model=model, in_chat=True, door=False)
            if _welcome_hint else _final_response))
    return result


def log_api_error_attempt(
    agent: Any, api_error: Exception, *, retry_count: int, max_retries: int,
    status_code: Optional[int], elapsed_time: float, api_messages: Any, approx_tokens: int,
) -> Tuple[str, str, Any, Any, Any]:
    """Log one failed API attempt (warning + buffered retry trace, OpenRouter "no tool
    endpoints" hint, bare-404 missing-vendor-prefix hint); the buffer only surfaces if every
    retry+fallback exhausts. Returns ``(error_type, error_msg, provider, base_url, model)``."""
    error_type = type(api_error).__name__
    error_msg = str(api_error).lower()
    _error_summary = agent._summarize_api_error(api_error)
    logger.warning(
        "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
        retry_count, max_retries, error_type, agent._client_log_context(), _error_summary,
    )

    _provider = getattr(agent, "provider", "unknown")
    _base = getattr(agent, "base_url", "unknown")
    _model = getattr(agent, "model", "unknown")
    _blines(agent, f"⚠️  Attempt {retry_count}/{max_retries} failed: {_error_summary}")
    # Exception class, endpoint, raw body and token counts are developer detail: verbose only.
    if getattr(agent, "verbose_logging", False):
        _status_code_str = f" [HTTP {status_code}]" if status_code else ""
        _blines(
            agent,
            f"   🔌 {error_type}{_status_code_str}  Provider: {_provider}  Model: {_model}",
            f"   🌐 Endpoint: {_base}",
        )
        if status_code and status_code < 500:
            _err_body = getattr(api_error, "body", None)
            _err_body_str = str(_err_body)[:300] if _err_body else None
            if _err_body_str:
                _blines(agent, f"   📋 Details: {_err_body_str}")
        _blines(agent, f"   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

    if agent._is_openrouter_url() and "support tool use" in error_msg:
        _blines(agent, f"   💡 No OpenRouter providers for {_model} support tool calling with your current settings.")
        from agent.chat_completion_helpers import _provider_preferences_for_agent
        if _provider_preferences_for_agent(agent).get("only"):
            _blines(
                agent,
                "      Your provider_routing.only restriction is filtering out tool-capable providers.",
                "      Try removing the restriction or adding providers that support tools for this model.",
            )
        _blines(agent, f"      Check which providers support tools: https://openrouter.ai/models/{_model}")

    # Bare 404 on a ``vendor/model`` catalogue usually means the id lost its prefix; the
    # provider never names the model, so we do.
    _suggestion = _missing_vendor_prefix_suggestion(api_error, _provider, _model)
    if _suggestion:
        _blines(
            agent,
            f"   💡 Model '{_model}' is not a valid id for provider {_provider} — it is missing its vendor prefix.",
            f"      Did you mean '{_suggestion}'?  Re-pick it with /model.",
        )
    return error_type, error_msg, _provider, _base, _model


def abort_turn_on_interrupt(
    agent: Any, messages: List[Dict[str, Any]], conversation_history: Any, api_call_count: int, *,
    abort_message: str, interrupt_text: str,
) -> Dict[str, Any]:
    """Announce ``abort_message``, close any open tool sequence with ``interrupt_text``,
    persist, clear the interrupt and return the ``interrupted`` result dict."""
    _vlines(agent, f"⚡ {abort_message}")
    close_interrupted_tool_sequence(messages, interrupt_text)
    agent._persist_session(messages, conversation_history)
    # The turn was stopped, not rebuilt: a pending steer was aimed at this turn's next
    # tool iteration, which will no longer happen — drop it (hard-cancel semantics).
    agent.clear_interrupt(hard_cancel=True)
    return {
        "final_response": interrupt_text, "messages": messages, "api_calls": api_call_count,
        "completed": False, "interrupted": True,
    }


def interruptible_backoff_sleep(
    agent: Any, wait_time: float, _retry: Optional[TurnRetryState], *,
    messages: List[Dict[str, Any]], conversation_history: Any, api_call_count: int,
    abort_message: str, interrupt_text: str, activity_label: str,
) -> Optional[Dict[str, Any]]:
    """Sleep ``wait_time`` in 200 ms slices so interrupts are honoured promptly, touching
    activity every ~30 s so the gateway's inactivity monitor knows we are alive.

    On interrupt with ``_retry`` given and a redirect pending: preserve the redirect, arm
    ``_retry.restart_with_redirected_messages`` and return ``None`` (caller rebuilds the
    turn). Otherwise return the ``interrupted`` result dict. ``None`` when the wait completed."""
    sleep_end = time.time() + wait_time
    _touch_counter = 0
    while time.time() < sleep_end:
        if agent._interrupt_requested:
            if _retry is not None and agent.clear_interrupt(preserve_redirect=True):
                _retry.restart_with_redirected_messages = True
                return None
            return abort_turn_on_interrupt(
                agent, messages, conversation_history, api_call_count,
                abort_message=abort_message, interrupt_text=interrupt_text,
            )
        time.sleep(0.2)
        _touch_counter += 1
        if _touch_counter % 150 == 0:  # 150 × 0.2s = 30s
            agent._touch_activity(f"{activity_label}, {int(sleep_end - time.time())}s remaining")
    return None


_ZAI_POLICY_NOTES = {
    "zai_coding_overload_long": " (Z.AI Coding overload adaptive long backoff)",
    "zai_coding_overload_short": " (Z.AI Coding overload short retry)",
}


def compute_error_backoff(
    agent: Any, api_error: Exception, *, retry_count: int, max_retries: int, is_rate_limited: bool,
    is_zai_coding_overload: bool, base_url: Any, model: Any,
) -> float:
    """Pick the wait before the next API retry and announce it. Retry-After wins for
    rate limits and any other retryable error (capped at 600s: Anthropic Tier 1 buckets
    reset in ~171s, so a 120s cap re-tripped the limit); otherwise jittered backoff,
    replaced by the adaptive policy for 429s / Z.AI overloads. Normal retries are
    buffered; long Z.AI Coding waits surface immediately."""
    # Imported lazily so tests that patch ``agent.retry_utils.jittered_backoff`` /
    # ``adaptive_rate_limit_backoff`` (incl. the run_agent conftest fast-backoff fixture) intercept.
    from agent.retry_utils import adaptive_rate_limit_backoff, jittered_backoff, parse_retry_after_seconds

    # Respect Retry-After on every retryable provider error, not just 429s. Retryable
    # 5xx responses (e.g. Cloudflare 520/524) also carry the header or a structured
    # ``retry_after`` problem-detail body field; ignoring either turns an origin
    # outage into a retry storm.
    _retry_after = parse_retry_after_seconds(
        getattr(getattr(api_error, "response", None), "headers", None)
    )
    if _retry_after is None:
        _error_body = getattr(api_error, "body", None)
        if isinstance(_error_body, dict):
            # Some providers nest it as error.retry_after (the same unwrap
            # extract_api_error_context uses), others put it at the top level.
            _nested = _error_body.get("error")
            _payload = _nested if isinstance(_nested, dict) else _error_body
            _retry_after = parse_retry_after_seconds(_payload.get("retry_after"))
    if _retry_after is not None:
        # Cap at 10 minutes. Anthropic Tier 1 input-token buckets reset in ~171s, so a 120s cap
        # caused us to retry before the actual reset window and re-trip the limit. 600s covers all
        # realistic provider reset windows while still rejecting pathological values. (#26293)
        _retry_after = min(_retry_after, 600)
        if _retry_after <= 0:
            # A zero/expired cooldown (retry-after: 0, or an HTTP-date in the
            # past, which the parser clamps to 0.0) carries no usable wait —
            # treat it as absent so we never hot-loop the provider.
            _retry_after = None
    wait_time = _retry_after if _retry_after is not None else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
    _backoff_policy = None
    _adaptive = is_rate_limited or is_zai_coding_overload
    if _adaptive and _retry_after is None:
        wait_time, _backoff_policy = adaptive_rate_limit_backoff(
            retry_count, base_url=str(base_url), model=model, error=api_error, default_wait=wait_time,
        )
    if _adaptive:
        _policy_note = _ZAI_POLICY_NOTES.get(_backoff_policy or "", "")
        _wait_reason = "Provider overloaded" if is_zai_coding_overload and not is_rate_limited else "Rate limited"
        _rate_limit_status = f"⏱️ {_wait_reason}. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries}){_policy_note}..."
        if _backoff_policy == "zai_coding_overload_long":
            agent._emit_diagnostic_status(_rate_limit_status)
        else:
            agent._buffer_diagnostic_status(_rate_limit_status)
    else:
        _retry_status = (
            f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})..."
        )
        if _retry_after is not None and _retry_after > 60:
            # A 5xx Retry-After can now reach the 600s cap; buffering that wait
            # would leave the user silent for minutes, so surface long provider
            # cooldowns immediately (mirrors the zai_coding_overload_long path).
            agent._emit_diagnostic_status(_retry_status)
        else:
            agent._buffer_diagnostic_status(_retry_status)
    # The buffered line only replays if every retry fails; the live status
    # line is the one thing the user sees meanwhile. Name the wait there so a
    # 60s backoff after a 5xx is not an anonymous spinner — this is transient
    # (rewritten by the next frame, cleared on recovery), so it does not add
    # the transcript chatter the buffer exists to avoid.
    agent._emit_diagnostic_wait(
        f"⏳ waiting on provider — retrying in {wait_time:.0f}s (attempt {retry_count}/{max_retries})"
    )
    logger.warning(
        "Retrying API call in %ss (attempt %s/%s) %s policy=%s error=%s",
        wait_time, retry_count, max_retries, agent._client_log_context(),
        _backoff_policy or "default", api_error,
    )
    return wait_time


def validate_response_shape(agent: Any, response: Any) -> Tuple[bool, List[str]]:
    """Validate the raw provider response via the transport; ``(response_invalid,
    error_details)``. A Codex ``failed``/``cancelled`` status (e.g. quota exhaustion) is
    invalid so the fallback chain triggers; an empty Codex ``output`` with non-empty
    ``output_text`` is deferred to normalization."""
    if agent._get_transport().validate_response(response):
        return False, []
    if response is None:
        return True, ["response is None"]
    if agent.api_mode == "codex_responses":
        _codex_resp_status = str(getattr(response, "status", "") or "").strip().lower()
        if _codex_resp_status in {"failed", "cancelled"}:
            _codex_error_obj = getattr(response, "error", None)
            _codex_error_msg = (
                _codex_error_obj.get("message") if isinstance(_codex_error_obj, dict)
                else str(_codex_error_obj) if _codex_error_obj
                else f"Responses API returned status '{_codex_resp_status}'"
            )
            logger.warning(
                "Codex response status='%s' (error=%s). Routing to fallback. %s",
                _codex_resp_status, _codex_error_msg, agent._client_log_context(),
            )
            return True, [f"response.status={_codex_resp_status}: {_codex_error_msg}"]
        # Stream backfill may have failed but normalize can still recover from output_text.
        _out_text = getattr(response, "output_text", None)
        _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
        if _out_text_stripped:
            logger.debug(
                "Codex response.output is empty but output_text is present "
                "(%d chars); deferring to normalization.",
                len(_out_text_stripped),
            )
            return False, []
        logger.warning(
            "Codex response.output is empty after stream backfill "
            "(status=%s, incomplete_details=%s, model=%s). %s",
            getattr(response, "status", None), getattr(response, "incomplete_details", None),
            getattr(response, "model", None),
            f"api_mode={agent.api_mode} provider={agent.provider}",
        )
        return True, ["response.output is empty"]
    if agent.api_mode == "anthropic_messages":
        detail = "response.content invalid (not a non-empty list)"
    elif agent.api_mode == "bedrock_converse":
        detail = "Bedrock response invalid (no output or choices)"
    elif not hasattr(response, 'choices'):
        detail = "response has no 'choices' attribute"
    elif response.choices is None:
        detail = "response.choices is None"
    else:
        detail = "response.choices is empty"
    return True, [detail]


def describe_invalid_response(agent: Any, response: Any, api_duration: float) -> Tuple[str, str, str]:
    """Diagnostics for an empty/malformed response: ``(error_msg, provider_name,
    failure_hint)``. The hint is derived from the provider error code (524/504/429/
    5xx) and the response time, instead of always assuming rate limiting."""
    error_msg = "Unknown"
    provider_name = "Unknown"
    _has_error = bool(response and hasattr(response, 'error') and response.error)
    if _has_error:
        error_msg = str(response.error)
        if hasattr(response.error, 'metadata') and response.error.metadata:
            provider_name = response.error.metadata.get('provider_name', 'Unknown')
    elif response and hasattr(response, 'message') and response.message:
        error_msg = str(response.message)

    # OpenRouter often returns the actual model used.
    if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
        provider_name = f"model={response.model}"

    if provider_name == "Unknown" and response:
        resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
        if agent.verbose_logging:
            logging.debug(f"Response attributes for invalid response: {resp_attrs}")

    _resp_error_code = None
    if _has_error:
        _code_raw = getattr(response.error, 'code', None)
        if _code_raw is None and isinstance(response.error, dict):
            _code_raw = response.error.get('code')
        if _code_raw is not None:
            try:
                _resp_error_code = int(_code_raw)
            except (TypeError, ValueError):
                pass

    return error_msg, provider_name, _failure_hint_for(_resp_error_code, api_duration)


def _failure_hint_for(code: Optional[int], api_duration: float) -> str:
    """Human-readable hint from the provider error code and response time."""
    if code == 524:
        return f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
    if code == 504:
        return f"upstream gateway timeout (504, {api_duration:.0f}s)"
    if code == 429:
        return "rate limited by upstream provider (429)"
    if code in {500, 502}:
        return f"upstream server error ({code}, {api_duration:.0f}s)"
    if code in {503, 529}:
        return f"upstream provider overloaded ({code})"
    if code is not None:
        return f"upstream error (code {code}, {api_duration:.0f}s)"
    if api_duration < 10:
        return f"fast response ({api_duration:.1f}s) — likely rate limited"
    if api_duration > 60:
        return f"slow response ({api_duration:.0f}s) — likely upstream timeout"
    return f"response time {api_duration:.1f}s"


@dataclass
class ClassifiedErrorVerdict:
    """Outcome of ``route_classified_error``. ``action``: ``"return"`` (terminal result),
    ``"break"`` (restart armed on ``_retry``), ``"continue"`` (re-enter the retry loop; Nous
    guard re-check) or ``"fallthrough"`` (proceed to overflow / client-error / backoff
    handling). The remaining fields are loop locals the router rebound or computed."""

    action: str
    result: Optional[Dict[str, Any]]
    status_code: Optional[int]
    messages: List[Dict[str, Any]]
    active_system_prompt: Any
    conversation_history: Any
    retry_count: int
    max_retries: int
    compression_attempts: int
    provider_overflow_recovery_pending: bool
    is_rate_limited: bool
    wrapped_output_cap_budget: Optional[int]
    is_zai_coding_overload: bool


_OVERFLOW_REASONS = frozenset({
    FailoverReason.long_context_tier, FailoverReason.payload_too_large, FailoverReason.context_overflow,
})
_RATE_LIMIT_REASONS = frozenset({
    FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit,
})
_TRANSPORT_FAILURE_REASONS = frozenset({FailoverReason.timeout, FailoverReason.overloaded})


_LONG_CONTEXT_TIER_CAP = 200000


def _cap_long_context_tier(agent: Any) -> int:
    """Cap the compressor's context window at the long-context tier limit; returns the
    previous ``context_length``."""
    compressor = agent.context_compressor
    old_ctx = compressor.context_length
    if old_ctx > _LONG_CONTEXT_TIER_CAP:
        compressor.update_model(
            model=agent.model, context_length=_LONG_CONTEXT_TIER_CAP, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
        )
        # Context probing flags exist only on the built-in compressor (plugin engines
        # manage their own). Don't persist — a tier limit, not a model capability;
        # 1M should return if extra usage is enabled.
        if hasattr(compressor, "_context_probed"):
            compressor._context_probed = True
            compressor._context_probe_persistable = False
        agent._buffer_vprint(
            f"⚠️  Anthropic long-context tier "
            f"requires extra usage — reducing context: "
            f"{old_ctx:,} → {_LONG_CONTEXT_TIER_CAP:,} tokens"
        )
    return old_ctx


def _eager_fallback_status(classified: Any, is_upstream: bool, is_transport_failure: bool) -> str:
    """Status line announcing an eager fallback switch."""
    if is_upstream:
        _upstream_name = (classified.error_context or {}).get("upstream_provider", "aggregator")
        return f"⚠️ Upstream {_upstream_name} rate-limited — switching to fallback model..."
    if classified.reason == FailoverReason.billing:
        if classified.billing_unverified:
            # Ambiguous body — don't assert billing.
            return (
                "⚠️ Provider reported usage/credit exhaustion "
                "(unverified — may be a content-filter rejection) "
                "— switching to fallback provider..."
            )
        return "⚠️ Billing or credits exhausted — switching to fallback provider..."
    if is_transport_failure:
        return "⚠️ Provider unreachable — switching to fallback provider..."
    return "⚠️ Rate limited — switching to fallback provider..."


def _is_genuine_nous_rate_limit(agent: Any, api_error: Exception, error_context: Any, classified: Any = None) -> bool:
    """Record a genuine account-level Nous 429 to the cross-session breaker; upstream
    capacity 429s (no exhausted bucket in headers or last-known state) are left alone.

    *error_context* is the turn's (``extract_api_error_context``); *classified* brings the
    classifier's own context, where a welcome-tier ``rate_limited`` refusal and its ``reset_at``
    live. A long welcome reset is an exhausted allowance whatever the headers say, and the one
    place the user is told that signing in lifts it."""
    _genuine = False
    try:
        from agent.nous_rate_guard import (
            is_genuine_nous_rate_limit, is_long_welcome_rate_limit, record_nous_rate_limit)
        _err_resp = getattr(api_error, "response", None)
        _err_hdrs = getattr(_err_resp, "headers", None) if _err_resp else None
        from hermes_cli.anon_auth import is_anonymous_agent
        anonymous = is_anonymous_agent(agent)
        _classified_ctx = getattr(classified, "error_context", None) or {}
        # Only an anonymous request's fairshare body is an allowance verdict; named
        # requests keep the exhausted-bucket rule, whatever their host or body says.
        _genuine = (
            (anonymous and is_long_welcome_rate_limit(_classified_ctx))
            or is_genuine_nous_rate_limit(headers=_err_hdrs, last_known_state=agent._rate_limit_state))
        if _genuine:
            _merged = {**(error_context if isinstance(error_context, dict) else {}), **_classified_ctx}
            record_nous_rate_limit(headers=_err_hdrs, error_context=_merged, anonymous=anonymous)
        else:
            logger.info(
                "Nous 429 looks like upstream capacity "
                "(no exhausted bucket in headers or "
                "last-known state) -- not tripping "
                "cross-session breaker."
            )
    except Exception:
        pass
    return _genuine


def route_classified_error(
    agent: Any, api_error: Exception, classified: Any, _retry: TurnRetryState, *, error_msg: str,
    error_context: Any, recovered_with_pool: bool, base_url: Any, model: Any,
    messages: List[Dict[str, Any]], api_messages: Any, system_message: Any,
    active_system_prompt: Any, conversation_history: Any, retry_count: int, max_retries: int,
    compression_attempts: int, max_compression_attempts: int, api_call_count: int,
    effective_task_id: Any,
) -> ClassifiedErrorVerdict:
    """Ordered (load-bearing) recovery steps between classification and overflow handling:
    compaction-disabled overflow → terminal error (output-cap errors exempt); Anthropic
    long-context tier 429 → cap at 200k and compress; eager fallback for rate-limit/billing
    (immediately) and transport failures (after 1 retry) unless credential-pool rotation may
    still recover (upstream-aggregator 429s always fall back); persistent 401/403 → fallback
    chain once; genuine Nous 429 → cross-session breaker + re-enter the loop exactly once."""
    from agent.conversation_compression import conversation_history_after_compression
    from agent.conversation_loop import _arm_fallback_restart, _ra
    from agent.model_metadata import estimate_request_tokens_rough

    _provider_overflow_recovery_pending = False
    is_rate_limited = False
    _wrapped_output_cap_budget = None
    _is_zai_coding_overload = False
    status_code = getattr(api_error, "status_code", None)

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ClassifiedErrorVerdict:
        return ClassifiedErrorVerdict(
            action=action, result=result, status_code=status_code, messages=messages,
            active_system_prompt=active_system_prompt, conversation_history=conversation_history,
            retry_count=retry_count, max_retries=max_retries,
            compression_attempts=compression_attempts,
            provider_overflow_recovery_pending=_provider_overflow_recovery_pending,
            is_rate_limited=is_rate_limited, wrapped_output_cap_budget=_wrapped_output_cap_budget,
            is_zai_coding_overload=_is_zai_coding_overload,
        )

    def _fallback_break() -> ClassifiedErrorVerdict:
        nonlocal active_system_prompt, retry_count, compression_attempts
        active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
        retry_count = 0
        compression_attempts = 0
        return _verdict("break")

    # ``compression.enabled: false`` forbids every automatic trigger, incl. these
    # overflow recovery paths; error out. Output-cap errors exempt.
    _is_output_cap_error = (
        is_output_cap_error(error_msg) or parse_available_output_tokens_from_error(error_msg) is not None
    )
    if (
        classified.reason in _OVERFLOW_REASONS
        and not getattr(agent, "compression_enabled", True)
        and not _is_output_cap_error
    ):
        agent._flush_status_buffer()
        _vlines(
            agent,
            "❌ The conversation is too long for the model and automatic shrinking is off (compression.enabled: false).",
            "   💡 Run /compress to shrink it now, /new to start fresh, "
            "pick a model with a bigger context window, or remove attachments.",
        )
        logger.error(
            f"{agent.log_prefix}Context overflow ({classified.reason.value}) with "
            f"auto-compaction disabled — not compressing."
        )
        agent._persist_session(messages, conversation_history)
        _final_response = site_copy("compression_disabled", model=agent.model)
        return _verdict("return", stamp_failure({
            "final_response": _final_response, "messages": messages, "completed": False,
            "api_calls": api_call_count, "error": _final_response, "partial": True, "failed": True,
            "compaction_disabled": True,
        }, "context_overflow", False))

    # Anthropic 429 "Extra usage is required for long context requests" is a
    # subscription-tier limit, not transient: cap at 200k and compress.
    if classified.reason == FailoverReason.long_context_tier:
        old_ctx = _cap_long_context_tier(agent)
        compression_attempts += 1
        if compression_attempts <= max_compression_attempts:
            original_len = len(messages)
            # Overhead-aware request size so recovery arms on the true request
            # (msgs + tools + system), not the tool-blind message count.
            messages, active_system_prompt = agent._compress_context(
                # Route the overhead-aware _real_tokens (computed above) into compression, not the bare
                # last_prompt_tokens — which is 0 in the no-usage fallback, hiding the true request size
                # from the engine's overflow guard (upstream PR #77169 review).
                messages, system_message,
                approx_tokens=estimate_request_tokens_rough(api_messages, tools=agent.tools or None),
                task_id=effective_task_id,
            )
            conversation_history = conversation_history_after_compression(agent, messages, conversation_history)
            if len(messages) < original_len or old_ctx > _LONG_CONTEXT_TIER_CAP:
                agent._buffer_diagnostic_status(
                    COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE.format(
                        new_ctx=_LONG_CONTEXT_TIER_CAP, old_ctx=old_ctx
                    )
                )
                time.sleep(2)
                # Provider proved the request doesn't fit the reduced window; row count
                # isn't proof the rebuilt one does. Recheck before the next call.
                _provider_overflow_recovery_pending = True
                _retry.restart_with_compressed_messages = True
                return _verdict("break")
        # Compression exhausted or didn't help: fall through to normal error handling.

    # Eager fallback: rate-limit/billing switch immediately (primary won't recover in
    # the retry window); transport errors get 1 retry first.
    is_rate_limited = classified.reason in _RATE_LIMIT_REASONS
    # Some relays wrap upstream output-cap 400s as 429 (rate_limit). Only the max_tokens
    # clamp fixes it. Parsed once; gates the eager-fallback exemption and overflow entry.
    # Relay-wrapped output-cap errors: some gateways wrap an upstream "[400]: max_tokens (...) exceeds
    # model's maximum output tokens (...)" as HTTP 429, which classifies as rate_limit. The failure is a
    # deterministic request-shape problem — falling back to another provider (or burning generic retries)
    # can't fix it, but the output-cap clamp below can, in one retry (#72281). Parse once here; the result
    # gates both the eager-fallback exemption and the widened is_context_length_error entry, and is reused
    # as available_out inside the handler.
    _wrapped_output_cap_budget = (
        parse_available_output_tokens_from_error(error_msg)
        if classified.reason == FailoverReason.rate_limit else None
    )
    _is_transport_failure = classified.reason in _TRANSPORT_FAILURE_REASONS
    # Z.AI overload 429s classify `overloaded`, which `is_rate_limited` excludes. Detect
    # directly so the long backoff runs, and raise the ceiling to reach it.
    _is_zai_coding_overload = is_zai_coding_overload_error(base_url=str(base_url), model=model, error=api_error)
    if _is_zai_coding_overload:
        max_retries = max(max_retries, zai_coding_overload_retry_ceiling())
    _should_fallback = (
        (is_rate_limited and _wrapped_output_cap_budget is None)
        or (_is_transport_failure and retry_count >= 2)
    )
    if _should_fallback and agent._fallback_index < len(agent._fallback_chain):
        # No eager fallback while credential pool rotation may recover. Exception: an
        # upstream-aggregator 429 — the pool can't help, always fall back.
        # Fixes #11314.
        _is_upstream = classified.reason == FailoverReason.upstream_rate_limit
        pool_may_recover = (
            False if _is_upstream else _ra()._pool_may_recover_from_rate_limit(agent._credential_pool)
        )
        if not pool_may_recover:
            agent._buffer_diagnostic_status(_eager_fallback_status(classified, _is_upstream, _is_transport_failure))
            if agent._try_activate_fallback(reason=classified.reason):
                return _fallback_break()

    # A 401/403 surviving credential refresh means a broken credential or endpoint:
    # escalate to the fallback chain once; False -> terminal handling.
    if (
        classified.is_auth
        and not _retry.auth_failover_attempted
        and agent._fallback_index < len(agent._fallback_chain)
    ):
        _retry.auth_failover_attempted = True
        agent._buffer_diagnostic_status(
            "🔐 Authentication failed and could not be refreshed — "
            "switching to fallback provider..."
        )
        if agent._try_activate_fallback(reason=classified.reason):
            return _fallback_break()

    # Nous Portal: a genuine account-level 429 is recorded to a shared file so ALL
    # sessions back off; is_genuine_nous_rate_limit excludes upstream 429s.
    if (
        is_rate_limited
        and agent.provider == "nous"
        and classified.reason == FailoverReason.rate_limit
        and not recovered_with_pool
        and _is_genuine_nous_rate_limit(agent, api_error, error_context, classified)
    ):
        # Re-enter the loop exactly once so the top-of-loop Nous guard runs
        # (retry_count = max_retries would skip it entirely).
        retry_count = max(0, max_retries - 1)
        return _verdict("continue")
    # Upstream capacity 429: normal retry logic will typically succeed.
    return _verdict("fallthrough")
