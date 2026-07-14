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

import json
import logging
import os
import re
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Optional

from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout
from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH
from agent.error_classifier import FailoverReason
from agent.gemini_native_adapter import is_native_gemini_base_url
from agent.model_metadata import is_local_endpoint
from agent.message_sanitization import (
    _sanitize_surrogates,
    _repair_tool_call_arguments,
)
from tools.terminal_tool import is_persistent_env
from utils import base_url_host_matches, base_url_hostname, env_float, env_int

logger = logging.getLogger(__name__)
_OPENROUTER_PROVIDER_SORT_VALUES = {"throughput", "latency", "price"}

# When the fallback chain is fully exhausted on a non-rate-limit failure
# (e.g. every provider returns a non-retryable client error like HTTP 400),
# arm a short cooldown so the NEXT turn's restore_primary_runtime stays gated
# and does not reset _fallback_index=0 to replay the entire chain again.
# Without this, a client/gateway that re-submits immediately would re-marshal
# the full (potentially 80k-token) context once per provider every turn and
# can drive a constrained host into memory/swap exhaustion.  Rate-limit /
# billing reasons keep their own 60s cooldown (set above); this is the
# narrower non-rate-limit case.  See issue #24996.
_FALLBACK_EXHAUSTED_COOLDOWN_S = 5.0


def _ra():
    """Lazy ``run_agent`` reference.

    Used to honor test patches like
    ``patch("run_agent.cleanup_vm")`` / ``patch("run_agent.cleanup_browser")``
    that target symbols imported into ``run_agent``'s namespace.
    """
    import run_agent
    return run_agent


def estimate_request_context_tokens(api_payload: Any) -> int:
    """Estimate context/load tokens from an API payload, dict or messages list.

    The stale-call detectors historically assumed a Chat Completions request:
    they pulled ``api_kwargs["messages"]`` and ran a cheap char/4 estimate.
    Codex / Responses API requests carry the conversational payload in
    ``input`` (with additional load in ``instructions`` and ``tools``), so the
    legacy estimator reported ~0 tokens for every Codex turn and the
    context-tier scaling never fired.

    This helper handles both shapes:
      - bare list -> treat as Chat Completions ``messages``
      - dict with ``messages`` -> Chat Completions (+ ``tools`` if present)
      - dict with ``input`` -> Responses API (+ ``instructions``/``tools``)
      - any other dict -> fall back to summing string values
    """

    def _chars(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        return len(str(value))

    def _message_chars(messages: Any) -> int:
        if not isinstance(messages, list):
            return _chars(messages)
        return sum(_chars(item) for item in messages)

    if isinstance(api_payload, list):
        return _message_chars(api_payload) // 4

    if isinstance(api_payload, dict):
        messages = api_payload.get("messages")
        if isinstance(messages, list):
            total_chars = _message_chars(messages)
            if "tools" in api_payload:
                total_chars += _chars(api_payload.get("tools"))
            return total_chars // 4

        if "input" in api_payload:
            total_chars = (
                _chars(api_payload.get("input"))
                + _chars(api_payload.get("instructions"))
                + _chars(api_payload.get("tools"))
            )
            return total_chars // 4

        return sum(_chars(value) for value in api_payload.values()) // 4

    return _chars(api_payload) // 4


def _is_openai_codex_backend(agent) -> bool:
    base_url_lower = str(getattr(agent, "_base_url_lower", "") or "")
    base_url_hostname = str(getattr(agent, "_base_url_hostname", "") or "")
    return (
        getattr(agent, "provider", None) == "openai-codex"
        or (
            base_url_hostname == "chatgpt.com"
            and "/backend-api/codex" in base_url_lower
        )
    )


def openai_codex_stale_timeout_floor(est_tokens: int) -> float:
    """Minimum wall-clock stale timeout for openai-codex by estimated context.

    Gateway/Telegram sessions routinely ship ~15–25k tokens of tools +
    instructions before the first user message. Subscription-backed Codex can
    legitimately spend several minutes in backend admission/prefill at that
    size; the generic 90s non-stream stale default aborts healthy calls. The
    floor engages above 10k estimated tokens so those gateway-scale payloads
    are covered; smaller requests keep the generic default.
    """
    if est_tokens > 100_000:
        return 1200.0
    if est_tokens > 50_000:
        return 900.0
    if est_tokens > 10_000:
        return 600.0
    return 0.0


def _validated_openrouter_provider_sort(raw_sort: Any) -> Optional[str]:
    """Return a normalized OpenRouter provider.sort value or None."""
    if not isinstance(raw_sort, str):
        return None
    sort_value = raw_sort.strip().lower()
    if not sort_value:
        return None
    if sort_value in _OPENROUTER_PROVIDER_SORT_VALUES:
        return sort_value
    logger.warning(
        "Ignoring invalid OpenRouter provider.sort value %r (allowed: %s)",
        raw_sort,
        ", ".join(sorted(_OPENROUTER_PROVIDER_SORT_VALUES)),
    )
    return None


def _provider_preferences_for_agent(agent) -> Dict[str, Any]:
    """Build the validated provider-routing object shared by request paths."""
    preferences: Dict[str, Any] = {}
    if agent.providers_allowed:
        preferences["only"] = agent.providers_allowed
    if agent.providers_ignored:
        preferences["ignore"] = agent.providers_ignored
    if agent.providers_order:
        preferences["order"] = agent.providers_order
    provider_sort = _validated_openrouter_provider_sort(agent.provider_sort)
    if provider_sort:
        preferences["sort"] = provider_sort
    if agent.provider_require_parameters:
        preferences["require_parameters"] = True
    if agent.provider_data_collection:
        preferences["data_collection"] = agent.provider_data_collection
    return preferences


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ── Cross-turn stale-call circuit breaker (#58962) ─────────────────────
# A session wedged against an unresponsive provider hits the stale detector
# on every call and loops forever (observed: 494 consecutive failures over
# 3+ days, each burning the full stale timeout × retries with no response).
# The agent carries ``_consecutive_stale_streams``: incremented on every
# stale kill, reset only when a call actually completes (or when the
# provider is swapped — switch_model / try_activate_fallback /
# restore_primary_runtime — since the streak measured the OLD provider).
# Past the give-up threshold, calls abort immediately with an actionable
# error instead of re-waiting out the stale timeout.

def _stale_streak(agent) -> int:
    try:
        return int(getattr(agent, "_consecutive_stale_streams", 0) or 0)
    except Exception:
        return 0


def _bump_stale_streak(agent) -> None:
    try:
        agent._consecutive_stale_streams = _stale_streak(agent) + 1
    except Exception:
        pass


def _reset_stale_streak(agent) -> None:
    try:
        agent._consecutive_stale_streams = 0
    except Exception:
        pass


def _check_stale_giveup(agent) -> None:
    """Raise immediately when the consecutive-stale streak is past the
    give-up threshold — no network attempt, no stale-timeout wait."""
    _giveup = env_int("HERMES_STREAM_STALE_GIVEUP", 5)
    _streak = _stale_streak(agent)
    if _giveup > 0 and _streak >= _giveup:
        raise RuntimeError(
            "Provider has been unresponsive (no response received) for "
            f"{_streak} consecutive stale attempts — aborting this call to "
            "avoid an indefinite stall. Switch models or start a new "
            "session, then retry."
        )


def _dispatch_nonstreaming_api_request(agent, api_kwargs: dict, *, make_client):
    """Run one non-streaming LLM request for the active api_mode and return it.

    Shared by the interrupt-worker path (``interruptible_api_call``) and the
    inline path (``direct_api_call``) so the per-api_mode dispatch — codex /
    anthropic / bedrock / MoA / OpenAI-compatible — lives in exactly one place.

    ``make_client(reason)`` builds the per-request OpenAI client for the codex
    and OpenAI-compatible branches; the worker path uses it to register the
    client with its stranger-thread abort machinery, the inline path uses it to
    capture the client for its own ``finally`` close. The anthropic / bedrock /
    MoA branches manage their own clients and never call it. All interrupt,
    abort, cancellation, and close semantics stay in the callers — this helper
    only issues the request.
    """
    if agent.api_mode == "codex_responses":
        request_client = make_client("codex_stream_request")
        return agent._run_codex_stream(
            api_kwargs,
            client=request_client,
            on_first_delta=getattr(agent, "_codex_on_first_delta", None),
        )
    if agent.api_mode == "anthropic_messages":
        return agent._anthropic_messages_create(api_kwargs)
    if agent.api_mode == "bedrock_converse":
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
        return normalize_converse_response(raw_response)
    if agent.provider == "moa":
        # MoA is a virtual chat-completions provider backed by the
        # in-process MoAClient facade. Do not rebuild a request-local
        # OpenAI client from the virtual runtime metadata.
        return agent.client.chat.completions.create(**api_kwargs)
    request_client = make_client("chat_completion_request")
    return request_client.chat.completions.create(**api_kwargs)


def should_use_direct_api_call(agent) -> bool:
    """Whether a cron OpenAI-wire request should skip the interrupt worker.

    Issue #62151 is specific to OpenRouter's chat-completions path inside the
    gateway cron thread stack. Keep native/Codex/Bedrock/MoA transports on their
    established workers: their cancellation and client ownership differ, and
    the report provides no evidence that those paths share the pre-HTTP wedge.
    """
    return (
        getattr(agent, "platform", None) == "cron"
        and getattr(agent, "api_mode", None) == "chat_completions"
        and getattr(agent, "provider", None) != "moa"
    )


def direct_api_call(agent, api_kwargs: dict):
    """Run a non-streaming LLM call inline on the conversation thread.

    Used when ``should_use_direct_api_call`` is True. Skips the interrupt worker
    (whose only job is interactive-interrupt responsiveness, which this context
    does not have) so the nested-pool deadlock (#62151) cannot occur. Because the
    request runs in-flight normally, the per-request OpenAI client's own httpx
    timeout (provider ``request_timeout_seconds`` / ``HERMES_API_TIMEOUT``) bounds
    a genuinely hung provider — the same bound interactive calls already rely on.
    """
    _check_stale_giveup(agent)
    agent._touch_activity("waiting for non-streaming API response")
    request_client_holder = {"client": None}
    request_client_lock = threading.Lock()

    def _abort_active_request(reason: str) -> None:
        """Abort the inline request from cron's watchdog/interrupt thread."""
        with request_client_lock:
            request_client = request_client_holder["client"]
        if request_client is not None:
            agent._abort_request_openai_client(request_client, reason=reason)

    def _make_client(reason: str):
        client = agent._create_request_openai_client(reason=reason, api_kwargs=api_kwargs)
        with request_client_lock:
            request_client_holder["client"] = client
        agent._active_request_abort = _abort_active_request
        return client

    try:
        response = _dispatch_nonstreaming_api_request(
            agent, api_kwargs, make_client=_make_client
        )
    except Exception:
        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError("Agent interrupted during API call") from None
        raise
    else:
        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError("Agent interrupted during API call")
        _reset_stale_streak(agent)
        return response
    finally:
        if getattr(agent, "_active_request_abort", None) is _abort_active_request:
            agent._active_request_abort = None
        with request_client_lock:
            request_client = request_client_holder["client"]
            request_client_holder["client"] = None
        if request_client is not None:
            agent._close_request_openai_client(request_client, reason="request_complete")


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
    # Cron and other non-interactive, nested-pool contexts must not spawn the
    # interrupt worker — it wedges before the socket opens on the 2nd+ call
    # (#62151). Run inline instead. See should_use_direct_api_call.
    if should_use_direct_api_call(agent):
        return direct_api_call(agent, api_kwargs)

    result = {"response": None, "error": None}

    # Cross-turn stale-call circuit breaker (#58962) — non-streaming sibling
    # of the guard in interruptible_streaming_api_call.  Quiet-mode /
    # subagent / no-stream-consumer sessions take THIS path, and a wedged
    # unattended session here has the same infinite stale-retry class.
    _check_stale_giveup(agent)

    request_client_holder = {"client": None, "owner_tid": None}
    request_client_lock = threading.Lock()
    # Request-local cancellation flag. Distinct from agent._interrupt_requested
    # because that flag is cleared at run_conversation() turn boundaries, but
    # this daemon worker thread can outlive the turn (the gateway caches
    # AIAgent instances per session). Tracks whether THIS specific request was
    # cancelled by the main thread's interrupt handler, so the transport error
    # that is the expected consequence of our own force-close isn't misread as
    # a network bug and surfaced to the caller. (PR #6600 — cascading interrupt
    # hang.)
    _request_cancelled = {"value": False}

    def _set_request_client(client):
        with request_client_lock:
            request_client_holder["client"] = client
            # #29507: stamp the owning thread so a stranger-thread interrupt
            # only shuts the connection down rather than racing the worker
            # for FD ownership during ``client.close()``.
            request_client_holder["owner_tid"] = threading.get_ident()
        return client

    def _close_request_client_once(reason: str) -> None:
        # #29507: dispatch on the calling thread.
        #
        # When ``_call`` (the worker) reaches its ``finally`` it owns the
        # close and we pop + fully close as before. When a *stranger* thread
        # (the interrupt-check loop, the stale-call detector) drives the
        # close, only shut the sockets down so the worker's blocked
        # ``recv``/``send`` unwinds with an ``EPIPE`` / EOF — and let the
        # worker close ``client`` from its own thread on its way out. That
        # avoids the FD-recycling race where the kernel reassigned a
        # just-closed TLS socket FD to ``kanban.db``, and the still-live SSL
        # BIO on the worker thread then wrote a 24-byte TLS application-data
        # record into the SQLite header (#29507).
        with request_client_lock:
            request_client = request_client_holder.get("client")
            owner_tid = request_client_holder.get("owner_tid")
            stranger_thread = (
                request_client is not None
                and owner_tid is not None
                and owner_tid != threading.get_ident()
            )
            if not stranger_thread:
                # Owning thread (or no recorded owner) → pop and fully close.
                request_client_holder["client"] = None
                request_client_holder["owner_tid"] = None
        if request_client is None:
            return
        if stranger_thread:
            agent._abort_request_openai_client(request_client, reason=reason)
        else:
            agent._close_request_openai_client(request_client, reason=reason)

    def _call():
        try:
            # _set_request_client registers each per-request OpenAI client with
            # the stranger-thread abort machinery above; the shared dispatch
            # helper builds it via this callback so the interrupt / stale-call
            # detectors can force-close the worker's connection.
            result["response"] = _dispatch_nonstreaming_api_request(
                agent,
                api_kwargs,
                make_client=lambda reason: _set_request_client(
                    agent._create_request_openai_client(
                        reason=reason, api_kwargs=api_kwargs
                    )
                ),
            )
        except Exception as e:
            # If the request was cancelled by the main thread's interrupt
            # handler, the transport error is the expected consequence of our
            # own force-close, NOT a network bug. Swallow it instead of
            # surfacing — the main thread raises InterruptedError. (#6600)
            if _request_cancelled["value"]:
                logger.debug(
                    "Non-streaming worker caught %s after request cancellation — "
                    "exiting without surfacing a network error.",
                    type(e).__name__,
                )
                return
            result["error"] = e
        finally:
            _close_request_client_once("request_complete")

    # ── Stale-call timeout (mirrors streaming stale detector) ────────
    # Non-streaming calls return nothing until the full response is
    # ready.  Without this, a hung provider can block for the full
    # httpx timeout (default 1800s) with zero feedback.  The stale
    # detector kills the connection early so the main retry loop can
    # apply richer recovery (credential rotation, provider fallback).
    _stale_timeout = agent._compute_non_stream_stale_timeout(api_kwargs)

    # ── Codex Responses stream watchdogs ────────────────────────────────
    # The chatgpt.com/backend-api/codex endpoint has an intermittent failure
    # mode where it accepts the connection but never emits a single stream
    # event (observed directly: 0 events, no HTTP status, the socket just
    # hangs). A fresh reconnect succeeds in ~2s, but the wall-clock stale
    # timeout (often 180–900s) makes us wait minutes before retrying. While no
    # stream event has arrived yet we apply a much shorter TTFB cutoff so the
    # main retry loop can reconnect promptly. Large subscription-backed Codex
    # requests can legitimately spend tens of seconds in backend admission /
    # prompt prefill before the first SSE event, so the no-byte TTFB watchdog
    # is disabled for large chatgpt.com/backend-api/codex requests. A second
    # failure mode emits an opening SSE frame and then stalls forever in SSL
    # read; for that we watch the gap since the last Codex stream event. This
    # matches Codex CLI's stream_idle_timeout model: any valid SSE event is
    # activity. Operators can tune via HERMES_CODEX_TTFB_TIMEOUT_SECONDS and
    # HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS (0 disables each).
    _codex_watchdog_enabled = agent.api_mode == "codex_responses"
    _openai_codex_backend = _is_openai_codex_backend(agent)
    _est_tokens_for_codex_watchdog = estimate_request_context_tokens(api_kwargs)
    if _codex_watchdog_enabled and _openai_codex_backend:
        _codex_floor = openai_codex_stale_timeout_floor(_est_tokens_for_codex_watchdog)
        if _codex_floor:
            _stale_timeout = max(_stale_timeout, _codex_floor)

    if _est_tokens_for_codex_watchdog > 100_000:
        _codex_idle_timeout_default = 180.0
    elif _est_tokens_for_codex_watchdog > 50_000:
        _codex_idle_timeout_default = 120.0
    elif _est_tokens_for_codex_watchdog > 10_000:
        _codex_idle_timeout_default = 60.0
    else:
        _codex_idle_timeout_default = 12.0

    # No-byte TTFB cutoff. The OpenAI SDK's own streaming read timeout is far
    # longer (openai 2.x DEFAULT_TIMEOUT.read = 600s), so a tight 12s default
    # killed subscription-backed Codex requests mid-prefill before the backend
    # had a chance to emit its first SSE event. Default to 120s — long enough to
    # clear normal backend admission / prompt prefill, short enough to still
    # reconnect promptly when the socket is genuinely wedged. Set
    # HERMES_CODEX_TTFB_TIMEOUT_SECONDS=0 to disable this watchdog entirely.
    _ttfb_enabled = _codex_watchdog_enabled
    _ttfb_timeout = _env_float("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", 120.0)
    if _ttfb_timeout <= 0:
        _ttfb_enabled = False
    elif _openai_codex_backend:
        _ttfb_disable_above = _env_float("HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS", 10_000.0)
        _ttfb_strict = os.environ.get("HERMES_CODEX_TTFB_STRICT", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if (
            not _ttfb_strict
            and _ttfb_disable_above > 0
            and _est_tokens_for_codex_watchdog >= _ttfb_disable_above
        ):
            _ttfb_enabled = False
            logger.info(
                "Disabling openai-codex no-byte TTFB watchdog for large request "
                "(context=~%s tokens >= %.0f). Waiting for backend response instead. "
                "Set HERMES_CODEX_TTFB_STRICT=1 to force early reconnects.",
                f"{_est_tokens_for_codex_watchdog:,}",
                _ttfb_disable_above,
            )
        else:
            _ttfb_cap = _env_float("HERMES_CODEX_TTFB_MAX_SECONDS", 120.0)
            if _ttfb_cap > 0 and _ttfb_timeout > _ttfb_cap:
                logger.info(
                    "Capping openai-codex no-byte TTFB timeout from %.0fs to %.0fs "
                    "(context=~%s tokens). Set HERMES_CODEX_TTFB_MAX_SECONDS to tune.",
                    _ttfb_timeout,
                    _ttfb_cap,
                    f"{_est_tokens_for_codex_watchdog:,}",
                )
                _ttfb_timeout = _ttfb_cap

    _codex_idle_enabled = _codex_watchdog_enabled
    _codex_idle_timeout = _env_float(
        "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS",
        _codex_idle_timeout_default,
    )
    if _codex_idle_timeout <= 0:
        _codex_idle_enabled = False

    if _codex_watchdog_enabled:
        # Reset before the worker starts so a marker left over from a previous
        # call on this agent can't be misread as first-byte for this one.
        agent._codex_stream_last_event_ts = None
        agent._codex_stream_last_progress_ts = None

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

        _elapsed = time.time() - _call_start

        # TTFB detector: the Codex stream has produced no event at all and
        # we're past the first-byte cutoff → the backend opened the
        # connection but isn't responding. Kill it so the retry loop can
        # reconnect (a fresh connection typically succeeds in seconds),
        # instead of waiting out the much longer wall-clock stale timeout.
        if (
            _ttfb_enabled
            and _elapsed > _ttfb_timeout
            and getattr(agent, "_codex_stream_last_event_ts", None) is None
        ):
            _silent_hint: Optional[str] = None
            _hint_fn = getattr(agent, "_codex_silent_hang_hint", None)
            if callable(_hint_fn):
                try:
                    _silent_hint = _hint_fn(model=api_kwargs.get("model"))
                except Exception:
                    _silent_hint = None
            logger.warning(
                "Codex stream produced no bytes within TTFB cutoff "
                "(%.0fs > %.0fs, model=%s). Backend accepted the connection "
                "but sent no stream events. Killing connection so the retry "
                "loop can reconnect.",
                _elapsed, _ttfb_timeout, api_kwargs.get("model", "unknown"),
            )
            if _silent_hint:
                agent._buffer_status(
                    f"⚠️ No first byte from provider in {int(_elapsed)}s "
                    f"(codex stream, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Reconnecting. {_silent_hint}"
                )
            else:
                agent._buffer_status(
                    f"⚠️ No first byte from provider in {int(_elapsed)}s "
                    f"(codex stream, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Reconnecting."
                )
            try:
                _close_request_client_once("codex_ttfb_kill")
            except Exception:
                pass
            agent._touch_activity(
                f"codex stream killed after {int(_elapsed)}s with no first byte"
            )
            # Wait briefly for the worker to notice the closed connection.
            t.join(timeout=2.0)
            if result["error"] is None and result["response"] is None:
                if _silent_hint:
                    result["error"] = TimeoutError(
                        f"Codex stream produced no bytes within {int(_elapsed)}s "
                        f"(TTFB threshold: {int(_ttfb_timeout)}s). {_silent_hint}"
                    )
                else:
                    result["error"] = TimeoutError(
                        f"Codex stream produced no bytes within {int(_elapsed)}s "
                        f"(TTFB threshold: {int(_ttfb_timeout)}s)"
                    )
            break

        # Stream-idle detector: the Codex backend emitted at least one SSE
        # frame, then stopped emitting events. Valid keepalive / in_progress
        # frames refresh _codex_stream_last_event_ts and should not be killed.
        _last_codex_event_ts = getattr(agent, "_codex_stream_last_event_ts", None)
        if (
            _codex_idle_enabled
            and _last_codex_event_ts is not None
            and (time.time() - _last_codex_event_ts) > _codex_idle_timeout
        ):
            _event_stale_elapsed = time.time() - _last_codex_event_ts
            logger.warning(
                "Codex stream produced no SSE events for %.0fs after first byte "
                "(threshold %.0fs, model=%s, context=~%s tokens). Killing "
                "connection so the retry loop can reconnect.",
                _event_stale_elapsed,
                _codex_idle_timeout,
                api_kwargs.get("model", "unknown"),
                f"{_est_tokens_for_codex_watchdog:,}",
            )
            agent._buffer_status(
                f"⚠️ Codex stream sent no events for {int(_event_stale_elapsed)}s "
                f"after first byte (model: {api_kwargs.get('model', 'unknown')}). "
                f"Reconnecting."
            )
            try:
                _close_request_client_once("codex_stream_idle_kill")
            except Exception:
                pass
            agent._touch_activity(
                f"codex stream killed after {int(_event_stale_elapsed)}s with no SSE events"
            )
            t.join(timeout=2.0)
            if result["error"] is None and result["response"] is None:
                result["error"] = TimeoutError(
                    f"Codex stream produced no SSE events for {int(_event_stale_elapsed)}s "
                    f"after first byte (threshold: {int(_codex_idle_timeout)}s)"
                )
            break

        # Stale-call detector: kill the connection if no response
        # arrives within the configured timeout.
        if _elapsed > _stale_timeout:
            _est_ctx = estimate_request_context_tokens(api_kwargs)
            _silent_hint: Optional[str] = None
            _hint_fn = getattr(agent, "_codex_silent_hang_hint", None)
            if callable(_hint_fn):
                try:
                    _silent_hint = _hint_fn(model=api_kwargs.get("model"))
                except Exception:
                    _silent_hint = None
            logger.warning(
                "Non-streaming API call stale for %.0fs (threshold %.0fs). "
                "model=%s context=~%s tokens. Killing connection.",
                _elapsed, _stale_timeout,
                api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
            )
            if _silent_hint:
                agent._buffer_status(
                    f"⚠️ No response from provider for {int(_elapsed)}s "
                    f"(non-streaming, model: {api_kwargs.get('model', 'unknown')}). "
                    f"{_silent_hint}"
                )
            else:
                agent._buffer_status(
                    f"⚠️ No response from provider for {int(_elapsed)}s "
                    f"(non-streaming, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Aborting call."
                )
            try:
                if agent.api_mode == "anthropic_messages":
                    agent._anthropic_client.close()
                    agent._rebuild_anthropic_client()
                else:
                    _close_request_client_once("stale_call_kill")
            except Exception:
                pass
            # Circuit breaker (#58962): count the stale kill.  See the
            # canonical comment block above ``_stale_streak()``.
            _bump_stale_streak(agent)
            agent._touch_activity(
                f"stale non-streaming call killed after {int(_elapsed)}s"
            )
            # Wait briefly for the thread to notice the closed connection.
            t.join(timeout=2.0)
            if result["error"] is None and result["response"] is None:
                if _silent_hint:
                    result["error"] = TimeoutError(
                        f"Non-streaming API call timed out after {int(_elapsed)}s "
                        f"with no response (threshold: {int(_stale_timeout)}s). "
                        f"{_silent_hint}"
                    )
                else:
                    result["error"] = TimeoutError(
                        f"Non-streaming API call timed out after {int(_elapsed)}s "
                        f"with no response (threshold: {int(_stale_timeout)}s)"
                    )
            break

        if agent._interrupt_requested:
            # Mark THIS request cancelled before force-closing so the worker's
            # exception handler recognizes the forced transport error as a
            # cancel and exits cleanly instead of surfacing a network error or
            # (in the streaming path) burning full retry cycles. (#6600)
            _request_cancelled["value"] = True
            logger.debug(
                "Force-closing httpx client due to interrupt (not a network error)."
            )
            # Force-close the in-flight worker-local HTTP connection to stop
            # token generation without poisoning the shared client used to
            # seed future retries.
            try:
                if agent.api_mode == "anthropic_messages":
                    agent._anthropic_client.close()
                    agent._rebuild_anthropic_client()
                else:
                    _close_request_client_once("interrupt_abort")
            except Exception:
                pass
            raise InterruptedError("Agent interrupted during API call")
    if result["error"] is not None:
        raise result["error"]
    # Success — clear the circuit breaker (#58962): the provider proved
    # responsive.  See the canonical comment block above ``_stale_streak()``.
    if result["response"] is not None:
        _reset_stale_streak(agent)
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
            or base_url_host_matches(agent.base_url, "githubcopilot.com")
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

        # xAI's /responses endpoint rejects ``pattern`` and ``format`` keywords
        # in tool schemas (HTTP 400 "Invalid arguments passed to the model").
        # Most commonly hit when MCP-derived tools carry JSON Schema validation
        # keywords through. Strip them before building kwargs. See #27197.
        # It also rejects ``enum`` values containing ``/`` (HuggingFace IDs
        # like ``Qwen/Qwen3.5-0.8B`` shipped by MCP servers) — same 400 with
        # the same opaque message; strip those enums too.
        #
        # Deep-copy ``tools_for_api`` before sanitizing: the sanitizers
        # mutate in place (documented contract on ``strip_slash_enum`` /
        # ``strip_pattern_and_format``), and ``tools_for_api`` is a direct
        # reference to ``agent.tools``.  Without the copy, the first xAI
        # request permanently strips constraints from the shared per-agent
        # tool registry — every subsequent non-xAI call from the same
        # agent (auxiliary task routed to Anthropic, OpenRouter fallback,
        # main-model swap) sees the already-stripped schema.  See #27907.
        if is_xai_responses:
            try:
                import copy as _copy
                from tools.schema_sanitizer import (
                    strip_pattern_and_format,
                    strip_slash_enum,
                )
                tools_for_api = _copy.deepcopy(tools_for_api)
                tools_for_api, _ = strip_pattern_and_format(tools_for_api)
                tools_for_api, _ = strip_slash_enum(tools_for_api)
            except Exception as exc:
                logger.warning(
                    "%s⚠️ Failed to sanitize tool schemas for xAI: %s",
                    getattr(agent, "log_prefix", ""), exc,
                )

        return _ct.build_kwargs(
            model=agent.model,
            messages=_msgs_for_codex,
            tools=tools_for_api,
            reasoning_config=agent.reasoning_config,
            session_id=getattr(agent, "session_id", None),
            max_tokens=agent.max_tokens,
            timeout=agent._resolved_api_call_timeout(),
            request_overrides=agent.request_overrides,
            is_github_responses=is_github_responses,
            is_codex_backend=is_codex_backend,
            is_xai_responses=is_xai_responses,
            github_reasoning_extra=agent._github_models_reasoning_extra_body() if is_github_responses else None,
            replay_encrypted_reasoning=bool(
                getattr(agent, "_codex_reasoning_replay_enabled", True)
            ),
        )

    # ── chat_completions (default) ─────────────────────────────────────
    _ct = agent._get_transport()

    # Provider detection flags
    _is_qwen = agent._is_qwen_portal()
    _is_or = agent._is_openrouter_url()
    _is_gh = (
        base_url_host_matches(agent._base_url_lower, "models.github.ai")
        or base_url_host_matches(agent._base_url_lower, "githubcopilot.com")
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

    # Provider preferences (aggregator profile decides whether to emit them).
    _prefs = _provider_preferences_for_agent(agent)

    # Anthropic-compatible max-output fallback (last resort only — applied in
    # build_kwargs *after* ephemeral/user/profile max_tokens, never overriding
    # an explicit value).  Model-gated, not URL-gated: any chat-completions
    # proxy serving a Claude/MiniMax/Qwen3 model needs max_tokens, because the
    # Anthropic Messages API treats it as mandatory and proxies that omit it
    # (AWS Bedrock, NVIDIA, LiteLLM, vLLM, corporate gateways) default as low
    # as 4096 output tokens — easily exhausted by thinking + large tool calls
    # like write_file/patch.  OpenRouter/Nous were the only routes covered
    # before; gating on _ANTHROPIC_OUTPUT_LIMITS membership covers them all.
    _ant_max = None
    try:
        from agent.anthropic_adapter import (
            _get_anthropic_max_output,
            _ANTHROPIC_OUTPUT_LIMITS,
        )
        _model_norm = (agent.model or "").lower().replace(".", "-")
        if any(key in _model_norm for key in _ANTHROPIC_OUTPUT_LIMITS):
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

    # Defence-in-depth: redact credentials (PATs, API keys, Bearer tokens)
    # from assistant content BEFORE the message enters conversation history.
    # If the model accidentally inlines a secret in its natural-language
    # response, catch it here at the persistence boundary so it never
    # reaches state.db, session_*.json, gateway delivery, or compression.
    # Respects HERMES_REDACT_SECRETS via redact_sensitive_text — no-op
    # when disabled. (#19798)
    if isinstance(_san_content, str) and _san_content:
        from agent.redact import redact_sensitive_text
        _san_content = redact_sensitive_text(_san_content)

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

    # Anthropic interleaved-thinking replay: when a turn interleaves signed
    # thinking blocks with tool_use, the parallel reasoning_details +
    # tool_calls fields lose the cross-type ordering, and reconstruction
    # front-loads thinking — reordering signed blocks and triggering HTTP 400
    # ("thinking ... blocks in the latest assistant message cannot be
    # modified"). Carry the verbatim ordered block list so the adapter can
    # replay the latest assistant message unchanged. See
    # agent/transports/anthropic.py and agent/anthropic_adapter.py.
    ordered_blocks = getattr(assistant_message, "anthropic_content_blocks", None)
    if ordered_blocks:
        msg["anthropic_content_blocks"] = ordered_blocks

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
            # Tool-call arguments are intentionally NOT redacted here. This
            # dict enters the in-memory conversation history that is replayed
            # to the model on every subsequent turn AND persisted to state.db,
            # which is itself replayed verbatim on session resume
            # (get_messages_as_conversation). Masking a credential to `***`
            # here poisons that replay: the model reads back its own
            # `PGPASSWORD='***' psql ...` call and copies the placeholder into
            # the next tool call, breaking every credential-dependent command
            # on the second turn (#43083). The masking also provided no real
            # protection — the same secret still leaks verbatim through tool
            # OUTPUT (file contents, command output, diffs, the compaction
            # block), none of which this pass ever touched. Keeping secrets
            # out of the replayable store is a separate tokenization/vault
            # concern, not something arg-redaction can deliver without
            # breaking replay. Storage-time redaction remains governed by the
            # `security.redact_secrets` toggle. (#19798 introduced this;
            # #43083 removed it.)
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



def rewrite_prompt_model_identity(agent, model: str, provider: str) -> None:
    """Point the cached system prompt's ``Model:``/``Provider:`` lines at
    the active runtime after a provider switch.

    The system prompt is session-stable and replayed verbatim for prefix-cache
    warmth, but after a failover the new backend's cache is cold anyway —
    while a stale identity line makes the agent misreport which model it is
    when asked.  Rewrite the lines in place WITHOUT persisting to the session
    DB: the stored row keeps the primary's labels, so when the primary is
    restored the prompt is byte-identical to the stored copy again and its
    prefix cache still matches.

    Only the LAST occurrence of each line is touched — the identity lines
    live in the volatile tail of the prompt, and earlier matches could be
    user content (memory snapshots, context files).
    """
    sp = getattr(agent, "_cached_system_prompt", None)
    if not isinstance(sp, str) or not sp:
        return
    for label, value in (("Model", model), ("Provider", provider)):
        if not value:
            continue
        matches = list(re.finditer(rf"(?m)^{label}: .*$", sp))
        if matches:
            last = matches[-1]
            sp = f"{sp[:last.start()]}{label}: {value}{sp[last.end():]}"
    agent._cached_system_prompt = sp


def _fallback_entry_key(fb: dict) -> tuple[str, str, str]:
    return (
        str(fb.get("provider") or "").strip().lower(),
        str(fb.get("model") or "").strip(),
        str(fb.get("base_url") or "").strip().rstrip("/"),
    )


def _fallback_entry_unavailable_without_network(agent, fb: dict) -> Optional[str]:
    """Return a skip reason for fallback entries known to be unusable locally."""
    fb_provider = (fb.get("provider") or "").strip().lower()
    if fb_provider != "nous":
        return None
    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
    except Exception as exc:
        return f"nous_auth_unreadable:{type(exc).__name__}"
    access_value = state.get("access_token")
    refresh_value = state.get("refresh_token")
    has_access = isinstance(access_value, str) and bool(access_value.strip())
    has_refresh = isinstance(refresh_value, str) and bool(refresh_value.strip())
    if not (has_access or has_refresh):
        return "nous_token_missing"
    return None



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
    if reason in {FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit}:
        # Only start cooldown when leaving the primary provider.  If we're
        # already on a fallback and chain-switching, the primary wasn't the
        # source of the 429 so the cooldown should not be reset/extended.
        fallback_already_active = bool(getattr(agent, "_fallback_activated", False))
        current_provider = (getattr(agent, "provider", "") or "").strip().lower()
        primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
        if (not fallback_already_active) or (primary_provider and current_provider == primary_provider):
            agent._rate_limited_until = time.monotonic() + 60
    if agent._fallback_index >= len(agent._fallback_chain):
        # Chain exhausted.  If we actually walked a non-empty chain and the
        # failure was NOT a rate-limit/billing event (those already armed
        # their own 60s cooldown above), arm a short cooldown so the next
        # turn's restore_primary_runtime stays gated instead of resetting
        # _fallback_index=0 and re-marshaling the whole context across every
        # provider again.  Guards the cross-turn replay storm in #24996.
        if (
            len(agent._fallback_chain) > 0
            and reason not in {FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit}
        ):
            _existing_cooldown = getattr(agent, "_rate_limited_until", 0) or 0
            agent._rate_limited_until = max(
                _existing_cooldown,
                time.monotonic() + _FALLBACK_EXHAUSTED_COOLDOWN_S,
            )
        return False
    fb = agent._fallback_chain[agent._fallback_index]
    agent._fallback_index += 1
    fb_key = _fallback_entry_key(fb)
    unavailable = getattr(agent, "_unavailable_fallback_keys", None)
    if unavailable is None:
        unavailable = set()
        agent._unavailable_fallback_keys = unavailable
    if fb_key in unavailable:
        logger.debug("Fallback skip: %s previously marked unavailable", fb_key)
        return agent._try_activate_fallback(reason)
    fb_provider = (fb.get("provider") or "").strip().lower()
    fb_model = (fb.get("model") or "").strip()
    if not fb_provider or not fb_model:
        return agent._try_activate_fallback(reason)  # skip invalid, try next

    local_skip_reason = _fallback_entry_unavailable_without_network(agent, fb)
    if local_skip_reason:
        unavailable.add(fb_key)
        logger.warning(
            "Fallback skip: %s/%s is not locally usable (%s); suppressing for this session",
            fb_provider,
            fb_model,
            local_skip_reason,
        )
        return agent._try_activate_fallback(reason)

    # Skip entries that resolve to the current (provider, model) — falling
    # back to the same backend that just failed loops the failure. Compare
    # base_url too so two distinct custom_providers entries pointing at the
    # same shim/proxy URL also dedup. See issue #22548.
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    current_model = (getattr(agent, "model", "") or "").strip()
    current_base_url = str(getattr(agent, "base_url", "") or "").rstrip("/").lower()
    fb_base_url_for_dedup = (fb.get("base_url") or "").strip().rstrip("/").lower()
    if fb_provider == current_provider and fb_model == current_model:
        logger.warning(
            "Fallback skip: chain entry %s/%s matches current provider/model",
            fb_provider, fb_model,
        )
        return agent._try_activate_fallback(reason)
    if (
        fb_base_url_for_dedup
        and current_base_url
        and fb_base_url_for_dedup == current_base_url
        and fb_model == current_model
    ):
        logger.warning(
            "Fallback skip: chain entry base_url %s matches current backend",
            fb_base_url_for_dedup,
        )
        return agent._try_activate_fallback(reason)

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
            logger.warning(
                "Fallback to %s failed: provider not configured",
                fb_provider)
            unavailable.add(fb_key)
            return agent._try_activate_fallback(reason)  # try next in chain
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider

            fb_model = normalize_model_for_provider(fb_model, fb_provider)
        except Exception as _norm_err:
            logger.warning(
                "Could not normalize fallback model %r for provider %r: %s",
                fb_model, fb_provider, _norm_err,
            )

        # Determine api_mode from provider / base URL / model
        fb_api_mode = "chat_completions"
        fb_base_url = str(fb_client.base_url)
        _fb_is_azure = agent._is_azure_openai_url(fb_base_url)
        if fb_provider == "openai-codex":
            fb_api_mode = "codex_responses"
        elif (
            fb_provider == "anthropic"
            or fb_base_url.rstrip("/").lower().endswith("/anthropic")
            or base_url_hostname(fb_base_url) == "api.anthropic.com"
        ):
            # Custom providers (e.g. cron-anthropic) point at the native
            # api.anthropic.com host with no "/anthropic" path suffix, so the
            # name/suffix checks above miss them and they default to
            # chat_completions → POST /v1/chat/completions → 404. Match the
            # host the same way determine_api_mode() and _detect_api_mode_for_url()
            # do on the primary path. (#32243, #49247)
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
        old_provider = agent.provider

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

        # Rebind the credential pool to the fallback provider when the provider
        # changes.  Keeping the primary pool attached would make downstream
        # recovery (rate_limit / billing / auth) mutate the wrong credential
        # set and can overwrite the fallback's base_url back to the primary
        # endpoint.  See #33163.
        #
        # When the fallback shares the pool's provider (e.g. both openrouter
        # entries with different routing) the pool is preserved.  When the
        # providers differ, load the fallback provider's own pool if one exists
        # so provider-specific rotation continues to work after the switch.
        _existing_pool = getattr(agent, "_credential_pool", None)
        if _existing_pool is not None:
            _pool_provider = (getattr(_existing_pool, "provider", "") or "").strip().lower()
            if _pool_provider and _pool_provider != fb_provider:
                logger.info(
                    "Fallback to %s/%s: clearing primary credential pool "
                    "(pool_provider=%s) to prevent cross-provider contamination",
                    fb_provider, fb_model, _pool_provider,
                )
                agent._credential_pool = None
        if getattr(agent, "_credential_pool", None) is None:
            try:
                from agent.credential_pool import load_pool

                fallback_pool = load_pool(fb_provider)
                if fallback_pool and fallback_pool.has_credentials():
                    agent._credential_pool = fallback_pool
                    logger.info(
                        "Fallback to %s/%s: attached fallback credential pool",
                        fb_provider, fb_model,
                    )
            except Exception as exc:
                logger.debug(
                    "Fallback to %s/%s: could not attach credential pool: %s",
                    fb_provider, fb_model, exc,
                )

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
            # ``agent.api_key`` may be callable (Entra ID); the
            # context-length resolver expects a string for live
            # probes. Foundry typically resolves via config/static
            # catalogs anyway, so coerce defensively.
            _fb_ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
            fb_context_length = get_model_context_length(
                agent.model, base_url=agent.base_url,
                api_key=_fb_ctx_api_key, provider=agent.provider,
                config_context_length=getattr(agent, "_config_context_length", None),
                custom_providers=getattr(agent, "_custom_providers", None),
            )
            agent.context_compressor.update_model(
                model=agent.model,
                context_length=fb_context_length,
                base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),  # callable preserved → call_llm
                provider=agent.provider,
                api_mode=agent.api_mode,
            )

        # Keep the prompt's self-identity in sync with the model actually
        # answering, so "what model are you?" doesn't report the primary.
        rewrite_prompt_model_identity(agent, fb_model, fb_provider)

        agent._buffer_status(
            f"🔄 Primary model failed — switching to fallback: "
            f"{fb_model} via {fb_provider}"
        )
        # The buffered line above is dropped on successful recovery, but a
        # provider/model switch is a durable state change operators must see
        # even when the fallback succeeds.  Record a one-shot notice that the
        # success path surfaces exactly once via _emit_pending_fallback_notice
        # (see run_agent.py); it is discarded on terminal failure since the
        # buffered line is flushed instead.  See fallback-observability fix.
        agent._pending_fallback_notice = (
            f"🔄 Switched to fallback model: {old_model} via {old_provider} "
            f"→ {fb_model} via {fb_provider}"
        )
        logger.info(
            "Fallback activated: %s → %s (%s)",
            old_model, fb_model, fb_provider,
        )
        # Reset the stale-call circuit breaker (#58962): the streak measured
        # the OLD provider's unresponsiveness.  Carrying it over would
        # short-circuit the freshly activated fallback before it gets a
        # single stream attempt.
        _reset_stale_streak(agent)
        return True
    except Exception as e:
        if fb_provider == "nous":
            unavailable.add(fb_key)
        logger.error("Failed to activate fallback %s: %s", fb_model, e)
        return agent._try_activate_fallback(reason)  # try next in chain



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
            # Strict OpenAI-compatible gateways (Fireworks-backed OpenCode Go,
            # Mistral, Moonshot/Kimi) reject any message key outside the Chat
            # Completions schema. The main loop drops these via
            # ChatCompletionsTransport.convert_messages(), but the summary path
            # hand-builds messages and calls chat.completions.create() directly,
            # bypassing the transport — so mirror that sanitization here:
            # tool_name (SQLite FTS bookkeeping), the codex_* reasoning carriers,
            # timestamp (preserved on gateway user replay entries for the
            # stale-confirmation expiry check — #47868 rejection class),
            # and every Hermes-internal underscore-prefixed scaffolding key.
            for schema_foreign in ("tool_name", "codex_reasoning_items", "codex_message_items", "timestamp"):
                api_msg.pop(schema_foreign, None)
            for internal_key in [k for k in api_msg if isinstance(k, str) and k.startswith("_")]:
                api_msg.pop(internal_key, None)
            if _needs_sanitize:
                agent._sanitize_tool_calls_for_strict_api(api_msg, model=agent.model)
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

            # Merge the profile's canonical body even when routing is unset:
            # profiles may always emit required metadata such as Portal tags.
            provider_preferences = _provider_preferences_for_agent(agent)
            profile_extra_body = {}
            try:
                from providers import get_provider_profile

                provider_profile = get_provider_profile(agent.provider)
                if provider_profile is not None:
                    profile_extra_body = provider_profile.build_extra_body(
                        session_id=getattr(agent, "session_id", None),
                        provider_preferences=provider_preferences or None,
                        model=agent.model,
                        base_url=agent.base_url,
                        reasoning_config=agent.reasoning_config,
                    )
            except Exception:
                pass

            if profile_extra_body:
                summary_extra_body.update(profile_extra_body)
            if provider_preferences and "provider" not in profile_extra_body and (
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
        logger.warning(f"Failed to get summary response: {e}")
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
            logger.warning(f"Failed to cleanup VM for task {task_id}: {e}")
    try:
        _ra().cleanup_browser(task_id)
    except Exception as e:
        if agent.verbose_logging:
            logger.warning(f"Failed to cleanup browser for task {task_id}: {e}")




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

    # Cron and other non-interactive, nested-pool contexts deadlock on the
    # spawned worker thread (#62151). They also have no stream consumer, so the
    # deltas this path produces go nowhere. Delegate to the non-streaming entry
    # (which runs inline via should_use_direct_api_call) exactly like the codex
    # branch below — routing through the _interruptible_api_call method keeps the
    # outer loop's per-request retry/refresh seam intact.
    if should_use_direct_api_call(agent):
        return agent._interruptible_api_call(api_kwargs)

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
                    is_streaming_access_denied_error,
                    normalize_converse_response,
                    stream_converse_with_callbacks,
                )
                region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                api_kwargs.pop("__bedrock_converse__", None)
                client = _get_bedrock_runtime_client(region)
                try:
                    raw_response = client.converse_stream(**api_kwargs)
                except Exception as _bedrock_exc:
                    # IAM policies scoped to bedrock:InvokeModel only (no
                    # InvokeModelWithResponseStream) reject converse_stream()
                    # with AccessDeniedException. That denial is permanent for
                    # the session — fall back to the non-streaming converse()
                    # inline (it maps to bedrock:InvokeModel) and disable
                    # streaming for subsequent calls so we don't re-fail every
                    # turn.
                    if is_streaming_access_denied_error(_bedrock_exc):
                        agent._disable_streaming = True
                        agent._safe_print(
                            "\n⚠  AWS IAM denied bedrock:InvokeModelWithResponseStream — "
                            "falling back to non-streaming InvokeModel.\n"
                            "   Grant that action to restore streaming output.\n"
                        )
                        logger.info(
                            "bedrock: converse_stream denied by IAM (%s) — "
                            "using non-streaming converse() for this session.",
                            type(_bedrock_exc).__name__,
                        )
                        result["response"] = normalize_converse_response(
                            client.converse(**api_kwargs)
                        )
                        return
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
        # Worker exited before the poll loop observed the interrupt flag. The
        # Bedrock stream callback breaks out and returns a PARTIAL response
        # without raising on interrupt (see bedrock_adapter.py
        # stream_converse_with_callbacks / on_interrupt_check), so result[
        # "response"] is populated with error=None and the in-loop raise above
        # never fires. Re-check here so /stop is not silently swallowed on the
        # Bedrock path — mirrors the post-worker guard on the main streaming
        # loop. (#59999 area)
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted during Bedrock API call (post-worker)")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    result = {"response": None, "error": None, "partial_tool_names": []}

    # Cross-turn stale-stream circuit breaker (#58962) — see the canonical
    # comment block above ``_stale_streak()``.  Raises past the give-up
    # threshold instead of burning another stale-timeout×retries cycle.
    _check_stale_giveup(agent)

    request_client_holder = {"client": None, "diag": None, "owner_tid": None}
    request_client_lock = threading.Lock()
    # Request-local cancellation flag — see interruptible_api_call for the full
    # rationale. The streaming retry loop is where the 7-minute cascading-
    # interrupt hang originated: a force-close raised RemoteProtocolError, the
    # loop classified it as a transient network error, and burned full retry
    # cycles (and emitted "reconnecting" noise) on a request the user already
    # cancelled. The token lets the worker recognize its own forced close and
    # exit immediately instead of retrying. (PR #6600.)
    _request_cancelled = {"value": False}

    def _set_request_client(client):
        with request_client_lock:
            request_client_holder["client"] = client
            # See #29507 explanation in the non-streaming variant above.
            request_client_holder["owner_tid"] = threading.get_ident()
        return client

    def _close_request_client_once(reason: str) -> None:
        # See #29507 explanation in the non-streaming variant above. A
        # stranger thread (the interrupt-check / stale-stream detector loop)
        # only aborts sockets — never pops, never calls ``client.close()`` —
        # so the worker thread retains ownership of the FD release.
        with request_client_lock:
            request_client = request_client_holder.get("client")
            owner_tid = request_client_holder.get("owner_tid")
            stranger_thread = (
                request_client is not None
                and owner_tid is not None
                and owner_tid != threading.get_ident()
            )
            if not stranger_thread:
                request_client_holder["client"] = None
                request_client_holder["owner_tid"] = None
        if request_client is None:
            return
        if stranger_thread:
            agent._abort_request_openai_client(request_client, reason=reason)
        else:
            agent._close_request_openai_client(request_client, reason=reason)

    first_delta_fired = {"done": False}
    deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)
    # Wall-clock timestamp of the last real streaming chunk.  The outer
    # poll loop uses this to detect stale connections that keep receiving
    # SSE keep-alive pings but no actual data.
    last_chunk_time = {"t": time.time()}
    # Stale-stream patience, shared between the httpx socket read timeout
    # (built in ``_call_chat_completions`` below) and the stale-stream detector
    # (computed further down, before the worker thread starts).  Initialized
    # here so the read-timeout builder can floor itself at the stale value and
    # never fire before the detector.  ``None`` until the detector value is
    # resolved, so the builder degrades to its plain default if it ever runs
    # first.
    _stream_stale_timeout = None

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
            else env_float("HERMES_API_TIMEOUT", 1800.0)
        )
        # Read timeout: config wins here too.  Otherwise use
        # HERMES_STREAM_READ_TIMEOUT (default 120s) for cloud providers.
        if _provider_timeout_cfg is not None:
            _stream_read_timeout = _provider_timeout_cfg
        else:
            _stream_read_timeout = env_float("HERMES_STREAM_READ_TIMEOUT", 120.0)
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
            elif (
                _stream_read_timeout == 120.0
                and _stream_stale_timeout is not None
                and _stream_stale_timeout != float("inf")
                and _stream_stale_timeout > _stream_read_timeout
            ):
                # Cloud reasoning models (e.g. Opus) routinely pause mid-stream
                # for minutes during extended thinking.  The stale-stream
                # detector is deliberately scaled up to tolerate this (180–300s,
                # see the stale-timeout block below), but the raw httpx socket
                # read timeout defaulted to a flat 120s and fired *first* —
                # tearing down a healthy reasoning stream before the stale
                # detector (which owns retry + diagnostics) could act.  Keep the
                # socket read timeout in step with the detector so it no longer
                # preempts it.
                _stream_read_timeout = _stream_stale_timeout
                logger.debug(
                    "Cloud reasoning stream — read timeout raised to %.0fs to "
                    "match stale-stream detector", _stream_read_timeout,
                )
        # Cap connect/pool at 60s even when provider timeout is higher.
        # connect/pool cover TCP handshake, not model inference.
        _conn_cap = min(_base_timeout, 60.0) if _provider_timeout_cfg is not None else 30.0
        stream_kwargs = {
            **api_kwargs,
            "stream": True,
            "timeout": _httpx.Timeout(
                connect=_conn_cap,
                read=_stream_read_timeout,
                write=_base_timeout,
                pool=_conn_cap,
            ),
        }
        # OpenAI's `stream_options={"include_usage": True}` drives usage
        # accounting on OpenAI-compatible endpoints (incl. the Gemini OpenAI
        # compat shim and aggregators like OpenRouter).  Google's *native*
        # Gemini REST endpoint rejects the keyword outright
        # (`Completions.create() got an unexpected keyword argument
        # 'stream_options'`), so omit it only for that endpoint.
        if not is_native_gemini_base_url(agent.base_url):
            stream_kwargs["stream_options"] = {"include_usage": True}
        request_client = _set_request_client(
            agent._create_request_openai_client(
                reason="chat_completion_stream_request",
                api_kwargs=stream_kwargs,
            )
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
        stream = request_client.chat.completions.create(**stream_kwargs)

        # Some OpenAI-compatible adapters (for example copilot-acp, and the MoA
        # openai-codex aggregator) accept stream=True but still return a
        # completed response object rather than an iterator of chunks.  Treat
        # that as "streaming unsupported" for the rest of this session instead
        # of crashing on ``for chunk in stream`` with ``'types.SimpleNamespace'
        # object is not iterable`` (#11732, #55933).
        #
        # Discriminate on the mere PRESENCE of a ``choices`` attribute, not on
        # it being a non-empty list: an adapter may hand back a completed
        # response whose ``choices`` is ``None`` or empty (an error /
        # content-filter / terminal frame), and every such shape is still a
        # whole response — not a token stream — that would crash iteration just
        # the same.  A genuine provider stream (SDK ``Stream`` object,
        # generator) exposes no ``choices`` attribute, so it is left untouched.
        if hasattr(stream, "choices"):
            logger.info(
                "Streaming request returned a final response object instead of "
                "an iterator; switching %s/%s to non-streaming for this session.",
                agent.provider or "unknown",
                agent.model or "unknown",
            )
            agent._disable_streaming = True
            # An empty/None ``choices`` carries no message to surface; return the
            # completed object as-is so the outer loop's normal invalid-response
            # validation (conversation_loop.py) handles it via the retry path,
            # never ``for chunk in stream``.
            choices = stream.choices
            first_choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
            message = getattr(first_choice, "message", None)
            if message is not None:
                reasoning_text = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                )
                if isinstance(reasoning_text, str) and reasoning_text:
                    _fire_first_delta()
                    agent._fire_reasoning_delta(reasoning_text)
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    _fire_first_delta()
                    agent._fire_stream_delta(content)
            return stream

        # Capture rate limit headers from the initial HTTP response.
        # The OpenAI SDK Stream object exposes the underlying httpx
        # response via .response before any chunks are consumed.
        agent._capture_rate_limits(getattr(stream, "response", None))
        agent._capture_credits(getattr(stream, "response", None))
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
            agent._touch_activity("receiving stream response")
            _now_chunk = time.time()
            # Reset the stale-stream timer ONLY on chunks that carry real
            # progress (content / reasoning / tool-call deltas).  A
            # content-less SSE keepalive ping must NOT reset it: the
            # ChatGPT codex backend (chatgpt.com/backend-api/codex) keeps a
            # stalled stream alive with pings while delivering zero tokens.
            # Resetting on every raw chunk (the previous behaviour) let
            # those pings defeat both this detector AND the httpx read
            # timeout, leaving a hung request alive until the backend gave
            # up (~10 min) instead of killing+reconnecting at the stale
            # threshold (the immediate retry succeeds in seconds).
            _pg = chunk.choices[0].delta if chunk.choices else None
            if _pg is not None and (
                getattr(_pg, "content", None)
                or getattr(_pg, "reasoning_content", None)
                or getattr(_pg, "reasoning", None)
                or getattr(_pg, "tool_calls", None)
                or getattr(_pg, "function_call", None)
            ):
                last_chunk_time["t"] = _now_chunk

            # Update per-attempt diagnostic counters.  Best-effort —
            # failures are swallowed so the streaming hot path is never
            # interrupted by diagnostic accounting.
            try:
                _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                if _diag.get("first_chunk_at") is None:
                    _diag["first_chunk_at"] = _now_chunk
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
                        # Poolside may send integer id instead of string
                        _tc_id = tc_delta.id
                        if isinstance(_tc_id, int):
                            _tc_id = str(_tc_id)
                        tool_calls_acc[idx] = {
                            "id": _tc_id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                            "extra_content": None,
                        }
                    entry = tool_calls_acc[idx]
                    if tc_delta.id is not None:
                        _new_id = tc_delta.id
                        if isinstance(_new_id, int):
                            _new_id = str(_new_id)
                        if _new_id:
                            entry["id"] = _new_id
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
                        extra = (tc_delta.model_extra if isinstance(tc_delta.model_extra, dict) else {}).get("extra_content")
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

        # Zero-chunk guard: stream yielded nothing usable — a provider/upstream
        # error or malformed SSE, not a legitimate empty completion. Raise so the
        # retry machinery handles it instead of fabricating a successful turn.
        if (
            finish_reason is None
            and not content_parts
            and not reasoning_parts
            and not tool_calls_acc
        ):
            raise RuntimeError(
                "Provider returned an empty stream with no finish_reason "
                "(possible upstream error or malformed SSE response)."
            )

        # A stream that delivered a tool call but only partial/unparseable
        # JSON args splits into two very different cases:
        #
        #   1. Provider sent finish_reason="length" → a genuine output-cap
        #      truncation.  Boosting max_tokens on retry is the right move.
        #
        #   2. Provider sent NO finish_reason (the SSE simply stopped after
        #      the opening "{" with no terminator and no [DONE]) → the
        #      upstream dropped/stalled the connection mid tool-call.  This
        #      is NOT an output cap — the model never reported hitting one.
        #      Some dedicated endpoints (e.g. NVIDIA Nemotron Ultra on the
        #      Nous dedicated endpoint) stall for minutes during large
        #      tool-arg generation, then close the stream cleanly without a
        #      finish_reason.  Stamping "length" here sends it down the
        #      max_tokens-boost truncation path, which retries 3× to no
        #      effect and finally reports the misleading "Response truncated
        #      due to output length limit" — the red herring this guards
        #      against.  Route it through the partial-stream-stub path
        #      instead so the loop reports an honest mid-tool-call stream
        #      drop and fails fast rather than escalating output budget.
        _tool_args_dropped_no_finish = has_truncated_tool_args and finish_reason is None
        if _tool_args_dropped_no_finish:
            _dropped_names = [
                (tool_calls_acc[idx]["function"]["name"] or "?")
                for idx in sorted(tool_calls_acc)
            ]
            logger.warning(
                "Stream ended with no finish_reason while a tool call's "
                "arguments were still incomplete (tools=%s); treating as a "
                "mid-tool-call stream drop, not an output-length truncation.",
                _dropped_names,
            )
            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=None,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=FINISH_REASON_LENGTH,
            )
            return SimpleNamespace(
                id=PARTIAL_STREAM_STUB_ID,
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
                _dropped_tool_names=_dropped_names or None,
            )

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
        # Defensive: strip Responses-only kwargs (instructions, input, ...)
        # that can leak in under an api_mode-flip race. The Anthropic SDK
        # raises a non-retryable TypeError on them, killing the turn. See
        # #31673 / sanitize_anthropic_kwargs().
        from agent.anthropic_adapter import sanitize_anthropic_kwargs
        sanitize_anthropic_kwargs(
            api_kwargs, log_prefix=getattr(agent, "log_prefix", "")
        )
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

            # Return the native Anthropic Message for downstream processing.
            # If the stream was interrupted (the event loop broke out above on
            # agent._interrupt_requested), do NOT call get_final_message() — on
            # a partially-consumed stream the SDK may hang draining remaining
            # events or return a Message with incomplete tool_use blocks (partial
            # JSON in `input`). The outer poll loop raises InterruptedError, so
            # this return value is discarded anyway.
            if agent._interrupt_requested:
                return None
            return stream.get_final_message()

    def _call():
        import httpx as _httpx

        _max_stream_retries = env_int("HERMES_STREAM_RETRIES", 2)

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
                    # If the main poll loop force-closed this request because
                    # of an interrupt, the resulting transport error is the
                    # expected consequence of our own close — NOT a transient
                    # network error. Exit immediately: no retry, no fallback,
                    # no "reconnecting" status. The outer poll loop raises
                    # InterruptedError. This is the fix for the cascading-
                    # interrupt hang where doomed retries burned full
                    # stream-stale-timeout cycles. (#6600)
                    if _request_cancelled["value"]:
                        logger.debug(
                            "Streaming worker caught %s after request "
                            "cancellation — exiting without retry.",
                            type(e).__name__,
                        )
                        return
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
                        _close_request_client_once("stream_mid_tool_retry_cleanup")
                        if agent.api_mode == "anthropic_messages":
                            try:
                                agent._anthropic_client.close()
                                agent._rebuild_anthropic_client()
                            except Exception:
                                pass
                        else:
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
                            _close_request_client_once("stream_retry_cleanup")
                            # Also rebuild the primary client to purge
                            # any dead connections from the pool.
                            if agent.api_mode == "anthropic_messages":
                                try:
                                    agent._anthropic_client.close()
                                    agent._rebuild_anthropic_client()
                                except Exception:
                                    pass
                            else:
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
                        agent._buffer_status(
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
                        # AWS Bedrock (AnthropicBedrock SDK path): IAM policies
                        # with bedrock:InvokeModel but not
                        # InvokeModelWithResponseStream reject messages.stream()
                        # with a permission error naming the streaming action.
                        # Permanent for the session — flip to non-streaming
                        # (messages.create() maps to bedrock:InvokeModel).
                        _is_bedrock_stream_denied = False
                        if (
                            not _is_stream_unsupported
                            and "invokemodelwithresponsestream" in _err_lower
                        ):
                            # Cheap message pre-check before importing the
                            # adapter — bedrock_adapter triggers a lazy boto3
                            # install at import time, which must not run for
                            # unrelated providers' stream errors.
                            from agent.bedrock_adapter import (
                                is_streaming_access_denied_error,
                            )
                            _is_bedrock_stream_denied = (
                                is_streaming_access_denied_error(e)
                            )
                        if _is_stream_unsupported or _is_bedrock_stream_denied:
                            agent._disable_streaming = True
                            agent._safe_print(
                                "\n⚠  AWS IAM denied bedrock:InvokeModelWithResponseStream. "
                                "Switching to non-streaming.\n"
                                "   Grant that action to restore streaming output.\n"
                                if _is_bedrock_stream_denied else
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
            _close_request_client_once("stream_request_complete")

    # Provider-configured stale timeout takes priority over env default.
    _cfg_stale = get_provider_stale_timeout(agent.provider, agent.model)
    if _cfg_stale is not None:
        _stream_stale_timeout_base = _cfg_stale
    else:
        _stream_stale_timeout_base = env_float("HERMES_STREAM_STALE_TIMEOUT", 180.0)
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
        _est_tokens = estimate_request_context_tokens(api_kwargs)
        if _est_tokens > 100_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
        elif _est_tokens > 50_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
        else:
            _stream_stale_timeout = _stream_stale_timeout_base
        # Reasoning-model floor: known reasoning models (Nemotron 3 Ultra,
        # OpenAI o1/o3, Anthropic Opus 4.x thinking, DeepSeek R1, Qwen QwQ,
        # xAI Grok reasoning, etc.) routinely exceed the default 180s chat-
        # model threshold during their thinking phase.  The cloud gateway
        # upstream kills the socket first, surfacing as BrokenPipeError.
        # Raises the floor only — never overrides explicit user config
        # (handled by get_provider_stale_timeout above).
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        _reasoning_floor = get_reasoning_stale_timeout_floor(api_kwargs.get("model"))
        if _reasoning_floor is not None:
            _stream_stale_timeout = max(_stream_stale_timeout, _reasoning_floor)

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
            _est_ctx = estimate_request_context_tokens(api_kwargs)
            logger.warning(
                "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                "model=%s context=~%s tokens. Killing connection.",
                _stale_elapsed, _stream_stale_timeout,
                api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
            )
            agent._buffer_status(
                f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                f"(model: {api_kwargs.get('model', 'unknown')}, "
                f"context: ~{_est_ctx:,} tokens). "
                f"Reconnecting..."
            )
            try:
                _close_request_client_once("stale_stream_kill")
            except Exception:
                pass
            # Circuit breaker (#58962): count the stale kill.  See the
            # canonical comment block above ``_stale_streak()``.
            _bump_stale_streak(agent)
            # Rebuild the primary client too — its connection pool
            # may hold dead sockets from the same provider outage.
            if agent.api_mode == "anthropic_messages":
                try:
                    agent._anthropic_client.close()
                    agent._rebuild_anthropic_client()
                except Exception:
                    pass
            else:
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
            # Mark THIS request cancelled before force-closing so the worker's
            # exception handler recognizes the forced transport error as a
            # cancel and exits without retrying or surfacing a network error.
            # (#6600)
            _request_cancelled["value"] = True
            logger.debug(
                "Force-closing streaming httpx client due to interrupt "
                "(not a network error)."
            )
            try:
                if agent.api_mode == "anthropic_messages":
                    agent._anthropic_client.close()
                    agent._rebuild_anthropic_client()
                else:
                    _close_request_client_once("stream_interrupt_abort")
            except Exception:
                pass
            raise InterruptedError("Agent interrupted during streaming API call")
    # Worker thread exited before the main thread's poll loop could check
    # the interrupt flag.  If the worker returned early due to an interrupt
    # (e.g. _call_anthropic() detected _interrupt_requested and returned
    # None), the InterruptedError above was never raised.  Re-check the
    # flag here so /stop is not silently swallowed.  (#59999 area)
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted during streaming API call (post-worker)")
    if result["error"] is not None:
        if deltas_were_sent["yes"]:
            # Streaming failed AFTER some tokens were already delivered to
            # the platform.  Re-raising would let the outer retry loop make
            # Return a partial response stub with finish_reason="length"
            # so the conversation loop's continuation machinery fires.
            # tool_calls=None prevents auto-execution of incomplete calls.
            _partial_text = (
                getattr(agent, "_current_streamed_assistant_text", "") or ""
            ).strip() or None

            # Append a user-visible warning if tool calls were dropped so
            # the user and model both know what was attempted.
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
                # Fire as streaming delta so the user sees it immediately.
                try:
                    agent._fire_stream_delta(_warn)
                except Exception:
                    pass
                logger.warning(
                    "Partial stream dropped tool call(s) %s after %s chars "
                    "of text; surfaced warning to user: %s",
                    _partial_names, len(_partial_text or ""), result["error"],
                )
                _stub_finish_reason = FINISH_REASON_LENGTH
            else:
                logger.warning(
                    "Partial stream delivered before error; returning "
                    "length-truncated stub with %s chars of recovered "
                    "content so the loop can continue from where the "
                    "stream died: %s",
                    len(_partial_text or ""),
                    result["error"],
                )
                _stub_finish_reason = FINISH_REASON_LENGTH
            _stub_msg = SimpleNamespace(
                role="assistant", content=_partial_text, tool_calls=None,
                reasoning_content=None,
            )
            # Detect provider output-layer content filtering (e.g. MiniMax
            # "output new_sensitive (1027)", Azure/OpenAI content_filter,
            # Anthropic safety refusal).  The raw error is about to be
            # swallowed into a finish_reason=length stub, so classify it HERE
            # while we still have it and stamp the stub.  Retrying such a
            # content-deterministic filter on the same primary just re-hits
            # the filter — the conversation loop reads this tag and activates
            # the fallback chain instead of burning continuation retries.
            # error_classifier is the single source of truth for "what counts
            # as a content filter" (#32421).
            _content_filter_terminated = False
            try:
                from agent.error_classifier import classify_api_error, FailoverReason
                _cls = classify_api_error(
                    result["error"],
                    provider=str(getattr(agent, "provider", "") or ""),
                    model=str(getattr(agent, "model", "") or ""),
                )
                _content_filter_terminated = (
                    _cls.reason == FailoverReason.content_policy_blocked
                )
            except Exception:
                _content_filter_terminated = False
            _stub = SimpleNamespace(
                id=PARTIAL_STREAM_STUB_ID,
                model=getattr(agent, "model", "unknown"),
                choices=[SimpleNamespace(
                    index=0, message=_stub_msg, finish_reason=_stub_finish_reason,
                )],
                usage=None,
                _dropped_tool_names=_partial_names or None,
            )
            if _content_filter_terminated:
                _stub._content_filter_terminated = True
            # Partial-stream stub: chunks WERE received (deltas fired), so
            # the provider is demonstrably responsive — clear the circuit
            # breaker (#58962) just like the full-success return below.
            _reset_stale_streak(agent)
            return _stub
        raise result["error"]
    # Success — clear the circuit breaker (#58962): the provider proved
    # responsive.  See the canonical comment block above ``_stale_streak()``.
    if result["response"] is not None:
        _reset_stale_streak(agent)
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
