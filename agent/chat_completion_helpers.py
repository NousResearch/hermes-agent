"""API-call helpers extracted from :class:`AIAgent`: non-streaming and streaming
request drivers, request kwargs builder, assistant-message materializer,
provider-fallback activator, max-iterations handler, per-turn resource cleanup.

Each function takes the parent ``AIAgent`` as ``agent``; AIAgent keeps thin
forwarders. Symbols tests patch on ``run_agent`` (``cleanup_vm`` /
``cleanup_browser``) are resolved through :func:`_ra` at call time.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout
from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH
from agent.error_classifier import (FailoverReason, PROVIDER_STREAM_NON_JSON_ERROR_CODE)
from agent.errors import EmptyStreamError
from agent.fast_mode import effective_request_overrides
from agent.turn_context import substitute_api_content
from agent.gemini_native_adapter import is_native_gemini_base_url
# Remote endpoints must never be fingerprinted: the probe waterfall is only valid for local/LM-Studio/Ollama
# boxes. Non-Ollama remotes (sglang, vLLM, OpenAI-compat) expose Ollama-compat endpoints that can
# misidentify and, without an api_key, return 401 on every leg (issue #89863).
from agent.model_metadata import is_local_endpoint
from agent.message_content import flatten_message_text
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.message_sanitization import (_sanitize_surrogates, _repair_tool_call_arguments)
from agent.reasoning_summaries import separate_glued_reasoning_blocks
from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current
from tools.terminal_tool_lifecycle import is_persistent_env
from utils import base_url_host_matches, base_url_hostname, env_float, env_int

logger = logging.getLogger(__name__)
_OPENROUTER_PROVIDER_SORT_VALUES = {"throughput", "latency", "price"}
_PROVIDER_STREAM_ERROR_FINISH_REASONS = {"error", "error_finish"}
_PROVIDER_STREAM_SSE_FIELDS = {"event", "data", "id", "retry"}
_PROVIDER_STREAM_ERROR_TEXT_LIMIT = 4096

# Fallback chain exhausted on a non-rate-limit failure (#24996): arm a short
# cooldown so the NEXT turn's restore_primary_runtime stays gated instead of
# resetting _fallback_index=0 and re-marshaling the whole context across every
# provider again (memory/swap exhaustion on constrained hosts). Rate-limit /
# billing reasons keep their own longer cooldown.
_FALLBACK_EXHAUSTED_COOLDOWN_S = 5.0


def _context_thread_target(callback):
    """Bind a no-argument thread target to the caller's ContextVars."""
    context = contextvars.copy_context()
    return lambda: context.run(callback)


def _join_worker_for_relay_teardown(worker, *, label: str) -> None:
    """Bounded worker join before raising InterruptedError (#81521).

    Raising immediately lets turn teardown race a still-open Relay LLM scope and
    corrupt the LIFO stack (CLI EIO / redraw storm). Only joins when Relay managed
    execution is live — otherwise the join would just delay interrupt detection.
    """
    try:
        from agent import relay_runtime
        runtime = relay_runtime.get_runtime(create=False)
        if runtime is None or not runtime.managed_execution_enabled():
            return
    except Exception:
        return
    worker.join(timeout=2.0)
    if worker.is_alive():
        logger.warning("%s worker still alive after interrupt abort (2.0s join "
            "timeout); Relay teardown will best-effort drain orphaned scopes (#81521).", label)


def _ra():
    """Lazy ``run_agent`` reference so ``patch("run_agent.cleanup_vm")`` etc. intercept."""
    import run_agent
    return run_agent


class ProviderStreamError(Exception):
    """Provider encoded an API error as streaming content instead of an SDK error."""

    def __init__(self, *, status_code: Optional[int], body: dict, raw_text: str, headers: Any = None):
        self.status_code = status_code
        self.body = body
        self.raw_text = raw_text
        self.response = SimpleNamespace(headers=headers or {})
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        error_obj = self.body.get("error", {}) if isinstance(self.body, dict) else {}
        if not isinstance(error_obj, dict):
            error_obj = {}
        parts = ["Provider stream returned an error event"]
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        if error_obj.get("code"):
            parts.append(str(error_obj["code"]))
        text = " - ".join(parts)
        if error_obj.get("message"):
            text += f": {error_obj['message']}"
        return text


def _status_code_from_value(value: Any) -> Optional[int]:
    if isinstance(value, int) and 100 <= value < 600:
        return value
    if not isinstance(value, str):
        return None
    match = re.search(r"(?:HTTP_STATUS/)?\b([1-5]\d\d)\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _status_code_from_payload(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    candidates = [payload.get(k) for k in ("status_code", "status", "http_status")]
    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        candidates.extend(error_obj.get(k) for k in ("status_code", "status", "http_status", "code"))
    candidates.append(payload.get("code"))
    for candidate in candidates:
        status_code = _status_code_from_value(candidate)
        if status_code is not None:
            return status_code
    return None


def _json_object_from_text(text: str) -> Optional[dict]:
    stripped = (text or "").strip()
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        if stripped.startswith("{"):
            decoded = json.loads(stripped)
            return decoded if isinstance(decoded, dict) else None
    return None


def _parse_provider_sse_events(text: str) -> list[dict]:
    """Parse provider text that looks like Server-Sent Events."""
    events: list[dict] = []
    current = {"event": None, "data": [], "comments": [], "fields": {}}

    def _flush_current():
        nonlocal current
        if any(current.values()):
            status_candidates = list(current["comments"]) + [
                current["fields"][key]
                for key in ("status", "status_code", "http_status")
                if key in current["fields"]
            ]
            events.append({
                "event": current["event"],
                "data": "\n".join(current["data"]),
                "comments": list(current["comments"]),
                "fields": dict(current["fields"]),
                "status_code": next(
                    (s for s in map(_status_code_from_value, status_candidates) if s is not None), None),
            })
        current = {"event": None, "data": [], "comments": [], "fields": {}}

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip("\r")
        if line == "":
            _flush_current()
            continue
        if line.startswith(":"):
            current["comments"].append(line[1:].strip())
            continue

        field, sep, value = line.partition(":")
        if not sep:
            current["fields"][field.strip().lower()] = ""
            continue
        field = field.strip().lower()
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            current["event"] = value.strip()
        elif field == "data":
            current["data"].append(value)
        else:
            current["fields"][field] = value

    _flush_current()
    return events


def _provider_error_body(payload: dict, status_code: Optional[int]) -> dict:
    """Normalize common provider error payloads to OpenAI-style body.error."""
    if not isinstance(payload, dict):
        payload = {}
    elif isinstance(payload.get("error"), dict):
        return payload
    code = (payload.get("code") or payload.get("error_code") or payload.get("type")
            or (f"HTTP_{status_code}" if status_code else "provider_stream_error"))
    message = (payload.get("message") or payload.get("error_description") or payload.get("error")
               or "Provider stream returned an error event.")
    normalized_error = {"message": str(message)}
    if code:
        normalized_error["code"] = str(code)
    for key in ("request_id", "param", "type"):
        if payload.get(key):
            normalized_error[key] = payload[key]
    return {"error": normalized_error}


def _provider_stream_error_from_json_decode_error(error: json.JSONDecodeError, *,
    response: Any = None) -> ProviderStreamError:
    """Preserve plain-text SSE data rejected inside the OpenAI SDK: on a non-JSON
    ``event: error`` the SDK raises from ``sse.json()`` before yielding a chunk,
    but ``JSONDecodeError.doc`` still carries the provider's original message."""
    from agent.redact import redact_sensitive_text
    raw_text = str(getattr(error, "doc", "") or "").strip()
    safe_text = redact_sensitive_text(_sanitize_surrogates(raw_text), force=True)
    safe_text = safe_text[:_PROVIDER_STREAM_ERROR_TEXT_LIMIT]
    return ProviderStreamError(
        status_code=None,
        body=_provider_error_body(
            {"code": PROVIDER_STREAM_NON_JSON_ERROR_CODE,
                "message": safe_text or "Provider stream returned non-JSON SSE data."},
            None,
        ),
        raw_text=safe_text,
        headers=getattr(response, "headers", None) if response is not None else None,
    )


def _iter_provider_stream_chunks(stream, *, response: Any = None):
    """Yield SDK chunks while translating SDK-level SSE decode failures."""
    try:
        yield from stream
    except json.JSONDecodeError as error:
        stream_response = response() if callable(response) else response
        if stream_response is None:
            stream_response = getattr(stream, "response", None)
        raise _provider_stream_error_from_json_decode_error(error, response=stream_response) from error


def _payload_has_error_shape(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("error"), (dict, str)):
        return True
    return bool(payload.get("message")) and bool(
        payload.get("code") or payload.get("error_code") or _status_code_from_payload(payload) is not None)


def _provider_stream_text_may_be_sse(text: str) -> bool:
    """Return True while pending text still looks like an SSE control block."""
    stripped = (text or "").lstrip()
    if not stripped:
        return False

    lines = stripped.splitlines()
    trailing_newline = stripped.endswith(("\n", "\r"))
    saw_sse_field = False

    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r")
        if line == "":
            continue
        if line.startswith(":"):
            saw_sse_field = True
            continue

        field, sep, _value = line.partition(":")
        field_name = field.strip().lower()
        if sep and field_name in _PROVIDER_STREAM_SSE_FIELDS:
            saw_sse_field = True
            continue

        is_last_incomplete = index == len(lines) - 1 and not trailing_newline
        if is_last_incomplete and any(
            sse_field.startswith(field_name) for sse_field in _PROVIDER_STREAM_SSE_FIELDS):
            return True
        return False

    return saw_sse_field


def _provider_stream_error_from_text(text: str, finish_reason: Optional[str], *,
    response: Any = None) -> Optional[ProviderStreamError]:
    """Convert provider-streamed error text into an exception for retry logic."""
    if not text:
        return None

    if str(finish_reason or "").lower() not in _PROVIDER_STREAM_ERROR_FINISH_REASONS:
        return None

    headers = getattr(response, "headers", None) if response is not None else None

    def _error(payload: dict, status_code: Optional[int]) -> ProviderStreamError:
        return ProviderStreamError(status_code=status_code, body=_provider_error_body(payload, status_code),
            raw_text=text, headers=headers)

    for event in _parse_provider_sse_events(text):
        is_error_event = str(event.get("event") or "").strip().lower() == "error"
        payload = _json_object_from_text(event.get("data") or "") or {}
        status_code = event.get("status_code") or _status_code_from_payload(payload)
        # The finish_reason is an error here, so an error event always qualifies;
        # a non-error event needs an error-shaped payload or an HTTP error code.
        if (status_code is not None and status_code >= 400) or is_error_event or _payload_has_error_shape(payload):
            return _error(payload, status_code)

    payload = _json_object_from_text(text)
    if payload is not None:
        return _error(payload, _status_code_from_payload(payload))

    if text.strip():
        return _error({}, None)
    return None


def estimate_request_context_tokens(api_payload: Any) -> int:
    """Cheap char/4 context estimate for the stale-call detectors. Handles both
    wire shapes so Codex turns don't report ~0 tokens: list -> Chat ``messages``;
    dict with ``messages`` (+``tools``); dict with ``input`` (Responses API,
    +``instructions``/``tools``); any other dict -> sum of its values."""

    def _chars(value: Any) -> int:
        return 0 if value is None else len(str(value))

    if isinstance(api_payload, list):
        return sum(_chars(item) for item in api_payload) // 4
    if not isinstance(api_payload, dict):
        return _chars(api_payload) // 4
    messages = api_payload.get("messages")
    if isinstance(messages, list):
        total_chars = sum(_chars(item) for item in messages)
        if "tools" in api_payload:
            total_chars += _chars(api_payload.get("tools"))
        return total_chars // 4
    if "input" in api_payload:
        return sum(_chars(api_payload.get(k)) for k in ("input", "instructions", "tools")) // 4
    return sum(_chars(value) for value in api_payload.values()) // 4


def _is_openai_codex_backend(agent) -> bool:
    from agent.codex_responses_adapter import classify_responses_route
    return classify_responses_route(agent).is_codex_backend


def openai_codex_stale_timeout_floor(est_tokens: int) -> float:
    """Minimum wall-clock stale timeout for openai-codex by estimated context:
    subscription-backed Codex can spend minutes in admission/prefill on
    gateway-scale payloads, so the generic default would abort healthy calls.
    The floor engages above 10k estimated tokens."""
    for threshold, floor in ((100_000, 1200.0), (50_000, 900.0), (10_000, 600.0)):
        if est_tokens > threshold:
            return floor
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
    logger.warning("Ignoring invalid OpenRouter provider.sort value %r (allowed: %s)", raw_sort,
        ", ".join(sorted(_OPENROUTER_PROVIDER_SORT_VALUES)))
    return None


def _provider_preferences_for_agent(agent) -> Dict[str, Any]:
    """Build the validated provider-routing object shared by request paths."""
    preferences: Dict[str, Any] = {}
    for key, value in (("only", agent.providers_allowed), ("ignore", agent.providers_ignored),
        ("order", agent.providers_order), ("sort", _validated_openrouter_provider_sort(agent.provider_sort)),
        ("require_parameters", True if agent.provider_require_parameters else None),
        ("data_collection", agent.provider_data_collection)):
        if value:
            preferences[key] = value
    return preferences


def _prompt_cache_scope_for_agent(agent) -> "str | None":
    """Rotation-stable logical cache scope for *agent*, or None (transports then
    fall back to the physical session_id, so a failure never blocks the build)."""
    try:
        from agent.prompt_cache_scope import resolve_prompt_cache_scope_safe
        return resolve_prompt_cache_scope_safe(agent)
    except Exception:
        logger.debug("prompt-cache scope resolution failed", exc_info=True)
        return None


def _merge_nous_portal_messages_extra_body(agent, anthropic_kwargs: dict) -> dict:
    """Merge Portal ``tags`` / ``session_id`` onto an Anthropic Messages kwargs dict.
    The Nous profile is only consulted by the OpenAI-wire transport; ``session_id``
    only — never ``provider_preferences`` (an OpenAI-wire routing object)."""
    if getattr(agent, "provider", None) not in {"nous", "nous-portal", "nousresearch"}:
        return anthropic_kwargs
    try:
        from providers import get_provider_profile
        nous_profile = get_provider_profile("nous")
        if nous_profile is not None:
            anthropic_kwargs.setdefault("extra_body", {}).update(
                nous_profile.build_extra_body(session_id=getattr(agent, "session_id", None)))
    except Exception as exc:  # noqa: BLE001 — never block a turn on tagging
        logger.debug("Nous Portal extra_body merge failed: %s", exc)
    return anthropic_kwargs


def _estimate_chunk_bytes(chunk: Any) -> int:
    """Cheap per-chunk size estimate for the stream diagnostic counters: delta
    string lengths plus a framing floor (~3x cheaper than ``len(repr(chunk))``
    in the agent's hottest loop). Unknown shapes just keep the floor."""
    size = 40  # SSE/JSON framing floor per chunk

    def _add(obj, *attrs):
        nonlocal size
        for attr in attrs:
            v = getattr(obj, attr, None)
            if isinstance(v, str):
                size += len(v)

    with contextlib.suppress(Exception):
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                _add(delta, "content", "reasoning_content", "reasoning")
                for tc in getattr(delta, "tool_calls", None) or ():
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        _add(fn, "arguments", "name")
        else:
            _add(getattr(chunk, "delta", None), "text", "partial_json")
    return size


def _codex_wait_notice_recovery(*, stale_timeout: float, ttfb_enabled: bool, ttfb_timeout: float,
    last_event_ts: Optional[float], call_start: float, idle_enabled: bool, idle_timeout: float,
    elapsed: float) -> str:
    """Describe the earliest enabled Codex watchdog on the call timeline."""
    deadlines: list[float] = []
    if math.isfinite(stale_timeout):
        deadlines.append(stale_timeout)
    if last_event_ts is None:
        if ttfb_enabled and math.isfinite(ttfb_timeout):
            deadlines.append(ttfb_timeout)
    elif idle_enabled and math.isfinite(idle_timeout):
        deadlines.append(max(0.0, last_event_ts - call_start) + idle_timeout)
    if not deadlines or min(deadlines) <= elapsed:
        return ""
    return f"; auto-reconnect at {int(min(deadlines))}s"


# ── Cross-turn stale-call circuit breaker (#58962) ─────────────────────
# A session wedged against an unresponsive provider would otherwise hit the
# stale detector on every call forever. ``agent._consecutive_stale_streams``
# is bumped on every stale kill and reset only when a call completes or the
# provider is swapped (switch_model / try_activate_fallback /
# restore_primary_runtime — the streak measured the OLD provider). Past the
# give-up threshold, calls abort immediately with an actionable error.

def _stale_streak(agent) -> int:
    try:
        return int(getattr(agent, "_consecutive_stale_streams", 0) or 0)
    except Exception:
        return 0


def _bump_stale_streak(agent) -> None:
    with contextlib.suppress(Exception):
        agent._consecutive_stale_streams = _stale_streak(agent) + 1


def _reset_stale_streak(agent) -> None:
    with contextlib.suppress(Exception):
        agent._consecutive_stale_streams = 0


_INTERRUPTED_WAIT_STALE_SECONDS = 30.0


def _record_interrupted_provider_wait(agent, elapsed: float, *, response_started: bool) -> bool:
    """Count a user-aborted pre-response stall toward the stale breaker: past the
    wait-notice interval an interrupt is evidence of an unresponsive attempt.
    Mid-response and early interrupts stay neutral."""
    if response_started or elapsed < _INTERRUPTED_WAIT_STALE_SECONDS:
        return False
    _bump_stale_streak(agent)
    logger.warning("Interrupted provider wait counted as stale after %.0fs with no output; "
        "consecutive stale attempts=%d.", elapsed, _stale_streak(agent))
    return True


def _report_stale_nonstream_kill(agent, api_kwargs: dict, elapsed: float, stale_timeout: float, *,
    inline: bool = False, hint: Optional[str] = None) -> None:
    """Log + status message for a stale non-streaming kill, shared by the worker
    poll loop and the inline ``direct_api_call`` watchdog (their kill/state
    sequences differ deliberately: different locking models)."""
    model = api_kwargs.get("model", "unknown")
    logger.warning("%son-streaming API call stale for %.0fs (threshold %.0fs). "
        "model=%s context=~%s tokens. Killing connection.", "Inline n" if inline else "N", elapsed,
        stale_timeout, model, f"{estimate_request_context_tokens(api_kwargs):,}")
    try:
        agent._buffer_status(
            f"⚠️ No response from provider for {int(elapsed)}s (non-streaming, model: {model}). {hint or 'Aborting call.'}")
    except Exception:
        logger.debug("stale status buffering failed", exc_info=True)


def _touch_stale_kill_activity(agent, elapsed: float) -> None:
    try:
        agent._touch_activity(f"stale non-streaming call killed after {int(elapsed)}s")
    except Exception:
        logger.debug("stale activity touch failed", exc_info=True)


def _check_stale_giveup(agent) -> None:
    """Raise immediately when the consecutive-stale streak is past the
    give-up threshold — no network attempt, no stale-timeout wait."""
    _giveup = env_int("HERMES_STREAM_STALE_GIVEUP", 5)
    _streak = _stale_streak(agent)
    if _giveup > 0 and _streak >= _giveup:
        raise RuntimeError(
            "Provider has been unresponsive (no response received) for "
            f"{_streak} consecutive stale attempts — aborting this call to "
            "avoid an indefinite stall. Switch models or start a new session, then retry."
        )


def _configured_stale_base(agent) -> float:
    """Per-provider ``stale_timeout_seconds`` config, else HERMES_STREAM_STALE_TIMEOUT (180s)."""
    cfg = get_provider_stale_timeout(agent.provider, agent.model)
    return cfg if cfg is not None else env_float("HERMES_STREAM_STALE_TIMEOUT", 180.0)


def _scale_stale_timeout_for_context(base: float, est_tokens: int) -> float:
    """Large contexts: slow models think for minutes before the first token;
    scale the threshold or the detector kills healthy streams."""
    if est_tokens > 100_000:
        return max(base, 300.0)
    if est_tokens > 50_000:
        return max(base, 240.0)
    return base


def _cloud_stale_timeout(base: float, api_kwargs: dict) -> float:
    """Cloud stale-stream patience: ``base`` scaled for context size, then floored for
    known reasoning models. ``model`` (OpenAI/Anthropic) wins over ``modelId`` (Bedrock);
    Bedrock's dotted, region-prefixed profile id can't match the floor's slug regex
    directly, so it is normalized as a fallback."""
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    timeout = _scale_stale_timeout_for_context(base, estimate_request_context_tokens(api_kwargs))
    floor = get_reasoning_stale_timeout_floor(api_kwargs.get("model") or api_kwargs.get("modelId") or "")
    if floor is None and api_kwargs.get("modelId"):
        floor = _bedrock_reasoning_stale_floor(api_kwargs["modelId"])
    return timeout if floor is None else max(timeout, floor)


def _derive_stream_stale_timeout(agent, api_kwargs: dict) -> float:
    """Stale-stream patience for a provider that is never a local endpoint (Bedrock):
    the OpenAI/Anthropic stale detector's budget minus its local branch."""
    return _cloud_stale_timeout(_configured_stale_base(agent), api_kwargs)


def _bedrock_reasoning_stale_floor(model_id: object) -> "float | None":
    """Map a Bedrock inference-profile id to its reasoning stale-timeout floor.

    ``us.anthropic.claude-opus-4-6-v1:0`` -> strip the region prefix, then try the
    segment after the provider namespace (``claude-opus-4-6-v1:0``) and the id with
    the provider dot dashed (``deepseek-r1-v1:0``). The floor table mixes dashed
    and dotted versions while Bedrock always dashes, so each candidate is also
    tried with digit-dash-digit <-> digit-dot-digit swapped (version separators
    only). First non-None wins; None for unknown models.
    """
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    if not model_id or not isinstance(model_id, str):
        return None
    name = model_id.strip().lower()
    for prefix in ("global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.", "ca.", "sa.", "me.", "af."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    base_candidates = [name]
    if "." in name:
        base_candidates.append(name.rsplit(".", 1)[1])   # claude-opus-4-6-v1:0
        base_candidates.append(name.replace(".", "-", 1))  # deepseek-r1-v1:0
    candidates = dict.fromkeys(
        form for cand in base_candidates
        for form in (cand, re.sub(r"(?<=\d)-(?=\d)", ".", cand), re.sub(r"(?<=\d)\.(?=\d)", "-", cand)))
    return next((f for f in map(get_reasoning_stale_timeout_floor, candidates) if f is not None), None)


def _bedrock_converse_call(api_kwargs: dict, *, stream: bool, on_stream_denied=None):
    """Pop the Hermes routing keys and call ``converse`` / ``converse_stream`` (boto3
    directly) with the shared recovery: a cachePoint rejection (Nova: toolConfig.tools,
    #97281) drops the marker and resends once inside the same attempt; a streaming IAM
    denial hands off to ``on_stream_denied(client, kwargs, exc)``; a stale connection
    evicts the cached client so the outer retry builds a fresh pool. Streaming returns the
    event stream; non-streaming an OpenAI-shaped SimpleNamespace."""
    from agent.bedrock_adapter import (_get_bedrock_runtime_client, invalidate_runtime_client,
        is_stale_connection_error, is_streaming_access_denied_error, normalize_converse_response,
        recover_from_cache_point_rejection)
    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
    api_kwargs.pop("__bedrock_converse__", None)
    client = _get_bedrock_runtime_client(region)
    method = client.converse_stream if stream else client.converse
    finish = (lambda raw: raw.get("stream", [])) if stream else normalize_converse_response
    try:
        raw_response = method(**api_kwargs)
    except Exception as exc:
        retry_kwargs = recover_from_cache_point_rejection(exc, api_kwargs)
        if retry_kwargs is not None:
            return finish(method(**retry_kwargs))
        if on_stream_denied is not None and is_streaming_access_denied_error(exc):
            return on_stream_denied(client, api_kwargs, exc)
        if is_stale_connection_error(exc):
            invalidate_runtime_client(region)
        raise
    return finish(raw_response)


def _dispatch_nonstreaming_api_request(agent, api_kwargs: dict, *, make_client):
    """Run one non-streaming LLM request for the active api_mode and return it.

    Shared by ``interruptible_api_call`` and ``direct_api_call``. ``make_client(reason,
    kind=...)`` builds the per-request client (``"openai"`` / ``"anthropic_messages"``)
    so callers can register it with their abort/close machinery; bedrock / MoA
    manage their own clients. Interrupt/abort/close semantics stay in callers.
    """
    if agent.api_mode == "codex_responses":
        return agent._run_codex_stream(api_kwargs, client=make_client("codex_stream_request"),
            on_first_delta=getattr(agent, "_codex_on_first_delta", None))
    if agent.api_mode == "anthropic_messages":
        # Request-local client so the stale/interrupt watchdog aborts sockets
        # from the stranger thread while the worker owns the SDK close (#67142).
        request_client = make_client("anthropic_messages_request", kind="anthropic_messages")
        return agent._anthropic_messages_create(api_kwargs, client=request_client)
    if agent.api_mode == "bedrock_converse":
        return _bedrock_converse_call(api_kwargs, stream=False)
    if agent.provider == "moa":
        # MoA is a virtual provider backed by the in-process MoAClient facade — never
        # rebuild a request-local client from the virtual metadata. After a client
        # replacement agent.client may be a native OpenAI client while provider stays
        # "moa": pop the MoA-internal key ONLY then (the facade consumes it; stripping
        # it there forces a duplicate fan-out). Only the facade exposes ``prepare()`` (#78382).
        _completions = getattr(getattr(agent.client, "chat", None), "completions", None)
        if not callable(getattr(_completions, "prepare", None)):
            api_kwargs.pop("_moa_prepared_request", None)
        return agent.client.chat.completions.create(**api_kwargs)
    return make_client("chat_completion_request").chat.completions.create(**api_kwargs)


def should_use_direct_api_call(agent) -> bool:
    """Whether an OpenAI-wire request should skip the interrupt worker.

    Gateway cron turns (#62151) and delegated children (#60203) run inside nested
    thread pools that wedge before the socket opens when the request is pushed onto
    yet another daemon worker. Running inline drops the deepest layer; interrupts
    still work because the inline path registers ``agent._active_request_abort``,
    which ``interrupt()`` invokes cross-thread (#72227). Native/Codex/Bedrock/MoA
    keep their workers: their cancellation and client ownership differ.
    """
    if getattr(agent, "api_mode", None) != "chat_completions" or getattr(agent, "provider", None) == "moa":
        return False
    if getattr(agent, "platform", None) == "cron":
        return True
    # Delegated child — via the execution ContextVar set by _run_single_child,
    # with the agent's platform stamp as a fallback for callers that bypass it.
    with contextlib.suppress(Exception):
        from agent.delegation_context import is_delegated_child_context
        if is_delegated_child_context():
            return True
    return getattr(agent, "platform", None) == "subagent"


# How often an in-flight direct_api_call refreshes last_activity_ts. Must stay well
# under the async-delegation idle stall threshold (450s) and below the 30s monitor sweep.
_DIRECT_API_ACTIVITY_HEARTBEAT_SECONDS = 15.0


def _managed_local_load_notice(agent, api_kwargs: dict) -> "Optional[str]":
    """Live phase notice ("⏳ loading <model> into memory — N%" / "⚙ processing
    prompt — P%") while the managed local server works before the first token;
    None when neither applies. Otherwise a cold load reads as a generic stall."""
    try:
        base = str(getattr(agent, "base_url", "") or "")
        if not base:
            return None
        from urllib.parse import urlparse
        from hermes_cli.local_runtime.load_progress import get_loading_progress, get_prefill_progress
        from hermes_cli.local_runtime.supervisor import state_path
        state = json.loads(state_path().read_text(encoding="utf-8"))
        managed = urlparse(str(state.get("base_url", ""))).netloc.lower()
        if not managed or urlparse(base).netloc.lower() != managed:
            return None
        model = str(api_kwargs.get("model", ""))
        progress = get_loading_progress().get(model)
        if progress is not None:
            return (f"⏳ loading {model} into memory — {progress['percent']}% "
                "(responses start once the model is loaded)")
        prefill = get_prefill_progress(model)
        if prefill is None:
            return None
        processed = int(prefill["processed"])
        total = estimate_request_context_tokens(api_kwargs)
        if total and total >= processed:
            return f"⚙ processing prompt — {max(0, min(100, round(processed / total * 100)))}%"
        # Counter past the estimate (estimator undercounted): no honest denominator, label-only.
        return "⚙ processing prompt"
    except Exception:  # noqa: BLE001 — a status nicety must never break a call
        return None


def _resolve_direct_stale_timeout(agent, api_kwargs: dict) -> float:
    """Stale budget for the inline call via ``agent._compute_non_stream_stale_timeout``.
    A non-numeric result (stub agent) leaves the watchdog disarmed; a resolver
    that *raises* propagates — swallowing into ``inf`` would reinstate the hang."""
    resolver = getattr(agent, "_compute_non_stream_stale_timeout", None)
    value = resolver(api_kwargs) if callable(resolver) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("inf")
    return float(value)


def _inline_nonstream_hard_timeout(stale_timeout: float):
    """Socket-level backstop for inline non-streaming calls (#85252): the keepalive
    client uses ``read=None`` and the stranger-thread abort must not ``close()`` the
    FD (#29507), so a hung provider otherwise waits until TCP dies. Returns an
    ``httpx.Timeout`` with read == stale budget, a float if httpx is unavailable,
    or ``None`` when the watchdog is disarmed (non-finite budget)."""
    if not math.isfinite(stale_timeout) or stale_timeout <= 0:
        return None
    conn_cap = min(stale_timeout, 60.0)
    try:
        import httpx as _httpx
        return _httpx.Timeout(connect=conn_cap, read=stale_timeout, write=conn_cap, pool=conn_cap)
    except Exception:
        return stale_timeout


class _InlineRequest:
    """Lifecycle state for one inline non-streaming request (#75301). Every transition
    happens under ``lock``: ``done`` stops a late timer bumping the stale streak after
    unwind; ``cancelled`` lets an interrupt own the outcome so a racing timer can't
    misclassify the kill as staleness; ``stale`` is the one-shot transition."""

    def __init__(self, agent, api_kwargs: dict, stale_timeout: float, call_start: float):
        self.agent = agent
        self.api_kwargs = api_kwargs
        self.stale_timeout = stale_timeout
        self.call_start = call_start
        self.client = None
        self.done = False
        self.stale = False
        self.cancelled = False
        self.lock = threading.Lock()
        self.abort_hook = self.abort  # single bound object: identity-checked on cleanup
        self._hb_stop = threading.Event()
        self._hb = threading.Thread(target=self._activity_heartbeat, name="direct-api-activity-hb", daemon=True)
        self._watchdog = None

    def _activity_heartbeat(self) -> None:
        # Never put the API call itself on another worker thread — that is the nested-pool
        # deadlock this path exists to avoid (#60203). This ticker only refreshes the clock.
        while not self._hb_stop.wait(_DIRECT_API_ACTIVITY_HEARTBEAT_SECONDS):
            with contextlib.suppress(Exception):
                self.agent._touch_activity("waiting for non-streaming API response")

    def _on_stale(self) -> None:
        # Timer thread: aborts sockets only, never issues a request (keeps the no-worker
        # property). False = request finished or an interrupt owns the outcome; stay silent.
        if not self.abort("stale_call_kill"):
            return
        elapsed = time.time() - self.call_start
        _report_stale_nonstream_kill(self.agent, self.api_kwargs, elapsed, self.stale_timeout, inline=True)
        _touch_stale_kill_activity(self.agent, elapsed)

    def start_watchdogs(self) -> None:
        """Start the activity heartbeat and (for a finite budget) the stale timer."""
        self._hb.start()
        if math.isfinite(self.stale_timeout) and self.stale_timeout > 0:
            self._watchdog = threading.Timer(self.stale_timeout, self._on_stale)
            self._watchdog.name = "direct-api-stale-watchdog"
            self._watchdog.daemon = True
            self._watchdog.start()

    def stop_watchdogs(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
        self.mark_done()
        self._hb_stop.set()
        self._hb.join(timeout=2.0)

    def _abort_client(self, client, reason: str, log_msg: str) -> None:
        try:
            self.agent._abort_request_openai_client(client, reason=reason)
        except Exception:
            logger.debug(log_msg, exc_info=True)

    def abort(self, reason: str) -> bool:
        """Abort the inline request from a watchdog/interrupt thread. Returns True
        when this call owned the stale transition (the timer reports/bumps once,
        never after an interrupt or a completed request). Aborts under the lock
        (same contract as _RequestClientRegistry): once released the finally may
        cache the client and the NEXT call check it out."""
        with self.lock:
            if self.done:
                return False
            if reason == "stale_call_kill":
                if self.cancelled:
                    return False
                newly_stale = not self.stale
                if newly_stale:
                    self.stale = True
                    # Bump BEFORE releasing: a fast retry's reset must not be
                    # overtaken by this older timer restoring the streak.
                    _bump_stale_streak(self.agent)
            else:
                # Interrupt wins the lock -> owns the outcome; a later timer
                # must not count it as staleness.
                self.cancelled = True
                newly_stale = False
            if self.client is not None:
                self._abort_client(self.client, reason, f"Inline request abort failed ({reason})")
            return newly_stale

    def make_client(self, reason: str, kind: str = "openai"):
        # Only OpenAI-wire requests reach direct_api_call; ``kind`` exists
        # for signature parity with the dispatch helper.
        client = self.agent._create_request_openai_client(reason=reason, api_kwargs=self.api_kwargs)
        with self.lock:
            self.client = client
            stale_before_dispatch = self.stale
            if stale_before_dispatch:
                # Timer fired during client construction: the abort found no
                # socket, so dispatching now would open one AFTER the only
                # watchdog fired. Fail here instead. (Residual ms-scale window
                # before httpx opens its socket is accepted.)
                self._abort_client(client, "stale_call_kill", "Inline abort after late client registration failed")
        if stale_before_dispatch:
            raise TimeoutError(
                f"Non-streaming API call timed out before request dispatch (threshold: {int(self.stale_timeout)}s)")
        self.agent._active_request_abort = self.abort_hook
        return client

    def mark_done(self) -> None:
        with self.lock:
            self.done = True

    def pop_client(self):
        with self.lock:
            client, self.client = self.client, None
        return client


def direct_api_call(agent, api_kwargs: dict):
    """Run a non-streaming LLM call inline on the conversation thread (cron turns,
    delegated children — see ``should_use_direct_api_call``): no interrupt worker,
    so the nested-pool deadlock cannot occur. An activity heartbeat keeps
    ``last_activity_ts`` advancing (else the stall monitor interrupts a healthy
    wait at ~450s). A stale-call watchdog bounds the request (#80759): the timer
    aborts in-flight sockets via the registered hook, and a per-call ``timeout``
    equal to the stale budget is the backstop when the abort finds nothing (#85252).
    Both surface a retryable ``TimeoutError`` for the outer retry loop."""
    _check_stale_giveup(agent)
    agent._touch_activity("waiting for non-streaming API response")
    # Resolve the budget BEFORE the heartbeat starts: the resolver may raise
    # (fail-closed), and a leaked heartbeat thread would mask real stalls forever.
    call_start = time.time()
    stale_timeout = _resolve_direct_stale_timeout(agent, api_kwargs)
    # Never override an explicit per-call timeout; otherwise pin read=stale_timeout so a
    # no-op abort can't leave the read=None socket hanging until TCP dies (#85252).
    hard_timeout = _inline_nonstream_hard_timeout(stale_timeout)
    if hard_timeout is not None and "timeout" not in api_kwargs:
        api_kwargs = {**api_kwargs, "timeout": hard_timeout}
    request = _InlineRequest(agent, api_kwargs, stale_timeout, call_start)
    request.start_watchdogs()

    # Only a clean return reports the reuse reason; errors/interrupts really
    # close the client so the retry builds a fresh pool.
    succeeded = False
    try:
        response = _dispatch_nonstreaming_api_request(agent, api_kwargs, make_client=request.make_client)
    except Exception:
        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError("Agent interrupted during API call") from None
        with request.lock:
            was_stale = request.stale
        if was_stale:
            # Our own abort caused the transport error: raise a retryable
            # TimeoutError, never InterruptedError ("the user wants to stop").
            raise TimeoutError(
                f"Non-streaming API call timed out after {int(time.time() - call_start)}s with no response "
                f"(threshold: {int(stale_timeout)}s)") from None
        raise
    else:
        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError("Agent interrupted during API call")
        # Mark ``done`` under the lock so a timer firing between response
        # arrival and unwind is a no-op and cannot overwrite the reset below.
        # If a timer already won, the request still completed: return it (the
        # reset undoes the bump; the finally discards the poisoned client).
        request.mark_done()
        _reset_stale_streak(agent)
        succeeded = True
        return response
    finally:
        request.stop_watchdogs()
        if getattr(agent, "_active_request_abort", None) is request.abort_hook:
            agent._active_request_abort = None
        request_client = request.pop_client()
        if request_client is not None:
            agent._close_request_openai_client(request_client,
                reason="request_complete" if succeeded else "request_error_cleanup")


class _RequestClientRegistry:
    """Per-request client / stream-handle registry shared by the request worker
    and the stranger threads (interrupt loop, stale detector) that may abort it.

    ``kind`` (``"openai"`` / ``"anthropic_messages"`` / ``"stream"``) routes
    :meth:`close_once` (#67142). ``"stream"`` registers a stream handle: under the
    MoA facade the singleton client has no per-request sockets, so interrupts
    must close the stream object itself (#57354).

    Thread-ownership rule (#29507): the owning worker pops + fully closes on its
    way out. A *stranger* thread only aborts the sockets — never ``client.close()``
    — avoiding the FD-recycling race where a just-closed TLS FD was reassigned to
    ``kanban.db`` and the live SSL BIO wrote into the SQLite header. The abort
    happens under the lock: once released the worker may cache the client and the
    NEXT call check it out. Stream handles are safe to close from any thread.
    """

    def __init__(self, agent):
        self.agent = agent
        self.client = None
        self.kind = "openai"
        self.owner_tid = None
        self.diag = None  # per-attempt stream diagnostics (streaming path)
        self.lock = threading.Lock()

    def set_client(self, client, *, kind: str = "openai"):
        with self.lock:
            self.client, self.kind, self.owner_tid = client, kind, threading.get_ident()
        return client

    @staticmethod
    def _stream_close_callable(stream):
        for owner in (stream, getattr(stream, "response", None)):
            close = getattr(owner, "close", None)
            if callable(close):
                return close
        return None

    def set_stream_handle(self, stream):
        return stream if self._stream_close_callable(stream) is None else self.set_client(stream, kind="stream")

    def _close_stream_handle(self, stream, reason: str) -> None:
        close = self._stream_close_callable(stream)
        if close is None:
            return
        try:
            close()
            logger.info("Streaming response handle closed (%s)", reason)
        except Exception as exc:
            logger.debug("Streaming response handle close failed (%s): %s", reason, exc)

    def close_once(self, reason: str) -> None:
        with self.lock:
            request_client, request_kind, owner_tid = self.client, self.kind, self.owner_tid
            stranger_thread = (
                request_kind != "stream"
                and request_client is not None
                and owner_tid is not None
                and owner_tid != threading.get_ident()
            )
            if stranger_thread:
                abort = (self.agent._abort_request_anthropic_client if request_kind == "anthropic_messages"
                         else self.agent._abort_request_openai_client)
                abort(request_client, reason=reason)
                return
            self.client = None
            self.owner_tid = None
        if request_client is None:
            return
        if request_kind == "stream":
            self._close_stream_handle(request_client, reason)
        elif request_kind == "anthropic_messages":
            self.agent._close_request_anthropic_client(request_client, reason=reason)
        else:
            self.agent._close_request_openai_client(request_client, reason=reason)


@dataclass
class _NonStreamWatchdogs:
    """Poll-loop thresholds for one non-streaming request."""
    stale_timeout: float
    codex: bool            # api_mode == codex_responses (codex watchdogs armed)
    est_tokens: int
    ttfb_enabled: bool
    ttfb_timeout: float
    idle_enabled: bool
    idle_timeout: float


def _resolve_nonstream_watchdogs(agent, api_kwargs: dict) -> _NonStreamWatchdogs:
    """Stale-call timeout plus the Codex Responses stream watchdogs.

    The stale detector kills a hung provider early so the retry loop can rotate
    credentials / fall back. Codex adds two failure modes: accepting the connection
    but never emitting an event (no-byte TTFB cutoff; a reconnect succeeds in ~2s)
    and stalling after the opening SSE frame (event-idle gap; any SSE event is
    activity). Tunables: HERMES_CODEX_TTFB_TIMEOUT_SECONDS,
    HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS (0 disables each),
    HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS / HERMES_CODEX_TTFB_STRICT,
    HERMES_CODEX_TTFB_MAX_SECONDS, HERMES_CODEX_HARD_TIMEOUT_SECONDS.
    """
    stale_timeout = agent._compute_non_stream_stale_timeout(api_kwargs)
    codex = agent.api_mode == "codex_responses"
    openai_codex_backend = _is_openai_codex_backend(agent)
    est_tokens = estimate_request_context_tokens(api_kwargs)
    if codex and openai_codex_backend:
        # Raise the stale floor for large payloads so healthy gateway-scale
        # requests aren't aborted mid-prefill.
        codex_floor = openai_codex_stale_timeout_floor(est_tokens)
        if codex_floor:
            stale_timeout = max(stale_timeout, codex_floor)
        # Flat hard ceiling (#64507) for a request that emits SOME bytes then wedges.
        # Default sits ABOVE the max floor (1200s) — a backstop, never tighter. 0 disables.
        hard_timeout = env_float("HERMES_CODEX_HARD_TIMEOUT_SECONDS", 1500.0)
        if hard_timeout > 0:
            stale_timeout = min(stale_timeout, hard_timeout)

    idle_default = next(
        (default for threshold, default in ((100_000, 180.0), (50_000, 120.0), (10_000, 60.0)) if est_tokens > threshold),
        12.0)

    # No-byte TTFB cutoff. Default 120s: the SDK's own read timeout is 600s,
    # and a tight 12s killed subscription-backed requests mid-prefill.
    ttfb_enabled = codex
    ttfb_timeout = env_float("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", 120.0)
    if ttfb_timeout <= 0:
        ttfb_enabled = False
    elif openai_codex_backend:
        # Large requests legitimately spend tens of seconds in admission/prefill before the
        # first SSE event: scale the cutoff up to the idle default unless TTFB_STRICT is set.
        disable_above = env_float("HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS", 10_000.0)
        strict = os.environ.get("HERMES_CODEX_TTFB_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}
        if not strict and disable_above > 0 and est_tokens >= disable_above and ttfb_timeout < idle_default:
            logger.info("Scaling openai-codex no-byte TTFB watchdog from %.0fs to %.0fs "
                "for large request (context=~%s tokens >= %.0f). "
                "Set HERMES_CODEX_TTFB_STRICT=1 to keep the smaller cutoff.", ttfb_timeout, idle_default,
                f"{est_tokens:,}", disable_above)
            ttfb_timeout = idle_default
        ttfb_cap = env_float("HERMES_CODEX_TTFB_MAX_SECONDS", 120.0)
        if ttfb_cap > 0 and ttfb_timeout > ttfb_cap:
            logger.info("Capping openai-codex no-byte TTFB timeout from %.0fs to %.0fs "
                "(context=~%s tokens). Set HERMES_CODEX_TTFB_MAX_SECONDS to tune.", ttfb_timeout, ttfb_cap,
                f"{est_tokens:,}")
            ttfb_timeout = ttfb_cap

    idle_timeout = env_float("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", idle_default)
    return _NonStreamWatchdogs(stale_timeout=stale_timeout, codex=codex, est_tokens=est_tokens,
        ttfb_enabled=ttfb_enabled, ttfb_timeout=ttfb_timeout, idle_enabled=codex and idle_timeout > 0,
        idle_timeout=idle_timeout)


def _codex_silent_hang_hint(agent, api_kwargs: dict) -> Optional[str]:
    hint_fn = getattr(agent, "_codex_silent_hang_hint", None)
    with contextlib.suppress(Exception):
        if callable(hint_fn):
            return hint_fn(model=api_kwargs.get("model"))
    return None


class _NonStreamRequest:
    """One non-streaming request on a worker thread, polled by the caller.

    State shared between the worker (``_call``) and the poll loop lives on the
    instance; ``_abort_request`` may run from the poll (stranger) thread.
    """

    def __init__(self, agent, api_kwargs: dict):
        self.agent = agent
        self.api_kwargs = api_kwargs
        self.result = {"response": None, "error": None}
        self.clients = _RequestClientRegistry(agent)
        # Request-local cancel flag: agent._interrupt_requested is cleared at turn
        # boundaries but this daemon worker can outlive the turn, so it must know THIS
        # request was force-closed and not surface the transport error as a bug (#6600).
        self.cancelled = False
        # Codex retirement token: the worker checks ``agent._active_codex_stream_request_token``
        # to know it still owns the turn; a watchdog kill clears it so a worker still
        # draining SSE raises instead of returning partial output as "completed"
        # (run_codex_stream._request_is_current). ``codex_retired`` mirrors it locally.
        self.codex_token = object() if agent.api_mode == "codex_responses" else None
        self.codex_retired = False
        self.wd = _resolve_nonstream_watchdogs(agent, api_kwargs)
        self.call_start = time.time()
        self.thread = None

    def _install_codex_request_token(self) -> None:
        if self.codex_token is not None and not self.codex_retired:  # retired before start: don't re-publish
            self.agent._active_codex_stream_request_token = self.codex_token

    def _retire_codex_request_token(self) -> None:
        if self.codex_token is None:
            return
        self.codex_retired = True
        if getattr(self.agent, "_active_codex_stream_request_token", None) is self.codex_token:
            self.agent._active_codex_stream_request_token = None

    def _make_client(self, reason: str, kind: str = "openai"):
        # Per-request clients are registered with the abort machinery so the watchdogs
        # force-close the worker's connection, never the shared client (#67142).
        if kind == "anthropic_messages":
            client = self.agent._create_request_anthropic_client(reason=reason)
        else:
            client = self.agent._create_request_openai_client(reason=reason, api_kwargs=self.api_kwargs)
        return self.clients.set_client(client, kind=kind)

    def _call(self):
        try:
            self._install_codex_request_token()
            self.result["response"] = _dispatch_nonstreaming_api_request(
                self.agent, self.api_kwargs, make_client=self._make_client)
        except Exception as e:
            # Our own force-close caused this error: swallow it, the main
            # thread raises InterruptedError (#6600). Retirement logs at info
            # (a watchdog discarded output the provider already sent — what an
            # operator debugging a truncated reply needs); cancellation at debug.
            if self.codex_retired:
                logger.info("Codex worker caught %s after request retirement — "
                    "discarding the stale partial instead of surfacing it as a completed response. %s",
                    type(e).__name__, self.agent._client_log_context())
                return
            if self.cancelled:
                logger.debug("Non-streaming worker caught %s after request "
                    "cancellation — exiting without surfacing a network error.", type(e).__name__)
                return
            self.result["error"] = e
        finally:
            # Retire first: close_once can raise, and a leaked token would let
            # a later worker mistake itself for the owning attempt.
            self._retire_codex_request_token()
            # Reuse reason only on a clean response; error or cancel-swallow
            # really closes so the next attempt builds a fresh pool.
            self.clients.close_once(
                "request_complete" if self.result["response"] is not None else "request_error_cleanup")

    def _abort_request(self, reason: str) -> None:
        """Watchdog/interrupt kill: abort the request client (kind-aware, #67142)
        and retire the codex token; the worker sees its own forced close via
        the cancel flags."""
        with contextlib.suppress(Exception):
            self.clients.close_once(reason)
        self._retire_codex_request_token()

    def _await_worker_after_kill(self, timeout_message: str) -> None:
        # Wait briefly for the worker to notice the closed connection.
        self.thread.join(timeout=2.0)
        if self.result["error"] is None and self.result["response"] is None:
            self.result["error"] = TimeoutError(timeout_message)

    def _model(self) -> str:
        return self.api_kwargs.get("model", "unknown")

    def _emit_wait_notice(self, elapsed: float) -> None:
        wd = self.wd
        try:
            recovery = _codex_wait_notice_recovery(stale_timeout=wd.stale_timeout,
                ttfb_enabled=wd.ttfb_enabled, ttfb_timeout=wd.ttfb_timeout,
                last_event_ts=getattr(self.agent, "_codex_stream_last_event_ts", None),
                call_start=self.call_start, idle_enabled=wd.idle_enabled, idle_timeout=wd.idle_timeout,
                elapsed=elapsed)
            self.agent._emit_wait_notice(
                f"⏳ waiting on {self.api_kwargs.get('model', 'the provider')} — "
                f"{int(elapsed)}s with no response yet (provider may be slow or overloaded{recovery})")
        except Exception:
            logger.debug("wait-notice construction failed", exc_info=True)

    def _ttfb_kill(self, elapsed: float) -> None:
        """No Codex event past the first-byte cutoff — kill so the retry loop
        reconnects instead of waiting out the stale timeout."""
        agent, wd = self.agent, self.wd
        silent_hint = _codex_silent_hang_hint(agent, self.api_kwargs)
        logger.warning("Codex stream produced no bytes within TTFB cutoff "
            "(%.0fs > %.0fs, model=%s). Backend accepted the connection "
            "but sent no stream events. Killing connection so the retry loop can reconnect.", elapsed,
            wd.ttfb_timeout, self._model())
        agent._buffer_status(
            f"⚠️ No first byte from provider in {int(elapsed)}s (codex stream, model: {self._model()}). "
            f"Reconnecting." + (f" {silent_hint}" if silent_hint else ""))
        self._abort_request("codex_ttfb_kill")
        agent._emit_wait_notice(f"⚠ no response from provider in {int(elapsed)}s — reconnecting...")
        agent._touch_activity(f"codex stream killed after {int(elapsed)}s with no first byte")
        self._await_worker_after_kill(
            f"Codex stream produced no bytes within {int(elapsed)}s (TTFB threshold: {int(wd.ttfb_timeout)}s)"
            + (f". {silent_hint}" if silent_hint else ""))

    def _idle_kill(self, event_stale_elapsed: float) -> None:
        """First byte arrived, then SSE events stopped (keepalive/in_progress
        frames refresh the timestamp and don't count)."""
        agent, wd = self.agent, self.wd
        logger.warning("Codex stream produced no SSE events for %.0fs after first byte "
            "(threshold %.0fs, model=%s, context=~%s tokens). Killing "
            "connection so the retry loop can reconnect.", event_stale_elapsed, wd.idle_timeout,
            self._model(), f"{wd.est_tokens:,}")
        agent._buffer_status(
            f"⚠️ Codex stream sent no events for {int(event_stale_elapsed)}s after first byte "
            f"(model: {self._model()}). Reconnecting.")
        self._abort_request("codex_stream_idle_kill")
        agent._touch_activity(f"codex stream killed after {int(event_stale_elapsed)}s with no SSE events")
        self._await_worker_after_kill(
            f"Codex stream produced no SSE events for {int(event_stale_elapsed)}s "
            f"after first byte (threshold: {int(wd.idle_timeout)}s)")

    def _stale_kill(self, elapsed: float) -> None:
        """No response within the stale timeout: kill and count toward the
        circuit breaker (#58962, see ``_stale_streak``)."""
        agent, wd = self.agent, self.wd
        silent_hint = _codex_silent_hang_hint(agent, self.api_kwargs)
        _report_stale_nonstream_kill(agent, self.api_kwargs, elapsed, wd.stale_timeout, hint=silent_hint)
        self._abort_request("stale_call_kill")
        _bump_stale_streak(agent)
        _touch_stale_kill_activity(agent, elapsed)
        self._await_worker_after_kill(
            f"Non-streaming API call timed out after {int(elapsed)}s with no response (threshold: {int(wd.stale_timeout)}s)"
            + (f". {silent_hint}" if silent_hint else ""))

    def _interrupt(self, elapsed: float) -> None:
        agent = self.agent
        _record_interrupted_provider_wait(agent, elapsed,
            response_started=self.wd.codex and getattr(agent, "_codex_stream_last_event_ts", None) is not None
        )
        # Mark cancelled BEFORE force-closing so the worker treats the transport
        # error as a cancel (#6600). Never close the shared client (releasing a
        # TLS FD mid-SSL-BIO corrupted an unrelated SQLite DB, #67142). Then let
        # the worker unwind Relay scopes before raising (#81521).
        self.cancelled = True
        logger.debug("Force-closing httpx client due to interrupt (not a network error).")
        self._abort_request("interrupt_abort")
        _join_worker_for_relay_teardown(self.thread, label="Non-streaming")
        raise InterruptedError("Agent interrupted during API call")

    def run(self):
        agent, wd = self.agent, self.wd
        if wd.codex:
            # Reset before the worker starts so a marker left over from a previous
            # call on this agent can't be misread as first-byte for this one.
            agent._codex_stream_last_event_ts = None
            agent._codex_stream_last_progress_ts = None
        agent._touch_activity("waiting for non-streaming API response")

        self.thread = t = threading.Thread(target=_context_thread_target(self._call), daemon=True)
        t.start()
        poll_count = 0
        while t.is_alive():
            t.join(timeout=0.3)
            poll_count += 1
            # Every ~30s: gateway inactivity heartbeat + rewrite the status line
            # so users see WHAT the wait is (the "infinite thinking" complaint).
            elapsed = time.time() - self.call_start
            if poll_count % 100 == 0:  # 100 × 0.3s = 30s
                self._emit_wait_notice(elapsed)
            last_event_ts = getattr(agent, "_codex_stream_last_event_ts", None)
            if wd.ttfb_enabled and elapsed > wd.ttfb_timeout and last_event_ts is None:
                self._ttfb_kill(elapsed)
                break
            if wd.idle_enabled and last_event_ts is not None and (time.time() - last_event_ts) > wd.idle_timeout:
                self._idle_kill(time.time() - last_event_ts)
                break
            if elapsed > wd.stale_timeout:
                self._stale_kill(elapsed)
                break
            if agent._interrupt_requested:
                self._interrupt(elapsed)
        if self.result["error"] is not None:
            raise self.result["error"]
        # Success — the provider proved responsive: clear the breaker (#58962).
        if self.result["response"] is not None:
            _reset_stale_streak(agent)
        return self.result["response"]


def interruptible_api_call(agent, api_kwargs: dict):
    """Run the API call on a worker thread so the caller can detect interrupts
    without waiting for the full HTTP round-trip. Each worker gets its own
    per-request client (interrupts close only that one); a stale-call detector
    kills the connection and raises so the main retry loop can back off / rotate
    credentials / fall back."""
    # Nested-pool contexts (cron, delegated children) wedge on a worker thread
    # (#62151): run inline. See should_use_direct_api_call.
    if should_use_direct_api_call(agent):
        return direct_api_call(agent, api_kwargs)
    _check_stale_giveup(agent)  # cross-turn stale breaker (#58962), non-streaming sibling
    return _NonStreamRequest(agent, api_kwargs).run()


def _consume_ephemeral_reasoning_off(agent) -> bool:
    """Consume the one-shot "answer without thinking" continuation flag.

    Set by the length-continuation path when a request returned reasoning but NO
    visible content (thinking ate the output cap); continuation turns never replay
    prior reasoning, so thinking ON would re-burn the budget. When True the caller
    overrides the wire reasoning_config with ``{"enabled": False, "effort": "none"}``
    for exactly the next call. Prompt-cache cost is bounded to ONE cold prefix write
    on config-sensitive providers (Anthropic, OpenAI) — far cheaper than four futile
    full-budget continuations.
    """
    consumed = bool(getattr(agent, "_ephemeral_reasoning_off", False))
    if consumed:
        agent._ephemeral_reasoning_off = False
    return consumed


def _reasoning_config_for_wire(agent):
    """``agent.reasoning_config`` with the one-shot reasoning-off override applied.

    Once the route has answered a disable with "reasoning is mandatory"
    (``agent._reasoning_disable_rejected``), every disable — configured or
    the one-shot continuation override — is dropped for the rest of the
    session: the request goes out without a reasoning config and the route
    applies its own default.
    """
    cfg = agent.reasoning_config
    ephemeral_off = _consume_ephemeral_reasoning_off(agent)
    if getattr(agent, "_reasoning_disable_rejected", False):
        # The route rejects disables. Resend exactly what the session has
        # been sending — the user's own config — so the retry lands on the
        # same provider cache key as every prior request. Only a config that
        # is itself a disable is dropped (omitted → route default), and that
        # session has never sent anything else, so nothing warm is lost.
        if isinstance(cfg, dict) and (
            cfg.get("enabled") is False or cfg.get("effort") == "none"
        ):
            return None
        return cfg
    if ephemeral_off:
        cfg = {**(cfg or {}), "enabled": False, "effort": "none"}
    return cfg


def _alias_tool_search_bridge_for_xai(agent, transport, tools_for_api):
    """xAI chat-completions reserves ``tool_search`` and 400s when the bridge declares
    it (#95003): rename the wire declaration; ``normalize_response`` maps calls back
    via the transport's ``_last_wire_aliases`` (reset here so a stale map can't
    reverse-map a name this request never aliased). Deep-copy first (#27907)."""
    if transport is not None and hasattr(transport, "_last_wire_aliases"):
        transport._last_wire_aliases = {}
    is_xai_chat = agent.provider in {"xai", "xai-oauth"} or agent._base_url_hostname == "api.x.ai"
    if not (is_xai_chat and tools_for_api):
        return tools_for_api
    try:
        import copy as _copy_xai
        from agent.transports.chat_completions import _rename_tool_search_bridge_for_xai
        has_bridge = any(
            (t.get("function") or {}).get("name") == "tool_search" for t in tools_for_api if isinstance(t, dict)
        )
        if has_bridge:
            tools_for_api = _copy_xai.deepcopy(tools_for_api)
            tools_for_api, alias_map = _rename_tool_search_bridge_for_xai(tools_for_api)
            if transport is not None:
                transport._last_wire_aliases = alias_map
    except Exception as exc:
        logger.warning("%s⚠️ Failed to alias tool_search bridge for xAI: %s", getattr(agent, "log_prefix", ""), exc)
    return tools_for_api


def _consume_ephemeral_max_output(agent):
    """Pop the one-shot ephemeral output cap; whichever path builds the request consumes it."""
    ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
    if ephemeral_out is not None:
        agent._ephemeral_max_output_tokens = None
    return ephemeral_out


def _build_anthropic_kwargs(agent, api_messages, tools_for_api, reasoning_config, request_overrides):
    ctx_len = getattr(agent, "context_compressor", None)
    ephemeral_out = _consume_ephemeral_max_output(agent)
    anthropic_kwargs = agent._get_transport().build_kwargs(model=agent.model,
        messages=agent._prepare_anthropic_messages_for_api(api_messages), tools=tools_for_api,
        max_tokens=ephemeral_out if ephemeral_out is not None else agent.max_tokens,
        reasoning_config=reasoning_config, is_oauth=agent._is_anthropic_oauth,
        preserve_dots=agent._anthropic_preserve_dots(),
        context_length=ctx_len.context_length if ctx_len else None,
        base_url=getattr(agent, "_anthropic_base_url", None),
        fast_mode=request_overrides.get("speed") == "fast",
        drop_context_1m_beta=bool(getattr(agent, "_oauth_1m_beta_disabled", False)))
    # Portal reads ``tags`` / ``session_id`` on its Messages route too, but the profile hook
    # is only consulted by the OpenAI-wire transport — merge here to keep sticky routing.
    return _merge_nous_portal_messages_extra_body(agent, anthropic_kwargs)


def _build_bedrock_kwargs(agent, api_messages, tools_for_api):
    # Bedrock Converse — the adapter converts messages/tools and calls boto3 directly.
    return agent._get_transport().build_kwargs(model=agent.model, messages=api_messages, tools=tools_for_api,
        max_tokens=agent.max_tokens or 4096, region=getattr(agent, "_bedrock_region", None) or "us-east-1",
        guardrail_config=getattr(agent, "_bedrock_guardrail_config", None))


def _build_codex_kwargs(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id):
    from agent.codex_responses_adapter import classify_responses_route
    from agent.native_compaction import native_compaction_context_management
    is_codex_backend, is_xai_responses, is_github_responses = classify_responses_route(agent)
    # Native server-side compaction (gpt-5.6 on direct OpenAI / ChatGPT Codex routes
    # only) — None on every other route/model, leaving the request unchanged.
    context_management = native_compaction_context_management(agent, is_codex_backend=is_codex_backend,
        is_xai_responses=is_xai_responses, is_github_responses=is_github_responses)
    # xAI's /responses endpoint 400s on ``pattern``/``format`` schema keywords and on
    # ``enum`` values containing ``/`` — strip them (#27197). Deep-copy first: the
    # sanitizers mutate in place and tools_for_api aliases agent.tools (#27907).
    if is_xai_responses:
        try:
            import copy as _copy
            from tools.schema_sanitizer import strip_pattern_and_format, strip_slash_enum
            tools_for_api = _copy.deepcopy(tools_for_api)
            tools_for_api, _ = strip_pattern_and_format(tools_for_api)
            tools_for_api, _ = strip_slash_enum(tools_for_api)
        except Exception as exc:
            logger.warning("%s⚠️ Failed to sanitize tool schemas for xAI: %s", getattr(agent, "log_prefix", ""), exc)
    return agent._get_transport().build_kwargs(model=agent.model,
        messages=agent._prepare_messages_for_non_vision_model(api_messages), tools=tools_for_api,
        reasoning_config=reasoning_config, session_id=getattr(agent, "session_id", None),
        cache_scope_id=cache_scope_id, base_url=agent.base_url, max_tokens=agent.max_tokens,
        timeout=agent._resolved_api_call_timeout(), request_overrides=request_overrides,
        provider=getattr(agent, "provider", None), is_github_responses=is_github_responses,
        is_codex_backend=is_codex_backend, is_xai_responses=is_xai_responses,
        github_reasoning_extra=agent._github_models_reasoning_extra_body() if is_github_responses else None,
        replay_encrypted_reasoning=bool(getattr(agent, "_codex_reasoning_replay_enabled", True)),
        context_management=context_management)


def _anthropic_max_output_for_model(agent):
    """Anthropic-compatible max-output fallback (last resort in build_kwargs, never
    overriding an explicit value). Model-gated, not URL-gated: any proxy serving a
    Claude/MiniMax/Qwen3 model needs max_tokens (Messages API treats it as
    mandatory; proxies that omit it default as low as 4096)."""
    with contextlib.suppress(Exception):
        from agent.anthropic_adapter import _get_anthropic_max_output, _ANTHROPIC_OUTPUT_LIMITS
        model_norm = (agent.model or "").lower().replace(".", "-")
        if any(key in model_norm for key in _ANTHROPIC_OUTPUT_LIMITS):
            return _get_anthropic_max_output(agent.model)
    return None


def _build_chat_completions_kwargs(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id):
    transport = agent._get_transport()
    tools_for_api = _alias_tool_search_bridge_for_xai(agent, transport, tools_for_api)

    _is_qwen = agent._is_qwen_portal()
    _is_or = agent._is_openrouter_url()
    _host = agent._base_url_lower
    _is_gh = base_url_host_matches(_host, "models.github.ai") or base_url_host_matches(_host, "githubcopilot.com")
    _is_lmstudio = (agent.provider or "").strip().lower() == "lmstudio"

    # _fixed_temperature_for_model may return the OMIT_TEMPERATURE sentinel
    # (temperature omitted entirely), a numeric override, or None.
    _omit_temp, _fixed_temp = False, None
    with contextlib.suppress(Exception):
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
        _ft = _fixed_temperature_for_model(agent.model, agent.base_url)
        _omit_temp = _ft is OMIT_TEMPERATURE
        _fixed_temp = None if _omit_temp else _ft

    _prefs = _provider_preferences_for_agent(agent)
    _ant_max = _anthropic_max_output_for_model(agent)
    _qwen_meta = {"sessionId": agent.session_id or "hermes", "promptId": str(uuid.uuid4())} if _is_qwen else None
    _profile = None
    with contextlib.suppress(Exception):
        from providers import get_provider_profile
        _profile = get_provider_profile(agent.provider)

    _ephemeral_out = _consume_ephemeral_max_output(agent)
    # Strip image parts for non-vision models on BOTH paths (registered
    # providers with profiles used to bypass it).
    _common = dict(model=agent.model, messages=agent._prepare_messages_for_non_vision_model(api_messages),
        tools=tools_for_api, base_url=agent.base_url, timeout=agent._resolved_api_call_timeout(),
        max_tokens=agent.max_tokens, ephemeral_max_output_tokens=_ephemeral_out,
        max_tokens_param_fn=agent._max_tokens_param, reasoning_config=reasoning_config,
        request_overrides=request_overrides, session_id=getattr(agent, "session_id", None),
        cache_scope_id=cache_scope_id, ollama_num_ctx=agent._ollama_num_ctx,
        provider_preferences=_prefs or None, openrouter_min_coding_score=agent.openrouter_min_coding_score,
        anthropic_max_output=_ant_max, supports_reasoning=agent._supports_reasoning_extra_body(),
        qwen_session_metadata=_qwen_meta)
    if _profile:
        # Profiles handle per-provider quirks via hooks fed the context above.
        return transport.build_kwargs(provider_profile=_profile, **_common)

    # Legacy flag path: only for a provider absent from the providers/ registry.
    return transport.build_kwargs(
        **_common,
        model_lower=(agent.model or "").lower(),
        is_openrouter=_is_or,
        is_nous=base_url_host_matches(_host, "nousresearch.com"),
        is_qwen_portal=_is_qwen,
        is_github_models=_is_gh,
        is_nvidia_nim=base_url_host_matches(_host, "integrate.api.nvidia.com"),
        is_kimi=any(base_url_host_matches(agent.base_url, h) for h in ("api.kimi.com", "moonshot.ai", "moonshot.cn")),
        is_tokenhub=base_url_host_matches(_host, "tokenhub.tencentmaas.com"),
        is_lmstudio=_is_lmstudio,
        is_custom_provider=agent.provider == "custom",
        qwen_prepare_fn=agent._qwen_prepare_chat_messages if _is_qwen else None,
        qwen_prepare_inplace_fn=agent._qwen_prepare_chat_messages_inplace if _is_qwen else None,
        fixed_temperature=_fixed_temp,
        omit_temperature=_omit_temp,
        github_reasoning_extra=agent._github_models_reasoning_extra_body() if _is_gh else None,
        lmstudio_reasoning_options=agent._lmstudio_reasoning_options_cached() if _is_lmstudio else None,
        provider_name=agent.provider,
    )


def build_api_kwargs(agent, api_messages: list, tools_for_api: list | None = None) -> dict:
    """Build the keyword arguments dict for the active API mode.

    Wraps the per-api_mode builder so the OpenCode ``x-opencode-session``
    affinity header rides on every OpenCode request regardless of transport
    (chat_completions / codex_responses / anthropic_messages all route
    OpenCode models). No-op for every other provider.
    """
    from agent.opencode_affinity import merge_opencode_session_headers

    kwargs = _build_api_kwargs_for_mode(agent, api_messages, tools_for_api)
    return merge_opencode_session_headers(
        kwargs,
        getattr(agent, "provider", None),
        getattr(agent, "base_url", None),
        getattr(agent, "session_id", None),
    )


def _build_api_kwargs_for_mode(agent, api_messages: list, tools_for_api: list | None = None) -> dict:
    # One-shot continuation override — consumed exactly once, on the FIRST
    # request this call builds (only one api_mode branch runs per invocation).
    reasoning_config = _reasoning_config_for_wire(agent)
    if tools_for_api is None:
        tools_for_api = agent.tools
    # The one place request_overrides are consumed: static /fast values are already pinned
    # in agent.request_overrides; auto/cold windows layer the fast override per request.
    request_overrides = effective_request_overrides(agent)
    if agent.api_mode == "anthropic_messages":
        return _build_anthropic_kwargs(agent, api_messages, tools_for_api, reasoning_config, request_overrides)
    if agent.api_mode == "bedrock_converse":
        return _build_bedrock_kwargs(agent, api_messages, tools_for_api)
    # Rotation-stable logical cache scope shared by every OpenAI-wire branch
    # (memoized on the agent); anthropic/bedrock above don't use it.
    cache_scope_id = _prompt_cache_scope_for_agent(agent)
    builder = _build_codex_kwargs if agent.api_mode == "codex_responses" else _build_chat_completions_kwargs
    return builder(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id)


def _model_dump_safe(obj):
    """``model_dump(warnings=False)`` (avoids pydantic serializer UserWarnings on
    generic-union SDK models), falling back for shims that reject the kwarg."""
    try:
        return obj.model_dump(warnings=False)
    except TypeError:
        return obj.model_dump()


def _dump_if_model(value):
    return _model_dump_safe(value) if hasattr(value, "model_dump") else value


def _assistant_reasoning_text(agent, assistant_message) -> Optional[str]:
    """Structured reasoning, else inline ``<think>`` blocks embedded in content."""
    reasoning_text = agent._extract_reasoning(assistant_message)
    if not reasoning_text:
        content = flatten_message_text(getattr(assistant_message, "content", None))
        think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
        if think_blocks:
            reasoning_text = "\n\n".join(b.strip() for b in think_blocks if b.strip()) or None
    if reasoning_text and agent.verbose_logging:
        logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")
    # When streaming is active the reasoning was already displayed during the
    # stream (structured deltas or <think> tag extraction); fire only for
    # non-streaming modes (gateway, batch, quiet). Anything not shown during
    # streaming is caught by the CLI post-response fallback.
    if reasoning_text and agent.reasoning_callback and not agent.stream_delta_callback and not agent._stream_callback:
        with contextlib.suppress(Exception):
            agent.reasoning_callback(reasoning_text)
    return _sanitize_surrogates(reasoning_text) if reasoning_text else reasoning_text


def _assistant_content_for_storage(agent, assistant_message):
    # Sanitize surrogates (Kimi/GLM via Ollama emit code points that crash json.dumps),
    # strip inline <think> tags at the storage boundary (they leaked to platforms and
    # polluted titles), then redact inlined credentials before the message enters
    # history / state.db / gateway delivery (no-op with HERMES_REDACT_SECRETS off).
    content = _sanitize_surrogates(flatten_message_text(getattr(assistant_message, "content", None)))
    if isinstance(content, str) and content:
        content = agent._strip_think_blocks(content).strip()
        if content:
            from agent.redact import redact_sensitive_text
            content = redact_sensitive_text(content)
    return content


def _assistant_tool_call_dict(agent, tool_call, index: int) -> dict:
    raw_id = getattr(tool_call, "id", None)
    call_id = getattr(tool_call, "call_id", None)
    if not isinstance(call_id, str) or not call_id.strip():
        call_id, _ = agent._split_responses_tool_id(raw_id)
    if not isinstance(call_id, str) or not call_id.strip():
        if isinstance(raw_id, str) and raw_id.strip():
            call_id = raw_id.strip()
        else:
            _fn = getattr(tool_call, "function", None)
            call_id = agent._deterministic_call_id(getattr(_fn, "name", "") if _fn else "",
                getattr(_fn, "arguments", "{}") if _fn else "{}", index)
    call_id = call_id.strip()

    response_item_id = getattr(tool_call, "response_item_id", None)
    if not isinstance(response_item_id, str) or not response_item_id.strip():
        _, response_item_id = agent._split_responses_tool_id(raw_id)
    response_item_id = agent._derive_responses_function_call_id(call_id,
        response_item_id if isinstance(response_item_id, str) else None)
    # Arguments are deliberately NOT redacted: this dict is replayed to the model every
    # turn, so a ``***`` mask would break credential-dependent commands (#43083).
    tc_dict = {"id": call_id, "call_id": call_id, "response_item_id": response_item_id,
        "type": tool_call.type,
        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}}
    # Preserve extra_content (Gemini thought_signature) or Gemini 3 thinking
    # models 400 on the next request.
    # Tool-call arguments are intentionally NOT redacted here. This dict enters the in-memory conversation
    # history that is replayed to the model on every subsequent turn AND persisted to state.db, which is
    # itself replayed verbatim on session resume (get_messages_as_conversation). Masking a credential to
    # `***` here poisons that replay: the model reads back its own `PGPASSWORD='***' psql ...` call and
    # copies the placeholder into the next tool call, breaking every credential-dependent command on the
    # second turn (#43083). The masking also provided no real protection — the same secret still leaks
    # verbatim through tool OUTPUT (file contents, command output, diffs, the compaction block), none of
    # which this pass ever touched. Keeping secrets out of the replayable store is a separate
    # tokenization/vault concern, not something arg-redaction can deliver without breaking replay.
    # Storage-time redaction remains governed by the `security.redact_secrets` toggle. (#19798 introduced
    # this; #43083 removed it.) Preserve extra_content (e.g. Gemini thought_signature) so it is sent back on
    # subsequent API calls. Without this, Gemini 3 thinking models reject the request with a 400 error.
    extra = getattr(tool_call, "extra_content", None)
    if extra is not None:
        tc_dict["extra_content"] = _dump_if_model(extra)
    return tc_dict


def build_assistant_message(agent, assistant_message, finish_reason: str) -> dict:
    """Build a normalized assistant message dict (reasoning, reasoning_details,
    optional tool_calls) shared by the tool-call and final-response paths.
    Textless turns are NOT padded here: ``repair_empty_non_final_messages`` is the
    single owner — write-time padding broke codex commentary turns and cannot
    survive ``_rows_to_conversation``."""
    assistant_tool_calls = getattr(assistant_message, "tool_calls", None)
    reasoning_text = _assistant_reasoning_text(agent, assistant_message)
    msg = stamp_message_timestamp({"role": "assistant",
        "content": _assistant_content_for_storage(agent, assistant_message), "reasoning": reasoning_text,
        "finish_reason": finish_reason})

    raw_reasoning_content = getattr(assistant_message, "reasoning_content", None)
    if raw_reasoning_content is None:
        model_extra = getattr(assistant_message, "model_extra", None) or {}
        if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
            raw_reasoning_content = model_extra["reasoning_content"]
    if raw_reasoning_content is not None:
        msg["reasoning_content"] = _sanitize_surrogates(raw_reasoning_content)
    elif assistant_tool_calls and agent._needs_thinking_reasoning_pad():
        # DeepSeek v4 / Kimi thinking modes 400 on a replayed tool-call message without
        # reasoning_content; pad with a single space (empty string is rejected too).
        # Without it, replaying the persisted message causes HTTP 400 ("The reasoning_content in the
        # thinking mode must be passed back to the API"). Include streamed reasoning text when captured;
        # otherwise pad with a single space — DeepSeek V4 Pro tightened validation and rejects empty string
        # ("The reasoning content in the thinking mode must be passed back to the API"). A space satisfies
        # non-empty checks everywhere without leaking fabricated reasoning. Refs #15250, #17400, #17341.
        msg["reasoning_content"] = reasoning_text or " "
    elif reasoning_text:
        # Streaming-only providers accumulate reasoning via deltas and never set
        # it on the message; replaying through a thinking model then 400s.
        # Promote ONLY when nothing set the field: SDK reasoning_content and the
        # tool-call pad win, and reasoning-less turns leave the field absent so
        # the replay-time leak guard and promotion tiers still apply.
        # Additive fallback (refs #16844, #16884). Streaming-only providers (glm, MiniMax, gpt-5.x via aigw,
        # Anthropic via openai-compat shims) accumulate reasoning through ``delta.reasoning_content`` chunks
        # but never land it on the message object as a top-level attribute, so neither branch above fires
        # and the chain-of-thought is stored only under the internal ``reasoning`` key. When the user later
        # replays that history through a DeepSeek-v4 / Kimi thinking model, the missing
        # ``reasoning_content`` causes HTTP 400 ("The reasoning_content in the thinking mode must be passed
        # back to the API."). Promote the already-sanitized streamed ``reasoning_text`` to
        # ``reasoning_content`` at write time, but ONLY when no prior branch already set it AND we actually
        # captured reasoning text. This preserves every existing behavior: - SDK-exposed
        # ``reasoning_content`` (OpenAI/Moonshot/DeepSeek SDK) still wins.
        msg["reasoning_content"] = reasoning_text

    if getattr(assistant_message, "reasoning_details", None):
        # Preserve reasoning_details exactly (opaque signature /
        # encrypted_content fields) for cross-turn reasoning continuity.
        preserved = []
        for d in assistant_message.reasoning_details:
            if isinstance(d, dict):
                preserved.append(d)
            elif hasattr(d, "__dict__"):
                preserved.append(d.__dict__)
            elif hasattr(d, "model_dump"):
                preserved.append(_model_dump_safe(d))
        if preserved:
            msg["reasoning_details"] = preserved

    # Provider-native carriers replayed verbatim on later turns:
    # anthropic_content_blocks keeps interleaved thinking + tool_use order
    # (reconstruction reorders signed blocks -> HTTP 400); codex_* items are
    # the encrypted reasoning / exact message items Responses prefix caching
    # needs.
    for attr in ("anthropic_content_blocks", "bedrock_content_blocks", "codex_reasoning_items", "codex_message_items"):
        value = getattr(assistant_message, attr, None)
        if value:
            msg[attr] = value

    if assistant_tool_calls:
        msg["tool_calls"] = [_assistant_tool_call_dict(agent, tc, i) for i, tc in enumerate(assistant_tool_calls)]
    return msg


def rewrite_prompt_model_identity(agent, model: str, provider: str) -> None:
    """Rewrite the cached prompt's ``Model:``/``Provider:`` lines after a provider switch.

    Not persisted: the stored row keeps the primary's labels so a restored primary replays a
    byte-identical prompt (prefix cache intact). Only the LAST occurrence of each line is touched —
    earlier matches may be user content (memory snapshots, context files)."""
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
    return (str(fb.get("provider") or "").strip().lower(), str(fb.get("model") or "").strip(),
            str(fb.get("base_url") or "").strip().rstrip("/"))


def _fallback_entry_unavailable_without_network(agent, fb: dict) -> Optional[str]:
    """Return a skip reason for fallback entries known to be unusable locally."""
    if (fb.get("provider") or "").strip().lower() != "nous":
        return None
    try:
        from hermes_cli.auth import get_provider_auth_state
        state = get_provider_auth_state("nous") or {}
    except Exception as exc:
        return f"nous_auth_unreadable:{type(exc).__name__}"
    has_token = any(isinstance(t, str) and t.strip() for t in (state.get("access_token"), state.get("refresh_token")))
    return None if has_token else "nous_token_missing"


_FALLBACK_REASON_LABELS = {
    FailoverReason.auth: "authentication failed",
    FailoverReason.auth_permanent: "authentication permanently failed",
    FailoverReason.billing: "billing or quota exhausted",
    FailoverReason.rate_limit: "rate limit",
    FailoverReason.upstream_rate_limit: "upstream model rate limit",
    FailoverReason.overloaded: "provider overloaded",
    FailoverReason.server_error: "provider server error",
    FailoverReason.timeout: "request timeout",
    FailoverReason.ssl_cert_verification: "TLS certificate verification failed",
    FailoverReason.context_overflow: "context window exceeded",
    FailoverReason.payload_too_large: "request payload too large",
    FailoverReason.image_too_large: "image payload too large",
    FailoverReason.model_not_found: "model not found",
    FailoverReason.provider_policy_blocked: "provider policy blocked the request",
    FailoverReason.content_policy_blocked: "content policy blocked the request",
    FailoverReason.format_error: "request format rejected",
    FailoverReason.invalid_encrypted_content: "encrypted reasoning state rejected",
    FailoverReason.multimodal_tool_content_unsupported: "multimodal tool content unsupported",
    FailoverReason.thinking_signature: "thinking signature rejected",
    FailoverReason.long_context_tier: "long-context tier unavailable",
    FailoverReason.oauth_long_context_beta_forbidden: "OAuth long-context beta unavailable",
    FailoverReason.llama_cpp_grammar_pattern: "grammar pattern rejected",
    FailoverReason.unknown: "provider failure",
}


def _fallback_reason_text(reason: "FailoverReason | None") -> str:
    """Return a concise operator-facing explanation for a fallback switch."""
    label = _FALLBACK_REASON_LABELS.get(reason)
    return label or str(getattr(reason, "value", None) or reason or "provider failure").replace("_", " ")


def _is_anthropic_wire_url(url: str) -> bool:
    """Same host match as determine_api_mode() / _detect_api_mode_for_url()."""
    return url.rstrip("/").lower().endswith("/anthropic") or base_url_hostname(url) == "api.anthropic.com"


def _fallback_api_mode_hint(fb: dict, fb_provider: str, fb_base_url_hint: Optional[str]) -> tuple[bool, str]:
    """(explicit, api_mode) for a fallback entry from its ORIGINAL base_url: resolve_provider_client()
    rewrites a dual-surface /anthropic base to /v1, losing the Anthropic wire signal. An explicit
    ``api_mode`` always wins (even "chat_completions") and suppresses later re-detection;
    ``provider: anthropic`` without a base_url still resolves to anthropic_messages."""
    explicit = str(fb.get("api_mode") or "").strip()
    if explicit:
        return True, explicit
    if fb_provider == "anthropic" or (fb_base_url_hint and _is_anthropic_wire_url(fb_base_url_hint)):
        return False, "anthropic_messages"
    return False, "chat_completions"


def _fallback_api_mode_resolved(agent, fb_provider: str, fb_model: str, fb_base_url: str) -> str:
    """Re-detect api_mode from provider / resolved base URL / model when the hint pass
    landed on the chat_completions default (never called for an explicit api_mode)."""
    if fb_provider == "openai-codex":
        return "codex_responses"
    if fb_provider in {"nous", "nous-portal", "nousresearch"}:
        # Portal is dual-wire: anthropic/* must land on /v1/messages (the swap rebuilds the native client).
        from hermes_cli.providers import nous_api_mode
        return nous_api_mode(fb_model)
    if _is_anthropic_wire_url(fb_base_url):
        # Named custom providers (cron-anthropic) resolve base_url from config; the hint pass never saw it.
        return "anthropic_messages"
    if agent._is_azure_openai_url(fb_base_url):
        return "chat_completions"  # Azure serves gpt-5.x on /chat/completions — no Responses API.
    # Provider exceptions (Copilot gpt-5-mini) stay inside the requires-responses predicate.
    if agent._is_direct_openai_url(fb_base_url) or agent._provider_model_requires_responses_api(fb_model, provider=fb_provider):
        return "codex_responses"
    host = base_url_hostname(fb_base_url)
    if fb_provider == "bedrock" or (host.startswith("bedrock-runtime.") and base_url_host_matches(fb_base_url, "amazonaws.com")):
        return "bedrock_converse"
    return "chat_completions"


def _rebind_fallback_credential_pool(agent, fb_provider: str, fb_model: str) -> None:
    """Rebind the credential pool when the provider changes (else rate_limit/billing/auth recovery
    mutates the wrong credentials and overwrites the fallback's base_url). Same-provider pool: kept."""
    existing_pool = getattr(agent, "_credential_pool", None)
    if existing_pool is not None:
        pool_provider = (getattr(existing_pool, "provider", "") or "").strip().lower()
        if pool_provider and pool_provider != fb_provider:
            logger.info(
                "Fallback to %s/%s: clearing primary credential pool (pool_provider=%s) to prevent cross-provider contamination",
                fb_provider, fb_model, pool_provider)
            agent._credential_pool = agent._credential_pool_entry_id = None
    if getattr(agent, "_credential_pool", None) is None:
        try:
            from agent.credential_pool import load_pool
            fallback_pool = load_pool(fb_provider)
            if fallback_pool and fallback_pool.has_credentials():
                agent._credential_pool = fallback_pool
                logger.info("Fallback to %s/%s: attached fallback credential pool", fb_provider, fb_model)
        except Exception as exc:
            logger.debug("Fallback to %s/%s: could not attach credential pool: %s", fb_provider, fb_model, exc)


_RATE_LIMIT_FAILOVER_REASONS = frozenset({FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit})


def _arm_rate_limit_cooldown(agent, reason: "FailoverReason | None") -> None:
    """Arm the primary's exponential cooldown (60s → 2m → ... → 4h cap) on CONSECUTIVE rate-limits;
    restore_primary_runtime resets the counter. Only when leaving the primary: chain-switching from
    an active fallback means the primary was not the 429 source, so its cooldown is left alone."""
    if reason not in _RATE_LIMIT_FAILOVER_REASONS:
        return
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
    if getattr(agent, "_fallback_activated", False) and not (primary_provider and current_provider == primary_provider):
        return
    backoff_count = getattr(agent, "_rate_limit_backoff_count", 0)
    agent._rate_limit_backoff_count = backoff_count + 1
    backoff_seconds = min(60 * (2 ** backoff_count), 14400)
    agent._rate_limited_until = time.monotonic() + backoff_seconds
    logging.info("Rate-limit backoff level %d: cooldown %d s (%.1f min, backoff#%d)", backoff_count, backoff_seconds, backoff_seconds / 60, backoff_count + 1)


def _fallback_chain_exhausted(agent, reason: "FailoverReason | None") -> bool:
    """Chain exhausted (always False). A non-empty chain walked on a non-rate-limit failure arms a
    short cooldown so next turn's restore_primary_runtime stays gated instead of replaying the whole
    context across every provider again."""
    if agent._fallback_chain and reason not in _RATE_LIMIT_FAILOVER_REASONS:
        agent._rate_limited_until = max(
            getattr(agent, "_rate_limited_until", 0) or 0, time.monotonic() + _FALLBACK_EXHAUSTED_COOLDOWN_S)
    return False


def _should_skip_fallback_candidate(agent, fb: dict, fb_key: tuple, fb_provider: str, fb_model: str, unavailable: set) -> bool:
    """True when the entry is already unavailable, malformed, locally unusable, or resolves
    to the backend that just failed (falling back to it would loop the failure)."""
    if fb_key in unavailable:
        logger.debug("Fallback skip: %s previously marked unavailable", fb_key)
        return True
    if not fb_provider or not fb_model:
        return True
    local_skip_reason = _fallback_entry_unavailable_without_network(agent, fb)
    if local_skip_reason:
        unavailable.add(fb_key)
        logger.warning("Fallback skip: %s/%s is not locally usable (%s); suppressing for this session", fb_provider, fb_model, local_skip_reason)
        return True
    # Identity semantics (axes, shim aliases, credential surfaces, multi-endpoint pools)
    # are owned by agent.backend_identity — do not re-implement comparisons here.
    # Skip entries that resolve to the same backend that just failed — falling back to it loops the failure.
    # See #22548, #62984, #70893.
    from agent.backend_identity import BackendIdentity, should_skip_candidate
    current_ident = BackendIdentity.build(provider=getattr(agent, "provider", ""),
        model=getattr(agent, "model", ""), base_url=str(getattr(agent, "base_url", "") or ""))
    fb_ident = BackendIdentity.build(provider=fb_provider, model=fb_model, base_url=(fb.get("base_url") or ""))
    if should_skip_candidate(fb_ident, current_ident):
        logger.warning(
            "Fallback skip: chain entry %s/%s resolves to the same backend as the current one (%s)",
            fb_provider, fb_model, current_ident.base_url or current_ident.provider)
        return True
    return False


def _swap_fallback_clients(agent, fb_client, fb_provider: str, fb_model: str, fb_base_url: str, fb_api_mode: str) -> None:
    """Install the fallback client(s) in place, honoring request_timeout_seconds (None = SDK default)."""
    timeout = get_provider_request_timeout(fb_provider, fb_model)
    if fb_api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_client
        from agent.anthropic_credentials import resolve_anthropic_token, _is_oauth_token
        is_anthropic = fb_provider == "anthropic"
        effective_key = fb_client.api_key or (resolve_anthropic_token() if is_anthropic else None) or ""
        agent.api_key = agent._anthropic_api_key = effective_key
        agent._anthropic_base_url = fb_base_url
        agent._anthropic_client = build_anthropic_client(effective_key, fb_base_url, timeout=timeout)
        agent._is_anthropic_oauth = _is_oauth_token(effective_key) if is_anthropic else False
        agent.client, agent._client_kwargs = None, {}
        return
    agent.api_key = fb_client.api_key
    agent.client = fb_client
    # Keep provider headers resolve_provider_client() baked into fb_client (SDK: _custom_headers), else
    # later request-client rebuilds drop them and User-Agent-sentinel providers (Kimi Coding) 403.
    fb_headers = getattr(fb_client, "_custom_headers", None) or getattr(fb_client, "default_headers", None)
    agent._client_kwargs = {"api_key": fb_client.api_key, "base_url": fb_base_url}
    if fb_headers:
        agent._client_kwargs["default_headers"] = dict(fb_headers)
    if timeout is not None:
        agent._client_kwargs["timeout"] = timeout
        # Rebuild now so the timeout applies to the very next request, not only after a rotation rebuild.
        agent._replace_primary_openai_client(reason="fallback_timeout_apply")


def _update_fallback_context_compressor(agent) -> None:
    """Point compression limits at the fallback model's context window (not the primary's),
    respecting the explicit model.context_length config override."""
    compressor = getattr(agent, "context_compressor", None)
    if not compressor:
        return
    from agent.model_metadata import get_model_context_length
    fb_context_length = get_model_context_length(
        agent.model, base_url=agent.base_url,
        api_key=agent.api_key if isinstance(agent.api_key, str) else "",  # callable (Entra ID) → probes need str
        provider=agent.provider,
        config_context_length=getattr(agent, "_config_context_length", None),
        custom_providers=getattr(agent, "_custom_providers", None),
    )
    compressor.update_model(  # callable api_key preserved → call_llm
        model=agent.model, context_length=fb_context_length, base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
    )


def _reresolve_fallback_reasoning_config(agent) -> None:
    """Per-model override > global reasoning_effort (YAML False = disabled); a config load
    failure keeps the current reasoning_config rather than killing the swap."""
    try:
        # Re-resolve reasoning_config for the new fallback model (Closes #21256). Wrapped in try/except
        # because a config load failure must not kill the swap.
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config
        agent.reasoning_config = resolve_reasoning_config(load_config() or {}, agent.model)
        logger.info("Fallback %s: reasoning_config resolved: %s", agent.model, agent.reasoning_config)
    except Exception as _reasoning_err:
        logger.debug("Failed to resolve reasoning_config for fallback %s; keeping current: %s", agent.model, _reasoning_err)


def _rescope_fallback_extra_body(agent, old_model: str, old_provider: str, old_base_url: str) -> None:
    """Drop the OLD provider's custom_providers-contributed extra_body keys, then merge the fallback
    provider's own. KEY-SCOPED: a key is dropped only if its value still equals what the old provider's
    config injected — a caller override of the same key won at init and differs, so it survives;
    keys the new provider redefines are re-added by the merge."""
    try:
        from agent.agent_init import _custom_provider_extra_body_for_agent, _merge_custom_provider_extra_body
        custom_providers = getattr(agent, "_custom_providers", None) or []
        old_provider_eb = _custom_provider_extra_body_for_agent(provider=old_provider, model=old_model, base_url=old_base_url, custom_providers=custom_providers) or {}
        overrides = dict(getattr(agent, "request_overrides", {}) or {})
        existing_eb = overrides.get("extra_body")
        if isinstance(existing_eb, dict) and old_provider_eb:
            scrubbed = {k: v for k, v in existing_eb.items() if not (k in old_provider_eb and v == old_provider_eb[k])}
            if scrubbed:
                overrides["extra_body"] = scrubbed
            else:
                overrides.pop("extra_body", None)
            agent.request_overrides = overrides
        _merge_custom_provider_extra_body(agent, custom_providers)
        logger.info("Fallback %s: extra_body resolved: %s", agent.model, (getattr(agent, "request_overrides", {}) or {}).get("extra_body"))
    except Exception as _eb_err:
        logger.debug("Failed to resolve extra_body for fallback %s; keeping current: %s", agent.model, _eb_err)


def _buffer_fallback_notice(agent, notice: str) -> None:
    """Buffer the switch notice for terminal failure AND retain it as a durable one-shot for
    _emit_pending_fallback_notice (a successful fallback clears retry chatter)."""
    agent._buffer_status(notice)
    pending = getattr(agent, "_pending_fallback_notice", None)
    if isinstance(pending, list):
        pending.append(notice)
    else:
        agent._pending_fallback_notice = [str(pending), notice] if pending else [notice]


def try_activate_fallback(agent, reason: "FailoverReason | None" = None) -> bool:
    """Switch to the next fallback model/provider in the chain; False when exhausted. Swaps client,
    model slug and provider in place so the retry loop continues on the new backend; client
    construction goes through resolve_provider_client (no duplicated provider→key mappings)."""
    _arm_rate_limit_cooldown(agent, reason)
    if agent._fallback_index >= len(agent._fallback_chain):
        return _fallback_chain_exhausted(agent, reason)
    fb = agent._fallback_chain[agent._fallback_index]
    agent._fallback_index += 1
    fb_key = _fallback_entry_key(fb)
    if getattr(agent, "_unavailable_fallback_keys", None) is None:
        agent._unavailable_fallback_keys = set()
    unavailable = agent._unavailable_fallback_keys
    fb_provider = (fb.get("provider") or "").strip().lower()
    fb_model = (fb.get("model") or "").strip()
    if _should_skip_fallback_candidate(agent, fb, fb_key, fb_provider, fb_model, unavailable):
        return agent._try_activate_fallback(reason)

    try:
        from agent.auxiliary_client import resolve_provider_client
        from hermes_cli.fallback_config import resolve_entry_api_key
        # Pass the entry's base_url/api_key so custom endpoints (Ollama Cloud) resolve instead
        # of falling through to OpenRouter defaults.
        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = resolve_entry_api_key(fb)
        fb_api_mode_explicit, fb_api_mode = _fallback_api_mode_hint(fb, fb_provider, fb_base_url_hint)
        # Ollama Cloud: OLLAMA_API_KEY from env when the entry has no key. Host match, not
        # substring — GHSA-76xc-57q6-vm5m.
        if fb_base_url_hint and base_url_host_matches(fb_base_url_hint, "ollama.com") and not fb_api_key_hint:
            from agent.secret_scope import get_secret
            fb_api_key_hint = get_secret("OLLAMA_API_KEY") or None
        # raw_codex=True: the main agent needs direct responses.stream() access for Codex providers.
        fb_client, _resolved_fb_model = resolve_provider_client(
            fb_provider, model=fb_model, raw_codex=True, explicit_base_url=fb_base_url_hint, explicit_api_key=fb_api_key_hint, api_mode=fb_api_mode)
        if fb_client is None:
            logger.warning("Fallback to %s failed: provider not configured", fb_provider)
            unavailable.add(fb_key)
            return agent._try_activate_fallback(reason)
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider
            fb_model = normalize_model_for_provider(fb_model, fb_provider)
        except Exception as _norm_err:
            logger.warning("Could not normalize fallback model %r for provider %r: %s", fb_model, fb_provider, _norm_err)

        fb_base_url = str(fb_client.base_url)
        if not fb_api_mode_explicit and fb_api_mode == "chat_completions":
            fb_api_mode = _fallback_api_mode_resolved(agent, fb_provider, fb_model, fb_base_url)

        old_model, old_provider, old_base_url = agent.model, agent.provider, agent.base_url

        # Clear the per-config context_length override so the fallback model's own context
        # window is resolved instead of the previous model's stale value.
        # See #22387.
        agent._config_context_length = None
        agent.model, agent.provider, agent.requested_provider = fb_model, fb_provider, fb_provider
        agent.base_url, agent.api_mode = fb_base_url, fb_api_mode
        # reasoning_content echo opt-in travels with the active provider; restore_primary_runtime reverts it.
        agent._reasoning_echo_flag = bool(fb.get("reasoning_echo", False))
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent._fallback_activated = True

        _rebind_fallback_credential_pool(agent, fb_provider, fb_model)
        _swap_fallback_clients(agent, fb_client, fb_provider, fb_model, fb_base_url, fb_api_mode)

        from agent.agent_runtime_helpers import sync_credential_pool_entry_id
        sync_credential_pool_entry_id(agent)

        agent._use_prompt_caching, agent._use_native_cache_layout = agent._anthropic_prompt_cache_policy(
            provider=fb_provider, base_url=fb_base_url, api_mode=fb_api_mode, model=fb_model)
        agent._ensure_lmstudio_runtime_loaded()  # LM Studio: preload before probing context length
        _update_fallback_context_compressor(agent)
        _reresolve_fallback_reasoning_config(agent)
        _rescope_fallback_extra_body(agent, old_model, old_provider, old_base_url)
        rewrite_prompt_model_identity(agent, fb_model, fb_provider)

        _buffer_fallback_notice(agent, (
            f"⚠️ Model fallback: {old_model} via {old_provider} unavailable "
            f"({_fallback_reason_text(reason)}); using {fb_model} via {fb_provider}."))
        # ``_fallback_activated`` is also reused by `/model --once` restoration; separate
        # provenance so the restore path only emits a recovery notice after a real fallback.
        agent._provider_fallback_active = True
        agent._provider_fallback_route = (str(fb_model), str(fb_provider))
        logger.info("Fallback activated: %s → %s (%s)", old_model, fb_model, fb_provider)
        # The stale-call streak measured the OLD provider; carrying it over would
        # short-circuit the fresh fallback before its first stream attempt.
        _reset_stale_streak(agent)
        from agent.native_compaction import resolve_native_compaction_capabilities
        agent.runtime_capabilities = resolve_native_compaction_capabilities(
            model=agent.model, base_url=agent.base_url, provider=fb_provider, is_codex_backend=fb_provider == "openai-codex")
        return True
    except Exception as e:
        if fb_provider == "nous":
            unavailable.add(fb_key)
        logger.error("Failed to activate fallback %s: %s", fb_model, e)
        return agent._try_activate_fallback(reason)  # try next in chain


# Keys outside the Chat Completions schema that strict gateways (Fireworks-backed OpenCode
# Go, Mistral, Moonshot/Kimi) reject with 422. The transport's convert_messages() drops them
# in the main loop; the summary path calls chat.completions.create() directly, so mirror it.
_SUMMARY_FOREIGN_MESSAGE_KEYS = ("reasoning", "finish_reason", "tool_name", "codex_reasoning_items",
    "codex_message_items", "timestamp", "platform_message_id")
_EMPTY_SUMMARY_RESPONSE = "I reached the iteration limit and couldn't generate a summary."


def _iteration_summary_api_messages(agent, messages: list) -> list:
    """Wire-ready messages for the summary call, mirroring the main loop's api_messages build
    (sidecar substitution, tool-call repair, thinking-only drop, underscore-key sweep)."""
    needs_sanitize = agent._should_sanitize_tool_calls()
    sanitize_model = agent.model
    if needs_sanitize and agent.provider == "moa":
        # MoA: agent.model is the virtual preset; use the real aggregator so Gemini keeps thought_signature.
        agg_slot = getattr(getattr(agent, "client", None), "last_aggregator_slot", None)
        sanitize_model = (agg_slot or {}).get("model") or sanitize_model
    api_messages = []
    for msg in messages:
        api_msg = msg.copy()
        agent._copy_reasoning_content_for_api(msg, api_msg)
        for key in _SUMMARY_FOREIGN_MESSAGE_KEYS:
            api_msg.pop(key, None)
        # api_content holds the exact bytes the main loop sent; substituting (not popping)
        # keeps the summary's prefix identical instead of re-prefilling the largest context.
        # Strict OpenAI-compatible gateways (Fireworks-backed OpenCode Go, Mistral, Moonshot/Kimi) reject
        # any message key outside the Chat Completions schema. The main loop drops these via
        # ChatCompletionsTransport.convert_messages(), but the summary path hand-builds messages and calls
        # chat.completions.create() directly, bypassing the transport — so mirror that sanitization here:
        # tool_name (SQLite FTS bookkeeping), the codex_* reasoning carriers, timestamp (preserved on
        # gateway user replay entries for the stale-confirmation expiry check — #47868 rejection class), and
        # every Hermes-internal underscore-prefixed scaffolding key.
        substitute_api_content(api_msg)
        if needs_sanitize:
            agent._sanitize_tool_calls_for_strict_api(api_msg, model=sanitize_model)
        api_messages.append(api_msg)

    effective_system = agent._cached_system_prompt or ""
    if agent.ephemeral_system_prompt:
        effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
    if effective_system:
        api_messages = [{"role": "system", "content": effective_system}] + api_messages
    for idx, pfm in enumerate(agent.prefill_messages or ()):
        api_messages.insert((1 if effective_system else 0) + idx, pfm.copy())

    # Compression/resume can orphan a tool result whose parent tool_call was summarized away.
    api_messages = agent._sanitize_api_messages(api_messages)
    # Same send-path vision eviction as the main loop (#89296).
    from agent.context_compressor import evict_stale_outbound_tool_images
    evict_stale_outbound_tool_images(api_messages)
    # Thinking-only assistant turns 400 on Anthropic-family providers; _thinking_prefill must
    # survive until here so the drop pass recognizes stubs after reasoning is stripped.
    api_messages = agent._drop_thinking_only_and_merge_users(api_messages)
    for api_msg in api_messages:  # underscore scaffolding: the transport's sweeper is bypassed here
        if isinstance(api_msg, dict):
            for internal_key in [k for k in api_msg if isinstance(k, str) and k.startswith("_")]:
                del api_msg[internal_key]
    return api_messages


def _managed_summary_call(agent, api_request_id: str, request, callback, *, retry_count: int):
    from agent import relay_llm
    return relay_llm.execute_current(
        request, callback,
        name=str(getattr(agent, "provider", "") or "provider"), model_name=str(getattr(agent, "model", "") or ""),
        metadata={"api_mode": str(getattr(agent, "api_mode", "") or "chat_completions"),
            "api_request_id": api_request_id, "call_role": "iteration_summary", "retry_count": retry_count},
        defer_logical_completion=True,
    )


def _iteration_summary_chat_kwargs(agent, api_messages: list) -> dict:
    """chat.completions.create kwargs for the summary, mirroring ChatCompletionsTransport.build_kwargs()."""
    try:
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
    except Exception:
        _fixed_temperature_for_model = _OMIT_TEMP = None
    raw_temp = _fixed_temperature_for_model(agent.model, agent.base_url) if _fixed_temperature_for_model is not None else None
    temperature = None if raw_temp is _OMIT_TEMP else raw_temp
    provider_name = (agent.provider or "").strip().lower()
    # LM Studio uses top-level `reasoning_effort` (not extra_body.reasoning).
    is_lmstudio = provider_name == "lmstudio" and agent._supports_reasoning_extra_body()
    lm_reasoning_effort = agent._resolve_lmstudio_summary_reasoning_effort() if is_lmstudio else None

    extra_body = {}
    if not is_lmstudio and agent._supports_reasoning_extra_body():
        extra_body["reasoning"] = agent.reasoning_config if agent.reasoning_config is not None else {"enabled": True, "effort": "medium"}
    if "nousresearch" in agent._base_url_lower:
        from agent.portal_tags import nous_portal_tags
        extra_body["tags"] = nous_portal_tags()

    summary_kwargs = {"model": agent.model, "messages": api_messages}
    if temperature is not None:
        summary_kwargs["temperature"] = temperature
    if agent.max_tokens is not None:
        summary_kwargs.update(agent._max_tokens_param(agent.max_tokens))
    if lm_reasoning_effort is not None:
        summary_kwargs["reasoning_effort"] = lm_reasoning_effort

    # Merge the profile's canonical body even when routing is unset (e.g. required Portal tags).
    provider_preferences = _provider_preferences_for_agent(agent)
    profile_extra_body = {}
    with contextlib.suppress(Exception):
        from providers import get_provider_profile
        provider_profile = get_provider_profile(agent.provider)
        if provider_profile is not None:
            profile_extra_body = provider_profile.build_extra_body(
                session_id=getattr(agent, "session_id", None), provider_preferences=provider_preferences or None,
                model=agent.model, base_url=agent.base_url, reasoning_config=agent.reasoning_config)
    if profile_extra_body:
        extra_body.update(profile_extra_body)

    def _is_openrouter() -> bool:
        return provider_name == "openrouter" or agent._is_openrouter_url()

    if provider_preferences and "provider" not in profile_extra_body and _is_openrouter():
        extra_body["provider"] = provider_preferences
    # Pareto Code router plugin — model-gated, same shape as the main-loop emission.
    _score = agent.openrouter_min_coding_score
    if agent.model == "openrouter/pareto-code" and _is_openrouter() and _score is not None and _score != "":
        with contextlib.suppress(TypeError, ValueError):
            _ps = float(_score)
            if 0.0 <= _ps <= 1.0:
                extra_body["plugins"] = [{"id": "pareto-router", "min_coding_score": _ps}]
    if extra_body:
        summary_kwargs["extra_body"] = extra_body
    return summary_kwargs


def _summary_text(agent, response, **normalize_kwargs) -> str:
    return (agent._get_transport().normalize_response(response, **normalize_kwargs).content or "").strip()


def _codex_summary_attempt(agent, api_messages: list, api_request_id: str):
    def _attempt(retry_count: int) -> str:
        codex_kwargs = agent._build_api_kwargs(api_messages)
        codex_kwargs.pop("tools", None)
        return _summary_text(agent, agent._run_codex_stream(codex_kwargs))
    return _attempt


def _anthropic_summary_attempt(agent, api_messages: list, api_request_id: str):
    def _attempt(retry_count: int) -> str:
        ant_kw = agent._get_transport().build_kwargs(
            model=agent.model, messages=api_messages, tools=None, max_tokens=agent.max_tokens,
            reasoning_config=agent.reasoning_config, is_oauth=agent._is_anthropic_oauth,
            preserve_dots=agent._anthropic_preserve_dots(), base_url=getattr(agent, "_anthropic_base_url", None))
        ant_kw = _merge_nous_portal_messages_extra_body(agent, ant_kw)
        response = _managed_summary_call(agent, api_request_id, ant_kw, agent._anthropic_messages_create, retry_count=retry_count)
        return _summary_text(agent, response, strip_tool_prefix=agent._is_anthropic_oauth)
    return _attempt


def _chat_summary_attempt(agent, api_messages: list, api_request_id: str):
    summary_kwargs = _iteration_summary_chat_kwargs(agent, api_messages)

    def _attempt(retry_count: int) -> str:
        summary_client = agent._ensure_primary_openai_client(reason="iteration_limit_summary_retry" if retry_count else "iteration_limit_summary")
        response = _managed_summary_call(
            agent, api_request_id, summary_kwargs, lambda request: summary_client.chat.completions.create(**request), retry_count=retry_count)
        return _summary_text(agent, response)
    return _attempt


_SUMMARY_ATTEMPT_BUILDERS = {"codex_responses": _codex_summary_attempt, "anthropic_messages": _anthropic_summary_attempt}


def handle_max_iterations(agent, messages: list, api_call_count: int) -> str:
    """Request a summary when max iterations are reached. Returns the final response text."""
    warning = f"⚠️  Reached maximum iterations ({agent.max_iterations}). Requesting summary..."
    if getattr(agent, "suppress_status_output", False):
        # Strict machine-readable mode (-Q, oneshot): keep diagnostics off stdout. quiet_mode is
        # NOT the gate — the interactive CLI runs quiet_mode=True by default and must see this.
        # Strict machine-readable mode (hermes chat -Q, oneshot, background review): keep diagnostics out of
        # stdout so wrappers receive only the final assistant content (#93220 class).
        logger.warning(warning)
    else:
        agent._safe_print(warning)

    summary_api_request_id = f"iteration-summary:{uuid.uuid4()}"
    summary_call_outcome = "failed"

    # Shared constant so compaction recognizers can identify this runtime nudge by its stable
    # content after SessionDB projection strips metadata flags.
    from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST
    append_message(messages, {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST})

    try:
        api_messages = _iteration_summary_api_messages(agent, messages)
        build_attempt = _SUMMARY_ATTEMPT_BUILDERS.get(agent.api_mode, _chat_summary_attempt)
        attempt = build_attempt(agent, api_messages, summary_api_request_id)

        # One retry on an empty summary; a summary empty once its <think> block is stripped is NOT retried.
        final_response = _EMPTY_SUMMARY_RESPONSE
        for retry_count in (0, 1):
            text = attempt(retry_count)
            if not text:
                continue
            if "<think>" in text:
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
            if text:
                summary_call_outcome = "success"
                append_message(messages, {"role": "assistant", "content": text})
                final_response = text
            break

    except Exception as e:
        logger.warning("Failed to get summary response: %s", e)
        final_response = f"I reached the maximum iterations ({agent.max_iterations}) but couldn't summarize. Error: {str(e)}"
    finally:
        from agent import relay_llm
        relay_llm.complete_logical_call(summary_api_request_id, outcome=summary_call_outcome)

    return final_response


def cleanup_task_resources(agent, task_id: str) -> None:
    """Per-turn VM + browser cleanup for a task. Skips ``cleanup_vm`` for persistent
    terminal envs (``_cleanup_inactive_envs`` reaps them after ``terminal.lifetime_seconds``)
    and ``cleanup_browser`` in headed mode (the inactivity reaper handles idle sessions)."""
    def _headed() -> bool:
        try:
            from tools.browser_tool_cloud import _is_headed_mode
            return _is_headed_mode()
        except Exception:
            return bool(os.environ.get("AGENT_BROWSER_HEADED"))

    for label, skip, skip_what, cleanup in (
        ("VM", is_persistent_env, "cleanup_vm for persistent env", lambda: _ra().cleanup_vm(task_id)),
        ("browser", lambda _tid: _headed(), "cleanup_browser for headed session", lambda: _ra().cleanup_browser(task_id)),
    ):
        try:
            if skip(task_id):
                if agent.verbose_logging:
                    logging.debug(f"Skipping per-turn {skip_what} {task_id}; idle reaper will handle it.")
            else:
                cleanup()
        except Exception as e:
            if agent.verbose_logging:
                logger.warning("Failed to cleanup %s for task %s: %s", label, task_id, e)


def _build_partial_stream_stub(role, full_content, full_reasoning, model_name, usage_obj, *,
    dropped_tool_names=None):
    """Stub for an SSE stream that ended without ``finish_reason`` after
    delivering content. Tagged ``PARTIAL_STREAM_STUB_ID`` + ``FINISH_REASON_LENGTH``
    so the loop enters its continuation/retry path instead of accepting
    truncated output as a complete turn (#32086)."""
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model=model_name,
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(role=role, content=full_content, tool_calls=None,
                reasoning_content=full_reasoning),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=usage_obj,
        _dropped_tool_names=dropped_tool_names or None,
    )


# SSE error events from proxies (OpenRouter's {"error":{"message":"Network
# connection lost."}}) surface as SDK APIError without a status_code (unlike
# APIStatusError). They mean the upstream stream died: retry with a fresh
# connection like an httpx drop.
_SSE_CONN_PHRASES = ("connection lost", "connection reset", "connection closed", "connection terminated",
    "network error", "network connection", "terminated", "peer closed", "broken pipe",
    "upstream connect error")


def _is_sse_connection_error(exc: BaseException) -> bool:
    from openai import APIError as _APIError
    if not isinstance(exc, _APIError) or getattr(exc, "status_code", None):
        return False
    err_lower = str(exc).lower()
    return any(phrase in err_lower for phrase in _SSE_CONN_PHRASES)


def _relay_stream_identity(agent, name_default: str) -> dict:
    """``session_id``/``name``/``model_name`` kwargs for ``relay_llm.stream``."""
    return {"session_id": str(getattr(agent, "session_id", "") or ""),
        "name": str(getattr(agent, "provider", "") or name_default),
        "model_name": str(getattr(agent, "model", "") or "")}


def _relay_stream_metadata(agent, api_mode: str) -> dict:
    call_role = ("delegated" if getattr(agent, "is_subagent", False)
                 else "fallback" if int(getattr(agent, "_fallback_index", 0) or 0) > 0 else "primary")
    return {"api_mode": api_mode, "api_request_id": getattr(agent, "_current_api_request_id", None),
        "call_role": call_role}


def _stream_final_text(response) -> str:
    with contextlib.suppress(Exception):
        choices = getattr(response, "choices", None)
        first_choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
        content = getattr(getattr(first_choice, "message", None), "content", None)
        if isinstance(content, str):
            return content
    with contextlib.suppress(Exception):
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(t for t in (getattr(part, "text", None) for part in content) if isinstance(t, str))
    return ""


def _with_stream_emitters(agent, run):
    """Bracket ``run()`` with the agent's ``_emit_stream_start`` / ``_emit_stream_end``
    hooks when present (end carries the final text on success, the error string on
    failure) and re-raise."""
    start = getattr(agent, "_emit_stream_start", None)
    if start is not None:
        start()
    try:
        response = run()
    except Exception as exc:
        end = getattr(agent, "_emit_stream_end", None)
        if end is not None:
            end(final_text="", finished=False, error=str(exc))
        raise
    end = getattr(agent, "_emit_stream_end", None)
    if end is not None:
        end(final_text=_stream_final_text(response), finished=True, error=None)
    return response


def _stream_codex_passthrough(agent, api_kwargs: dict, on_first_delta):
    """Codex streams internally via _run_codex_stream (reached through
    _interruptible_api_call); park ``on_first_delta`` on the agent so it can pick
    it up, and bracket the call with the stream start/end emitters."""
    agent._codex_on_first_delta = on_first_delta
    try:
        return _with_stream_emitters(agent, lambda: agent._interruptible_api_call(api_kwargs))
    finally:
        agent._codex_on_first_delta = None


class _BedrockStream:
    """Bedrock Converse streaming: boto3 ``converse_stream()`` on a worker thread
    with real-time delta callbacks, polled by an interrupt / stale-event watchdog
    (same UX as the Anthropic and chat_completions streams)."""

    def __init__(self, agent, api_kwargs: dict, on_first_delta):
        self.agent = agent
        self.api_kwargs = api_kwargs
        self.on_first_delta = on_first_delta
        self.result = {"response": None, "error": None}
        self.first_delta_fired = False
        self.response_started = False
        # Liveness for the boto3 worker: ``for event in event_stream`` has NO read timeout,
        # so on_event stamps every event and the poll loop trips a watchdog on a long gap.
        self.started_at = time.time()
        self.last_event = self.started_at
        # Read (not popped): the worker's own pop inside _open_stream must
        # still resolve the same region.
        self.region = api_kwargs.get("__bedrock_region__", "us-east-1")
        # Same patience budget as the OpenAI/Anthropic stale detector.
        self.stale_timeout = _derive_stream_stale_timeout(agent, api_kwargs)

    def _model(self) -> str:
        return self.api_kwargs.get("modelId", "unknown")

    def _fire_first(self):
        self.response_started = True
        if not self.first_delta_fired and self.on_first_delta:
            self.first_delta_fired = True
            with contextlib.suppress(Exception):
                self.on_first_delta()

    def _after_first(self, fire):
        """Wrap a delta callback so the first delivered event also fires ``on_first_delta``."""
        def _on(value):
            self._fire_first()
            fire(value)
        return _on

    def _open_stream(self, next_api_kwargs: dict[str, Any]):
        return _bedrock_converse_call(dict(next_api_kwargs), stream=True, on_stream_denied=self._fall_back_to_converse)

    def _fall_back_to_converse(self, client, final_kwargs: dict, exc: Exception):
        # InvokeModel-only IAM policies cannot stream; fall back inside the same Relay
        # attempt (one lifecycle boundary).
        from agent.bedrock_adapter import normalize_converse_response
        self.agent._disable_streaming = True
        self.agent._safe_print("\n⚠  AWS IAM denied bedrock:InvokeModelWithResponseStream — "
            "falling back to non-streaming InvokeModel.\n"
            "   Grant that action to restore streaming output.\n")
        logger.info("bedrock: converse_stream denied by IAM (%s) — "
            "using non-streaming converse() for this session.", type(exc).__name__)
        return normalize_converse_response(client.converse(**final_kwargs))

    def _worker(self):
        agent = self.agent
        stream = None
        try:
            from agent import relay_llm
            from agent.bedrock_adapter import stream_converse_with_callbacks
            intercepted_events = []
            writer_token = {"value": None}

            def _stream_created(_stream: Any) -> None:
                writer_token["value"] = claim_stream_writer(agent)

            def _accept_event(_event: Any) -> bool:
                token = writer_token["value"]
                return token is None or stream_writer_is_current(agent, token)

            def _stamp_event() -> None:
                self.last_event = time.time()

            try:
                from agent.plugin_stream_hooks import has_reasoning_stream_observer_hooks
                plugin_reasoning_observer = has_reasoning_stream_observer_hooks()
            except Exception:
                logger.debug("plugin reasoning stream observer check failed", exc_info=True)
                plugin_reasoning_observer = False

            stream = relay_llm.stream(dict(self.api_kwargs), self._open_stream,
                **_relay_stream_identity(agent, "bedrock"),
                finalizer=lambda: stream_converse_with_callbacks({"stream": list(intercepted_events)}),
                on_stream_created=_stream_created, on_chunk=intercepted_events.append,
                chunk_adapter=lambda chunk: chunk, accept_chunk=_accept_event,
                completed_response_predicate=lambda response: bool(getattr(response, "choices", None)),
                metadata=_relay_stream_metadata(agent, "custom"), defer_logical_completion=True)
            wants_reasoning = agent.reasoning_callback or agent.stream_delta_callback or plugin_reasoning_observer
            streamed_response = stream_converse_with_callbacks({"stream": stream},
                on_text_delta=self._after_first(agent._fire_stream_delta) if agent._has_stream_consumers() else None,
                on_tool_start=self._after_first(agent._fire_tool_gen_started),
                on_reasoning_delta=self._after_first(agent._fire_reasoning_delta) if wants_reasoning else None,
                on_interrupt_check=lambda: agent._interrupt_requested, on_event=_stamp_event)
            self.result["response"] = stream.final_response or streamed_response
        except Exception as e:
            self.result["error"] = e
        finally:
            if stream is not None:
                stream.close()

    def _raise_if_interrupted(self, message: str, worker=None) -> None:
        if not self.agent._interrupt_requested:
            return
        _record_interrupted_provider_wait(
            self.agent, time.time() - self.started_at, response_started=self.response_started)
        if worker is not None:
            # Let the worker unwind Relay scopes before raising (#81521).
            _join_worker_for_relay_teardown(worker, label="Bedrock streaming")
        raise InterruptedError(message)

    def _on_stale(self, stale_elapsed: float) -> None:
        """No event past the stale timeout = wedged stream (the worker would
        block in the event loop forever)."""
        agent = self.agent
        logger.warning("Bedrock stream stale for %.0fs (threshold %.0fs) — no events "
            "received. region=%s model=%s. Aborting call.", stale_elapsed, self.stale_timeout, self.region,
            self._model())
        agent._buffer_status(f"⚠️ No events from Bedrock for {int(stale_elapsed)}s (model: {self._model()}). Aborting...")
        _bump_stale_streak(agent)
        # Evict the region's cached client so the NEXT call gets a fresh pool.
        # This does NOT abort the in-flight botocore EventStream (no external
        # cancellation exists); the daemon worker keeps reading until its
        # socket errors, so THIS call ends via the TimeoutError below.
        try:
            from agent.bedrock_adapter import invalidate_runtime_client
            invalidate_runtime_client(self.region)
        except Exception as _inval_exc:
            logger.debug("bedrock: stale client eviction failed: %s", _inval_exc)
        self.last_event = time.time()
        # Raises RuntimeError past HERMES_STREAM_STALE_GIVEUP; otherwise end
        # THIS call with a TimeoutError and let the streak carry forward.
        _check_stale_giveup(agent)
        self.result["error"] = TimeoutError(
            f"Bedrock stream produced no events for {int(stale_elapsed)}s (threshold {int(self.stale_timeout)}s) "
            f"— aborting stalled stream so the retry/fallback path can recover.")

    def _poll(self):
        t = threading.Thread(target=_context_thread_target(self._worker), daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            self._raise_if_interrupted("Agent interrupted during Bedrock API call", worker=t)
            stale_elapsed = time.time() - self.last_event
            if stale_elapsed > self.stale_timeout:
                self._on_stale(stale_elapsed)
                break
        # The Bedrock callback returns a PARTIAL response on interrupt without raising
        # (on_interrupt_check), so the in-loop raise may never fire. Re-check (#59999 area).
        self._raise_if_interrupted("Agent interrupted during Bedrock API call (post-worker)")
        if self.result["error"] is not None:
            raise self.result["error"]
        # Success clears the cross-turn breaker (#58962).
        if self.result["response"] is not None:
            _reset_stale_streak(self.agent)
        return self.result["response"]

    def run(self):
        # Cross-turn stale-stream circuit breaker (#58962), as on the OpenAI/
        # Anthropic path.
        _check_stale_giveup(self.agent)
        return _with_stream_emitters(self.agent, self._poll)


class _ToolCallAccumulator:
    """Assemble streamed tool-call deltas into complete ``tool_calls`` entries
    (``acc``: slot index -> entry dict). Ollama-compatible endpoints reuse index 0
    for every call in a parallel batch, distinguishing them only by id, so a new
    id at an already-seen raw index is redirected to a fresh slot."""

    def __init__(self):
        self.acc: dict = {}
        self._notified: set = set()
        self._last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
        self._active_slot_by_idx: dict = {}  # raw_index -> current slot in acc
        # Argument deltas are collected per slot and joined once in ``materialize`` —
        # ``+=`` per chunk rebuilds the whole string every delta (quadratic on big args).
        self._argument_parts: dict[int, list[str]] = {}

    def materialize(self) -> dict:
        """Join buffered argument deltas into each entry's ``arguments``; idempotent. Returns ``acc``."""
        for idx, parts in self._argument_parts.items():
            self.acc[idx]["function"]["arguments"] = "".join(parts)
        return self.acc

    def feed(self, tc_delta) -> Optional[str]:
        """Merge one delta; return the tool name the first time it is complete."""
        raw_idx = getattr(tc_delta, "index", None)
        if raw_idx is None:
            raw_idx = 0
        tc_id = getattr(tc_delta, "id", None)
        delta_id = tc_id or ""
        if isinstance(tc_id, int):  # Poolside sends integer ids
            tc_id = str(tc_id)

        self._active_slot_by_idx.setdefault(raw_idx, raw_idx)
        if delta_id and raw_idx in self._last_id_at_idx and delta_id != self._last_id_at_idx[raw_idx]:
            self._active_slot_by_idx[raw_idx] = max(self.acc, default=-1) + 1
        if delta_id:
            self._last_id_at_idx[raw_idx] = delta_id
        idx = self._active_slot_by_idx[raw_idx]

        entry = self.acc.setdefault(
            idx, {"id": tc_id or "", "type": "function", "function": {"name": "", "arguments": ""}, "extra_content": None},
        )
        parts = self._argument_parts.setdefault(idx, [])
        if tc_id:
            entry["id"] = tc_id
        tc_function = getattr(tc_delta, "function", None)
        if tc_function:
            if getattr(tc_function, "name", None):
                # Assignment, not +=: names arrive complete and some providers (MiniMax via
                # NVIDIA NIM) resend the full name every chunk — += gives "read_fileread_file".
                entry["function"]["name"] = tc_function.name
            if getattr(tc_function, "arguments", None):
                parts.append(tc_function.arguments)
        extra = getattr(tc_delta, "extra_content", None)
        if extra is None and hasattr(tc_delta, "model_extra"):
            extra = (tc_delta.model_extra if isinstance(tc_delta.model_extra, dict) else {}).get("extra_content")
        if extra is not None:
            entry["extra_content"] = _dump_if_model(extra)
        name = entry["function"]["name"]
        if name and idx not in self._notified:
            self._notified.add(idx)
            return name
        return None


class _StreamingCall:
    """One streaming request on the chat_completions / anthropic_messages wire.
    State shared between the request worker and the poll-loop monitor (heartbeat,
    stale kill, interrupt abort) lives on the instance, mutated from both threads."""

    def __init__(self, agent, api_kwargs: dict, on_first_delta):
        self.agent = agent
        self.api_kwargs = api_kwargs
        self.on_first_delta = on_first_delta
        self.worker = None  # request thread; None in inline mode
        self.result = {"response": None, "error": None, "partial_tool_names": []}
        self.clients = _RequestClientRegistry(agent)
        # Request-local cancel flag: the worker recognizes its own interrupt
        # force-close (RemoteProtocolError) and exits instead of retrying (#6600).
        self._request_cancelled = {"value": False}
        self.first_delta_fired = {"done": False}
        self.deltas_were_sent = {"yes": False}  # for the partial-delivery fallback
        self.provider_tool_in_flight = {"yes": False}
        # Last REAL chunk; the monitor detects SSE-ping-only connections with it.
        self.last_chunk_time = {"t": time.time()}
        # Shared by the socket read timeout (``_stream_timeouts``) and the stale
        # detector (``_resolve_stale_timeout``); None until resolved.
        self._stream_stale_timeout = None
        self.stream_attempt_lock = threading.Lock()
        self.stream_attempt_state = {"current": 0, "cancelled": set(), "discarded_chunks": 0, "discarded_bytes": 0}
        self.managed_stream_holder = {"stream": None}
        # Per-attempt: single-writer token, request-local client, raw HTTP response (chat wire).
        self._writer_token = self._attempt_request_client = self._attempt_stream_response = None

    # ── shared small helpers ────────────────────────────────────────────

    @staticmethod
    def _quiet(fn, *args) -> None:
        """Best-effort callback: never let a display hook break the stream."""
        with contextlib.suppress(Exception):
            fn(*args)

    def _set_managed_stream(self, stream: Any) -> Any:
        self.managed_stream_holder["stream"] = stream
        return stream

    def _close_managed_stream(self) -> None:
        close = getattr(self.managed_stream_holder.pop("stream", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Managed provider stream cleanup failed", exc_info=True)

    def _start_stream_attempt(self) -> int:
        with self.stream_attempt_lock:
            self.stream_attempt_state["current"] += 1
            attempt_id = int(self.stream_attempt_state["current"])
        self.provider_tool_in_flight["yes"] = False
        return attempt_id

    def _cancel_current_stream_attempt(self, reason: str) -> None:
        with self.stream_attempt_lock:
            current = int(self.stream_attempt_state["current"])
            if current:
                self.stream_attempt_state["cancelled"].add(current)
        if current:
            logger.debug("Marked stream attempt %s cancelled: %s", current, reason)

    def _stream_attempt_is_active(self, stream_attempt_id: int) -> bool:
        with self.stream_attempt_lock:
            state = self.stream_attempt_state
            return stream_attempt_id == int(state["current"]) and stream_attempt_id not in state["cancelled"]

    def _stream_attempt_was_cancelled(self, stream_attempt_id: int) -> bool:
        with self.stream_attempt_lock:
            return stream_attempt_id in self.stream_attempt_state["cancelled"]

    def _discard_stale_stream_chunk(self, stream_attempt_id: int, chunk) -> None:
        try:
            chunk_bytes = len(repr(chunk))
        except Exception:
            chunk_bytes = 0
        with self.stream_attempt_lock:
            state = self.stream_attempt_state
            state["discarded_chunks"] += 1
            state["discarded_bytes"] += chunk_bytes
            discarded_chunks, discarded_bytes = state["discarded_chunks"], state["discarded_bytes"]
        first = discarded_chunks == 1
        (logger.warning if first else logger.debug)(
            ("Discarding chunk from superseded stream attempt %s " if first else "Discarded stale stream chunk from attempt %s ")
            + "(discarded_chunks=%s discarded_bytes=%s)",
            stream_attempt_id, discarded_chunks, discarded_bytes,
        )

    def _fire_first_delta(self):
        if not self.first_delta_fired["done"] and self.on_first_delta:
            self.first_delta_fired["done"] = True
            self._quiet(self.on_first_delta)

    def _emit_text(self, text: str) -> None:
        self._fire_first_delta()
        self.agent._fire_stream_delta(text)
        self.deltas_were_sent["yes"] = True

    def _emit_reasoning(self, text: str) -> None:
        self._fire_first_delta()
        self.agent._fire_reasoning_delta(text)

    def _emit_tool_started(self, name: str) -> None:
        self._fire_first_delta()
        self.agent._fire_tool_gen_started(name)

    def _route_suppressed_text(self, text: str) -> None:
        """Tool-call turns suppress content streaming (no chatty preamble), but
        reasoning tags inside it must still reach the display: route through
        the delta callback for tag extraction (the CLI drops non-reasoning text
        once the stream box is closed)."""
        if self.agent.stream_delta_callback:
            self._quiet(lambda: (self.agent.stream_delta_callback(text), self.agent._record_streamed_assistant_text(text)))

    def _new_diag(self) -> dict:
        diag = self.agent._stream_diag_init()
        self.clients.diag = diag
        return diag

    def _count_chunk(self, diag, chunk) -> None:
        """Stamp liveness for a real chunk; diagnostics are best-effort."""
        self.last_chunk_time["t"] = time.time()
        self.agent._touch_activity("receiving stream response")
        with contextlib.suppress(Exception):
            diag["chunks"] = int(diag.get("chunks", 0)) + 1
            if diag.get("first_chunk_at") is None:
                diag["first_chunk_at"] = self.last_chunk_time["t"]
            # Delta-length estimate: ~3x cheaper than repr() per chunk.
            diag["bytes"] = int(diag.get("bytes", 0)) + _estimate_chunk_bytes(chunk)

    # ── chat_completions wire ───────────────────────────────────────────

    def _stream_timeouts(self) -> tuple[float, float, float]:
        """``(write, read, connect/pool)`` socket timeouts. Per-provider
        ``request_timeout_seconds`` wins over HERMES_API_TIMEOUT (1800s) and
        HERMES_STREAM_READ_TIMEOUT (120s); connect/pool cover the handshake, not
        inference: 30s, or capped at 60s when configured."""
        cfg = get_provider_request_timeout(self.agent.provider, self.agent.model)
        base = cfg if cfg is not None else env_float("HERMES_API_TIMEOUT", 1800.0)
        if cfg is not None:
            return base, cfg, min(base, 60.0)
        read = env_float("HERMES_STREAM_READ_TIMEOUT", 120.0)
        stale = self._stream_stale_timeout
        if read == 120.0 and self.agent.base_url and is_local_endpoint(self.agent.base_url):
            read = base  # local providers prefill for minutes
            logger.debug("Local provider detected (%s) — stream read timeout raised to %.0fs", self.agent.base_url, read)
        elif read == 120.0 and stale is not None and stale != float("inf") and stale > read:
            # Reasoning models pause mid-stream for minutes; the stale detector
            # tolerates that, so the raw read timeout must not fire first.
            read = stale
            logger.debug("Cloud reasoning stream — read timeout raised to %.0fs to match stale-stream detector", read)
        return base, read, 30.0

    @staticmethod
    def _choiceless_chunk(chunk, finish_reason):
        """Chunk with empty ``choices`` -> ``(usage, finish_reason)``. Raises
        ProviderStreamError for providers (DeepInfra) that send validation errors
        as in-stream chunks (choices=None + error_type/error_message), which
        would otherwise surface as a misleading EmptyStreamError plus retries."""
        usage = chunk.usage if hasattr(chunk, "usage") and chunk.usage else None  # final usage chunk
        # Without this check the error is silently dropped and the stream ends empty → EmptyStreamError →
        # misleading "empty stream" message and pointless retries on the same bad request. (#65631)
        _err_type = getattr(chunk, "error_type", None)
        _err_msg = getattr(chunk, "error_message", None)
        if _err_type or _err_msg:
            _status = _status_code_from_payload({"code": _err_type, "message": _err_msg}) or _status_code_from_value(_err_type)
            body = _provider_error_body(
                {"code": _err_type or "provider_in_stream_error", "message": str(_err_msg or chunk)}, _status)
            raise ProviderStreamError(status_code=_status, body=body, raw_text=f"{_err_type}: {_err_msg}")
        # Nous Portal usage frames (choices=[] + lastOne=true, no [DONE]) are a
        # clean terminal, not a drop; relabelled upstreams send 1 / "true".
        # See #90848.
        last_one = getattr(chunk, "lastOne", None)
        if last_one is None and isinstance(getattr(chunk, "model_extra", None), dict):
            last_one = chunk.model_extra.get("lastOne")
        if last_one in (True, 1, "true") and finish_reason is None:
            finish_reason = "stop"
        return usage, finish_reason

    def _open_chat_stream(self, stream_kwargs: dict[str, Any]):
        # Native Gemini rejects OpenAI's usage-streaming extension.
        if not is_native_gemini_base_url(self.agent.base_url):
            stream_kwargs["stream_options"] = {"include_usage": True}
        request_client = self._attempt_request_client = self.clients.set_client(
            self.agent._create_request_openai_client(reason="chat_completion_stream_request", api_kwargs=stream_kwargs))
        self.last_chunk_time["t"] = time.time()
        self.agent._touch_activity("waiting for provider response (streaming)")
        return request_client.chat.completions.create(**stream_kwargs)

    def _chat_stream_created(self, raw_stream: Any) -> None:
        response = self._attempt_stream_response = getattr(raw_stream, "response", None)
        self.agent._capture_rate_limits(response)
        self.agent._capture_credits(response)
        self.agent._stream_diag_capture_response(self.clients.diag, response)
        self.agent._check_openrouter_cache_status(response)
        self._writer_token = claim_stream_writer(self.agent)

    def _accept_chat_chunk(self, stream_attempt_id: int, chunk: Any) -> bool:
        with contextlib.suppress(Exception):
            choices = getattr(chunk, "choices", None)
            choice = choices[0] if choices else None
            delta = getattr(choice, "delta", None)
            # A stale-attempt fence can win while Relay hands back a tool-call chunk: record
            # the in-flight tool call (retry policy must not see a partial text response).
            if getattr(delta, "tool_calls", None):
                self.provider_tool_in_flight["yes"] = True
            # Marker-only finish chunk (no writable delta) always passes: the fence only stops
            # MORE text; fending the completion signal would mislabel a clean end as a drop.
            if getattr(choice, "finish_reason", None) and not any(
                getattr(delta, attr, None) for attr in ("content", "tool_calls", "reasoning_content", "reasoning")):
                return True
        if not self._stream_attempt_is_active(stream_attempt_id):
            return False
        if not self._writer_still_current("Streaming"):
            return False
        # Stamp BEFORE Relay processes the chunk so the watchdog can't cancel
        # a live stream mid-interceptor.
        self.last_chunk_time["t"] = time.time()
        return True

    def _writer_still_current(self, label: str) -> bool:
        """Single-writer fence: False (with a warning) once a newer stream claimed the writer slot."""
        token = self._writer_token
        if token is None or stream_writer_is_current(self.agent, token):
            return True
        logger.warning(
            "%s attempt superseded by a newer stream; stopping consumption to preserve the "
            "single-writer invariant (model=%s).", label, self.api_kwargs.get("model", "unknown"))
        return False

    def _call_chat_completions(self, stream_attempt_id: int):
        """Stream a chat completions response."""
        import httpx as _httpx
        base_timeout, read_timeout, conn_cap = self._stream_timeouts()
        content_parts: list = []
        reasoning_parts: list = []
        pending_text_parts: list[str] = []
        tool_calls = _ToolCallAccumulator()
        tool_calls_acc = tool_calls.acc
        finish_reason = model_name = usage_obj = None
        role = "assistant"
        _diag = self._new_diag()
        self._writer_token = self._attempt_request_client = self._attempt_stream_response = None

        def _open_stream(next_api_kwargs: dict[str, Any]):
            timeout = _httpx.Timeout(connect=conn_cap, read=read_timeout, write=base_timeout, pool=conn_cap)
            return self._open_chat_stream({**next_api_kwargs, "stream": True, "timeout": timeout})

        def _relay_final_response() -> dict[str, Any]:
            tool_calls.materialize()
            message = {"role": role, "content": "".join(content_parts) or None,
                "reasoning_content": "".join(reasoning_parts) or None,
                "tool_calls": [tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None}
            return {"model": model_name, "usage": usage_obj,
                "choices": [{"message": message, "finish_reason": finish_reason or "stop"}]}

        def _flush_pending_stream_text():
            pending_parts = list(pending_text_parts)
            pending_text_parts.clear()
            for text in pending_parts:
                (self._route_suppressed_text if tool_calls_acc else self._emit_text)(text)

        from agent import relay_llm
        stream = self._set_managed_stream(relay_llm.stream(self.api_kwargs, _open_stream,
            **_relay_stream_identity(self.agent, "provider"), finalizer=_relay_final_response,
            on_stream_created=self._chat_stream_created,
            accept_chunk=lambda chunk: self._accept_chat_chunk(stream_attempt_id, chunk),
            completed_response_predicate=lambda value: hasattr(value, "choices"),
            metadata=_relay_stream_metadata(self.agent, "chat_completions"), defer_logical_completion=True))
        if self.agent.provider == "moa":
            # Hermes interrupts the managed stream; Relay alone closes the provider stream.
            self.clients.set_stream_handle(stream)

        for chunk in _iter_provider_stream_chunks(stream, response=lambda: self._attempt_stream_response):
            self._count_chunk(_diag, chunk)
            if self.agent._interrupt_requested:
                # A half-read SSE response stays checked out of the httpx pool and the finally
                # would cache the client WITH the leaked connection: close on the owner first.
                try:
                    stream.close()
                except Exception:
                    # Still checked out: poison the slot so the finally really closes the pool.
                    if self._attempt_request_client is not None:
                        self.agent._abort_request_openai_client(
                            self._attempt_request_client, reason="interrupt_stream_close_failed")
                break
            if not self._stream_attempt_is_active(stream_attempt_id):
                self._discard_stale_stream_chunk(stream_attempt_id, chunk)
                continue
            if hasattr(chunk, "model") and chunk.model:
                model_name = chunk.model
            if not chunk.choices:
                usage, finish_reason = self._choiceless_chunk(chunk, finish_reason)
                usage_obj = usage or usage_obj
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            # Read finish_reason/usage BEFORE any content-shape `continue`: the SSE-echo
            # guard can swallow a merged finish chunk (vLLM standalone ':' tokens).
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            if hasattr(chunk, "usage") and chunk.usage:
                usage_obj = chunk.usage

            reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning_text:
                # Summary-part models omit the separator between markdown blocks; re-insert it.
                reasoning_text = separate_glued_reasoning_blocks(
                    reasoning_parts[-1] if reasoning_parts else "", reasoning_text)
                reasoning_parts.append(reasoning_text)
                self._emit_reasoning(reasoning_text)

            # Text (list-of-blocks deltas flattened once); possible echoed SSE is
            # buffered until it can be judged.
            delta_content = flatten_message_text(getattr(delta, "content", None), sep="")
            if delta_content:
                content_parts.append(delta_content)
                if tool_calls_acc:
                    self._route_suppressed_text(delta_content)
                elif pending_text_parts or _provider_stream_text_may_be_sse(delta_content):
                    pending_text_parts.append(delta_content)
                    if not _provider_stream_text_may_be_sse("".join(pending_text_parts)):
                        _flush_pending_stream_text()
                    continue
                else:
                    self._emit_text(delta_content)

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                _flush_pending_stream_text()
                for tc_delta in delta_tool_calls:
                    name = tool_calls.feed(tc_delta)
                    if name is not None:
                        self._emit_tool_started(name)
                        # Lets the stub-builder warn if streaming dies before the args
                        # complete instead of silently discarding the action.
                        self.result["partial_tool_names"].append(name)

        tool_calls.materialize()
        self._close_managed_stream()
        if self._stream_attempt_was_cancelled(stream_attempt_id):
            raise _httpx.RemoteProtocolError(f"stream attempt {stream_attempt_id} was superseded")
        if stream.final_response is not None:
            return self._adopt_final_response(stream.final_response)
        return self._finish_chat_stream(stream, role, content_parts, reasoning_parts, tool_calls_acc,
            finish_reason, model_name, usage_obj, flush_pending=_flush_pending_stream_text)

    def _adopt_final_response(self, final_response):
        """Adapter returned a completed response for ``stream=True``: switch the
        session to non-streaming and replay its content as deltas."""
        logger.info("Streaming request returned a final response object instead of an iterator; "
            "switching %s/%s to non-streaming for this session.", self.agent.provider or "unknown",
            self.agent.model or "unknown")
        self.agent._disable_streaming = True
        choices = final_response.choices
        message = getattr(choices[0] if isinstance(choices, (list, tuple)) and choices else None, "message", None)
        if message is not None:
            reasoning_text = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            if isinstance(reasoning_text, str) and reasoning_text:
                self._emit_reasoning(reasoning_text)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                self._fire_first_delta()
                self.agent._fire_stream_delta(content)  # not _emit_text: deltas_were_sent stays False here
        return final_response

    @staticmethod
    def _assemble_tool_calls(tool_calls_acc, finish_reason):
        """Materialize accumulated tool calls; flag truncated/unrepairable args."""
        mock_tool_calls = []
        has_truncated_tool_args = False
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            arguments = tc["function"]["arguments"]
            if arguments and arguments.strip():
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    # Repair before flagging (GLM via Ollama); "{}" = unrepairable.
                    repaired = _repair_tool_call_arguments(arguments, tc["function"]["name"] or "?")
                    if repaired != "{}":
                        arguments = repaired
                    else:
                        has_truncated_tool_args = True
            elif finish_reason is None:
                # Name arrived, zero arg bytes, no finish_reason: unflagged this
                # becomes a "stop" turn executing "{}" with no retry.
                has_truncated_tool_args = True
            mock_tool_calls.append(SimpleNamespace(
                id=tc["id"], type=tc["type"], extra_content=tc.get("extra_content"),
                function=SimpleNamespace(name=tc["function"]["name"], arguments=arguments)))
        return mock_tool_calls or None, has_truncated_tool_args

    def _finish_chat_stream(self, stream, role, content_parts, reasoning_parts, tool_calls_acc, finish_reason,
        model_name, usage_obj, *, flush_pending):
        """Assemble the non-streaming-shaped response after the chunk loop. A
        stream ending with no finish_reason is a drop, not a completion: return a
        partial-stream stub so the loop fails fast instead of executing empty
        args or stamping "stop"."""
        full_content = "".join(content_parts) or None
        full_reasoning = "".join(reasoning_parts) or None
        mock_tool_calls, has_truncated_tool_args = self._assemble_tool_calls(tool_calls_acc, finish_reason)
        # Zero-chunk guard: nothing usable = upstream error / malformed SSE.
        if finish_reason is None and not content_parts and not reasoning_parts and not tool_calls_acc:
            raise EmptyStreamError(
                "Provider returned an empty stream with no finish_reason (possible upstream error or malformed SSE response).")
        if has_truncated_tool_args and finish_reason is None:
            # Partial args WITH finish_reason="length" is a real output cap; with NONE the
            # upstream dropped mid tool-call, and stamping "length" burns 3 useless retries.
            _dropped_names = [(tool_calls_acc[idx]["function"]["name"] or "?") for idx in sorted(tool_calls_acc)]
            logger.warning(
                "Stream ended with no finish_reason while a tool call's arguments were still incomplete "
                "(tools=%s); treating as a mid-tool-call stream drop, not an output-length truncation.",
                _dropped_names)
            return _build_partial_stream_stub(
                role, full_content, full_reasoning, model_name, usage_obj, dropped_tool_names=_dropped_names or None)
        if finish_reason is None and content_parts and not tool_calls_acc and usage_obj is None:
            # Text-only drop: otherwise the partial text is stamped "stop" and the next step is
            # lost. A usage object proves the provider finished (include_usage's final chunk).
            logger.warning(
                "Stream ended with no finish_reason after delivering text with no tool calls; treating as a mid-stream drop.")
            return _build_partial_stream_stub(role, full_content, full_reasoning, model_name, usage_obj)
        effective_finish_reason = "length" if has_truncated_tool_args else (finish_reason or "stop")
        provider_stream_error = _provider_stream_error_from_text(
            full_content or "", effective_finish_reason, response=getattr(stream, "response", None))
        if provider_stream_error is not None:
            raise provider_stream_error
        flush_pending()
        message = SimpleNamespace(role=role, content=full_content, tool_calls=mock_tool_calls, reasoning_content=full_reasoning)
        return SimpleNamespace(id="stream-" + str(uuid.uuid4()), model=model_name, usage=usage_obj,
            choices=[SimpleNamespace(index=0, message=message, finish_reason=effective_finish_reason)])

    # ── anthropic_messages wire ─────────────────────────────────────────

    @staticmethod
    def _check_anthropic_message(message, *, tool_drop: bool = True):
        """Raise EmptyStreamError for a message the stream never completed: no
        content and no stop_reason (eventless -> retry), or with ``tool_drop`` a
        ``tool_use`` block and no stop_reason — the SSE closed mid tool call and
        its input is a partial snapshot (usually ``{}``), so raising blocks the
        empty-args execution (bounded retry, or stub/continuation after text)."""
        content = getattr(message, "content", None)
        if not content and getattr(message, "stop_reason", None) is None:
            raise EmptyStreamError(
                "Provider returned an empty stream with no stop_reason (possible upstream error or malformed event stream).")
        if tool_drop and getattr(message, "stop_reason", None) is None and any(
            getattr(block, "type", None) == "tool_use" for block in content or []):
            raise EmptyStreamError(
                "Stream ended with no stop_reason while a tool_use block was still incomplete; "
                "treating as a mid-tool-call stream drop (#80498).")
        return message

    def _call_anthropic(self, request_client):
        """Stream an Anthropic Messages API response; fires delta callbacks but
        returns the native Message from get_final_message(). Runs on the
        per-request ``request_client`` so the watchdog can abort this socket
        without closing the shared client mid-flight."""
        has_tool_use = False
        # Eventless stream: the SDK's get_final_message() raises AssertionError (no
        # message_start); shims may fabricate a contentless Message. All -> EmptyStreamError.
        saw_stream_event = False
        self.last_chunk_time["t"] = time.time()
        _diag = self._new_diag()
        self._writer_token = None
        _stream_context = {"manager": None, "stream": None}
        base_final_message = None

        from agent import relay_llm
        from agent.anthropic_adapter import sanitize_anthropic_kwargs
        accumulator = relay_llm.AnthropicStreamAccumulator()

        def _open_anthropic_stream(next_api_kwargs: dict[str, Any]):
            final_kwargs = dict(next_api_kwargs)
            sanitize_anthropic_kwargs(final_kwargs, log_prefix=getattr(self.agent, "log_prefix", ""))
            manager = request_client.messages.stream(**final_kwargs)
            _stream_context["manager"] = manager
            return manager.__enter__()

        def _anthropic_stream_created(raw_stream: Any) -> None:
            _stream_context["stream"] = raw_stream
            # Snapshot response diagnostics now so they survive a stream dying before the first event.
            self._quiet(
                lambda: self.agent._stream_diag_capture_response(_diag, getattr(raw_stream, "response", None)))
            self._writer_token = claim_stream_writer(self.agent)

        stream = self._set_managed_stream(relay_llm.stream(self.api_kwargs, _open_anthropic_stream,
            **_relay_stream_identity(self.agent, "anthropic"), finalizer=accumulator.finalize,
            on_stream_created=_anthropic_stream_created, on_chunk=accumulator.observe,
            accept_chunk=lambda _event: self._writer_still_current("Anthropic streaming"),
            metadata=_relay_stream_metadata(self.agent, "anthropic_messages"), defer_logical_completion=True))
        try:
            for event in stream:
                saw_stream_event = True
                self._count_chunk(_diag, event)
                if self.agent._interrupt_requested:
                    break
                event_type = getattr(event, "type", None)
                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        has_tool_use = True
                        if getattr(block, "name", None):
                            self._emit_tool_started(block.name)
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None) if delta else None
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        if text and not has_tool_use:
                            self._emit_text(text)
                    elif delta_type == "thinking_delta" and getattr(delta, "thinking", ""):
                        self._emit_reasoning(delta.thinking)
            raw_stream = _stream_context["stream"]
            if not self.agent._interrupt_requested and raw_stream is not None:
                try:
                    base_final_message = raw_stream.get_final_message()
                except AssertionError:
                    if not saw_stream_event:
                        raise EmptyStreamError(
                            "Provider returned an empty stream with no events (possible upstream error or malformed event stream).") from None
                    raise
        finally:
            try:
                self._close_managed_stream()
            finally:
                manager = _stream_context["manager"]
                if manager is not None:
                    manager.__exit__(None, None, None)

        if self.agent._interrupt_requested:
            return None
        if base_final_message is not None:
            self._check_anthropic_message(base_final_message, tool_drop=False)
            if not stream.output_modified:
                return self._check_anthropic_message(base_final_message)
        return self._check_anthropic_message(accumulator.response(base_final_message))

    # ── retry loop ──────────────────────────────────────────────────────

    def _retry_after_drop(self, e, attempt: int, max_retries: int, *, mid_tool_call: bool, reason: str) -> None:
        """Warn about the drop and tear down the request-local client. Shared
        clients are never closed from inside a request (FD-recycle hazard); the
        OpenAI primary is replaced lazily."""
        self.agent._emit_stream_drop(
            error=e, attempt=attempt + 2, max_attempts=max_retries + 1, mid_tool_call=mid_tool_call, diag=self.clients.diag)
        self._cancel_current_stream_attempt(reason)
        self.clients.close_once(reason)

    def _maybe_disable_streaming(self, e) -> None:
        """Flip to non-streaming when the provider rejects streaming outright or
        AnthropicBedrock IAM lacks InvokeModelWithResponseStream."""
        _err_lower = str(e).lower()
        _is_stream_unsupported = "stream" in _err_lower and "not supported" in _err_lower
        _is_bedrock_stream_denied = False
        if not _is_stream_unsupported and "invokemodelwithresponsestream" in _err_lower:
            # Message pre-check first: importing bedrock_adapter triggers a lazy boto3 install.
            from agent.bedrock_adapter import is_streaming_access_denied_error
            _is_bedrock_stream_denied = is_streaming_access_denied_error(e)
        if _is_stream_unsupported or _is_bedrock_stream_denied:
            self.agent._disable_streaming = True
            self.agent._safe_print(
                "\n⚠  AWS IAM denied bedrock:InvokeModelWithResponseStream. Switching to non-streaming.\n"
                "   Grant that action to restore streaming output.\n"
                if _is_bedrock_stream_denied else
                "\n⚠  Streaming is not supported for this model/provider. Switching to non-streaming.\n"
                "   To avoid this delay, set display.streaming: false in config.yaml\n"
            )

    def _handle_stream_error(self, e: Exception, attempt: int, max_retries: int) -> bool:
        """Classify a failed attempt: True = retry; False = stop with
        ``result["error"]`` set (unless our own interrupt force-closed the
        socket). Runs inside the ``except`` so ``logger.exception`` works."""
        import httpx as _httpx
        # Our own interrupt force-close: no retry/fallback/"reconnecting" (the
        # poll loop raises InterruptedError).
        if self._request_cancelled["value"]:
            logger.debug("Streaming worker caught %s after request cancellation — exiting without retry.", type(e).__name__)
            return False
        _is_timeout = isinstance(e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout))
        _is_conn_err = isinstance(e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError))
        _is_stream_parse_err = self.agent._is_provider_stream_parse_error(e)
        _is_empty_stream = isinstance(e, EmptyStreamError)
        _is_sse_conn_err = not _is_timeout and not _is_conn_err and _is_sse_connection_error(e)
        _is_transient = _is_timeout or _is_conn_err or _is_sse_conn_err or _is_stream_parse_err

        if self.deltas_were_sent["yes"]:
            # Died AFTER tokens were delivered: normally no retry (would duplicate
            # text). Exception: a tool call in flight — aborting discards it, so
            # retry TRANSIENT errors (a "reconnecting" marker + duplicated
            # preamble beats a failed action; no tool has executed yet).
            _partial_tool_in_flight = bool(self.result.get("partial_tool_names")) or self.provider_tool_in_flight["yes"]
            if not (_partial_tool_in_flight and _is_transient and attempt < max_retries):
                logger.warning("Streaming failed after partial delivery, not retrying: %s", e)
                self.result["error"] = e
                return False
            # Marker explains the re-streamed preamble (``_emit_stream_drop`` logs the WARNING);
            # reset the streamed-text buffer so it isn't double-recorded; fresh accumulators.
            self._quiet(self.agent._fire_stream_delta, "\n\n⚠ Connection dropped mid tool-call; reconnecting…\n\n")
            self._quiet(self.agent._reset_stream_delivery_tracking)
            self.result["partial_tool_names"] = []
            self.deltas_were_sent["yes"] = False
            self.first_delta_fired["done"] = False
            self._retry_after_drop(e, attempt, max_retries, mid_tool_call=True, reason="stream_mid_tool_retry_cleanup")
            return True

        if _is_transient or _is_empty_stream:
            # Transient network / timeout error: retry with a fresh connection first.
            if attempt < max_retries:
                self._retry_after_drop(e, attempt, max_retries, mid_tool_call=False, reason="stream_retry_cleanup")
                return True
            # Exhausted: log full diagnostics (chain, headers, bytes/elapsed).
            self.agent._log_stream_retry(kind="exhausted", error=e, attempt=max_retries + 1,
                max_attempts=max_retries + 1, mid_tool_call=False, diag=self.clients.diag)
            # Empty stream: "connection failed" would send users chasing network issues.
            _what = ("Provider returned malformed streaming data after" if _is_stream_parse_err
                     else "Provider returned an empty response stream after" if _is_empty_stream
                     else "Connection to provider failed after")
            self.agent._buffer_status(
                f"❌ {_what} {max_retries + 1} attempts. The provider may be experiencing issues — try again in a moment.")
        else:
            self._maybe_disable_streaming(e)
            logger.exception("Streaming failed before delivery: %s", e)
        # Propagate to the main retry loop (credential rotation, fallback, backoff).
        self.result["error"] = e
        return False

    def _call_wire(self, stream_attempt_id: int):
        if self.agent.api_mode != "anthropic_messages":
            return self._call_chat_completions(stream_attempt_id)
        # Per-request client so the watchdog aborts its socket, not the shared one.
        request_client = self.clients.set_client(
            self.agent._create_request_anthropic_client(reason="anthropic_stream_request"), kind="anthropic_messages")
        return self._call_anthropic(request_client)

    def _call(self):
        _max_stream_retries = env_int("HERMES_STREAM_RETRIES", 2)
        try:
            for _stream_attempt in range(_max_stream_retries + 1):
                stream_attempt_id = self._start_stream_attempt()
                # Otherwise /stop closes the connection and the retry opens a
                # FRESH one, blocking up to a full read timeout per attempt.
                if self.agent._interrupt_requested:
                    self._cancel_current_stream_attempt("interrupt_before_stream_retry")
                    raise InterruptedError("Agent interrupted before stream retry")
                try:
                    self.result["response"] = _with_stream_emitters(
                        self.agent, lambda: self._call_wire(stream_attempt_id))
                    return  # success
                except Exception as e:
                    self._close_managed_stream()
                    if not self._handle_stream_error(e, _stream_attempt, _max_stream_retries):
                        return
        except InterruptedError as e:
            # Fast pre-retry interrupt surfaces through the normal result channel.
            self.result["error"] = e
            return
        finally:
            self._close_managed_stream()
            # Reuse only after a clean stream; otherwise really close (fresh pool next).
            self.clients.close_once(
                "stream_request_complete" if self.result["response"] is not None else "stream_error_cleanup")

    # ── poll-loop monitor (heartbeat / stale kill / interrupt) ──────────

    def _run_call(self):
        try:
            self._call()
        finally:
            self._call_done.set()

    def _poll_local_load_notice(self, now: float) -> bool:
        """Managed local server: surface a cold model's weight-load progress
        instead of the 30s "provider may be slow" copy. Polled ~1s only while no
        REAL chunk arrived for 2s+ (never during healthy token flow); in-memory,
        no network. True while loading = heartbeat liveness, skip the rest of
        this iteration (the stale detector's local floor dwarfs any load)."""
        m = self._mon
        if now - self.last_chunk_time["t"] < 2.0 or now - m.last_load_poll < 1.0:
            return False
        m.last_load_poll = now
        _load_notice = _managed_local_load_notice(self.agent, self.api_kwargs)
        if _load_notice is not None:
            self.agent._emit_wait_notice(_load_notice)
            self.agent._touch_activity("local model loading")
            m.load_notice_shown, m.load_notice_misses, m.last_heartbeat = True, 0, now  # loading IS liveness
            return True
        if m.load_notice_shown:
            # One missed sample is routine (probe timeout under load); clearing on it strobed the line.
            m.load_notice_misses += 1
            if m.load_notice_misses >= 3:
                m.load_notice_shown, m.load_notice_misses = False, 0
                self.agent._emit_wait_notice("")
        return False

    def _heartbeat(self, waiting_secs: int, interval: float) -> None:
        """Gateway inactivity heartbeat: the start-to-first-chunk gap (thinking,
        local prefill) can exceed the gateway timeout."""
        if waiting_secs >= interval:
            # No chunks for 30s+: say WHAT the wait is and WHEN recovery kicks in.
            stale = self._stream_stale_timeout
            _recovery = f"; auto-reconnect at {int(stale)}s" if stale is not None and stale != float("inf") else ""
            self.agent._emit_wait_notice(
                f"⏳ waiting on {self.api_kwargs.get('model', 'the provider')} — {waiting_secs}s with no output yet "
                f"(provider may be slow or overloaded, or the model is thinking{_recovery})")
        else:
            # Chunks are flowing — keep the tracker fresh, leave the display alone.
            self.agent._touch_activity(f"waiting for stream response ({waiting_secs}s, no chunks yet)")

    def _kill_stale_stream(self, elapsed: float) -> None:
        """SSE pings but no chunks: cancel the attempt and abort the request-local
        client so the retry loop opens a fresh one. The shared client is never
        closed from this (stranger) thread — earlier stale-killed workers may
        still be unwinding SSL BIOs (FD-recycle corruption); the OpenAI primary
        is replaced lazily."""
        _est_ctx = estimate_request_context_tokens(self.api_kwargs)
        logger.warning(
            "Stream stale for %.0fs (threshold %.0fs) — no chunks received. model=%s context=~%s tokens. Killing connection.",
            elapsed, self._stream_stale_timeout, self.api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
        )
        self.agent._buffer_status(
            f"⚠️ No response from provider for {int(elapsed)}s (model: {self.api_kwargs.get('model', 'unknown')}, "
            f"context: ~{_est_ctx:,} tokens). Reconnecting...")
        with contextlib.suppress(Exception):
            self._cancel_current_stream_attempt("stale_stream_kill")
            self.clients.close_once("stale_stream_kill")
        _bump_stale_streak(self.agent)  # circuit breaker, see ``_stale_streak()``
        # Reset the timer so we don't kill repeatedly while the worker unwinds.
        self.last_chunk_time["t"] = time.time()
        self.agent._emit_wait_notice(f"⚠ no output from provider for {int(elapsed)}s — reconnecting...")
        self.agent._touch_activity(f"stale stream detected after {int(elapsed)}s, reconnecting")

    def _abort_for_interrupt(self, stale_elapsed: float) -> None:
        """/stop seen by the monitor: mark cancelled, abort the request-local
        socket, wait for the worker, flag the interrupt."""
        # The stale branch already counted this iteration if its deadline won the race.
        if stale_elapsed <= self._stream_stale_timeout:
            _record_interrupted_provider_wait(self.agent, stale_elapsed, response_started=self.deltas_were_sent["yes"])
        # Mark cancelled BEFORE force-closing so the worker treats the forced
        # transport error as a cancel, not a network error (#6600).
        self._request_cancelled["value"] = True
        logger.debug("Force-closing streaming httpx client due to interrupt (not a network error).")
        with contextlib.suppress(Exception):
            self._cancel_current_stream_attempt("stream_interrupt_abort")
            # Kind-aware: only the request-local socket; the shared _anthropic_client is never closed here.
            self.clients.close_once("stream_interrupt_abort")
        # Let the worker unwind Relay-managed scopes first; raising first lets
        # turn teardown race a still-open scope and corrupt the LIFO stack.
        if self.worker is not None:
            _join_worker_for_relay_teardown(self.worker, label="Streaming")
        self._monitor_interrupted["yes"] = True

    def _monitor_loop(self) -> None:
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        self._mon = SimpleNamespace(last_heartbeat=time.time(), last_load_poll=0.0, load_notice_shown=False, load_notice_misses=0)
        _is_local_base = bool(self.agent.base_url) and is_local_endpoint(self.agent.base_url)
        while not self._call_done.is_set():
            self._call_done.wait(timeout=0.3)
            _hb_now = time.time()
            if _is_local_base and self._poll_local_load_notice(_hb_now):
                continue
            if _hb_now - self._mon.last_heartbeat >= _HEARTBEAT_INTERVAL:
                self._mon.last_heartbeat = _hb_now
                self._heartbeat(int(_hb_now - self.last_chunk_time["t"]), _HEARTBEAT_INTERVAL)
            _stale_elapsed = time.time() - self.last_chunk_time["t"]
            if _stale_elapsed > self._stream_stale_timeout:
                self._kill_stale_stream(_stale_elapsed)
            if self.agent._interrupt_requested:
                self._abort_for_interrupt(_stale_elapsed)
                return

    # ── orchestration ───────────────────────────────────────────────────

    def _resolve_stale_timeout(self) -> None:
        """Set ``_stream_stale_timeout``. Local endpoints (unless the env is set) get
        long but FINITE patience — 900s / ``agent.local_stream_stale_timeout`` /
        HERMES_LOCAL_STREAM_STALE_TIMEOUT — an infinite one stalled sessions on a
        crashed endpoint forever. Cloud values scale with context size and are
        floored for known reasoning models (else BrokenPipeError from the gateway)."""
        base = _configured_stale_base(self.agent)
        if base == 180.0 and self.agent.base_url and is_local_endpoint(self.agent.base_url):
            _local_default = 900.0
            with contextlib.suppress(Exception):
                from hermes_cli.config import load_config_readonly
                _cfg = load_config_readonly()  # read-only consumer — no deepcopy
                _agent_cfg = _cfg.get("agent") if isinstance(_cfg, dict) else None
                _v = _agent_cfg.get("local_stream_stale_timeout") if isinstance(_agent_cfg, dict) else None
                if isinstance(_v, (int, float)):
                    _local_default = float(_v)
            self._stream_stale_timeout = env_float("HERMES_LOCAL_STREAM_STALE_TIMEOUT", _local_default)
            logger.debug("Local provider detected (%s) — stale stream timeout set to %.0fs",
                self.agent.base_url, self._stream_stale_timeout)
            return
        self._stream_stale_timeout = _cloud_stale_timeout(base, self.api_kwargs)

    def _partial_stream_stub(self):
        """Tokens already reached the platform: a finish_reason="length" stub fires the
        continuation machinery; tool_calls=None blocks executing incomplete calls.
        Content may be EMPTY on purpose — the loop skips appending an empty stub and
        only sends the nudge (placeholder text leaked into the stitched response)."""
        error = self.result["error"]
        _partial_text = (getattr(self.agent, "_current_streamed_assistant_text", "") or "").strip() or None
        _partial_names = list(self.result.get("partial_tool_names") or [])
        if _partial_names:
            # User-visible warning so the user and model both know what was attempted.
            _name_str = ", ".join(_partial_names[:3])
            if len(_partial_names) > 3:
                _name_str += f", +{len(_partial_names) - 3} more"
            _warn = (f"\n\n⚠ Stream stalled mid tool-call ({_name_str}); the action was not executed. "
                     f"Ask me to retry if you want to continue.")
            _partial_text = (_partial_text or "") + _warn
            self._quiet(self.agent._fire_stream_delta, _warn)  # visible immediately
            logger.warning(
                "Partial stream dropped tool call(s) %s after %s chars of text; surfaced warning to user: %s",
                _partial_names, len(_partial_text or ""), error)
        else:
            logger.warning(
                "Partial stream delivered before error; returning length-truncated stub with %s chars of "
                "recovered content so the loop can continue from where the stream died: %s",
                len(_partial_text or ""), error)
        # Classify content filtering (MiniMax 1027, Azure content_filter, Anthropic refusal)
        # before the error is swallowed into the stub: the loop reads the tag and falls back.
        _stub = _build_partial_stream_stub("assistant", _partial_text, None,
            getattr(self.agent, "model", "unknown"), None, dropped_tool_names=_partial_names)
        with contextlib.suppress(Exception):
            from agent.error_classifier import classify_api_error
            _cls = classify_api_error(
                error, provider=str(getattr(self.agent, "provider", "") or ""), model=str(getattr(self.agent, "model", "") or ""))
            if _cls.reason == FailoverReason.content_policy_blocked:
                _stub._content_filter_terminated = True
        _reset_stale_streak(self.agent)  # deltas fired => provider responsive: clear the breaker
        return _stub

    def run(self):
        """Resolve the stale timeout, run the request (worker thread or inline),
        drive the heartbeat/stale/interrupt monitor, then translate the outcome."""
        self._resolve_stale_timeout()
        # Delegated children and cron turns run the request INLINE (a worker inside
        # their nested pools wedges before the socket opens) but must still STREAM
        # (edge proxies kill silent POSTs). Only the poll loop moves to a monitor
        # thread, which never issues a request, so the no-worker deadlock fix holds.
        self._call_done = threading.Event()
        self._monitor_interrupted = {"yes": False}
        if should_use_direct_api_call(self.agent):
            self.worker = None
            monitor = threading.Thread(
                target=_context_thread_target(self._monitor_loop), name="stream-inline-monitor", daemon=True)
            monitor.start()
            try:
                self._run_call()
            finally:
                monitor.join(timeout=2.0)
        else:
            self.worker = threading.Thread(target=_context_thread_target(self._run_call), daemon=True)
            self.worker.start()
            self._monitor_loop()
        if self._monitor_interrupted["yes"]:
            raise InterruptedError("Agent interrupted during streaming API call")
        if self.agent._interrupt_requested:  # worker returned early before the monitor saw the flag
            raise InterruptedError("Agent interrupted during streaming API call (post-worker)")
        if self.result["error"] is not None:
            if self.deltas_were_sent["yes"]:
                return self._partial_stream_stub()
            raise self.result["error"]
        if self.result["response"] is not None:
            _reset_stale_streak(self.agent)  # provider proved responsive: clear the breaker
        # Propagate first-chunk timing for the ``post_api_request`` hook.
        if isinstance(self.clients.diag, dict) and self.clients.diag.get("first_chunk_at"):
            self.agent._last_api_first_chunk_at = float(self.clients.diag["first_chunk_at"])
        return self.result["response"]


def interruptible_streaming_api_call(agent, api_kwargs: dict, *, on_first_delta=None):
    """Streaming variant of _interruptible_api_call: fires the delta callbacks per
    text token (tool-call turns suppress them) and returns a SimpleNamespace in
    the non-streaming response shape. codex_responses delegates to the already-
    streaming codex runner; cron turns and delegated children run inline."""
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted before streaming API call")
    if agent.api_mode == "codex_responses":
        return _stream_codex_passthrough(agent, api_kwargs, on_first_delta)
    if agent.api_mode == "bedrock_converse":
        return _BedrockStream(agent, api_kwargs, on_first_delta).run()
    # Cross-turn stale-stream circuit breaker (see ``_stale_streak()``).
    _check_stale_giveup(agent)
    return _StreamingCall(agent, api_kwargs, on_first_delta).run()


__all__ = ["interruptible_api_call", "build_api_kwargs", "build_assistant_message", "try_activate_fallback",
    "handle_max_iterations", "cleanup_task_resources", "interruptible_streaming_api_call"]
