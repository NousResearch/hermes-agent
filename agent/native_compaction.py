"""Native OpenAI Responses server-side compaction for direct OpenAI routes.

``context_management=[{"type": "compaction", "compact_threshold": N}]`` makes the server
summarize older context into an opaque ``compaction`` item once the input crosses N tokens.
Automatic ``context_management`` remains limited to the gpt-5.6 family. Direct API-key
GPT-6 Astra instead uses an explicit final ``compaction_trigger`` maintenance item,
persisting the returned canonical window before replay. The local compressor stays armed
as fallback. Transport imports are lazy to keep the shared gate cycle-free.
"""

from __future__ import annotations

import logging
import copy
import hashlib
import json
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from agent.context_compressor import is_compaction_summary_message
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# Native compaction fires this far below the local trigger so the server gets the first shot.
LOCAL_TRIGGER_SAFETY_MARGIN = 8_192
# Fallback when automatic mode has no local trigger to follow.
DEFAULT_COMPACT_THRESHOLD = 200_000
# Substring match so dated snapshots and variants (gpt-5.6-mini) stay eligible.
_ELIGIBLE_MODEL_MARKER = "gpt-5.6"

# Durable sidecar key for the explicit Astra path.  The transcript itself remains
# complete; this is only the provider's canonical wire window and its exact
# boundary, so a restart can replay it without re-running compaction.
ASTRA_COMPACTION_METADATA_KEY = "astra_native_compaction"
ASTRA_COMPACTION_VERSION = 1


def is_native_compaction_model(model: Optional[str]) -> bool:
    """True when the model is in the gpt-5.6 family."""
    return _ELIGIBLE_MODEL_MARKER in (model or "").lower()


def _is_astra_model(model: Any) -> bool:
    """Match only the explicit Astra model, including provider-qualified names."""
    return str(model or "").strip().lower().rsplit("/", 1)[-1] == "gpt-6-astra"


def resolve_native_compaction_capabilities(
    *, model: Optional[str], base_url: Optional[str], provider: Optional[str] = None, is_codex_backend: bool = False,
) -> Dict[str, bool]:
    """Resolve the native-compaction capability for a runtime destination (a resolved ``False``
    is distinct from "unresolved" and must survive model switches unchanged)."""
    direct_default = (provider or "").strip().lower() == "openai" and not base_url
    return {"native_compaction": is_native_compaction_model(model) and (
        direct_default or is_direct_openai_route(base_url, is_codex_backend=is_codex_backend))}


def is_direct_openai_route(base_url: Optional[str], *, is_codex_backend: bool = False) -> bool:
    """True for api.openai.com or the ChatGPT Codex backend — nothing else."""
    if is_codex_backend:
        return True
    try:
        hostname = (urlsplit(base_url or "").hostname or "").lower()
    except ValueError:
        return False
    return hostname == "api.openai.com"


def _positive_int(value: Any, *, reject: tuple = (bool,)) -> Optional[int]:
    """``int(value)`` when it is a positive integer-like (never a bool), else None."""
    if value is None or isinstance(value, reject):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_compact_threshold(configured_threshold: Any, local_trigger_tokens: Any = None) -> int:
    """Resolve automatic mode or clamp an explicit native threshold.

    Omitted/invalid follows the local compressor trigger minus the safety margin. An
    explicit positive integer is absolute unless it must be clamped so native compaction
    fires first. Booleans are never thresholds.
    """
    local = _positive_int(local_trigger_tokens)
    upper = None if local is None else max(
        1_024, local - LOCAL_TRIGGER_SAFETY_MARGIN if local > LOCAL_TRIGGER_SAFETY_MARGIN else int(local * 0.8))
    configured = _positive_int(configured_threshold, reject=(bool, float))
    if configured is None:
        return upper if upper is not None else DEFAULT_COMPACT_THRESHOLD
    if upper is None:
        return configured
    return max(1_024, min(configured, upper))


_checkpoint_suppression_logged = False


def _warn_native_compaction_suppressed_by_checkpoint_gate() -> None:
    """Log once per process; the suppression itself is re-evaluated per request."""
    global _checkpoint_suppression_logged
    if not _checkpoint_suppression_logged:
        _checkpoint_suppression_logged = True
        logger.warning(
            "compression.checkpoint_required is enabled: server-side native "
            "compaction (context_management) is disabled for this agent so the "
            "checkpoint-aware Hermes compressor stays authoritative."
        )


def native_compaction_context_management(agent: Any, *, is_codex_backend: bool, is_xai_responses: bool = False,
                                         is_github_responses: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Return the ``context_management`` payload for this request, or None ("do not send").

    Every gate is re-checked per request so a mid-session model switch or the in-session
    kill switch (``agent.codex_responses_native_compaction = False``) takes effect next call.
    """
    capabilities = getattr(agent, "runtime_capabilities", None)
    if isinstance(capabilities, dict) and not capabilities.get("native_compaction", False):
        return None
    # compression.enabled: false disables ALL automatic compaction, native included.
    if not getattr(agent, "codex_responses_native_compaction", False) or not getattr(agent, "compression_enabled", True):
        return None
    # Server-side compaction is a lossy boundary the provider owns (no pre-compress checkpoint
    # can run first), so the checkpoint-aware compressor stays authoritative. Explicit-True
    # matches compress_context().
    if getattr(agent, "compression_checkpoint_required", False) is True:
        _warn_native_compaction_suppressed_by_checkpoint_gate()
        return None
    if is_xai_responses or is_github_responses or not is_native_compaction_model(getattr(agent, "model", None)):
        return None
    # Astra configuration updates invalidate the automatic context-management
    # contract.  Its explicit maintenance request is scheduled at a completed
    # turn boundary instead; never put both mechanisms on one request.
    if (
        _is_astra_model(getattr(agent, "model", None))
        and is_direct_openai_route(getattr(agent, "base_url", None))
        and not is_codex_backend
    ):
        return None
    trusted_proxy = bool(getattr(agent, "capabilities", {}).get("openai_native_compaction", False))
    if not trusted_proxy and not is_direct_openai_route(getattr(agent, "base_url", None), is_codex_backend=is_codex_backend):
        return None

    compressor = getattr(agent, "context_compressor", None)
    local_trigger = getattr(compressor, "threshold_tokens", None) if compressor is not None else None
    threshold = resolve_compact_threshold(getattr(agent, "codex_responses_compact_threshold", None), local_trigger)
    return [{"type": "compaction", "compact_threshold": threshold}]


def _is_direct_astra_agent(agent: Any, *, is_codex_backend: bool = False) -> bool:
    """Exact direct API-key Astra route (no OAuth, proxy, relay, or delegation)."""
    from agent.transports.codex import is_astra_reasoning_cache_eligible

    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    if provider not in {"", "openai", "openai-api"}:
        return False
    return is_astra_reasoning_cache_eligible(
        getattr(agent, "model", None), getattr(agent, "base_url", None),
        api_mode=getattr(agent, "api_mode", None), api_key=getattr(agent, "api_key", None),
        auth_mode=getattr(agent, "auth_mode", "api_key"), provider=getattr(agent, "provider", None),
        is_subagent=getattr(agent, "is_subagent", False), platform=getattr(agent, "platform", None),
        delegate_depth=getattr(agent, "_delegate_depth", 0),
        compression_checkpoint_required=getattr(agent, "compression_checkpoint_required", False),
    ) and not is_codex_backend


def is_astra_native_compaction_eligible(agent: Any) -> bool:
    """Whether the explicit maintenance path may issue a request for *agent*."""
    return bool(
        _is_direct_astra_agent(agent)
        and getattr(agent, "codex_responses_native_compaction", False)
        and getattr(agent, "compression_enabled", True)
        and not getattr(agent, "_astra_native_compaction_disabled", False)
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_value(value: Any) -> Any:
    """Convert an SDK response item without dropping opaque provider fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_value(dump(mode="json", warnings=False))
        except TypeError:
            return _json_value(dump())
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    values = getattr(value, "__dict__", None)
    return _json_value(values) if isinstance(values, dict) else None


def astra_compaction_content_digest(messages: Any) -> Optional[str]:
    """Hash the covered chat prefix while ignoring local-only sidecar fields.

    Steering is intentionally injected into an existing tool result. A persisted
    canonical window must therefore be bypassed if that covered result changed after
    maintenance, including after a restart. Returning ``None`` fails open to the
    ordinary full-transcript serializer when a fixture contains a non-JSON value.
    """
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        return None
    comparable = []
    for message in messages:
        content = message.get("api_content")
        if isinstance(content, str) and content:
            message = {**message, "content": content}
        comparable.append({
            key: _json_value(value)
            for key, value in message.items()
            if key not in {"_row_id", "api_content", "display_kind", "display_metadata"}
        })
    try:
        encoded = json.dumps(comparable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_astra_window(window: Any) -> bool:
    """Validate the minimum canonical output needed for durable replay."""
    if not isinstance(window, list) or not window or not all(isinstance(item, dict) for item in window):
        return False
    checkpoints = [item for item in window if item.get("type") == "compaction"]
    return bool(checkpoints) and all(
        isinstance(item.get("encrypted_content"), str) and bool(item["encrypted_content"].strip())
        for item in checkpoints
    )


@dataclass(frozen=True)
class AstraCompactionResult:
    """Provider canonical window staged before durable activation."""

    window: List[Dict[str, Any]]
    covered_boundary: Dict[str, Any]
    route: str = "direct_openai"


def astra_compaction_boundary(messages: Any) -> Dict[str, Any]:
    """Return a stable, non-content boundary marker for the completed prefix."""
    prefix = messages if isinstance(messages, list) else []
    last = prefix[-1] if prefix else {}
    row_id = last.get("_row_id") if isinstance(last, dict) else None
    return {
        "message_count": len(prefix),
        "last_row_id": row_id if isinstance(row_id, int) and not isinstance(row_id, bool) else None,
        "last_role": last.get("role") if isinstance(last, dict) else None,
    }


def astra_compaction_prefix_with_row_ids(agent: Any, messages: Any) -> List[Dict[str, Any]]:
    """Attach durable row ids to a completed prefix without content-based fallback.

    Normal turn history intentionally omits row ids from its wire-facing copy.  Resolve
    them from the same ordered durable transcript, and reject any positional mismatch so
    duplicate user text or a concurrent append cannot move a checkpoint carrier.
    """
    prefix = messages if isinstance(messages, list) else []
    if not prefix or not all(isinstance(item, dict) for item in prefix):
        return []
    if all(isinstance(item.get("_row_id"), int) and not isinstance(item.get("_row_id"), bool) for item in prefix):
        return [dict(item) for item in prefix]
    db = getattr(agent, "_session_db", None) or getattr(agent, "session_db", None)
    loader = getattr(db, "get_messages_as_conversation", None) if db is not None else None
    if not callable(loader):
        return []
    try:
        durable = loader(getattr(agent, "session_id", None), include_row_ids=True)
    except Exception:
        logger.warning("Astra compaction durable boundary lookup failed", exc_info=True)
        return []
    if not isinstance(durable, list) or len(durable) < len(prefix):
        return []
    resolved: List[Dict[str, Any]] = []
    for source, stored in zip(prefix, durable):
        row_id = stored.get("_row_id") if isinstance(stored, dict) else None
        if (
            not isinstance(stored, dict)
            or not isinstance(row_id, int)
            or isinstance(row_id, bool)
            or source.get("role") != stored.get("role")
            or source.get("content") != stored.get("content")
        ):
            return []
        resolved.append({**source, "_row_id": row_id})
    return resolved


def _maintenance_kwargs(agent: Any, messages: List[Dict[str, Any]], system_message: str, tools: Any) -> dict:
    """Build the explicit trigger request using the normal Responses serializer."""
    from agent.transports.codex import ResponsesApiTransport
    transport = ResponsesApiTransport()
    reasoning_config = getattr(agent, "reasoning_config", None)
    if not isinstance(reasoning_config, dict):
        reasoning_config = {"effort": getattr(agent, "_astra_effective_effort", None)
                            or getattr(agent, "_astra_base_effort", None) or "low"}
    kwargs = transport.build_kwargs(
        model=getattr(agent, "model", "gpt-6-astra"), messages=messages, tools=tools,
        reasoning_config=reasoning_config,
        instructions=system_message or "", session_id=getattr(agent, "session_id", None),
        cache_scope_id=getattr(agent, "_prompt_cache_scope_id", None),
        base_url=getattr(agent, "base_url", None), api_key=getattr(agent, "api_key", None),
        auth_mode=getattr(agent, "auth_mode", "api_key"), provider=getattr(agent, "provider", None),
        api_mode=getattr(agent, "api_mode", "codex_responses"),
        is_subagent=getattr(agent, "is_subagent", False), platform=getattr(agent, "platform", None),
        delegate_depth=getattr(agent, "_delegate_depth", 0),
        compression_checkpoint_required=False, astra_state=getattr(agent, "_astra_reasoning_state", {}),
        replay_encrypted_reasoning=bool(getattr(agent, "_codex_reasoning_replay_enabled", True)),
        context_management=None, astra_configuration_updates=True,
    )
    kwargs["input"] = list(kwargs.get("input") or []) + [{"type": "compaction_trigger"}]
    # Tool schemas stay available to the model's context, but the invisible
    # maintenance response can never dispatch a tool call.
    if kwargs.get("tools"):
        kwargs["tool_choice"] = "none"
        kwargs["parallel_tool_calls"] = False
    kwargs.pop("context_management", None)
    return transport.preflight_kwargs(kwargs, allow_stream=False)


def _record_astra_maintenance_usage(agent: Any, response: Any, messages: List[Dict[str, Any]], duration: float) -> None:
    """Fold invisible maintenance usage once through the normal accounting path."""
    if not hasattr(agent, "session_api_calls") or not getattr(response, "usage", None):
        return
    try:
        from agent.turn_usage import record_response_usage
        record_response_usage(
            agent, response, messages=messages, api_call_count=max(1, int(getattr(agent, "_api_call_count", 0) or 0)),
            api_duration=duration, compression_attempts=0,
            max_compression_attempts=int(getattr(agent, "max_compression_attempts", 3) or 3),
        )
    except Exception:
        logger.debug("Astra maintenance usage accounting failed", exc_info=True)


def request_astra_compaction(
    agent: Any, messages: List[Dict[str, Any]], *, system_message: str = "", tools: Any = None,
    commit_fence: Any = None,
) -> Optional[AstraCompactionResult]:
    """Issue one direct HTTP maintenance request and stage its canonical output.

    This function never mutates transcript or executes output tools.  Callers must
    persist the returned result and activate it only after that write succeeds.
    """
    if not is_astra_native_compaction_eligible(agent) or not isinstance(messages, list) or not messages:
        return None
    if commit_fence is not None and commit_fence.is_cancelled:
        return None
    request = _maintenance_kwargs(agent, messages, system_message, tools)
    client = None
    started = time.monotonic()
    try:
        create = getattr(agent, "_create_request_openai_client", None)
        client = create(reason="astra_compaction_maintenance", api_kwargs=request) if callable(create) \
            else agent._ensure_primary_openai_client(reason="astra_compaction_maintenance")
        from agent.codex_runtime import _bypass_sdk_request_transform
        response = client.responses.create(**_bypass_sdk_request_transform(request))
        status = str(_field(response, "status", "") or "").strip().lower()
        raw_output = _field(response, "output")
        output = _json_value(raw_output)
        if status and status != "completed":
            raise RuntimeError(f"Astra compaction response incomplete: {status}")
        if not _valid_astra_window(output):
            raise RuntimeError("Astra compaction response did not return a compaction checkpoint")
        _record_astra_maintenance_usage(agent, response, messages, time.monotonic() - started)
        if commit_fence is not None and commit_fence.is_cancelled:
            return None
        return AstraCompactionResult(
            window=copy.deepcopy(output), covered_boundary=astra_compaction_boundary(messages),
        )
    except Exception as exc:
        status_code = _field(exc, "status_code")
        if is_native_compaction_rejection(str(exc), status_code=status_code):
            agent._astra_native_compaction_disabled = True
        logger.warning("Astra explicit compaction maintenance failed (%s)", type(exc).__name__)
        return None
    finally:
        close = getattr(agent, "_close_request_openai_client", None)
        if client is not None and callable(close):
            with suppress(Exception):
                close(client, reason="request_complete")


def persist_astra_compaction_result(
    agent: Any, messages: List[Dict[str, Any]], result: AstraCompactionResult, *, commit_fence: Any = None,
) -> bool:
    """Persist then activate the canonical window behind the compression fence."""
    if (
        not isinstance(result, AstraCompactionResult)
        or result.route != "direct_openai"
        or not _valid_astra_window(result.window)
        or not isinstance(result.covered_boundary, dict)
        or not isinstance(result.covered_boundary.get("message_count"), int)
        or isinstance(result.covered_boundary.get("message_count"), bool)
        or result.covered_boundary["message_count"] < 0
        or result.covered_boundary != astra_compaction_boundary(messages)
    ):
        return False
    db = getattr(agent, "_session_db", None)
    if db is None:
        db = getattr(agent, "session_db", None)
    if db is None:
        return False
    prefix = messages if isinstance(messages, list) else []
    carrier = next((msg for msg in reversed(prefix) if isinstance(msg, dict) and msg.get("role") == "assistant"), None)
    row_id = carrier.get("_row_id") if carrier else None
    if not isinstance(row_id, int) or isinstance(row_id, bool):
        return False
    metadata = {
        "version": ASTRA_COMPACTION_VERSION, "window": copy.deepcopy(result.window),
        "covered_boundary": dict(result.covered_boundary), "route": result.route,
    }
    covered_digest = astra_compaction_content_digest(prefix)
    if not covered_digest:
        return False
    metadata["covered_boundary"]["covered_digest"] = covered_digest
    merge = getattr(db, "merge_message_display_metadata", None)
    if not callable(merge):
        return False
    admitted = False
    if commit_fence is not None:
        admitted = bool(commit_fence.begin_commit(getattr(agent, "_hard_interrupt_requested", None)))
        if not admitted:
            return False
    try:
        try:
            merged = merge(getattr(agent, "session_id", None), row_id, {ASTRA_COMPACTION_METADATA_KEY: metadata})
        except Exception:
            logger.warning("Astra explicit compaction checkpoint persistence failed", exc_info=True)
            return False
        if merged != 1:
            return False
        state = dict(metadata)
        state["row_id"] = row_id
        agent._astra_native_compaction = state
        agent._astra_native_compaction_boundary = dict(result.covered_boundary)
        return True
    finally:
        if admitted:
            commit_fence.finish_commit()


def restore_astra_compaction_state(agent: Any, messages: Any) -> Optional[dict]:
    """Restore a validated canonical window from durable message metadata."""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        metadata = message.get("display_metadata") if isinstance(message, dict) else None
        state = metadata.get(ASTRA_COMPACTION_METADATA_KEY) if isinstance(metadata, dict) else None
        if (
            not isinstance(state, dict)
            or state.get("version") != ASTRA_COMPACTION_VERSION
            or state.get("route") != "direct_openai"
            or not _valid_astra_window(state.get("window"))
        ):
            continue
        boundary = state.get("covered_boundary")
        if (
            not isinstance(boundary, dict)
            or not isinstance(boundary.get("message_count"), int)
            or isinstance(boundary.get("message_count"), bool)
            or boundary["message_count"] < 0
            or boundary.get("last_role") != "assistant"
            or (
                boundary.get("last_row_id") is not None
                and (
                    not isinstance(boundary.get("last_row_id"), int)
                    or isinstance(boundary.get("last_row_id"), bool)
                )
            )
        ):
            continue
        restored = copy.deepcopy(state)
        agent._astra_native_compaction = restored
        return restored
    return None


def refresh_astra_compaction_boundary(agent: Any, messages: Any) -> Optional[dict]:
    """Resolve the durable row boundary to this in-memory transcript's count."""
    state = getattr(agent, "_astra_native_compaction", None)
    if not isinstance(state, dict):
        state = restore_astra_compaction_state(agent, messages)
    if not isinstance(state, dict):
        return None
    boundary = state.get("covered_boundary") or {}
    row_id = boundary.get("last_row_id")
    if isinstance(row_id, int):
        for index, message in enumerate(messages or []):
            if isinstance(message, dict) and message.get("_row_id") == row_id:
                boundary = dict(boundary, message_count=index + 1)
                state["covered_boundary"] = boundary
                return state
    return state


# Retention budgets for plaintext user messages / local summaries carried across a native
# compaction boundary (mirrors Codex CLI's RETAINED_MESSAGE_TOKEN_BUDGET).
RETAINED_USER_MESSAGE_TOKEN_BUDGET = 64_000
RETAINED_SUMMARY_TOKEN_BUDGET = 32_000


def _approx_tokens(text: str) -> int:
    """Cheap chars//4 token estimate — same shape Codex uses for retention."""
    return max(1, len(text) // 4)


def _extract_item_text(item: Any) -> Optional[str]:
    """Measurable text from a Responses item (string/multipart/metadata), or None."""
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if content is None and "output_text" in item:
        content = item.get("output_text")
    if isinstance(content, str):
        return content if content.strip() else None
    if not isinstance(content, list):
        return None
    parts = []
    for part in content:
        candidates: tuple = (part,)  # non-str, non-dict parts filter out below
        if isinstance(part, dict):
            part_meta = part.get("metadata")
            candidates = (part.get("text") or part.get("input_text") or part.get("output_text"),
                          part_meta.get("text") if isinstance(part_meta, dict) else None)
        parts.extend(c.strip() for c in candidates if isinstance(c, str) and c.strip())
    text = " ".join(parts)
    return text if text.strip() else None


def _has_retainable_image_content(item: Any) -> bool:
    """True for a converted Responses message with a valid ``input_image`` part (only the
    adapter-owned shape counts, so empty multipart placeholders never become durable history)."""
    content = item.get("content") if isinstance(item, dict) else None
    return isinstance(content, list) and any(
        isinstance(part, dict) and str(part.get("type") or "").strip().lower() == "input_image"
        and isinstance(part.get("image_url"), str) and part["image_url"].strip() for part in content
    )


# Canonical provenance check. Deliberately NOT a second heuristic (no underscore-key scan,
# no ad-hoc headings) — either could promote adversarial content to durable history.
_is_summary_item = is_compaction_summary_message


def _is_compaction_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "compaction"


def prune_pre_checkpoint_items(
    items: List[Dict[str, Any]],
    retained_user_token_budget: int = RETAINED_USER_MESSAGE_TOKEN_BUDGET,
    retained_summary_token_budget: int = RETAINED_SUMMARY_TOKEN_BUDGET,
    enable_summary_retention: bool = True, item_sources: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Restructure Responses input around the newest compaction checkpoint.

    The server drops every input item preceding a replayed ``compaction`` item, erasing the
    user's plaintext asks and any local-compression summary. Rebuild as::

        [checkpoint run] + [retained user & summary messages (newest-first budget)] + [post]

    - The NEWEST contiguous run of checkpoints wins; relative order is preserved.
    - User messages are kept verbatim within ``retained_user_token_budget``; the boundary
      message is head-truncated when it only partially fits (string content only). A
      recognized image-only user message is retained whole at one-token cost.
    - Summaries are retained whole within ``retained_summary_token_budget``, never sliced
      (framing would corrupt) and never duplicated.
    - ``item_sources`` (parallel to ``items``) is the raw chat message each item came from.
      Conversion can be lossy for summaries (merge-into-tail carrier → typed
      ``function_call_output``; assistant carrier shadowed by a stale replay), so a source
      that is itself a canonical summary carrier is read from the SOURCE and retained as a
      synthesized ``role="assistant"`` message.
    - ``enable_summary_retention`` is a test override, not a config surface.

    The server drops every input item that precedes a replayed ``compaction`` item (live-verified Aug 2026),
    so sending pre-checkpoint history is dead weight AND silently erases the user's plaintext asks —
    including any local-compression summary the agent already produced, which previously vanished here
    because it carries ``role="assistant"``, not ``"user"`` (#90975).
    A summary is never byte/character-sliced: Hermes summaries carry structural framing (handoff prefix, end
    marker, merge-into-tail delimiters) that a blind slice can corrupt, so one that doesn't fit whole is
    dropped instead. A summary already retained once (identical text) is never duplicated, so repeated
    checkpoints stay idempotent. - ``enable_summary_retention`` is a function-level override (used by tests
    and callers that need the pre-#90975 behavior back); it is not wired to a user-facing config surface.
    Without ``item_sources`` (default), retention only sees what survived conversion, matching pre-#90976
    behavior (#90976).
    """
    if not isinstance(items, list) or not items:
        return items
    last_cp = max((i for i, item in enumerate(items) if _is_compaction_item(item)), default=None)
    if last_cp is None:
        return items
    first_cp = last_cp
    while first_cp > 0 and _is_compaction_item(items[first_cp - 1]):
        first_cp -= 1

    pre = items[:first_cp]
    has_sources = isinstance(item_sources, list) and len(item_sources) == len(items)
    pre_sources: List[Any] = item_sources[:first_cp] if has_sources else [None] * len(pre)

    retained_reversed: List[Dict[str, Any]] = []
    user_remaining = max(0, int(retained_user_token_budget))
    summary_remaining = max(0, int(retained_summary_token_budget))
    seen_summary_texts: set = set()

    def _retain_summary(text: Optional[str], retained_item: Dict[str, Any]) -> None:
        """Retain a summary whole when it fits the budget and is not a duplicate (never sliced)."""
        nonlocal summary_remaining
        if not text or summary_remaining <= 0 or text in seen_summary_texts:
            return
        cost = _approx_tokens(text)
        if cost <= summary_remaining:
            seen_summary_texts.add(text)
            retained_reversed.append(retained_item)
            summary_remaining -= cost

    for item, source in zip(reversed(pre), reversed(pre_sources)):
        if not isinstance(item, dict):
            continue
        # Source-based detection sees past a lossy conversion; it only fires
        # when the source itself is a provenance-tagged summary carrier.
        # Canonical source-based summary detection: reads the ORIGINAL chat message's own content, so it
        # sees past a lossy conversion (a typed `function_call_output` wrapper, or a stale exact-replay
        # message) that erased the summary from `item` itself (#90976).
        if enable_summary_retention and isinstance(source, dict) and _is_summary_item(source):
            text = flatten_message_text(source.get("content"))
            _src_role = source.get("role")
            _retain_summary(text if text.strip() else None,
                            {"role": _src_role if _src_role in ("user", "assistant") else "assistant", "content": text})
            continue
        # Typed non-message items never carry role=user or a summary flag.
        if "type" in item and item.get("type") != "message":
            continue
        is_summary = enable_summary_retention and _is_summary_item(item)
        is_user = item.get("role") == "user"
        if not is_user and not is_summary:
            continue
        text = _extract_item_text(item)
        if text is None:
            if not (is_user and _has_retainable_image_content(item)):
                continue
            text = ""
        if is_summary:
            _retain_summary(text, item)
        elif user_remaining > 0:
            cost = _approx_tokens(text)
            if cost <= user_remaining:
                retained_reversed.append(item)
                user_remaining -= cost
            elif isinstance(item.get("content"), str):
                truncated = {**item, "content": item["content"][: user_remaining * 4]}
                if truncated["content"].strip():
                    retained_reversed.append(truncated)
                user_remaining = 0

    result = items[first_cp : last_cp + 1] + list(reversed(retained_reversed)) + items[last_cp + 1 :]
    logger.debug("Pruned pre-checkpoint items: %d input -> %d retained (user_rem=%d, summary_rem=%d)",
                 len(items), len(result), user_remaining, summary_remaining)
    return result


_REJECTION_MARKERS = (
    "unknown", "unsupported", "invalid", "unexpected", "not permitted",
    "not allowed", "unrecognized", "extra field", "no such", "bad request",
    "not supported",
)


def is_native_compaction_rejection(error: Any, status_code: Any = None) -> bool:
    """True when a provider error is a STRUCTURED rejection of ``context_management``.

    Drives one-shot recovery (strip, disable for the session, retry), so matching is narrow:
    a transient 5xx that merely ECHOES the request must not downgrade native compaction.
    Requires ``status_code`` 400 (or unknown) AND the field name with rejection language.

    See #82777.
    * ``status_code`` is 400 (or unknown/None — some transports surface only a message string; field-name
    matching alone is then the best available signal, preserving pre-#82777 behavior for them), and * the
    error text names ``context_management`` / ``compact_threshold`` alongside rejection language ("unknown",
    "unsupported", "invalid", "unexpected", "not permitted"...). A bare field-name echo without rejection
    language does not match.
    """
    text = str(error or "").lower()
    if (
        "context_management" not in text
        and "compact_threshold" not in text
        and "compaction_trigger" not in text
    ):
        return False
    try:
        if status_code is not None and int(status_code) != 400:
            return False
    except (TypeError, ValueError):
        pass
    return any(marker in text for marker in _REJECTION_MARKERS)


def has_compaction_checkpoint(items: Any) -> bool:
    """Does this ``codex_reasoning_items`` sidecar carry a compaction checkpoint? A checkpoint is
    cumulative context living in exactly one place: rewrite/discard the sidecar only after asking."""
    return isinstance(items, list) and any(_is_compaction_item(item) for item in items)


def merge_interim_reasoning_items(prior_items: Any, new_items: Any) -> List[Dict[str, Any]]:
    """Merge ``codex_reasoning_items`` across Codex incomplete-continuation dedup.

    A checkpoint on the EARLIER response is not re-emitted by the continuation, so a blind
    overwrite drops the only copy: newer items win, prior checkpoints are prepended unless
    the newer payload has its own.
    """
    prior = prior_items if isinstance(prior_items, list) else []
    kept_checkpoints = [item for item in prior if _is_compaction_item(item)]
    new_list = list(new_items) if isinstance(new_items, list) else []
    if has_compaction_checkpoint(new_list) or not kept_checkpoints:
        return new_list
    return kept_checkpoints + new_list
