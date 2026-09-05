"""Native OpenAI Responses server-side compaction — gpt-5.6 on direct OpenAI routes only.

``context_management=[{"type": "compaction", "compact_threshold": N}]`` makes the server
summarize older context into an opaque ``compaction`` item once the input crosses N tokens.
Deliberately narrow (live-verified): gpt-5.6 only (5.1/5.2 fail server-side with no
structured rejection) on api.openai.com or the ChatGPT Codex backend. The local compressor
stays armed as fallback (native threshold clamped below the local trigger); compaction items
ride the ``codex_reasoning_items`` sidecar. No transport imports (shared gate, no cycles).
"""

from __future__ import annotations

import logging
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


def is_native_compaction_model(model: Optional[str]) -> bool:
    """True when the model is in the gpt-5.6 family."""
    return _ELIGIBLE_MODEL_MARKER in (model or "").lower()


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
    trusted_proxy = bool(getattr(agent, "capabilities", {}).get("openai_native_compaction", False))
    if not trusted_proxy and not is_direct_openai_route(getattr(agent, "base_url", None), is_codex_backend=is_codex_backend):
        return None

    compressor = getattr(agent, "context_compressor", None)
    local_trigger = getattr(compressor, "threshold_tokens", None) if compressor is not None else None
    threshold = resolve_compact_threshold(getattr(agent, "codex_responses_compact_threshold", None), local_trigger)
    return [{"type": "compaction", "compact_threshold": threshold}]


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
    if "context_management" not in text and "compact_threshold" not in text:
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
