"""Native OpenAI Responses server-side compaction — gpt-5.6 on direct OpenAI routes only.

OpenAI's Responses API supports server-side compaction: include
``context_management=[{"type": "compaction", "compact_threshold": N}]`` in a
``/v1/responses`` request and, when the rendered input crosses N tokens, the
server summarizes older context into an opaque ``compaction`` output item
(``encrypted_content``, sealed to the issuing endpoint). Replaying that item
as an input item on later requests stands in for the pruned history, so the
model keeps long-horizon recall without the client ever seeing a summary.
Docs: https://developers.openai.com/api/docs/guides/compaction

Hermes' support is deliberately narrow (live verification, Aug 2026):

* **gpt-5.6 family only.** gpt-5.6 and its variants compact correctly.
  Sending the field to gpt-5.1 / gpt-5.2 reliably fails server-side —
  HTTP 500 on the blocking path and a permanent stall on the streaming
  path (90s watchdog x 3 retries = a dead turn). There is no structured
  "unsupported" rejection to downgrade on, so the only safe gate is an
  explicit model-family check.
* **Direct OpenAI routes only:** api.openai.com (API key) or the ChatGPT
  Codex backend (subscription OAuth). Every other Responses surface
  (xAI, GitHub/Copilot, relays, local servers) never sees the field —
  most would 400 on the unknown parameter, and none can mint or decrypt
  the compaction blob.

Ownership model: Hermes' local compression stays fully armed as the
fallback owner. The native threshold is clamped safely below the local
compressor's trigger so the server compacts first; if it doesn't (native
disabled mid-session, provider hiccup, non-eligible route), the local
summarizer fires exactly as before. There is no new custody state — the
captured compaction items ride the existing ``codex_reasoning_items``
sidecar, which already handles persistence (state.db), gateway session
replay, cross-issuer stamping, and the encrypted-replay kill switch.

This module stays free of transport/adapter dependencies so the transport,
adapter, and conversation loop can share the gate without import cycles. The
two exceptions — ``agent.context_compressor`` and ``agent.message_content`` —
sit below this module in the dependency graph (neither imports
``native_compaction``), so importing their provenance/text primitives here
introduces no cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from agent.context_compressor import is_compaction_summary_message
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# Native compaction fires this many tokens below the local compressor's
# trigger so the server always gets the first shot at compaction.
LOCAL_TRIGGER_SAFETY_MARGIN = 8_192

DEFAULT_COMPACT_THRESHOLD = 200_000

# Model-family gate. Substring match on the lowercased model id so dated
# snapshots (gpt-5.6-2026-07-xx) and variants (gpt-5.6-mini) stay eligible.
_ELIGIBLE_MODEL_MARKER = "gpt-5.6"


def is_native_compaction_model(model: Optional[str]) -> bool:
    """True when the model is in the gpt-5.6 family."""
    return _ELIGIBLE_MODEL_MARKER in (model or "").lower()


def is_direct_openai_route(
    base_url: Optional[str],
    *,
    is_codex_backend: bool = False,
) -> bool:
    """True for api.openai.com or the ChatGPT Codex backend — nothing else."""
    if is_codex_backend:
        return True
    try:
        hostname = (urlsplit(base_url or "").hostname or "").lower()
    except ValueError:
        return False
    return hostname == "api.openai.com"


def resolve_compact_threshold(
    configured_threshold: Any,
    local_trigger_tokens: Any = None,
) -> int:
    """Clamp the configured native threshold below the local compressor trigger.

    Without the clamp a native threshold above the local trigger would let the
    local summarizer fire first every time, making native compaction dead
    config. ``local_trigger_tokens`` is ``ContextCompressor.threshold_tokens``
    when a compressor is attached, else None.
    """
    try:
        configured = int(configured_threshold)
    except (TypeError, ValueError):
        configured = DEFAULT_COMPACT_THRESHOLD
    if isinstance(configured_threshold, bool) or configured <= 0:
        configured = DEFAULT_COMPACT_THRESHOLD

    local = None
    try:
        if local_trigger_tokens is not None and not isinstance(local_trigger_tokens, bool):
            local = int(local_trigger_tokens)
    except (TypeError, ValueError):
        local = None
    if local is None or local <= 0:
        return configured

    if local > LOCAL_TRIGGER_SAFETY_MARGIN:
        upper = local - LOCAL_TRIGGER_SAFETY_MARGIN
    else:
        upper = max(1_024, int(local * 0.8))
    return max(1_024, min(configured, upper))


def native_compaction_context_management(
    agent: Any,
    *,
    is_codex_backend: bool,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Return the ``context_management`` payload for this request, or None.

    None means "do not send the field" — the request is byte-identical to
    pre-feature behavior. All gates are re-checked per request so a
    mid-session model switch or the in-session kill switch
    (``agent.codex_responses_native_compaction = False``, set by the
    conversation loop's rejection recovery) takes effect on the next call.
    """
    if not bool(getattr(agent, "codex_responses_native_compaction", False)):
        return None
    # compression.enabled: false disables ALL automatic compaction, native
    # included — mirrors the codex_app_server_auto contract.
    if not bool(getattr(agent, "compression_enabled", True)):
        return None
    if is_xai_responses or is_github_responses:
        return None
    if not is_native_compaction_model(getattr(agent, "model", None)):
        return None
    if not is_direct_openai_route(
        getattr(agent, "base_url", None), is_codex_backend=is_codex_backend
    ):
        return None

    compressor = getattr(agent, "context_compressor", None)
    threshold = resolve_compact_threshold(
        getattr(agent, "codex_responses_compact_threshold", DEFAULT_COMPACT_THRESHOLD),
        getattr(compressor, "threshold_tokens", None) if compressor is not None else None,
    )
    return [{"type": "compaction", "compact_threshold": threshold}]


# Retention budget for plaintext user messages carried across a native
# compaction boundary (mirrors Codex CLI's RETAINED_MESSAGE_TOKEN_BUDGET).
# Live verification (Aug 2026, gpt-5.6 @ api.openai.com): the server renders
RETAINED_USER_MESSAGE_TOKEN_BUDGET = 64_000

# Retention budget for local compression summary messages carried across a native
# compaction boundary to prevent summary token inflation.
RETAINED_SUMMARY_TOKEN_BUDGET = 32_000


def _approx_tokens(text: str) -> int:
    """Cheap chars//4 token estimate — same shape Codex uses for retention."""
    return max(1, len(text) // 4)


def _extract_item_text(item: Any) -> Optional[str]:
    """Extract measurable text from string, list content, output_text, or nested metadata text.

    Returns None when the item carries no measurable text.
    Handles string content, multipart lists (input_text/text/output_text), and fallback keys.
    """
    if not isinstance(item, dict):
        return None

    content = item.get("content")
    if content is None and "output_text" in item:
        content = item.get("output_text")

    if isinstance(content, str):
        return content if content.strip() else None

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    parts.append(part.strip())
            elif isinstance(part, dict):
                part_text = part.get("text") or part.get("input_text") or part.get("output_text")
                if isinstance(part_text, str) and part_text.strip():
                    parts.append(part_text.strip())
                part_meta = part.get("metadata")
                if isinstance(part_meta, dict) and isinstance(part_meta.get("text"), str):
                    if part_meta["text"].strip():
                        parts.append(part_meta["text"].strip())
        text = " ".join(parts)
        return text if text.strip() else None

    return None


def _is_summary_item(item: Any) -> bool:
    """True when *item* is a canonical Hermes compression-summary message.

    Delegates entirely to
    ``agent.context_compressor.is_compaction_summary_message`` — the single
    authoritative provenance check already used by every other summary
    consumer (memory providers, frontends, the compactor itself). It prefers
    the exact, truthy ``COMPRESSED_SUMMARY_METADATA_KEY`` marker and falls
    back to the canonical prefix classifier (``SUMMARY_PREFIX`` /
    ``LEGACY_SUMMARY_PREFIX`` / historical prefixes, including the
    merge-into-tail shape) for the case where the underscore-prefixed key
    was already stripped by a wire sanitizer.

    Deliberately NOT a second heuristic: no arbitrary underscore-key scan, no
    inference from a falsy or unrelated metadata key, and no matching on
    ad-hoc content headings like ``"## Summary"`` in ordinary text — any of
    those can promote a normal user/assistant message (or adversarial
    content) to durable retained history (#90975 review).
    """
    return is_compaction_summary_message(item)


def prune_pre_checkpoint_items(
    items: List[Dict[str, Any]],
    retained_user_token_budget: int = RETAINED_USER_MESSAGE_TOKEN_BUDGET,
    retained_summary_token_budget: int = RETAINED_SUMMARY_TOKEN_BUDGET,
    enable_summary_retention: bool = True,
    item_sources: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Restructure Responses input around the newest compaction checkpoint.

    The server drops every input item that precedes a replayed ``compaction``
    item (live-verified Aug 2026), so sending pre-checkpoint history is dead
    weight AND silently erases the user's plaintext asks — including any
    local-compression summary the agent already produced, which previously
    vanished here because it carries ``role="assistant"``, not ``"user"``
    (#90975). When a checkpoint is present, rebuild the wire as::

        [checkpoint run] + [retained user & summary messages (newest-first budget)] + [post]

    - The NEWEST contiguous run of checkpoints wins.
    - Retained user messages are kept verbatim within
      ``retained_user_token_budget``; the boundary message is head-truncated
      when it only partially fits (string content only) — goals are usually
      stated up front, so the head is the valuable end.
    - Compression summary messages (``_is_summary_item``, the canonical
      ``agent.context_compressor`` provenance check) are retained whole
      within ``retained_summary_token_budget``. A summary is never
      byte/character-sliced: Hermes summaries carry structural framing
      (handoff prefix, end marker, merge-into-tail delimiters) that a blind
      slice can corrupt, so one that doesn't fit whole is dropped instead.
      A summary already retained once (identical text) is never duplicated,
      so repeated checkpoints stay idempotent.
    - ``enable_summary_retention`` is a function-level override (used by
      tests and callers that need the pre-#90975 behavior back); it is not
      wired to a user-facing config surface.
    - Original relative chronological order between user messages and
      summaries is preserved.
    - ``item_sources`` (optional, parallel to ``items``) is the raw chat
      message each Responses item was converted from. By the time a summary
      reaches this function as a converted ``item`` it can already be lossy:
      a merge-into-tail tool-result carrier becomes a typed
      ``function_call_output`` (no ``content``/``role`` survives the
      conversion at all), and a merge-into-tail assistant carrier can be
      shadowed by a stale exact ``codex_message_items`` replay captured
      before the merge rewrote its content. When a source is provided and is
      itself a canonical summary carrier (``is_compaction_summary_message``),
      its content is read directly from the source — never from the
      converted item — and it is retained as a synthesized
      ``role="assistant"`` message regardless of what shape the original
      item took. Without ``item_sources`` (default), retention only sees
      what survived conversion, matching pre-#90976 behavior (#90976).
    """
    if not isinstance(items, list) or not items:
        return items

    last_cp = None
    for i, item in enumerate(items):
        if isinstance(item, dict) and item.get("type") == "compaction":
            last_cp = i
    if last_cp is None:
        return items

    # Extend backwards over the contiguous run ending at last_cp.
    first_cp = last_cp
    while (
        first_cp > 0
        and isinstance(items[first_cp - 1], dict)
        and items[first_cp - 1].get("type") == "compaction"
    ):
        first_cp -= 1

    pre = items[:first_cp]
    checkpoint_run = items[first_cp : last_cp + 1]
    post = items[last_cp + 1 :]

    if isinstance(item_sources, list) and len(item_sources) == len(items):
        pre_sources: List[Any] = item_sources[:first_cp]
    else:
        pre_sources = [None] * len(pre)

    retained_reversed: List[Dict[str, Any]] = []
    user_remaining = max(0, int(retained_user_token_budget))
    summary_remaining = max(0, int(retained_summary_token_budget))
    seen_summary_texts: set = set()

    def _try_retain_summary(text: Optional[str]) -> Optional[Dict[str, Any]]:
        """Check budget/dedup/cost for a summary; return cost info or None."""
        if not text or summary_remaining <= 0 or text in seen_summary_texts:
            return None
        cost = _approx_tokens(text)
        if cost > summary_remaining:
            # Never byte-slice a summary's structural framing — drop it
            # whole rather than corrupt the handoff prefix / end marker.
            return None
        seen_summary_texts.add(text)
        return {"cost": cost}

    for item, source in zip(reversed(pre), reversed(pre_sources)):
        if not isinstance(item, dict):
            continue

        # Canonical source-based summary detection: reads the ORIGINAL chat
        # message's own content, so it sees past a lossy conversion (a
        # typed `function_call_output` wrapper, or a stale exact-replay
        # message) that erased the summary from `item` itself (#90976).
        # This is never a heuristic promotion of arbitrary item content —
        # it only fires when the source message itself is a canonical,
        # provenance-tagged summary carrier.
        if enable_summary_retention and isinstance(source, dict) and _is_summary_item(source):
            text = flatten_message_text(source.get("content")) if isinstance(source, dict) else ""
            text = text if text.strip() else None
            result = _try_retain_summary(text)
            if result:
                _src_role = source.get("role")
                retained_reversed.append({
                    "role": _src_role if _src_role in ("user", "assistant") else "assistant",
                    "content": text,
                })
                summary_remaining -= result["cost"]
            continue

        # Skip typed non-message items (function_call_output etc. never
        # carry role=user or a summary flag, but stay defensive about
        # future shapes).
        if "type" in item and item.get("type") != "message":
            continue

        is_summary = enable_summary_retention and _is_summary_item(item)
        is_user = item.get("role") == "user"

        if not is_user and not is_summary:
            continue

        text = _extract_item_text(item)
        if text is None:
            continue
        # Image-only user messages have empty text but non-empty content —
        # main retains them at 1-token cost (images count as zero, matching
        # Codex's retention accounting). Don't skip them just because text
        # is falsy.
        if not text and not is_user:
            continue

        if is_summary:
            result = _try_retain_summary(text)
            if result:
                retained_reversed.append(item)
                summary_remaining -= result["cost"]
        elif is_user:
            if user_remaining <= 0:
                continue
            cost = _approx_tokens(text)
            if cost <= user_remaining:
                retained_reversed.append(item)
                user_remaining -= cost
            elif isinstance(item.get("content"), str):
                truncated = dict(item)
                truncated["content"] = item["content"][: user_remaining * 4]
                if truncated["content"].strip():
                    retained_reversed.append(truncated)
                user_remaining = 0

    retained_ordered = list(reversed(retained_reversed))
    result = checkpoint_run + retained_ordered + post

    logger.debug(
        "Pruned pre-checkpoint items: %d input -> %d retained (user_rem=%d, summary_rem=%d)",
        len(items),
        len(result),
        user_remaining,
        summary_remaining,
    )

    return result


def is_native_compaction_rejection(error: Any, status_code: Any = None) -> bool:
    """True when a provider error is a STRUCTURED rejection of the
    context_management field.

    Used by the conversation loop's one-shot recovery: strip the field,
    disable native compaction for the rest of the session, retry. Matching
    is deliberately narrow — a transient 5xx/timeout whose body merely
    ECHOES the request (and therefore contains the field name) must NOT
    permanently downgrade native compaction for the session (#82777).

    Two conditions, both required when a status is known:

    * ``status_code`` is 400 (or unknown/None — some transports surface
      only a message string; field-name matching alone is then the best
      available signal, preserving pre-#82777 behavior for them), and
    * the error text names ``context_management`` / ``compact_threshold``
      alongside rejection language ("unknown", "unsupported", "invalid",
      "unexpected", "not permitted"...). A bare field-name echo without
      rejection language does not match.
    """
    text = str(error or "").lower()
    if "context_management" not in text and "compact_threshold" not in text:
        return False
    if status_code is not None:
        try:
            if int(status_code) != 400:
                return False
        except (TypeError, ValueError):
            pass
    rejection_markers = (
        "unknown", "unsupported", "invalid", "unexpected", "not permitted",
        "not allowed", "unrecognized", "extra field", "no such", "bad request",
        "not supported",
    )
    return any(marker in text for marker in rejection_markers)


def has_compaction_checkpoint(items: Any) -> bool:
    """Does this ``codex_reasoning_items`` sidecar carry a compaction checkpoint?

    A ``type: "compaction"`` item is the server-side stand-in for history that
    has already been pruned — cumulative context, not per-turn reasoning. It
    rides the same sidecar as ordinary reasoning items, so anything that
    rewrites or discards that sidecar (or the message carrying it) has to ask
    this question first: the checkpoint exists in exactly one place, and the
    request that loses it loses the compacted history with it.
    """
    return any(
        isinstance(item, dict) and item.get("type") == "compaction"
        for item in (items if isinstance(items, list) else ())
    )


def merge_interim_reasoning_items(
    prior_items: Any,
    new_items: Any,
) -> List[Dict[str, Any]]:
    """Merge ``codex_reasoning_items`` across Codex incomplete-continuation
    dedup, preserving native compaction checkpoints.

    The incomplete-retry path updates a visually-duplicate interim assistant
    message in place with the newer response's replay payload. A checkpoint
    captured on the EARLIER response is a cumulative context carrier the
    continuation won't re-emit (the replayed checkpoint keeps the server
    render under threshold), so a blind overwrite drops the only copy and the
    next request balloons back to full history. Rule: newer items win, but
    prior checkpoints are prepended unless the newer payload carries its own.
    """
    kept_checkpoints = [
        item
        for item in (prior_items if isinstance(prior_items, list) else [])
        if isinstance(item, dict) and item.get("type") == "compaction"
    ]
    new_list = list(new_items) if isinstance(new_items, list) else []
    if has_compaction_checkpoint(new_list) or not kept_checkpoints:
        return new_list
    return kept_checkpoints + new_list
