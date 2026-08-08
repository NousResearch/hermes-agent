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

This module is dependency-free on purpose so the transport, adapter, and
conversation loop can share the gate without import cycles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

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


def is_native_compaction_rejection(error: Any) -> bool:
    """True when a provider error names the context_management field.

    Used by the conversation loop's one-shot recovery: strip the field,
    disable native compaction for the rest of the session, retry. Matching
    is deliberately narrow — generic 4xx/5xx/timeouts must NOT permanently
    downgrade native compaction, they take the normal retry path.
    """
    text = str(error or "").lower()
    return "context_management" in text or "compact_threshold" in text


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
    new_has_checkpoint = any(
        isinstance(item, dict) and item.get("type") == "compaction"
        for item in new_list
    )
    if new_has_checkpoint or not kept_checkpoints:
        return new_list
    return kept_checkpoints + new_list
