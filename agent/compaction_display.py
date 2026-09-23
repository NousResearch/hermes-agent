"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent.context_compressor import ContextCompressor, is_compaction_summary_message


_COMPACTION_INTERNAL_FIELDS = (
    "tool_calls",
    "finish_reason",
    "reasoning",
    # Provider replay/metadata fields that ride the wire on every request but are invisible to
    # ``msg["content"]``/``msg["tool_calls"]`` accounting. Codex Responses sessions in particular carry
    # ``codex_reasoning_items`` blobs of ``encrypted_content`` that can dominate the serialized session (a
    # measured 214-turn session held ~115K tokens / 27% of its payload there — #55572).
    # ``reasoning_details`` is handled separately (see ``_reasoning_details_text_chars``): its signed/base64
    # envelope is excluded from the budget, mirroring the preflight estimator's exclusion in
    # ``model_metadata._estimate_message_tokens_without_images`` (#73298).
    # An assistant turn may carry only reasoning/thinking content with no visible text (extended-thinking
    # turns, thinking-only recovery responses). Such a turn is persisted with its reasoning fields and is
    # recallable from the transcript, but dropping it here as "empty" makes it vanish from the
    # resumed/reloaded session view while the desktop's reasoning disclosure has nothing to render. Keep it
    # when it carries reasoning so the "Thinking…" block still shows. (#44022)
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


# ``display_metadata`` key: the user's own view of a multimodal (list) turn whose ``content`` also
# carries injected context — memory prefetch / ``pre_llm_call`` (#71998) and gateway must-deliver
# notes. That context is model-facing and stays in ``content`` so the model replays what it saw;
# display projections show the user only their own part. ``text`` is the stripped text projection
# (the shape flushes persist), ``parts`` the part count (the shape in-place compaction persists).
INJECTED_CONTEXT_META_KEY = "injected_context"


def record_user_view_before_injection(message: Any) -> None:
    """Stamp the user's own view on a list user turn before context is appended to its content.

    Only the FIRST injection records: later ones (gateway notes, then the prologue's context part)
    must not overwrite the view with content that already carries injected text."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return
    meta = message.get("display_metadata")
    meta = dict(meta) if isinstance(meta, dict) else {}
    if INJECTED_CONTEXT_META_KEY in meta:
        return
    from agent.session_persistence import _durable_content

    text = _durable_content(content)
    if not isinstance(text, str) or not text.strip():
        return
    # Leading whitespace only: load paths strip the whole content's outer ends, and the user's part
    # is the head — trimming its tail too would stop the recorded view matching the stored row.
    meta[INJECTED_CONTEXT_META_KEY] = {"text": text.lstrip(), "parts": len(content)}
    message["display_metadata"] = meta


def user_view_without_injected_context(message: Any) -> Any:
    """The user's own content for a user row that carries injected context, else ``None``.

    Verified against the row before anything is dropped: raw parts must end in text parts past the
    recorded count; a text projection (sanitized on some load paths) must still start with the
    recorded view followed by the separator the projection writes. A row rewritten since (a persist
    override, a redaction) no longer matches and is shown unchanged."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    meta = message.get("display_metadata")
    view = meta.get(INJECTED_CONTEXT_META_KEY) if isinstance(meta, dict) else None
    if not isinstance(view, dict):
        return None
    content = message.get("content")
    if isinstance(content, list):
        count = view.get("parts")
        if (isinstance(count, int) and not isinstance(count, bool) and 0 < count < len(content)
                and all(isinstance(p, dict) and p.get("type") == "text" for p in content[count:])):
            return content[:count]
        return None
    text = view.get("text")
    if isinstance(content, str) and isinstance(text, str) and text:
        if content.lstrip().startswith(text + "\n"):
            return text
    return None


def project_compaction_message_for_display(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return authentic transcript content, or ``None`` for a pure handoff.

    Model-facing recovery history retains the complete carrier. Display
    projections instead remove the handoff, inherited tool state, and internal
    reasoning while preserving any real prior-tail content or live user ask
    embedded in the carrier. A user row carrying injected turn context shows
    only the user's own part (``user_view_without_injected_context``).
    """
    if not isinstance(message, dict):
        return None
    if not is_compaction_summary_message(message):
        projected = message.copy()
        user_view = user_view_without_injected_context(message)
        if user_view is not None:
            projected["content"] = user_view
        return projected

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    if projected is None:
        return None

    projected = projected.copy()
    for key in _COMPACTION_INTERNAL_FIELDS:
        projected.pop(key, None)
    projected.pop("display_kind", None)
    return projected
