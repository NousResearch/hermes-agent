"""Per-request image COUNT ceilings (vLLM ``--limit-mm-per-prompt``, SGLang, DeepInfra 8, Fireworks 60,
DashScope 250 data-URIs, OpenAI-compatible gateways at 30).

A count rejection fails byte-identically on every resend, and history only grows, so without help one
such 400 locks the session for good. Recovery records the ceiling the provider stated (or half the
rejected count when it states none) per ``(provider, model)``; from then on every request for that
route drops its OLDEST images from the per-call copy before sending. History keeps the pixels, and the
newest user message is never touched (that is what the user just asked about). Retirement goes through
the shared step function so the cached prefix moves in batches, never once per new image.

Port of MiniMax-AI/minimax-code#431 (request-level image cap + deterministic rejection), learned from
the rejection instead of configured per model.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.context_compressor import _is_image_part, _replace_image_parts, _strip_images_from_tool_msg, _tool_result_parts
from agent.image_eviction_policy import outbound_image_retire_count
from agent.turn_context import drop_stale_api_content
from agent.vision_message_prep import _provider_model_key

# A stated ceiling outside this band is a misparse (a byte size, a request id), not an image count.
_SANE_LIMIT = range(1, 1001)

# Ordered most-specific first; group 1 is always the ceiling, never the rejected count.
_LIMIT_PATTERNS = (
    re.compile(r"(?:at most|no more than|maximum of|maximum number of|too many)\s+(\d+)\s+image"),  # vLLM-style
    re.compile(r"(\d+)\s+image\(s\)\s+may be provided"),  # vLLM
    re.compile(r"exceeds limit\s+(\d+)"),  # SGLang "Image count 5 exceeds limit 2 per request."
    re.compile(r"images? in request:\s*\d+\s*>\s*(\d+)"),  # DeepInfra / gateways "Too many images in request: 31 > 30"
    re.compile(r"number of images[^\d]*?\bto\s+(\d+)"),  # Fireworks "...limit the number of images per conversation to 60"
    re.compile(r"data-uri per request:\s*(\d+)"),  # DashScope "Exceeded limit on max data-uri per request: 250"
)


def image_count_limit_from_error(error: Exception) -> Optional[int]:
    """The per-request image ceiling a count rejection states, or None when it names no usable number."""
    parts = []
    for value in (error, getattr(error, "message", None), getattr(error, "body", None)):
        if value:
            parts.append(str(value))
    text = " ".join(parts).lower()
    for pattern in _LIMIT_PATTERNS:
        match = pattern.search(text)
        if match and int(match.group(1)) in _SANE_LIMIT:
            return int(match.group(1))
    return None


def _image_count(content: Any) -> int:
    parts = _tool_result_parts(content)
    return sum(1 for p in parts if _is_image_part(p)) if isinstance(parts, list) else 0


def _without_images(msg: dict[str, Any], limit: int) -> Optional[dict[str, Any]]:
    """Copy of ``msg`` with its image parts replaced by a note; None when it carries none."""
    if msg.get("role") == "tool":
        return _strip_images_from_tool_msg(msg)
    stripped = _replace_image_parts(
        msg.get("content"), f"[Earlier image omitted: this provider accepts at most {limit} images per request]"
    )
    if stripped is None:
        return None
    new_msg = {**msg, "content": stripped}
    drop_stale_api_content(new_msg)
    return new_msg


def strip_images_to_count_limit(api_messages: list[dict[str, Any]], limit: int) -> int:
    """Retire the oldest image-bearing rows of the per-call copy until at most ``limit`` images remain.

    The newest user message is reserved wherever it sits (after a tool round it precedes the
    assistant/tool rows). Returns how many images were removed; 0 when the request already fits or
    when the reserved message alone exceeds the ceiling (no retirement can help — the caller surfaces
    the rejection). Rows are replaced, never mutated, so history sharing the row dicts is untouched.
    """
    latest_user = next(
        (i for i in range(len(api_messages) - 1, -1, -1)
         if isinstance(api_messages[i], dict) and api_messages[i].get("role") == "user"),
        -1,
    )
    reserved = _image_count(api_messages[latest_user].get("content")) if latest_user >= 0 else 0
    if reserved > limit:
        return 0
    carriers = [
        (i, n) for i in range(len(api_messages) - 1, -1, -1)
        if i != latest_user and isinstance(api_messages[i], dict)
        and (n := _image_count(api_messages[i].get("content")))
    ]
    retire = outbound_image_retire_count([n for _, n in carriers], reserved, limit=limit, floor=0)
    removed = 0
    for i, n in carriers[len(carriers) - retire:]:
        new_msg = _without_images(api_messages[i], limit)
        if new_msg is not None:
            api_messages[i] = new_msg
            removed += n
    return removed


def apply_learned_image_count_limit(agent: Any, api_messages: Any) -> int:
    """Send-path cap for a route that already rejected an image count this session."""
    limit = getattr(agent, "_image_count_limits", {}).get(_provider_model_key(agent))
    if limit is None or not isinstance(api_messages, list):
        return 0
    return strip_images_to_count_limit(api_messages, limit)


def learn_image_count_limit(agent: Any, api_error: Exception, api_messages: Any) -> tuple[int, int]:
    """Record the route's ceiling from a count rejection and apply it to this attempt's copy.

    Returns ``(removed, limit)``; nothing is recorded when nothing could be removed, so a rejection
    the newest message causes on its own is not mistaken for a learnable ceiling.
    """
    if not isinstance(api_messages, list):
        return 0, 0
    sent = sum(_image_count(m.get("content")) for m in api_messages if isinstance(m, dict))
    limit = image_count_limit_from_error(api_error) or max(1, sent // 2)
    removed = strip_images_to_count_limit(api_messages, limit)
    if removed:
        limits = vars(agent).setdefault("_image_count_limits", {})  # per agent, so per session
        key = _provider_model_key(agent)
        limits[key] = min(limit, limits.get(key, limit))
    return removed, limit
