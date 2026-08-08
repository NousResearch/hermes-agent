"""Total request-body budget for size-limited transports (ChatGPT Codex).

The ``chatgpt.com/backend-api/codex`` transport hard-drops oversized request
bodies: the server reads the whole upload, then closes the connection without
sending any response, which the client can only surface as a retryable-looking
``APIConnectionError``.  Measured by bisection (2026-08-02): bodies up to
~1,175,000 bytes return HTTP 200; bodies of ~1,200,000 bytes and above are
dropped after ~15 s.  Because sessions run with ``store=False``, the entire
history is re-uploaded on every call — one oversized item permanently wedges
its session, and retries re-send the identical body.

This module enforces a total serialized-body budget immediately before send
(transport preflight).  Token-based context compression cannot protect this
limit: images and large tool outputs are byte-dense but token-cheap, so a body
can be 6x over the wire ceiling while token accounting reports plenty of room.

Degradation order — stop as soon as the body fits, never touch user text:

1. Truncate ``function_call_output`` items *behind* the active tail (the
   most recent tool exchanges the model needs verbatim to continue), largest
   first, keeping head + tail around an omission marker.
2. Re-encode embedded data-URL images down (oldest first) via
   :func:`agent.image_payloads.constrain_image_payload`.
3. Drop history ``reasoning`` items (opaque encrypted blobs — untruncatable).
4. Stub oversized history ``function_call.arguments`` with valid JSON.
5. Replace still-oversized images behind the active tail with a text
   placeholder.
6. Truncate active-tail tool outputs, with a more generous floor.
7. Re-truncate history outputs with minimal floors (hundreds of outputs can
   exceed the budget even at the first-pass floors).
8. Last resort: replace remaining images anywhere — regardless of individual
   size, since under-cap images can collectively bust the budget — with a
   text placeholder. A lost image beats a dropped request.

If the body still exceeds the budget after all five steps, it is returned
as-is with an ERROR log: that means instructions + tool schemas + the current
user content alone exceed the transport limit, which no history surgery can
fix — the send will fail and the log tells a human exactly why.

Cost model: the body is fully serialized twice per over-budget call (once to
measure, once to verify); every ladder step tracks progress through exact
per-item deltas (an item serializes identically standalone and inside the
``input`` array).  Retries re-run the ladder from scratch: a
content-blind memo was measured cheaper but can substitute an unrelated
same-size request, so correctness wins.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agent.image_payloads import (
    DEFAULT_NATIVE_IMAGE_MAX_DIMENSION,
    agent_config_int,
    constrain_image_payload,
    data_url_payload_size,
)

logger = logging.getLogger(__name__)

# Measured ceiling ~1.17-1.2 MB (see module docstring); default leaves
# headroom for serialization drift and transport framing.
DEFAULT_REQUEST_BODY_BUDGET_BYTES = 1_000_000

# History tool outputs are context, not the active exchange: keep enough of
# each end to preserve what the output was and how it ended.
_HISTORY_OUTPUT_HEAD = 4_096
_HISTORY_OUTPUT_TAIL = 1_024
# Active-tail outputs are what the model is currently working from — keep an
# order of magnitude more before cutting.
_TAIL_OUTPUT_HEAD = 16_384
_TAIL_OUTPUT_TAIL = 4_096

# How many of the most recent tool exchanges stay verbatim: the model needs
# its latest results to continue the loop, but a long agentic run's early
# outputs are context, not the active exchange (a cron session can carry 75
# outputs after a single user message — protecting "everything after the last
# user message" would protect all of them).
_PROTECTED_RECENT_OUTPUTS = 3

_OMISSION_MARKER = (
    "\n[... {omitted} characters omitted by hermes: request exceeded the "
    "transport body-size limit ...]\n"
)
_IMAGE_PLACEHOLDER = (
    "[image omitted by hermes: request exceeded the transport body-size limit]"
)

# api_kwargs keys that never reach the wire body.
_NON_WIRE_KEYS = ("extra_headers",)

_DATA_URL_RE = re.compile(r"^data:(?P<mime>image/[a-z0-9.+-]+);base64,(?P<b64>.*)$", re.S)

def get_request_body_budget() -> int:
    """Return the configured body budget (``agent.max_request_body_bytes``).

    Malformed, missing, zero, or negative values fall back to the default.
    """
    return agent_config_int(
        "max_request_body_bytes", DEFAULT_REQUEST_BODY_BUDGET_BYTES
    )


def serialized_body_size(api_kwargs: Dict[str, Any]) -> int:
    """Approximate the wire size of the JSON body built from ``api_kwargs``.

    Excludes client-side keys (underscore-prefixed, ``extra_headers``) that
    the SDK does not serialize into the request body.
    """
    wire = {
        k: v
        for k, v in api_kwargs.items()
        if not (isinstance(k, str) and (k.startswith("_") or k in _NON_WIRE_KEYS))
    }
    try:
        return len(json.dumps(wire, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return len(str(wire).encode("utf-8"))


def _active_tail_start(input_items: List[Any]) -> int:
    """First index of the protected active tail.

    The tail covers everything after the last user message *or* the last
    ``_PROTECTED_RECENT_OUTPUTS`` tool exchanges, whichever protects less.
    User message items appear both as ``{"type": "message", "role": "user"}``
    and as bare ``{"role": "user", ...}`` shapes.
    """
    last_user = -1
    output_indices: List[int] = []
    for idx, item in enumerate(input_items):
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            last_user = idx
        elif item.get("type") == "function_call_output":
            output_indices.append(idx)

    from_user = last_user + 1
    if not output_indices:
        return from_user
    kth_output = output_indices[-_PROTECTED_RECENT_OUTPUTS:][0]
    # Include the function_call immediately preceding the k-th output.
    from_outputs = max(0, kth_output - 1)
    return max(from_user, from_outputs)


def _json_size(obj: Any) -> int:
    """Exact serialized size of one input item or content part.

    An element serializes identically standalone and inside its enclosing
    array (same separators), so mutation deltas computed with this are exact.
    """
    try:
        return len(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return len(str(obj).encode("utf-8"))


def _truncate_text(text: str, head: int, tail: int) -> str:
    omitted = len(text) - head - tail
    if omitted <= len(_OMISSION_MARKER):
        return text
    return (
        text[:head]
        + _OMISSION_MARKER.format(omitted=omitted)
        + (text[-tail:] if tail > 0 else "")
    )


def _truncate_output_item(item: Dict[str, Any], head: int, tail: int) -> bool:
    """Truncate a ``function_call_output`` item in place. Returns True if changed."""
    output = item.get("output")
    if isinstance(output, str):
        truncated = _truncate_text(output, head, tail)
        if len(truncated) < len(output):
            item["output"] = truncated
            return True
        return False
    if isinstance(output, list):
        changed = False
        for part in output:
            if isinstance(part, dict) and part.get("type") == "input_text":
                text = part.get("text")
                if isinstance(text, str):
                    truncated = _truncate_text(text, head, tail)
                    if len(truncated) < len(text):
                        part["text"] = truncated
                        changed = True
        return changed
    return False


def _constrain_data_url(data_url: str, max_payload_bytes: int) -> Optional[str]:
    """Re-encode a data-URL image to fit ``max_payload_bytes``; None on failure.

    Returns the original URL when it already fits.
    """
    if len(data_url) <= max_payload_bytes:
        return data_url
    match = _DATA_URL_RE.match(data_url)
    if not match:
        return None
    try:
        raw = base64.b64decode(match.group("b64"), validate=False)
    except (binascii.Error, ValueError):
        return None
    constrained = constrain_image_payload(
        raw,
        match.group("mime"),
        max_payload_bytes=max_payload_bytes,
        # The byte cap dominates here; keep the standard dimension ceiling.
        max_dimension=DEFAULT_NATIVE_IMAGE_MAX_DIMENSION,
    )
    if constrained is None:
        return None
    new_raw, new_mime = constrained
    if data_url_payload_size(len(new_raw), new_mime) > max_payload_bytes:
        return None
    return "data:%s;base64,%s" % (
        new_mime,
        base64.b64encode(new_raw).decode("ascii"),
    )


def _iter_image_parts(
    input_items: List[Any],
) -> List[Tuple[int, Dict[str, Any]]]:
    """Return (item_index, part) for every data-URL input_image part."""
    found: List[Tuple[int, Dict[str, Any]]] = []
    for idx, item in enumerate(input_items):
        if not isinstance(item, dict):
            continue
        parts: List[Any] = []
        if item.get("role") is not None and isinstance(item.get("content"), list):
            parts = item["content"]
        elif item.get("type") == "function_call_output" and isinstance(
            item.get("output"), list
        ):
            parts = item["output"]
        for part in parts:
            if (
                isinstance(part, dict)
                and part.get("type") == "input_image"
                and isinstance(part.get("image_url"), str)
                and part["image_url"].startswith("data:")
            ):
                found.append((idx, part))
    return found


def apply_request_body_budget(
    api_kwargs: Dict[str, Any],
    budget_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Return ``api_kwargs`` with its serialized body constrained to budget.

    Under budget, the original object is returned untouched.  Otherwise the
    ``input`` list is deep-copied and degraded per the module docstring; the
    original is never mutated.
    """
    budget = budget_bytes if isinstance(budget_bytes, int) and budget_bytes > 0 else (
        get_request_body_budget()
    )
    size = serialized_body_size(api_kwargs)
    if size <= budget:
        return api_kwargs

    input_items = api_kwargs.get("input")
    if not isinstance(input_items, list) or not input_items:
        logger.error(
            "request_budget: body is %d bytes (budget %d) but has no input "
            "items to degrade; sending as-is and the transport will likely "
            "drop it.",
            size,
            budget,
        )
        return api_kwargs

    constrained = dict(api_kwargs)
    # Strings are shared by reference under deepcopy (atomic in CPython), so
    # this copies structure only, not megabyte payloads.
    items: List[Any] = copy.deepcopy(input_items)
    constrained["input"] = items
    tail_start = _active_tail_start(items)
    actions: Counter = Counter()
    # Running total maintained through exact per-item deltas; one final full
    # serialization verifies it before we log success.
    current = size

    def _image_allowance() -> int:
        """Per-image byte allowance from the budget headroom left right now.

        Non-image content (instructions, tools, truncated history) is charged
        first; the remainder — minus a serialization-drift margin — is split
        evenly across the images still present. A lone photo in a small
        conversation keeps most of the envelope; many images share it. Floor
        keeps degenerate cases usable.
        """
        parts = _iter_image_parts(items)
        if not parts:
            return 64_000
        image_bytes = sum(len(prt.get("image_url", "")) for _i, prt in parts)
        room = budget - (current - image_bytes) - 16_384
        return max(64_000, room // len(parts))

    def _sorted_outputs(lo: int, hi: int) -> List[Dict[str, Any]]:
        return sorted(
            (
                item
                for item in items[lo:hi]
                if isinstance(item, dict)
                and item.get("type") == "function_call_output"
            ),
            key=_json_size,
            reverse=True,
        )

    def truncate_pass(candidates: List[Dict[str, Any]], head: int, tail: int, label: str) -> None:
        nonlocal current
        for item in candidates:
            if current <= budget:
                return
            before = _json_size(item)
            if _truncate_output_item(item, head, tail):
                current += _json_size(item) - before
                actions[label] += 1

    def placeholder_pass(
        candidates: List[Tuple[int, Dict[str, Any]]],
        label: str,
        respect_cap: bool = True,
    ) -> None:
        nonlocal current
        allowance = _image_allowance()
        for _idx, part in candidates:
            if current <= budget:
                return
            if respect_cap and len(part.get("image_url", "")) <= allowance:
                continue
            before = _json_size(part)
            part.clear()
            part["type"] = "input_text"
            part["text"] = _IMAGE_PLACEHOLDER
            current += _json_size(part) - before
            actions[label] += 1

    # Step 1: truncate history tool outputs, largest first.
    truncate_pass(
        _sorted_outputs(0, tail_start),
        _HISTORY_OUTPUT_HEAD,
        _HISTORY_OUTPUT_TAIL,
        "history outputs truncated",
    )

    # Step 2: re-encode embedded images down, oldest first.
    if current > budget:
        allowance = _image_allowance()
        for _idx, part in _iter_image_parts(items):
            if current <= budget:
                break
            constrained_url = _constrain_data_url(part["image_url"], allowance)
            if constrained_url and len(constrained_url) < len(part["image_url"]):
                current += len(constrained_url) - len(part["image_url"])
                part["image_url"] = constrained_url
                actions["images re-encoded"] += 1

    # Step 3: drop history reasoning items (opaque encrypted blobs that can
    # be large and cannot be truncated without corruption; the adapter already
    # drops foreign-issuer reasoning, so absence is a supported state).
    if current > budget:
        kept: List[Any] = []
        for idx, item in enumerate(items):
            if (
                current > budget
                and idx < tail_start
                and isinstance(item, dict)
                and item.get("type") == "reasoning"
            ):
                current -= _json_size(item) + 2  # item + list separator
                actions["history reasoning dropped"] += 1
                continue
            kept.append(item)
        if actions["history reasoning dropped"]:
            tail_start -= actions["history reasoning dropped"]
            items[:] = kept

    # Step 4: stub oversized history function_call arguments (kept as valid
    # JSON — replayed calls are context, never re-executed).
    if current > budget:
        for item in items[:tail_start]:
            if current <= budget:
                break
            if not (isinstance(item, dict) and item.get("type") == "function_call"):
                continue
            args = item.get("arguments")
            if isinstance(args, str) and len(args) > 4_096:
                before = _json_size(item)
                item["arguments"] = json.dumps(
                    {
                        "_hermes_truncated": "%d-character arguments omitted: "
                        "request exceeded the transport body-size limit" % len(args)
                    }
                )
                current += _json_size(item) - before
                actions["history call arguments stubbed"] += 1

    # Step 5: placeholder history images that still don't fit.
    if current > budget:
        placeholder_pass(
            _iter_image_parts(items[:tail_start]), "history images replaced"
        )

    # Step 6: truncate active-tail tool outputs (generous floor), largest first.
    if current > budget:
        truncate_pass(
            _sorted_outputs(tail_start, len(items)),
            _TAIL_OUTPUT_HEAD,
            _TAIL_OUTPUT_TAIL,
            "active outputs truncated",
        )

    # Step 7: escalate — the first-pass floors (4KB head each) can themselves
    # exceed the budget when history holds hundreds of outputs; re-truncate
    # with minimal floors.
    if current > budget:
        truncate_pass(
            _sorted_outputs(0, tail_start), 512, 128, "history outputs re-truncated"
        )

    # Step 8: last resort — placeholder remaining images anywhere regardless
    # of individual size (several under-cap images can collectively bust the
    # budget), oldest first so the current turn's image goes last. Also covers
    # images Pillow cannot re-encode (corrupt/unsupported): a lost image beats
    # a request the transport is guaranteed to drop.
    if current > budget:
        placeholder_pass(
            _iter_image_parts(items), "images replaced (last resort)", respect_cap=False
        )

    summary = ", ".join("%s: %d" % (k, v) for k, v in actions.items()) or "no-op"
    final_size = serialized_body_size(constrained)
    if final_size <= budget:
        logger.info(
            "request_budget: constrained body %d -> %d bytes (budget %d): %s",
            size,
            final_size,
            budget,
            summary,
        )
        return constrained

    logger.error(
        "request_budget: body still %d bytes after degradation (budget %d, "
        "started %d, actions: %s). Instructions + tool schemas + current user "
        "content alone exceed the transport limit; the send will likely be "
        "dropped by the server.",
        final_size,
        budget,
        size,
        summary,
    )
    return constrained


__all__ = [
    "DEFAULT_REQUEST_BODY_BUDGET_BYTES",
    "apply_request_body_budget",
    "get_request_body_budget",
    "serialized_body_size",
]
