"""API-call-time prefill messages: fold system-role prefills into the leading system prompt.

Strict chat templates (Qwen-family Jinja templates, several llama.cpp / LM Studio
gateways) reject any system message after the first one ("System message must be at
the beginning"), and the native Anthropic converter keeps only the LAST system message,
so a separate system prefill silently replaced the real system prompt there. Folding
keeps exactly one leading system message on every route.

Content policy (gateway prefill files are raw JSON, so content may be a string or a
list of content blocks):

* every piece is a string -> one string, pieces stripped and joined with a blank line
  (the system prompt stays a byte prefix, so its static-prefix cache split still applies);
* any piece is a block list -> one block list: lists keep their blocks verbatim (order,
  ``cache_control`` and any other keys preserved) and a single dict block is treated as a
  one-block list. Each string piece becomes one stripped ``{"type": "text"}`` block that
  carries the blank-line separator toward its neighbours, so a renderer that ``""``-joins
  plain text parts (the send path's cache-decoration undo flattens one- and two-part
  system lists that way) still reads ``"<base>\n\n<prefill>"``. Two adjacent lists are
  concatenated with no separator added, because their blocks are never rewritten;
* empty pieces (``None``, ``""``, whitespace-only strings, ``[]``) are skipped.

Nothing is ever coerced through ``str()``: a content type outside that policy raises
``TypeError`` instead of being sent as a Python repr.
"""

from __future__ import annotations

import copy
from typing import Any

_Content = Any  # str | list[dict] | dict


def _is_empty(content: _Content) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, (list, dict)):
        return not content
    raise TypeError(
        f"system prefill content must be a string or a list of content blocks, got {type(content).__name__}"
    )


_SEPARATOR = "\n\n"


def _combine_system_content(pieces: list[_Content]) -> _Content:
    pieces = [piece for piece in pieces if not _is_empty(piece)]
    if all(isinstance(piece, str) for piece in pieces):
        return _SEPARATOR.join(piece.strip() for piece in pieces)
    blocks: list[Any] = []
    last = len(pieces) - 1
    for i, piece in enumerate(pieces):
        if isinstance(piece, str):
            text = piece.strip()
            if i > 0:
                text = _SEPARATOR + text
            if i < last and not isinstance(pieces[i + 1], str):
                text += _SEPARATOR
            blocks.append({"type": "text", "text": text})
        elif isinstance(piece, dict):
            blocks.append(piece)
        else:
            blocks.extend(piece)
    return blocks


def fold_system_prefill_messages(
    api_messages: list[dict[str, Any]],
    prefill_messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return the request with prefills applied: system-role prefill content folded into the
    single leading system message (created if the request has none), every other prefill
    inserted right after it in its original order.

    Prefill messages are deep-copied, so send-path sanitizers that rewrite the request in
    place never reach ``agent.prefill_messages``. ``api_messages`` and its rows are not
    mutated; the leading system row is replaced by a new dict when it changes.
    """
    if not prefill_messages:
        return api_messages

    system_pieces: list[_Content] = []
    other_prefill: list[dict[str, Any]] = []
    for prefill in prefill_messages:
        cloned = copy.deepcopy(prefill)
        if cloned.get("role") == "system":
            system_pieces.append(cloned.get("content"))
        else:
            other_prefill.append(cloned)

    messages = list(api_messages)
    has_leading_system = bool(messages) and messages[0].get("role") == "system"
    if any(not _is_empty(piece) for piece in system_pieces):
        if has_leading_system:
            existing = messages[0]
            content = _combine_system_content([existing.get("content"), *system_pieces])
            messages[0] = {**existing, "content": content}
        else:
            messages.insert(0, {"role": "system", "content": _combine_system_content(system_pieces)})
            has_leading_system = True

    offset = 1 if has_leading_system else 0
    messages[offset:offset] = other_prefill
    return messages
