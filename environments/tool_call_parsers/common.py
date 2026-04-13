"""W4 / F-013: shared utilities for tool-call parsers.

Every parser in this package builds a `ChatCompletionMessageToolCall` from
a model-specific raw text format. The per-parser regex / token scheme
stays per-parser, but argument normalisation, call construction, and
(for tag-based parsers) tag-pair extraction are identical boilerplate —
extracted here so a fix applies everywhere at once.

The audit flagged this package as having ~0 unit tests; most of the
test value for F-013 is in `tests/environments/tool_call_parsers/`.
This module is a supporting win that reduces surface area.
"""
from __future__ import annotations

import json
import random
import re
import string
import uuid
from typing import Any, Iterable, List, Optional, Tuple

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


def normalize_arguments(args: Any) -> str:
    """Coerce an arbitrary `arguments` payload to a JSON string.

    Parsers historically varied:
      - some received dicts, json.dumps'd them;
      - some received JSON strings, passed through;
      - some received lists or numbers, json.dumps'd them;
      - some fell through silently on unknown types.

    Now: dicts / lists / scalars → `json.dumps(..., ensure_ascii=False)`;
    str → returned as-is; anything else → json.dumps.
    """
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def make_uuid_id(prefix: str = "call_") -> str:
    """Short UUID-based tool-call id — used by hermes/llama/deepseek/etc."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def make_mistral_id() -> str:
    """Mistral-style 9-char alphanumeric id.

    Historically each parser kept its own local copy of this. Centralised
    here so future Mistral variants pick up any change.
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=9))


def make_tool_call(
    *,
    name: str,
    arguments: Any,
    call_id: Optional[str] = None,
) -> ChatCompletionMessageToolCall:
    """Build a ChatCompletionMessageToolCall with normalised arguments.

    Pass `call_id=None` (default) to get a fresh `call_<uuid8>` id.
    Pass a pre-generated id (e.g. from `make_mistral_id()`) to override.
    """
    return ChatCompletionMessageToolCall(
        id=call_id or make_uuid_id(),
        type="function",
        function=Function(name=name, arguments=normalize_arguments(arguments)),
    )


def extract_tagged_payloads(
    text: str,
    open_tag: str,
    close_tag: str,
) -> List[str]:
    """Return each `<OPEN>payload</CLOSE>` body (plus the trailing unclosed
    `<OPEN>...` if the generation was truncated mid-call).

    Used by hermes, longcat, qwen, and any future tag-delimited format.
    Matches the historical regex `OPEN\\s*(.*?)\\s*CLOSE|OPEN\\s*(.*)`
    with DOTALL semantics.
    """
    escaped_open = re.escape(open_tag)
    escaped_close = re.escape(close_tag)
    pattern = re.compile(
        rf"{escaped_open}\s*(.*?)\s*{escaped_close}|{escaped_open}\s*(.*)",
        re.DOTALL,
    )
    payloads: List[str] = []
    for match in pattern.findall(text):
        # match is a tuple: (closed_content, unclosed_content). Pick
        # whichever group captured, skip empties.
        raw = match[0] if match[0] else match[1]
        if raw.strip():
            payloads.append(raw)
    return payloads


def split_content_from_first_marker(text: str, marker: str) -> Tuple[Optional[str], bool]:
    """Return the text before `marker`, stripped — or None if empty/absent.

    Second tuple element is True when the marker was found in `text`.
    """
    idx = text.find(marker)
    if idx < 0:
        return text.strip() or None, False
    prefix = text[:idx].strip()
    return (prefix or None), True
