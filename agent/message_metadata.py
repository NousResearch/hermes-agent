"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

import hashlib
import logging
import re
from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp"})

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])

logger = logging.getLogger(__name__)

# Keep this detector diagnostic-only: mutating a model response at this point would change both the
# provider-visible transcript and the durable copy. Scan a bounded prefix so a pathological response
# cannot turn the append chokepoint into another expensive operation.
_GUARDRAIL_SCAN_CHARS = 256_000
_RAW_ROLE_PREFIX_RE = re.compile(r"^\s*(?:user|assistant|system|tool)\s*(?::|\r?\n)", re.IGNORECASE)
_DIVIDER_LINE_RE = re.compile(r"^\s*(?:[-_=*]){8,}\s*$")
_REPEATED_DIVIDER_LINES = 8


def _observe_assistant_guardrail(message: MutableMapping[str, Any]) -> None:
    """Log bounded evidence when assistant text resembles a pasted transcript or divider loop."""
    if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
        return
    content = message["content"]
    if not content:
        return
    scanned = content[:_GUARDRAIL_SCAN_CHARS]
    raw_role_prefix = bool(_RAW_ROLE_PREFIX_RE.match(scanned))
    divider_lines = 0
    longest_divider_run = 0
    current_run = 0
    for line in scanned.splitlines():
        if _DIVIDER_LINE_RE.fullmatch(line):
            divider_lines += 1
            current_run += 1
            longest_divider_run = max(longest_divider_run, current_run)
        else:
            current_run = 0
    repeated_dividers = (
        divider_lines >= _REPEATED_DIVIDER_LINES or longest_divider_run >= 3)
    if not (raw_role_prefix or repeated_dividers):
        return
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
    logger.warning(
        "Assistant output guardrail observation: chars=%d scanned=%d raw_role_prefix=%s "
        "divider_lines=%d longest_divider_run=%d digest=%s",
        len(content), len(scanned), raw_role_prefix, divider_lines, longest_divider_run, digest,
    )


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time; all other callers use
    the local wall clock. Returns the same mapping for use at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    _observe_assistant_guardrail(message)
    messages.append(stamp_message_timestamp(message, timestamp=timestamp))
    return message
