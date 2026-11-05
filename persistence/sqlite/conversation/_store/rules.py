"""Pure helpers and constants of the application modules the store functions run, each block equal to the module its comment names."""


from __future__ import annotations
from typing import (
    Any, MutableMapping, Optional,
)
from session_rules.turns import (
    MESSAGE_UID, message_uid_or_none, mint_uid,
    _AUTO_LOAD_PREFIX, _SKILL_INVOCATION_PREFIX,
)
import logging
import re
import time
import errno
import sys
from datetime import datetime
import math


# Row snapshot and message uid keys, as agent.message_metadata defines them.
DB_ROW_SNAPSHOT = "_db_row_snapshot"


CANONICAL_ROW = "_canonical_row"


TOOL_CALL_UID = "_tool_call_uid"


DATA_USE = "data_use"


def lift_data_use(message: MutableMapping[str, Any]) -> None:
    """Restore a replayed row's ``data_use`` key from its durable ``display_metadata`` copy."""
    metadata = message.get("display_metadata")
    if isinstance(metadata, dict) and metadata.get(DATA_USE):
        message[DATA_USE] = list(metadata[DATA_USE])


def stamp_message_uid(msg: MutableMapping[str, Any]) -> str:
    """The dict's ``message_uid``, minting one (``mint_uid``) when it carries none.

    Minted ONCE per logical message, at its first insert, and stamped on the caller's dict so every later
    insert of that dict (compaction generation, rotation handoff, replace) writes the same uid. Never
    derived from content, timestamp or tool-call ids: two distinct messages may share all three.
    """
    uid = message_uid_or_none(msg)
    if uid is None:
        uid = msg[MESSAGE_UID] = mint_uid()
    return uid


# As agent.conversation_compression_archive defines it.
ABSORBED_ROW_IDS = "_absorbed_row_ids"


# Checkpoint carriers, as agent.context_compressor finds them.
def _is_checkpoint_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "compaction"


def _newest_checkpoint_carrier(messages: list[dict[str, Any]], key: str) -> int:
    """Index of the last assistant message carrying a ``type: "compaction"`` item under *key*, or -1.
    Transcript-side mirror of ``native_compaction.prune_pre_checkpoint_items``' newest-run-wins rule:
    the wire builder drops every checkpoint before the last one, so this is the only carrier whose
    checkpoint can still reach a request."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        items = msg.get(key)
        if isinstance(items, list) and any(_is_checkpoint_item(item) for item in items):
            return i
    return -1


# Lone surrogates, as agent.message_sanitization replaces them.
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD; no-op when none present."""
    # ``str.isascii`` is an O(1) flag check; surrogates are never ASCII, so the
    # regex scan only runs for the (rare) non-ASCII leaf.
    if text.isascii():
        return text
    return _SURROGATE_RE.sub('\ufffd', text)


# LIKE patterns of the skill scaffolding prefixes agent.skill_commands writes.
SKILL_SCAFFOLD_SQL_LIKE = _SKILL_INVOCATION_PREFIX + "%"


AUTO_LOAD_SCAFFOLD_SQL_LIKE = _AUTO_LOAD_PREFIX + "%"


# Leading message timestamps, as gateway.message_timestamps strips them.
_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\[(?:"
    r"(?P<dow>[A-Z][a-z]{2}) "
    r"(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?: (?P<tz>[A-Za-z0-9_+\-/:]+))?"
    r"|(?P<iso>\d{4}-\d{2}-\d{2}T[^\]]+)"
    r")\]\s*"
)


def _localize(dt: datetime, tz) -> Optional[float]:
    """Epoch for ``dt`` in ``tz`` or the local zone; None outside platform limits."""
    naive_local = dt.tzinfo is None and tz is None
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz) if tz is not None else dt.astimezone()
        return float(dt.timestamp())
    except (OverflowError, ValueError):
        return None
    except OSError as exc:
        if not (sys.platform == "win32" and naive_local and exc.errno == errno.EINVAL):
            return None
        # Windows' fold probe can reject the first Unix day. mktime can
        # resolve those local wall times without changing their meaning.
        # Limit recovery to that fold-free interval and reject normalization.
        wall_time = dt.timetuple()
        try:
            epoch = time.mktime(wall_time)
            if 0 <= epoch < 86_400 and time.localtime(epoch)[:6] == wall_time[:6]:
                return float(epoch + dt.microsecond / 1_000_000)
        except (OSError, OverflowError, ValueError):
            pass
        return None


def _parse_iso(text: str, tz=None) -> Optional[float]:
    """Parse an ISO-8601 string (incl. ``+0200`` offsets fromisoformat rejects)."""
    for parse in (datetime.fromisoformat, lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z")):
        try:
            return _localize(parse(text), tz)
        except (TypeError, ValueError):
            continue
    return None


def _parse_timestamp_match(match: re.Match, tz=None) -> Optional[float]:
    if match.group("iso"):
        return _parse_iso(match.group("iso"), tz)
    try:
        dt = datetime.strptime(f"{match.group('date')} {match.group('time')}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return _localize(dt, tz)


def strip_leading_message_timestamps(content: str, tz=None) -> tuple[str, Optional[float]]:
    """Strip leading gateway timestamp prefixes → ``(clean_content, embedded_epoch)``.
    With several prefixes the one closest to the text wins, preserving the platform-send
    time of legacy rows like ``[processing time] [platform time] [sender] message``."""
    if not isinstance(content, str) or not content:
        return content, None
    text, embedded_epoch = content, None
    while (match := _TIMESTAMP_PREFIX_RE.match(text)) is not None:
        parsed = _parse_timestamp_match(match, tz=tz)
        if parsed is not None:
            embedded_epoch = parsed
        text = text[match.end():]
    return text, embedded_epoch


# Epoch coercion, as hermes_cli.timefmt does it.
logger = logging.getLogger("hermes_cli.timefmt")


EPOCH_MIN = 0.0


EPOCH_MAX = 4_200_000_000.0


def coerce_epoch(value: Any, *, session_id: Optional[str] = None, field: str = "timestamp") -> Optional[float]:
    """A stored timestamp cell as float epoch seconds, or ``None`` when it cannot be trusted.

    Numbers, numeric strings and ``datetime`` are accepted; anything else, non-finite values and
    values outside ``EPOCH_MIN..EPOCH_MAX`` return ``None`` after a WARNING naming the session so
    the corrupt row can be found. ``None``/``""`` mean "unset" and stay silent. Every reader that
    renders a row timestamp goes through here (a bad row degrades to one ``?`` cell, never a dead
    command) and every writer uses it to refuse persisting a new bad row.
    """
    if value is None or value == "":
        return None
    try:
        ts = float(value.timestamp()) if isinstance(value, datetime) else float(value)
    except (TypeError, ValueError):
        ts = math.nan
    if not (EPOCH_MIN <= ts <= EPOCH_MAX):  # also False for nan
        logger.warning("Ignoring corrupt %s %r%s", field, value, f" on session {session_id}" if session_id else "")
        return None
    return ts


# As tools.todo_tool defines it.
MAX_TODO_RESULT_CHARS = 512_000


# Spillover file names, as tools.tool_result_storage writes them.
_SPILLOVER_NAME_RE = re.compile(r"spillover(?:/|\\+)([A-Za-z0-9_.-]+)")


SPILLOVER_SQL_FILTER = "(instr(content, 'spillover') > 0 OR instr(api_content, 'spillover') > 0)"


def spillover_names_in_text(*texts) -> set:
    """Spillover archive basenames referenced by transcript message text (empty set on none)."""
    names: set = set()
    for text in texts:
        if isinstance(text, str) and "spillover" in text:
            names.update(_SPILLOVER_NAME_RE.findall(text))
    return names


RESTORED_FROM_STORE = "_restored_from_store"


_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _is_image_part(part: Any) -> bool:
    """True if ``part`` is an image block (``image_url``, ``input_image``, or ``image``)."""
    return isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES


_PERSIST_AFTER_ADMISSION_INTERRUPT = "_persist_after_admission_interrupt"
