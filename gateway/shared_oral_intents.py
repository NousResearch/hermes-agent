"""Platform-neutral oral intent vocabulary for group/send shortcuts."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Pattern

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent / "data" / "shared_oral_intents.json"
_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "DOTALL": re.DOTALL,
    "MULTILINE": re.MULTILINE,
}


@lru_cache(maxsize=1)
def _load_data() -> dict[str, Any]:
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Shared oral intents data file missing: %s", _DATA_PATH)
        return {}
    except Exception as exc:
        logger.warning("Failed to load shared oral intents from %s: %s", _DATA_PATH, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_term_sequence(section: str, key: str) -> tuple[str, ...]:
    data = _load_data()
    section_value = data.get(section) or {}
    raw_values = section_value.get(key) if isinstance(section_value, dict) else None
    if not isinstance(raw_values, list):
        return ()
    values: list[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if text:
            values.append(text)
    return tuple(values)


def _compile_pattern_entry(entry: Any) -> Pattern[str] | None:
    if isinstance(entry, str):
        pattern = entry
        flags_value: Any = []
    elif isinstance(entry, dict):
        pattern = str(entry.get("pattern") or "").strip()
        flags_value = entry.get("flags") or []
    else:
        return None

    if not pattern:
        return None

    compiled_flags = 0
    if isinstance(flags_value, str):
        flags_iterable = [flags_value]
    elif isinstance(flags_value, list):
        flags_iterable = flags_value
    else:
        flags_iterable = []
    for flag_name in flags_iterable:
        compiled_flags |= _FLAG_MAP.get(str(flag_name or "").strip().upper(), 0)
    return re.compile(pattern, compiled_flags)


def _load_pattern_sequence(section: str, key: str) -> tuple[Pattern[str], ...]:
    data = _load_data()
    section_value = data.get(section) or {}
    raw_entries = section_value.get(key) if isinstance(section_value, dict) else None
    if raw_entries is None:
        return ()
    if isinstance(raw_entries, (str, dict)):
        raw_entries = [raw_entries]
    if not isinstance(raw_entries, list):
        return ()

    patterns: list[Pattern[str]] = []
    for entry in raw_entries:
        compiled = _compile_pattern_entry(entry)
        if compiled is not None:
            patterns.append(compiled)
    return tuple(patterns)


GROUP_CURRENT_TARGET_TERMS = _load_term_sequence("term_sequences", "GROUP_CURRENT_TARGET_TERMS")
GROUP_LISTEN_DISABLE_TERMS = _load_term_sequence("term_sequences", "GROUP_LISTEN_DISABLE_TERMS")
GROUP_LISTEN_ENABLE_TERMS = _load_term_sequence("term_sequences", "GROUP_LISTEN_ENABLE_TERMS")
GROUP_LISTEN_HINT_TERMS = _load_term_sequence("term_sequences", "GROUP_LISTEN_HINT_TERMS")
GROUP_CHAT_ENABLE_TERMS = _load_term_sequence("term_sequences", "GROUP_CHAT_ENABLE_TERMS")
GROUP_REPORT_ENABLE_TERMS = _load_term_sequence("term_sequences", "GROUP_REPORT_ENABLE_TERMS")
GROUP_REPORT_DISABLE_TERMS = _load_term_sequence("term_sequences", "GROUP_REPORT_DISABLE_TERMS")
GROUP_REPORT_NOW_TERMS = _load_term_sequence("term_sequences", "GROUP_REPORT_NOW_TERMS")
GROUP_REPORT_DM_TERMS = _load_term_sequence("term_sequences", "GROUP_REPORT_DM_TERMS")
GROUP_REPORT_CURRENT_CHAT_TERMS = _load_term_sequence("term_sequences", "GROUP_REPORT_CURRENT_CHAT_TERMS")
GROUP_STATUS_QUERY_TERMS = _load_term_sequence("term_sequences", "GROUP_STATUS_QUERY_TERMS")
SEND_QUERY_TERMS = _load_term_sequence("term_sequences", "SEND_QUERY_TERMS")
SEND_CONFIRM_TERMS = _load_term_sequence("term_sequences", "SEND_CONFIRM_TERMS")

GROUP_LISTEN_DISABLE_PATTERNS = _load_pattern_sequence("pattern_sequences", "GROUP_LISTEN_DISABLE_PATTERNS")
GROUP_LISTEN_ENABLE_PATTERNS = _load_pattern_sequence("pattern_sequences", "GROUP_LISTEN_ENABLE_PATTERNS")
_DEFAULT_DIRECT_CONTROL_WRAPPER_PATTERNS = (
    re.compile(r"^(?:我让你|我叫你|我说了|我说的是|我是说)\s*"),
    re.compile(r"^(?:帮我把|请你把|麻烦你把)\s*"),
    re.compile(r"^(?:帮我|请你|麻烦你)\s*"),
)
DIRECT_CONTROL_WRAPPER_PATTERNS = (
    _load_pattern_sequence("pattern_sequences", "DIRECT_CONTROL_WRAPPER_PATTERNS")
    or _DEFAULT_DIRECT_CONTROL_WRAPPER_PATTERNS
)
