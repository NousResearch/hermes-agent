"""Repair model-mangled ``computer_use`` screenshot paths in final responses.

Some models rewrite the Windows path ``computer_use`` reported into a POSIX-looking
one (``C:\\Users\\Alice\\...`` -> ``/Users/Alice/...``) inside an explicit ``MEDIA:``
directive, so delivery-path validation rejects it and drops the attachment.
Deliberately narrow: only rewrites paths in a response that *already* carries a
``MEDIA:`` directive whose ``computer_use_<uuid>`` basename exactly matches a
canonical path returned this turn.  Never auto-attaches; validation still runs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List

logger = logging.getLogger(__name__)

# Absolute-path prefix for canonical capture paths: Windows drive letter, POSIX
# root, or UNC share. Shared string so the summary regex stays in sync.
_ABS_PATH_PREFIX_PATTERN = r"(?:[A-Za-z]:[/\\]|/|\\\\)"
_ABS_PATH_PREFIX_RE = re.compile(r"^" + _ABS_PATH_PREFIX_PATTERN)
_CAPTURE_BASENAME_PATTERN = r"computer_use_[0-9a-f]{32}\.(?:png|jpe?g)"

_COMPUTER_USE_CAPTURE_BASENAME_RE = re.compile(r"^" + _CAPTURE_BASENAME_PATTERN + r"$", re.IGNORECASE)
_COMPUTER_USE_CAPTURE_SUMMARY_RE = re.compile(
    r"\(shareable screenshot saved to "
    r"(?P<path>" + _ABS_PATH_PREFIX_PATTERN + r"[^\r\n]*?" + _CAPTURE_BASENAME_PATTERN + r")\)",
    re.IGNORECASE,
)


def tool_name_by_call_id(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map assistant tool-call ids to tool names for the given messages."""
    mapping: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id") or call.get("call_id")
            name = str((call.get("function") or {}).get("name") or call.get("name") or "")
            if call_id and name:
                mapping[str(call_id)] = name
    return mapping


def _computer_use_capture_basename(path: Any) -> str:
    """Canonical (lowercased) capture basename for either separator style, or ''."""
    basename = re.split(r"[/\\]", str(path or "").strip().strip("`\"'"))[-1]
    return basename.lower() if _COMPUTER_USE_CAPTURE_BASENAME_RE.fullmatch(basename) else ""


def _iter_computer_use_capture_paths(content: Any) -> Iterator[str]:
    """Yield persisted screenshot paths from computer_use result content (JSON, a
    multimodal list, or text; the latter two keep the canonical path in the summary
    line since the envelope's ``meta`` is not stored in the tool message)."""
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped.startswith(("{", "[")):
            for match in _COMPUTER_USE_CAPTURE_SUMMARY_RE.finditer(content):
                yield match.group("path").strip()
            return
        # Parse JSON, never regex-scan it: JSON escaping doubles backslashes, so a raw-text
        # hit would yield a path that exists nowhere.  Fail closed on truncated JSON.
        try:
            payload = json.loads(stripped)
        except Exception:
            return
        if isinstance(payload, (dict, list)):
            yield from _iter_computer_use_capture_paths(payload)
    elif isinstance(content, list):
        for part in content:
            yield from _iter_computer_use_capture_paths(part)
    elif isinstance(content, dict):
        meta = content.get("meta")
        for holder in (content, meta if isinstance(meta, dict) else {}):
            if isinstance(holder.get("screenshot_path"), str):
                yield holder["screenshot_path"]
        # Producer shapes (computer_use ``_capture_response``): content/text = multimodal
        # parts; text_summary/summary = the "(shareable screenshot saved to ...)" line.
        for field in ("content", "text", "text_summary", "summary"):
            nested = content.get(field)
            if isinstance(nested, (str, dict, list)):
                yield from _iter_computer_use_capture_paths(nested)


def _current_turn_messages(messages: List[Dict[str, Any]], history_offset: int) -> List[Dict[str, Any]]:
    if not history_offset or len(messages) >= history_offset:
        return messages[history_offset:]
    # Compression can invalidate the slice boundary: recover the turn from its last user
    # message, fail closed if none remains.  Narrower than run.py's scan-everything fallback
    # for auto-attach — that decides whether to ATTACH; this only rewrites emitted paths.
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return messages[index:]
    return []


def repair_explicit_computer_use_media_paths(response: str, messages: List[Dict[str, Any]], history_offset: int = 0) -> str:
    """Recover model-mangled paths in explicit ``MEDIA:`` directives whose basename
    matches (case-insensitively) a canonical screenshot path from this turn.
    Fail-open: the repair is cosmetic, so any error returns the response unchanged."""
    try:
        return _repair_explicit_computer_use_media_paths_inner(response, messages, history_offset)
    except Exception:
        logger.debug("computer_use media path repair failed", exc_info=True)
        return response


def _canonical_capture_paths(turn_messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """``{lowercase basename: absolute canonical path}`` from this turn's computer_use results."""
    call_id_names = tool_name_by_call_id(turn_messages)
    canonical: Dict[str, str] = {}
    for msg in turn_messages:
        if msg.get("role") not in {"tool", "function"}:
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if str(msg.get("name") or msg.get("tool_name") or call_id_names.get(call_id) or "") != "computer_use":
            continue
        for path in _iter_computer_use_capture_paths(msg.get("content")):
            basename = _computer_use_capture_basename(path)
            if basename and _ABS_PATH_PREFIX_RE.match(path):
                canonical[basename] = path
    return canonical


def _repair_explicit_computer_use_media_paths_inner(response: str, messages: List[Dict[str, Any]], history_offset: int = 0) -> str:
    if "MEDIA:" not in response:
        return response
    canonical_by_basename = _canonical_capture_paths(_current_turn_messages(messages, history_offset))
    if not canonical_by_basename:
        return response

    # Lazy: keeps the import cheap for standalone cron processes that may never hit MEDIA:.
    from gateway.platforms.base import BasePlatformAdapter

    repaired = response
    for emitted_path, _is_voice in BasePlatformAdapter.extract_media(response)[0]:
        canonical = canonical_by_basename.get(_computer_use_capture_basename(emitted_path))
        if canonical and emitted_path != canonical:
            repaired = repaired.replace(emitted_path, canonical)
    return repaired
