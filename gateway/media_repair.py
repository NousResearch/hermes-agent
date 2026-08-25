"""Repair model-mangled ``computer_use`` screenshot paths in final responses.

``computer_use`` persists a bounded screenshot into the Hermes image cache and
tells the model its absolute path.  Some models rewrite a Windows path into a
POSIX-looking one (``C:\\Users\\Alice\\...`` -> ``/Users/Alice/...``) when
emitting an explicit ``MEDIA:`` directive, so delivery-path validation rejects
the nonexistent path and the attachment is dropped.

The repair is deliberately narrow: it only rewrites paths inside a response
that *already* carries an explicit ``MEDIA:`` directive, and only when the
directive's generated ``computer_use_<uuid>`` basename exactly matches a
canonical screenshot path returned by ``computer_use`` in the current turn.
It never auto-attaches captures, and normal media path validation still runs
after the repair.

This lives in its own module (mirroring ``gateway/media_policy.py``) so every
delivery surface — the messaging gateway's main turn path, gateway background
tasks, and cron job delivery — shares one implementation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List

# Absolute-path prefix accepted for canonical capture paths: Windows drive
# letter, POSIX root, or UNC share. Kept as a pattern string so the summary
# regex below and the compiled prefix check stay in sync.
_ABS_PATH_PREFIX_PATTERN = r"(?:[A-Za-z]:[/\\]|/|\\\\)"
_ABS_PATH_PREFIX_RE = re.compile(r"^" + _ABS_PATH_PREFIX_PATTERN)

_COMPUTER_USE_CAPTURE_BASENAME_RE = re.compile(
    r"^computer_use_[0-9a-f]{32}\.(?:png|jpe?g)$",
    re.IGNORECASE,
)
_COMPUTER_USE_CAPTURE_SUMMARY_RE = re.compile(
    r"\(shareable screenshot saved to "
    r"(?P<path>" + _ABS_PATH_PREFIX_PATTERN + r"[^\r\n]*?"
    r"computer_use_[0-9a-f]{32}\.(?:png|jpe?g))\)",
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
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "")
            if call_id and name:
                mapping[str(call_id)] = name
    return mapping


def _computer_use_capture_basename(path: Any) -> str:
    """Return a canonical capture basename for either path separator style."""
    value = str(path or "").strip().strip("`\"'")
    basename = re.split(r"[/\\]", value)[-1]
    if _COMPUTER_USE_CAPTURE_BASENAME_RE.fullmatch(basename):
        return basename.lower()
    return ""


def _iter_computer_use_capture_paths(content: Any) -> Iterator[str]:
    """Yield persisted screenshot paths from computer_use result content.

    The tool can return JSON, a multimodal content list, or a text fallback.
    The latter two retain the canonical path in the human-readable summary
    even though the multimodal envelope's ``meta`` dictionary is not stored in
    the tool message.
    """
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            # JSON-looking content: parse first, never regex-scan the raw
            # text. JSON escaping doubles backslashes, so a summary-regex hit
            # on the raw string would yield ``C:\\\\Users\\\\...`` — a path
            # that exists nowhere. Fail closed on unparseable JSON (tool
            # output truncation) rather than repair to a corrupted path.
            try:
                payload = json.loads(stripped)
            except Exception:
                return
            if isinstance(payload, (dict, list)):
                yield from _iter_computer_use_capture_paths(payload)
            return
        for match in _COMPUTER_USE_CAPTURE_SUMMARY_RE.finditer(content):
            yield match.group("path").strip()
        return

    if isinstance(content, list):
        for part in content:
            yield from _iter_computer_use_capture_paths(part)
        return

    if not isinstance(content, dict):
        return

    screenshot_path = content.get("screenshot_path")
    if isinstance(screenshot_path, str):
        yield screenshot_path
    meta = content.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("screenshot_path"), str):
        yield meta["screenshot_path"]
    # Producer shapes (tools/computer_use/tool.py::_capture_response):
    # "content"/"text" — multimodal envelope parts; "text_summary"/"summary" —
    # the human-readable summary carrying the "(shareable screenshot saved
    # to ...)" line.
    for field in ("content", "text", "text_summary", "summary"):
        nested = content.get(field)
        if isinstance(nested, (str, dict, list)):
            yield from _iter_computer_use_capture_paths(nested)


def repair_explicit_computer_use_media_paths(
    response: str,
    messages: List[Dict[str, Any]],
    history_offset: int = 0,
) -> str:
    """Recover model-mangled paths for explicitly requested screenshots.

    Repair only an already-explicit ``MEDIA:`` directive whose unique
    generated basename matches a canonical screenshot path from this turn.
    This does not auto-attach ordinary computer-use captures, and normal
    media path validation still runs after the repair.
    """
    if "MEDIA:" not in response:
        return response

    if history_offset and len(messages) >= history_offset:
        turn_messages = messages[history_offset:]
    elif history_offset:
        # Compression can invalidate the original slice boundary. Recover the
        # current turn from its last user message; fail closed if none
        # remains. (Deliberately narrower than the scan-everything fallback
        # in gateway/run.py::_collect_auto_append_media_tags — that helper
        # decides whether to ATTACH, this one only rewrites paths the model
        # already explicitly emitted.)
        last_user = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        turn_messages = messages[last_user:] if last_user is not None else []
    else:
        turn_messages = messages

    call_id_names = tool_name_by_call_id(turn_messages)

    canonical_by_basename: Dict[str, str] = {}
    for msg in turn_messages:
        if msg.get("role") not in {"tool", "function"}:
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        tool_name = str(
            msg.get("name")
            or msg.get("tool_name")
            or call_id_names.get(call_id)
            or ""
        )
        if tool_name != "computer_use":
            continue
        for path in _iter_computer_use_capture_paths(msg.get("content")):
            basename = _computer_use_capture_basename(path)
            if basename and _ABS_PATH_PREFIX_RE.match(path):
                canonical_by_basename[basename] = path

    if not canonical_by_basename:
        return response

    from gateway.platforms.base import BasePlatformAdapter

    media_files, _ = BasePlatformAdapter.extract_media(response)
    repaired = response
    for emitted_path, _is_voice in media_files:
        canonical = canonical_by_basename.get(
            _computer_use_capture_basename(emitted_path)
        )
        if canonical and emitted_path != canonical:
            repaired = repaired.replace(emitted_path, canonical)
    return repaired
