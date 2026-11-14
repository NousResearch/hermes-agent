"""Trajectory saving + scratchpad helpers (``_convert_to_trajectory_format`` stays an AIAgent method — batch_runner.py calls it)."""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


# Tool calls and responses sit in the turn text as JSON (agent_runtime_helpers); a text pass can eat
# their quotes and braces (``KEY=***`` swallowing ``"}}``), so their values are redacted decoded.
_EMBEDDED_JSON_RE = re.compile(r"(<(tool_call|tool_response)>\s*)(.*?)(\s*</\2>)", re.DOTALL)


def _redact_text(text: str) -> str:
    parts, pos = [], 0
    for m in _EMBEDDED_JSON_RE.finditer(text):
        try:
            body = json.dumps(_redact_json(json.loads(m.group(3))), ensure_ascii=False)
        except json.JSONDecodeError:
            continue  # not JSON: the text pass below covers it
        parts += [redact_sensitive_text(text[pos:m.start()], force=True), m.group(1), body, m.group(4)]
        pos = m.end()
    parts.append(redact_sensitive_text(text[pos:], force=True))
    return "".join(parts)


def _redact_json(value: Any) -> Any:
    """Decoded tool-call/response JSON: a value the plain pass leaves unchanged is retried on a
    one-field document, because decoding dropped the key the JSON-field rules match on
    (``"api_key"``, ``"x-api-key"``, ``"Cookie"``)."""
    if isinstance(value, dict):
        return {
            key: _redact_field(key, item) if isinstance(item, str) else _redact_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    return redact_trajectory_value(value)


def _redact_field(key: Any, value: str) -> str:
    cleaned = _redact_text(value)
    if cleaned != value:
        return cleaned
    try:
        return json.loads(redact_sensitive_text(json.dumps({key: value}, ensure_ascii=False), force=True))[key]
    except (ValueError, KeyError):
        return "***"


def redact_trajectory_value(value: Any) -> Any:
    """Redact string leaves without allowing redaction markers to change JSON syntax."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [redact_trajectory_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_trajectory_value(item) for key, item in value.items()}
    return value


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Whether content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    return bool(content) and "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def _lock_append_handle(f, acquire: bool) -> None:
    """Exclusive whole-file lock on an append handle: ``flock`` on POSIX, a 1-byte
    ``msvcrt.locking`` range at offset 0 on Windows (append position is restored by the OS)."""
    if os.name == "nt":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if acquire else msvcrt.LK_UNLCK, 1)
        f.seek(0, os.SEEK_END)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if acquire else fcntl.LOCK_UN)


def save_trajectory(trajectory: list[dict[str, Any]], model: str, completed: bool, filename: str | None = None):
    """Append a ShareGPT-format entry to a JSONL file (default trajectory_samples.jsonl / failed_trajectories.jsonl by ``completed``)."""
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
    entry = {"conversations": trajectory, "timestamp": datetime.now().isoformat(), "model": model, "completed": completed}
    try:
        # Trajectories are durable files and may contain tool arguments or user text.  Apply the
        # central redactor at this persistence boundary regardless of the display/config toggle.
        safe_entry = redact_trajectory_value(entry)
        line = json.dumps(safe_entry, ensure_ascii=False) + "\n"  # serialize before taking the lock
        with open(filename, "a", encoding="utf-8") as f:
            # Gateway sessions and batch workers append to the SAME default file; without an
            # exclusive lock around write+flush, entries larger than one write() interleave and the
            # JSONL stops parsing (#12684).
            _lock_append_handle(f, True)
            try:
                f.write(line)
                f.flush()
            finally:
                _lock_append_handle(f, False)
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
