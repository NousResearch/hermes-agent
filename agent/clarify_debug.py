"""Always-on diagnostics for clarify tool-call Unicode corruption."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

from hermes_constants import get_hermes_home


_MAX_LOG_BYTES = 5 * 1024 * 1024


def _log_path():
    return get_hermes_home() / "clarify_debug.jsonl"


def _is_enabled() -> bool:
    value = os.environ.get("HERMES_CLARIFY_DEBUG")
    return value is None or value.strip().lower() not in {"0", "false", "off", "no"}


def _rotate_if_oversized(log_path) -> None:
    try:
        if log_path.stat().st_size <= _MAX_LOG_BYTES:
            return
        lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
        log_path.write_text("".join(lines[len(lines) // 2 :]), encoding="utf-8")
    except OSError:
        return


def log_clarify_debug(stage: str, value, **extra) -> None:
    """Append one clarify diagnostic record without affecting tool execution."""
    if not _is_enabled():
        return
    try:
        text = value if isinstance(value, str) else repr(value)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "repr": repr(value),
            "utf8_codepoints": [ord(char) for char in text],
            **extra,
        }
        log_path = _log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            _rotate_if_oversized(log_path)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return
