"""SOC audit hook — always-on structured logging for cybersecurity operations.

Activated when the environment variable HERMES_CYBER_AUDIT=true is set.
Writes a newline-delimited JSON (NDJSON) audit log to:

    $HERMES_HOME/logs/cyber_audit.jsonl

Every agent:step and agent:end event is recorded with:
  - ISO-8601 timestamp (UTC)
  - session_id / session_key (gateway sessions) or task description
  - platform / user identity (gateway context only)
  - event_type
  - tool call name + truncated input (agent:step)
  - tool result summary (agent:step)
  - completion reason + token counts (agent:end)

The log is append-only and never truncated by this module.
Rotate externally with logrotate or equivalent.

Security notes:
  - Credential-like values in tool inputs are redacted (env vars that match
    _REDACT_RE are replaced with ***).
  - The log file is created with mode 0600 (owner-read-write only).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Match common secret-bearing key names inside tool argument dicts
_REDACT_RE = re.compile(
    r"api[_-]?key|token|password|secret|credential|auth|bearer",
    re.IGNORECASE,
)


def _is_enabled() -> bool:
    return os.environ.get("HERMES_CYBER_AUDIT", "").lower() in ("1", "true", "yes")


def _audit_log_path() -> Path:
    try:
        from hermes_cli.config import get_hermes_home
        logs_dir = get_hermes_home() / "logs"
    except Exception:
        logs_dir = Path.home() / ".hermes" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "cyber_audit.jsonl"
    # Ensure restrictive permissions on creation
    if not log_path.exists():
        try:
            log_path.touch(mode=0o600)
        except Exception:
            pass
    return log_path


def _redact(obj: Any, depth: int = 0) -> Any:
    if depth > 4:
        return obj
    if isinstance(obj, dict):
        return {
            k: "***" if _REDACT_RE.search(str(k)) else _redact(v, depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(item, depth + 1) for item in obj[:20]]
    if isinstance(obj, str) and len(obj) > 500:
        return obj[:500] + "…"
    return obj


def _write(record: dict) -> None:
    try:
        path = _audit_log_path()
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:
        logger.warning("cyber_audit: failed to write audit log: %s", exc)


# ---------------------------------------------------------------------------
# Public hook entry point
# ---------------------------------------------------------------------------

async def handle(event_type: str, context: dict) -> None:
    if not _is_enabled():
        return

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record: dict[str, Any] = {
        "ts": ts,
        "event": event_type,
    }

    # Attach session identity when available (gateway context)
    for key in ("session_id", "session_key", "platform", "user_id", "user_name", "chat_id"):
        val = context.get(key)
        if val is not None:
            record[key] = str(val)

    if event_type == "agent:step":
        tool_call = context.get("tool_call") or {}
        tool_result = context.get("tool_result") or {}
        record["tool"] = tool_call.get("name") or tool_call.get("function", {}).get("name")
        raw_input = tool_call.get("input") or tool_call.get("function", {}).get("arguments") or {}
        if isinstance(raw_input, str):
            try:
                raw_input = json.loads(raw_input)
            except Exception:
                raw_input = {"_raw": raw_input[:300]}
        record["tool_input"]  = _redact(raw_input)
        # Truncate result to keep log manageable
        result_str = str(tool_result)
        record["tool_result_preview"] = result_str[:300] + ("…" if len(result_str) > 300 else "")

    elif event_type == "agent:end":
        record["stop_reason"]  = context.get("stop_reason")
        record["input_tokens"] = context.get("input_tokens")
        record["output_tokens"] = context.get("output_tokens")
        record["iterations"]   = context.get("iterations")

    elif event_type in ("session:start", "session:end", "gateway:startup"):
        # Minimal record — just timestamp + event type + identity already added
        pass

    _write(record)
