from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home


_SENSITIVE_KEYS = {
    "aeskey",
    "authorization",
    "ice",
    "key",
    "raw",
    "rawaudio",
    "rtccandidate",
    "rtcicecandidates",
    "rtcsession",
    "secret",
    "sharedkey",
    "token",
}


def _safe_call_id(call_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(call_id or "unknown"))
    return safe[:128] or "unknown"


def _redact_value(value: Any, *, key: str = "") -> Any:
    normalized_key = re.sub(r"[^a-z0-9]+", "", key.lower())
    if normalized_key in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    return value


class NativeCallTraceWriter:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (get_hermes_home() / "logs" / "calls")

    def record(self, call_id: str, event: str, **fields: Any) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        safe_id = _safe_call_id(call_id)
        path = self.root / f"{safe_id}.jsonl"
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "call_id": str(call_id or ""),
            "event": str(event or "event"),
        }
        row.update(_redact_value(fields))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return path
