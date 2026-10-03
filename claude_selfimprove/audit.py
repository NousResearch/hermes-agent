"""Append-only audit log.

Every state change this pipeline makes — a candidate created, evidence
merged, a threshold met, a file written, a conflict blocked, a rollback —
gets one JSON line here. The log is never rewritten or truncated by normal
operation (only ``rollback`` reads it, and only the operator's own tooling
prunes it), so it stays a trustworthy record of what the pipeline actually
did, independent of the mutable candidate database.

No raw transcript text belongs in an audit entry — only identifiers,
decisions, and already-redacted summaries the caller passed in.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from . import atomic, paths


def record(event: str, **fields: Any) -> None:
    """Append one audit entry. Never raises — a logging failure must not
    abort the pipeline run that triggered it."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        atomic.append_line(paths.audit_log_path(), line)
    except (OSError, TypeError, ValueError):
        pass


def read_all() -> list[dict[str, Any]]:
    """Read every audit entry, oldest first. Used by ``status``/tests."""
    path = paths.audit_log_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries
