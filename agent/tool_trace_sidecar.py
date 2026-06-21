"""Tool-timing sidecar.

Optional, env-gated JSONL writer that captures REAL per-tool wall latency
for evaluation/benchmarking pipelines (see ~/.hermes/scripts/trace-test-prompt.sh).

Why: `state.db.messages.timestamp` is persistence-time (assigned in bulk at
end-of-turn), so it can't be used to reconstruct per-tool latency. The
`tool_executor` *does* measure `tool_duration` (time.time() before/after the
tool call) but never persists it. This module is a thin opt-in shim that
appends each tool's measured duration to a JSONL sidecar.

Enable:
    export HERMES_TOOL_TRACE=1
    # optional custom dir (default: ~/.hermes/tool-trace/)
    export HERMES_TOOL_TRACE_DIR=/some/path

Output:
    <dir>/<session_id>.jsonl, one JSON object per tool result:
        {"ts": 1780331669.221, "session_id": "...", "tool_call_id": "...",
         "tool_name": "read_file", "duration_s": 0.234,
         "is_error": false, "result_size_bytes": 5998}

The trace script merges this back into the message timeline.
Never raises. Failures are silently swallowed so this can never break tool
execution.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any


def _enabled() -> bool:
    return os.environ.get("HERMES_TOOL_TRACE", "").strip() not in ("", "0", "false", "False")


def _trace_path(session_id: str) -> Path:
    base = os.environ.get("HERMES_TOOL_TRACE_DIR") or os.path.expanduser("~/.hermes/tool-trace")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in (session_id or "no-session"))
    return p / f"{safe}.jsonl"


def record_tool_timing(
    session_id: str | None,
    tool_call_id: str | None,
    tool_name: str | None,
    duration_s: float | None,
    is_error: bool = False,
    result: Any = None,
) -> None:
    """Append one tool-timing entry to the per-session JSONL sidecar.

    Silently no-ops when HERMES_TOOL_TRACE is unset. Never raises.
    """
    if not _enabled():
        return
    try:
        try:
            size = len(result) if result is not None and hasattr(result, "__len__") else 0
        except Exception:
            size = 0
        entry = {
            "ts": time.time(),
            "session_id": session_id or "",
            "tool_call_id": tool_call_id or "",
            "tool_name": tool_name or "",
            "duration_s": float(duration_s) if duration_s is not None else None,
            "is_error": bool(is_error),
            "result_size_bytes": int(size),
        }
        path = _trace_path(session_id or "no-session")
        # Append; one-line-per-record JSONL. Open per-call so concurrent tools
        # in the concurrent path don't fight over a shared file handle.
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Never let evaluation tracing break tool execution.
        pass
