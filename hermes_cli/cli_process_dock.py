"""Background-process rows for the classic CLI live-work dock.

Sibling of ``cli_subagent_monitor``: the dock paints one "Processes" block under the
subagent rows from the same snapshot the ``process_manage`` tool reads, so a
``terminal(background=true)`` spawn is visible the moment it starts instead of only
when its completion notification lands.
"""
from __future__ import annotations

import time

# A finished process stays on the dock long enough to read its exit line, then
# leaves; the completion notification in the transcript is the durable record.
RETAIN_SECONDS = 60

# Reader-thread exits report "exited"; `kill_process` / `kill_all` stamp "killed".
_STATUS_BY_REASON = {"killed": "killed", "lost": "lost", "failed_start": "failed"}


def _last_output_line(buffer: str) -> str:
    for line in reversed((buffer or "").splitlines()):
        text = " ".join(line.split())
        if text:
            return text
    return ""


def process_status(entry: dict) -> str:
    if entry.get("status") != "exited":
        return "running"
    reason = _STATUS_BY_REASON.get(entry.get("completion_reason") or "")
    if reason:
        return reason
    return "done" if not entry.get("exit_code") else "failed"


def process_rows(now: float | None = None) -> list:
    """Running processes plus those that exited within ``RETAIN_SECONDS``, newest-running first.

    The classic CLI is one conversation per OS process, so every registry entry (its own
    spawns and any a child handed back) belongs to this dock; gateway-hosted surfaces scope
    by session key in ``tui_gateway`` instead.
    """
    from tools.process_registry import process_registry

    now = time.time() if now is None else now
    rows = []
    for entry in process_registry.list_sessions():
        proc = process_registry.get(entry["session_id"])
        if proc is None:
            continue
        status = process_status(entry)
        exited_at = proc.exited_at if status != "running" else 0.0
        if status != "running" and now - exited_at > RETAIN_SECONDS:
            continue
        end = exited_at or now
        rows.append({
            "kind": "process",
            "key": proc.id,
            "id": proc.id,
            "command": " ".join((proc.command or "").split()) or "background process",
            "status": status,
            "exit_code": proc.exit_code,
            "elapsed": max(0, int(end - proc.started_at)),
            "since_exit": max(0, int(now - exited_at)) if exited_at else 0,
            "detail": _last_output_line(proc.output_buffer),
        })
    rows.sort(key=lambda r: (r["status"] != "running", -r["elapsed"] if r["status"] == "running" else r["since_exit"]))
    return rows


_GLYPH = {"running": "⚙", "done": "✔", "failed": "✘", "killed": "✘", "lost": "?"}


def process_activity(row: dict) -> str:
    """The trailing ``· …`` part of a dock line: elapsed + latest output while running, the exit
    verdict + age once finished."""
    if row["status"] == "running":
        return f"{row['elapsed']}s · " + (f"last: {row['detail']}" if row["detail"] else "starting")
    verdict = {"done": f"exit {row['exit_code']}", "failed": f"exit {row['exit_code']}",
               "killed": "killed", "lost": "lost"}[row["status"]]
    return f"{verdict} · {row['since_exit']}s ago"


def process_glyph(row: dict) -> str:
    return _GLYPH.get(row["status"], "?")


def process_tail(process_id: str) -> str:
    from tools.process_registry import process_registry

    proc = process_registry.get(process_id)
    if proc is None:
        return "This process is no longer tracked."
    text = (proc.output_buffer or "")[-32768:]
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    return text or "No output yet."


def kill(process_id: str) -> dict:
    from tools.process_registry import process_registry

    return process_registry.kill_process(process_id)
