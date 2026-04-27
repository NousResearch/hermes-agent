"""Subprocess lifecycle manager for the google_meet bot.

Single active meeting at a time. Stores the running pid + out_dir in a
session-scoped state file under ``$HERMES_HOME/workspace/meetings/.active.json``
so tool calls across turns can find the bot, and ``on_session_end`` can clean
it up.

The bot runs as a detached subprocess — we don't hold file descriptors open,
so the parent agent loop can't block on it. We communicate via files only.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

# File + directory layout (under $HERMES_HOME):
#
#   workspace/meetings/
#       .active.json                # pointer to current session's bot
#       <meeting-id>/
#           status.json             # live bot state (written by bot each tick)
#           transcript.txt          # scraped captions
#
# .active.json holds:
#   {"pid": 12345, "meeting_id": "abc-defg-hij", "out_dir": "...",
#    "url": "https://meet.google.com/...", "started_at": 1714159200.0,
#    "session_id": "optional"}


def _root() -> Path:
    return Path(get_hermes_home()) / "workspace" / "meetings"


def _active_file() -> Path:
    return _root() / ".active.json"


def _read_active() -> Optional[Dict[str, Any]]:
    p = _active_file()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_active(data: Dict[str, Any]) -> None:
    p = _active_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _clear_active() -> None:
    try:
        _active_file().unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — treat as alive.
        return True
    return True


# ---------------------------------------------------------------------------
# Public API — used by tool handlers + CLI
# ---------------------------------------------------------------------------

def start(
    url: str,
    *,
    out_dir: Optional[Path] = None,
    headed: bool = False,
    auth_state: Optional[str] = None,
    guest_name: str = "Hermes Agent",
    duration: Optional[str] = None,
    session_id: Optional[str] = None,
    mode: str = "transcribe",
    realtime_model: Optional[str] = None,
    realtime_voice: Optional[str] = None,
    realtime_instructions: Optional[str] = None,
    realtime_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Spawn the meet_bot subprocess for *url*.

    If a bot is already running for this hermes install, leave it first —
    we enforce single-active-meeting semantics.

    Returns a dict summarizing the started bot.
    """
    from plugins.google_meet.meet_bot import _is_safe_meet_url, _meeting_id_from_url

    if not _is_safe_meet_url(url):
        return {
            "ok": False,
            "error": (
                "refusing: only https://meet.google.com/ URLs are allowed. "
                "got: " + repr(url)
            ),
        }

    existing = _read_active()
    if existing and _pid_alive(int(existing.get("pid", 0))):
        stop(reason="replaced by new meet_join")

    meeting_id = _meeting_id_from_url(url)
    out = out_dir or (_root() / meeting_id)
    out.mkdir(parents=True, exist_ok=True)

    # Wipe any stale transcript/status files from a previous run of this
    # meeting id so polling isn't confused.
    for name in ("transcript.txt", "status.json"):
        f = out / name
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass

    env = os.environ.copy()
    env["HERMES_MEET_URL"] = url
    env["HERMES_MEET_OUT_DIR"] = str(out)
    env["HERMES_MEET_GUEST_NAME"] = guest_name
    if headed:
        env["HERMES_MEET_HEADED"] = "1"
    if auth_state:
        env["HERMES_MEET_AUTH_STATE"] = auth_state
    if duration:
        env["HERMES_MEET_DURATION"] = duration
    # v2: realtime mode + passthroughs. The bot defaults to transcribe
    # mode if HERMES_MEET_MODE isn't set, matching v1 behavior.
    if mode:
        env["HERMES_MEET_MODE"] = mode
    if realtime_model:
        env["HERMES_MEET_REALTIME_MODEL"] = realtime_model
    if realtime_voice:
        env["HERMES_MEET_REALTIME_VOICE"] = realtime_voice
    if realtime_instructions:
        env["HERMES_MEET_REALTIME_INSTRUCTIONS"] = realtime_instructions
    if realtime_api_key:
        env["HERMES_MEET_REALTIME_KEY"] = realtime_api_key

    log_path = out / "bot.log"
    # Detach: stdin=devnull, stdout/stderr → log file, new session so parent
    # signals don't propagate.
    log_fh = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "plugins.google_meet.meet_bot"],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        # The subprocess now owns the log fd; we can close ours.
        log_fh.close()

    record = {
        "pid": proc.pid,
        "meeting_id": meeting_id,
        "out_dir": str(out),
        "url": url,
        "started_at": time.time(),
        "session_id": session_id,
        "log_path": str(log_path),
        "mode": mode,
    }
    _write_active(record)
    return {"ok": True, **record}


def status() -> Dict[str, Any]:
    """Return the current meeting state, or ``{"ok": False, "reason": ...}``."""
    active = _read_active()
    if not active:
        return {"ok": False, "reason": "no active meeting"}

    pid = int(active.get("pid", 0))
    alive = _pid_alive(pid) if pid else False

    status_path = Path(active.get("out_dir", "")) / "status.json"
    bot_status: Dict[str, Any] = {}
    if status_path.is_file():
        try:
            bot_status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "ok": True,
        "alive": alive,
        "pid": pid,
        "meetingId": active.get("meeting_id"),
        "url": active.get("url"),
        "startedAt": active.get("started_at"),
        "outDir": active.get("out_dir"),
        **bot_status,
    }


def transcript(last: Optional[int] = None) -> Dict[str, Any]:
    """Read the current transcript file. Returns ok=False if none exists."""
    active = _read_active()
    if not active:
        return {"ok": False, "reason": "no active meeting"}

    tp = Path(active.get("out_dir", "")) / "transcript.txt"
    if not tp.is_file():
        return {
            "ok": True,
            "meetingId": active.get("meeting_id"),
            "lines": [],
            "total": 0,
            "path": str(tp),
        }
    text = tp.read_text(encoding="utf-8", errors="replace")
    all_lines = [ln for ln in text.splitlines() if ln.strip()]
    lines = all_lines[-last:] if last else all_lines
    return {
        "ok": True,
        "meetingId": active.get("meeting_id"),
        "lines": lines,
        "total": len(all_lines),
        "path": str(tp),
    }


def enqueue_say(text: str) -> Dict[str, Any]:
    """Append a ``say`` request to the active bot's JSONL queue.

    Returns ``{"ok": False, "reason": ...}`` when no meeting is active or
    the active bot is in transcribe-only mode. Otherwise writes a line to
    ``<out_dir>/say_queue.jsonl`` that the bot's realtime speaker thread
    will consume.
    """
    import uuid

    text = (text or "").strip()
    if not text:
        return {"ok": False, "reason": "text is required"}

    active = _read_active()
    if not active:
        return {"ok": False, "reason": "no active meeting"}
    if active.get("mode") != "realtime":
        return {
            "ok": False,
            "reason": (
                "active meeting is in transcribe mode — pass mode='realtime' "
                "to meet_join to enable agent speech"
            ),
        }

    out_dir = Path(active.get("out_dir", ""))
    if not out_dir.is_dir():
        return {"ok": False, "reason": f"out_dir missing: {out_dir}"}

    queue_path = out_dir / "say_queue.jsonl"
    entry = {"id": uuid.uuid4().hex[:12], "text": text}
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {
        "ok": True,
        "meetingId": active.get("meeting_id"),
        "enqueued_id": entry["id"],
        "queue_path": str(queue_path),
    }


def stop(*, reason: str = "requested") -> Dict[str, Any]:
    """Signal the active bot to leave cleanly, then clear the active pointer.

    Sends SIGTERM and waits up to 10s for the bot to exit. Falls back to
    SIGKILL if the bot doesn't respond.
    """
    active = _read_active()
    if not active:
        return {"ok": False, "reason": "no active meeting"}

    pid = int(active.get("pid", 0))
    out_dir = active.get("out_dir")
    transcript_path = Path(out_dir) / "transcript.txt" if out_dir else None

    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.5)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    _clear_active()
    return {
        "ok": True,
        "reason": reason,
        "meetingId": active.get("meeting_id"),
        "transcriptPath": str(transcript_path) if transcript_path else None,
    }
