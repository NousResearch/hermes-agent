"""Copilot subprocess launcher: spawns ``copilot -p`` in print mode.

Uses the GitHub Copilot CLI in non-interactive print mode (``-p``) with
``--allow-all --silent --output-format json``.  The process runs in the
repo working directory, edits files autonomously, and exits when done.

A background monitor thread waits for the process to finish and
transitions the job state accordingly (running → idle on success,
running → failed on error).
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_state import SessionDB
from copilot_jobs.models import RepoEntry

logger = logging.getLogger(__name__)

# Patterns to extract session info from copilot output (JSONL or plain text).
SESSION_ID_PATTERN = re.compile(
    r"session[_\s-]?id[:\s]+([a-zA-Z0-9_-]+)", re.IGNORECASE
)


def parse_copilot_output(output: str) -> Dict[str, Optional[str]]:
    """Parse copilot stdout for session handles.

    Checks JSONL lines first (``--output-format json``), then falls back
    to regex matching on plain text.

    Returns ``{"session_id": ... or None}``.
    """
    result: Dict[str, Optional[str]] = {"session_id": None}

    # Try JSONL parsing — each line may be a JSON object.
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                sid = obj.get("sessionId") or obj.get("session_id")
                if sid:
                    result["session_id"] = str(sid)
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: regex on plain text.
    m = SESSION_ID_PATTERN.search(output)
    if m:
        result["session_id"] = m.group(1)

    return result


def build_copilot_command(
    prompt: str,
    *,
    copilot_bin: str = "copilot",
    model: Optional[str] = None,
    json_output: bool = True,
) -> List[str]:
    """Build the ``copilot -p`` command list.

    Flags used:
      -p <prompt>       non-interactive one-shot
      --allow-all       auto-approve all tool use
      --silent          suppress stats
      --no-auto-update  skip update check
      --no-ask-user     fully autonomous
      --output-format json   JSONL output for parsing
    """
    cmd = [
        copilot_bin,
        "-p", prompt,
        "--allow-all",
        "--silent",
        "--no-auto-update",
        "--no-ask-user",
    ]
    if json_output:
        cmd.extend(["--output-format", "json"])
    if model:
        cmd.extend(["--model", model])
    return cmd


def build_resume_command(
    session_id: str,
    *,
    copilot_bin: str = "copilot",
) -> str:
    """Build a human-readable command to resume or inspect a past session."""
    return f"{copilot_bin} --resume={session_id}"


def _monitor_process(
    proc: subprocess.Popen,
    db: SessionDB,
    job_id: str,
) -> None:
    """Wait for the copilot process to exit and transition the job state.

    Runs in a daemon thread.  Reads all remaining stdout, captures the
    exit code, and transitions: running → idle (exit 0) or
    running → failed (non-zero exit).
    """
    try:
        stdout_data, _ = proc.communicate()
        exit_code = proc.returncode

        parsed = parse_copilot_output(stdout_data or "")
        if parsed.get("session_id"):
            db.update_copilot_job_remote(
                job_id, copilot_session_id=parsed["session_id"]
            )

        payload = json.dumps({
            "exit_code": exit_code,
            "session_id": parsed.get("session_id") or "",
            "output_bytes": len(stdout_data or ""),
        })

        if exit_code == 0:
            db.mark_copilot_job_idle(job_id)
            db.record_copilot_job_event(
                job_id, event_type="process_completed", payload_json=payload
            )
        else:
            db.transition_copilot_job(
                job_id, "failed",
                event_type="process_failed",
                error_text=f"copilot exited with code {exit_code}",
                payload_json=payload,
            )

    except Exception as exc:
        logger.error("Monitor thread error for job %s: %s", job_id, exc)
        try:
            db.transition_copilot_job(
                job_id, "failed",
                event_type="monitor_error",
                error_text=str(exc),
            )
        except Exception:
            pass


def launch_copilot(
    db: SessionDB,
    job_id: str,
    repo: RepoEntry,
    prompt: str,
    *,
    copilot_bin: str = "copilot",
    model: Optional[str] = None,
    dry_run: bool = False,
    _spawn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Launch a ``copilot -p`` session for a job.

    1. Transitions the job from pending → running.
    2. Spawns ``copilot -p <prompt> --allow-all --silent`` in *repo.path*.
    3. Starts a daemon thread to monitor exit and transition the job.
    4. Returns immediately with process details.

    If *dry_run* is True, skips the subprocess and returns placeholders.
    The *_spawn* hook allows tests to inject a fake Popen.

    Returns ``{"pid": int, "cmd": [...], "session_id": str|None}``.
    """
    # Transition to running.
    db.transition_copilot_job(job_id, "running", event_type="launch_started")

    cmd = build_copilot_command(
        prompt, copilot_bin=copilot_bin, model=model
    )

    if dry_run:
        fake_sid = f"dry-run-{uuid.uuid4().hex[:8]}"
        db.update_copilot_job_remote(job_id, copilot_session_id=fake_sid, pid=0)
        return {"pid": 0, "cmd": cmd, "session_id": fake_sid}

    try:
        if _spawn:
            proc = _spawn(cmd, repo.path)
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=repo.path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

        db.update_copilot_job_remote(job_id, pid=proc.pid)
        db.record_copilot_job_event(
            job_id,
            event_type="launched",
            payload_json=json.dumps({"pid": proc.pid, "cmd": cmd}),
        )

        # Start monitor thread.
        monitor = threading.Thread(
            target=_monitor_process,
            args=(proc, db, job_id),
            daemon=True,
            name=f"copilot-monitor-{job_id}",
        )
        monitor.start()

        return {"pid": proc.pid, "cmd": cmd, "session_id": None}

    except Exception as exc:
        logger.error("Failed to launch copilot: %s", exc)
        db.transition_copilot_job(
            job_id, "failed",
            event_type="launch_failed",
            error_text=str(exc),
        )
        raise
