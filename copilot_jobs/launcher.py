"""Copilot --remote launcher: spawns a Copilot session and captures handles.

Launches `copilot --remote` with the appropriate cwd, parses stdout/stderr
for session_id and remote connection details, and persists them into the
SessionDB copilot_jobs table.
"""

import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional

from hermes_state import SessionDB
from copilot_jobs.models import RepoEntry

logger = logging.getLogger(__name__)

# Patterns to extract session info from copilot --remote output.
# These are best-effort and will need updating if Copilot CLI output changes.
SESSION_ID_PATTERN = re.compile(r"session[_\s-]?id[:\s]+([a-zA-Z0-9_-]+)", re.IGNORECASE)
REMOTE_NAME_PATTERN = re.compile(r"remote[_\s-]?name[:\s]+([a-zA-Z0-9._-]+)", re.IGNORECASE)


def parse_copilot_output(output: str) -> Dict[str, Optional[str]]:
    """Parse copilot --remote stdout/stderr for session handles.

    Returns a dict with 'session_id' and 'remote_name' keys (values may be None).
    """
    result: Dict[str, Optional[str]] = {
        "session_id": None,
        "remote_name": None,
    }

    m = SESSION_ID_PATTERN.search(output)
    if m:
        result["session_id"] = m.group(1)

    m = REMOTE_NAME_PATTERN.search(output)
    if m:
        result["remote_name"] = m.group(1)

    return result


def build_attach_command(
    container_name: str,
    session_id: str,
) -> str:
    """Build the docker exec command for a human to attach to a running session."""
    return f"docker exec -it {container_name} copilot --connect={session_id}"


def launch_copilot_remote(
    db: SessionDB,
    job_id: str,
    repo: RepoEntry,
    prompt: str,
    container_name: str = "ryanwalden-ryanwalden",
    copilot_bin: str = "copilot",
    dry_run: bool = False,
) -> Dict[str, Optional[str]]:
    """Launch a copilot --remote session for a job.

    1. Transitions the job from pending -> running
    2. Spawns `copilot --remote` with cwd = repo.path
    3. Parses stdout for session handles
    4. Persists handles and attach_command into the DB

    If dry_run is True, skips the actual subprocess and returns placeholder values.

    Returns a dict with session_id, remote_name, attach_command, and pid.
    Raises on subprocess failure (job is transitioned to 'failed').
    """
    # Transition to running
    db.transition_copilot_job(job_id, "running", event_type="launch_started")

    if dry_run:
        fake_sid = f"dry-run-{uuid.uuid4().hex[:8]}"
        attach_cmd = build_attach_command(container_name, fake_sid)
        db.update_copilot_job_remote(
            job_id,
            copilot_session_id=fake_sid,
            remote_name="dry-run",
            pid=0,
            attach_command=attach_cmd,
        )
        return {
            "session_id": fake_sid,
            "remote_name": "dry-run",
            "attach_command": attach_cmd,
            "pid": 0,
        }

    cmd = [copilot_bin, "--remote"]
    if prompt:
        cmd.extend(["-q", prompt])

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=repo.path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "COPILOT_HEADLESS": "1"},
        )

        # Read initial output to capture session info
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            # Stop reading after we have enough lines or find a session ID
            if len(output_lines) > 50 or SESSION_ID_PATTERN.search(line):
                break

        full_output = "".join(output_lines)
        parsed = parse_copilot_output(full_output)

        session_id = parsed["session_id"]
        remote_name = parsed["remote_name"]
        attach_cmd = None
        if session_id:
            attach_cmd = build_attach_command(container_name, session_id)

        db.update_copilot_job_remote(
            job_id,
            copilot_session_id=session_id,
            remote_name=remote_name,
            pid=proc.pid,
            attach_command=attach_cmd,
        )

        db.record_copilot_job_event(
            job_id,
            event_type="launched",
            payload_json=f'{{"pid": {proc.pid}, "session_id": "{session_id or ""}"}}',
        )

        return {
            "session_id": session_id,
            "remote_name": remote_name,
            "attach_command": attach_cmd,
            "pid": proc.pid,
        }

    except Exception as exc:
        logger.error("Failed to launch copilot --remote: %s", exc)
        db.transition_copilot_job(
            job_id, "failed",
            event_type="launch_failed",
            error_text=str(exc),
        )
        raise
