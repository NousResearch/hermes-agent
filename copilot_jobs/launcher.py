"""Copilot subprocess launcher: spawns ``copilot -p`` with ``--remote``.

Uses the GitHub Copilot CLI in non-interactive print mode (``-p``) with
``--allow-all --silent --remote --output-format json``.  The process runs
in the repo working directory, edits files autonomously, and exits when
done.  The ``--remote`` flag makes the session connectable from any
authenticated terminal via ``copilot --connect=<session_id>``.

The session ID is pre-generated and passed via ``--resume=<uuid>`` so it
is known immediately — no output parsing delay.

When launched for real (not via ``_spawn`` or ``dry_run``), the copilot
process is fully detached (``start_new_session=True``) with stdout/stderr
redirected to a log file.  A shell wrapper runs copilot and then invokes
``complete_job.py`` to update the DB — no daemon thread required, so the
parent ``hermes`` process can exit immediately without killing copilot.
"""

import json
import logging
import re
import shlex
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from copilot_jobs.models import RepoEntry

logger = logging.getLogger(__name__)

# Patterns to extract session info from copilot output (JSONL or plain text).
SESSION_ID_PATTERN = re.compile(
    r"session[_\s-]?id[:\s]+([a-zA-Z0-9_-]+)", re.IGNORECASE
)


def _parse_line_for_session_id(line: str) -> Optional[str]:
    """Try to extract a session ID from a single output line."""
    line = line.strip()
    if not line:
        return None

    # Try JSON first.
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            sid = obj.get("sessionId") or obj.get("session_id")
            if sid:
                return str(sid)
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback.
    m = SESSION_ID_PATTERN.search(line)
    return m.group(1) if m else None


def parse_copilot_output(output: str) -> Dict[str, Optional[str]]:
    """Parse copilot stdout for session handles.

    Checks JSONL lines first (``--output-format json``), then falls back
    to regex matching on plain text.

    Returns ``{"session_id": ... or None}``.
    """
    for line in output.splitlines():
        sid = _parse_line_for_session_id(line)
        if sid:
            return {"session_id": sid}
    return {"session_id": None}


def build_copilot_command(
    prompt: str,
    *,
    copilot_bin: str = "copilot",
    model: Optional[str] = None,
    json_output: bool = True,
    session_id: Optional[str] = None,
) -> List[str]:
    """Build the ``copilot -p`` command list.

    Flags used:
      -p <prompt>       non-interactive one-shot
      --allow-all       auto-approve all tool use
      --silent          suppress stats
      --remote          enable cloud relay for --connect
      --resume=<uuid>   pin session to a pre-generated ID
      --no-auto-update  skip update check
      --no-ask-user     fully autonomous
      --output-format json   JSONL output for parsing
    """
    cmd = [
        copilot_bin,
        "-p", prompt,
        "--allow-all",
        "--silent",
        "--remote",
        "--no-auto-update",
        "--no-ask-user",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    if json_output:
        cmd.extend(["--output-format", "json"])
    if model:
        cmd.extend(["--model", model])
    return cmd


def _log_dir() -> Path:
    """Return (and create) the copilot log directory."""
    d = Path.home() / ".hermes" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def launch_copilot(
    repo: RepoEntry,
    prompt: str,
    *,
    session_id: str,
    copilot_bin: str = "copilot",
    model: Optional[str] = None,
    dry_run: bool = False,
    on_complete: Optional[Callable[[str, int], None]] = None,
    _spawn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Launch ``copilot -p`` with ``--remote`` for a repo.

    *session_id* is the pre-generated UUID used as both the hermes job ID
    and the copilot session (passed via ``--resume=<uuid>``).

    **Real launches** (no ``_spawn``): copilot runs fully detached via a
    shell wrapper that redirects stdout to a log file and calls
    ``complete_job.py`` on exit.  The parent process can exit immediately.

    **Test launches** (``_spawn`` provided): a daemon thread waits for the
    fake process and calls ``on_complete`` so tests can assert on exit
    behaviour synchronously.

    If *dry_run* is True, skips the subprocess and returns placeholders.

    Returns ``{"session_id": str, "cmd": [...], "proc": Popen|None}``.
    """
    cmd = build_copilot_command(
        prompt, copilot_bin=copilot_bin, model=model, session_id=session_id
    )

    if dry_run:
        if on_complete:
            on_complete(session_id, 0)
        return {"session_id": session_id, "exit_code": 0, "cmd": cmd, "proc": None}

    try:
        if _spawn:
            # Test path: use the fake process with a daemon thread.
            proc = _spawn(cmd, repo.path)

            def _wait_and_finish():
                try:
                    proc.stdout.read()
                    proc.wait()
                    if on_complete:
                        on_complete(session_id, proc.returncode)
                except Exception as exc:
                    logger.error("Background wait error: %s", exc)
                    if on_complete:
                        on_complete(session_id, -1)

            waiter = threading.Thread(
                target=_wait_and_finish,
                daemon=True,
                name="copilot-wait",
            )
            waiter.start()
        else:
            # Real path: fully detached process via shell wrapper.
            log_path = _log_dir() / f"copilot-{session_id}.log"
            complete_script = str(
                Path(__file__).resolve().parent / "complete_job.py"
            )
            python_bin = sys.executable

            # Shell command: run copilot, capture its exit code, then
            # update the DB via complete_job.py.
            shell_cmd = (
                f'{shlex.join(cmd)} > {shlex.quote(str(log_path))} 2>&1; '
                f'_ec=$?; '
                f'{shlex.quote(python_bin)} {shlex.quote(complete_script)} '
                f'{shlex.quote(session_id)} $_ec'
            )

            proc = subprocess.Popen(
                ["bash", "-c", shell_cmd],
                cwd=repo.path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        return {
            "session_id": session_id,
            "cmd": cmd,
            "proc": proc,
        }

    except Exception as exc:
        logger.error("Failed to launch copilot: %s", exc)
        raise
