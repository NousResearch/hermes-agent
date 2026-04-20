"""Copilot subprocess launcher: spawns ``copilot -p`` with ``--remote``.

Uses the GitHub Copilot CLI in non-interactive print mode (``-p``) with
``--allow-all --silent --remote --output-format json``.  The process runs
in the repo working directory, edits files autonomously, and exits when
done.  The ``--remote`` flag makes the session connectable from any
authenticated terminal via ``copilot --connect=<session_id>``.
"""

import json
import logging
import re
import subprocess
import uuid
from typing import Any, Callable, Dict, List, Optional

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
      --remote          enable cloud relay for --connect
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
    if json_output:
        cmd.extend(["--output-format", "json"])
    if model:
        cmd.extend(["--model", model])
    return cmd


def launch_copilot(
    repo: RepoEntry,
    prompt: str,
    *,
    copilot_bin: str = "copilot",
    model: Optional[str] = None,
    dry_run: bool = False,
    _spawn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Launch ``copilot -p`` with ``--remote`` for a repo.

    Spawns the process, waits for it to finish, and captures the session
    ID from the JSON output.  The session is connectable from any
    authenticated terminal via ``copilot --connect=<session_id>``.

    If *dry_run* is True, skips the subprocess and returns placeholders.
    The *_spawn* hook allows tests to inject a fake Popen.

    Returns ``{"session_id": str|None, "exit_code": int, "cmd": [...]}``.
    """
    cmd = build_copilot_command(
        prompt, copilot_bin=copilot_bin, model=model
    )

    if dry_run:
        fake_sid = f"dry-run-{uuid.uuid4().hex[:8]}"
        return {"session_id": fake_sid, "exit_code": 0, "cmd": cmd}

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

        stdout_data, _ = proc.communicate()
        exit_code = proc.returncode

        parsed = parse_copilot_output(stdout_data or "")
        session_id = parsed.get("session_id")

        return {
            "session_id": session_id,
            "exit_code": exit_code,
            "cmd": cmd,
        }

    except Exception as exc:
        logger.error("Failed to launch copilot: %s", exc)
        raise
