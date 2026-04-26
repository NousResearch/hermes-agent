"""Copilot subprocess launcher: spawns ``copilot -i`` with ``--remote``.

Uses the GitHub Copilot CLI in interactive mode (``-i <prompt>``) with
``--allow-all --remote``. Interactive mode is required because
``--remote`` (and therefore ``copilot --connect=<task-id>``) only works
against a persistent session — the non-interactive ``-p`` flag exits as
soon as the prompt completes and never registers with the cloud relay.

Because interactive mode renders a TUI, copilot is wrapped in
``script -qfc`` to allocate a PTY, with stdout/stderr captured to a log
file. Hermes keeps its own pre-generated job ID for bookkeeping, but it
does not force that UUID into Copilot via ``--resume``. Recent Copilot
CLI builds treat ``--resume`` as a resume path where startup prompts do
not auto-run, which would make ``/copilot_remote launch <prompt>`` open a
remote session without executing the requested work.

When launched for real (not via ``_spawn`` or ``dry_run``), the wrapper
is fully detached (``start_new_session=True``). A shell wrapper runs
copilot and then invokes ``complete_job.py`` to update the DB — no
daemon thread required, so the parent ``hermes`` process can exit
immediately without killing copilot.
"""

import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from copilot_remote.models import RepoEntry
from copilot_remote.router import _sanitize_for_log

logger = logging.getLogger(__name__)

_DEFAULT_COPILOT_PATHS = [
    "/usr/local/share/npm-global/bin/copilot",
    "/usr/local/bin/copilot",
    "/usr/bin/copilot",
]
_DEFAULT_COPILOT_API_URL = "https://api.githubcopilot.com"
_REMOTE_PROMPT_CHECK_TIMEOUT = 3.0
_REMOTE_PROMPT_POLL_INTERVAL = 0.25
_REMOTE_API_TIMEOUT = 10.0

# Patterns to extract session info from copilot output (JSONL or plain text).
SESSION_ID_PATTERN = re.compile(
    r"session[_\s-]?id[:\s]+([a-zA-Z0-9_-]+)", re.IGNORECASE
)
REMOTE_TASK_ID_PATTERN = re.compile(
    r"Remote session active \(steerable\): .*?/tasks/([0-9a-f-]+)",
    re.IGNORECASE,
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

    Inspects each output line and tries to parse a session ID from JSON
    content when present, falling back to regex matching on plain text.
    ``build_copilot_command()`` does not pass ``--output-format json``
    (interactive ``--remote`` does not support it), so in practice the
    regex path is what fires — the JSON branch only matters if a
    future Copilot release prints structured lines unprompted.

    Returns ``{"session_id": ... or None}``.
    """
    for line in output.splitlines():
        sid = _parse_line_for_session_id(line)
        if sid:
            return {"session_id": sid}
    return {"session_id": None}


def _parse_remote_task_id(
    log_text: str,
    requested_session_id: Optional[str] = None,
) -> Optional[str]:
    """Extract the exported remote task ID for a launched Copilot session."""
    if requested_session_id and requested_session_id not in log_text:
        return None

    match = REMOTE_TASK_ID_PATTERN.search(log_text)
    return match.group(1) if match else None


def _snapshot_process_logs() -> Dict[Path, int]:
    """Capture current Copilot process log sizes before launching."""
    logs_dir = Path.home() / ".copilot" / "logs"
    snapshot: Dict[Path, int] = {}

    for path in logs_dir.glob("process-*.log"):
        try:
            snapshot[path] = path.stat().st_size
        except OSError:
            continue

    return snapshot


def _resolve_copilot_bin(copilot_bin: str) -> str:
    """Resolve the Copilot executable path with sensible fallbacks."""
    resolved = shutil.which(copilot_bin)
    if resolved:
        return resolved

    if os.path.sep in copilot_bin:
        return copilot_bin

    for candidate in _DEFAULT_COPILOT_PATHS:
        if Path(candidate).exists():
            return candidate

    return copilot_bin


def _wait_for_remote_task_id(
    requested_session_id: Optional[str] = None,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
    prior_logs: Optional[Dict[Path, int]] = None,
) -> Optional[str]:
    """Poll Copilot process logs for the exported remote task ID."""
    logs_dir = Path.home() / ".copilot" / "logs"
    deadline = time.time() + timeout
    prior_logs = prior_logs or {}

    while time.time() < deadline:
        # Snapshot (path, mtime) up-front with try/except so a log rotated or
        # deleted between glob() and stat() doesn't crash the launch flow.
        log_paths_with_mtime = []
        for path in logs_dir.glob("process-*.log"):
            try:
                log_paths_with_mtime.append((path, path.stat().st_mtime))
            except OSError:
                continue

        for path, _mtime in sorted(log_paths_with_mtime, key=lambda item: item[1], reverse=True):
            try:
                previous_size = prior_logs.get(path)
                current_size = path.stat().st_size
                if previous_size is not None and current_size <= previous_size:
                    continue

                if previous_size is None:
                    log_text = path.read_text(encoding="utf-8", errors="ignore")
                else:
                    log_text = path.read_bytes()[previous_size:].decode("utf-8", errors="ignore")

                task_id = _parse_remote_task_id(
                    log_text,
                    requested_session_id,
                )
            except OSError:
                continue
            if task_id:
                return task_id
        time.sleep(poll_interval)

    return None


def build_copilot_command(
    prompt: str,
    *,
    copilot_bin: str = "copilot",
    model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[str]:
    """Build the ``copilot -i`` command list.

    Flags used:
      -i <prompt>       interactive mode, auto-execute prompt (persists)
      --allow-all       auto-approve all tool use
      --remote          enable cloud relay for --connect
      --resume=<uuid>   pin session to a pre-generated ID
      --no-auto-update  skip update check
      --no-ask-user     fully autonomous

    Note: ``--silent`` and ``--output-format json`` are intentionally
    omitted because they conflict with the interactive TUI required by
    ``--remote``/``--connect``.
    """
    cmd = [
        copilot_bin,
        "-i", prompt,
        "--allow-all",
        "--remote",
        "--no-auto-update",
        "--no-ask-user",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    if model:
        cmd.extend(["--model", model])
    return cmd


def _log_dir() -> Path:
    """Return (and create) the copilot log directory."""
    d = Path.home() / ".hermes" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _copilot_api_url() -> str:
    """Return the Copilot API base URL used for remote task steering."""
    return os.environ.get("COPILOT_API_URL", _DEFAULT_COPILOT_API_URL).rstrip("/")


def _resolve_github_auth_token() -> str:
    """Resolve a GitHub auth token without exposing it in logs."""
    for env_var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            return token

    gh_bin = shutil.which("gh")
    if not gh_bin:
        raise RuntimeError(
            "GitHub CLI is not available to steer remote Copilot tasks"
        )

    result = subprocess.run(
        [gh_bin, "auth", "token"],
        check=False,
        capture_output=True,
        text=True,
    )
    token = result.stdout.strip()
    if result.returncode != 0 or not token:
        raise RuntimeError(
            "Unable to resolve a GitHub auth token for remote Copilot steering"
        )
    return token


def _copilot_api_request(
    method: str,
    path: str,
    *,
    token: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = _REMOTE_API_TIMEOUT,
) -> Dict[str, Any]:
    """Call the Copilot Mission Control API and return a decoded JSON body."""
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    integration_id = os.environ.get("GITHUB_COPILOT_INTEGRATION_ID")
    if integration_id:
        headers["Copilot-Integration-Id"] = integration_id

    request = Request(
        url=f"{_copilot_api_url()}{path}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="ignore")
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Copilot API {method} {path} failed: {exc.code} {response_body[:200]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Copilot API {method} {path} failed: {exc.reason}"
        ) from exc

    if not raw_body:
        return {}

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Copilot API {method} {path} returned invalid JSON"
        ) from exc


def _list_remote_task_events(task_id: str, token: str) -> List[Dict[str, Any]]:
    """Fetch Mission Control events for a remote task."""
    response = _copilot_api_request("GET", f"/agents/tasks/{task_id}/events", token=token)
    events = response.get("events")
    if not isinstance(events, list):
        raise RuntimeError(
            f"Copilot task events response for {task_id} did not include an events list"
        )
    return [event for event in events if isinstance(event, dict)]


def _remote_task_has_user_message(events: List[Dict[str, Any]]) -> bool:
    """Return True when the task history already contains a user message."""
    return any(event.get("type") == "user.message" for event in events)


def _steer_remote_task(task_id: str, prompt: str, token: str) -> None:
    """Submit the initial prompt directly to an existing remote task."""
    _copilot_api_request(
        "POST",
        f"/agents/tasks/{task_id}/steer",
        token=token,
        payload={"content": prompt, "type": "user_message"},
    )


def _ensure_initial_prompt_delivered(
    task_id: str,
    prompt: str,
    *,
    check_timeout: float = _REMOTE_PROMPT_CHECK_TIMEOUT,
    poll_interval: float = _REMOTE_PROMPT_POLL_INTERVAL,
) -> str:
    """Verify the launch prompt reached the task, steering it if needed."""
    token = _resolve_github_auth_token()
    deadline = time.time() + check_timeout

    while True:
        events = _list_remote_task_events(task_id, token)
        if _remote_task_has_user_message(events):
            return "already-submitted"

        if time.time() >= deadline:
            break

        time.sleep(poll_interval)

    _steer_remote_task(task_id, prompt, token)
    return "steered"


def _attempt_initial_prompt_delivery(
    task_id: Optional[str],
    prompt: str,
) -> Dict[str, Optional[str]]:
    """Best-effort prompt delivery that returns status plus user-facing warnings."""
    if not task_id:
        return {
            "status": "unverified",
            "warning": (
                "Hermes could not determine the remote task ID, so it could not verify "
                "that the launch prompt was delivered."
            ),
        }

    try:
        status = _ensure_initial_prompt_delivered(task_id, prompt)
    except Exception as exc:
        return {
            "status": "unverified",
            "warning": (
                f"Hermes could not verify or steer the launch prompt for task {task_id}: {exc}"
            ),
        }

    return {"status": status, "warning": None}


def _terminate_process_group(proc: Any) -> None:
    """Best-effort cleanup for a detached launch that cannot be steered."""
    pid = getattr(proc, "pid", None)
    if not pid:
        return

    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        return


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
    """Launch ``copilot -i`` with ``--remote`` for a repo.

    *session_id* is the hermes job ID used for DB tracking and log naming.
    Copilot gets its own fresh session so the startup prompt actually runs.

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
        prompt,
        copilot_bin=_resolve_copilot_bin(copilot_bin),
        model=model,
    )

    if dry_run:
        if on_complete:
            on_complete(session_id, 0)
        return {
            "session_id": session_id,
            "exit_code": 0,
            "cmd": cmd,
            "proc": None,
            "prompt_delivery_status": None,
            "prompt_delivery_warning": None,
        }

    try:
        connect_id = None
        prompt_delivery = {"status": None, "warning": None}

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
            # Interactive mode (-i) needs a PTY for its TUI to render and
            # for --remote to register with the cloud relay, so wrap with
            # ``script -qfc`` which allocates a PTY and captures output.
            prior_logs = _snapshot_process_logs()
            log_path = _log_dir() / f"copilot-{session_id}.log"
            complete_script = str(
                Path(__file__).resolve().parent / "complete_job.py"
            )
            python_bin = sys.executable

            script_inner = shlex.join(cmd)
            script_cmd = [
                "script",
                "-eqfc",
                script_inner,
                str(log_path),
            ]

            # Shell command: run copilot under script(1), capture exit
            # code, then update the DB via complete_job.py.
            shell_cmd = (
                f'{shlex.join(script_cmd)} > /dev/null 2>&1; '
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

            connect_id = _wait_for_remote_task_id(prior_logs=prior_logs)
            prompt_delivery = _attempt_initial_prompt_delivery(connect_id, prompt)
            if prompt_delivery["status"]:
                logger.info(
                    "Initial prompt delivery for task %s: %s",
                    connect_id or session_id,
                    prompt_delivery["status"],
                )
            if prompt_delivery["warning"]:
                logger.warning(_sanitize_for_log(prompt_delivery["warning"]))

        return {
            "session_id": session_id,
            "connect_id": connect_id if not _spawn else None,
            "cmd": cmd,
            "proc": proc,
            "prompt_delivery_status": prompt_delivery["status"],
            "prompt_delivery_warning": prompt_delivery["warning"],
        }

    except Exception as exc:
        if not dry_run and not _spawn and "proc" in locals():
            _terminate_process_group(proc)
        logger.error("Failed to launch copilot: %s", exc)
        raise
