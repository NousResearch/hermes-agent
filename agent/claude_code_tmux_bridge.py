"""Claude Code tmux bridge.

Submits prompts to a running Claude Code session in a named tmux pane,
waits for a unique completion marker, and returns the sanitised response.

Usage::

    from agent.claude_code_tmux_bridge import submit_prompt, BridgeStatus

    result = submit_prompt("claude-momentum", "Refactor utils.py to use dataclasses")
    if result.status == BridgeStatus.SUCCESS:
        print(result.response)

One pre-configured worker is registered at module level:
  session ``claude-momentum``, workspace ``/home/michael/code``.

Additional workers can be registered via ``register_worker()``.

Notes
-----
* Newlines in prompts are collapsed to spaces before sending; tmux
  ``send-keys`` treats each newline as an Enter keystroke.  Single-paragraph
  prompts work best.
* Auth failure is detected only at timeout (not during polling) to avoid
  false positives from unrelated pane history.
* No tmux session is created automatically; the session must already exist.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import re
import subprocess
import threading
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# ── ANSI / credential / auth patterns ────────────────────────────────────────

_ANSI_RE = re.compile(
    r"\x1b\[[0-9;]*[mGKHFABCDEJ]"
    r"|\x1b\][\S\s]*?\x07"
    r"|\x1b[()][AB012]"
    r"|\x1b[=>]"
    r"|\r",
)

# Redact token-shaped values that might appear in captured pane output.
_CREDENTIAL_RE = re.compile(
    r"(?:Bearer\s+|(?:api[-_]?key|token|password|oauth|secret)[\s=:]+)"
    r"[A-Za-z0-9_\-\.]{16,}",
    re.IGNORECASE,
)

_AUTH_FAILURE_RE = re.compile(
    r"(?:"
    r"authentication\s+(?:required|failed|expired|error)"
    r"|oauth\s+(?:\w+\s+)*(?:error|expired|invalid|required|revoked)"
    r"|sign[\s-]?in\s+(?:required|to|with)"
    r"|login\s+(?:required|to|with)"
    r"|please\s+(?:authenticate|log\s*in|sign\s*in)"
    r"|unauthorized"
    r"|session\s+expired"
    r"|credentials?\s+(?:expired|invalid|required)"
    r")",
    re.IGNORECASE,
)


# ── Public types ──────────────────────────────────────────────────────────────

class BridgeStatus(enum.Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    BUSY = "busy"
    AUTH_FAILURE = "auth_failure"
    SESSION_MISSING = "session_missing"
    SESSION_DEAD = "session_dead"
    FAILURE = "failure"


@dataclasses.dataclass
class BridgeResult:
    status: BridgeStatus
    response: str = ""
    error: str = ""


@dataclasses.dataclass
class WorkerConfig:
    session: str
    workspace: str
    timeout: float = 300.0
    poll_interval: float = 2.0


# ── Worker registry ───────────────────────────────────────────────────────────

_WORKERS: dict[str, WorkerConfig] = {}


def register_worker(config: WorkerConfig) -> None:
    """Register a named tmux worker configuration."""
    _WORKERS[config.session] = config


def get_worker(session: str) -> Optional[WorkerConfig]:
    """Return the WorkerConfig for *session*, or None."""
    return _WORKERS.get(session)


# Pre-configured worker for Momentum Studio
register_worker(WorkerConfig(session="claude-momentum", workspace="/home/michael/code"))


# ── Per-session locks ─────────────────────────────────────────────────────────

_session_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_session_lock(session: str) -> threading.Lock:
    with _registry_lock:
        if session not in _session_locks:
            _session_locks[session] = threading.Lock()
        return _session_locks[session]


# ── Low-level tmux helpers ────────────────────────────────────────────────────

def _run_tmux(*args: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def _session_exists(session: str) -> bool:
    result = _run_tmux("has-session", "-t", session)
    return result.returncode == 0


def _pane_alive(session: str) -> bool:
    result = _run_tmux("list-panes", "-t", session, "-F", "#{pane_id}")
    return result.returncode == 0 and bool(result.stdout.strip())


def _send_keys(session: str, text: str) -> bool:
    """Send *text* literally to *session* then press Enter."""
    r = _run_tmux("send-keys", "-t", session, "-l", text)
    if r.returncode != 0:
        return False
    return _run_tmux("send-keys", "-t", session, "Enter").returncode == 0


def _capture_pane(session: str, history_lines: int = 5000) -> str:
    result = _run_tmux(
        "capture-pane", "-p",
        "-t", session,
        "-S", f"-{history_lines}",
        timeout=15.0,
    )
    return result.stdout if result.returncode == 0 else ""


# ── Output sanitisation ───────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _redact_credentials(text: str) -> str:
    return _CREDENTIAL_RE.sub("[REDACTED]", text)


def _is_auth_failure(text: str) -> bool:
    return bool(_AUTH_FAILURE_RE.search(_strip_ansi(text)))


def _extract_response(pre_line_count: int, post_raw: str, done_marker: str) -> Optional[str]:
    """Extract response lines from pane content captured after sending.

    Returns lines from *pre_line_count* up to (not including) the done
    marker line, sanitised; or None if the marker is absent.
    """
    clean_lines = _strip_ansi(post_raw).splitlines()

    done_idx = None
    for i, line in enumerate(clean_lines):
        if done_marker in line:
            done_idx = i
            break

    if done_idx is None:
        return None

    response_lines = clean_lines[pre_line_count:done_idx]
    return _redact_credentials("\n".join(response_lines).strip())


# ── Public API ────────────────────────────────────────────────────────────────

def submit_prompt(
    session: str,
    prompt: str,
    *,
    timeout: Optional[float] = None,
    poll_interval: Optional[float] = None,
) -> BridgeResult:
    """Submit *prompt* to the Claude Code session *session*.

    Parameters
    ----------
    session:
        Name of the tmux session running Claude Code.
    prompt:
        Prompt text.  Newlines are collapsed to spaces before sending.
    timeout:
        Seconds to wait for a response.  Falls back to the worker's
        configured timeout (default 300 s).
    poll_interval:
        Seconds between pane captures.  Falls back to the worker's
        configured poll interval (default 2 s).

    Returns
    -------
    BridgeResult with status one of:

    SUCCESS         Response captured successfully.
    TIMEOUT         Completion marker not seen before deadline.
    BUSY            Another call already holds the per-session lock.
    AUTH_FAILURE    Pane shows authentication or OAuth error at timeout.
    SESSION_MISSING Named tmux session does not exist.
    SESSION_DEAD    Session exists but has no responsive pane.
    FAILURE         Subprocess error or unexpected condition.
    """
    worker = _WORKERS.get(session) or WorkerConfig(session=session, workspace="")
    eff_timeout = timeout if timeout is not None else worker.timeout
    eff_poll = poll_interval if poll_interval is not None else worker.poll_interval

    # Verify session exists
    try:
        if not _session_exists(session):
            return BridgeResult(
                status=BridgeStatus.SESSION_MISSING,
                error=f"tmux session {session!r} not found",
            )
    except subprocess.TimeoutExpired:
        return BridgeResult(status=BridgeStatus.FAILURE, error="has-session timed out")
    except FileNotFoundError:
        return BridgeResult(status=BridgeStatus.FAILURE, error="tmux not found on PATH")
    except OSError as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

    # Verify pane is alive
    try:
        if not _pane_alive(session):
            return BridgeResult(
                status=BridgeStatus.SESSION_DEAD,
                error=f"tmux session {session!r} has no live pane",
            )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

    # Acquire per-session lock (non-blocking — BUSY if already held)
    lock = _get_session_lock(session)
    if not lock.acquire(blocking=False):
        return BridgeResult(
            status=BridgeStatus.BUSY,
            error=f"Session {session!r} is already handling a prompt",
        )
    try:
        return _run_prompt(
            session=session,
            prompt=prompt,
            timeout=eff_timeout,
            poll_interval=eff_poll,
        )
    finally:
        lock.release()


def _run_prompt(
    *,
    session: str,
    prompt: str,
    timeout: float,
    poll_interval: float,
) -> BridgeResult:
    run_id = uuid.uuid4().hex
    done_marker = f"HERMES_BRIDGE_DONE_{run_id}"

    # Collapse newlines — tmux send-keys treats each \n as Enter
    flat_prompt = " ".join(prompt.splitlines())
    instrumented = (
        f"{flat_prompt} "
        f"(When your response is complete, output exactly on its own line: {done_marker})"
    )

    # Snapshot pane before sending so we can slice out only new content
    try:
        pre_raw = _capture_pane(session)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))
    pre_line_count = len(_strip_ansi(pre_raw).splitlines())

    logger.debug(
        "bridge: sending to %r run_id=%s pre_lines=%d",
        session, run_id, pre_line_count,
    )

    try:
        if not _send_keys(session, instrumented):
            return BridgeResult(status=BridgeStatus.FAILURE, error="tmux send-keys failed")
    except (subprocess.TimeoutExpired, OSError) as exc:
        return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

    # Poll for completion marker
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        try:
            post_raw = _capture_pane(session)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return BridgeResult(status=BridgeStatus.FAILURE, error=str(exc))

        response = _extract_response(pre_line_count, post_raw, done_marker)
        if response is not None:
            logger.debug("bridge: done run_id=%s chars=%d", run_id, len(response))
            return BridgeResult(status=BridgeStatus.SUCCESS, response=response)

    # Timeout — check whether pane shows an auth error
    try:
        final_raw = _capture_pane(session)
    except (subprocess.TimeoutExpired, OSError):
        final_raw = ""

    new_content = "\n".join(_strip_ansi(final_raw).splitlines()[pre_line_count:])
    if _is_auth_failure(new_content):
        return BridgeResult(
            status=BridgeStatus.AUTH_FAILURE,
            error="Authentication or OAuth failure detected in pane output",
        )

    return BridgeResult(
        status=BridgeStatus.TIMEOUT,
        error=f"Completion marker not seen within {timeout:.0f}s",
    )
