"""Proof completion contracts: ``proof:<command>`` gates ``done`` on an exit code.

A card created with ``completion_contract="proof:<command>"`` cannot reach
``done`` until that command exits 0 in the card's workspace. It is the
local-work counterpart of the PR acceptance gate (``kanban_pr_acceptance``):
the worker's summary is never the evidence, the command's exit status is.
Nothing parses prose, and no model judgment is involved.

The command runs outside SQLite transactions (same shape as PR acceptance),
from the card's workspace directory, with a credential-scrubbed environment,
a wall-clock cap, and bounded, redacted output captured onto a durable
``proof_acceptance`` event so the next worker (or a human) can read why the
completion was refused.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from utils import env_int

PROOF_PREFIX = "proof:"
PROOF_MAX_CHARS = 2000
OUTPUT_TAIL_CHARS = 2000
DEFAULT_TIMEOUT_SECONDS = 120
_KILL_GRACE_SECONDS = 5
_DETAIL_EXCERPT_CHARS = 300
_RECOVERY = (
    "Fix the work until the proof command exits 0, then retry completion. "
    "Use kanban_block if the proof itself is wrong or needs human input; "
    "receipts remain on the task event log."
)


def is_proof_contract(contract: Optional[str]) -> bool:
    return isinstance(contract, str) and contract.startswith(PROOF_PREFIX)


def validate_proof_contract(value: str) -> str:
    """Normalise ``proof:<command>``; the command is stripped and bounded."""
    command = value[len(PROOF_PREFIX):].strip()
    if not command:
        raise ValueError("proof: completion_contract needs a command after the prefix")
    if "\x00" in command:
        raise ValueError("proof: command cannot contain NUL bytes")
    if len(command) > PROOF_MAX_CHARS:
        raise ValueError(f"proof: command exceeds {PROOF_MAX_CHARS} characters")
    return PROOF_PREFIX + command


def proof_command(contract: str) -> str:
    return contract[len(PROOF_PREFIX):]


def proof_timeout_seconds() -> int:
    """Wall-clock cap for one proof run; ``HERMES_KANBAN_PROOF_TIMEOUT`` overrides."""
    return max(1, env_int("HERMES_KANBAN_PROOF_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))


def failure_detail(receipt: dict) -> str:
    """One line for ``last_failure_error``: classification, detail, and a short
    excerpt of the command's output so the worker sees *why* without opening
    the event log."""
    excerpt = (receipt.get("stderr_tail") or receipt.get("stdout_tail") or "").strip()
    if len(excerpt) > _DETAIL_EXCERPT_CHARS:
        excerpt = "…" + excerpt[-_DETAIL_EXCERPT_CHARS:]
    line = f"Proof {receipt['classification']}: {receipt.get('detail', '')}".rstrip()
    if excerpt:
        line += f" Output: {excerpt}"
    return f"{line} {receipt['recovery']}"


def collect_proof(
    contract: str, *, task_id: str, workspace_kind: Optional[str],
    workspace_path: Optional[str], timeout: Optional[int] = None,
) -> dict:
    """Run the card's proof command in its workspace and return a receipt.

    ``ok`` is True only on exit 0. Every other outcome keeps the card
    in-flight with a classification: ``failure`` (non-zero exit), ``timeout``,
    ``infra`` (the command could not start) or ``workspace_missing`` (the card
    has no existing absolute workspace directory to run in).
    """
    command = proof_command(contract)
    receipt: dict = {
        "ok": False, "gate": "proof", "classification": "missing", "command": command,
        "cwd": None, "exit_code": None, "stdout_tail": "", "stderr_tail": "",
        "duration_ms": 0, "recovery": _RECOVERY,
    }
    cwd = _workspace_dir(workspace_kind, workspace_path)
    if cwd is None:
        receipt.update(
            classification="workspace_missing",
            detail=("The card has no existing workspace directory to run the proof in; "
                    "proof contracts need a claimed scratch, dir: or worktree workspace."),
        )
        return receipt
    receipt["cwd"] = str(cwd)
    budget = timeout if timeout is not None else proof_timeout_seconds()
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=str(cwd), env=_proof_env(task_id, cwd),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            **({"start_new_session": True} if os.name == "posix" else {}),
        )
    except OSError as exc:
        receipt.update(classification="infra",
                       detail=f"Proof command could not start ({exc.__class__.__name__}).")
        return receipt
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=budget)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        stdout, stderr = _drain(proc)
    receipt.update(
        duration_ms=int((time.monotonic() - started) * 1000),
        stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
    )
    if timed_out:
        receipt.update(classification="timeout",
                       detail=f"Proof command exceeded {budget}s (HERMES_KANBAN_PROOF_TIMEOUT).")
        return receipt
    receipt["exit_code"] = proc.returncode
    if proc.returncode == 0:
        receipt.update(ok=True, classification="success")
    else:
        receipt.update(classification="failure",
                       detail=f"Proof command exited {proc.returncode}.")
    return receipt


def _workspace_dir(workspace_kind: Optional[str], workspace_path: Optional[str]) -> Optional[Path]:
    """The directory a proof runs in: the card's persisted workspace path when it
    is absolute and exists. Never guesses (a relative path would resolve
    against whatever CWD the completing process has)."""
    if not workspace_path:
        return None
    path = Path(workspace_path).expanduser()
    if not path.is_absolute() or not path.is_dir():
        return None
    return path


def _proof_env(task_id: str, cwd: Path) -> dict[str, str]:
    """Credential-scrubbed child env plus the two values a proof can key on."""
    from tools.environments.local import hermes_subprocess_env
    env = hermes_subprocess_env(inherit_credentials=False)
    env["HERMES_KANBAN_TASK"] = task_id
    env["HERMES_KANBAN_WORKSPACE"] = str(cwd)
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the shell and everything it spawned (its own session on POSIX)."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _drain(proc: subprocess.Popen) -> tuple[str, str]:
    """Collect whatever output a killed proof left; never hang on a pipe a
    surviving grandchild still holds open."""
    try:
        return proc.communicate(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return "", ""


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _tail(value) -> str:
    text = _as_text(value)
    if len(text) > OUTPUT_TAIL_CHARS:
        text = "…" + text[-OUTPUT_TAIL_CHARS:]
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(text)
