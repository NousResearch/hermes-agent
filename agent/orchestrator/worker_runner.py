"""Worker subprocess runner (real controlled execution).

Replaces the intent-only RUN_WORKER handler with a real subprocess call.
The runner enforces:
- Hard timeout via subprocess.communicate(timeout=...).
- Terminate → wait grace → kill fallback.
- stdout/stderr captured to ~/.hermes/traces/workers/<worker_run_id>.{stdout,stderr}.
- Secret redaction in captured output.
- Append-only worker_runs.jsonl trace.

The runner NEVER imports DecisionEngine. The runner is invoked by the
RUN_WORKER handler only — not by the engine itself.

Per-batch scoping: ``run_worker_subprocess`` reads ``get_active_batch_run_id()``
when set by the RUN_WORKER handler from Dispatcher metadata so the row in
``worker_runs.jsonl`` carries the same ``batch_run_id`` as the batch_trace row.
This avoids the double-counting bug fixed by
HERMES_ORCHESTRATOR_METRICS_BATCH_RUN_ID_SCOPING without coupling BatchRunner
directly to worker execution.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


WORKER_LOG_ROOT = Path("/home/jr-ubuntu/.hermes/traces/workers")
WORKER_RUN_LOG = Path("/home/jr-ubuntu/.hermes/traces/worker_runs.jsonl")

# Process-global active batch_run_id. Set by the RUN_WORKER handler from
# Dispatcher metadata before invoking worker execution. Workers spawned during
# that handler read it and stamp it onto the resulting worker_runs.jsonl row.
# Use a threading.local so concurrent dispatcher invocations on different
# threads do not clobber each other.
_active_batch_run_id_tls = threading.local()


def set_active_batch_run_id(batch_run_id: str | None) -> str | None:
    """Set the active batch_run_id for the current thread; returns the previous value."""
    previous = getattr(_active_batch_run_id_tls, "value", None)
    _active_batch_run_id_tls.value = batch_run_id
    return previous


def get_active_batch_run_id() -> str | None:
    return getattr(_active_batch_run_id_tls, "value", None)


@contextlib.contextmanager
def active_batch_run_id_scope(batch_run_id: str | None):
    """Context manager that temporarily sets the active batch_run_id."""
    previous = set_active_batch_run_id(batch_run_id)
    try:
        yield
    finally:
        set_active_batch_run_id(previous)

_SECRET_PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
)


def _redact_secrets(s):
    """Replace secret patterns with [REDACTED]."""
    for p in _SECRET_PATTERNS:
        s = p.sub("[REDACTED]", s)
    return s


@dataclass
class WorkerRunResult:
    """Output of run_worker_subprocess."""
    worker_run_id: str
    worker_id: str
    task_id: str
    command: list
    exitcode: int | None
    stdout_path: str | None
    stderr_path: str | None
    latency_ms: int
    timed_out: bool
    killed: bool
    error_type: str | None
    error_repr: str | None
    timestamp: str
    # batch_run_id scopes the row to a single BatchRunner.run_batch invocation.
    # New rows must always carry it; legacy rows (None) are tolerated for
    # backwards compatibility with traces written before this field existed.
    batch_run_id: str | None = None

    def to_dict(self):
        return {
            "worker_run_id": self.worker_run_id,
            "worker_id": self.worker_id,
            "task_id": self.task_id,
            "command": list(self.command),
            "exitcode": self.exitcode,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "latency_ms": self.latency_ms,
            "timed_out": self.timed_out,
            "killed": self.killed,
            "error_type": self.error_type,
            "error_repr": self.error_repr,
            "timestamp": self.timestamp,
            "batch_run_id": self.batch_run_id,
        }


def _utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_run_log(result, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def run_worker_subprocess(
    worker_id,
    task_id,
    command,
    *,
    timeout_s=60,
    cwd=None,
    env=None,
    worker_log_root=None,
    worker_run_log=None,
    batch_run_id=None,
):
    """Execute the worker command in a subprocess with timeout + kill fallback.

    batch_run_id is recorded into the worker_runs.jsonl row to scope the
    run to a single BatchRunner.run_batch invocation. When ``batch_run_id``
    is not provided explicitly, the runner falls back to the thread-local
    active value set by the RUN_WORKER handler via ``active_batch_run_id_scope``.
    """
    worker_log_root = worker_log_root or WORKER_LOG_ROOT
    worker_run_log = worker_run_log or WORKER_RUN_LOG

    if batch_run_id is None:
        batch_run_id = get_active_batch_run_id()

    worker_run_id = str(uuid.uuid4())
    timestamp = _utcnow_iso()
    t0 = time.monotonic()

    # Pre-flight: validate command.
    if not command or not isinstance(command, list):
        latency = int((time.monotonic() - t0) * 1000)
        result = WorkerRunResult(
            worker_run_id=worker_run_id,
            worker_id=worker_id,
            task_id=task_id,
            command=list(command) if command else [],
            exitcode=None,
            stdout_path=None,
            stderr_path=None,
            latency_ms=latency,
            timed_out=False,
            killed=False,
            error_type="worker_command_missing",
            error_repr="command is empty or not a list",
            timestamp=timestamp,
            batch_run_id=batch_run_id,
        )
        _append_run_log(result, worker_run_log)
        return result

    # Prepare log paths.
    worker_log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = worker_log_root / f"{worker_run_id}.stdout"
    stderr_path = worker_log_root / f"{worker_run_id}.stderr"

    # Merge env.
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    proc = None
    timed_out = False
    killed = False
    error_type = None
    error_repr = None
    exitcode = None
    stdout = ""
    stderr = ""

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            exitcode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            error_type = "worker_timeout"
            error_repr = f"timeout after {timeout_s}s"
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                killed = True
                proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except Exception as e:
                    stdout = ""
                    stderr = f"kill failed: {e!r}"
            except Exception as e:
                stdout = ""
                stderr = f"terminate-comm failed: {e!r}"
    except FileNotFoundError as e:
        error_type = "worker_command_not_found"
        error_repr = repr(e)
        stdout = ""
        stderr = ""
    except Exception as e:
        error_type = "worker_spawn_failed"
        error_repr = repr(e)
        stdout = ""
        stderr = ""

    latency = int((time.monotonic() - t0) * 1000)

    # Redact secrets before writing.
    if stdout:
        stdout = _redact_secrets(stdout)
    if stderr:
        stderr = _redact_secrets(stderr)

    # Write stdout/stderr files.
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")

    result = WorkerRunResult(
        worker_run_id=worker_run_id,
        worker_id=worker_id,
        task_id=task_id,
        command=list(command),
        exitcode=exitcode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        latency_ms=latency,
        timed_out=timed_out,
        killed=killed,
        error_type=error_type,
        error_repr=error_repr,
        timestamp=timestamp,
        batch_run_id=batch_run_id,
    )
    _append_run_log(result, worker_run_log)
    return result