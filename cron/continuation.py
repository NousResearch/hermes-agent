"""Job-scoped continuation for background processes launched by a cron run (opt-in, #110650).

A cron execution has no turn left to re-enter, so a finished child's generic completion event
must never be routed from the ambient ``HERMES_SESSION_*`` key — an ambient key can inject a late
child's output into an unrelated chat (see ``cron/scheduler._CronRunScope``, #53027 / #63142).

A job that declares ``background_continuation: true`` opts into a *job-scoped* path instead:

1. the spawning run binds :func:`bind`, which captures the job's own delivery target;
2. ``terminal(background=true, notify_on_complete=true)`` stamps that payload on the child's
   ``ProcessSession`` and leaves the session's ``notify_on_complete`` FALSE, so the shared
   completion queue (keyed on a live session) is never involved;
3. when the child exits, :func:`write_record` persists a durable completion record keyed by job;
4. a later scheduler tick :func:`claim_pending` and runs ONE continuation agent turn for the job,
   delivered through the job's own delivery config.

Absent/false = byte-identical pre-feature behaviour.
"""

import json
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger("cron.continuation")

# Job field that arms this path.
CONTINUATION_FIELD = "background_continuation"
_DIR_NAME = "continuations"
_OUTPUT_TAIL_CHARS = 4000

_bound: ContextVar[Optional[dict]] = ContextVar("hermes_cron_continuation", default=None)


def _dir() -> Path:
    return get_hermes_home() / "cron" / _DIR_NAME


def bind(job: dict, job_id: str, execution_id: Optional[str], target: Optional[dict]) -> Any:
    """Bind the job-scoped continuation context for this run. Returns a reset token.

    Routing is captured from the JOB's own delivery target here, at bind time. Nothing on this
    path ever reads the ambient session env: that fallback is the leak this feature exists to
    avoid.
    """
    thread_id = (target or {}).get("thread_id")
    payload = {
        "job_id": str(job_id or ""),
        "job_name": str(job.get("name") or job.get("prompt") or job_id or "cron job")[:200],
        "execution_id": str(execution_id or ""),
        "platform": str((target or {}).get("platform") or ""),
        "chat_id": str((target or {}).get("chat_id") or ""),
        "thread_id": "" if thread_id is None else str(thread_id),
    }
    return _bound.set(payload)


def unbind(token: Any) -> None:
    if token is not None:
        _bound.reset(token)


def active() -> Optional[dict]:
    """The armed job-scoped payload, or None outside an opted-in cron run."""
    return _bound.get()


def write_record(session) -> Optional[Path]:
    """Persist the completion record for a job-scoped child that just exited.

    Called from ``ProcessRegistry._move_to_finished`` — the record IS the completion event.
    The filename is keyed by process id, so a replayed move overwrites rather than duplicates,
    and :func:`claim_pending` unlinks before running: one continuation per process, ever.
    """
    payload = getattr(session, "cron_continuation", None)
    if not payload:
        return None
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from tools.process_registry import MAX_OUTPUT_CHARS

    with session._lock:
        output = session.output_buffer[-min(_OUTPUT_TAIL_CHARS, MAX_OUTPUT_CHARS):]
    record = {
        **payload,
        "process_id": session.id,
        # Redacted for the same reason the durable receipts are: this file outlives the run.
        "command": redact_sensitive_text(session.command, code_file=True, force=True),
        "exit_code": session.exit_code,
        "completion_reason": session.completion_reason,
        "termination_source": session.termination_source,
        "started_at": session.started_at,
        "output": redact_terminal_output(output, session.command, force=True),
    }
    try:
        directory = _dir()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / f"{payload['job_id']}_{session.id}.json"
        atomic_json_write(path, record, mode=0o600)
        return path
    except OSError:
        logger.warning(
            "Could not persist cron continuation record for %s", session.id, exc_info=True)
        return None


def claim_next() -> Optional[Dict[str, Any]]:
    """Claim the single oldest pending record (read, then unlink) — one continuation, ever.

    One-at-a-time so a crash while running continuation N never deletes records N+1.. that the
    scheduler has not attempted yet; they stay on disk for the next tick.
    """
    try:
        paths = sorted(_dir().glob("*.json"))
    except OSError:
        return None
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            path.unlink()
        except FileNotFoundError:
            continue  # defensive: the tick lock already serializes claimers
        except (OSError, ValueError):
            logger.warning("Dropping unreadable cron continuation record %s", path.name, exc_info=True)
            continue
        if isinstance(record, dict) and record.get("job_id") and record.get("process_id"):
            return record
        # Claimed (unlinked) but malformed: consumed, never retried.
        logger.warning("Dropping malformed cron continuation record %s", path.name)
    return None


def claim_pending() -> List[Dict[str, Any]]:
    """Claim every pending record — retained for drains that run immediately (tests, manual).

    The scheduler tick no longer uses this: it collects one-at-a-time (see ``claim_next``)
    and runs continuations outside the tick lock.
    """
    claimed: List[Dict[str, Any]] = []
    while True:
        record = claim_next()
        if record is None:
            break
        claimed.append(record)
    return claimed


def build_prompt(record: Dict[str, Any]) -> str:
    """The ``## Run Context`` block for the continuation turn (per-fire, never persisted)."""
    exit_code = record.get("exit_code")
    detail = ", ".join(
        str(part) for part in (record.get("completion_reason"), record.get("termination_source"))
        if part)
    output = str(record.get("output") or "").strip() or "(no output captured)"
    receipt = get_hermes_home() / "logs" / "process-results" / f"{record.get('process_id')}.json"
    return (
        "## Background process finished\n\n"
        f"Job '{record.get('job_name')}' (job id `{record['job_id']}`) started a background "
        f"process that has now exited — this is your one-shot continuation turn for it.\n\n"
        f"**Process:** `{record['process_id']}`\n"
        f"**Command:** {record.get('command') or '(unknown)'}\n"
        f"**Exit code:** {exit_code if exit_code is not None else 'unknown'}"
        + (f" ({detail})" if detail else "")
        + f"\n**Started:** {record.get('started_at')}\n"
        f"**Full output (if retained):** {receipt}\n\n"
        "**Output tail:**\n```\n" + output + "\n```\n\n"
        "Act on this result now: inspect it, report the outcome, retry on failure, or schedule the "
        "next dependent step. This turn runs once and is never repeated for this process.\n"
        "Do not start another notifying background process in this turn: a continuation never "
        "spawns a further continuation.\n"
    )
