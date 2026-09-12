"""Monitor-mode cron support — hash-suppressed change detection.

A monitor job attaches a cheap source (``monitor_script`` / ``monitor_url``) to an LLM cron job.
Each tick runs the source FIRST and hashes its EXACT output bytes (no timestamp/whitespace
normalization — scripts must emit stable output) against the hash from the last agent-triggering
tick: unchanged → agent run suppressed (silent ``no_change`` run); changed/first run → a "MONITOR
CHANGE DETECTED" block (capped unified diff + new output) is injected into the prompt; source
failure → an ERROR, never a change, and the stored hash is left untouched. State:
``job["monitor_state"]`` in jobs.json (hash + last_changed_at) and
``OUTPUT_DIR/<job_id>/monitor_last_output.txt`` (for the diff).

The default commit boundary is detection time. Jobs with
``monitor_commit_policy="after_delivery"`` defer the hash/snapshot commit until the agent finishes
AND delivery succeeds, so a failed run re-offers the same observation instead of consuming it. Such
a job may return the exact marker ``[MONITOR_RETRY]`` to suppress delivery and leave the prior hash
intact.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Prompt-injection caps: unified diff, and new-output block (mirrors the 8k context_from truncation
# in cron/scheduler.py). Then bounded-GET limits for monitor_url sources.
MAX_DIFF_CHARS = 4000
MAX_OUTPUT_CHARS = 8000
URL_TIMEOUT_SECONDS = 30
MAX_URL_BYTES = 262_144  # 256 KiB

_SNAPSHOT_FILENAME = "monitor_last_output.txt"


@dataclass
class MonitorOutcome:
    """Result of one monitor-source evaluation."""

    ok: bool
    changed: bool = False
    first_run: bool = False
    context_block: Optional[str] = None
    error: Optional[str] = None
    new_hash: Optional[str] = None
    output: Optional[str] = None
    pending_token: Optional[str] = None


def hash_monitor_output(output: str) -> str:
    """Hash the monitor output as exact UTF-8 bytes (no normalization)."""
    return hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()


def build_monitor_diff(old: str, new: str) -> str:
    """Unified diff of old vs new monitor output, capped at MAX_DIFF_CHARS."""
    diff = "\n".join(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), fromfile="previous", tofile="current", lineterm="",
        )
    )
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated]"
    return diff


def _snapshot_path(job_id: str):
    from cron.jobs import _job_output_dir

    return _job_output_dir(job_id) / _SNAPSHOT_FILENAME


def _read_last_output(job_id: str) -> str:
    try:
        path = _snapshot_path(job_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Monitor: failed to read last output for %r: %s", job_id, exc)
    return ""


def _write_last_output_strict(job_id: str, output: str) -> None:
    path = _snapshot_path(job_id)
    from cron.jobs import _ensure_cron_dir

    _ensure_cron_dir(path.parent)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(output, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_last_output(job_id: str, output: str) -> None:
    try:
        _write_last_output_strict(job_id, output)
    except Exception as exc:
        logger.warning("Monitor: failed to persist last output for %r: %s", job_id, exc)


def _fetch_monitor_url(url: str) -> tuple[bool, str]:
    """Bounded GET of a monitor URL. Returns (ok, body-or-error)."""
    import urllib.request

    if not str(url).lower().startswith(("http://", "https://")):
        return False, f"monitor_url must be http(s): {url!r}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-cron-monitor"})
        with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as resp:  # nosec B310 — scheme checked above
            body = resp.read(MAX_URL_BYTES + 1)
        return True, body[:MAX_URL_BYTES].decode("utf-8", errors="replace")
    except Exception as exc:
        return False, f"monitor_url fetch failed: {exc}"


def _field(job: dict, key: str) -> str:
    return (job.get(key) or "").strip()


def _run_monitor_source(job: dict) -> tuple[bool, str]:
    """Run the job's monitor source (script or URL). Returns (ok, output)."""
    monitor_script = _field(job, "monitor_script")
    if monitor_script:
        # Same containment + interpreter rules as the existing `script` field.
        from cron.scheduler_script import _run_job_script

        return _run_job_script(monitor_script, workdir=_field(job, "workdir") or None)
    monitor_url = _field(job, "monitor_url")
    if monitor_url:
        return _fetch_monitor_url(monitor_url)
    return False, "monitor job has neither monitor_script nor monitor_url"


def job_has_monitor(job: dict) -> bool:
    return bool(_field(job, "monitor_script") or _field(job, "monitor_url"))


def check_monitor(job: dict) -> MonitorOutcome:
    """Run the monitor source and decide whether the agent should run.

    By default, on change (or first run) the new hash + snapshot are persisted BEFORE the agent
    runs — detection time is the state boundary, so a failed agent run doesn't re-alert on the same
    content forever. ``monitor_commit_policy='after_delivery'`` instead returns the new hash/output
    to the scheduler for a downstream commit once the agent AND delivery both succeeded.
    On failure nothing is persisted.
    """
    job_id = str(job.get("id") or "")
    pending = None
    if job.get("monitor_commit_policy") == "safe_retry":
        from cron import monitor_pending
        try:
            pending = monitor_pending.resume(job)
        except monitor_pending.RecoveryRequired as exc:
            return MonitorOutcome(ok=False, error=str(exc))
    ok, output = (True, pending["output"]) if pending else _run_monitor_source(job)
    if not ok:
        return MonitorOutcome(ok=False, error=output)

    new_hash = hash_monitor_output(output)
    raw_state = job.get("monitor_state")
    last_hash = raw_state.get("last_output_hash") if isinstance(raw_state, dict) else None

    if pending is None and last_hash is not None and new_hash == last_hash:
        return MonitorOutcome(ok=True, changed=False)

    first_run = last_hash is None
    old_output = "" if first_run else _read_last_output(job_id)

    shown_output = output
    if len(shown_output) > MAX_OUTPUT_CHARS:
        shown_output = shown_output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

    current = f"### Current output\n\n```\n{shown_output}\n```"
    if first_run:
        context_block = (
            "## Monitor Baseline (first run)\n\n"
            "This is the first observation of the monitored source — there is "
            "no previous output to diff against.\n\n" + current
        )
    else:
        diff = build_monitor_diff(old_output, output)
        context_block = (
            "## MONITOR CHANGE DETECTED\n\n"
            "The monitored source's output changed since the last run.\n\n"
            f"### Diff (previous → current)\n\n```diff\n{diff}\n```\n\n" + current
        )

    if job.get("monitor_commit_policy") == "safe_retry" and pending is None:
        from cron import monitor_pending
        try:
            pending = monitor_pending.begin(job, output)
        except monitor_pending.RecoveryRequired as exc:
            return MonitorOutcome(ok=False, error=str(exc))
        if pending is None:
            return MonitorOutcome(ok=True, changed=False)
    elif job.get("monitor_commit_policy") not in {"after_delivery", "safe_retry"}:
        _persist_monitor_state(job_id, new_hash, output)
    return MonitorOutcome(
        ok=True,
        changed=True,
        first_run=first_run,
        context_block=context_block,
        new_hash=new_hash,
        output=output,
        pending_token=pending["token"] if pending else None,
    )


def _persist_monitor_state(job_id: str, new_hash: str, output: str) -> None:
    from cron.jobs import _hermes_now, update_job

    _write_last_output(job_id, output)
    try:
        update_job(
            job_id,
            {
                "monitor_state": {
                    "last_output_hash": new_hash,
                    "last_changed_at": _hermes_now().isoformat(),
                }
            },
        )
    except Exception as exc:
        logger.warning("Monitor: failed to persist state for %r: %s", job_id, exc)


def commit_monitor_state(job_id: str, new_hash: str, output: str) -> None:
    """Commit a deferred monitor observation after downstream success."""
    if not new_hash:
        raise ValueError("new_hash is required to commit monitor state")
    from cron.jobs import _hermes_now, update_job

    snapshot_path = _snapshot_path(job_id)
    old_snapshot_exists = snapshot_path.exists()
    old_output = (
        snapshot_path.read_text(encoding="utf-8") if old_snapshot_exists else ""
    )
    _write_last_output_strict(job_id, output)

    def rollback_snapshot() -> None:
        if old_snapshot_exists:
            _write_last_output_strict(job_id, old_output)
        else:
            snapshot_path.unlink(missing_ok=True)

    try:
        # update_job rewrites the job dict and persists jobs.json in one
        # step; hash and last_changed_at are therefore applied together or
        # not at all (a failed update_job raises before any partial state
        # survives). The rollback below only has to cover the snapshot
        # file, which is why it does not restore a timestamp.
        updated = update_job(
            job_id,
            {
                "monitor_state": {
                    "last_output_hash": new_hash,
                    "last_changed_at": _hermes_now().isoformat(),
                }
            },
        )
        if updated is None:
            raise RuntimeError(f"monitor job {job_id!r} disappeared during commit")
    except Exception:
        try:
            rollback_snapshot()
        except Exception:
            logger.exception(
                "Monitor: failed to roll back snapshot for %r after hash commit failure",
                job_id,
            )
        raise
