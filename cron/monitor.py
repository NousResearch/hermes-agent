"""Monitor-mode cron support — hash-suppressed change detection.

A monitor job attaches a cheap source (``monitor_script`` / ``monitor_url``) to an LLM cron job.
Each tick runs the source FIRST and hashes its EXACT output bytes (no timestamp/whitespace
normalization — scripts must emit stable output) against the hash from the last agent-triggering
tick: unchanged → agent run suppressed (silent ``no_change`` run); changed/first run → a "MONITOR
CHANGE DETECTED" block (capped unified diff + new output) is injected into the prompt; source
failure → an ERROR, never a change, and the stored hash is left untouched. A suppressed tick is
NEUTRAL for the job's health — it neither recovers the job nor touches its failure streak (see
``NO_CHANGE_FLAG`` below). State:
``job["monitor_state"]`` in jobs.json (hash + last_changed_at) and
``OUTPUT_DIR/<job_id>/monitor_last_output.txt`` (for the diff).
"""

from __future__ import annotations

import difflib
import hashlib
import logging
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
            return path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("Monitor: failed to read last output for %r: %s", job_id, exc)
    return ""


def _write_last_output(job_id: str, output: str) -> None:
    try:
        from cron.jobs import _ensure_cron_dir

        path = _snapshot_path(job_id)
        _ensure_cron_dir(path.parent)
        path.write_text(output, encoding="utf-8")
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

        return _run_job_script(monitor_script, workdir=_field(job, "workdir") or None,
                               interpreter=job.get("interpreter"))
    monitor_url = _field(job, "monitor_url")
    if monitor_url:
        return _fetch_monitor_url(monitor_url)
    return False, "monitor job has neither monitor_script nor monitor_url"


def job_has_monitor(job: dict) -> bool:
    return bool(_field(job, "monitor_script") or _field(job, "monitor_url"))


# --- A suppressed tick is NEUTRAL for the job's health ----------------------------------------
# The gate reports an unchanged tick as a successful run so it costs no LLM turn and delivers
# nothing, but that success says NOTHING about whether the job's own work succeeds. Reading it as
# a plain success resolved the job's open failure incidents and reset ``failure_streak`` on every
# unchanged poll, which made ``cron.failure_repeat_alert_hours`` and ``cron.failure_nudge_threshold``
# dead letters for monitor-gated jobs: every real failure arrived as the first failure of a fresh
# streak — one notice per failing run instead of one notice plus its reminder cadence (#140178).
# So a suppressed tick must leave the failure state exactly as the last real run left it.
#
# The flag rides on the in-flight job dict for one fire (the same idiom as ``_model_unreachable`` /
# ``_quota_hold_seconds``: set by the run, read by that fire's terminal bookkeeping). It travels to
# the firing scheduler through ``gateway.session_cron.RUN_JOB_FLAGS`` when the gateway owns the run.
NO_CHANGE_FLAG = "_monitor_no_change"


def flag_no_change_tick(job: dict) -> None:
    """Mark this fire as a suppressed (agent-less) monitor tick, for its bookkeeping tail."""
    job[NO_CHANGE_FLAG] = True


def suppressed_tick(job: object) -> bool:
    """True when *job*'s in-flight run was a suppressed monitor tick (no agent work ran)."""
    return bool(isinstance(job, dict) and job.get(NO_CHANGE_FLAG))


def resolve_incidents_after_run(job: dict) -> None:
    """Resolve the job's open incidents after a run that DID work; a suppressed tick resolves none."""
    if suppressed_tick(job):
        return
    from cron.incidents import close_incidents_for_recovered_job

    close_incidents_for_recovered_job(job["id"])


def check_monitor(job: dict) -> MonitorOutcome:
    """Run the monitor source and decide whether the agent should run.

    On change (or first run) the new hash + snapshot are persisted BEFORE the agent runs — detection
    time is the state boundary, so a failed agent run doesn't re-alert on the same content forever.
    On failure nothing is persisted.
    """
    job_id = str(job.get("id") or "")
    ok, output = _run_monitor_source(job)
    if not ok:
        return MonitorOutcome(ok=False, error=output)

    new_hash = hash_monitor_output(output)
    raw_state = job.get("monitor_state")
    last_hash = raw_state.get("last_output_hash") if isinstance(raw_state, dict) else None

    if last_hash is not None and new_hash == last_hash:
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

    _persist_monitor_state(job_id, new_hash, output)
    return MonitorOutcome(ok=True, changed=True, first_run=first_run, context_block=context_block)


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
