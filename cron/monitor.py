"""Monitor-mode cron support — hash-suppressed change detection.

A monitor job attaches a cheap *monitor source* (``monitor_script`` or
``monitor_url``) to an ordinary LLM cron job. Each tick the scheduler runs
the source FIRST and compares a hash of its exact output bytes against the
hash stored from the last agent-triggering tick:

* unchanged → the agent run is suppressed entirely (no LLM, no delivery);
  the tick is recorded as a silent ``no_change`` run.
* changed (or first run) → a "MONITOR CHANGE DETECTED" context block —
  unified diff of old vs new output (capped) plus the new output — is
  injected into the prompt and the agent runs normally.
* source failure → treated as an ERROR, never as a change. The stored hash
  is left untouched so a source that recovers to its previous output still
  suppresses.

Output is compared as EXACT BYTES — no timestamp stripping or whitespace
normalization. Monitor scripts should emit stable output (sort results,
omit "generated at" lines) or every tick will look like a change.

State lives in two places, both durable across scheduler restarts:

* ``job["monitor_state"]`` in jobs.json — ``last_output_hash`` +
  ``last_changed_at`` (additive JSON fields, no migration needed);
* ``OUTPUT_DIR/<job_id>/monitor_last_output.txt`` — the previous output
  text, kept only so the next change can render a diff.

Inspired by: ChatGPT Work monitor tasks (idea-level, docs-only);
enabler: #80774.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Cap for the unified diff injected into the prompt.
MAX_DIFF_CHARS = 4000
# Cap for the new-output block injected into the prompt (mirrors the 8k
# context_from truncation in cron/scheduler.py).
MAX_OUTPUT_CHARS = 8000
# Bounded GET limits for monitor_url sources.
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
            old.splitlines(),
            new.splitlines(),
            fromfile="previous",
            tofile="current",
            lineterm="",
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


def _write_last_output(job_id: str, output: str) -> None:
    try:
        path = _snapshot_path(job_id)
        from cron.jobs import _ensure_cron_dir
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
        if len(body) > MAX_URL_BYTES:
            body = body[:MAX_URL_BYTES]
        return True, body.decode("utf-8", errors="replace")
    except Exception as exc:
        return False, f"monitor_url fetch failed: {exc}"


def _run_monitor_source(job: dict) -> tuple[bool, str]:
    """Run the job's monitor source (script or URL). Returns (ok, output)."""
    monitor_script = (job.get("monitor_script") or "").strip()
    if monitor_script:
        # Same containment + interpreter rules as the existing `script` field.
        from cron.scheduler import _run_job_script

        workdir = (job.get("workdir") or "").strip() or None
        return _run_job_script(monitor_script, workdir=workdir)
    monitor_url = (job.get("monitor_url") or "").strip()
    if monitor_url:
        return _fetch_monitor_url(monitor_url)
    return False, "monitor job has neither monitor_script nor monitor_url"


def job_has_monitor(job: dict) -> bool:
    return bool((job.get("monitor_script") or "").strip() or (job.get("monitor_url") or "").strip())


def check_monitor(job: dict) -> MonitorOutcome:
    """Run the monitor source and decide whether the agent should run.

    On change (or first run) the new hash + snapshot are persisted BEFORE
    the agent runs — detection time is the state boundary, so a failed
    agent run doesn't re-alert on the same content forever.
    On failure nothing is persisted.
    """
    job_id = str(job.get("id") or "")
    ok, output = _run_monitor_source(job)
    if not ok:
        return MonitorOutcome(ok=False, error=output)

    new_hash = hash_monitor_output(output)
    raw_state = job.get("monitor_state")
    state = raw_state if isinstance(raw_state, dict) else {}
    last_hash = state.get("last_output_hash")

    if last_hash is not None and new_hash == last_hash:
        return MonitorOutcome(ok=True, changed=False)

    first_run = last_hash is None
    old_output = "" if first_run else _read_last_output(job_id)

    shown_output = output
    if len(shown_output) > MAX_OUTPUT_CHARS:
        shown_output = shown_output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

    if first_run:
        context_block = (
            "## Monitor Baseline (first run)\n\n"
            "This is the first observation of the monitored source — there is "
            "no previous output to diff against.\n\n"
            f"### Current output\n\n```\n{shown_output}\n```"
        )
    else:
        diff = build_monitor_diff(old_output, output)
        context_block = (
            "## MONITOR CHANGE DETECTED\n\n"
            "The monitored source's output changed since the last run.\n\n"
            f"### Diff (previous → current)\n\n```diff\n{diff}\n```\n\n"
            f"### Current output\n\n```\n{shown_output}\n```"
        )

    _persist_monitor_state(job_id, new_hash, output)
    return MonitorOutcome(
        ok=True, changed=True, first_run=first_run, context_block=context_block
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
