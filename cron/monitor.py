"""Monitor-mode cron support — hash-suppressed change detection.

A monitor job attaches a cheap source (``monitor_script`` / ``monitor_url``) to an LLM cron job.
Each tick runs the source FIRST and hashes its EXACT output bytes (no timestamp/whitespace
normalization — scripts must emit stable output) against the hash from the last agent-triggering
tick: unchanged → agent run suppressed (silent ``no_change`` run); changed/first run → a "MONITOR
CHANGE DETECTED" block (capped unified diff + new output) is injected into the prompt; source
failure → an ERROR, never a change, and the stored hash is left untouched. State:
``job["monitor_state"]`` in jobs.json (hash + last_changed_at, plus ``source_failed_at`` while a
failure window is open) and ``OUTPUT_DIR/<job_id>/monitor_last_output.txt`` (for the diff).

Failure-window contract: while ``source_failed_at`` is set, change detection is DOWN
— a hash pinned before the window cannot prove "unchanged". The first tick after the source works
again therefore NEVER suppresses: identical output surfaces a visible "MONITOR SOURCE RECOVERED"
re-baseline tick; different output takes the normal change path. Either success clears the flag
and re-arms detection from the current baseline.
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
MAX_URL_REDIRECTS = 5

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
    """Bounded GET of a monitor URL. Returns (ok, body-or-error).

    The URL can come from the agent's own cronjob tool call and the body is fed back into the
    agent's prompt, so every hop (initial URL and each redirect target) passes the shared SSRF
    policy, and the connection itself goes through the connect-time-guarded client.
    """
    from urllib.parse import urljoin

    from tools.url_safety import create_ssrf_safe_client, is_safe_url

    current_url = str(url)
    if not current_url.lower().startswith(("http://", "https://")):
        return False, f"monitor_url must be http(s): {url!r}"
    try:
        with create_ssrf_safe_client(timeout=URL_TIMEOUT_SECONDS, follow_redirects=False,
                                     headers={"User-Agent": "hermes-cron-monitor",
                                              "Accept-Encoding": "identity"}) as client:
            for _ in range(MAX_URL_REDIRECTS + 1):
                if not is_safe_url(current_url):
                    return False, ("monitor_url blocked: private/internal or unsafe destination "
                                   "(intranet monitors need security.allow_private_urls: true)")
                with client.stream("GET", current_url) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            return False, "monitor_url redirect without a Location header"
                        current_url = urljoin(current_url, location)
                        continue
                    resp.raise_for_status()
                    body = bytearray()
                    # Raw bytes, as urllib read them: decoding happens before this cap sees a
                    # chunk, and one read of a hostile br/gzip body inflates to GBs.
                    for chunk in resp.iter_raw():
                        body += chunk
                        if len(body) > MAX_URL_BYTES:
                            break
                return True, bytes(body[:MAX_URL_BYTES]).decode("utf-8", errors="replace")
        return False, f"monitor_url exceeded {MAX_URL_REDIRECTS} redirects"
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


def _persist_failure_flag(job_id: str, job: dict) -> None:
    """Open the failure window in the store: ``monitor_state.source_failed_at``.

    Written once per failure window (not per failing tick) and cleared by the next successful
    ``_persist_monitor_state``. Purpose: a hash pinned BEFORE the window must never
    silently suppress the tick that discovers the source healthy again — the recovery tick
    reads this flag and re-baselines visibly instead.
    """
    raw_state = job.get("monitor_state")
    if isinstance(raw_state, dict) and raw_state.get("source_failed_at"):
        return  # window already open — one flag per failure, not one write per failing tick
    from cron.jobs import _hermes_now, update_job

    prior = {k: v for k, v in raw_state.items() if k != "source_failed_at"} \
        if isinstance(raw_state, dict) else {}
    try:
        update_job(job_id, {"monitor_state": {
            **prior, "source_failed_at": _hermes_now().isoformat()}})
    except Exception:  # a store write failure must not kill the tick
        logger.warning("Monitor: failed to persist the failure flag for %r", job_id, exc_info=True)


def check_monitor(job: dict) -> MonitorOutcome:
    """Run the monitor source and decide whether the agent should run.

    On change (or first run) the new hash + snapshot are persisted BEFORE the agent runs — detection
    time is the state boundary, so a failed agent run doesn't re-alert on the same content forever.
    On failure nothing about the hash is persisted; the failure window opens instead (see
    ``_persist_failure_flag``).
    """
    job_id = str(job.get("id") or "")
    ok, output = _run_monitor_source(job)
    if not ok:
        _persist_failure_flag(job_id, job)
        return MonitorOutcome(ok=False, error=output)

    new_hash = hash_monitor_output(output)
    raw_state = job.get("monitor_state")
    last_hash = raw_state.get("last_output_hash") if isinstance(raw_state, dict) else None
    source_failed = bool(isinstance(raw_state, dict) and raw_state.get("source_failed_at"))

    if last_hash is not None and new_hash == last_hash:
        if not source_failed:
            return MonitorOutcome(ok=True, changed=False)
        # Recovery after a failure window, output identical to the pinned hash: the hash was
        # pinned BEFORE the window, so "unchanged" is no longer trustworthy evidence of "no
        # change" — the source was unrunnable in between and may have changed unseen.
        # Surface a visible re-baseline tick instead of a silent suppression; the
        # canonical persist below clears the flag and re-arms detection from this baseline.
        shown_output = output
        if len(shown_output) > MAX_OUTPUT_CHARS:
            shown_output = shown_output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        context_block = (
            "## MONITOR SOURCE RECOVERED\n\n"
            "The monitored source failed on one or more previous ticks, so change detection was "
            "down and the stored baseline predates the failure window. This tick succeeded and "
            "the output is identical to the pre-failure baseline — nothing changed while the "
            "source was down (or a change was missed and has since reverted). Detection is "
            "re-armed from this baseline.\n\n"
            f"### Current output\n\n```\n{shown_output}\n```"
        )
        _persist_monitor_state(job_id, new_hash, output)
        return MonitorOutcome(ok=True, changed=True, context_block=context_block)

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
