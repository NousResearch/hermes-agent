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
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# Cap for the unified diff injected into the prompt.
MAX_DIFF_CHARS = 4000
# Cap for the new-output block injected into the prompt (mirrors the 8k
# context_from truncation in cron/scheduler.py).
MAX_OUTPUT_CHARS = 8000
# Bounded GET limits for monitor_url sources.
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


def _hash_monitor_bytes(output: bytes) -> str:
    """Hash exact source bytes before lossy decoding for prompt display."""
    return hashlib.sha256(output).hexdigest()


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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    except Exception as exc:
        logger.warning("Monitor: failed to persist last output for %r: %s", job_id, exc)


def clear_monitor_snapshot(job_id: str) -> None:
    """Remove a monitor's persisted diff snapshot after its source changes."""
    try:
        _snapshot_path(job_id).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Monitor: failed to clear last output for %r: %s", job_id, exc)


def _fetch_monitor_url_bytes(url: str) -> tuple[bool, bytes | str]:
    """Bounded GET of a monitor URL. Success returns exact body bytes."""
    from tools.url_safety import create_ssrf_safe_client, is_safe_url

    current_url = str(url)
    try:
        with create_ssrf_safe_client(
            timeout=URL_TIMEOUT_SECONDS,
            follow_redirects=False,
            headers={"User-Agent": "hermes-cron-monitor"},
        ) as client:
            for _ in range(MAX_URL_REDIRECTS + 1):
                scheme = (urlparse(current_url).scheme or "").lower()
                if scheme not in {"http", "https"}:
                    return False, f"monitor_url must be http(s): {current_url!r}"
                if not is_safe_url(current_url):
                    return False, "monitor_url blocked by SSRF protection"

                with client.stream("GET", current_url) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            return False, "monitor_url redirect omitted Location"
                        current_url = urljoin(current_url, location)
                        continue

                    resp.raise_for_status()
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in resp.iter_bytes(65_536):
                        total += len(chunk)
                        if total > MAX_URL_BYTES:
                            return False, (
                                f"monitor_url response exceeds {MAX_URL_BYTES} bytes"
                            )
                        chunks.append(chunk)
                    return True, b"".join(chunks)
        return False, f"monitor_url exceeded {MAX_URL_REDIRECTS} redirects"
    except Exception as exc:
        return False, f"monitor_url fetch failed: {exc}"


def _fetch_monitor_url(url: str) -> tuple[bool, str]:
    """Text compatibility wrapper around the exact-byte URL fetcher."""
    ok, result = _fetch_monitor_url_bytes(url)
    if not ok:
        return False, _redact_monitor_text(result)
    return True, bytes(result).decode("utf-8", errors="replace")


def _redact_monitor_text(text: object) -> str:
    """Redact every monitor egress even when global redaction is disabled."""
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(
            str(text),
            force=True,
            redact_url_credentials=True,
        )
    except Exception as exc:
        logger.warning("Monitor: failed to redact source data: %s", exc)
        return "[REDACTED - redaction failed]"


def _redact_monitor_output(raw_output: bytes) -> str:
    """Decode source bytes for prompt/snapshot use without hashing this view."""
    return _redact_monitor_text(raw_output.decode("utf-8", errors="replace"))


def _run_monitor_source(job: dict) -> tuple[bool, str, bytes]:
    """Run one monitor source, returning display text plus exact hash bytes."""
    monitor_script = (job.get("monitor_script") or "").strip()
    if monitor_script:
        # Same containment + interpreter rules as the existing `script` field.
        from cron.scheduler import _run_job_script

        workdir = (job.get("workdir") or "").strip() or None
        ok, result = _run_job_script(
            monitor_script,
            workdir=workdir,
            raw_output=True,
        )
        if not ok:
            return False, _redact_monitor_text(result), b""
        raw_output = bytes(result)
        return True, _redact_monitor_output(raw_output), raw_output
    monitor_url = (job.get("monitor_url") or "").strip()
    if monitor_url:
        ok, result = _fetch_monitor_url_bytes(monitor_url)
        if not ok:
            return False, _redact_monitor_text(result), b""
        raw_output = bytes(result)
        return True, _redact_monitor_output(raw_output), raw_output
    return False, "monitor job has neither monitor_script nor monitor_url", b""


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
    ok, output, raw_output = _run_monitor_source(job)
    if not ok:
        return MonitorOutcome(ok=False, error=output)

    new_hash = _hash_monitor_bytes(raw_output)
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

    if _persist_monitor_state(job, new_hash, output) is False:
        # The source was edited or the job removed while this observation was
        # in flight. Do not run the agent for obsolete data; the new source
        # keeps its empty baseline and will run normally on its next tick.
        return MonitorOutcome(ok=True, changed=False)
    return MonitorOutcome(
        ok=True, changed=True, first_run=first_run, context_block=context_block
    )


def _persist_monitor_state(job: dict, new_hash: str, output: str) -> Optional[bool]:
    from cron.jobs import (
        _hermes_now,
        _monitor_source_generation,
        update_monitor_state_if_source_matches,
    )

    job_id = job["id"]
    try:
        persisted = update_monitor_state_if_source_matches(
            job_id,
            expected_monitor_script=job.get("monitor_script"),
            expected_monitor_url=job.get("monitor_url"),
            expected_monitor_source_generation=_monitor_source_generation(job),
            monitor_state={
                "last_output_hash": new_hash,
                "last_changed_at": _hermes_now().isoformat(),
            },
            monitor_output=output,
        )
        if not persisted:
            logger.info(
                "Monitor: discarded stale state for %r because its source changed",
                job_id,
            )
        return persisted
    except Exception as exc:
        logger.warning("Monitor: failed to persist state for %r: %s", job_id, exc)
        return None
