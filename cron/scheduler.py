"""Cron job scheduler: tick() runs due jobs (gateway calls it every 60s from a background thread).
A file lock (~/.hermes/cron/.tick.lock) keeps overlapping processes to one tick at a time.
"""

import atexit
import concurrent.futures
import contextlib
import contextvars
import errno
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

# fcntl is Unix-only; Windows uses msvcrt
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

# Must precede repo-level imports: standalone invocations (e.g. module reload after
# `hermes update`) otherwise fail with ModuleNotFoundError for hermes_time et al.
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home
from cron.env_settings import cron_env_setting
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import (
    _expand_env_vars, load_config, resolve_cron_model_drift_defaults)
from hermes_cli.fallback_config import get_fallback_chain
from hermes_time import now as _hermes_now
from agent.interrupt_compat import request_hard_interrupt
from agent.delegation_context import (
    enter_non_dispatcher_owned_context, exit_non_dispatcher_owned_context)

logger = logging.getLogger(__name__)


def _extract_cron_context_payload(text: str) -> str:
    """Extract the useful payload from a persisted cron Markdown artifact.

    Cron output files include the job's original prompt for auditability.  When
    one job consumes another through ``context_from``, reinjecting that prompt
    recursively duplicates instructions at every hop.  Prefer the final
    Response/Error section while preserving legacy plain-text output files.
    """
    artifact = (text or "").strip()
    if not artifact:
        return ""

    markers = list(re.finditer(r"(?m)^## (Response|Error)\s*$", artifact))
    if not markers:
        return artifact

    marker = markers[-1]
    marker_end = artifact.find("\n", marker.start())
    if marker_end == -1:
        return ""

    payload = artifact[marker_end + 1 :].strip()
    if not payload or payload == "(No response generated)":
        return ""
    if marker.group(1) == "Error":
        return f"## Error\n\n{payload}"
    return payload


def _close_late_session_db_result(future: "concurrent.futures.Future") -> None:
    """Done-callback: close a SessionDB whose constructor finished after run_job's init timeout
    (worker abandoned via ``shutdown(wait=False)``), else its .db/WAL/SHM handles leak to EMFILE.

    If the constructor later completes inside that abandoned worker, the Future's result — an open SessionDB
    holding .db / WAL / SHM file handles — would be orphaned and never closed, leaking descriptors until
    EMFILE (#72782). This callback retrieves and closes that eventual late result.
    """
    with contextlib.suppress(Exception):
        db = future.result()
        if db is not None:
            from hermes_state_registry import release_or_close
            release_or_close(db)


def _set_cron_session_title(session_db, session_id, base_title):
    """Persist a non-blank, unique title for a finished cron session; returns it (None if unset).
    Runs BEFORE end_session()/close() so no write races the close. Duplicate title (unique-index
    ValueError) -> get_next_title_in_lineage(); if unavailable, raise rather than end up untitled.

    Centralizes the title write so the cron finally block can guarantee a non-blank, unique title is
    persisted before end_session()/close() tear the connection down (issues #50535, #50536, #50537):
    - #50535: never leaves the session blank. base_title already carries a cron-id fallback for nameless
    jobs; this also guards a failed write. Recover by appending a #N suffix via get_next_title_in_lineage()
    when supported, instead of swallowing the error and ending up untitled. - #50536: this runs
    synchronously in the cron finally block ahead of the session close, so no in-flight title write can race
    the close.
    """
    if not session_db or not session_id:
        return None
    title = (base_title or "").strip()
    if not title:
        return None
    try:
        session_db.set_session_title(session_id, title)
        return title
    except ValueError:
        # Unique-title collision: fall back to the next lineage title (base #2, #3, ...).
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        if next_title_fn is None:
            raise
        deduped = next_title_fn(title)
        if not deduped or deduped == title:
            raise
        session_db.set_session_title(session_id, deduped)
        return deduped


def _fallback_chain_phrase() -> str:
    """Fallback-chain clause for a provider-failure message: "exhausted" vs "none configured" (most
    installs). Fails open to the ambiguous wording if config can't be read — never crash delivery.
    """
    try:
        cfg = load_config() or {}
        chain = get_fallback_chain(cfg)
    except Exception:
        return "Fallback chain was exhausted or unavailable."
    if chain:
        return "Fallback chain was exhausted or unavailable."
    return (
        "No fallback chain configured — add one with `hermes fallback add`, "
        "or set a cron fleet default via `cron.model` + `cron.model_provider` in config.yaml."
    )


def _failure_streak_nudge(job: dict) -> str:
    """Review nudge when a recurring job keeps failing, else "". The failure message is delivered
    BEFORE mark_job_run records this run, hence stored ``failure_streak`` + 1. Threshold:
    ``cron.failure_nudge_threshold`` (default 3, 0 disables)."""
    schedule_kind = (job.get("schedule") or {}).get("kind")
    if schedule_kind not in {"cron", "interval"}:
        return ""
    try:
        cfg = load_config() or {}
        threshold = int(
            ((cfg.get("cron") or {}) if isinstance(cfg, dict) else {}).get(
                "failure_nudge_threshold", 3
            )
        )
    except Exception:
        threshold = 3
    if threshold <= 0:
        return ""
    streak = int(job.get("failure_streak") or 0) + 1  # +1 = this run
    if streak < threshold:
        return ""
    job_ref = job.get("name") or job.get("id") or "this job"
    return (
        f"\nThis job has failed {streak} runs in a row — worth a review. "
        f"Fix its prompt/config, or pause it with `hermes cron pause {job_ref}` "
        "(resume/remove also available) to stop the noise."
    )


def _detect_gateway_code_skew() -> tuple[str, str] | None:
    """Boot-vs-disk revision skew for THIS process, or None. Test seam over
    ``gateway.code_skew.detect_code_skew``; a broken import must never take delivery down."""
    try:
        from gateway.code_skew import detect_code_skew

        return detect_code_skew()
    except Exception:
        return None


class CronTickYielded(RuntimeError):
    """A stale-code ticker yielded this tick to a fresh gateway.

    Raised by ``tick()`` BEFORE the tick lock when boot fingerprint ≠ disk, this process does NOT
    own the runtime lock and a fresh process holds it — the stale process must stay out of the
    dispatch race (contention would starve the fresh ticker). Skew ``None`` never yields (fail
    open). Raised, not returned, so ``record_ticker_error`` sees it and ``hermes cron status``
    isn't green.
    """

    def __init__(self, boot_rev: str, disk_rev: str) -> None:
        self.boot_rev = boot_rev
        self.disk_rev = disk_rev
        super().__init__(
            f"Cron tick yielded to a fresh gateway process (stale code: "
            f"booted on {boot_rev}, disk is at {disk_rev})"
        )


# Log the yield at most once per episode (reset when the skew changes) to avoid per-interval spam.
_YIELD_LOG_INTERVAL_SECONDS = 3600.0
_last_yield_log: dict[str, object] = {}


def _should_yield_tick_to_fresh_gateway() -> tuple[str, str] | None:
    """``(boot_rev, disk_rev)`` when this tick must yield to a fresher gateway, else None. Yields
    only when ALL hold: code skew, we don't own the runtime lock, another process holds it. Every
    probe failure returns None — yielding is a certainty claim, never a guess."""
    skew = _detect_gateway_code_skew()
    if skew is None:
        return None
    try:
        from gateway import status as _gateway_status
    except Exception:
        return None
    try:
        if _gateway_status.owns_gateway_runtime_lock():
            return None
        if not _gateway_status.is_gateway_runtime_lock_active():
            return None
    except Exception:
        return None
    return skew


def _log_tick_yield_once(reason: str) -> None:
    """Log the yield at error level once per episode (skew signature)."""
    global _last_yield_log
    now = time.monotonic()
    last_reason = _last_yield_log.get("reason")
    last_at = _last_yield_log.get("at", 0.0)
    if last_reason != reason or (now - float(last_at)) >= _YIELD_LOG_INTERVAL_SECONDS:
        logger.error(
            "Cron tick yielded: this process is running stale code (%s) and a "
            "fresher gateway owns the runtime lock — jobs will fire from that "
            "process. Restart this one to reclaim its ticks.",
            reason)
    _last_yield_log = {"reason": reason, "at": now}


def _summarize_cron_failure_for_delivery(job: dict, error: str | None) -> str:
    """Compact one-line failure message for chat delivery (full details stay in cron output)."""
    job_name = job.get("name") or job.get("id") or "cron job"
    text = (error or "unknown error").strip()
    lower = text.lower()

    # no_agent jobs never reach a model, so provider errors are structurally impossible for them.
    # Gate on job MODE before substring matching, or a script's own wording ("timed out", "429")
    # would blame the wrong subsystem; the generic cleaner below reports what actually happened.
    provider_reachable = not job.get("no_agent")

    # Script runner contract ("Script timed out after {n}s: {path}") — also for agent jobs with a
    # context script. Must precede generic timeout matching so it never claims a provider fallback.
    # See #78503, #82460.
    if lower.startswith("script timed out"):
        return (
            f"⚠️ Cron '{job_name}' failed: script timed out. "
            "No model was invoked. Full details saved in cron output."
        )

    # Whole-token 429: substrings in job ids/ports/hashes tripped false rate-limit alerts.
    if provider_reachable and (
        # Provider/API failures are the common noisy path. Keep these short. Match 429 as a whole token
        # (#83188 @cation98): bare substring matching let identifiers containing those digits (job ids,
        # ports, hashes) trip a false "provider rate limit" alert.
        re.search(r"\b429\b", text) or "rate limit" in lower or "usage limit" in lower
    ):
        reason = "rate limit"
        if "weekly usage limit" in lower:
            reason = "weekly usage limit"
        elif "quota" in lower:
            reason = "quota limit"
        return (
            f"⚠️ Cron '{job_name}' failed: provider {reason}. "
            f"{_fallback_chain_phrase()} "
            "Full details saved in cron output."
        )

    # Scheduler inactivity watchdog shape ("idle for {n}s (limit {m}s)"). Must precede the generic
    # provider-timeout branch: the job's own tool going quiet involves no provider/fallback chain.
    # The scheduler's own inactivity watchdog (see the TimeoutError raised above at "Cron job '{job_name}'
    # idle for {secs}s (limit {limit}s) — last activity: {desc}") produces a message that contains the
    # substring "timed out"/"timeout" nowhere, but DOES contain "idle for ... (limit ...)" — however
    # older/other call sites can still phrase an inactivity abort using "timed out" wording, so match on the
    # "idle for Ns (limit" shape specifically (case-insensitive) BEFORE the generic provider- timeout branch
    # below. Without this, an inactivity timeout — the job's OWN tool call/turn going quiet, no provider or
    # fallback chain ever involved — gets rewritten into a misleading "provider timeout / fallback chain
    # exhausted" message, sending the operator to debug the wrong system entirely (field-reported: a stuck
    # `terminal` tool call tripped the 600s inactivity limit and was reported as a provider/fallback
    # failure). Mirrors the same reordering fix upstream issue #59549 applied for script timeouts vs
    # provider timeouts — check the more specific, deterministic signature first.
    if re.search(r"idle for \d+s\s*\(limit \d+s\)", lower):
        return (
            f"⚠️ Cron '{job_name}' failed: the job itself stalled — no tool/API "
            "activity for the configured inactivity window. Not a provider or "
            "fallback-chain issue; check what the job was doing when it went "
            "quiet. Full details saved in cron output."
        )

    if provider_reachable and (
        "readtimeout" in lower or "timed out" in lower or "timeout" in lower
    ):
        return (
            f"⚠️ Cron '{job_name}' failed: provider timeout. "
            f"{_fallback_chain_phrase()} "
            "Full details saved in cron output."
        )

    # Whole-token 401/403 and auth wording so "oauth", "4015" etc. don't trip a false auth message.
    if provider_reachable and (
        re.search(r"authenticat|authoriz", lower) or re.search(r"\b(401|403)\b", text)
    ):
        return (
            f"⚠️ Cron '{job_name}' failed: provider authentication error. "
            "Full details saved in cron output."
        )

    # Strip exception wrappers; bound input first so a multi-KB blob can't slow the regexes.
    cleaned = re.sub(r"^(RuntimeError|Exception|ValueError|HTTPStatusError):\s*", "", text[:2000])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 180:
        cleaned = cleaned[:177].rstrip() + "..."
    message = f"⚠️ Cron '{job_name}' failed: {cleaned}"

    # Import-class failures in a gateway whose checkout changed underneath it (mixed sys.modules)
    # read like code bugs. When boot SHA ≠ disk HEAD, APPEND cause + fix — never replace the raw
    # error, which carries the failing symbol. Fail-safe: skew is None on non-git/no-fingerprint
    # (message unchanged); no_agent jobs excluded via the same mode gate (a fresh subprocess
    # resolves imports against disk, so its ImportError is the script's own problem).
    # Import-class failures (#95294 part 3): a long-lived gateway whose checkout was updated underneath it
    # (interrupted `hermes update`, manual git pull) serves MIXED modules — old entries frozen in
    # sys.modules, new files loaded by lazy imports — and every agent cron job then dies with `cannot import
    # name X` / ModuleNotFoundError. The error itself reads like a code bug, so operators debug the wrong
    # thing (2 days on the reporting incident, 15 missed jobs).
    if provider_reachable and re.search(
        r"cannot import name|modulenotfounderror|importerror", lower
    ):
        try:
            skew = _detect_gateway_code_skew()
        except Exception:
            skew = None  # delivery must never die on a diagnostics probe
        if skew is not None:
            boot_rev, disk_rev = skew
            message += (
                f" Likely cause: the gateway is running stale code (booted "
                f"on {boot_rev}, disk is at {disk_rev}) — run "
                "`hermes gateway restart` to fix it."
            )

    return message


def _upsert_incident_for_failure(
    job: dict, error: str, *, output_file: Optional[Any] = None
) -> tuple[bool, Optional[str]]:
    """Record a durable failure incident (grouped by job + error signature). Returns
    ``(acked, incident_id)``; acked=True when the signature's incident is already ``closed`` ->
    suppress the per-run ping. Store errors log at debug; the caller delivers as if none existed."""
    try:
        from cron.incidents import get_incident, upsert_incident

        incident_id, _is_new = upsert_incident(
            job["id"], str(error or ""), job_name=job.get("name"), output_file=output_file)
        incident = get_incident(incident_id)
        acked = bool(incident and incident.get("state") == "closed")
        return acked, incident_id
    except Exception as exc:
        logger.debug(
            "Incident store unavailable for job %s (delivery unaffected): %s",
            job["id"], exc)
        return False, None


def _mark_incident_alerted(incident_id: Optional[str]) -> None:
    """Best-effort: mark incident ``alerted`` (no-op for closed; never resurrects an acked one)."""
    if not incident_id:
        return
    try:
        from cron.incidents import set_incident_state

        set_incident_state(incident_id, "alerted")
    except Exception as exc:
        logger.debug("Failed marking incident %s alerted: %s", incident_id, exc)


class CronPromptInjectionBlocked(Exception):
    """Raised by _build_job_prompt when the assembled prompt (incl. runtime-loaded skill content,
    unseen by create-time scanning) trips the injection scanner; run_job turns it into a clean
    "job blocked" delivery.

    Assembled-prompt scanning (including loaded skill content) plugs the gap from #3968: create-time
    scanning only covers the user-supplied prompt field; skill content loaded at runtime was never scanned,
    so a malicious skill could carry an injection payload that reached the non-interactive (auto-approve)
    cron agent.
    """


def _resolve_cron_disabled_toolsets(cfg: dict) -> list[str]:
    """Toolsets a cron-spawned agent must never receive: ``messaging``/``clarify`` always
    (interactive); ``cronjob`` by default (loop prevention, not a security boundary —
    ``cron.allow_agent_scheduling: true`` lifts only that); ``agent.disabled_toolsets`` layered on
    top so per-job ``enabled_toolsets`` cannot widen past config.yaml's denylist.

    See #25752.
    """
    cron_cfg = (cfg or {}).get("cron") or {}
    if cron_cfg.get("allow_agent_scheduling"):
        disabled = ["messaging", "clarify"]
    else:
        disabled = ["cronjob", "messaging", "clarify"]
    agent_cfg = (cfg or {}).get("agent") or {}
    from agent.skill_utils import parse_config_string_list

    user_disabled = parse_config_string_list(agent_cfg.get("disabled_toolsets"))
    for name in user_disabled:
        name = str(name).strip()
        if name and name not in disabled:
            disabled.append(name)
    return disabled


def _merge_mcp_into_per_job_toolsets(per_job: list[str], cfg: dict) -> list[str]:
    """Layer enabled MCP servers onto a per-job ``enabled_toolsets`` allowlist (else a per-job list
    silently drops every MCP server). Mirrors ``_get_platform_tools``: ``no_mcp`` sentinel -> none
    (stripped); any MCP server already listed -> allowlist, add nothing; else union all enabled."""
    result = [t for t in per_job if t != "no_mcp"]
    if "no_mcp" in per_job:
        return result
    # lazy: avoid heavy hermes_cli import at module load; shares MCP-membership with gateway/CLI
    from hermes_cli.tools_config import enabled_mcp_server_names
    enabled_mcp = enabled_mcp_server_names(cfg)
    if set(result) & enabled_mcp:
        return result
    for name in sorted(enabled_mcp):
        if name not in result:
            result.append(name)
    return result


def _resolve_cron_enabled_toolsets(job: dict, cfg: dict) -> list[str] | None:
    """Toolset list for a cron job. Precedence: per-job ``enabled_toolsets`` (+ MCP merge) >
    ``cron`` platform config (``_get_platform_tools``, which strips _DEFAULT_OFF_TOOLSETS so fresh
    installs run without ``moa``) > ``None`` on any failure (full default set).

    1. Per-job ``enabled_toolsets`` (set via ``cronjob`` tool on create/update). Keeps the agent's
    job-scoped toolset override intact — #6130. Enabled MCP servers are layered on per
    ``_merge_mcp_into_per_job_toolsets`` so a native-toolset allowlist does not silently strip MCP tools. 2.
    Mirrors gateway behavior (``_get_platform_tools(cfg, platform_key)``) so users can gate cron toolsets
    globally without recreating every job. 3. ``None`` on any lookup failure — AIAgent loads the full
    default set (legacy behavior before this change, preserved as the safety net).
    """
    per_job = job.get("enabled_toolsets")
    if per_job:
        return _merge_mcp_into_per_job_toolsets(list(per_job), cfg or {})
    try:
        from hermes_cli.tools_config import _get_platform_tools  # lazy: avoid heavy import at cron module load
        return sorted(_get_platform_tools(cfg or {}, "cron"))
    except Exception as exc:
        logger.warning(
            "Cron toolset resolution failed, falling back to full default toolset: %s",
            exc)
        return None


def _resolve_job_reasoning_config(job: dict, cfg: dict, model: str) -> dict | None:
    """Effective reasoning config for a cron run. A per-job ``reasoning_effort`` pin beats global
    and per-model config and is model-independent by design (also governs an auth-fallback swap);
    clamping stays with provider transports. An unparseable pin warns and falls back to config."""
    from hermes_constants import parse_reasoning_effort, resolve_reasoning_config

    pinned = job.get("reasoning_effort")
    if pinned is not None:
        parsed = parse_reasoning_effort(pinned)
        if parsed is not None:
            logger.info("Job '%s': using per-job reasoning_effort '%s'", job.get("id", "?"), pinned)
            return parsed
        logger.warning(
            "Job '%s': invalid stored reasoning_effort %r — ignoring the pin "
            "and falling back to config resolution. Fix with `cronjob "
            "action=update job_id=%s reasoning_effort=<level>` (valid: none, "
            "minimal, low, medium, high, xhigh, max, ultra).",
            job.get("id", "?"),
            pinned,
            job.get("id", "?"))
    return resolve_reasoning_config(cfg if isinstance(cfg, dict) else {}, str(model))


from cron.jobs import (
    _ensure_cron_dir, advance_next_runs, claim_dispatch, claim_job_for_fire, fire_claim_fence,
    clear_run_claim, get_due_jobs, heartbeat_fire_claim, heartbeat_run_claim, mark_job_run,
    save_job_output, use_cron_store)
from cron.executions import (
    _TERMINAL_STATES, create_execution, finish_execution, get_execution,
    mark_execution_handoff_pending, mark_execution_running, recover_interrupted_executions)

# Response marker that suppresses delivery (output is still saved locally for audit).
SILENT_MARKER = "[SILENT]"


def _is_cron_silence_response(text: str) -> bool:
    """True when a cron final response should suppress delivery: ``[SILENT]`` (or SILENT /
    NO_REPLY / NO REPLY) as the whole response OR its own first/last line — NOT mid-sentence.
    Shares the webhook-lane matcher in :mod:`gateway.response_filters` so the two cannot drift.

    Recognizes the bracketed ``[SILENT]`` sentinel (whole-response, first line, or last line) plus the
    bracketless ``SILENT`` / ``NO_REPLY`` / ``NO REPLY`` variants the model emits when it drops the brackets
    (#51438, #46917). Whitespace-trimmed and case-insensitive. A token buried mid-sentence is treated as
    real content and delivered.
    """
    from gateway.response_filters import is_autonomous_silence_response

    return is_autonomous_silence_response(text)

# Persistent pool for parallel cron jobs: tick() submits and returns; long jobs never block it.
_parallel_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_parallel_pool_max_workers: Optional[int] = None
_running_job_ids: set = set()
_running_fire_owners: dict[str, dict[object, tuple[Optional[str], Path]]] = {}
# Parent gateway threads synchronously waiting on restart-safe scope workers.
# Shutdown must not misclassify these as ownerless in-process runs: the tool
# process sweep cannot reach the worker's transient scope.
_restart_safe_waiter_job_ids: set[str] = set()
_running_lock = threading.Lock()

# Per in-flight id: time.time() claim instant + the future owning its release (``_FUTURE_PENDING``
# until pool.submit returns). Past-allowance with no live future = leak; the sweep force-releases.
_running_since: dict = {}
_running_futures: dict = {}

# Installed in ``_running_futures`` at claim time so a sweep landing before ``pool.submit`` returns
# never sees ``missing`` and releases a claim about to get its future.
_FUTURE_PENDING = object()

# Forced-release count/history for ``get_inflight_guard_stats()``; mirrored to JSONL for probes.
_forced_release_count: int = 0
_forced_releases: list = []
_FORCED_RELEASE_HISTORY = 20

# Stale-allowance floor (minutes); per-job allowance is max(2 * interval, this).
_INFLIGHT_MIN_ALLOWANCE_MINUTES = 30.0


# Execution tokens (``_running_fire_owners`` identity keys) force-interrupted at shutdown; see
# ``mark_running_jobs_interrupted``. ``run_one_job`` checks its OWN token before writing
# ``last_status`` so a still-running agent thread can't overwrite "interrupted" with a false "ok".
# Token keying scopes the flag to one execution (recurring jobs reuse IDs); legacy paths without a
# fire owner fall back to the bare job ID.
# ``run_one_job``'s own completion path checks its OWN token before writing ``last_status`` so a cron agent
# thread that keeps running in-process after its tool was killed out from under it — and produces a
# plausible-looking final response from truncated output — can never overwrite the interrupted status with a
# false "ok" (#60432). Token keying keeps an interruption scoped to that exact execution: a later run of the
# same job ID (recurring jobs reuse the ID every fire) must not inherit the stale flag.
_interrupted_job_ids: set = set()


class _CancelEventLike(Protocol):
    """Structural type for cancellation sources (``threading.Event``, ``_CombinedCancelEvent``)."""

    def is_set(self) -> bool: ...
    def set(self) -> None: ...


class _CombinedCancelEvent:
    """Duck-typed ``threading.Event`` ORing several cancellation sources (fire-claim heartbeat
    ``lost_ownership`` + per-transport events). Workers only call is_set()/set(), so no pump thread.
    """

    def __init__(self, *events: Optional["_CancelEventLike"]) -> None:
        self._events = [event for event in events if event is not None]

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def set(self) -> None:
        for event in self._events:
            event.set()


def get_running_job_ids() -> "frozenset[str]":
    """Thread-safe snapshot of executing job IDs (dispatch until ``_process_job`` returns). Read by
    the gateway shutdown drain, otherwise blind to cron work (runs outside ``_running_agents``).

    _drain_active_agents``) reads this to treat in-flight cron work as active the same way it already treats
    in-flight chat sessions via ``_running_agents`` — cron jobs run through their own thread pool here,
    entirely outside that dict, so without this the drain is structurally blind to them (#60432).
    """
    with _running_lock:
        return frozenset(_running_job_ids | _running_fire_owners.keys())


def try_register_running_job(job_id: str) -> bool:
    """Atomically add ``job_id`` to the in-flight set; False (caller must skip) if already mid-run.
    Single dedupe owner for ticker + manual runs (the fire claim's 300s TTL is outlived by real
    jobs). Callers MUST pair success with ``release_running_job`` in a ``finally``.

    This is the single dedupe owner shared by the ticker's ``_submit_with_guard`` and manual runs
    (``tools/cronjob_tools``): the fire claim alone cannot prevent a double-fire because its TTL (300s) is
    routinely outlived by real jobs, after which a manual ``cronjob(action='run')`` would claim successfully
    and run the same job concurrently (idea from #53395 by @izumi0uu).
    Registration also makes the run visible to ``get_running_job_ids`` (the gateway shutdown drain, #60432)
    and ``mark_running_jobs_interrupted``.
    """
    with _running_lock:
        if job_id in _running_job_ids:
            return False
        _running_job_ids.add(job_id)
        # Same critical section as the add: no window where an in-flight id lacks an age the sweep
        # can bound. Sentinel is replaced by the real future once ``pool.submit`` returns.
        _running_since[job_id] = time.time()
        _running_futures[job_id] = _FUTURE_PENDING
        return True


def release_running_job(job_id: str) -> None:
    """Remove ``job_id`` from the in-flight running set (idempotent)."""
    with _running_lock:
        _running_job_ids.discard(job_id)
        _running_since.pop(job_id, None)
        _running_futures.pop(job_id, None)


def _inflight_min_allowance_minutes() -> float:
    """Stale allowance floor (min): ``cron.inflight_max_minutes``, else env escape hatch/default."""
    with contextlib.suppress(Exception):
        _ucfg = load_config() or {}
        _cfg_val = (
            _ucfg.get("cron", {}) if isinstance(_ucfg, dict) else {}
        ).get("inflight_max_minutes")
        if _cfg_val is not None:
            val = float(_cfg_val)
            if val > 0:
                return val
    raw = cron_env_setting("HERMES_CRON_INFLIGHT_MAX_MINUTES").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (ValueError, TypeError):
            logger.warning(
                "Invalid HERMES_CRON_INFLIGHT_MAX_MINUTES=%r; using default %s",
                raw,
                _INFLIGHT_MIN_ALLOWANCE_MINUTES)
    return _INFLIGHT_MIN_ALLOWANCE_MINUTES


# expr -> minutes; cadence never changes, so avoid re-evaluating croniter every tick.
_cron_interval_cache: dict = {}


def _cron_interval_minutes(expr: str) -> Optional[float]:
    """Cron expression cadence (gap between next two fires) in minutes; None -> floor allowance."""
    if expr in _cron_interval_cache:
        return _cron_interval_cache[expr]
    result = None
    with contextlib.suppress(Exception):
        from cron.jobs import _ensure_croniter

        if _ensure_croniter():
            from cron.jobs import croniter as _croniter
            from datetime import datetime

            base = datetime.now()
            it = _croniter(expr, base)
            first = it.get_next(datetime)
            second = it.get_next(datetime)
            gap = (second - first).total_seconds() / 60.0
            result = gap if gap > 0 else None
    _cron_interval_cache[expr] = result
    return result


def _job_interval_minutes(job: dict) -> Optional[float]:
    """Best-effort job interval in minutes (None if unknown / one-shot -> floor). ``schedule`` is
    persisted as a parsed dict; the string path is only a fallback for programmatic callers."""
    with contextlib.suppress(Exception):
        schedule = job.get("schedule")
        if isinstance(schedule, str) and schedule.strip():
            from cron.jobs import parse_schedule

            schedule = parse_schedule(schedule) or {}
        if isinstance(schedule, dict):
            kind = schedule.get("kind")
            if kind == "interval":
                minutes = schedule.get("minutes")
                return float(minutes) if minutes else None
            if kind == "cron":
                return _cron_interval_minutes(str(schedule.get("expr") or ""))
    return None


def get_inflight_guard_stats() -> dict:
    """Probe-visible snapshot; non-zero ``forced_releases`` means a job wedged and was recovered."""
    now = time.time()
    with _running_lock:
        return {
            "running": sorted(_running_job_ids),
            "running_ages_seconds": {
                jid: round(now - started, 1)
                for jid, started in _running_since.items()
            },
            "forced_releases": _forced_release_count,
            "recent_forced_releases": list(_forced_releases)}


def _record_forced_release(job_id: str, name: str, age_seconds: float, allowance_seconds: float) -> None:
    """Persist a countable signal for one forced release (best-effort)."""
    entry = {
        "job_id": job_id,
        "name": name,
        "age_seconds": round(age_seconds, 1),
        "allowance_seconds": round(allowance_seconds, 1),
        "at": _hermes_now().isoformat()}
    with _running_lock:
        _forced_releases.append(entry)
        del _forced_releases[:-_FORCED_RELEASE_HISTORY]
    try:
        path = _get_hermes_home() / "cron" / "inflight_forced_releases.jsonl"
        _ensure_cron_dir(path.parent)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as e:  # never let telemetry break a tick
        logger.debug("Could not append forced-release record: %s", e)


def _latest_executions_for_releasable_claims() -> dict:
    """Latest durable execution per releasable-looking claim (missing/pending/done future), one
    indexed query so the healthy path pays no DB work. Snapshot under _running_lock — iterating
    the set while try_register/release mutate it raises RuntimeError."""
    with _running_lock:
        claim_futures = {job_id: _running_futures.get(job_id) for job_id in _running_job_ids}
    candidates = [
        job_id for job_id, fut in claim_futures.items()
        if fut is None or fut is _FUTURE_PENDING or fut.done()
    ]
    if not candidates:
        return {}
    try:
        from cron.executions import latest_executions as _latest_execs
        return _latest_execs(candidates)
    except Exception:
        return {}


def _row_belongs_to_claim(row: dict, claim_started: float) -> bool:
    """True when the ledger row was claimed at/after this in-memory claim. An older terminal row is
    the PREVIOUS run's (try_register->create_execution window); releasing on it would
    double-dispatch. Unparseable timestamps fail closed (the age path still bounds the claim)."""
    claimed_at = row.get("claimed_at")
    if not claimed_at:
        return False
    try:
        from cron.jobs import _ensure_aware as _ensure_aware_ts
        row_ts = _ensure_aware_ts(datetime.fromisoformat(claimed_at))
        return row_ts.timestamp() >= claim_started
    except (ValueError, TypeError, OSError):
        return False


def _record_stale_release(job: dict, job_id: str, age: float, allowance: float, fut, reason: str) -> None:
    """WARNING log + probe record for one forced release, then ``last_error`` unless the ledger
    already holds the run's real outcome or the job has a finite repeat budget."""
    name = job.get("name") or job_id
    future_state = "pending" if fut is _FUTURE_PENDING else "missing" if fut is None else "finished"
    logger.warning(
        "cron.inflight.forced_release event=forced_release reason=%s job='%s' "
        "id=%s age=%.0fs allowance=%.0fs future=%s — stale in-flight claim "
        "released; the job was skipping every fire with 'already running'",
        reason, name, job_id, age, allowance, future_state)
    _record_forced_release(job_id, name, age, allowance)
    # Ledger already records how the run ended: mark_job_run here would clobber an honest
    # ok status with a synthetic failure or double-write a failure.
    if reason == "ledger-terminal":
        return
    # Age release may lack a ledger row, so last_error is how it surfaces. But a forced release
    # is NOT a real run: never consume a finite repeat budget or let mark_job_run auto-delete.
    repeat = job.get("repeat") or {}
    if isinstance(repeat, dict) and repeat.get("times") is not None:
        logger.warning(
            "cron.inflight.forced_release.job_untouched job='%s' id=%s — "
            "finite-repeat job released without mark_job_run (repeat budget "
            "preserved); row left in place so it re-fires normally",
            name, job_id)
        return
    try:
        mark_job_run(
            job_id, False,
            f"Stale in-flight claim force-released after {age / 60:.1f}m "
            f"(allowance {allowance / 60:.1f}m); previous run never released "
            f"the scheduler in-flight guard")
    except Exception as e:
        logger.warning("Could not record forced release for job %s: %s", job_id, e)


def sweep_stale_inflight(due_jobs: Optional[list] = None) -> list:
    """Force-release in-flight claims that can no longer be making progress; returns released ids.

    Stale = older than ``max(2 * interval, floor)`` AND (no live future — submit path hung before
    ``pool.submit`` returned — or finished without discarding the id) — or the claim's OWN ledger
    row is terminal regardless of age. Each release logs WARNING ``event=forced_release``, bumps
    the probe counter, mirrors JSONL, and writes ``last_error``.
    """
    global _forced_release_count

    by_id = {j.get("id"): j for j in (due_jobs or []) if isinstance(j, dict)}
    floor_seconds = _inflight_min_allowance_minutes() * 60.0
    now = time.time()
    stale: list = []
    from cron.executions import _TERMINAL_STATES as _terminal_states

    _latest = _latest_executions_for_releasable_claims()
    # Compute intervals OUTSIDE _running_lock so croniter doesn't block try_register/release.
    _intervals = {jid: _job_interval_minutes(j) for jid, j in by_id.items()}

    with _running_lock:
        for job_id in list(_running_job_ids):
            started = _running_since.get(job_id)
            if started is None:
                # Claim predates this guard — adopt it; sweepable one allowance from now.
                _running_since[job_id] = now
                continue
            age = now - started
            interval_minutes = _intervals.get(job_id)
            allowance = floor_seconds
            if interval_minutes:
                allowance = max(allowance, 2.0 * interval_minutes * 60.0)
            fut = _running_futures.get(job_id)
            if fut is _FUTURE_PENDING:
                # Submit path hung before ``pool.submit`` returned — the wedge class; release it.
                pass
            elif fut is not None and not fut.done():
                continue  # genuinely still executing
            # Ledger reconciliation: a terminal row belonging to THIS claim proves it stale even
            # inside its age allowance. Row must be this claim's, else a recurring job's previous
            # run would double-dispatch a fresh claim.
            latest = _latest.get(job_id)
            if (
                latest is not None
                and latest.get("status") in _terminal_states
                and _row_belongs_to_claim(latest, started)
            ):
                reason = "ledger-terminal"
            elif age >= allowance:
                reason = "age"
            else:
                continue
            _running_job_ids.discard(job_id)
            _running_since.pop(job_id, None)
            _running_futures.pop(job_id, None)
            _forced_release_count += 1
            stale.append((job_id, age, allowance, fut, reason))

    for job_id, age, allowance, fut, _reason in stale:
        _record_stale_release(by_id.get(job_id) or {}, job_id, age, allowance, fut, _reason)
    return [s[0] for s in stale]


def mark_running_jobs_interrupted(
    reason: str, *, only_owners: Optional[set] = None,
) -> list:
    """Best-effort: mark every in-flight cron job interrupted; returns the job IDs marked.

    Called by gateway shutdown right after ``process_registry.kill_all()``: a job whose tool was
    killed must never report success. ``only_owners`` (``(job_id, fire_owner)`` pairs) restricts
    marking. Tokens go into ``_interrupted_job_ids`` BEFORE ``last_status`` is written so
    ``run_one_job`` sees them.
    """
    with _running_lock:
        restart_safe_waiters = set(_restart_safe_waiter_job_ids)
        active_fires = [
            (token, job_id, owner, profile_home)
            for job_id, executions in _running_fire_owners.items()
            if job_id not in restart_safe_waiters
            for token, (owner, profile_home) in executions.items()
        ]
        if only_owners is not None:
            active_fires = [fire for fire in active_fires if (fire[1], fire[2]) in only_owners]
        registered_ids = {job_id for _t, job_id, _o, _p in active_fires}
        if only_owners is None:
            active_fires.extend(
                (None, job_id, None, _get_hermes_home())
                for job_id in (
                    _running_job_ids - registered_ids - restart_safe_waiters
                )
            )
        _interrupted_job_ids.update(
            token if token is not None else job_id
            for token, job_id, _owner, _profile_home in active_fires
        )
    marked = []
    for _token, job_id, fire_owner, profile_home in active_fires:
        if not fire_owner:
            logger.warning(
                "Job '%s' interrupted before its durable fire owner was registered; "
                "leaving persisted state untouched",
                job_id)
            # Still report it: shutdown uses the returned IDs for the interrupted-cron notice. The
            # in-memory flag WAS recorded above; only the persisted last_status write is skipped.
            # See #82232.
            marked.append(job_id)
            continue
        try:
            with use_cron_store(profile_home):
                if mark_job_run(
                    job_id, False, reason, expected_fire_owner=fire_owner):
                    marked.append(job_id)
        except Exception as e:
            logger.warning("Failed to mark job %s interrupted: %s", job_id, e)
    return marked


def _is_interrupted(job_id: str, token: Optional[object] = None) -> bool:
    """Non-destructive peek: has shutdown marked THIS execution interrupted? Used before deciding
    what to deliver; does not clear the flag (the authoritative pre-``last_status`` check needs it).
    ``token`` scopes to one execution so a fresh run reusing the job ID isn't poisoned."""
    with _running_lock:
        if token is not None and token in _interrupted_job_ids:
            return True
        return job_id in _interrupted_job_ids


def _consume_interrupted_flag(job_id: str, token: Optional[object] = None) -> bool:
    """Return True and clear the flag if shutdown marked THIS execution interrupted. Called right
    before ``last_status`` is written; consuming stops the flag leaking into a later run."""
    with _running_lock:
        hit = False
        if token is not None and token in _interrupted_job_ids:
            _interrupted_job_ids.discard(token)
            hit = True
        if job_id in _interrupted_job_ids:
            _interrupted_job_ids.discard(job_id)
            hit = True
        return hit


def _inactivity_watchdog_loop(
    *, get_idle_seconds: Callable[[], float], limit_s: float, poll_s: float, stop: threading.Event,
    future_done: Callable[[], bool],
) -> bool:
    """Poll idle time until limit (-> True), stop, or the future completes (-> False). Uses
    ``threading.Event.wait``, not asyncio, so a blocked event loop cannot disable the watchdog.

    Driven by ``threading.Event.wait`` (a kernel timeout), not asyncio, so a blocked event-loop /
    ``run_job`` thread cannot disable this watchdog the way ``asyncio.sleep`` / ``wait_for`` would (family A
    of #94285 — the 4118s-idle-on-a-600s-limit cron hang). Returns True when *limit_s* of inactivity was
    observed.
    """
    while not stop.wait(poll_s):
        if future_done():
            return False
        try:
            idle = float(get_idle_seconds() or 0.0)
        except Exception:
            idle = 0.0
        if idle >= limit_s:
            return True
    return False


def _cron_inactivity_seconds() -> float:
    """Parse HERMES_CRON_TIMEOUT (seconds). 0 = unlimited; bad input = 600. Shared by the
    inactivity monitor and the cwd-lock bound so they can't drift: the lock bound must stay >= the
    inactivity limit or waiters fail while a healthy holder runs."""
    raw = cron_env_setting("HERMES_CRON_TIMEOUT").strip()
    if not raw:
        return 600.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid HERMES_CRON_TIMEOUT=%r; using default 600s", raw)
        return 600.0


def _get_parallel_pool(max_workers: Optional[int]) -> concurrent.futures.ThreadPoolExecutor:
    """Return (or create) the persistent parallel pool."""
    global _parallel_pool, _parallel_pool_max_workers
    if _parallel_pool is None or _parallel_pool_max_workers != max_workers:
        if _parallel_pool is not None:
            _parallel_pool.shutdown(wait=False, cancel_futures=False)
        _parallel_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cron-parallel")
        _parallel_pool_max_workers = max_workers
    return _parallel_pool


def _shutdown_parallel_pool() -> None:
    """Shut down the persistent pool on process exit."""
    global _parallel_pool, _parallel_pool_max_workers
    if _parallel_pool is not None:
        _parallel_pool.shutdown(wait=True, cancel_futures=False)
        _parallel_pool = None
        _parallel_pool_max_workers = None


atexit.register(_shutdown_parallel_pool)
# Per-fire usage audit log; resolves via _get_hermes_home() so profile-scoped paths work.
def _usage_audit_path() -> Path:
    return _get_hermes_home() / "cron" / "usage_audit.jsonl"


def _utcnow_iso_ms() -> str:
    """RFC3339 UTC timestamp with millisecond precision and 'Z' suffix."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _write_usage_audit(record: dict) -> None:
    """Append one JSONL line to cron/usage_audit.jsonl. NEVER raises — a logger bug must not
    break cron jobs (the whole write is inside one try)."""
    try:
        path = _usage_audit_path()
        _ensure_cron_dir(path.parent)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning("usage_audit write failed: %s", e)


def _interpreter_shutting_down(exc: Optional[BaseException] = None) -> bool:
    """True when the interpreter is finalizing (tick fired during gateway teardown): concurrent.
    futures/asyncio refuse new work, so delivery attempts only pollute errors.log — callers skip
    with a warning. ``exc`` lets an already-raised scheduling error count as a shutdown signal.

    A cron tick can fire while the gateway is tearing down — SIGTERM from ``hermes update`` / ``hermes
    gateway stop`` / systemd restart, or an OOM-kill. Once finalization starts, ``concurrent.futures``
    refuses new work with ``RuntimeError: cannot schedule new futures after interpreter shutdown`` and
    asyncio's default executor is gone, so *any* attempt to schedule delivery (live-adapter,
    ``asyncio.run``, or a fresh pool) is doomed and only pollutes ``errors.log`` with a traceback. See
    #55924, #58720.
    """
    from tools.interpreter_shutdown import interpreter_shutting_down

    return interpreter_shutting_down(exc)


# Module override hook for tests / emergency monkeypatches.
_hermes_home: Path | None = None


def _get_hermes_home() -> Path:
    """Hermes home at call time (honouring the test override). Cron is per-profile: never freeze
    this at import or anchor it at the shared default root — either breaks profile isolation.

    Cron is per-profile by design (#4707): the in-process ticker runs inside a profile-scoped gateway, so
    resolving the active HERMES_HOME at call time means a profile's jobs are stored AND executed under that
    profile's home (its .env, config.yaml, scripts, skills).
    """
    return _hermes_home or get_hermes_home()


def _get_lock_paths() -> tuple[Path, Path]:
    """Resolve cron lock paths at call time so profile/env changes are honored."""
    hermes_home = _get_hermes_home()
    lock_dir = hermes_home / "cron"
    return lock_dir, lock_dir / ".tick.lock"


# Errnos that mean "another ticker (or manual tick) holds the tick lock", as opposed to a real failure
# opening/locking the file. Everything else — most importantly EMFILE/ENFILE (fd exhaustion, #87644) and
# EACCES on open() — must be surfaced, never swallowed as lock contention.
def _is_lock_contention_errno(err: OSError) -> bool:
    """True when *err* from the lock syscall means another ticker holds the lock (POSIX flock:
    EWOULDBLOCK/EAGAIN, EACCES on some NFS; msvcrt.locking: EACCES/EDEADLK). Everything else —
    notably EMFILE/ENFILE and EACCES on open() — must be surfaced, never swallowed as contention."""
    if err.errno is None:
        return False
    if fcntl is not None:
        return err.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES)
    if msvcrt is not None:
        return err.errno in (errno.EACCES, errno.EDEADLK)
    return False


def _is_fd_exhaustion_text(text: str) -> bool:
    """Text half of _is_fd_exhaustion (shared with the CLI hint)."""
    lowered = text.lower()
    return "too many open files" in lowered or "emfile" in lowered


def _is_fd_exhaustion(exc: BaseException) -> bool:
    """True when *exc* indicates fd exhaustion: EMFILE/ENFILE errno, or the "Too many open files"
    wording for wrapped exceptions (load_jobs wraps the OSError in a RuntimeError).

    See #87644.
    """
    if isinstance(exc, OSError) and exc.errno in (errno.EMFILE, errno.ENFILE):
        return True
    return _is_fd_exhaustion_text(str(exc))


def _reclaim_fds_best_effort() -> None:
    """Best-effort fd reclamation: gc.collect() closes file objects stuck in reference cycles;
    apply_nofile_soft_limit() raises the RLIMIT_NOFILE soft limit for headroom. Never raises.

    The cron FD-leak family (#60859, #79742, #80792) leaks descriptors from abandoned workers/sessions. Two
    safe, idempotent levers:
    """
    with contextlib.suppress(Exception):
        import gc

        gc.collect()
    with contextlib.suppress(Exception):
        from hermes_cli.resource_limits import apply_nofile_soft_limit

        apply_nofile_soft_limit(None)


def drain_delivery_queue(adapters, loop) -> int:
    """Send queued worker results through this gateway's live adapters."""
    from cron.delivery_queue import _path, drain

    # Only restart-safe workers create the queue file.  Every gateway (macOS,
    # Windows, launchd, Docker) runs this housekeeping tick, so skip the sqlite
    # open/create entirely until a worker has actually queued something.
    if not _path().exists():
        return 0
    return drain(
        lambda queued_job, queued_content, queued_for_failure: _deliver_result(
            queued_job,
            queued_content,
            adapters=adapters,
            loop=loop,
            for_failure=queued_for_failure,
        )
    )


_DEFAULT_SCRIPT_TIMEOUT = 3600  # seconds (1 hour)
# Backward-compatible module override used by tests and emergency monkeypatches.
_SCRIPT_TIMEOUT = _DEFAULT_SCRIPT_TIMEOUT
_RUN_CLAIM_HEARTBEAT_SECONDS = 60.0
_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS = _RUN_CLAIM_HEARTBEAT_SECONDS * 3


def _cron_cleanup_timeout_seconds() -> float:
    """Return the wall-clock bound for cron post-run cleanup."""
    default = 10.0
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("cleanup_timeout_seconds")
        if configured is not None:
            timeout = float(configured)
            if timeout >= 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron cleanup timeout from config: %s", exc)
    return default


def _target_mirror_eligible(
    job: dict,
    target: dict,
    *,
    global_mirror: bool,
    origin_match: Optional[bool] = None,
) -> bool:
    """Whether a resolved delivery target may receive the transcript mirror.

    The June origin-scoping refactor gated mirroring on target == origin,
    which correctly excluded broadcasts but also silenced two legitimate
    conversation shapes — both hit by script-provisioned ("managed") crons,
    which never capture an origin (``_origin_from_env`` only fires for jobs
    created from a live gateway chat):

    - ``origin_fallback``: ``deliver=origin`` with no captured origin resolves
      to the home channel — the user's primary conversation standing in for
      the origin, not a broadcast. Eligible under the same flags as a true
      origin target. (Field report 2026-08-17: brief delivered to the Slack
      DM, mirror silently skipped, reply hit a context-less session.)
    - ``explicit``: a ``platform:chat_id`` target is eligible ONLY when the
      job itself opts in via ``attach_to_session: true`` — the job author
      declaring this target a conversation (managed per-user DM briefings).
      The global ``cron.mirror_delivery`` flag never activates explicit
      targets: it must not start writing transcript entries into arbitrary
      explicitly-addressed chats (shared channels, other users' DMs).

    Broadcast expansions (``all``, bare-platform home targets) carry no
    provenance tag and are never eligible — unchanged invariant.

    ``origin_match`` lets the caller pass a precomputed
    ``_target_matches_origin`` result (``_deliver_result`` already computes it
    for the same target); when ``None`` it is computed here so tests and
    future callers stay self-contained.
    """
    if origin_match is None:
        origin = _resolve_origin(job) or {}
        origin_match = _target_matches_origin(
            origin, target.get("platform", ""), target.get("chat_id", ""),
            target.get("thread_id"),
        )
    if origin_match:
        return True
    resolved_from = target.get("_resolved_from")
    if resolved_from == "origin_fallback":
        # Same activation rules as an origin target: per-job attach wins,
        # else the global flag. This deliberately restates the precedence
        # _cron_mirror_delivery_enabled encodes (keep the two in sync): the
        # sole production caller pre-merges it into `global_mirror`, but the
        # helper must stay correct standalone — a per-job False must beat a
        # raw global True for any caller that does not pre-merge.
        per_job = job.get("attach_to_session")
        if isinstance(per_job, bool):
            return per_job
        return bool(global_mirror)
    if resolved_from == "explicit":
        return job.get("attach_to_session") is True
    return False


def _inchannel_seed_allowed(*, is_dm: bool, user_id: Optional[str]) -> bool:
    """Whether the flat in_channel session seed may run for a target.

    Group-channel session keys are user-isolated
    (``…:group:<chat_id>:<user_id>`` — see _seed_cron_channel_session); a
    seed without a real user_id would create an orphan session that no
    inbound reply ever resolves to, which is worse than no seed (the plain
    mirror can still land if a session exists). DM keys don't embed
    user_id, so DM targets are always seedable. Origin-captured jobs carry
    the scheduler's user_id; origin-less managed jobs typically don't, and
    their group-channel targets must fall back to the plain mirror.
    """
    return bool(is_dm or user_id)


def _maybe_mirror_cron_delivery(
    job: dict,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    enabled: bool = False,
) -> None:
    """Best-effort mirror of a cron delivery into the origin chat's session.

    No-op unless ``enabled`` (resolved once by the caller, and already scoped to
    the origin target — see ``_target_matches_origin``). Reuses the shipped
    ``mirror_to_session`` so cron rides exactly the same path that interactive
    ``send_message`` mirroring already uses, including passing ``user_id`` so a
    per-user-isolated group chat resolves to the exact member who scheduled the
    job (parity with ``send_message``). All failures are swallowed — a delivery
    that succeeded must never be reported as failed because the transcript
    mirror hit a problem.

    Because the caller only enables this for the target that equals the job's
    origin conversation, the session is expected to exist (the job was born in
    that session). A missing session therefore indicates an origin-less /
    fan-out delivery that should not have been mirrored anyway, and is treated
    as a silent no-op — never a synthetic session is created.
    """
    if not enabled:
        return
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.mirror import mirror_to_session

        # Mirror as a USER turn with a labelled prefix, NOT an assistant turn.
        # The brief is not the agent speaking; an assistant-role mirror lands as
        # assistant→assistant after the agent's last turn and breaks strict
        # alternation (issue #2221, the exact failure #2313 removed). A
        # user-role turn collapses safely via repair_message_sequence's
        # consecutive-user merge on every provider, and the prefix preserves the
        # "this came from cron" context that the dropped SQLite mirror metadata
        # would otherwise lose on replay.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=thread_id,
            user_id=user_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': mirrored delivery into %s:%s session transcript",
                job.get("id", "?"), platform_name, chat_id,
            )
        else:
            logger.debug(
                "Job '%s': delivery mirror skipped for %s:%s "
                "(no matching gateway session — cold start)",
                job.get("id", "?"), platform_name, chat_id,
            )
    except Exception as e:
        logger.debug(
            "Job '%s': delivery mirror failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )


def _open_continuable_cron_thread(
    job: dict,
    adapter,
    chat_id: str,
    loop,
) -> Optional[str]:
    """Open a dedicated thread for a continuable cron job (thread-preferred).

    Returns the new ``thread_id`` on success, or ``None`` when the platform has
    no thread primitive (WhatsApp/Signal/SMS) or creation failed — the ``None``
    return is the caller's signal to fall back to the origin-DM mirror, the same
    open-thread-or-fallback shape as ``GatewayRunner._process_handoff``. Reuses
    the shipped ``adapter.create_handoff_thread``; no new adapter surface.
    """
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None
    task_name = job.get("name") or job.get("id", "cron")
    thread_name = f"Hermes — {task_name}"
    try:
        from agent.async_utils import safe_schedule_threadsafe

        coro = create_thread(str(chat_id), thread_name)
        future = safe_schedule_threadsafe(coro, loop)  # type: ignore[arg-type]
        if future is None:
            return None
        new_thread_id = future.result(timeout=30)
        return str(new_thread_id) if new_thread_id else None
    except Exception as e:
        logger.debug(
            "Job '%s': create_handoff_thread failed on %s — falling back to "
            "DM-session mirror: %s",
            job.get("id", "?"), getattr(adapter, "name", "?"), e,
        )
        return None


def _seed_cron_thread_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    thread_id: str,
    mirror_text: str,
    chat_name: Optional[str] = None,
    is_dm: bool = False,
    scope_id: Optional[str] = None,
) -> None:
    """Seed the freshly-opened cron thread's session with the brief.

    Without this the brief is *visible* in the new thread but absent from any
    transcript, so the user's first reply in-thread would hit a session with no
    record of it ("what is Task #2?"). We create the thread-keyed session (the
    same key the user's reply will resolve to — ``build_session_key`` keys
    threads as participant-shared, so no ``user_id`` is needed) and append the
    brief as an assistant turn via the shipped ``mirror_to_session``.

    ``scope_id`` is the workspace/server scope (Slack team id).
    ``build_session_key`` embeds it in every Slack key, so a scoped reply's
    key carries it — the seed must reproduce it or the seeded row is
    unreachable (the scope-less flat-seed sibling of the is_dm keying bug).
    Best-effort None for platforms without scope.

    ``is_dm`` selects the seeded ``chat_type``: a thread under a DM must seed
    ``chat_type="dm"`` because the user's in-thread DM reply arrives with
    chat_type="dm" and ``build_session_key`` routes DM threads through the DM
    arm (``...:dm:<chat>:<thread>``) — a "thread"-typed seed lands in
    ``...:thread:<chat>:<thread>``, a row no DM reply ever resolves to
    (continuation amnesia, Alice live 2026-08-20, job 8e21a957b77b). Channel
    threads keep ``chat_type="thread"`` (their replies really do arrive as
    threads). Same sibling-lane class as the flat seed's ``is_dm``
    (dcca9d8cfe).

    Mirrors ``GatewayRunner._process_handoff``'s seed step, but standalone:
    cron reaches the live ``SessionStore`` through the adapter's
    ``_session_store`` handle rather than the gateway object. Best-effort — a
    delivery that already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        seeded_session_id: Optional[str] = None
        session_store = getattr(adapter, "_session_store", None)
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                # Discord thread destinations must key on the thread's OWN id
                # to match how the Discord adapter keys organic in-thread
                # messages (chat_id == thread_id). Other platforms (Slack,
                # Telegram) use chat_id == parent_channel for thread messages,
                # so the parent chat_id is correct for them. See the matching
                # guard in GatewayRunner._process_handoff.
                if platform_enum == Platform.DISCORD:
                    seed_chat_id = str(thread_id)
                else:
                    seed_chat_id = str(chat_id)
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=seed_chat_id,
                    chat_name=chat_name,
                    # DM threads key through the DM arm (see docstring); the
                    # reply's chat_type is what the seed must reproduce.
                    chat_type="dm" if is_dm else "thread",
                    user_id="system:cron",
                    user_name="Cron",
                    thread_id=str(thread_id),
                    scope_id=str(scope_id) if scope_id else None,
                )
                # Ensure the thread-keyed session row exists so the mirror has
                # a target and the user's later reply joins the same session.
                # Capture the exact id — the mirror writes into THIS row, not
                # an origin-heuristic rediscovery (which bails on populated
                # chats; same class as the flat-seed live failure 2026-08-19).
                _entry = session_store.get_or_create_session(dest_source)
                seeded_session_id = getattr(_entry, "session_id", None)

        from gateway.mirror import mirror_to_session

        # User-role + labelled prefix (see _maybe_mirror_cron_delivery): the
        # seeded brief must not read as an assistant turn, or the user's first
        # in-thread reply produces assistant→user→... off a phantom assistant
        # message. Pass the seed user_id so the mirror resolves the exact
        # thread-keyed session row we just created.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=str(thread_id),
            user_id="system:cron",
            role="user",
            session_id=seeded_session_id,
        )
        if ok:
            logger.info(
                "Job '%s': opened continuable thread %s on %s:%s and seeded the brief",
                job.get("id", "?"), thread_id, platform_name, chat_id,
            )
        else:
            logger.warning(
                "Job '%s': thread seed did NOT land on %s:%s thread=%s — an "
                "in-thread reply will not see this brief",
                job.get("id", "?"), platform_name, chat_id, thread_id,
            )
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the continuation-
        # amnesia bug (Alice 2026-08-19) — it must be visible in production.
        logger.warning(
            "Job '%s': seeding cron thread session failed for %s:%s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, thread_id, e,
        )


def _seed_cron_channel_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    *,
    is_dm: bool,
    user_id: Optional[str],
    chat_name: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> bool:
    """Seed the FLAT (thread_id=None) session for an ``in_channel`` cron delivery.

    The ``in_channel`` surface (D1/D2) delivers the brief flat into the channel
    with no thread, so the continuation surface is the whole-channel /
    whole-DM session keyed ``thread_id=None`` — the same bucket
    ``reply_in_thread: false`` routes an inbound plain reply to.

    Unlike the thread path, the shipped delivery-mirror alone is NOT sufficient
    here: ``mirror_to_session`` only APPENDS to a session that already EXISTS
    (``_find_session_id`` → no-op when none matches), and a flat channel
    ``(…, None)`` row is only created when a human posts a top-level message the
    bot processes — a ``chat_postMessage`` cron delivery never goes through the
    inbound handler, so the row is usually absent and the mirror silently drops
    the brief (verified live: the brief never landed, the reply had no context).
    So we CREATE the flat session row first, exactly like
    ``_seed_cron_thread_session`` does for threads, then mirror into it.

    The session KEY must match what the user's later inbound reply resolves to
    (``build_session_key``):
    - **Channel** (``chat_type="group"``): key is
      ``…:group:<chat_id>:<user_id>`` — user-isolated — so the seed MUST carry
      the **origin's real ``user_id``** (the member who scheduled the job), NOT
      a synthetic ``system:cron`` id, or the reply keys to a different session.
    - **1:1 DM** (``chat_type="dm"``): the key is ``…:dm:<chat_id>`` and does
      NOT embed ``user_id``, so any ``user_id`` resolves to the same session.
    ``chat_type`` mirrors the inbound handler's own choice
    (``"dm" if is_dm else "group"``, ``adapter.py``), so the seeded key is
    byte-identical to the reply's key.

    Returns True if a seed row was created and the brief mirrored, else False
    (caller falls back to the plain mirror). Best-effort — a delivery that
    already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return False
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        chat_type = "dm" if is_dm else "group"
        session_store = getattr(adapter, "_session_store", None)
        seeded_session_id: Optional[str] = None
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=str(chat_id),
                    chat_name=chat_name,
                    chat_type=chat_type,
                    user_id=str(user_id) if user_id else None,
                    thread_id=None,  # flat — the whole-channel/DM session
                    # Workspace scope: build_session_key embeds it in every
                    # Slack key, so a scoped reply only resolves to this row
                    # when the seed carries it too (see thread-seed docstring).
                    scope_id=str(scope_id) if scope_id else None,
                )
                # Create the flat session row so the mirror has a target and the
                # user's later plain reply joins the SAME session. Capture the
                # exact session id: the mirror must write into THIS row, not
                # re-discover it via origin heuristics (which bail out on
                # populated chats where the flat session coexists with
                # per-message thread sessions — live failure, Alice 2026-08-19).
                _entry = session_store.get_or_create_session(dest_source)
                seeded_session_id = getattr(_entry, "session_id", None)

        from gateway.mirror import mirror_to_session

        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=None,
            user_id=str(user_id) if user_id else None,
            session_id=seeded_session_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': seeded flat in_channel session on %s:%s (chat_type=%s)",
                job.get("id", "?"), platform_name, chat_id, chat_type,
            )
        return bool(ok)
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the "agent has no idea
        # about its own brief" bug (Alice 2026-08-19) — it must be visible in
        # production logs.
        logger.warning(
            "Job '%s': seeding in_channel session failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )
        return False


def _cron_job_origin_log_suffix(job: dict) -> str:
    """Return safe provenance details for security warnings about a cron job.

    The scheduler normally has no live HTTP request object when it detects a
    bad stored ``context_from`` reference. Including the job's saved origin
    makes future probe logs actionable without exposing secrets: platform/chat
    metadata for gateway-created jobs, and optional source-IP fields for API
    surfaces that persist them in origin metadata.
    """
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return ""

    fields = []
    for key in ("platform", "chat_id", "thread_id", "source_ip", "remote", "forwarded_for"):
        value = origin.get(key)
        if value is None:
            continue
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if text:
            fields.append(f"origin_{key}={text[:200]!r}")
    return " " + " ".join(fields) if fields else ""


def _plugin_cron_env_var(platform_name: str) -> str:
    """Return the cron home-channel env var registered by a plugin platform.

    Falls through the platform registry so plugins that set
    ``cron_deliver_env_var`` on their ``PlatformEntry`` get cron delivery
    support without editing this module.
    """
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name.lower())
        if entry and entry.cron_deliver_env_var:
            return entry.cron_deliver_env_var
    except Exception:
        pass
    return ""


def _is_known_delivery_platform(platform_name: str) -> bool:
    """Whether ``platform_name`` is a valid cron delivery target.

    Hardcoded built-ins in ``_KNOWN_DELIVERY_PLATFORMS`` are checked first;
    plugin platforms registered via ``PlatformEntry`` are accepted if they
    provide a ``cron_deliver_env_var``.
    """
    name = platform_name.lower()
    if name in _KNOWN_DELIVERY_PLATFORMS:
        return True
    return bool(_plugin_cron_env_var(name))


def _resolve_home_env_var(platform_name: str) -> str:
    """Return the env var name for a platform's cron home channel.

    Built-in platforms are in ``_HOME_TARGET_ENV_VARS``; plugin platforms are
    resolved from the platform registry.
    """
    name = platform_name.lower()
    env_var = _HOME_TARGET_ENV_VARS.get(name)
    if env_var:
        return env_var
    return _plugin_cron_env_var(name)


def _get_config_home_channel(platform_name: str):
    """Return the persisted ``HomeChannel`` for a platform from gateway config.

    ``/sethome`` declares ``config.yaml`` canonical (it is the only store that
    survives for relay-fronted logical platforms, whose adapters are not
    natively enabled) and mirrors the value into the legacy
    ``<PLATFORM>_HOME_CHANNEL`` env var only as a best-effort compatibility
    shim.  Cron historically read ONLY the env mirror, so a home channel that
    existed solely in config.yaml — e.g. Discord fronted by the relay
    connector, where no ``DISCORD_HOME_CHANNEL`` was ever exported — was
    invisible and jobs silently fell back to local-only.  Reading the
    canonical store here fixes that for every relay-fronted platform at once.
    """
    try:
        from gateway.config import load_gateway_config, Platform

        config = load_gateway_config()
        platform = Platform(platform_name.lower())
        return config.get_home_channel(platform)
    except Exception:
        logger.debug(
            "config home_channel lookup failed for platform %r",
            platform_name, exc_info=True,
        )
        return None


def _env_home_target_chat_id(platform_name: str) -> str:
    """Return the home chat id from the legacy env mirror only (no config)."""
    env_var = _resolve_home_env_var(platform_name)
    if not env_var:
        return ""
    value = os.getenv(env_var, "")
    if not value:
        legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
        if legacy:
            value = os.getenv(legacy, "")
    return value


def _get_home_target_chat_id(platform_name: str) -> str:
    """Return the configured home target chat/room ID for a delivery platform.

    Resolution order: platform env var (legacy mirror, kept first so an
    operator override keeps winning) → legacy env var name → the canonical
    ``home_channel`` block persisted in config.yaml by ``/sethome``.
    """
    value = _env_home_target_chat_id(platform_name)
    if value:
        return value
    home = _get_config_home_channel(platform_name)
    if home is not None and home.chat_id:
        return str(home.chat_id)
    return ""


def _get_home_target_thread_id(platform_name: str) -> Optional[str]:
    """Return the optional thread/topic ID for a platform home target.

    Telegram-only override: ``TELEGRAM_CRON_THREAD_ID`` takes precedence over
    ``TELEGRAM_HOME_CHANNEL_THREAD_ID`` for cron delivery. When topic mode is
    enabled, deliveries that land in the root DM (thread_id unset) end up in
    the system-only lobby where the user cannot reply — the gateway returns
    the lobby reminder and drops ``reply_to_message_id`` (#24409). Pointing
    cron at a dedicated topic via this env var lets replies work as expected
    without changing the lobby invariant.
    """
    env_var = _resolve_home_env_var(platform_name)
    if platform_name.lower() == "telegram":
        cron_thread = os.getenv("TELEGRAM_CRON_THREAD_ID", "").strip()
        if cron_thread:
            return cron_thread
    value = os.getenv(f"{env_var}_THREAD_ID", "").strip() if env_var else ""
    if not value and env_var:
        legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
        if legacy:
            value = os.getenv(f"{legacy}_THREAD_ID", "").strip()
    if value:
        return value
    # Canonical config.yaml fallback — same rationale as
    # _get_home_target_chat_id, and thread affinity only applies when the
    # chat itself resolved from the same config block (an env-provided chat
    # id keeps its env-provided thread semantics).
    if not _env_home_target_chat_id(platform_name):
        home = _get_config_home_channel(platform_name)
        if home is not None and home.thread_id:
            return str(home.thread_id)
    return None


def _iter_home_target_platforms():
    """Iterate built-in + plugin platform names that expose a home channel.

    Used by the ``deliver=origin`` fallback when the job has no origin.
    """
    for name in _HOME_TARGET_ENV_VARS:
        yield name
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.cron_deliver_env_var and entry.name not in _HOME_TARGET_ENV_VARS:
                yield entry.name
    except Exception:
        pass


def _relay_fronted_delivery_platforms(connected: set) -> set:
    """Logical platforms deliverable through a connected relay connector.

    ``get_connected_platforms()`` only sees NATIVELY configured platforms.
    On a relay-fronted deployment (relay in ``config.platforms``, the real
    platform credential living in the connector) the fronted platforms are
    absent from that set although fire-time routing delivers to them via
    ``resolve_delivery_transport`` + ``RelayAdapter.fronts_platform``. This
    keeps validation symmetric with routing by consulting the same
    env-derived deploy stamp (``GATEWAY_RELAY_PLATFORMS``) the live
    adapter's identity set is seeded from. No relay connected -> empty set,
    so native topologies keep the strict credential check unchanged.
    """
    if "relay" not in connected:
        return set()
    try:
        from gateway.relay import relay_fronted_platforms

        return relay_fronted_platforms()
    except Exception:
        logger.debug("relay fronted-platform lookup failed", exc_info=True)
        return set()


def cron_delivery_targets() -> list[dict]:
    """Return the platforms a cron job can auto-deliver to.

    Single source of truth for any UI (dashboard dropdown, etc.) that lets a
    user pick a cron delivery target. A platform is included when it is a valid
    cron delivery platform AND its gateway is configured (enabled + credentials
    present). Each entry reports whether the platform's home target (the
    room/channel cron posts to) is set — a platform can be configured for
    interactive use but still lack the home target an unattended cron job needs.

    Returns a list of dicts: ``{"id", "name", "home_target_set", "home_env_var"}``
    ordered by the gateway's canonical platform order. Callers should always
    prepend the implicit ``local`` option themselves — it needs no config.
    """
    targets: list[dict] = []
    try:
        from gateway.config import load_gateway_config

        gateway_config = load_gateway_config()
        connected = {p.value for p in gateway_config.get_connected_platforms()}
        connected |= _relay_fronted_delivery_platforms(connected)
    except Exception:
        logger.debug("cron_delivery_targets: gateway config unavailable", exc_info=True)
        connected = set()

    for name in _iter_home_target_platforms():
        if name not in connected:
            continue
        if not _is_known_delivery_platform(name):
            continue
        env_var = _resolve_home_env_var(name)
        targets.append(
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "home_target_set": bool(_get_home_target_chat_id(name)),
                "home_env_var": env_var or None,
            }
        )

    # Bot Chat targets: one per local profile. Machine-local by design (the
    # scheduler delivers via a local chat subprocess), so the names listed
    # here are exactly the names that resolve at fire time — no gateway
    # config, no home channel needed.
    try:
        from hermes_cli.profiles import list_profile_names

        for profile_name in list_profile_names():
            targets.append(
                {
                    "id": f"{BOT_CHAT_PLATFORM}:{profile_name}",
                    "name": f"Bot Chat ({profile_name})",
                    "home_target_set": True,
                    "home_env_var": None,
                }
            )
    except Exception:
        logger.debug("cron_delivery_targets: profile listing unavailable", exc_info=True)
    return targets


def _origin_thread_is_stale(origin: dict) -> bool:
    """True when a Slack origin's thread is a stale creation-turn artifact.

    Relay-fronted Slack in thread-per-message mode stamps each top-level
    message's own id as the session thread (a session KEY, not a durable
    location). Jobs persisted before origin capture learned to drop that
    stamp carry it as ``origin.thread_id`` forever. Heuristic that repairs
    them at fire time without touching genuine threads: when the origin
    chat IS the configured Slack home chat (the ``/sethome`` conversation),
    a pinned origin thread is the creation-message artifact — the user's
    delivery expectation for their home conversation is top-level (or the
    home target's own configured thread). Non-home chats keep their
    threads: a job deliberately created inside a working thread stays there.
    """
    if str(origin.get("platform") or "").lower() != "slack":
        return False
    if not origin.get("thread_id"):
        return False
    home_chat = _get_home_target_chat_id("slack")
    return bool(home_chat) and str(origin.get("chat_id")) == str(home_chat)


def _origin_delivery_thread(origin: dict):
    """The thread a deliver=origin job should use, stale stamps dropped."""
    if _origin_thread_is_stale(origin):
        home_thread = _get_home_target_thread_id("slack")
        return home_thread if home_thread else None
    return origin.get("thread_id")


def _resolve_single_delivery_target(job: dict, deliver_value: str) -> Optional[dict]:
    """Resolve one concrete auto-delivery target for a cron job."""

    origin = _resolve_origin(job)

    if deliver_value == "local":
        return None

    # bot-chat[:<profile>] — checked before the generic platform:chat_id
    # split below so the profile-name argument is never misparsed as a
    # chat_id on an unknown platform.
    bot_chat_profile = parse_bot_chat_deliver_token(deliver_value)
    if bot_chat_profile is not None:
        return _resolve_bot_chat_target(job, bot_chat_profile)

    if deliver_value == "origin":
        if origin:
            return {
                "platform": origin["platform"],
                "chat_id": str(origin["chat_id"]),
                "thread_id": _origin_delivery_thread(origin),
                # Resolution provenance for mirror eligibility (see
                # _target_mirror_eligible): this IS the origin conversation.
                "_resolved_from": "origin",
            }
        # Origin missing (e.g. job created via API/script) — try each
        # platform's home channel as a fallback instead of silently dropping.
        for platform_name in _iter_home_target_platforms():
            chat_id = _get_home_target_chat_id(platform_name)
            if chat_id:
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                return {
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": _get_home_target_thread_id(platform_name),
                    # The fallback stands in for the user's primary
                    # conversation (NOT a broadcast) — mirror-eligible so
                    # continuable crons work for script-provisioned jobs
                    # that never captured an origin.
                    "_resolved_from": "origin_fallback",
                }
        return None

    if ":" in deliver_value:
        platform_name, rest = deliver_value.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import (
            prepare_send_message_platforms,
            resolve_send_target,
        )

        prepare_send_message_platforms()
        # pass_unresolved_references: stored jobs have no model in the loop to react
        # to a resolution error, and a target the directory doesn't know
        # (fresh install, platform-native id) used to be handed to the
        # adapter as written. Dropping it here silently loses the job's
        # output.
        chat_id, thread_id, resolution_error = resolve_send_target(
            platform_key, rest, pass_unresolved_references=True
        )
        if resolution_error:
            logger.warning(
                "Invalid cron delivery target '%s': %s",
                deliver_value,
                resolution_error,
            )
            return None

        if (
            thread_id is None
            and platform_key == "slack"
            and origin
            and str(origin.get("platform") or "").lower() == platform_key
            and str(origin.get("chat_id")) == str(chat_id)
            and origin.get("thread_id")
            and not _origin_thread_is_stale(origin)
        ):
            thread_id = origin.get("thread_id")

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
            # Explicit platform:chat target — mirror-eligible only under the
            # job's own attach_to_session opt-in (see _target_mirror_eligible).
            "_resolved_from": "explicit",
        }

    platform_name = deliver_value
    if origin and origin.get("platform") == platform_name:
        chat_id = _get_home_target_chat_id(platform_name)
        if chat_id:
            return {
                "platform": platform_name,
                "chat_id": chat_id,
                "thread_id": _get_home_target_thread_id(platform_name),
            }
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if not _is_known_delivery_platform(platform_name):
        return None
    chat_id = _get_home_target_chat_id(platform_name)
    if not chat_id:
        return None

    return {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": _get_home_target_thread_id(platform_name),
    }


def _get_bot_chat_delivery_timeout() -> int:
    """Timeout for one bot-chat delivery turn (the target bot runs a full
    agent turn on the injected output, so this is minutes, not seconds).

    ``cron.bot_chat_delivery_timeout_seconds`` in config.yaml; default 600.
    """
    try:
        cfg = load_config()
        value = int(cfg.get("cron", {}).get("bot_chat_delivery_timeout_seconds", 600))
        return value if value > 0 else 600
    except Exception:
        return 600


def _deliver_to_bot_chat(job: dict, content: str, profile: str) -> Optional[str]:
    """Deliver job output into a profile's canonical Bot Chat as an inbound turn.

    Runs ``hermes [-p <profile>] chat --in ~ -c "Bot Chat" --create-if-missing
    -Q --query-file <tmp>`` — the exact lane Bot Mode agent-to-agent messages
    use, so the adopt-before-mint canonical-session rules apply and the target
    bot receives the output as a real user-role message it can act on.
    Alternation-safe by construction: this is an inbound turn on the chat
    command lane, not a transcript splice.

    ``profile`` is ``""`` for the job's own profile (subprocess inherits this
    scheduler's HERMES_HOME) or a validated local profile name.  Returns None
    on success or an error string for ``last_delivery_error``.
    """
    import shutil as _shutil
    import tempfile

    job_id = job.get("id", "?")
    job_name = job.get("name", job_id)

    hermes_bin = _shutil.which("hermes")
    if hermes_bin:
        argv = [hermes_bin]
    else:
        try:
            import importlib.util as _ilu

            if _ilu.find_spec("hermes_cli") is not None:
                argv = [sys.executable, "-m", "hermes_cli.main"]
            else:
                return "bot-chat delivery failed: hermes CLI not resolvable"
        except Exception:
            return "bot-chat delivery failed: hermes CLI not resolvable"

    env = os.environ.copy()
    if profile:
        argv += ["-p", profile]
        # -p owns profile resolution in the child; a leftover HERMES_HOME
        # from THIS scheduler's profile must not shadow it.
        env.pop("HERMES_HOME", None)

    # The prefix tells the receiving bot this is scheduled output, not the
    # human typing — mirrors the Bot Mode sender-attribution convention.
    message = (
        f'[Cronjob "{job_name}" output — scheduled job, not the user. '
        f"Review it, act on anything that needs action, and summarize "
        f"for the chat.]\n\n{content}"
    )

    query_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", prefix="hermes-cron-botchat-",
            delete=False,
        ) as fh:
            fh.write(message)
            query_file = fh.name

        argv += [
            "chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing",
            "-Q", "--query-file", query_file,
        ]

        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_get_bot_chat_delivery_timeout(),
            env=env,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-500:]
            msg = (
                f"bot-chat delivery to profile "
                f"'{profile or '(own)'}' failed (exit {result.returncode})"
                + (f": {tail}" if tail else "")
            )
            logger.warning("Job '%s': %s", job_id, msg)
            return msg
        logger.info(
            "Job '%s': delivered to Bot Chat of profile '%s'",
            job_id, profile or "(own)",
        )
        return None
    except subprocess.TimeoutExpired:
        msg = (
            f"bot-chat delivery to profile '{profile or '(own)'}' timed out "
            f"after {_get_bot_chat_delivery_timeout()}s (the bot's turn may "
            "still complete; raise cron.bot_chat_delivery_timeout_seconds if "
            "this recurs)"
        )
        logger.warning("Job '%s': %s", job_id, msg)
        return msg
    except Exception as e:
        msg = f"bot-chat delivery failed: {str(e) or type(e).__name__}"
        logger.warning("Job '%s': %s", job_id, msg, exc_info=True)
        return msg
    finally:
        if query_file:
            try:
                os.unlink(query_file)
            except OSError:
                pass


def _normalize_deliver_value(deliver) -> str:
    """Normalize a stored/submitted ``deliver`` value to its canonical string form.

    The contract is that ``deliver`` is a string (``"local"``, ``"origin"``,
    ``"telegram"``, ``"telegram:-1001:17"``, or comma-separated combinations).
    Historically some callers — MCP clients passing an array, direct edits of
    ``jobs.json``, or stale code paths — have stored a list/tuple like
    ``["telegram"]``.  ``str(["telegram"])`` would serialize to the literal
    string ``"['telegram']"``, which is not a known platform and fails
    resolution silently.  Flatten lists/tuples into a comma-separated string
    so both forms work.  Returns ``"local"`` for anything falsy.
    """
    if deliver is None or deliver == "":
        return "local"
    if isinstance(deliver, (list, tuple)):
        parts = [str(p).strip() for p in deliver if str(p).strip()]
        return ",".join(parts) if parts else "local"
    return str(deliver)


# Routing intent tokens — resolved at fire time, not create time, so a
# job created before Telegram was wired up will pick up Telegram once it
# comes online.  ``all`` expands into the set of connected platforms
# (those with a configured home chat_id) in _expand_routing_tokens.
_ROUTING_TOKENS = frozenset({"all"})

# Pseudo-platform for delivering job output INTO a profile's canonical
# "Bot Chat" session as a real inbound turn (the bot sees it, runs a turn,
# and can respond — Bot Mode's agent-to-agent lane, not a transcript
# mirror).  ``bot-chat`` targets the job's own profile; ``bot-chat:<name>``
# targets a named profile on THIS machine.  Deliberately excluded from the
# ``all`` routing token: ``all`` fans out to messaging home channels, and a
# bot-chat delivery costs a full agent turn.
BOT_CHAT_PLATFORM = "bot-chat"


def parse_bot_chat_deliver_token(part: str) -> Optional[str]:
    """Return the target profile for a ``bot-chat[:<name>]`` deliver token.

    Returns ``""`` for the bare token (the job's own profile), the profile
    name for the explicit form, or ``None`` when ``part`` is not a bot-chat
    token at all.  Case-insensitive on the token; the profile name is
    normalized by the profile layer at resolve time.
    """
    raw = (part or "").strip()
    lowered = raw.lower()
    if lowered == BOT_CHAT_PLATFORM:
        return ""
    prefix = BOT_CHAT_PLATFORM + ":"
    if lowered.startswith(prefix):
        return raw[len(prefix):].strip()
    return None


def _resolve_bot_chat_target(job: dict, profile_arg: str) -> Optional[dict]:
    """Resolve a bot-chat deliver token to a concrete delivery target.

    ``profile_arg`` is ``""`` for the job's own profile (the HERMES_HOME
    this scheduler runs under — machine-local and self-referential, so no
    ``-p`` flag is needed at send time) or an explicit profile name that
    must exist in THIS machine's profile root.  Cross-machine delivery is
    intentionally unsupported: names resolve only against the local
    ``~/.hermes/profiles/`` tree, so same-named profiles on other gateways
    can never be targeted by accident.
    """
    if not profile_arg:
        # Own profile: chat subprocess inherits HERMES_HOME, no name needed.
        return {"platform": BOT_CHAT_PLATFORM, "chat_id": "", "thread_id": None}
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists

        canon = normalize_profile_name(profile_arg)
        if not profile_exists(canon):
            logger.warning(
                "Job '%s': bot-chat delivery profile '%s' not found on this "
                "machine — skipping target",
                job.get("id", "?"), profile_arg,
            )
            return None
        return {"platform": BOT_CHAT_PLATFORM, "chat_id": canon, "thread_id": None}
    except Exception:
        logger.warning(
            "Job '%s': failed to resolve bot-chat profile '%s'",
            job.get("id", "?"), profile_arg, exc_info=True,
        )
        return None


def _expand_routing_tokens(part: str) -> List[str]:
    """Expand a routing-intent token to concrete platform names.

    ``all`` expands to every platform in ``_iter_home_target_platforms()``
    that has a configured home chat_id right now.  Unknown / non-token
    values pass through unchanged as a single-element list, so the caller
    can treat every token uniformly.
    """
    token = part.lower()
    if token not in _ROUTING_TOKENS:
        return [part]
    expanded: List[str] = []
    for platform_name in _iter_home_target_platforms():
        if _get_home_target_chat_id(platform_name):
            expanded.append(platform_name)
    return expanded


def _resolve_delivery_targets(job: dict) -> List[dict]:
    """Resolve all concrete auto-delivery targets for a cron job.

    Accepts the legacy comma-separated ``deliver`` string plus the
    ``all`` routing-intent token, which expands to every platform with
    a configured home channel.  Tokens may be combined with explicit
    targets: ``origin,all`` and ``all,telegram:-100:17`` both work.
    Duplicate (platform, chat_id, thread_id) tuples are collapsed by the
    existing dedup pass.
    """
    deliver = _normalize_deliver_value(job.get("deliver", "local"))
    if deliver == "local":
        return []

    raw_parts = [p.strip() for p in deliver.split(",") if p.strip()]

    # Expand routing intents.
    parts: List[str] = []
    for raw in raw_parts:
        parts.extend(_expand_routing_tokens(raw))

    seen = {}
    targets = []
    for part in parts:
        target = _resolve_single_delivery_target(job, part)
        if target:
            key = (target["platform"].lower(), str(target["chat_id"]), target.get("thread_id"))
            if key not in seen:
                seen[key] = target
                targets.append(target)
            else:
                # OR-merge resolution provenance on dedup: "origin,all" (either
                # order) resolving to the same chat must keep the
                # origin/origin_fallback tag — a mirror-eligible token must not
                # lose eligibility to token order (see _target_mirror_eligible).
                kept = seen[key]
                if _MIRROR_PROVENANCE_RANK.get(str(target.get("_resolved_from") or ""), 0) > \
                        _MIRROR_PROVENANCE_RANK.get(str(kept.get("_resolved_from") or ""), 0):
                    kept["_resolved_from"] = target.get("_resolved_from")
    return targets


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    targets = _resolve_delivery_targets(job)
    return targets[0] if targets else None


# Media extension sets — audio routing is centralized in gateway.platforms.base
# via should_send_media_as_audio() so Telegram-specific rules stay in one place.
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(
    adapter,
    chat_id: str,
    media_files: list,
    metadata: dict | None,
    loop,
    job: dict,
    platform=None,
) -> list:
    """Send extracted MEDIA files as native platform attachments via a live adapter.

    Routes each file to the appropriate adapter method (send_voice, send_image_file,
    send_video, send_document) based on file extension — mirroring the routing logic
    in ``BasePlatformAdapter._process_message_background``.

    Returns a list of per-file error strings (empty when every attachment
    delivered). Callers surface these into the job's delivery errors so a
    dropped attachment is visible in ``last_error``/run status instead of
    only in the gateway log (the silent-drop half of the manual-run
    attachment bug: text delivered, file vanished, job marked ok).
    """
    from pathlib import Path

    from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio

    errors: list = []
    requested = [(str(p), v) for p, v in (media_files or [])]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Report paths the safety filter dropped: the model referenced them in
    # MEDIA: tags but they will never be sent (missing file, denied prefix,
    # or strict-mode policy miss).
    kept = {p for p, _ in media_files}
    for raw_path, _v in requested:
        try:
            from gateway.platforms.base import validate_media_delivery_path

            if validate_media_delivery_path(raw_path) not in kept:
                errors.append(
                    f"attachment dropped by media path policy: {raw_path}"
                )
        except Exception:
            errors.append(f"attachment dropped by media path policy: {raw_path}")

    for media_path, _is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            route_platform = platform if platform is not None else getattr(adapter, "platform", None)
            if should_send_media_as_audio(route_platform, ext, is_voice=_is_voice):
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=metadata)

            from agent.async_utils import safe_schedule_threadsafe
            future = safe_schedule_threadsafe(coro, loop)
            if future is None:
                msg = f"cannot send media {media_path}: gateway loop unavailable"
                logger.warning("Job '%s': %s", job.get("id", "?"), msg)
                errors.append(msg)
                return errors
            try:
                # Large attachments (long TTS audio, concatenated recordings,
                # big exports) can legitimately exceed a fixed 30s upload
                # window. Configurable, matching the other cron timeouts
                # (cron.media_send_timeout_seconds in config.yaml, or the
                # HERMES_CRON_MEDIA_SEND_TIMEOUT env override).
                result = future.result(timeout=_get_media_send_timeout())
            except TimeoutError:
                future.cancel()
                raise
            if result and not getattr(result, "success", True):
                msg = (
                    f"media send failed for {media_path}: "
                    f"{getattr(result, 'error', 'unknown')}"
                )
                logger.warning("Job '%s': %s", job.get("id", "?"), msg)
                errors.append(msg)
        except Exception as e:
            # Argument-less exceptions (notably TimeoutError, the most likely
            # failure on this path) have an empty str(), which would render
            # the reason as nothing at all. Fall back to the class name.
            msg = (
                f"failed to send media {media_path}: {str(e) or type(e).__name__}"
            )
            logger.warning("Job '%s': %s", job.get("id", "?"), msg)
            errors.append(msg)
    return errors


def _confirm_adapter_delivery(send_result) -> bool:
    """Return True only if ``send_result`` unambiguously confirms delivery.

    A live adapter that returns ``None`` (e.g. a swallowed exception, a busy
    platform, or a code path that returns early without producing a
    ``SendResult``) must NOT be treated as success — doing so causes the
    scheduler to log ``"delivered to <chat> via live adapter"`` while the
    gateway never actually sees the message (#47056).

    Likewise, an object missing a ``success`` attribute (e.g. a bare ``dict``
    or a partial mock) is a contract violation: it does not actually tell us
    whether the send succeeded.  Require an explicit, truthy ``success``
    attribute to count as confirmed.
    """
    if send_result is None:
        return False
    if not hasattr(send_result, "success"):
        return False
    return bool(getattr(send_result, "success"))


def _is_channel_dm_topic(
    runtime_adapter: Any,
    chat_id: Any,
    loop: Any,
    job_id: str,
) -> bool:
    """Decide whether an (already-ambiguous) Telegram topic target is a genuine
    Bot API *channel* Direct-Messages topic (route via
    ``direct_messages_topic_id``) rather than a forum-style topic in a private
    chat (route via ``message_thread_id``).

    Callers gate this on the ambiguous shape first
    (``telegram:<positive_chat_id>:<numeric_thread_id>``) — that shape is
    identical for both cases, so shape alone cannot decide (this was the #52060
    regression).  The real signal is the chat *type*: a genuine channel DM topic
    lives on a ``channel`` chat.  Probe the live adapter's ``get_chat_info`` once
    and only return True when the chat is a channel.

    Fails SAFE to ``message_thread_id`` (returns False) for adapters without a
    probe, or any probe error/timeout — that is the pre-#22773 behaviour and the
    correct default for the common forum-topic case.
    """
    # Resolve on the CLASS, not the instance (general pitfall #11): a MagicMock
    # instance auto-creates a truthy ``get_chat_info`` attribute, so an
    # instance-level probe would misclassify test doubles. Real adapters expose
    # the coroutine on the class regardless.
    get_chat_info = getattr(type(runtime_adapter), "get_chat_info", None)
    if not callable(get_chat_info):
        return False
    try:
        from agent.async_utils import safe_schedule_threadsafe

        future = safe_schedule_threadsafe(
            get_chat_info(runtime_adapter, str(chat_id)), loop,  # type: ignore[arg-type]
        )
        if future is None:
            return False
        # Lighter than a send (metadata-only Bot API call), so a shorter bound
        # than the 30s/60s send waits elsewhere in this file is intentional.
        info = future.result(timeout=10)
    except Exception:
        logger.debug(
            "Job '%s': get_chat_info probe failed for chat=%s — "
            "defaulting to message_thread_id routing",
            job_id, chat_id, exc_info=True,
        )
        return False
    is_channel = isinstance(info, dict) and str(info.get("type") or "").lower() == "channel"
    if is_channel:
        logger.info(
            "Job '%s': chat=%s is a channel — routing via direct_messages_topic_id",
            job_id, chat_id,
        )
    return is_channel


def _deliver_result(job: dict, content: str, adapters=None, loop=None) -> Optional[str]:
    """
    Deliver job output to the configured target(s) (origin chat, specific platform, etc.).

    When ``adapters`` and ``loop`` are provided (gateway is running), tries to
    use the live adapter first — this supports E2EE rooms (e.g. Matrix) where
    the standalone HTTP path cannot encrypt.  Falls back to standalone send if
    the adapter path fails or is unavailable.

    Returns None on success, or an error string on failure.
    """
    targets = _resolve_delivery_targets(job)
    if not targets:
        deliver_value = _normalize_deliver_value(job.get("deliver", "local"))
        if deliver_value == "local":
            return None  # local-only jobs don't deliver — not a failure
        # deliver=origin with no resolvable origin and no configured home
        # channels: treat as local rather than reporting an error.  CLI-created
        # jobs never capture a {platform, chat_id} origin, so failing here would
        # make every CLI `deliver=origin` (or auto-detect) job emit a spurious
        # "no delivery target resolved" error on every run (#43014).  The output
        # is still persisted in last_output for `cron list`/resume.
        if deliver_value == "origin":
            logger.info(
                "Job '%s': deliver=origin but no origin or home channels — "
                "skipping delivery (output saved in last_output)",
                job.get("name", job.get("id", "?")),
            )
            return None
        msg = f"no delivery target resolved for deliver={deliver_value}"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    # Optionally wrap the content with a header/footer so the user knows this
    # is a cron delivery.  Wrapping is on by default; set cron.wrap_response: false
    # in config.yaml for clean output.
    wrap_response = True
    user_cfg = None
    try:
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    except Exception:
        pass

    if wrap_response:
        task_name = job.get("name", job["id"])
        job_id = job.get("id", "")
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"(job_id: {job_id})\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"To stop or manage this job, send me a new message (e.g. \"stop reminder {task_name}\")."
        )
    else:
        delivery_content = content

    # Extract MEDIA: tags so attachments are forwarded as files, not raw text
    from gateway.platforms.base import BasePlatformAdapter

    # Bridge gateway media-policy config (strict / allow_dirs / trust_recent)
    # into the env vars the path validator reads. Gateway startup does this
    # at boot; a standalone process (manual `hermes cron run` from the CLI,
    # a cron tick without the gateway) historically did NOT — so manual runs
    # filtered attachment paths under a DIFFERENT policy than scheduled runs
    # and silently dropped files the gateway would deliver. Idempotent,
    # env-wins, never raises.
    from gateway.media_policy import apply_media_policy_env

    apply_media_policy_env(user_cfg)

    media_files, cleaned_delivery_content = BasePlatformAdapter.extract_media(delivery_content)
    requested_media = [(str(p), v) for p, v in media_files]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Attachments the policy filter dropped will never be sent on ANY lane —
    # record them up front so the run status says so (previously one
    # stderr WARNING was the only trace: text delivered, file vanished).
    _policy_dropped = len(requested_media) - len(media_files)
    policy_drop_errors = (
        [
            f"{_policy_dropped} media attachment(s) dropped by media path "
            "policy (missing file, denied prefix, or strict-mode miss); "
            "see gateway.strict / media_delivery_allow_dirs in config.yaml"
        ]
        if _policy_dropped > 0
        else []
    )

    # Resolve the delivery-mirror gate ONCE (default off). When on, each
    # successful delivery is also appended to the target chat's gateway session
    # transcript so a user reply in that chat sees the cron output in context.
    # Mirror the CLEAN, unwrapped output (not the cron header/footer).
    try:
        mirror_enabled = _cron_mirror_delivery_enabled(job, user_cfg)
    except Exception:
        mirror_enabled = False
    # Keep the cleaned delivery text available independently of the optional
    # transcript-mirror knob. Continuable surfaces (notably in_channel) must
    # seed their target session even when attach_to_session=false and
    # cron.mirror_delivery=false; gating this value on mirror_enabled makes
    # the seed receive an empty string and return False, which is exactly the
    # live failure reproduced three times on Alice (job ef7bd2869d15).
    _, mirror_text = BasePlatformAdapter.extract_media(content)
    mirror_text = (mirror_text or "").strip()

    try:
        config = load_gateway_config()
    except Exception as e:
        msg = f"failed to load gateway config: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    delivery_errors = []

    for target in targets:
        platform_name = target["platform"]
        chat_id = target["chat_id"]
        thread_id = target.get("thread_id")

        # bot-chat targets don't ride a gateway adapter: the output becomes a
        # real inbound turn in the target profile's canonical Bot Chat via the
        # chat CLI lane (the same one Bot Mode agent-to-agent sends use). The
        # bot runs a turn and can respond — handled before the Platform enum
        # below, which knows nothing about this pseudo-platform.
        if platform_name == BOT_CHAT_PLATFORM:
            bot_chat_error = _deliver_to_bot_chat(job, content, chat_id)
            if bot_chat_error:
                delivery_errors.append(bot_chat_error)
            continue

        # Diagnostic: log thread_id for topic-aware delivery debugging
        origin = _resolve_origin(job) or {}
        origin_thread = origin.get("thread_id")
        if origin_thread and not thread_id:
            logger.warning(
                "Job '%s': origin has thread_id=%s but delivery target lost it "
                "(deliver=%s, target=%s)",
                job["id"], origin_thread, job.get("deliver", "local"), target,
            )
        elif thread_id:
            logger.debug(
                "Job '%s': delivering to %s:%s thread_id=%s",
                job["id"], platform_name, chat_id, thread_id,
            )

        # Mirror scope: the origin conversation, the home-channel FALLBACK for
        # an origin-less deliver=origin job (a script-provisioned managed cron
        # standing in for the user's primary conversation — not a broadcast),
        # or an explicit target the job opted into via attach_to_session.
        # Broadcast/fan-out targets are never mirrored (_target_mirror_eligible).
        origin_target = _target_matches_origin(origin, platform_name, chat_id, thread_id)
        mirror_this_target = mirror_enabled and _target_mirror_eligible(
            job, target, global_mirror=mirror_enabled, origin_match=origin_target,
        )
        # Pass the origin's user_id so a per-user-isolated group chat resolves to
        # the exact member who scheduled the job — parity with send_message.
        # Resolved for ANY origin-matching target (not just mirror-enabled):
        # the in_channel seed below needs it too, and it must not depend on
        # the attach_to_session/mirror opt-in.
        origin_user_id = origin.get("user_id") if origin_target else None

        # DM shape of this target, needed by BOTH the in_channel flatten gate
        # below and the seed/_seed_cron_channel_session chat_type further down:
        # a 1:1 DM keys as ``dm`` (Slack DM channel ids start with "D"; or the
        # origin says so), everything else as ``group``.
        origin_chat_type = str(origin.get("chat_type") or "").lower()
        is_dm_target = origin_chat_type == "dm" or (
            not origin_chat_type and str(chat_id).startswith("D")
        )

        # Shared continuable-target gate for the in_channel surface. The
        # thread-flatten and the flat-session seed MUST use the SAME gate —
        # if they drift, the brief and its continuation session land in
        # different places (the split-surface bug the flatten exists to
        # prevent). Origin targets qualify unconditionally (independent of the
        # attach_to_session / mirror opt-in — see 3c52d3589f); non-origin
        # mirror-eligible targets (origin_fallback / opted-in explicit)
        # qualify only when the seed can actually create a resolvable session
        # (_inchannel_seed_allowed: DM-shaped, or a known user_id for
        # user-isolated group keys).
        inchannel_continuable = origin_target or (
            mirror_this_target
            and _inchannel_seed_allowed(is_dm=is_dm_target, user_id=origin_user_id)
        )

        # Built-in names resolve to their enum member; plugin platform names
        # create dynamic members via Platform._missing_().
        try:
            platform = Platform(platform_name.lower())
        except (ValueError, KeyError):
            msg = f"unknown platform '{platform_name}'"
            logger.warning("Job '%s': %s", job["id"], msg)
            delivery_errors.append(msg)
            continue

        from gateway.delivery import resolve_delivery_transport

        transport = resolve_delivery_transport(platform, config, adapters)
        if transport is not None:
            pconfig = transport.config
            runtime_adapter = transport.adapter
        else:
            # No live transport: preserve the existing standalone delivery path,
            # which uses the logical platform's configured credential.
            pconfig = config.platforms.get(platform)
            runtime_adapter = None

        if transport is not None and transport.is_relay:
            # A relay transport carries the RELAY adapter's config, and
            # resolve_delivery_transport already applied relay's enablement
            # rule (config block absent OR enabled). The logical platform is
            # deliberately NOT natively enabled in a relay-fronted deployment
            # (its credential lives in the connector), so the native
            # configured/enabled gate below must not apply — it used to
            # reject exactly the targets the relay was resolved to serve.
            if pconfig is None:
                from gateway.config import PlatformConfig
                pconfig = PlatformConfig(enabled=True)
        elif not pconfig or not pconfig.enabled:
            msg = f"platform '{platform_name}' not configured/enabled"
            logger.warning("Job '%s': %s", job["id"], msg)
            delivery_errors.append(msg)
            continue

        # Prefer the resolved live transport when the gateway is running. This
        # supports E2EE native adapters and relay-fronted logical platforms.
        # The live-send path (which SEEDS the flat in_channel continuation
        # session via _seed_cron_channel_session) needs not just a live adapter
        # but a running event loop to schedule the async send onto. Compute that
        # gate ONCE so the in_channel thread_id clear below stays in lockstep
        # with the live-send/seed block further down (they used to drift): an
        # adapter can be present while the loop is absent/not-running, in which
        # case the live-send block is skipped and delivery falls through to the
        # standalone path — which cannot seed the flat session (r3609147550).
        live_adapter_ready = (
            runtime_adapter is not None
            and loop is not None
            and getattr(loop, "is_running", lambda: False)()
        )
        delivered = False
        target_errors = []

        # Continuable cron surface (D1/D2/D6): resolve the delivery surface for
        # this platform generically from its config ``extra``. Default "thread"
        # (today's behaviour, byte-identical). "in_channel" delivers the brief
        # FLAT into the channel (no dedicated thread) so a plain channel reply
        # continues the job in-context via the shared-channel session
        # ``(platform, chat_id, None)`` — the same bucket ``reply_in_thread:
        # false`` routes inbound channel messages to. The key is read
        # generically here (any platform); the ``in_channel`` branch is gated on
        # the adapter capability flag ``supports_inchannel_continuable`` so an
        # unsupported platform fails SAFE to "thread" (Slack is the first
        # consumer; "first consumer ≠ definition").
        surface_mode = _resolve_cron_surface_mode(pconfig, platform_name)
        in_channel_surface = surface_mode == "in_channel"
        if in_channel_surface and runtime_adapter is not None:
            # Per-platform capability first: one RelayAdapter fronts N
            # platforms and the connector advertises the bit per platform at
            # handshake — the scalar attr only carries the PRIMARY identity's
            # bit. Native adapters (no per-platform query) keep the class
            # attribute path unchanged.
            per_platform_check = getattr(
                runtime_adapter, "supports_inchannel_continuable_for_platform",
                None,
            )
            if callable(per_platform_check):
                try:
                    surface_supported = bool(per_platform_check(platform_name))
                except Exception:
                    surface_supported = False
            else:
                surface_supported = bool(getattr(
                    runtime_adapter, "supports_inchannel_continuable", False
                ))
            if not surface_supported:
                # Fail safe (D6): platform has no in_channel continuation
                # primitive.
                logger.debug(
                    "Job '%s': cron_continuable_surface=in_channel not supported on "
                    "%s, using thread",
                    job.get("id", "?"), platform_name,
                )
                in_channel_surface = False

        if in_channel_surface and inchannel_continuable and live_adapter_ready:
            # Force flat delivery (D2): the continuable-channel target must
            # ignore any inherited origin/target thread_id, or the flat
            # continuable session seeded below (thread_id=None, via
            # _seed_cron_channel_session) never matches where the brief is
            # actually delivered — route_thread_id further down in this loop
            # reads `thread_id` and would otherwise route into the origin
            # thread instead of flat into the channel.
            #
            # Gated on `inchannel_continuable` (the SAME gate as the seed
            # below), NOT `mirror_this_target` alone: for origin targets the
            # seed fires on origin-match alone (in_channel is the
            # continuation surface, independent of the attach_to_session /
            # mirror opt-in), so the flatten must use the SAME gate — with
            # the default knobs off, a mirror-gated flatten kept delivering
            # into the origin thread while the flat session got seeded,
            # leaving the brief and its continuation surface in different
            # places.
            # Gated on `live_adapter_ready` (adapter present AND a running loop)
            # so the clear fires ONLY on the live-send path that actually seeds
            # the flat session — the SAME condition as the live-send block
            # below. `runtime_adapter is not None` alone is broader than that
            # path: an adapter can be present while the event loop is absent or
            # not running, in which case the live-send/seed block is skipped and
            # delivery falls through to the standalone path. Clearing thread_id
            # there would flatten a brief into a channel with NO seeded
            # continuable session behind it (and bypass the D6 capability
            # check), so the standalone fallback must keep the origin thread
            # (review r3609147550).
            #
            # Fan-out / broadcast / explicit-thread targets keep their thread_id
            # (they are not continuable and are never seeded). Placed AFTER
            # mirror_this_target / origin_user_id are computed above — those
            # need the ORIGINAL thread_id to match the origin conversation.
            thread_id = None

        # For an in_channel delivery the flat continuation session is created
        # explicitly below (the shipped mirror only APPENDS to an existing
        # session, and the flat channel row is otherwise absent for a
        # chat_postMessage delivery). ``is_dm_target`` (computed above with
        # origin_user_id) selects the session chat_type so the seeded key
        # matches the inbound reply's key. ``inchannel_seeded`` suppresses the
        # generic mirror below so the brief is not double-written.
        inchannel_seeded = False

        # Continuable cron (thread-preferred): when mirroring is enabled for the
        # origin target and the gateway is live, try to open a DEDICATED thread
        # for this job and deliver the brief into it. On thread-capable
        # platforms (Telegram/Discord/Slack) the brief + the user's replies live
        # in their own scrollback; the thread-keyed session is seeded so a reply
        # continues with full context. On DM-only platforms (WhatsApp/Signal)
        # create_handoff_thread returns None and we fall back to mirroring into
        # the origin DM session (handled after delivery). Cf. _process_handoff.
        #
        # in_channel surface (D2): SKIP thread creation entirely — leave
        # thread_id=None so the delivery posts flat, then
        # ``_seed_cron_channel_session`` (below) CREATES the shared-channel
        # session and mirrors the brief into it. The shipped mirror alone is
        # NOT enough here: ``mirror_to_session`` only APPENDS to an existing
        # session and a flat ``(platform, chat_id, None)`` row is otherwise
        # absent for a ``chat_postMessage`` delivery, so the seed must create
        # the row first (F5).
        thread_seeded = False
        opened_thread_id: Optional[str] = None
        if (
            mirror_this_target
            and not in_channel_surface
            and runtime_adapter is not None
            and loop is not None
            and not thread_id  # never override an explicit origin thread/topic
        ):
            new_thread_id = _open_continuable_cron_thread(
                job, runtime_adapter, chat_id, loop,
            )
            if new_thread_id:
                # Route THIS delivery into the new thread now (the send needs the
                # thread_id), but defer seeding the thread session until the
                # delivery actually succeeds — otherwise an open-succeeds /
                # deliver-fails case leaves a seeded brief the user never saw,
                # and (worse) suppresses the DM-fallback mirror via thread_seeded.
                thread_id = new_thread_id
                opened_thread_id = new_thread_id

        if live_adapter_ready:
            # Telegram topic routing (#22773, regression fixed #52060): a
            # ``telegram:<positive_chat_id>:<numeric_thread_id>`` cron target is
            # ambiguous — a forum-style topic in a private chat and a genuine
            # Bot API channel Direct-Messages topic share the same shape and
            # need OPPOSITE routing. Disambiguate at delivery time via
            # ``_is_channel_dm_topic`` (see its docstring for the full
            # rationale); ``thread_id`` goes in ``route_metadata`` so the
            # anchorless cron send bypasses the DeliveryRouter's private-chat
            # reply-anchor requirement. Compute the routed metadata ONCE so both
            # the text send (via DeliveryRouter) and the media send agree.
            from gateway.delivery import (
                DeliveryRouter,
                DeliveryTarget,
                _looks_like_int,
                looks_like_telegram_private_chat_id,
            )

            is_ambiguous_telegram_topic = (
                platform == Platform.TELEGRAM
                and thread_id is not None
                and looks_like_telegram_private_chat_id(str(chat_id))
                and _looks_like_int(str(thread_id))
            )
            route_via_dm_topic = is_ambiguous_telegram_topic and _is_channel_dm_topic(
                runtime_adapter, chat_id, loop, job["id"],
            )
            if route_via_dm_topic:
                # Genuine Bot API channel Direct-Messages topic (#22773 mode 2):
                # routed via direct_messages_topic_id, no bare thread_id.
                route_thread_id = None
                route_metadata = {
                    "direct_messages_topic_id": str(thread_id),
                    "job_id": job["id"],
                }
                # Media metadata mirrors the text routing so attachments land in
                # the same DM topic instead of the General lane (#22773).
                media_metadata = {"direct_messages_topic_id": str(thread_id)}
            else:
                # Forum-style topic (private chat / supergroup) or non-topic
                # target: route via message_thread_id (#52060).  Put thread_id in
                # *route_metadata* (not just the DeliveryTarget) deliberately —
                # the DeliveryRouter's private-chat topic detection
                # (gateway/delivery.py) demands a reply anchor when thread_id is
                # absent from metadata; cron deliveries have no inbound reply
                # anchor, so the metadata key bypasses that check and lets the
                # adapter route via a plain message_thread_id.
                route_thread_id = str(thread_id) if thread_id is not None else None
                route_metadata = {"job_id": job["id"]}
                if route_thread_id:
                    route_metadata["thread_id"] = route_thread_id
                media_metadata = {"thread_id": thread_id} if thread_id else None

            # Relay egress needs a tenant discriminator on the frame: the
            # connector's fail-closed guard resolves the workspace/guild from
            # metadata.scope_id, and after a gateway restart the RelayAdapter's
            # per-chat scope cache is COLD (learned only from inbound), while
            # DeliveryRouter stamps scope only for the configured HOME channel
            # (gateway/delivery.py). A scoped origin that is not the home chat
            # therefore egressed with no scope_id at all and could be rejected
            # before delivery — the delivery-leg sibling of the seed-key scope
            # fix. Origin-matching targets only: a fan-out/broadcast target's
            # tenant is NOT the origin's, and stamping the wrong scope is worse
            # than none (the router/home path handles fan-out home targets).
            if origin_target and origin.get("scope_id"):
                route_metadata.setdefault("scope_id", str(origin["scope_id"]))
                media_metadata = dict(media_metadata or {})
                media_metadata.setdefault("scope_id", str(origin["scope_id"]))

            try:
                # Send cleaned text (MEDIA tags stripped) — not the raw content.
                # Route through the gateway's DeliveryRouter so the live send
                # gets the same platform-specific routing as live messages —
                # in particular Telegram's three-mode topic routing.  The
                # standalone cron path lacked this, so DM-topic cron deliveries
                # landed in the General topic or were rejected by Bot API 10.0
                # (#22773).
                text_to_send = cleaned_delivery_content.strip()
                adapter_ok = True
                timed_out = False
                delivered_message_id = None
                if text_to_send:
                    from agent.async_utils import safe_schedule_threadsafe

                    router = DeliveryRouter(config, adapters)
                    route_target = DeliveryTarget(
                        platform=platform,
                        chat_id=str(chat_id),
                        thread_id=route_thread_id,
                        is_explicit=True,
                    )
                    # Pass thread routing via the target (not a bare metadata
                    # "thread_id"): the router only applies its Telegram DM-topic
                    # detection when "thread_id"/"message_thread_id" are absent
                    # from metadata, deriving the routing from target.thread_id
                    # or the explicit direct_messages_topic_id above.
                    future = safe_schedule_threadsafe(
                        router._deliver_to_platform(
                            route_target,
                            text_to_send,
                            route_metadata,
                        ),
                        loop,
                    )
                    if future is None:
                        adapter_ok = False
                        target_errors.append("live adapter event loop scheduling failed")
                    else:
                        send_result = None
                        timeout_handled = False
                        try:
                            send_result = future.result(timeout=60)
                        except TimeoutError:
                            # #38922: a slow confirmation does NOT necessarily
                            # mean the send failed — but we must distinguish two
                            # cases via future.cancel()'s return value:
                            #
                            #   cancel() == False -> the coroutine was already
                            #     running on the gateway loop when the timeout
                            #     fired; the request is in flight on the wire and
                            #     cannot be un-sent.  Re-sending via standalone
                            #     would be a guaranteed DUPLICATE, so treat it as
                            #     delivered (assume-delivered).
                            #
                            #   cancel() == True -> the scheduled callback never
                            #     started executing (loop wedged/backlogged for
                            #     the full 60s), so nothing was sent.  We MUST
                            #     fall through to the standalone path or the
                            #     message is silently dropped (worse than a
                            #     duplicate).
                            cancelled = future.cancel()
                            if cancelled:
                                msg = (
                                    f"live adapter send to {platform_name}:{chat_id} "
                                    "timed out before the coroutine was dispatched"
                                )
                                logger.warning(
                                    "Job '%s': %s, falling back to standalone",
                                    job["id"], msg,
                                )
                                target_errors.append(msg)
                                adapter_ok = False  # fall through to standalone path
                                timeout_handled = True
                            else:
                                timed_out = True
                                timeout_handled = True
                                logger.warning(
                                    "Job '%s': live adapter send to %s:%s timed out "
                                    "after 60s; already dispatched (in flight), "
                                    "assuming delivered (skipping standalone fallback "
                                    "to avoid duplicate)",
                                    job["id"], platform_name, chat_id,
                                )
                        except Exception as ex:
                            # A real send error (not a slow confirmation) — fall
                            # through to the standalone path so the message is
                            # still delivered.
                            target_errors.append(f"live adapter send failed: {ex}")
                            raise

                        if timeout_handled:
                            # The timeout branch above already decided the
                            # outcome (assume-delivered if in flight, or
                            # adapter_ok=False to fall through if never
                            # dispatched).  send_result is None, so skip the
                            # confirmation/thread-fallback inspection below.
                            pass
                        else:
                            # _deliver_to_platform returns either a SendResult
                            # (.success attr) or, when the silence-narration
                            # filter drops the message, a plain dict
                            # {"success": True, "delivered": False, ...}.
                            # Normalize both shapes so a getattr default doesn't
                            # misread a dict, and so a None / success-less object
                            # is NOT counted as delivered (#47056).
                            if isinstance(send_result, dict):
                                send_success = bool(send_result.get("success", False))
                                send_raw_response = send_result.get("raw_response")
                                delivered_message_id = send_result.get("message_id")
                            else:
                                send_success = _confirm_adapter_delivery(send_result)
                                send_raw_response = getattr(send_result, "raw_response", None)
                                delivered_message_id = getattr(send_result, "message_id", None)

                            if not send_success:
                                if isinstance(send_result, dict):
                                    err = send_result.get("error", "unknown")
                                    shape = "dict"
                                elif send_result is not None:
                                    err = getattr(send_result, "error", None)
                                    shape = type(send_result).__name__
                                else:
                                    err = "no response from adapter"
                                    shape = "None"
                                msg = (
                                    f"live adapter send to {platform_name}:{chat_id} "
                                    f"returned unconfirmed result ({shape}, error={err})"
                                )
                                if transport is not None and transport.is_relay:
                                    logger.warning("Job '%s': %s", job["id"], msg)
                                else:
                                    logger.warning(
                                        "Job '%s': %s, falling back to standalone",
                                        job["id"], msg,
                                    )
                                target_errors.append(msg)
                                adapter_ok = False  # fall through to standalone path
                            elif (
                                send_raw_response
                                and thread_id
                                and send_raw_response.get("thread_fallback")
                            ):
                                requested_thread_id = send_raw_response.get("requested_thread_id") or thread_id
                                msg = (
                                    f"configured thread_id {requested_thread_id} for "
                                    f"{platform_name}:{chat_id} was not found; delivered without thread_id"
                                )
                                logger.warning("Job '%s': %s", job["id"], msg)
                                delivery_errors.append(msg)

                # Send extracted media files as native attachments via the live
                # adapter, using the same DM-topic-aware routing as the text send
                # (#22773 — media previously used a bare thread_id and landed in
                # the General lane for private DM topics).  Skip on an in-flight
                # confirmation timeout: the gateway loop is contended, so each
                # media send would also block its 30s budget, and the text
                # payload is already assumed delivered (#38922).  Record the
                # skipped attachments so the drop is visible rather than silently
                # lost.
                if adapter_ok and not timed_out and media_files:
                    routed_media_metadata = dict(media_metadata or {})
                    if transport is not None and transport.is_relay:
                        routed_media_metadata["_relay_logical_platform"] = platform.value
                        logical_home = config.get_home_channel(platform)
                        if logical_home is not None and logical_home.chat_id == chat_id:
                            if logical_home.user_id:
                                routed_media_metadata["user_id"] = logical_home.user_id
                            if logical_home.scope_id:
                                routed_media_metadata["scope_id"] = logical_home.scope_id
                    _media_errors = _send_media_via_adapter(
                        runtime_adapter,
                        chat_id,
                        media_files,
                        routed_media_metadata or None,
                        loop,
                        job,
                        platform=platform,
                    )
                    # Surface per-file failures into the run status (parity
                    # with the standalone lane): text delivered but an
                    # attachment didn't is a visible partial failure, not ok.
                    for _me in _media_errors:
                        _msg = f"{_me} (target {platform_name}:{chat_id})"
                        delivery_errors.append(_msg)
                elif timed_out and media_files:
                    msg = (
                        f"{len(media_files)} media attachment(s) not delivered to "
                        f"{platform_name}:{chat_id} (live adapter confirmation timed out)"
                    )
                    logger.warning("Job '%s': %s", job["id"], msg)
                    delivery_errors.append(msg)

                if adapter_ok:
                    logger.info("Job '%s': delivered to %s:%s via live adapter", job["id"], platform_name, chat_id)
                    delivered = True
                    # Seed the thread session only now that delivery into it
                    # succeeded (deferred from thread-open above).
                    if opened_thread_id and not thread_seeded:
                        _seed_cron_thread_session(
                            job, runtime_adapter, platform_name, chat_id,
                            opened_thread_id, mirror_text,
                            chat_name=origin.get("chat_name"),
                            is_dm=is_dm_target,
                            scope_id=origin.get("scope_id"),
                        )
                        thread_seeded = True
                    # in_channel surface: CREATE + seed the flat channel/DM
                    # session (the shipped mirror only appends to an existing
                    # session — the flat row is otherwise absent for a
                    # chat_postMessage delivery, so the brief would be lost).
                    # Gated on `inchannel_continuable` — the SHARED gate with
                    # the thread-flatten above (they must not drift, or the
                    # brief and its continuation session land in different
                    # places). Origin targets seed without requiring the
                    # mirror opt-in: in_channel IS the continuation surface —
                    # a continuable flat cron without its seed is a brief the
                    # next reply can't see (the bug Victor hit live
                    # 2026-08-19: agent had "no idea about the delivery
                    # message"). Mirror-eligible NON-origin targets
                    # (origin_fallback / opted-in explicit — see
                    # _target_mirror_eligible) also seed, guarded by
                    # _inchannel_seed_allowed inside the gate: group-channel
                    # keys are user-isolated, so a seed without a user_id
                    # (origin-less managed cron into a shared channel) would
                    # create an orphan session no reply resolves to — those
                    # fall back to the plain mirror instead.
                    if in_channel_surface and inchannel_continuable and not thread_seeded:
                        inchannel_seeded = _seed_cron_channel_session(
                            job, runtime_adapter, platform_name, chat_id,
                            mirror_text, is_dm=is_dm_target,
                            user_id=origin_user_id,
                            chat_name=origin.get("chat_name"),
                            scope_id=origin.get("scope_id"),
                        )
                        if not inchannel_seeded:
                            logger.warning(
                                "Job '%s': in_channel seed did NOT land on %s:%s "
                                "— a plain reply will not see this brief",
                                job["id"], platform_name, chat_id,
                            )
                        # Companion THREAD-surface seed (live gap, Alice
                        # 2026-08-19): a flat brief is still a Slack message
                        # the user can reply to IN ITS THREAD — the natural
                        # mobile/desktop affordance — and that reply keys to
                        # (chat, thread=<brief ts>), a session the flat seed
                        # never touches. Seed it too so BOTH reply surfaces
                        # continue the job. Uses the delivered message id as
                        # the thread anchor; best-effort like every seed.
                        if delivered_message_id:
                            _seed_cron_thread_session(
                                job, runtime_adapter, platform_name, chat_id,
                                str(delivered_message_id), mirror_text,
                                chat_name=origin.get("chat_name"),
                                is_dm=is_dm_target,
                                scope_id=origin.get("scope_id"),
                            )
                    elif in_channel_surface and not inchannel_continuable:
                        logger.warning(
                            "Job '%s': in_channel delivery to %s:%s is not a "
                            "continuable target (origin=%s:%s thread=%s; not the "
                            "origin conversation, and not a mirror-eligible "
                            "fallback/opted-in target the seed can key) — seed "
                            "skipped; the plain mirror below may still apply",
                            job["id"], platform_name, chat_id,
                            origin.get("platform"), origin.get("chat_id"),
                            origin.get("thread_id"),
                        )
                    _maybe_mirror_cron_delivery(
                        job, platform_name, chat_id, mirror_text,
                        thread_id=thread_id, user_id=origin_user_id,
                        enabled=mirror_this_target and not thread_seeded and not inchannel_seeded,
                    )
            except Exception as e:
                err_msg = f"live adapter delivery to {platform_name}:{chat_id} failed: {e}"
                if not any(err_msg in err for err in target_errors):
                    target_errors.append(err_msg)
                if transport is not None and transport.is_relay:
                    logger.warning("Job '%s': %s", job["id"], err_msg)
                else:
                    logger.warning(
                        "Job '%s': %s, falling back to standalone",
                        job["id"], err_msg,
                    )

        if not delivered:
            if transport is not None and transport.is_relay:
                # Relay owns the logical destination and its connector owns the
                # platform credential. A native retry could duplicate delivery
                # and cannot be authenticated correctly, so fail closed.
                if not target_errors:
                    target_errors.append(
                        f"relay delivery to {platform_name}:{chat_id} failed"
                    )
                delivery_errors.extend(target_errors)
                continue
            # If the interpreter is finalizing (gateway SIGTERM / restart /
            # OOM), scheduling any new delivery is futile — asyncio.run and a
            # fresh ThreadPoolExecutor both raise "cannot schedule new futures
            # after interpreter shutdown". Skip gracefully with a warning
            # rather than emitting an ERROR traceback on every restart-race
            # (#58720, #55924).
            if _interpreter_shutting_down():
                msg = f"delivery to {platform_name}:{chat_id} skipped — interpreter is shutting down"
                logger.warning("Job '%s': %s", job["id"], msg)
                target_errors.append(msg)
                delivery_errors.extend(target_errors)
                continue
            # Standalone path: run the async send in a fresh event loop (safe from any thread)
            coro = _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files)
            try:
                result = asyncio.run(coro)
            except RuntimeError as run_err:
                # asyncio.run() checks for a running loop before awaiting the coroutine;
                # when it raises, the original coro was never started — close it to
                # prevent "coroutine was never awaited" RuntimeWarning, then retry in a
                # fresh thread that has no running loop.
                coro.close()
                # If the RuntimeError is the interpreter-finalization signal,
                # the fresh-thread fallback would fail identically — skip
                # gracefully instead of logging a shutdown-race traceback.
                if _interpreter_shutting_down(run_err):
                    msg = f"delivery to {platform_name}:{chat_id} skipped — interpreter is shutting down"
                    logger.warning("Job '%s': %s", job["id"], msg)
                    target_errors.append(msg)
                    delivery_errors.extend(target_errors)
                    continue
                # The thread-pool fallback can itself raise (SMTP ConnectionError,
                # future.result timeout, etc.). An exception raised inside this
                # `except RuntimeError` block is NOT caught by the sibling
                # `except Exception` below — it would escape _deliver_result()
                # and crash the whole delivery loop, silently skipping every
                # remaining target (#47163). Wrap the fallback in its own
                # try/except so a per-target failure is logged and the loop
                # continues to the next target.
                try:
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content, thread_id=thread_id, media_files=media_files))
                        result = future.result(timeout=30)
                    finally:
                        pool.shutdown(wait=False)
                except Exception as e:
                    # A shutdown-race here is expected during teardown; downgrade
                    # to a warning so it doesn't read as a genuine failure.
                    if _interpreter_shutting_down(e):
                        msg = f"delivery to {platform_name}:{chat_id} skipped — interpreter is shutting down"
                        logger.warning("Job '%s': %s", job["id"], msg)
                        target_errors.append(msg)
                        delivery_errors.extend(target_errors)
                        continue
                    msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
                    logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
                    target_errors.extend([msg])
                    delivery_errors.extend(target_errors)
                    continue
            except Exception as e:
                msg = f"delivery to {platform_name}:{chat_id} failed: {e}"
                logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
                target_errors.extend([msg])
                delivery_errors.extend(target_errors)
                continue

            if result and result.get("error"):
                # Include target context (platform/chat) so a bare error string
                # like "Discord send failed: TimeoutError: " is attributable.
                # Not inside an except block — the error comes from the send
                # result dict, so there is no traceback to attach.
                msg = f"delivery error: {result['error']} (target {platform_name}:{chat_id})"
                logger.error("Job '%s': %s", job["id"], msg)
                target_errors.extend([msg])
                delivery_errors.extend(target_errors)
                continue

            # Standalone senders report per-file attachment failures in
            # ``warnings`` while still returning success (the text leg
            # delivered). Surface them: a cron whose PDF/image silently
            # vanished used to mark the run ok with no trace — the exact
            # "manual run delivers text but no attachment" field report.
            _sender_warnings = (
                result.get("warnings") if isinstance(result, dict) else None
            ) or []
            for _w in _sender_warnings:
                msg = f"delivery warning: {_w} (target {platform_name}:{chat_id})"
                logger.error("Job '%s': %s", job["id"], msg)
                delivery_errors.append(msg)

            logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
            _maybe_mirror_cron_delivery(
                job, platform_name, chat_id, mirror_text,
                thread_id=thread_id, user_id=origin_user_id,
                enabled=mirror_this_target and not thread_seeded,
            )

    if policy_drop_errors:
        # Filter-time drops apply to every target; report them once.
        delivery_errors.extend(policy_drop_errors)
    if delivery_errors:
        return "; ".join(delivery_errors)
    return None


_DEFAULT_SCRIPT_TIMEOUT = 3600  # seconds (1 hour)
# Backward-compatible module override used by tests and emergency monkeypatches.
_SCRIPT_TIMEOUT = _DEFAULT_SCRIPT_TIMEOUT
_RUN_CLAIM_HEARTBEAT_SECONDS = 60.0
_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS = _RUN_CLAIM_HEARTBEAT_SECONDS * 3


def _get_script_timeout() -> int:
    """Resolve cron pre-run script timeout from module/env/config with a safe default."""
    if _SCRIPT_TIMEOUT != _DEFAULT_SCRIPT_TIMEOUT:
        try:
            timeout = int(float(_SCRIPT_TIMEOUT))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default", _SCRIPT_TIMEOUT)

    env_value = os.getenv("HERMES_CRON_SCRIPT_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid HERMES_CRON_SCRIPT_TIMEOUT=%r; using config/default", env_value)

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("script_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron script timeout from config: %s", exc)

    return _DEFAULT_SCRIPT_TIMEOUT


_DEFAULT_MEDIA_SEND_TIMEOUT = 300


def _get_media_send_timeout() -> int:
    """Resolve the per-attachment media-send timeout from env/config.

    Mirrors the ``script_timeout_seconds`` resolution pattern: the
    HERMES_CRON_MEDIA_SEND_TIMEOUT env var wins, then
    ``cron.media_send_timeout_seconds`` in config.yaml, then the default
    (300s — large attachments like long TTS audio can legitimately exceed
    the old fixed 30s upload window).
    """
    env_value = os.getenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning(
                "Invalid HERMES_CRON_MEDIA_SEND_TIMEOUT=%r; using config/default",
                env_value,
            )

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("media_send_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron media-send timeout from config: %s", exc)

    return _DEFAULT_MEDIA_SEND_TIMEOUT


def _read_windows_pyvenv_cfg(venv_dir: Path) -> dict[str, str]:
    cfg_path = venv_dir / "pyvenv.cfg"
    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    parsed: dict[str, str] = {}
    for raw in lines:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _windows_cron_python_invocation(python_exe: str) -> tuple[str, dict[str, str]]:
    """Return an output-capable hidden Python invocation for Windows scripts.

    Cron scripts capture stdout/stderr, so using ``pythonw.exe`` directly can
    lose script output.  uv-created venv ``python.exe`` launchers are also a
    problem: even with CREATE_NO_WINDOW, the launcher can re-exec the base
    console interpreter and flash a visible window.  For uv venvs, bypass the
    launcher and run the base ``python.exe`` directly with the venv paths
    overlaid in the environment.
    """
    if sys.platform != "win32":
        return python_exe, {}

    interpreter = Path(python_exe)
    venv_dir = interpreter.parent.parent
    env_overlay: dict[str, str] = {}

    if interpreter.name.lower() == "pythonw.exe":
        sibling = interpreter.with_name("python.exe")
        if sibling.exists():
            interpreter = sibling

    cfg = _read_windows_pyvenv_cfg(venv_dir)
    home = cfg.get("home", "")
    site_packages = venv_dir / "Lib" / "site-packages"
    if "uv" in cfg and home:
        base_python = Path(home) / "python.exe"
        if base_python.exists() and site_packages.exists():
            interpreter = base_python
            env_overlay["VIRTUAL_ENV"] = str(venv_dir)
            pythonpath_entries = [
                str(Path(__file__).resolve().parents[1]),
                str(site_packages),
            ]
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)
            env_overlay["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return str(interpreter), env_overlay


def _terminate_cron_script_process(proc: subprocess.Popen) -> None:
    """Best-effort hard stop of a cron script and every child it spawned."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                creationflags=windows_hide_flags(),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    else:
        try:
            process_group: Optional[int] = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            process_group = None
        if process_group is not None:
            try:
                os.killpg(process_group, signal.SIGTERM)  # windows-footgun: ok — POSIX-only branch (win32 handled above)
            except (ProcessLookupError, PermissionError, OSError):
                process_group = None
            if process_group is not None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
                # Escalate whenever ANY group member survived the TERM: a
                # TERM-ignoring descendant keeps the stdio pipe write ends
                # open, and the caller's communicate() would then block on
                # EOF forever.  killpg(pgid, 0) probes group liveness.
                try:
                    os.killpg(process_group, 0)  # windows-footgun: ok — POSIX-only branch
                except (ProcessLookupError, OSError):
                    process_group = None
                if process_group is not None:
                    try:
                        os.killpg(process_group, getattr(signal, "SIGKILL", signal.SIGTERM))
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


def _drain_script_pipes(proc: subprocess.Popen) -> None:
    """Reap a terminated script process without ever blocking indefinitely.

    A descendant that survived the tree kill can hold the pipe write ends
    open, so a bare ``communicate()`` would wait for EOF forever.  Bound the
    drain, then abandon the pipes — the caller only needs the process reaped
    and the worker thread unblocked, not the output.
    """
    try:
        proc.communicate(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except OSError:
        pass
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        # Truly wedged — leave the zombie to the OS reaper rather than
        # blocking the cron worker thread forever.
        pass


def _windows_cron_bootstrap_argv(
    python_exe: str,
    env_overlay: dict[str, str],
    script_path: str,
) -> list[str]:
    """Bootstrap a cron script under the base interpreter with ``.pth`` support.

    The uv-venv overlay mode runs the base ``python.exe`` (to avoid the
    launcher re-execing a console interpreter and flashing a window) and
    re-attaches the venv via ``PYTHONPATH``.  But ``PYTHONPATH`` entries are
    plain ``sys.path`` additions — Python's site initialization never
    processes ``.pth`` files for them (only ``site.addsitedir()`` does) — so
    editable installs (``pip install -e``, ``__editable__*.pth`` links) are
    invisible to cron script jobs.

    Bootstrap with ``site.addsitedir()`` on the venv ``site-packages``, then
    exec the script as ``__main__``.  ``runpy.run_path`` keeps ``__file__``
    correct; ``sys.path[0]`` is set to the script's directory to preserve the
    ``python script.py`` import semantics.  Note: ``runpy`` does not set
    ``__package__``/``__spec__`` the way a direct invocation does, so
    package-relative imports (``from . import x``) may behave differently.
    Falls back to a plain invocation if the venv layout is unresolvable —
    the pre-existing PYTHONPATH behaviour is strictly better than failing
    to run at all.
    """
    site_packages = Path(env_overlay.get("VIRTUAL_ENV", "")) / "Lib" / "site-packages"
    if not site_packages.is_dir():
        # Silent here would make the "editable installs invisible" failure
        # undiagnosable; the pre-existing PYTHONPATH-only behaviour applies.
        logger.warning(
            "Windows cron script: venv site-packages %s not found; running "
            "without .pth processing (editable installs may be unimportable)",
            site_packages,
        )
        return [python_exe, script_path]
    bootstrap = (
        "import os, runpy, site, sys;"
        f"site.addsitedir({str(site_packages)!r});"
        "script = sys.argv[1];"
        "sys.argv = [script] + sys.argv[2:];"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(script)));"
        "runpy.run_path(script, run_name='__main__')"
    )
    return [python_exe, "-c", bootstrap, script_path]


def _run_job_script(
    script_path: str,
    workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
) -> tuple[bool, str]:
    """Execute a cron job's data-collection script and capture its output.

    Scripts must reside within HERMES_HOME/scripts/.  Both relative and
    absolute paths are resolved and validated against this directory to
    prevent arbitrary script execution via path traversal or absolute
    path injection.

    Supported interpreters (chosen by file extension):

    * ``.sh`` / ``.bash`` — run with ``/bin/bash``
    * anything else — run with the current Python interpreter
      (``sys.executable``), preserving the original behaviour for
      Python-based pre-check and data-collection scripts.

    Shell support lets ``no_agent=True`` jobs ship classic bash watchdogs
    (the `memory-watchdog.sh` pattern) without wrapping them in Python.

    Subprocess environment is passed through ``_sanitize_subprocess_env`` so
    provider credentials and other Hermes-managed secrets are not inherited
    (SECURITY.md §2.3), matching terminal and MCP child processes.

    Args:
        script_path: Path to the script.  Relative paths are resolved
            against HERMES_HOME/scripts/.  Absolute and ~-prefixed paths
            are also validated to ensure they stay within the scripts dir.
        workdir: Optional absolute path to use as the script's cwd.
            When set, the subprocess runs in this directory instead of
            the scripts-dir parent.  The Python process cwd is NEVER
            mutated, avoiding the global-side-effect bug where a cron
            job's ``os.chdir()`` leaks into concurrent gateway sessions
            (#69396).

    Returns:
        (success, output) — on failure *output* contains the error message so the
        LLM can report the problem to the user.
    """
    scripts_dir = _get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir_resolved = scripts_dir.resolve()

    # Same ingestion contract as cron.lifecycle_guard._expand_candidate_path:
    # a NUL-bearing value can never name a real script, and on Windows the
    # Path operations raise ValueError *after* expanduser (expanduser never
    # expands "~user" there, so the try below never fires) — reject eagerly
    # so both platforms fail cleanly instead of crashing the scheduler.
    # str() first so the guard itself can never raise TypeError on a
    # non-str script_path (e.g. a Path passed by a future caller) — the
    # guard must be crash-proof even though every current call site
    # passes a plain str (#86832 review).
    if "\x00" in str(script_path):
        return False, f"Blocked: script path contains a NUL byte: {script_path!r}"

    try:
        raw = Path(script_path).expanduser()
    except (ValueError, RuntimeError, OSError):
        # Same ingestion contract as cron.lifecycle_guard: a NUL-bearing
        # value (ValueError) or an unexpandable ``~`` (RuntimeError with no
        # resolvable HOME) can never name a real script. The creation-time
        # guard tolerates such values as "nothing to scan", so they can
        # reach fire time — fail the run with a report instead of crashing
        # the scheduler with an unhandled exception.
        return False, f"Blocked: script path is not a valid filesystem path: {script_path!r}"
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (scripts_dir / raw).resolve()

    # Guard against path traversal, absolute path injection, and symlink
    # escape — scripts MUST reside within HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    script_timeout = _get_script_timeout()

    # Pick an interpreter by extension.  Bash for .sh/.bash, Python for
    # everything else.  We deliberately do NOT honour the file's own
    # shebang: the scripts dir is trusted, but keeping the interpreter
    # choice explicit here keeps the allowed surface small and auditable.
    suffix = path.suffix.lower()
    if suffix in {".sh", ".bash"}:
        # Resolve bash dynamically so Windows (Git Bash) and Linux/macOS
        # all work.  On native Windows without Git for Windows installed
        # shutil.which returns None — fall back to a clear error rather
        # than a FileNotFoundError with a confusing "[WinError 2]"
        # traceback.
        _bash = shutil.which("bash") or (
            "/bin/bash" if os.path.isfile("/bin/bash") else None
        )
        if _bash is None:
            return False, (
                f"Cannot run .sh/.bash script {path.name!r}: bash not found on PATH. "
                "On Windows, install Git for Windows (which ships Git Bash) "
                "or rewrite the script as Python (.py)."
        )
        argv = [_bash, str(path)]
        env_overlay: dict[str, str] = {}
    else:
        python_exe, env_overlay = _windows_cron_python_invocation(sys.executable)
        if env_overlay:
            # Overlay mode (Windows uv venv): PYTHONPATH alone cannot make
            # editable installs importable — .pth processing needs
            # site.addsitedir() (see _windows_cron_bootstrap_argv).
            argv = _windows_cron_bootstrap_argv(python_exe, env_overlay, str(path))
        else:
            argv = [python_exe, str(path)]

    try:
        from tools.environments.local import build_subprocess_env

        popen_kwargs: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32":
            popen_kwargs = {
                "creationflags": windows_hide_flags()
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                "encoding": "utf-8",
                "errors": "replace",
            }
        env = build_subprocess_env()
        env.update(env_overlay)
        # Use the job's workdir as the subprocess cwd when configured,
        # otherwise default to the scripts-dir parent (back-compat).
        # NEVER mutate the Python process cwd — that would leak into
        # concurrent gateway sessions (#69396).
        _script_cwd = workdir or str(path.parent)
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=_script_cwd,
            env=env,
            **popen_kwargs,
        )
        deadline = time.monotonic() + script_timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                _terminate_cron_script_process(proc)
                _drain_script_pipes(proc)
                return False, "Script cancelled because cron fire ownership was lost"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_cron_script_process(proc)
                _drain_script_pipes(proc)
                return False, f"Script timed out after {script_timeout}s: {path}"
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        stdout = (stdout_raw or "").strip()
        stderr = (stderr_raw or "").strip()

        # Redact secrets from both stdout and stderr before any return path.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception as e:
            logger.warning("Failed to redact sensitive text from output: %s", e)
            stdout = "[REDACTED - redaction failed]"
            stderr = "[REDACTED - redaction failed]"

        if proc.returncode != 0:
            parts = [f"Script exited with code {proc.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout

    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _run_job_script_with_claim_heartbeat(
    job: dict,
    script_path: str,
    workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
) -> tuple[bool, str]:
    """Run a cron script while keeping its owned one-shot claim fresh.

    Script execution is synchronous and may legitimately outlive the stale
    claim TTL.  Without a concurrent heartbeat, another scheduler process can
    mistake the live run for a dead owner and dispatch the same one-shot again.
    Recurring jobs and unclaimed/manual runs have no durable one-shot claim and
    therefore use the ordinary script path without starting a thread.

    The claim owner is captured from the dispatched job and never re-read from
    storage.  ``heartbeat_run_claim`` compares that stable owner before every
    refresh, so a stale runner cannot extend a replacement owner's claim.
    """
    schedule = job.get("schedule")
    claim = job.get("run_claim")
    owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    if not (
        isinstance(schedule, dict)
        and schedule.get("kind") == "once"
        and owner
    ):
        return _run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)

    job_id = str(job.get("id") or "")
    stop = threading.Event()
    heartbeat_context = contextvars.copy_context()

    def _heartbeat_loop() -> None:
        while not stop.wait(_RUN_CLAIM_HEARTBEAT_SECONDS):
            try:
                heartbeat_run_claim(job_id, expected_owner=owner)
            except Exception:
                logger.debug(
                    "Job '%s': script run_claim heartbeat failed",
                    job_id,
                    exc_info=True,
                )

    heartbeat_thread = threading.Thread(
        target=heartbeat_context.run,
        args=(_heartbeat_loop,),
        name="cron-script-claim-heartbeat",
        daemon=True,
    )
    try:
        heartbeat_thread.start()
    except Exception:
        logger.debug(
            "Job '%s': could not start script run_claim heartbeat",
            job_id,
            exc_info=True,
        )
        return _run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)

    try:
        return _run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)
    finally:
        stop.set()
        # Event.wait() wakes immediately.  Keep completion bounded if the
        # heartbeat is already waiting on another process's jobs-file lock.
        heartbeat_thread.join(timeout=1.0)


def _parse_wake_gate(script_output: str) -> bool:
    """Parse the last non-empty stdout line of a cron job's pre-check script
    as a wake gate.

    The convention (ported from nanoclaw #1232): if the last stdout line is
    JSON like ``{"wakeAgent": false}``, the agent is skipped entirely — no
    LLM run, no delivery. Any other output (non-JSON, missing flag, gate
    absent, or ``wakeAgent: true``) means wake the agent normally.

    Returns True if the agent should wake, False to skip.
    """
    if not script_output:
        return True
    stripped_lines = [line for line in script_output.splitlines() if line.strip()]
    if not stripped_lines:
        return True
    last_line = stripped_lines[-1].strip()
    try:
        gate = json.loads(last_line)
    except (json.JSONDecodeError, ValueError):
        return True
    if not isinstance(gate, dict):
        return True
    return gate.get("wakeAgent", True) is not False


def _build_job_prompt(
    job: dict,
    prerun_script: Optional[tuple] = None,
    extra_prompt: Optional[str] = None,
) -> str:
    """Build the effective prompt for a cron job, optionally loading one or more skills first.

    Args:
        job: The cron job dict.
        prerun_script: Optional ``(success, stdout)`` from a script that has
            already been executed by the caller (e.g. for a wake-gate check).
            When provided, the script is not re-executed and the cached
            result is used for prompt injection. When omitted, the script
            (if any) runs inline as before.
        extra_prompt: Optional per-run context (from ``cronjob(action='run')``,
            #57331 — salvaged from #57342 by @liuhao1024). Appended to the
            stored prompt under a ``## Run Context`` header for this single
            fire only — never persisted to the job definition.
    """
    user_prompt = str(job.get("prompt") or "")
    if extra_prompt:
        user_prompt = f"{user_prompt}\n\n## Run Context\n{extra_prompt}"
    prompt = user_prompt
    skills = job.get("skills")
    # True when runtime-collected DATA (script stdout, upstream-job output)
    # has been injected into the prompt. Data content legitimately quotes
    # command-shape strings (a triage feed ingesting a bug report that
    # pastes `rm -rf /`), so it must not be scanned with the strict
    # user-prompt pattern set — see _scan_assembled_cron_prompt.
    has_injected_data = False

    # Run data-collection script if configured, inject output as context.
    script_path = job.get("script")
    if script_path:
        if prerun_script is not None:
            success, script_output = prerun_script
        else:
            success, script_output = _run_job_script(script_path)
        if success:
            if script_output:
                prompt = (
                    "## Script Output\n"
                    "The following data was collected by a pre-run script. "
                    "Use it as context for your analysis.\n\n"
                    f"```\n{script_output}\n```\n\n"
                    f"{prompt}"
                )
                has_injected_data = True
            else:
                # Script produced no output — nothing to report, skip AI call.
                return None
        else:
            prompt = (
                "## Script Error\n"
                "The data-collection script failed. Report this to the user.\n\n"
                f"```\n{script_output}\n```\n\n"
                f"{prompt}"
            )
            has_injected_data = True

    # Inject output from referenced cron jobs as context.
    context_from = job.get("context_from")
    if context_from:
        from cron.jobs import get_cron_output_dir
        output_dir = get_cron_output_dir()
        if isinstance(context_from, str):
            context_from = [context_from]
        for source_job_id in context_from:
            # "self" resolves to the job's own id: the job wakes up with its
            # most recent output injected, giving recurring jobs continuity
            # across runs (dedupe against what was already reported, continue
            # where the last run left off) without touching session history.
            is_self = False
            if isinstance(source_job_id, str) and source_job_id.strip().lower() == "self":
                source_job_id = str(job.get("id") or "")
                is_self = True
            elif source_job_id == job.get("id"):
                is_self = True
            # Guard against path traversal — valid job IDs are 12-char hex strings
            if not source_job_id or not all(c in "0123456789abcdef" for c in source_job_id):
                logger.warning(
                    "context_from: skipping invalid job_id %r for job_id=%r name=%r%s",
                    source_job_id,
                    job.get("id"),
                    job.get("name"),
                    _cron_job_origin_log_suffix(job),
                )
                continue
            try:
                job_output_dir = output_dir / source_job_id
                if not job_output_dir.exists():
                    continue  # silent skip — no output yet
                output_files = sorted(
                    job_output_dir.glob("*.md"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                if not output_files:
                    continue  # silent skip — no output yet
                latest_output = _extract_cron_context_payload(
                    output_files[0].read_text(encoding="utf-8")
                )
                # Truncate to 8K characters to avoid prompt bloat
                _MAX_CONTEXT_CHARS = 8000
                if len(latest_output) > _MAX_CONTEXT_CHARS:
                    latest_output = latest_output[:_MAX_CONTEXT_CHARS] + "\n\n[... output truncated ...]"
                if latest_output:
                    if is_self:
                        prompt = (
                            "## Your previous run's output\n"
                            "The following is this job's most recent output from its "
                            "previous run. Use it for continuity: avoid repeating what "
                            "was already reported, and continue where the last run "
                            "left off.\n\n"
                            f"```\n{latest_output}\n```\n\n"
                            f"{prompt}"
                        )
                    else:
                        prompt = (
                            f"## Output from job '{source_job_id}'\n"
                            "The following is the most recent output from a preceding "
                            "cron job. Use it as context for your analysis.\n\n"
                            f"```\n{latest_output}\n```\n\n"
                            f"{prompt}"
                        )
                    has_injected_data = True
                else:
                    continue  # silent skip — empty output
            except (OSError, PermissionError) as e:
                logger.warning("context_from: failed to read output for job %r: %s", source_job_id, e)
                # silent skip — do not pollute the prompt with error messages

    # Inject the job's durable notepad (per-job KV scratchpad surviving
    # scheduled wake-ups). Empty notepad renders as "" so jobs that never
    # use the feature get a byte-identical prompt.
    from cron import notepad as cron_notepad

    notepad_section = cron_notepad.render_notepad_section(str(job.get("id") or ""))
    if notepad_section:
        prompt = f"{notepad_section}{prompt}"
        has_injected_data = True

    # Always prepend cron execution guidance so the agent knows how
    # delivery works and can suppress delivery when appropriate.
    cron_hint = (
        "[IMPORTANT: You are running as a scheduled cron job. "
        "DELIVERY: Your final response will be automatically delivered "
        "to the user — do NOT use send_message or try to deliver "
        "the output yourself. Just produce your report/output as your "
        "final response and the system handles the rest. "
        "SILENT: If there is genuinely nothing new to report, respond "
        "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    prompt = cron_hint + prompt
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []
    elif isinstance(skills, str):
        skills = [skills]

    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return _scan_assembled_cron_prompt(
            prompt,
            job,
            has_skills=False,
            has_injected_data=has_injected_data,
            user_prompt=user_prompt,
        )

    from tools.skills_tool import skill_view
    from tools.skill_usage import bump_use
    from agent.skill_bundles import build_bundle_invocation_message, resolve_bundle_command_key
    from agent.skill_utils import normalize_skill_lookup_name

    parts = []
    skipped: list[str] = []
    for skill_name in skill_names:
        # Cron jobs historically accepted only skill names here, but the CLI/gateway
        # slash-command path lets bundles shadow skills with the same slug. Mirror
        # that behavior so `skills: ["my-bundle"]` expands bundle members instead
        # of being treated as a missing skill.
        bundle_key = resolve_bundle_command_key(skill_name.lstrip("/"))
        if bundle_key:
            bundle_payload = build_bundle_invocation_message(
                bundle_key,
                user_instruction="",
                task_id=str(job.get("id") or "") or None,
            )
            if bundle_payload:
                bundle_message, _loaded_bundle_skills, _missing_bundle_skills = bundle_payload
                if parts:
                    parts.append("")
                parts.append(bundle_message)
                continue
            logger.warning(
                "Cron job '%s': bundle '%s' could not load any skills, skipping",
                job.get("name", job.get("id")),
                skill_name,
            )
            skipped.append(skill_name)
            continue

        try:
            loaded = json.loads(skill_view(normalize_skill_lookup_name(skill_name)))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Cron job '%s': skill '%s' returned invalid JSON, skipping", job.get("name", job.get("id")), skill_name)
            skipped.append(skill_name)
            continue
        if not loaded.get("success"):
            error = loaded.get("error") or f"Failed to load skill '{skill_name}'"
            logger.warning("Cron job '%s': skill not found, skipping — %s", job.get("name", job.get("id")), error)
            skipped.append(skill_name)
            continue

        # Bump usage so the curator sees this skill as actively used.
        try:
            bump_use(skill_name, task_id=str(job.get("id") or "") or None)
        except Exception:
            logger.debug("Cron job: failed to bump skill usage for '%s'", skill_name, exc_info=True)

        content = str(loaded.get("content") or "").strip()
        if parts:
            parts.append("")
        parts.extend(
            [
                f'[IMPORTANT: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
                "",
                content,
            ]
        )

    if skipped:
        notice = (
            f"[IMPORTANT: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        )
        parts.insert(0, notice)

    stable_prefix = None
    if prompt:
        from agent.skill_commands import append_user_instruction

        parts.append("")
        # The skill blocks (and any skipped-skill notice) above are stable per
        # job config; the appended instruction carries the volatile per-run
        # data (cron hint + prompt + script output + run context). Declare
        # that boundary for the Anthropic cache planner (#81867).
        stable_prefix = append_user_instruction(parts, prompt)
    assembled = _scan_assembled_cron_prompt("\n".join(parts), job, has_skills=True)
    if stable_prefix and len(assembled) > len(stable_prefix) and assembled.startswith(stable_prefix):
        # Guarded because the injection scanner may sanitize (mutate) the
        # assembled bytes; a mismatch simply falls back to whole-message
        # caching.
        from agent.prompt_cache_boundary import register_stable_prefix

        register_stable_prefix(stable_prefix)
    return assembled


def _scan_assembled_cron_prompt(
    assembled: str,
    job: dict,
    *,
    has_skills: bool = False,
    has_injected_data: bool = False,
    user_prompt: Optional[str] = None,
) -> str:
    """Scan the fully-assembled cron prompt for injection patterns. Raises
    ``CronPromptInjectionBlocked`` when a match fires so ``run_job`` can
    surface a clear refusal to the operator.

    Plugs the #3968 gap: ``_scan_cron_prompt`` runs on the user-supplied
    prompt at create/update, but skill content is loaded from disk at
    runtime and was never scanned. Since cron runs non-interactively
    (auto-approves tool calls), a malicious skill carrying an injection
    payload bypassed every gate.

    Two pattern tiers, selected by what the assembled prompt CONTAINS,
    not just whether skills are attached:

    - When the assembled prompt is essentially the user prompt + the cron
      hint (no skills, no injected data), the STRICT ``_scan_cron_prompt``
      patterns apply: a bare ``rm -rf /`` in a small directive prompt is a
      smoking gun, not prose.
    - When the assembled prompt includes runtime-loaded content — skill
      markdown (``has_skills=True``) or DATA injected from a job script's
      stdout / an upstream job's output (``has_injected_data=True``) — the
      LOOSER ``_scan_cron_skill_assembled`` pattern set is used: only
      unambiguous prompt-injection directives block; command-shape
      patterns are dropped and invisible unicode is sanitized (stripped +
      logged) rather than blocked, to avoid false-positives that
      permanently kill a job. Skill bodies are vetted at install time by
      ``skills_guard.py``; script output is produced by operator-authored
      code, the same trust class — and data feeds (e.g. a triage bot
      ingesting bug reports) legitimately quote dangerous commands.

    When the looser tier is selected because of injected data only,
    ``user_prompt`` (the raw, pre-assembly prompt) is additionally scanned
    with the STRICT set so the user-authored surface keeps the full
    create/update-time guarantee at runtime (defense-in-depth for legacy
    jobs that predate the create-time scanner).
    """
    from tools.cronjob_tools import _scan_cron_prompt, _scan_cron_skill_assembled

    if has_skills or has_injected_data:
        # Runtime-loaded content (vetted skill markdown and/or data from
        # operator-authored scripts) legitimately contains command-shape
        # strings. Invisible unicode is sanitized (not blocked) so a stray
        # zero-width space can't permanently kill the job; the cleaned
        # prompt is what actually runs.
        cleaned, scan_error = _scan_cron_skill_assembled(assembled)
        assembled = cleaned
        if not scan_error and not has_skills and user_prompt:
            # Data-injection path: keep the strict guarantee on the
            # user-authored prompt itself.
            scan_error = _scan_cron_prompt(user_prompt)
    else:
        scan_error = _scan_cron_prompt(assembled)
    if scan_error:
        job_label = job.get("name") or job.get("id") or "<unknown>"
        logger.warning(
            "Cron job '%s': assembled prompt blocked by injection scanner — %s",
            job_label,
            scan_error,
        )
        raise CronPromptInjectionBlocked(scan_error)
    return assembled


def _guard_job_credential_exfil(job: dict) -> None:
    """Fail closed if a job's stored provider/base_url pair would exfiltrate a
    credential (F8 runtime backstop; CWE-200/CWE-522).

    The model-callable cron tool validates this on create/update, but a job
    persisted before that guard — or written directly to the jobs store —
    reaches the scheduler's provider-resolution sink unchecked. Re-validate the
    EFFECTIVE stored pair with the same guard the tool uses, so a named
    provider's stored key is never paired with an off-host base_url at fire
    time. Raises ``RuntimeError`` (caught by the run_job failure path → the run
    is aborted and reported) when the pair is unsafe; returns ``None`` otherwise.

    Fallback providers come from operator config, not the model-callable job, so
    they are trusted and validated by the caller, not here.
    """
    try:
        from tools.cronjob_tools import _validate_cron_base_url
        err = _validate_cron_base_url(job.get("provider"), job.get("base_url"))
    except Exception as exc:
        # Fail CLOSED: this is the last guard before provider resolution, so an
        # unexpected validator/import error must not silently allow an unvetted
        # pair through. A job that carries no base_url override cannot exfiltrate
        # a stored credential via this path (there is nothing to validate, and
        # the validator would return None), so it still runs — that keeps the
        # overwhelmingly-common no-override jobs from wedging on an unrelated
        # error. But any job that DID set a base_url is refused until the
        # validator can actually vet the pair. Operator fallback providers come
        # from config, not the job, so they are unaffected.
        if job.get("base_url"):
            err = (
                f"could not validate provider/base_url pair "
                f"({exc.__class__.__name__}: {exc}); refusing to run a job with "
                "an unverified base_url override"
            )
        else:
            err = None
    if err:
        job_id = job.get("id")
        logger.error(
            "Job '%s': refusing to run — unsafe provider/base_url pair could "
            "exfiltrate a stored credential: %s",
            job_id, err,
        )
        raise RuntimeError(f"Cron job '{job_id}' blocked for safety: {err}")


def _block_and_pause_job(
    job_id: str, job_name: str, reason: str
) -> tuple[bool, str, str, Optional[str]]:
    """Fail a run closed and pause the job so it stops being scheduled.

    Used for job shapes that can never run (a5e29e688dc0). Returning an error
    alone is not enough — an unrunnable job that stays enabled re-fires on
    every tick forever. Pausing writes ``paused_at``/``paused_reason``, giving
    an auditable record of why the scheduler stopped it.
    """
    from cron.jobs import pause_job

    logger.error("Job '%s': %s", job_id, reason)
    try:
        pause_job(job_id, f"Auto-paused by scheduler: {reason}")
    except Exception:
        logger.exception("Job '%s': failed to auto-pause unrunnable job", job_id)

    now_iso = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")
    doc = (
        f"# Cron Job: {job_name}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {now_iso}\n"
        f"**Status:** blocked (unrunnable job) — auto-paused\n\n"
        f"{reason}\n"
    )
    alert = f"⚠ Cron job '{job_name}' was auto-paused\n\n{reason}"
    return False, doc, alert, reason


# Marker prefix stamped into the error string returned by ``run_job`` when the
# pre-dispatch configuration validation (T1-26) refuses to run the agent.
# ``run_one_job`` keys off it to record ``last_status='blocked_config'`` and to
# apply the alert-once dedup. The ``:silent`` variant means "already alerted on
# a previous tick — do not deliver again".
BLOCKED_CONFIG_MARKER = "[blocked_config]"
BLOCKED_CONFIG_SILENT_MARKER = "[blocked_config:silent]"

# Marker prefix for a #44585 drift-guard skip. Same alert-once contract as
# blocked_config: run_one_job keys off it to record last_status and the
# ``:silent`` variant means "already alerted on a previous tick — do not
# deliver again" (the drift_alerted bit on the job record, #73506 shape).
DRIFT_SKIP_MARKER = "[drift_skip]"
DRIFT_SKIP_SILENT_MARKER = "[drift_skip:silent]"



def _is_transient_provider_resolve_error(exc: BaseException) -> bool:
    """True when primary provider resolution failed for a transient network reason.

    Agent crons resolve OAuth credentials (token refresh / discovery) before the
    agent loop starts. A short DNS outage (Cloudflare WARP / macOS resolver blip)
    surfaces as httpx/httpcore ConnectError or raw OSError errno 8 ("nodename nor
    servname provided") and must be eligible for ``fallback_providers`` the same
    way AuthError already is — otherwise a healthy XAI_API_KEY / Anthropic rung
    never gets tried and the whole job dies before the first model call.
    """
    # Walk the cause chain; scheduler wraps raw transport errors.
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__
        module = type(cur).__module__ or ""
        msg = str(cur).lower()
        # Explicit transport classes from httpx/httpcore/aiohttp.
        if name in {
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "NetworkError",
            "TimeoutException",
            "ClientConnectorError",
            "ClientConnectorDNSError",
            "ServerTimeoutError",
            "ClientOSError",
        }:
            return True
        if "httpx" in module or "httpcore" in module or "aiohttp" in module:
            if any(
                needle in msg
                for needle in (
                    "nodename nor servname",
                    "name or service not known",
                    "temporary failure in name resolution",
                    "failed to resolve",
                    "connection refused",
                    "network is unreachable",
                    "timed out",
                    "timeout",
                )
            ):
                return True
        if isinstance(cur, OSError):
            # Platform-safe classification (the raw-literal set {8, 7, 11, ...}
            # from the first revision mixed macOS getaddrinfo constants with
            # errno values and does not hold on Linux — see PR review).
            # socket.gaierror carries getaddrinfo codes (EAI_*), plain OSError
            # carries errno; compare each against its own constant namespace.
            import errno as _errno
            import socket as _socket

            if isinstance(cur, _socket.gaierror):
                _eai_transient = {
                    getattr(_socket, _n)
                    for _n in ("EAI_NONAME", "EAI_AGAIN", "EAI_FAIL", "EAI_NODATA")
                    if hasattr(_socket, _n)
                }
                if cur.errno in _eai_transient:
                    return True
            else:
                err_no = getattr(cur, "errno", None)
                if err_no in {
                    _errno.ECONNREFUSED,
                    _errno.ECONNRESET,
                    _errno.EHOSTUNREACH,
                    _errno.ENETUNREACH,
                    _errno.ENETDOWN,
                    _errno.ETIMEDOUT,
                    _errno.EAGAIN,
                }:
                    return True
            if any(
                needle in msg
                for needle in (
                    "nodename nor servname",
                    "name or service not known",
                    "temporary failure in name resolution",
                    "network is unreachable",
                )
            ):
                return True
        # Bare RuntimeError/Exception that already carries the DNS text
        # (format_runtime_provider_error sometimes surfaces the raw message).
        if "nodename nor servname" in msg or "name or service not known" in msg:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _cron_preflight_enabled(cfg: dict) -> bool:
    """Whether cron pre-dispatch configuration validation is enabled.

    Default ON; only the literal boolean ``false`` under ``cron.preflight``
    opts out (mirrors ``cron_model_drift_guard_enabled`` semantics).
    """
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get("preflight", True) is not False


def _preflight_check_provider_key(job: dict, cfg: dict) -> Optional[str]:
    """READ-ONLY probe: would provider resolution fail for lack of a key?

    Mirrors the effective requested-provider computation from run_job's
    resolution block without any side effects on the run. When a fallback
    chain is configured the check is skipped entirely — the existing
    auth-fallback path may legitimately rescue a missing primary key, so
    blocking here would break that contract (and burning zero LLM calls is
    already guaranteed by the fallback resolution being config-local).
    """
    try:
        if get_fallback_chain(cfg):
            return None
    except Exception:
        return None  # fail-open: never block on a preflight-internal error

    _cron_cfg = cfg.get("cron") if isinstance(cfg.get("cron"), dict) else {}
    requested = (
        job.get("provider")
        or str((_cron_cfg or {}).get("model_provider") or "").strip()
        or None
    )
    model = job.get("model") or os.getenv("HERMES_MODEL") or ""

    from hermes_cli.auth import AuthError

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        kwargs = {"requested": requested, "target_model": model}
        if job.get("base_url"):
            kwargs["explicit_base_url"] = job.get("base_url")
        resolve_runtime_provider(**kwargs)
    except AuthError as exc:
        return (
            f"provider credential missing: {exc}. "
            "Set the provider API key in .env (or `hermes setup`), or pin a "
            "working provider via `hermes cron edit "
            f"{job.get('id')} --provider <p>`."
        )
    except Exception:
        # Non-auth resolution errors (bad config shapes, network probes,
        # import issues) are NOT a missing-credential condition — let the
        # real resolution path handle and report them as before.
        return None
    return None


def _preflight_check_delivery(job: dict) -> Optional[str]:
    """Check the job's delivery target(s) resolve to configured platforms.

    ``local``/``origin`` (and the ``all`` routing token) need no gateway
    credentials and are never checked — a deliver=local job must not pay a
    gateway-config load. For concrete platform targets, an unknown platform
    always blocks; a known platform additionally blocks when the gateway
    config is loadable and reports it unconnected (enabled + credentials —
    the same source `cron_delivery_targets` uses). Gateway-config load
    failures fail OPEN so a transient config hiccup never wedges delivery
    that would have worked.
    """
    deliver_value = _normalize_deliver_value(job.get("deliver", "local"))
    platform_parts: list[str] = []
    for part in deliver_value.split(","):
        part = part.strip()
        if not part or part.lower() in {"local", "origin", "all"}:
            continue
        # bot-chat targets need no gateway credentials — they deliver via a
        # local chat subprocess. Unknown-profile failures surface per run in
        # last_delivery_error (and are validated at create time).
        if parse_bot_chat_deliver_token(part) is not None:
            continue
        platform_parts.append(part.split(":", 1)[0].strip())
    if not platform_parts:
        return None

    connected: Optional[set] = None
    for platform_name in platform_parts:
        if not _is_known_delivery_platform(platform_name):
            return (
                f"delivery platform '{platform_name}' is not a known cron "
                "delivery target. Fix the job's `deliver` value or configure "
                "the platform's gateway credentials."
            )
        if connected is None:
            try:
                from gateway.config import load_gateway_config

                gateway_config = load_gateway_config()
                connected = {
                    p.value for p in gateway_config.get_connected_platforms()
                }
                connected |= _relay_fronted_delivery_platforms(connected)
            except Exception:
                logger.debug(
                    "preflight: gateway config unavailable — skipping "
                    "delivery credential check", exc_info=True,
                )
                return None  # fail-open
        if platform_name.lower() not in connected:
            return (
                f"delivery platform '{platform_name}' has no gateway "
                "credentials configured (not connected). Configure it via "
                "`hermes setup` or change the job's `deliver` target."
            )
    return None


def _preflight_check_skills(job: dict) -> Optional[str]:
    """Check attached skills report ready (no missing required env/commands).

    Consults the same ``readiness_status`` payload ``skill_view`` computes
    for interactive use. Skills that fail to load at all are left to the
    existing skipped-skill handling in ``_build_job_prompt`` (fail-open):
    this check only blocks on an affirmative "setup needed" verdict, i.e.
    the skill exists but its required environment is missing — a run that
    is guaranteed to misfire.
    """
    skills = job.get("skills")
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []
    elif isinstance(skills, str):
        skills = [skills]
    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return None

    from tools.skills_tool import skill_view

    for skill_name in skill_names:
        try:
            payload = json.loads(skill_view(skill_name))
        except Exception:
            continue  # unreadable/missing skill → existing skip handling
        if not isinstance(payload, dict) or not payload.get("success"):
            continue
        if (
            payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"
        ):
            missing = [
                f"env ${name}"
                for name in payload.get(
                    "missing_required_environment_variables"
                ) or []
            ]
            missing += [
                f"command '{name}'"
                for name in payload.get("missing_required_commands") or []
            ]
            missing += [
                f"credential file {name}"
                for name in payload.get("missing_credential_files") or []
            ]
            detail = ", ".join(missing) or "required setup incomplete"
            return (
                f"attached skill '{skill_name}' is not ready: missing "
                f"{detail}. Provide the missing prerequisites or detach the "
                "skill from this job."
            )
    return None


def _preflight_job_config(job: dict, cfg: dict) -> Optional[str]:
    """Pre-dispatch configuration validation (T1-26).

    Returns a human-readable reason when the job's configuration cannot
    produce a successful run — missing provider API key, unconfigured
    delivery platform, or an attached skill with missing required env —
    so the caller can refuse the run BEFORE any agent machinery is
    constructed and no LLM call is burned. Returns ``None`` when the
    configuration validates (or when a check cannot be evaluated: every
    check fails open, so preflight can only ever block on an affirmative
    misconfiguration verdict).

    Same fail-before-spend spirit as the #44585 drift guard and the
    fail-loud-on-hidden-tools direction in #27948; alert dedup follows the
    alert-once pattern from the dead-pin auto-pause (#73506).
    """
    for name, check in (
        ("provider_key", lambda: _preflight_check_provider_key(job, cfg)),
        ("skills", lambda: _preflight_check_skills(job)),
        ("delivery", lambda: _preflight_check_delivery(job)),
    ):
        try:
            reason = check()
        except Exception:
            logger.debug(
                "preflight check %s raised — failing open", name, exc_info=True
            )
            continue
        if reason:
            return reason
    return None


def _cron_cleanup_timeout_seconds() -> float:
    """Return the wall-clock bound for cron post-run cleanup."""
    default = 10.0
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("cleanup_timeout_seconds")
        if configured is not None:
            timeout = float(configured)
            if timeout >= 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron cleanup timeout from config: %s", exc)
    return default


def _run_cron_cleanup_with_timeout(
    cleanup,
    *,
    job_id: str,
    label: str,
    timeout_seconds: Optional[float] = None,
def _run_cron_cleanup_with_timeout(
    cleanup, *, job_id: str, label: str, timeout_seconds: Optional[float] = None,
) -> bool:
    """Run fallible post-run cleanup without permanently wedging a cron ID."""
    timeout = (_cron_cleanup_timeout_seconds() if timeout_seconds is None else float(timeout_seconds))
    if timeout <= 0:
        try:
            cleanup()
            return True
        except (Exception, KeyboardInterrupt) as exc:
            logger.debug("Job '%s': %s failed: %s", job_id, label, exc)
            return False

    done = threading.Event()
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            cleanup()
        except BaseException as exc:
            error.append(exc)
        finally:
            done.set()

    # Daemon thread is deliberate: unlike ThreadPoolExecutor workers it is not joined at interpreter
    # exit if cleanup never returns, so the gateway can still shut down.
    worker = threading.Thread(
        target=_runner, name=f"cron-cleanup-{job_id}", daemon=True)
    worker.start()
    if not done.wait(timeout):
        logger.error(
            "Job '%s': %s exceeded %.1fs; abandoning cleanup so future runs remain dispatchable",
            job_id,
            label,
            timeout)
        return False
    if error:
        logger.debug("Job '%s': %s failed: %s", job_id, label, error[0])
        return False
    return True


class _BoundedCronSessionDB:
    """Proxy SessionDB cleanup calls through the cron cleanup timeout; after the first failure or
    timeout all later calls fail immediately (a damaged connection leaks at most one worker)."""

    def __init__(self, session_db, job_id: str):
        self._session_db = session_db
        self._job_id = job_id
        self._disabled = False

    def __getattr__(self, name):
        target = getattr(self._session_db, name)
        if not callable(target):
            return target

        def _bounded(*args, **kwargs):
            if self._disabled:
                raise RuntimeError("session finalization disabled after prior cleanup failure")

            result = {}

            def _call():
                try:
                    result["value"] = target(*args, **kwargs)
                except BaseException as exc:
                    result["error"] = exc
                    raise

            ok = _run_cron_cleanup_with_timeout(
                _call, job_id=self._job_id, label=f"session finalization ({name})")
            if not ok:
                error = result.get("error")
                if error is not None:
                    raise error
                # No error yet not complete == timeout: disable so later steps fail fast.
                self._disabled = True
                raise TimeoutError(f"session finalization method {name} timed out")
            return result.get("value")

        return _bounded


def _job_doc_header(job_name: str, job_id: str, now_iso: str, mode: str) -> str:
    """Common markdown header for the short-circuit run docs (no_agent / monitor)."""
    return (
        f"# Cron Job: {job_name}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {now_iso}\n"
        f"**Mode:** {mode}\n"
    )


def _resolve_job_workdir(job: dict, job_id: str) -> Optional[str]:
    """Configured job workdir, or None when unset / no longer a directory (logged)."""
    workdir = (job.get("workdir") or "").strip() or None
    if workdir and not Path(workdir).is_dir():
        logger.warning(
            "Job '%s': configured workdir %r no longer exists — running without it",
            job_id, workdir)
        return None
    return workdir


def _run_no_agent_job(
    job: dict, job_id: str, job_name: str, cancel_event,
) -> tuple[bool, str, str, Optional[str]]:
    """no_agent short-circuit — the script IS the job (no AIAgent, no tokens). stdout → delivered
    verbatim; empty stdout or wakeAgent=false → silent success; non-zero exit/timeout → error alert.
    """
    # Load .env first so auto-delivery can resolve *_HOME_CHANNEL: the agent path's per-run dotenv
    # reload never runs for no_agent jobs. Does not override existing values.
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(hermes_home=_get_hermes_home())
    except Exception:
        logger.debug("Job '%s': no_agent .env reload failed", job_id, exc_info=True)

    script_path = job.get("script")
    # Legacy/hand-edited no_agent job without a script: pause it, or it re-fires every tick.
    if not str(script_path or "").strip():
        from cron.jobs import NO_AGENT_WITHOUT_SCRIPT_ERROR

        return _block_and_pause_job(job_id, job_name, NO_AGENT_WITHOUT_SCRIPT_ERROR)

    # Pass workdir as subprocess cwd; never os.chdir() (leaks into concurrent gateway sessions).
    _job_workdir = _resolve_job_workdir(job, job_id)
    try:
        ok, output = _run_job_script_with_claim_heartbeat(
            job, script_path, workdir=_job_workdir, cancel_event=cancel_event)
    except Exception as exc:
        logger.exception("Job '%s': script execution raised unexpectedly", job_id)
        ok, output = False, f"Script execution failed: {exc}"

    now_iso = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")
    header = _job_doc_header(job_name, job_id, now_iso, "no_agent (script)")

    if not ok:
        # Deliver the error: a silently broken watchdog is the worst-case outcome.
        alert = (
            f"⚠ Cron watchdog '{job_name}' script failed\n\n"
            f"{output}\n\n"
            f"Time: {now_iso}"
        )
        return False, f"{header}**Status:** script failed\n\n{output}\n", alert, output

    # wakeAgent=false is a silent signal, same as empty stdout.
    if not _parse_wake_gate(output):
        logger.info("Job '%s' (no_agent): wakeAgent=false gate — silent run", job_id)
        return True, f"{header}**Status:** silent (wakeAgent=false)\n", SILENT_MARKER, None

    if not output.strip():
        logger.info("Job '%s' (no_agent): empty stdout — silent run", job_id)
        return True, f"{header}**Status:** silent (empty output)\n", SILENT_MARKER, None

    return True, f"{header}\n---\n\n{output}\n", output, None


def _apply_monitor_gate(
    job: dict, job_id: str, job_name: str, extra_prompt: Optional[str],
) -> tuple[Optional[tuple], Optional[str]]:
    """Monitor gate (hash-suppressed change detection). Must run BEFORE any agent machinery so an
    unchanged tick costs no LLM/delivery. Returns ``(early_result | None, extra_prompt)``; when
    early_result is None, extra_prompt may carry the injected monitor context.
    """
    from cron.monitor import check_monitor, job_has_monitor

    if not job_has_monitor(job):
        return None, extra_prompt
    _mon = check_monitor(job)
    _mon_now = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")
    header = _job_doc_header(job_name, job_id, _mon_now, "monitor")
    if not _mon.ok:
        # Source failure is an ERROR, never a change: alert so a broken monitor can't silently
        # stop watching. Stored hash untouched.
        logger.error("Job '%s': monitor source failed: %s", job_id, _mon.error)
        _mon_alert = (
            f"⚠ Cron monitor '{job_name}' source failed\n\n"
            f"{_mon.error}\n\n"
            f"Time: {_mon_now}"
        )
        return (
            False, f"{header}**Status:** monitor source failed\n\n{_mon.error}\n", _mon_alert, _mon.error,
        ), extra_prompt
    if not _mon.changed:
        # Unchanged: silent no_change tick (ledger doc kept; SILENT_MARKER blocks delivery).
        logger.info("Job '%s': monitor output unchanged — suppressing agent run", job_id)
        return (
            True, f"{header}**Status:** no_change (agent run suppressed)\n", SILENT_MARKER, None,
        ), extra_prompt
    # Changed (or first run): inject monitor context via the per-run seam, then normal agent run.
    if _mon.context_block:
        extra_prompt = (
            f"{_mon.context_block}\n\n{extra_prompt}" if extra_prompt else _mon.context_block
        )
    return None, extra_prompt


@dataclass
class _CronJobConfig:
    """Config-derived inputs for one agent-backed cron run."""

    cfg: dict
    model: str
    model_cfg: Any
    cron_default_provider: str


def _snapshot_pin(job: dict, axis: str, current: str, job_id: str) -> str:
    """The creation snapshot is an unpinned axis's effective pin: return it, logging once when it
    differs from *current* (the live global default); ``""`` for legacy jobs without one, which keep
    following the global default. A global model/provider change must never stop a cron job; a job
    keeps running on what it was created under until the operator pins it or sets a cron.* fleet
    default (#44585)."""
    snapshot = str(job.get(f"{axis}_snapshot") or "").strip()
    if snapshot and current and snapshot.lower() != current.lower():
        logger.info(
            "Job '%s': running on creation-snapshot %s %r (global default is now %r); "
            "`hermes cron edit %s --%s <value>` or cron.%s in config.yaml moves it.",
            job_id, axis, snapshot, current, job_id, axis,
            "model" if axis == "model" else "model_provider")
    return snapshot


def _load_cron_job_config(job: dict, job_id: str, job_name: str) -> _CronJobConfig:
    """Load config.yaml and resolve the run's model: per-job override > cron.model (fleet default) >
    creation snapshot > HERMES_MODEL > config ``model:``. Re-read every tick (no cache) so
    ``hermes cron edit --model`` applies next tick."""
    model = job.get("model") or cron_env_setting("HERMES_MODEL") or ""
    _cron_default_provider = ""
    _cfg: dict = {}
    _model_cfg: Any = {}
    try:
        from hermes_cli.config import read_user_config_raw
        _cfg_path = str(_get_hermes_home() / "config.yaml")
        if os.path.exists(_cfg_path):
            _cfg = read_user_config_raw(Path(_cfg_path))
            # Honor administrator-pinned managed scope (fail-open; no-op without managed scope).
            with contextlib.suppress(Exception):
                from hermes_cli import managed_scope
                _cfg = managed_scope.apply_managed_overlay(_cfg)
            _cfg = _expand_env_vars(_cfg)
            # Coerce null to {} so a falsy default never clobbers a resolved env value.
            _model_cfg = _cfg.get("model") or {}
            _cron_cfg_for_model = _cfg.get("cron") or {}
            _cron_default_model = ""
            if isinstance(_cron_cfg_for_model, dict):
                _cron_default_model = str(_cron_cfg_for_model.get("model") or "").strip()
                _cron_default_provider = str(_cron_cfg_for_model.get("model_provider") or "").strip()
            if not job.get("model"):
                if _cron_default_model:
                    model = _cron_default_model
                else:
                    _, _global_model = resolve_cron_model_drift_defaults(
                        _cfg, environ={"HERMES_MODEL": cron_env_setting("HERMES_MODEL")})
                    model = _snapshot_pin(job, "model", _global_model, job_id) or _global_model or model
    except Exception as e:
        logger.warning("Job '%s': failed to load config.yaml, using defaults: %s", job_id, e)

    # Fail fast: an empty model otherwise reaches the provider as an opaque 400.
    # See #23979.
    if not (isinstance(model, str) and model.strip()):
        raise RuntimeError(
            f"Cron job '{job_name}' has no model configured "
            f"(job.model={job.get('model')!r}, "
            f"HERMES_MODEL={cron_env_setting('HERMES_MODEL')!r}, "
            "config.yaml model.default missing or empty). "
            f"Set a per-job model via "
            f"`hermes cron edit {job_id} --model <name>` or set a "
            "default with `hermes model <name>`."
        )

    with contextlib.suppress(Exception):
        from hermes_constants import apply_ipv4_preference
        _net_cfg = _cfg.get("network", {})
        if isinstance(_net_cfg, dict) and _net_cfg.get("force_ipv4"):
            apply_ipv4_preference(force=True)
    return _CronJobConfig(_cfg, model, _model_cfg, _cron_default_provider)


def _load_prefill_messages(cfg: dict, job_id: str) -> Optional[list]:
    """Prefill messages from env or config.yaml (top-level key canonical; agent.* is legacy)."""
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg.get("agent", {}), dict) else {}
    prefill_file = (
        cron_env_setting("HERMES_PREFILL_MESSAGES_FILE")
        or cfg.get("prefill_messages_file", "")
        or agent_cfg.get("prefill_messages_file", "")
    )
    if not prefill_file:
        return None
    pfpath = Path(prefill_file).expanduser()
    if not pfpath.is_absolute():
        pfpath = _get_hermes_home() / pfpath
    if not pfpath.exists():
        return None
    try:
        with open(pfpath, "r", encoding="utf-8") as _pf:
            prefill_messages = json.load(_pf)
        return prefill_messages if isinstance(prefill_messages, list) else None
    except Exception as e:
        logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
        return None


def _preflight_or_block(job: dict, job_id: str, job_name: str, cfg: dict) -> Optional[tuple]:
    """Pre-dispatch config validation: refuse unrunnable jobs (missing key, unready skill,
    unconfigured delivery) BEFORE AIAgent is built. run_one_job keys off BLOCKED_CONFIG_MARKER to
    record blocked_config and alert once (`preflight_alerted` bit). Must run after the wake gate so
    silent ticks stay silent. Opt-out: `cron.preflight: false`. Returns failure tuple or None.
    """
    # --------------------------------------------------------------- Pre-dispatch configuration validation
    # (T1-26). A job whose configuration cannot possibly produce a successful run — missing provider API key
    # (no fallback chain), unready attached skill, unconfigured delivery platform — is refused HERE, before
    # AIAgent is constructed and before the resolution below can feed a doomed runtime into it, so a
    # misconfigured job never burns an LLM call. run_one_job keys off the BLOCKED_CONFIG_MARKER in the
    # returned error to record last_status='blocked_config' and alert exactly once (dedup persisted via the
    # job's `preflight_alerted` bit — the #73506 alert-once shape).
    _pf_reason = None
    try:
        if _cron_preflight_enabled(cfg):
            _pf_reason = _preflight_job_config(job, cfg)
            if not _pf_reason and job.get("preflight_alerted"):
                # Config healthy again: clear alert-once marker so a future break re-alerts.
                with contextlib.suppress(Exception):
                    from cron.jobs import clear_preflight_alerted
                    clear_preflight_alerted(job_id)
    except Exception:
        # Fail open: the validator must never take down a runnable job.
        logger.debug("Job '%s': preflight validation errored — failing open", job_id, exc_info=True)
        _pf_reason = None
    if not _pf_reason:
        return None

    logger.warning(
        "Job '%s' (ID: %s): BLOCKED by pre-dispatch config validation — %s (no LLM call was made)",
        job_name, job_id, _pf_reason)
    already_alerted = False
    try:
        from cron.jobs import mark_preflight_alerted
        already_alerted = mark_preflight_alerted(job_id)
    except Exception:
        logger.debug("Job '%s': could not persist preflight alert marker", job_id, exc_info=True)
    marker = BLOCKED_CONFIG_SILENT_MARKER if already_alerted else BLOCKED_CONFIG_MARKER
    blocked_doc = (
        f"# Cron Job: {job_name}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Status:** BLOCKED (configuration)\n\n"
        "Pre-dispatch validation found a configuration problem and "
        "the agent was NOT run (no tokens spent).\n\n"
        f"**Reason:** {_pf_reason}\n\n"
        "The job will stay blocked (without re-alerting) until the "
        "configuration is fixed; the next healthy run clears this "
        "state. Set `cron.preflight: false` in config.yaml to disable this validation."
    )
    return False, blocked_doc, "", f"{marker} {_pf_reason}"


def _resolve_job_runtime(job: dict, job_id: str, jc: _CronJobConfig) -> tuple[dict, str]:
    """Resolve the runtime, walking the fallback chain on auth/transient-network errors. Returns
    ``(runtime, model)``; provider+model swap atomically (never swap only the provider while keeping
    a paid primary model). Provider precedence: per-job pin > cron.model_provider > creation
    snapshot > persisted global config."""
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider, format_runtime_provider_error)
    from hermes_cli.auth import AuthError

    model = jc.model
    requested = job.get("provider") or jc.cron_default_provider or None
    if not requested:
        global_provider = (
            str(jc.model_cfg.get("provider") or "").strip() if isinstance(jc.model_cfg, dict) else "")
        # None (not the config provider) keeps the legacy no-snapshot path resolving from persisted
        # config exactly as before.
        requested = _snapshot_pin(job, "provider", global_provider, job_id) or None
    try:
        # Do NOT pass HERMES_INFERENCE_PROVIDER as `requested`: it would override persisted config
        # and resurrect stale providers for unpinned jobs.
        runtime_kwargs = {
            "requested": requested,
            # api_mode must derive from the model actually run, not the stale persisted default.
            "target_model": model,
        }
        if job.get("base_url"):
            runtime_kwargs["explicit_base_url"] = job.get("base_url")
        return resolve_runtime_provider(**runtime_kwargs), model
    except Exception as resolve_exc:
        # Walk the fallback chain on AuthError AND transient network/DNS failures (e.g. during
        # OAuth refresh); anything else re-raises.
        is_auth = isinstance(resolve_exc, AuthError)
        is_transient_net = _is_transient_provider_resolve_error(resolve_exc)
        if not (is_auth or is_transient_net):
            raise RuntimeError(format_runtime_provider_error(resolve_exc)) from resolve_exc

        logger.warning(
            "Job '%s': primary provider resolve failed (%s: %s), trying fallback",
            job_id, "auth" if is_auth else "transient network", resolve_exc)
        for entry in get_fallback_chain(jc.cfg):
            if not isinstance(entry, dict):
                continue
            fb_provider = str(entry.get("provider") or "").strip()
            fb_model = str(entry.get("model") or "").strip()
            if not fb_provider or not fb_model:
                continue
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key

                fb_kwargs = {"requested": fb_provider, "target_model": fb_model}
                if entry.get("base_url"):
                    fb_kwargs["explicit_base_url"] = entry["base_url"]
                fb_api_key = resolve_entry_api_key(entry)
                if fb_api_key:
                    fb_kwargs["explicit_api_key"] = fb_api_key
                runtime = resolve_runtime_provider(**fb_kwargs)
                logger.info(
                    "Job '%s': fallback resolved to %s model %s",
                    job_id, runtime.get("provider"), fb_model)
                return runtime, fb_model
            except Exception as fb_exc:
                logger.debug("Job '%s': fallback %s failed: %s", job_id, fb_provider, fb_exc)
        raise RuntimeError(format_runtime_provider_error(resolve_exc)) from resolve_exc


def _load_credential_pool(runtime: dict, job_id: str):
    runtime_provider = str(runtime.get("provider") or "").strip().lower()
    if not runtime_provider:
        return None
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(runtime_provider)
        if pool.has_credentials():
            logger.info(
                "Job '%s': loaded credential pool for provider %s with %d entries",
                job_id, runtime_provider, len(pool.entries()))
            return pool
    except Exception as e:
        logger.debug("Job '%s': failed to load credential pool for %s: %s", job_id, runtime_provider, e)
    return None


def _init_cron_mcp_tools(job_id: str) -> None:
    """Register MCP servers for the agent's tool registry. Idempotent across ticks; non-fatal so a
    broken MCP server never kills a working job."""
    try:
        # Initialize MCP servers so configured mcp_servers are available to the agent's tool registry before
        # AIAgent is constructed. Without this, cron jobs never saw any MCP tools — only the gateway / CLI
        # paths called discover_mcp_tools() at startup. Idempotent: subsequent ticks short-circuit on
        # already-connected servers inside register_mcp_servers(). Non-fatal on failure: a broken MCP server
        # shouldn't kill an otherwise-working cron job. See #4219.
        from tools.mcp_tool_discovery import discover_mcp_tools
        _mcp_tools = discover_mcp_tools()
        if _mcp_tools:
            logger.info("Job '%s': %d MCP tool(s) available", job_id, len(_mcp_tools))
    except Exception as _mcp_exc:
        logger.warning("Job '%s': MCP initialization failed (non-fatal): %s", job_id, _mcp_exc)


def _open_cron_session_db(job: dict):
    """Open the SQLite session store under its own timeout (HERMES_CRON_TIMEOUT only watches
    run_conversation). A wedged sqlite3.connect returns None (no session store) instead of
    wedging the worker thread."""
    # Initialize the SQLite session store so cron job messages are persisted and discoverable via
    # session_search (same pattern as gateway/run.py) — only now, after every early-return path (wake-gate,
    # prompt validation, drift skip) has passed, so a gated run never opens state.db just to abandon the
    # handle (#96290). Bounded with its own timeout (separate from HERMES_CRON_TIMEOUT, which only watches
    # the agent's run_conversation below): SessionDB.__init__ opens/migrates state.db synchronously and has
    # no timeout of its own against a wedged sqlite3.connect (e.g. a stale flock left by a crashed sibling
    # process). An unbounded hang here would wedge the job's worker thread, so the init is bounded and a
    # timeout proceeds without a session store instead of blocking the run forever.
    _session_db_timeout = _get_session_db_timeout()
    try:
        from hermes_state_registry import acquire

        if _session_db_timeout <= 0:
            return acquire()
        _session_db_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Copy the context so a profile run resolves ITS OWN home/state.db on the worker thread
        # instead of the process-global default.
        _session_db_context = contextvars.copy_context()
        _session_db_future = _session_db_pool.submit(_session_db_context.run, acquire)
        try:
            return _session_db_future.result(timeout=_session_db_timeout)
        except concurrent.futures.TimeoutError:
            # The abandoned worker may still finish; close its late result or its SQLite FDs leak.
            # The worker is abandoned (shutdown below doesn't wait for it). If SessionDB() later completes
            # inside it, the future's result would be orphaned and its SQLite FDs (.db, WAL, SHM) leak until
            # process exit. Register a done-callback that retrieves and closes any eventual late result
            # (#72782).
            _session_db_future.add_done_callback(_close_late_session_db_result)
            raise
        finally:
            # Abandon a wedged connect() rather than blocking shutdown on it.
            _session_db_pool.shutdown(wait=False)
    except concurrent.futures.TimeoutError:
        logger.error(
            "Job '%s': SessionDB init did not return within %.0fs — proceeding "
            "without a session store for this run instead of blocking it forever",
            job.get("id", "?"), _session_db_timeout)
    except Exception as e:
        logger.debug("Job '%s': SQLite session store not available: %s", job.get("id", "?"), e)
    return None


def _raise_inactivity_timeout(agent, job_name: str, limit_s: float) -> None:
    """Log the agent's last activity, hard-interrupt it and raise TimeoutError."""
    _activity = {}
    if hasattr(agent, "get_activity_summary"):
        with contextlib.suppress(Exception):
            _activity = agent.get_activity_summary()
    _last_desc = _activity.get("last_activity_desc", "unknown")
    _secs_ago = _activity.get("seconds_since_activity", 0)
    logger.error(
        "Job '%s' idle for %.0fs (inactivity limit %.0fs) "
        "| last_activity=%s | iteration=%s/%s | tool=%s",
        job_name, _secs_ago, limit_s,
        _last_desc, _activity.get("api_call_count", 0), _activity.get("max_iterations", 0),
        _activity.get("current_tool") or "none")
    request_hard_interrupt(agent, "Cron job timed out (inactivity)")
    raise TimeoutError(
        f"Cron job '{job_name}' idle for "
        f"{int(_secs_ago)}s (limit {int(limit_s)}s) "
        f"— last activity: {_last_desc}")


def _run_agent_with_watchdog(
    agent, prompt: str, job: dict, job_id: str, job_name: str, task_id: str, cancel_event,
    worker_state: Optional[dict] = None,
) -> dict:
    """Run ``agent.run_conversation`` on a worker thread under the inactivity (not wall-clock)
    watchdog: default 600s, override HERMES_CRON_TIMEOUT, 0 = unlimited."""
    _cron_timeout = _cron_inactivity_seconds()
    _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
    _POLL_INTERVAL = 5.0
    # Heartbeat the one-shot run_claim while alive: without it a long run looks like a dead owner
    # and gets re-dispatched / stale-removed out from under the live run.
    # Keep the one-shot run_claim fresh while the run is alive (#62002): the claim TTL is a dead-owner
    # detector, but without a heartbeat a run that legitimately outlives it (stream stall, laptop asleep
    # mid-run) is indistinguishable from a dead tick — another process re-dispatches it and get_due_jobs
    # stale-removes the job record out from under the live run. Refreshing the claim from this monitor keeps
    # "expired claim" meaning "owner died".
    _job_schedule = job.get("schedule")
    _is_oneshot = isinstance(_job_schedule, dict) and _job_schedule.get("kind") == "once"
    _run_claim = job.get("run_claim")
    _run_claim_owner = str(_run_claim.get("by") or "") if isinstance(_run_claim, dict) else ""
    _last_claim_heartbeat = time.monotonic()

    def _abort_if_fire_claim_lost() -> None:
        if cancel_event is None or not cancel_event.is_set():
            return
        if agent is not None and hasattr(agent, "interrupt"):
            agent.interrupt("Cron fire claim ownership was lost")
        raise RuntimeError(f"Cron job '{job_name}' lost its durable fire claim ownership")

    def _heartbeat_run_claim_if_due():
        nonlocal _last_claim_heartbeat
        if not _is_oneshot or not _run_claim_owner:
            return
        _mono = time.monotonic()
        if _mono - _last_claim_heartbeat < _RUN_CLAIM_HEARTBEAT_SECONDS:
            return
        _last_claim_heartbeat = _mono
        try:
            heartbeat_run_claim(job_id, expected_owner=_run_claim_owner)
        except Exception:
            logger.debug("Job '%s': run_claim heartbeat failed", job_name, exc_info=True)

    _cron_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    # Carry scheduler-scoped ContextVar state (e.g. env passthrough) into the worker thread.
    _cron_context = contextvars.copy_context()
    _cron_future = _cron_pool.submit(
        _cron_context.run, agent.run_conversation, prompt, task_id=task_id)
    if worker_state is not None:
        worker_state["future"] = _cron_future
    _inactivity_timeout = False
    _watch_stop = threading.Event()

    def _idle_seconds() -> float:
        if not hasattr(agent, "get_activity_summary"):
            return 0.0
        try:
            _act = agent.get_activity_summary()
            return float(_act.get("seconds_since_activity", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _watch_inactivity() -> None:
        nonlocal _inactivity_timeout
        if _cron_inactivity_limit is None:
            return
        if _inactivity_watchdog_loop(
            get_idle_seconds=_idle_seconds, limit_s=_cron_inactivity_limit, poll_s=_POLL_INTERVAL,
            stop=_watch_stop, future_done=_cron_future.done):
            _inactivity_timeout = True

    _watch_thread = threading.Thread(
        target=_watch_inactivity, name=f"cron-inactivity-{str(job_id)[:8]}", daemon=True)
    try:
        if _cron_inactivity_limit is not None:
            # Separate daemon thread so a hung get_activity_summary can't stop the limit firing.
            # Daemon thread: kernel ``Event.wait`` timeout, independent of the ``run_job`` thread. A blocked
            # loop / hung ``get_activity_summary`` on this thread can no longer keep the 600s inactivity
            # limit from firing (#94285).
            _watch_thread.start()
        if _cron_inactivity_limit is None and not _is_oneshot and cancel_event is None:
            result = _cron_future.result()
        else:
            result = None
            while True:
                done, _ = concurrent.futures.wait({_cron_future}, timeout=_POLL_INTERVAL)
                if done:
                    _abort_if_fire_claim_lost()
                    result = _cron_future.result()
                    break
                if _inactivity_timeout:
                    break
                _abort_if_fire_claim_lost()
                _heartbeat_run_claim_if_due()
    except Exception:
        _cron_pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        _watch_stop.set()
        _cron_pool.shutdown(wait=False, cancel_futures=True)

    if _inactivity_timeout:
        _raise_inactivity_timeout(agent, job_name, _cron_inactivity_limit)

    if not isinstance(result, dict):
        raise RuntimeError(
            f"agent.run_conversation returned {type(result).__name__} instead of dict: {result!r}"
        )
    return result


def _final_response_from_result(result: dict, job_id: str, job_name: str, AIAgent) -> str:
    """Deliverable final response from a ``run_conversation`` result. Raises RuntimeError on
    `failed=True`/`completed=False`: the error text may sit in `final_response` and would otherwise
    be delivered as the reply with the job marked ok."""
    # If the agent itself reported failure (e.g. all retries exhausted on API errors, model abort, mid-run
    # interrupt), do not silently mark the job as successful. run_agent populates
    # `failed=True`/`completed=False` on these paths and may put the error into `final_response`, which
    # would otherwise be delivered as if it were the agent's reply and the job's `last_status` set to "ok".
    # Raise so the except handler below builds the proper failure tuple. (issue #17855)
    turn_exit_reason = str(result.get("turn_exit_reason") or "")
    final_response_text = (result.get("final_response") or "").strip()
    max_iteration_summary = (
        result.get("failed") is not True
        and result.get("completed") is False
        and turn_exit_reason.startswith("max_iterations_reached(")
        and bool(final_response_text)
    )
    if result.get("failed") is True or (result.get("completed") is False and not max_iteration_summary):
        raise RuntimeError(result.get("error") or final_response_text or "agent reported failure")
    if max_iteration_summary:
        logger.warning(
            "Job '%s' reached the iteration limit but produced a final fallback response; "
            "delivering the response instead of failing the cron run",
            job_name)

    final_response = result.get("final_response", "") or ""
    # Repair model-mangled computer_use media paths before delivery (fail-open, as in gateway).
    if final_response:
        from gateway.media_repair import repair_explicit_computer_use_media_paths

        final_response = repair_explicit_computer_use_media_paths(
            final_response, result.get("messages", []))
    if final_response.strip() == "(No response generated)":
        final_response = ""
    # The "⚠️ No reply" turn-completion explainer would be delivered as a cron warning; detect it
    # via the same formatter and treat as empty so cron stays silent on abnormal empty turns.
    if final_response.strip() and turn_exit_reason:
        # Render every persistence-cause variant or cause-refined text slips through.
        _explainer_variants = []
        try:
            from hermes_state_errors import PERSISTENCE_ERROR_CAUSES as _causes
        except Exception:
            _causes = ("locked", "disk", "unknown")
        for _cause in (None, *_causes):
            try:
                _variant = AIAgent._format_turn_completion_explanation(turn_exit_reason, _cause)
            except TypeError:
                try:
                    _variant = AIAgent._format_turn_completion_explanation(turn_exit_reason)
                except Exception:
                    _variant = ""
            except Exception:
                _variant = ""
            if _variant:
                _explainer_variants.append(_variant.strip())
        if final_response.strip() in _explainer_variants:
            logger.info(
                "Job '%s': abnormal empty turn (%s) — suppressing explainer for cron delivery",
                job_id, turn_exit_reason)
            final_response = ""
    return final_response


def _finalize_cron_session(session_db, agent, job_id: str, job_name: str, cron_session_id: str) -> None:
    """Title, classify, end and release the cron session after the agent turn has returned."""
    # Bound every DB op so storage failure cannot hold the dispatch guard.
    _session_db = _BoundedCronSessionDB(session_db, job_id)
    # Compression may have rotated the run onto a continuation: finalize that, not the stale cron
    # id. SessionDB lineage is authoritative; agent.session_id is only a fail-safe.
    _final_cron_session_id = cron_session_id
    try:
        _compression_tip = _session_db.get_compression_tip(cron_session_id)
        if _compression_tip:
            _final_cron_session_id = _compression_tip
    except (Exception, KeyboardInterrupt) as e:
        with contextlib.suppress((Exception, KeyboardInterrupt)):
            _agent_session_id = getattr(agent, "session_id", None)
            # CLI (single-process) path: the approval contextvar is only bound during gateway/TUI turns and
            # HERMES_SESSION_KEY is not in the CLI environment, so the key resolves empty here. Since #64240
            # the CLI drains completions through a positive-ownership filter keyed on the durable
            # AIAgent.session_id — an empty session_key would fail closed and the CLI could never claim its
            # own completions, while a restored foreign event with an empty key could leak into any
            # unfiltered consumer (#64484). Stamp the parent's durable session id instead; compression
            # rotations are handled on the drain side via resolve_resume_session_id lineage resolution.
            if _agent_session_id:
                _final_cron_session_id = _agent_session_id
        logger.debug("Job '%s': failed to resolve cron compression tip: %s", job_id, e)
    # Title must persist BEFORE end_session()/close(). Run-time suffix keeps it unique against the
    # sessions.title index; the fallbacks below guarantee a non-blank title.
    try:
        # Title the cron session from the job (name -> id) and PERSIST it BEFORE end_session()/close() tear
        # the connection down, so the close can never run over an in-flight title write (#50536).
        _title_base = " ".join(job_name.split())[:60].strip() or f"cron {job_id}"
        _cron_title = f"{_title_base} · {_hermes_now().strftime('%b %d %H:%M')}"
        if not _set_cron_session_title(_session_db, _final_cron_session_id, _cron_title):
            _set_cron_session_title(_session_db, _final_cron_session_id, f"cron {job_id}")
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': failed to set cron session title: %s", job_id, e)
        # Never leave the session untitled.
        # Try the next free title in the lineage, then a bare id-stamped title. See #50535.
        for _fallback in (
            getattr(_session_db, "get_next_title_in_lineage", lambda b: b)(f"cron {job_id}"),
            f"cron {job_id} {_final_cron_session_id[-6:]}"):
            try:
                if _set_cron_session_title(_session_db, _final_cron_session_id, _fallback):
                    break
            except (Exception, KeyboardInterrupt):
                continue
    # Book cron_complete only when the last row is a real assistant reply ([SILENT] counts). Only a
    # POSITIVELY recognized bad status downgrades (keep tuple in sync with
    # session_lifecycle_statuses); unknown values / probe failures fail OPEN.
    # Verified completion booking (#93820): the run may only be recorded as cron_complete when the session's
    # LAST message row is a real assistant reply — a plain answer or the [SILENT] sentinel (both are
    # assistant-text rows, so both classify as 'complete'). A turn that died after a tool call,
    # mid-API-wait, or without any assistant text leaves the last row as a tool result / pending call / user
    # prompt and must not surface as a healthy run. session_lifecycle_statuses is the existing cost-bounded
    # classifier for exactly this shape. Only a POSITIVELY recognized pathological status (see the status
    # vocabulary in hermes_state's session_lifecycle_statuses docstring — keep the tuple below in sync when
    # it grows) downgrades the booking: an unknown value (newer classifier shape, test doubles) keeps the
    # historical reason, and so does a failed probe — the booking itself is FAIL-OPEN on probe errors,
    # because classification is best-effort metadata and must not mislabel a healthy run.
    _end_reason = "cron_complete"
    try:
        _statuses = _session_db.session_lifecycle_statuses([_final_cron_session_id])
        _lifecycle = _statuses.get(_final_cron_session_id)
        if _lifecycle in ("interrupted", "error", "empty"):
            _end_reason = "cron_incomplete_no_output"
            logger.warning(
                "Job '%s': session ended without a final assistant "
                "message (lifecycle=%s) — booking run as %s",
                job_id, _lifecycle, _end_reason)
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': session lifecycle classification failed: %s", job_id, e)
    try:
        _session_db.end_session(_final_cron_session_id, _end_reason)
        # The scheduler owns cron-session finalization. AIAgent.close() also
        # finalizes owned session rows by default; once the shared SessionDB is
        # released below, that second end_session() would reopen the just-closed
        # SQLite handle (#94736). The reason is durably booked, so disarm only the
        # agent's redundant row-finalization; its resource teardown still runs in
        # _teardown_cron_agent.
        if agent is not None:
            agent._end_session_on_close = False
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': failed to end session: %s", job_id, e)
    try:
        from hermes_state_registry import release_or_close
        release_or_close(_session_db)
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': failed to close SQLite session store: %s", job_id, e)


def _run_doc_header(job: dict, title: str, job_id: str, prompt: str) -> str:
    """Header of the persisted run document (title, ids, schedule, prompt)."""
    return (
        f"# Cron Job: {title}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Schedule:** {job.get('schedule_display', 'N/A')}\n\n"
        f"## Prompt\n\n{prompt}\n\n"
    )


_RunResult = tuple[bool, str, str, Optional[str]]


def _prepare_job_prompt(
    job: dict, job_id: str, job_name: str, extra_prompt: Optional[str], cancel_event,
) -> tuple[Optional[_RunResult], Optional[str]]:
    """Run every pre-agent gate and build the prompt. Returns ``(early_result, prompt)``: an early
    result short-circuits ``run_job`` (no_agent job, empty payload, monitor gate, wake gate,
    injection block, empty prompt); otherwise ``prompt`` is set."""
    # Fail closed on a corrupt config.yaml: defaults would let auto-detection bill a provider the
    # user never chose. no_agent jobs are exempt. Escape hatch: HERMES_IGNORE_USER_CONFIG=1.
    if not job.get("no_agent"):
        from hermes_cli.config import InvalidUserConfigError, require_parseable_user_config

        try:
            require_parseable_user_config()
        except InvalidUserConfigError as exc:
            logger.error("Job '%s': refusing to run — %s", job_id, exc)
            return (False, f"# Cron Job: {job_name}\n\nError: {exc}\n", "", str(exc)), None

    # no_agent short-circuits BEFORE importing run_agent / opening SessionDB.
    if job.get("no_agent"):
        return _run_no_agent_job(job, job_id, job_name, cancel_event), None

    # Legacy / hand-edited job with nothing to run: pause it instead of waking the LLM every fire.
    from cron.jobs import EMPTY_PAYLOAD_ERROR, job_payload_is_empty

    if job_payload_is_empty(job):
        return _block_and_pause_job(job_id, job_name, EMPTY_PAYLOAD_ERROR), None

    _early, extra_prompt = _apply_monitor_gate(job, job_id, job_name, extra_prompt)
    if _early is not None:
        return _early, None

    # Wake-gate: run the pre-check script BEFORE building the prompt; its result is passed into
    # _build_job_prompt so the script runs only once.
    # NOTE: the SQLite session store used to be initialized here, BEFORE the wake-gate and prompt-validation
    # early returns below. Every gated run (``wakeAgent: false``, blocked prompt) opened state.db and
    # returned without reaching the finally that closes it, relying on GC to release the handle. Init now
    # happens inside the main try, right before the agent is constructed — after every early-return path
    # (#96290).
    prerun_script = None
    script_path = job.get("script")
    if script_path:
        prerun_script = _run_job_script_with_claim_heartbeat(
            job,
            script_path,
            workdir=_resolve_job_workdir(job, job_id),
            cancel_event=cancel_event,
        )
        _ran_ok, _script_output = prerun_script
        if _ran_ok and not _parse_wake_gate(_script_output):
            logger.info("Job '%s' (ID: %s): wakeAgent=false, skipping agent run", job_name, job_id)
            silent_doc = (
                f"# Cron Job: {job_name}\n\n"
                f"**Job ID:** {job_id}\n"
                f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Script gate returned `wakeAgent=false` — agent skipped.\n"
            )
            return (True, silent_doc, SILENT_MARKER, None), None

    try:
        prompt = _build_job_prompt(job, prerun_script=prerun_script, extra_prompt=extra_prompt)
    except CronPromptInjectionBlocked as block_exc:
        # Injection scanner tripped: refuse this tick and tell the operator WHY.
        logger.warning(
            "Job '%s' (ID: %s): blocked by prompt-injection scanner — %s", job_name, job_id, block_exc,
        )
        blocked_doc = (
            f"# Cron Job: {job_name}\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Status:** BLOCKED\n\n"
            "The assembled prompt (user prompt + loaded skill content) tripped "
            "the cron injection scanner and the agent was NOT run.\n\n"
            f"**Scanner result:** {block_exc}\n\n"
            "Audit the skill(s) attached to this job for prompt-injection "
            "payloads or invisible-unicode markers. If the skill is legitimate "
            "and the match is a false positive, rephrase the content to avoid "
            "the threat pattern (`tools/cronjob_tools.py::_CRON_THREAT_PATTERNS`)."
        )
        return (False, blocked_doc, "", str(block_exc)), None
    if prompt is None:
        logger.info("Job '%s': script produced no output, skipping AI call.", job_name)
        return (True, "", SILENT_MARKER, None), None
    return None, prompt


_CRON_DELIVERY_VARS = (
    "HERMES_CRON_AUTO_DELIVER_PLATFORM",
    "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
    "HERMES_CRON_AUTO_DELIVER_THREAD_ID")


class _CronRunScope:
    """Per-run ContextVar / tool-cwd scope for ``run_job`` (ContextVars, not os.environ, so
    parallel jobs don't clobber each other). Construct before the try, ``enter()`` as its first
    statement, ``exit()`` in the finally — every setter here has a matching reset there.

    HERMES_SESSION_* are deliberately NOT seeded from job["origin"]: it is delivery metadata, not
    a sender, and terminal/tts/skills/send_message tools would act as if the origin user were
    driving the agent. Delivery reads job["origin"] / HERMES_CRON_AUTO_DELIVER_* directly.
    """

    def __init__(self, job: dict, job_id: str, execution_id: Optional[str]):
        from gateway.session_context import set_session_vars, _VAR_MAP
        from tools.terminal_tool import record_session_cwd

        self._var_map = _VAR_MAP
        # Resolve workdir BEFORE set_session_vars so it owns the _SESSION_CWD set/clear.
        self.workdir = _resolve_job_workdir(job, job_id)
        self._ctx_tokens = set_session_vars(
            platform="",
            chat_id="",
            chat_name="",
            # Cron can't receive completions after its turn; async delegation output could
            # otherwise route to an unrelated chat via the ambient session key => inline delegation.
            # We clear the HERMES_SESSION_* routing keys just below, so an async delegation's completion
            # event carries session_key="" — _enrich_async_delegation_routing cannot resolve it and
            # _inject_watch_notification drops it ("no routing metadata"). And by the time a child finishes,
            # run_job has already shipped the job's final response via _deliver_result; there is no turn
            # left to re-enter. (Worse, get_current_session_key() can fall back to the ambient os.environ
            # HERMES_SESSION_KEY, which risks routing a cron subagent's output into an unrelated user chat.)
            # Declaring the channel stateless routes delegate_task to its existing inline/synchronous path,
            # so results return within the job's own turn. See declare_stateless_channel(). Upstream:
            # #53027, #63142.
            async_delivery=False,
            cwd=self.workdir or "",
        )
        for name in _CRON_DELIVERY_VARS:
            _VAR_MAP[name].set("")
        # Workdir binds to the per-run task id (tool-layer cwd authority) instead of mutating
        # global TERMINAL_CWD; _SESSION_CWD above remains the prompt/context-file authority.
        self.task_id = f"cron:{job_id}:{execution_id or job.get('execution_id') or uuid.uuid4().hex}"
        if self.workdir:
            record_session_cwd(self.task_id, self.workdir)
        self._cron_session_var = _VAR_MAP["HERMES_CRON_SESSION"]
        self._cron_session_token = None
        self._non_dispatcher_token = None

    def enter(self) -> None:
        # Scope cron approval policy; exit() RESETS via token (pinning "" would suppress the legacy
        # os.environ fallback used by standalone entrypoints/tests).
        self._cron_session_token = self._cron_session_var.set("1")
        # Mark NOT the kanban worker: a worker's cronjob(action="run") lands here with
        # HERMES_KANBAN_TASK in env, and an unrelated job could close the worker's task. Must be a
        # ContextVar, NOT an os.environ clear (env is shared with the worker heartbeat and
        # concurrent jobs); copy_context() carries it into the agent thread.
        self._non_dispatcher_token = enter_non_dispatcher_owned_context()

    def exit(self) -> None:
        from gateway.session_context import clear_session_vars
        from tools.terminal_tool import clear_session_cwd

        clear_session_cwd(self.task_id)
        clear_session_vars(self._ctx_tokens)  # also clears _SESSION_CWD
        if self._cron_session_token is not None:
            self._cron_session_var.reset(self._cron_session_token)
        if self._non_dispatcher_token is not None:
            exit_non_dispatcher_owned_context(self._non_dispatcher_token)
        for name in _CRON_DELIVERY_VARS:
            self._var_map[name].set("")


def _reload_dotenv_and_publish_delivery_target(job: dict) -> None:
    """Re-read .env for this run and publish the auto-deliver target into the session ContextVars."""
    # Reset the secret-source cache FIRST or a Bitwarden/BSM-backed secret is never re-resolved
    # (only the placeholder reloads -> 401s).
    from hermes_cli.env_loader import load_hermes_dotenv, reset_secret_source_cache
    from gateway.session_context import _VAR_MAP

    reset_secret_source_cache(_get_hermes_home())
    load_hermes_dotenv(hermes_home=_get_hermes_home())

    delivery_target = _resolve_delivery_target(job)
    if delivery_target:
        _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set(delivery_target["platform"])
        _VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"].set(str(delivery_target["chat_id"]))
        _VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"].set(
            "" if delivery_target.get("thread_id") is None else str(delivery_target["thread_id"])
        )


@dataclass
class _CronAgentSetup:
    """Everything ``AIAgent(...)`` needs that is resolved from job + config (or a preflight block)."""
    blocked: Optional[_RunResult] = None
    model: str = ""
    runtime: dict = None
    prefill_messages: Any = None
    max_iterations: Any = None
    reasoning_config: Any = None
    fallback_model: Any = None
    credential_pool: Any = None


def _resolve_cron_agent_setup(job: dict, job_id: str, job_name: str, jc) -> _CronAgentSetup:
    """Resolve model/runtime/reasoning/pool for the run, in the original gate order: exfil guard ->
    preflight (may block) -> runtime (+ fallback chain) -> credential pool -> MCP."""
    _cfg = jc.cfg
    setup = _CronAgentSetup(model=jc.model)
    setup.prefill_messages = _load_prefill_messages(_cfg, job_id)

    # resolve_turn_limit() honors none/unlimited (sys.maxsize) and explicit 0 / null.
    from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
    _mt = _cfg.get("agent", {}).get("max_turns")
    if _mt is None:
        _mt = _cfg.get("max_turns")
    setup.max_iterations = _resolve_turn_limit(_mt)

    # Runtime backstop (CWE-200/522): fail closed BEFORE resolution on a provider/base_url pair
    # that would ship a stored credential off-host; hand-written jobs bypass create-time checks.
    _guard_job_credential_exfil(job)

    setup.blocked = _preflight_or_block(job, job_id, job_name, _cfg)
    if setup.blocked is not None:
        return setup

    setup.runtime, setup.model = _resolve_job_runtime(job, job_id, jc)
    setup.reasoning_config = _resolve_job_reasoning_config(
        job, _cfg if isinstance(_cfg, dict) else {}, str(setup.model)
    )
    setup.fallback_model = get_fallback_chain(_cfg) or None
    setup.credential_pool = _load_credential_pool(setup.runtime, job_id)
    # MCP servers must be registered before AIAgent is constructed.
    _init_cron_mcp_tools(job_id)
    return setup


def _construct_cron_agent(AIAgent, job: dict, _cfg: dict, setup: _CronAgentSetup, *, workdir, session_id, session_db):
    runtime = setup.runtime
    pr = _cfg.get("provider_routing") or {}
    return AIAgent(
        model=setup.model,
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        requested_provider=runtime.get("requested_provider"),
        api_mode=runtime.get("api_mode"),
        request_overrides=runtime.get("request_overrides"),
        acp_command=runtime.get("command"),
        acp_args=runtime.get("args"),
        max_iterations=setup.max_iterations,
        reasoning_config=setup.reasoning_config,
        prefill_messages=setup.prefill_messages,
        fallback_model=setup.fallback_model,
        credential_pool=setup.credential_pool,
        providers_allowed=pr.get("only"),
        providers_ignored=pr.get("ignore"),
        providers_order=pr.get("order"),
        provider_sort=pr.get("sort"),
        openrouter_min_coding_score=(_cfg.get("openrouter") or {}).get("min_coding_score"),
        enabled_toolsets=_resolve_cron_enabled_toolsets(job, _cfg),
        disabled_toolsets=_resolve_cron_disabled_toolsets(_cfg),
        quiet_mode=True,
        # Project context files only with a configured workdir; SOUL.md always.
        skip_context_files=not bool(workdir),
        load_soul_identity=True,
        skip_memory=False,
        skip_background_review=True,  # Cron has no human-in-the-loop need for skill/memory review forks (~30K tok/event)
        platform="cron",
        session_id=session_id,
        session_db=session_db,
    )


class _FireAudit:
    """One usage_audit.jsonl line per fire (created once the agent exists; fire id + start clock)."""

    def __init__(self, job: dict, job_id: str, model: str):
        self.job, self.job_id, self.model = job, job_id, model
        self.fire_id = uuid.uuid4().hex
        self.t_start = time.monotonic()

    def write(self, result: dict, error: Optional[str]) -> None:
        _write_usage_audit({
            "ts": _utcnow_iso_ms(),
            "job_id": self.job_id,
            "fire_id": self.fire_id,
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "total_tokens": result.get("total_tokens"),
            "response_silent": bool(result.get("response_silent")),
            "deliver_target": self.job.get("deliver"),
            "model": self.model or None,
            "duration_ms": int((time.monotonic() - self.t_start) * 1000),
            "error": error})



def run_job(
    job: dict, *, defer_agent_teardown: Optional[list] = None, extra_prompt: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None, execution_id: Optional[str] = None,
) -> tuple[bool, str, str, Optional[str]]:
    """Execute a single cron job. Returns (success, full_output_doc, final_response, error).
    ``defer_agent_teardown``: if a list, the live agent is appended instead of torn down; the caller
    MUST call ``_teardown_cron_agent(agent)`` AFTER delivery (a torn-down async client can't
    deliver). ``extra_prompt``: per-fire context, never persisted.

    ``defer_agent_teardown``: when a caller passes a list, ``run_job`` skips the agent's async-resource
    teardown (``agent.close()`` + ``cleanup_stale_async_clients()``) in its ``finally`` block and instead
    appends the live agent to that list. The caller is then responsible for calling
    ``_teardown_cron_agent(agent)`` AFTER it has delivered the result. This closes the ordering window in
    #58720 where delivery ran against a torn-down async client (defense-in-depth alongside the
    interpreter-shutdown guard). When ``None`` (the default) teardown happens inline as before, so every
    existing caller is unchanged.
    ``extra_prompt``: optional per-run context from ``cronjob(action='run', prompt=...)`` (#57331). Appended
    to the stored prompt for this fire only — never persisted to the job definition.
    """
    job_id = job["id"]
    job_name = str(job.get("name") or job.get("prompt") or job_id or "cron job")

    early, prompt = _prepare_job_prompt(job, job_id, job_name, extra_prompt, cancel_event)
    if early is not None:
        return early
    from run_agent import AIAgent

    _cron_session_id = f"cron_{job_id}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    agent = None
    model = ""
    _session_db = None
    _audit: Optional[_FireAudit] = None
    _worker_state: dict = {}
    scope = _CronRunScope(job, job_id, execution_id)
    try:
        scope.enter()
        if scope.workdir:
            logger.info("Job '%s': using task-scoped workdir %s", job_id, scope.workdir)
        _reload_dotenv_and_publish_delivery_target(job)

        jc = _load_cron_job_config(job, job_id, job_name)
        _cfg = jc.cfg
        model = jc.model
        setup = _resolve_cron_agent_setup(job, job_id, job_name, jc)
        if setup.blocked is not None:
            return setup.blocked
        model = setup.model

        # Open state.db only after every early-return gate has passed.
        _session_db = _open_cron_session_db(job)
        agent = _construct_cron_agent(
            AIAgent, job, _cfg, setup, workdir=scope.workdir, session_id=_cron_session_id,
            session_db=_session_db)
        _audit = _FireAudit(job, job_id, model)

        result = _run_agent_with_watchdog(
            agent, prompt, job, job_id, job_name, scope.task_id, cancel_event,
            worker_state=_worker_state)
        final_response = _final_response_from_result(result, job_id, job_name, AIAgent)
        # Keep final_response clean for delivery logic (empty = no delivery).
        logged_response = final_response if final_response else "(No response generated)"
        output = _run_doc_header(job, job_name, job_id, prompt) + f"## Response\n\n{logged_response}\n"
        logger.info("Job '%s' completed successfully", job_name)
        _audit.write(dict(result, response_silent=_is_cron_silence_response(final_response or "")), None)
        return True, output, final_response, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception("Job '%s' failed: %s", job_name, error_msg)
        # No audit row when we failed before the agent existed; the audit write must never raise.
        if _audit is not None:
            _audit.write({}, error_msg)
        from cron.scheduler_diagnostics import format_run_error
        output = (
            _run_doc_header(job, f"{job_name} (FAILED)", job_id, prompt)
            + format_run_error(e)
        )
        return False, output, "", error_msg

    finally:
        from cron.scheduler_detached_worker import defer_teardown_to_running_worker
        _worker_teardown_deferred = defer_teardown_to_running_worker(
            _worker_state.get("future"), _session_db, agent, job_id, job_name, _cron_session_id)
        scope.exit()
        if _session_db and not _worker_teardown_deferred:
            _finalize_cron_session(_session_db, agent, job_id, job_name, _cron_session_id)
        # Tear down the ephemeral agent or the gateway leaks fds per tick (EMFILE). With deferred
        # teardown, hand the live agent back: delivery needs a live async client.
        # Release subprocesses, terminal sandboxes, browser daemons, and the main OpenAI/httpx client held
        # by this ephemeral cron agent. Without this, a gateway that ticks cron every N minutes leaks fds
        # per job until it hits EMFILE (#10200 / "too many open files"). When the caller opted to defer
        # teardown (passed a list), hand the live agent back instead of closing it here — delivery must run
        # against a live async client, and the caller tears down afterwards (#58720).
        if not _worker_teardown_deferred:
            if defer_agent_teardown is not None:
                if agent is not None:
                    defer_agent_teardown.append(agent)
            else:
                _teardown_cron_agent(agent, job_id)


def _teardown_cron_agent(
    agent, job_id: str, *, timeout_seconds: Optional[float] = None
) -> None:
    """Release an ephemeral cron agent's async resources within a hard bound (this runs outside the
    inactivity watchdog). Shared by ``run_job``'s finally and deferred post-delivery teardown.

    Split out of ``run_job``'s ``finally`` so a caller that defers teardown (to deliver first — #58720) can
    invoke the identical cleanup AFTER delivery. The timeout matters because this executes after
    ``run_conversation`` has returned, outside the agent inactivity watchdog.
    """
    def _cleanup_agent() -> None:
        try:
            if agent is not None:
                agent.close()
        except (Exception, KeyboardInterrupt) as e:
            logger.debug("Job '%s': failed to close agent resources: %s", job_id, e)
        # Worker-thread event loop dies with the executor; reap httpx clients cached under it.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception as e:
            logger.debug("Job '%s': failed to reap stale auxiliary clients: %s", job_id, e)

    _run_cron_cleanup_with_timeout(
        _cleanup_agent, job_id=job_id, label="agent resource teardown",
        timeout_seconds=timeout_seconds)


def _run_with_fire_claim_heartbeat(job: dict, run) -> bool:
    """Run ``run`` while keeping this job's owned durable fire claim fresh."""
    claim = job.get("fire_claim")
    owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    if not owner:
        return run(None)

    job_id = str(job.get("id") or "")
    stop = threading.Event()
    lost_ownership = threading.Event()

    def _finish_unstarted(error: str) -> None:
        execution_id = job.get("execution_id")
        if not execution_id:
            return
        try:
            finish_execution(execution_id, success=False, error=error)
        except Exception:
            logger.warning(
                "Job '%s': failed to close unstarted execution ledger row",
                job_id,
                exc_info=True)

    try:
        owns_fire_claim = heartbeat_fire_claim(job_id, expected_owner=owner)
    except Exception:
        logger.warning("Job '%s': initial fire_claim validation failed", job_id, exc_info=True)
        _finish_unstarted("Fire claim ownership could not be validated before execution started.")
        return True

    if owns_fire_claim is False:
        logger.warning("Job '%s': fire claim ownership was already lost before execution", job_id)
        _finish_unstarted("Fire claim ownership lost before execution started.")
        return True

    def _heartbeat_loop() -> None:
        last_confirmed = time.monotonic()
        while not stop.wait(_RUN_CLAIM_HEARTBEAT_SECONDS):
            try:
                if not heartbeat_fire_claim(job_id, expected_owner=owner):
                    lost_ownership.set()
                    logger.warning(
                        "Job '%s': fire claim ownership lost; interrupting stale run",
                        job_id)
                    return
                last_confirmed = time.monotonic()
            except Exception:
                logger.debug("Job '%s': fire_claim heartbeat failed", job_id, exc_info=True)
                if (
                    time.monotonic() - last_confirmed
                    >= _FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS
                ):
                    lost_ownership.set()
                    logger.warning(
                        "Job '%s': fire_claim could not be renewed within %.1fs; "
                        "interrupting uncertain run",
                        job_id,
                        _FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS)
                    return

    heartbeat_thread = _start_heartbeat_thread(
        _heartbeat_loop, "cron-fire-claim-heartbeat",
        lambda: logger.warning(
            "Job '%s': could not start fire_claim heartbeat", job_id, exc_info=True))
    if heartbeat_thread is None:
        _finish_unstarted("Fire claim heartbeat could not be started; execution was not run.")
        return True

    try:
        return run(lost_ownership)
    finally:
        stop.set()
        heartbeat_thread.join(timeout=1.0)


def run_one_job(
    job: dict, *, adapters=None, loop=None, verbose: bool = False,
    extra_prompt: Optional[str] = None, cancel_event: Optional[_CancelEventLike] = None,
) -> bool:
    """Run ONE due job end-to-end: execute → save output → deliver → mark. Shared by the built-in
    ticker and external providers' ``fire_due``; does NOT decide due-ness or acquire the initial
    claim (callers use the store CAS) but keeps it alive. True if processed (a job failure is
    recorded via ``mark_job_run``), False only if processing raised. ``cancel_event``: optional
    transport-level cancel (dashboard drain)."""
    # Every gateway path (built-in scheduler, external providers, and direct
    # API fires) crosses this seam.  Ensure the detached worker has a durable
    # attempt to adopt before any launch can occur.
    if not job.get("execution_id"):
        execution = create_execution(
            job["id"], source="direct", scheduled_instant=job.get("_scheduled_instant"))
        job["execution_id"] = execution["id"]

    execution_id = str(job["execution_id"])
    external_owner = os.environ.get("_HERMES_CRON_EXTERNAL_WORKER") == execution_id
    if not external_owner:
        try:
            if _launch_external_cron_worker(job):
                return True
        except Exception as handoff_error:
            error = f"Restart-safe cron worker dispatch failed: {handoff_error}"
            logger.error("Job '%s': %s", job["id"], error)
            claim = job.get("fire_claim")
            owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
            try:
                mark_job_run(
                    job["id"],
                    False,
                    error,
                    **({"expected_fire_owner": owner} if owner else {}),
                )
            finally:
                finish_execution(execution_id, success=False, error=error)
            return True
    if extra_prompt is None:
        # Gateway-forwarded manual run stamps its prompt on the job via trigger_job; the fire that
        # consumes the manual occurrence picks it up here. Single-fire: mark_job_run clears it.
        _stamped = job.get("manual_run_prompt")
        if _stamped and job.get("manual_run_at"):
            extra_prompt = str(_stamped)
    claim = job.get("fire_claim")
    fire_owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    execution_token = object()
    profile_home = _get_hermes_home().resolve()
    with _running_lock:
        _running_fire_owners.setdefault(job["id"], {})[execution_token] = (
            fire_owner or None, profile_home)
    try:
        return _run_with_fire_claim_heartbeat(
            job,
            lambda lost_ownership: _run_one_job_body(
                job,
                adapters=adapters,
                loop=loop,
                verbose=verbose,
                extra_prompt=extra_prompt,
                fire_claim_lost=(
                    _CombinedCancelEvent(lost_ownership, cancel_event)
                    if cancel_event is not None
                    else lost_ownership
                ),
                execution_token=execution_token))
    finally:
        with _running_lock:
            executions = _running_fire_owners.get(job["id"])
            if executions is not None:
                executions.pop(execution_token, None)
                if not executions:
                    _running_fire_owners.pop(job["id"], None)


_OWNERSHIP_LOST_INTERRUPTED = "Interrupted by shutdown before terminal completion."


def _record_fire_ownership_lost(job_id: str, fire_owner: Optional[str], execution_id: str) -> None:
    """Bookkeeping after fire-claim ownership loss. A transport-level cancel (dashboard drain) is
    not a real loss — we still own the claim, so record the interruption via the owner-fenced
    terminal write instead of leaving fire_claim/last_status stale; otherwise discard."""
    if fire_owner is not None and heartbeat_fire_claim(job_id, expected_owner=fire_owner):
        mark_job_run(job_id, False, _OWNERSHIP_LOST_INTERRUPTED, expected_fire_owner=fire_owner)
        finish_execution(execution_id, success=False, error=_OWNERSHIP_LOST_INTERRUPTED)
    else:
        finish_execution(
            execution_id, success=False,
            error="Fire claim ownership lost; stale result was discarded.")


def _classify_delivery_outcome(
    *, delivery_error, should_deliver: bool, unresolved_origin: bool,
    normalized_deliver: str, incident_acked: bool, success: bool,
    delivery_queued=None,
) -> str:
    if delivery_error:
        return "failed"
    if should_deliver and delivery_queued:
        return "queued"
    if should_deliver and unresolved_origin:
        return "not_configured"
    if should_deliver and normalized_deliver != "local":
        return "delivered"
    if incident_acked and not success:
        # Failure ping withheld: operator acked this exact signature (vs. plain "suppressed").
        return "suppressed_acked"
    return "suppressed"


def _compose_run_delivery(
    job: dict, *, success: bool, error, final_response: str, output_file,
) -> tuple[str, bool, bool, bool, Optional[str]]:
    """Text to deliver for a finished run. Returns ``(deliver_content, blocked_config,
    silent_alert, incident_acked, failure_incident_id)``; ``silent_alert``: an alert-once marker
    says the operator was already told, deliver nothing."""
    err = str(error) if error else ""
    # Failed jobs always deliver, except blocked-config runs, which alert exactly ONCE.
    blocked_config_silent = BLOCKED_CONFIG_SILENT_MARKER in err
    blocked_config = blocked_config_silent or BLOCKED_CONFIG_MARKER in err
    incident_acked = False
    failure_incident_id = None
    if blocked_config and not success:
        # Bypass the generic failure summarizer (its auth/timeout heuristics would mislabel this).
        _pf_text = re.sub(r"\[blocked_config[^\]]*\]\s*", "", err).strip()
        deliver_content = (
            f"⛔ Cron '{job.get('name') or job['id']}' blocked by "
            f"configuration validation (no LLM call was made): "
            f"{_pf_text} "
            "This alert is sent once; the job stays blocked until the configuration is fixed."
        )
    elif success:
        deliver_content = final_response
    else:
        # Record the job+error signature once; if already acked by the operator, suppress the
        # per-run ping. Best-effort: a ledger failure never breaks delivery.
        incident_acked, failure_incident_id = _upsert_incident_for_failure(
            job, error or "", output_file=output_file
        )
        if incident_acked:
            deliver_content = ""
        else:
            deliver_content = (
                _summarize_cron_failure_for_delivery(job, error) + _failure_streak_nudge(job)
            )
    return deliver_content, blocked_config, blocked_config_silent, incident_acked, failure_incident_id


class _FireClaimLostDuringSideEffect(Exception):
    """Raised inside a side-effect fence when the durable fire claim is no longer ours."""


class _FireOwnership:
    """Fire-claim ownership checks for one run (``owner`` is None when the job carries no claim)."""

    def __init__(self, job: dict, fire_claim_lost: Optional[_CancelEventLike]):
        self.job = job
        self.fire_claim_lost = fire_claim_lost
        claim = job.get("fire_claim")
        self.owner = str(claim.get("by") or "") if isinstance(claim, dict) else None

    def side_effect_fence(self):
        if self.owner is None:
            return contextlib.nullcontext(True)
        return fire_claim_fence(self.job["id"], expected_owner=self.owner)

    def lost(self) -> bool:
        if self.fire_claim_lost is not None and self.fire_claim_lost.is_set():
            return True
        if self.owner is None:
            return False
        try:
            if heartbeat_fire_claim(self.job["id"], expected_owner=self.owner):
                return False
        except Exception:
            logger.debug(
                "Job '%s': fire_claim ownership validation failed", self.job["id"], exc_info=True)
            return False
        if self.fire_claim_lost is not None:
            self.fire_claim_lost.set()
        return True


@dataclass
class _RunDelivery:
    """Mutable outcome of the save/compose/deliver phase, read back by the bookkeeping tail."""
    job: dict
    success: bool
    error: Optional[str]
    delivery_attempted: bool = False
    delivery_error: Optional[str] = None
    should_deliver: bool = False
    unresolved_origin: bool = False
    blocked_config: bool = False
    incident_acked: bool = False
    failure_incident_id: Optional[str] = None
    side_effect_ownership_lost: bool = False


def _save_compose_deliver(
    d: _RunDelivery, fence: _FireOwnership, final_response: str, output: str, *,
    adapters, loop, verbose: bool, execution_token,
) -> None:
    """Save output, compose the notice and deliver it (both side effects run under the fire-claim
    fence; a lost claim raises ``_FireClaimLostDuringSideEffect`` for the caller)."""
    job = d.job
    with fence.side_effect_fence() as owns_output:
        if not owns_output:
            raise _FireClaimLostDuringSideEffect
        output_file = save_job_output(job["id"], output)
    if verbose:
        logger.info("Output saved to: %s", output_file)

    # A shutdown-killed tool subprocess can leave a plausible final_response from truncated
    # output; force the honest "interrupted" failure path. Peek-only (consumed later).
    if d.success and _is_interrupted(job["id"], execution_token):
        d.success = False
        d.error = (
            "Interrupted by gateway shutdown before the run finished "
            "(tool subprocess was killed mid-flight)."
        )

    (
        deliver_content, d.blocked_config, _silent_alert, d.incident_acked, d.failure_incident_id,
    ) = _compose_run_delivery(
        job, success=d.success, error=d.error, final_response=final_response,
        output_file=output_file)
    # Whitespace-only == empty: skip delivery; the guard below marks it a soft failure.
    d.should_deliver = bool(deliver_content.strip()) and not _silent_alert
    # Not a substring check: bare "SILENT"/"NO_REPLY" or a report quoting "[SILENT]" must
    # not be swallowed; bracketed-prefix / trailing-line tolerance is kept.
    if d.should_deliver and d.success and _is_cron_silence_response(deliver_content):
        # Cron silence suppression — see _is_cron_silence_response. Replaces the old `SILENT_MARKER in
        # ...upper()` substring check, which both leaked bracketless near-markers ("SILENT" / "NO_REPLY")
        # and wrongly swallowed a real report that merely quoted "[SILENT]" mid-sentence (#51438, #46917).
        logger.info("Job '%s': agent returned %s — skipping delivery", job["id"], SILENT_MARKER)
        d.should_deliver = False

    if d.should_deliver and fence.lost():
        d.should_deliver = False
        logger.warning("Job '%s': skipping delivery after fire claim ownership loss", job["id"])

    if not d.should_deliver:
        return
    d.unresolved_origin = (
        _normalize_deliver_value(_delivery_lane_value(job, for_failure=not d.success)) == "origin"
        and not _resolve_delivery_targets(job, for_failure=not d.success)
    )
    try:
        with fence.side_effect_fence() as owns_delivery:
            if not owns_delivery:
                raise _FireClaimLostDuringSideEffect
            d.delivery_attempted = True
            d.delivery_error = _deliver_result(
                job,
                deliver_content,
                adapters=adapters,
                loop=loop,
                # Failure summaries (and drift/blocked-config alerts composed into deliver_content
                # on the failure path) honor the job's failure_deliver override (NS-788).
                for_failure=not d.success,
            )
    except Exception as de:
        if isinstance(de, _FireClaimLostDuringSideEffect):
            raise
        d.delivery_error = str(de)
        logger.error("Delivery failed for job %s: %s", job["id"], de)


def _finish_interrupted_run(job: dict, execution_id: str, delivery_error: Optional[str]) -> None:
    """Shutdown already wrote last_status, so mark_job_run is skipped (a second call would skip a
    fire or auto-delete the job); an unsent notice is recorded via update_job instead."""
    if delivery_error:
        try:
            # The gateway shutdown already wrote last_status for this run, so mark_job_run is skipped below
            # — but it could not know that the notice we just tried to send never left the process (the
            # adapters were torn down first, #82232). Record the delivery failure on its own via update_job:
            # mark_job_run also advances next_run_at and the repeat counter, and running that a second time
            # for one run would skip a fire or auto-delete the job early.
            from cron.jobs import update_job
            update_job(job["id"], {"last_delivery_error": delivery_error})
        except Exception as _rec_err:
            logger.debug(
                "Failed recording delivery_error for interrupted job %s: %s", job["id"], _rec_err)
    finish_execution(
        execution_id, success=False,
        error="Interrupted by gateway shutdown before terminal completion.")


def _finish_completed_run(d: _RunDelivery, fire_owner: Optional[str], execution_id: str) -> bool:
    """mark_job_run (owner-fenced) + execution ledger row for a run that reached delivery."""
    job = d.job
    if not d.should_deliver and job.get("last_delivery_queued"):
        from cron.jobs import update_job
        update_job(job["id"], {"last_delivery_queued": None})
        job["last_delivery_queued"] = None
    mark_kwargs = {"delivery_error": d.delivery_error}
    if d.success and not d.delivery_error and d.should_deliver and job.get("last_delivery_queued"):
        mark_kwargs["status"] = "delivery_queued"
    if fire_owner is not None:
        mark_kwargs["expected_fire_owner"] = fire_owner
    if d.blocked_config:
        mark_kwargs["status"] = "blocked_config"
    marked = mark_job_run(job["id"], d.success, d.error, **mark_kwargs)
    if fire_owner is not None and not marked:
        finish_execution(
            execution_id, success=False,
            error="Fire claim ownership lost before terminal completion.")
        return True
    delivery_outcome = _classify_delivery_outcome(
        delivery_error=d.delivery_error,
        delivery_queued=job.get("last_delivery_queued"),
        should_deliver=d.should_deliver,
        unresolved_origin=d.unresolved_origin,
        # Read the lane the notice was actually routed through (failure_deliver on failure).
        normalized_deliver=_normalize_deliver_value(_delivery_lane_value(job, for_failure=not d.success)),
        incident_acked=d.incident_acked,
        success=d.success,
    )
    if delivery_outcome in ("delivered", "not_configured") and not d.success:
        # Failure ping left the process (or had a configured target): mark the incident alerted.
        _mark_incident_alerted(d.failure_incident_id)
    finish_execution(
        execution_id, success=d.success, error=d.error, delivery_outcome=delivery_outcome)
    return True


def _deliver_crash_failure(
    job: dict, err_text: str, *, adapters, loop,
) -> tuple[Optional[str], str]:
    """Failure notice for a run that raised out of run_job. Returns (delivery_error, outcome)."""
    normalized_deliver = _normalize_deliver_value(_delivery_lane_value(job, for_failure=True))
    # Same ack gate as the normal failure delivery: acked signatures stay silent here too.
    incident_acked, failure_incident_id = _upsert_incident_for_failure(job, err_text)
    if incident_acked:
        return None, "suppressed_acked"
    delivery_error = None
    try:
        delivery_error = _deliver_result(
            job,
            # Same text as the normal failure delivery: this run also counts toward
            # failure_streak, so the nudge must leave through here too.
            _summarize_cron_failure_for_delivery(job, err_text) + _failure_streak_nudge(job),
            adapters=adapters,
            loop=loop,
            for_failure=True,
        )
    except Exception as delivery_exc:
        delivery_error = str(delivery_exc)
        logger.error("Delivery failed for job %s: %s", job["id"], delivery_exc)
    unresolved_origin = bool(
        not delivery_error
        and normalized_deliver == "origin"
        and not _resolve_delivery_targets(job, for_failure=True)
    )
    delivery_outcome = _classify_delivery_outcome(
        delivery_error=delivery_error, should_deliver=True, unresolved_origin=unresolved_origin,
        normalized_deliver=normalized_deliver, incident_acked=False, success=False,
        delivery_queued=job.get("last_delivery_queued"))
    if delivery_outcome in ("delivered", "not_configured"):
        _mark_incident_alerted(failure_incident_id)
    return delivery_error, delivery_outcome



def _run_one_job_body(
    job: dict, *, adapters=None, loop=None, verbose: bool = False,
    extra_prompt: Optional[str] = None, fire_claim_lost: Optional[_CancelEventLike] = None,
    execution_token: Optional[object] = None,
) -> bool:
    fence = _FireOwnership(job, fire_claim_lost)
    fire_owner = fence.owner
    _side_effect_fence = fence.side_effect_fence
    _fire_claim_ownership_lost = fence.lost

    execution_id = job.get("execution_id")
    if not execution_id:
        execution_id = create_execution(
            job["id"], source="direct", scheduled_instant=job.get("_scheduled_instant"))["id"]
    delivery_attempted = False
    delivery_error = None
    from agent.secret_scope import (
        build_profile_secret_scope, reset_secret_scope, set_secret_scope)

    _scope_token = None
    _terminal_scope_token = None
    try:
        # Commit a finite one-shot's dispatch BEFORE its side effect so a tick dying mid-run cannot
        # re-fire it forever on restart. No-op for recurring/infinite jobs (at-most-times).
        # This lives here in the shared body so BOTH the built-in ticker and the external provider (Chronos
        # fire_due) get at-most-times semantics. See #38758.
        if not claim_dispatch(job["id"]):
            logger.info(
                "Job '%s': one-shot dispatch limit reached — skipping",
                job.get("name", job["id"]))
            finish_execution(
                execution_id, success=False,
                error="Dispatch claim rejected; execution was not started.")
            return True  # not an error — already handled/removed

        # Claimed durably before dispatch; becomes running only right before the actual run.
        # Detached workers transition to running while adopting; in-process paths must win the
        # claimed->running CAS here before any user script or agent side effect may begin.
        external_owner = os.environ.get("_HERMES_CRON_EXTERNAL_WORKER") == execution_id
        if not external_owner and mark_execution_running(execution_id) is None:
            logger.warning("Cron job %s lost execution ownership before start; skipping", job["id"])
            return True

        # get_secret() fails closed outside a scope; the ticker thread has none. Delivery adapters
        # resolve credentials, so the scope must span delivery too (reset in the outer finally).
        _scope_token = set_secret_scope(build_profile_secret_scope(_get_hermes_home()))
        # Same for terminal policy (gateway/run.py _profile_runtime_scope): else the ticker reads
        # process-global TERMINAL_* env a concurrent profile pinned. Resolution failure installs a
        # refusal scope — terminal execution raises instead of using the launch process's policy.
        # Same isolation for terminal settings (third profile seam; see gateway/run.py
        # _profile_runtime_scope): installs the firing profile's COMPLETE terminal policy for this fire —
        # run, delivery, and bookkeeping — resetting in this function's finally alongside the secret scope.
        # See #68559.
        # Bind the profile's COMPLETE terminal policy for the agent build (fail-closed: malformed policy →
        # refusal scope) so _make_agent's terminal probing / cwd hints resolve the routed profile, never the
        # launch process (#98581 class).
        # Same authoritative terminal policy the gateway binds per turn (#68559): a docker-configured
        # dashboard profile must never resolve the launch process's pinned env.
        # Fourth profile seam: bind the session profile's COMPLETE terminal policy for this turn
        # (dashboard/TUI analogue of the gateway's per-turn scope). #98581's unified-desktop reproduction
        # ran a docker-configured profile on the host because terminal_tool read the launch process's pinned
        # env.
        from tools.terminal_scope import (
            install_profile_terminal_scope)

        _terminal_scope_token = install_profile_terminal_scope(_get_hermes_home())
        # Defer agent teardown until AFTER delivery; closing first races the live send against a
        # torn-down async client. run_job hands the agent back via this list.
        # Defer the cron agent's async-resource teardown until AFTER delivery. run_job normally closes the
        # agent (and reaps stale async clients) in its finally block; doing that before _deliver_result runs
        # means the live send races a torn-down async client (#58720). Passing a holder list makes run_job
        # hand the agent back instead, and we tear it down below once delivery is done. Defense-in-depth
        # alongside the interpreter-shutdown guard in _deliver_result.
        _deferred_agents: list = []

        def _teardown_deferred() -> None:
            # run_job's finally still hands back the agent when it raises; tear it down here so a failed run
            # never leaks its async resources (#10200), then re-raise into the outer handler. BaseException
            # (not just Exception) so a KeyboardInterrupt/SystemExit mid-run still triggers teardown before
            # propagating.
            # Tear down the deferred agent(s) now that save + delivery have run (or raised). Must happen on
            # every path so cron agents never leak their subprocesses/clients (#10200).
            for _deferred_agent in _deferred_agents:
                _teardown_cron_agent(_deferred_agent, job["id"])

        _run_kwargs = {
            "defer_agent_teardown": _deferred_agents,
            "extra_prompt": extra_prompt,
            "execution_id": execution_id}
        if fire_claim_lost is not None:
            _run_kwargs["cancel_event"] = fire_claim_lost
        try:
            success, output, final_response, error = run_job(job, **_run_kwargs)
        except BaseException:
            # run_job hands back the agent even when raising; tear down so a failed run never leaks.
            # BaseException so KeyboardInterrupt/SystemExit mid-run still trigger teardown.
            _teardown_deferred()
            raise

        if _fire_claim_ownership_lost():
            _teardown_deferred()
            _record_fire_ownership_lost(job["id"], fire_owner, execution_id)
            return True

        # Agent is still live through delivery; wrap ALL of save/compose/deliver in try/finally so a
        # raise anywhere still tears the deferred agent down.
        d = _RunDelivery(job=job, success=success, error=error)
        try:
            _save_compose_deliver(
                d, fence, final_response, output, adapters=adapters, loop=loop, verbose=verbose,
                execution_token=execution_token)
        except _FireClaimLostDuringSideEffect:
            d.side_effect_ownership_lost = True
        finally:
            delivery_attempted, delivery_error = d.delivery_attempted, d.delivery_error
            # Every path must tear down deferred agent(s) so they never leak subprocesses/clients.
            _teardown_deferred()

        if d.side_effect_ownership_lost or _fire_claim_ownership_lost():
            _record_fire_ownership_lost(job["id"], fire_owner, execution_id)
            return True

        # Empty final_response is a soft failure so last_status is not "ok".
        if d.success and not final_response.strip():
            d.success = False
            d.error = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"

        if _consume_interrupted_flag(job["id"], execution_token):
            _finish_interrupted_run(job, execution_id, delivery_error)
            return True

        return _finish_completed_run(d, fire_owner, execution_id)

    except BaseException as e:  # noqa: BLE001 — deliberate: see below
        # BaseException, not Exception: CancelledError/KeyboardInterrupt/SystemExit propagate here.
        # Without mark_job_run(False) a finite one-shot is wedged: claim_dispatch consumed
        # repeat.completed but last_run_at is never written. Record first, then re-raise
        # non-Exception. Owner fencing still applies.
        # BaseException, not Exception (#73973): the inner run_job handler re-raises CancelledError /
        # KeyboardInterrupt / SystemExit after agent teardown, and none of those are Exception subclasses.
        # If they escape without mark_job_run(False), a finite one-shot is left wedged — claim_dispatch()
        # already consumed repeat.completed, but last_run_at is never written, so the job sits in state
        # "scheduled" until the run-claim TTL expires and the dispatch-limit guard removes it with no output
        # and no error. Owner fencing still applies: a stale worker must not record over a replacement claim
        # owner.
        _err_text = str(e) or type(e).__name__
        logger.error(
            "Error processing job %s: %s",
            job["id"],
            _err_text,
            exc_info=(type(e), e, e.__traceback__))
        delivery_outcome = "suppressed"
        # Owner fencing: a stale worker whose claim was taken over (or transport-cancelled) must not
        # send a failure alert on top of the replacement run's; fall through to fenced bookkeeping.
        if (
            isinstance(e, Exception)
            and not delivery_attempted
            and not isinstance(e, _FireClaimLostDuringSideEffect)
            and not _fire_claim_ownership_lost()
        ):
            delivery_error, delivery_outcome = _deliver_crash_failure(
                job, _err_text, adapters=adapters, loop=loop)
        try:
            if not _consume_interrupted_flag(job["id"], execution_token):
                mark_kwargs = {}
                if fire_owner is not None:
                    mark_kwargs["expected_fire_owner"] = fire_owner
                if isinstance(e, Exception):
                    mark_kwargs["delivery_error"] = delivery_error
                mark_job_run(job["id"], False, _err_text, **mark_kwargs)
        except Exception as record_err:
            # Never let bookkeeping mask the original interruption.
            logger.error("Failed to record interrupted run for job %s: %s", job["id"], record_err)
        try:
            finish_execution(
                execution_id, success=False, error=_err_text, delivery_outcome=delivery_outcome)
        except Exception as record_err:
            logger.error("Failed to finish execution record for job %s: %s", job["id"], record_err)
        if not isinstance(e, Exception):
            raise
        return False
    finally:
        # Function-level on purpose: must scope delivery, deferred teardown, claim-loss handling and
        # bookkeeping — not just run_job. Do not move into the run block's finally.
        if _scope_token is not None:
            reset_secret_scope(_scope_token)
        if _terminal_scope_token is not None:
            from tools.terminal_scope import reset_terminal_scope

            reset_terminal_scope(_terminal_scope_token)


def _wait_for_external_cron_worker_body(
    process: subprocess.Popen,
    *,
    execution_id: str,
) -> bool:
    """Preserve ``run_one_job``'s synchronous contract after handoff.

    The worker owns the durable execution and survives this gateway process.
    The caller nevertheless waits while it remains alive so manual/background
    callers do not release their in-process guard or report stale job state.
    A gateway replacement may kill this waiter; it does not kill the scoped
    worker or change its ledger ownership.
    """
    def _is_terminal() -> bool:
        current = get_execution(execution_id)
        return bool(current and current.get("status") in _TERMINAL_STATES)

    # The worker commits its terminal row before its process exits, so exit is
    # the correct wakeup.  Each ledger read opens a connection and re-runs
    # schema init; polling it at 50ms for an hours-long agent run is ~72k
    # opens/hour of pure contention with the worker's own writes.
    while True:
        try:
            returncode = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            if _is_terminal():
                return True
            continue
        # The worker can commit its terminal row and exit between the first
        # read and wait(). Re-read the exact attempt before declaring that
        # it died without terminalizing.
        if _is_terminal():
            return True
        # If the adopted worker died without terminalizing, its owner is
        # now provably gone. Recover to ``unknown`` rather than routing the
        # exception through the pre-handoff dispatch-failure path, which
        # would falsely assert that no side effect could have happened.
        recover_interrupted_executions()
        if _is_terminal():
            return True
        raise RuntimeError(
            "cron external worker exited before durable recovery could "
            f"terminalize its execution state (exit {returncode})"
        )


def _wait_for_external_cron_worker(
    process: subprocess.Popen,
    *,
    execution_id: str,
    job_id: Optional[str] = None,
    handoff_files: tuple[Path, ...] = (),
) -> bool:
    try:
        return _wait_for_external_cron_worker_body(
            process, execution_id=execution_id
        )
    finally:
        if job_id is not None:
            with _running_lock:
                _restart_safe_waiter_job_ids.discard(job_id)
        # The execution is terminal or its worker is dead: nobody will read a
        # payload or acknowledgement left behind by a late/unread handoff.
        for stale in handoff_files:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass


def _launch_external_cron_worker(job: dict) -> bool:
    """Launch *job* outside a managed gateway cgroup when required.

    Returns ``False`` when the caller is not a managed systemd gateway and the
    existing in-process path should be used.  In managed topology, failure to
    establish the transient scope raises: falling back would recreate the
    restart interruption this handoff exists to prevent.
    """
    execution_id = str(job["execution_id"])
    job_id = str(job["id"])
    handoff_dir = _get_hermes_home() / "cron" / "external-workers"
    payload_path = handoff_dir / f"{execution_id}.json"
    ack_path = handoff_dir / f"{execution_id}.ready"
    command = [
        sys.executable,
        "-m",
        "cron.scheduler",
        "--external-worker-file",
        str(payload_path),
        "--ack-file",
        str(ack_path),
    ]

    from agent.secret_scope import (
        build_profile_secret_scope,
        is_multiplex_active,
        reset_secret_scope,
        set_secret_scope,
    )
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from tools.environments.local import build_subprocess_env, strip_launch_profile_env
    from tools.process_registry import (
        restart_safe_gateway_child_argv,
        systemd_user_bus_env,
    )

    multiplex_active = is_multiplex_active()
    scoped_command = restart_safe_gateway_child_argv(
        command,
        unit_suffix=f"cron-{job_id}-exec-{execution_id}",
    )
    if scoped_command == command:
        return False

    if mark_execution_handoff_pending(execution_id) is None:
        raise RuntimeError(
            "cron execution claim changed before external worker handoff"
        )

    _ensure_cron_dir(handoff_dir)
    try:
        handoff_dir.chmod(0o700)
    except OSError:
        pass
    fd = os.open(payload_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as payload_file:
            json.dump(
                {
                    "job": job,
                    "profile_home": str(_get_hermes_home().resolve()),
                    "multiplex_active": multiplex_active,
                },
                payload_file,
            )
            payload_file.flush()
            os.fsync(payload_file.fileno())
    except BaseException:
        payload_path.unlink(missing_ok=True)
        raise

    profile_home = _get_hermes_home().resolve()
    hydrate_profile_secret_sources(profile_home)
    secret_token = set_secret_scope(build_profile_secret_scope(profile_home))
    try:
        worker_env = strip_launch_profile_env(build_subprocess_env(
            scrub_secrets=multiplex_active,
            inherit_profile_home=True,
            extra={"HERMES_HOME": str(profile_home)},
        ))
    finally:
        reset_secret_scope(secret_token)
    worker_env = systemd_user_bus_env(worker_env)
    try:
        process = subprocess.Popen(
            scoped_command,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=worker_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            creationflags=windows_hide_flags(),
        )
    except BaseException:
        payload_path.unlink(missing_ok=True)
        raise

    with _running_lock:
        _restart_safe_waiter_job_ids.add(job_id)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if ack_path.exists():
            try:
                acknowledgement = json.loads(ack_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception(
                    "Cron external worker %s published an unreadable acknowledgement; "
                    "treating handoff as ownership-uncertain",
                    execution_id,
                )
                return _wait_for_external_cron_worker(
                    process,
                    execution_id=execution_id,
                    job_id=job_id,
                    handoff_files=(payload_path,),
                )
            finally:
                ack_path.unlink(missing_ok=True)
            if (
                not isinstance(acknowledgement, dict)
                or acknowledgement.get("execution_id") != execution_id
            ):
                logger.error(
                    "Cron external worker acknowledgement mismatch for %s; "
                    "treating handoff as ownership-uncertain",
                    execution_id,
                )
                return _wait_for_external_cron_worker(
                    process,
                    execution_id=execution_id,
                    job_id=job_id,
                    handoff_files=(payload_path,),
                )
            logger.info(
                "Cron job '%s' handed to restart-safe worker pid=%s execution=%s",
                job_id,
                acknowledgement.get("pid"),
                execution_id,
            )
            return _wait_for_external_cron_worker(
                process,
                execution_id=execution_id,
                job_id=job_id,
                handoff_files=(payload_path,),
            )
        returncode = process.poll()
        if returncode is not None:
            with _running_lock:
                _restart_safe_waiter_job_ids.discard(job_id)
            payload_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"cron external worker exited before ownership acknowledgement "
                f"(exit {returncode})"
            )
        time.sleep(0.05)

    # The child may have adopted the durable row just before publishing its
    # acknowledgement.  Never fall back to in-process execution on an uncertain
    # handoff: that could duplicate side effects.  The execution owner/dead-owner
    # recovery ledger remains the authority.
    logger.warning(
        "Cron external worker for job '%s' did not acknowledge within 5s; "
        "leaving the durable execution claim untouched",
        job_id,
    )
    return _wait_for_external_cron_worker(
        process,
        execution_id=execution_id,
        job_id=job_id,
        handoff_files=(payload_path, ack_path),
    )


def _run_external_worker_payload(payload_path: Path, ack_path: Path) -> bool:
    """Adopt and execute one gateway-dispatched cron payload.

    The execution row is created by the gateway before spawn, then transferred
    here before the ready acknowledgement is published.  No side effect runs
    unless that durable ownership transfer succeeds.
    """
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        job = payload["job"]
        profile_home = Path(payload["profile_home"]).resolve()
        execution_id = str(job["execution_id"])
    except Exception:
        logger.exception("Cron external worker could not load payload %s", payload_path)
        return False
    finally:
        try:
            payload_path.unlink(missing_ok=True)
        except OSError:
            pass

    from agent.secret_scope import (
        build_profile_secret_scope,
        is_multiplex_active,
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from cron.executions import adopt_claimed_execution
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    home_token = set_hermes_home_override(profile_home)
    previous_multiplex = is_multiplex_active()
    multiplex_active = bool(payload.get("multiplex_active", False))
    set_multiplex_active(multiplex_active)
    hydrate_profile_secret_sources(profile_home)
    secret_token = set_secret_scope(build_profile_secret_scope(profile_home))
    try:
        with use_cron_store(profile_home):
            if adopt_claimed_execution(execution_id) is None:
                logger.error(
                    "Cron external worker refused execution %s: durable ownership "
                    "could not be established",
                    execution_id,
                )
                return False
            try:
                ack_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(ack_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as ack_file:
                    json.dump({"pid": os.getpid(), "execution_id": execution_id}, ack_file)
                    ack_file.flush()
                    os.fsync(ack_file.fileno())
            except Exception:
                logger.exception(
                    "Cron external worker could not publish ready acknowledgement for %s",
                    execution_id,
                )
                return False
            old_external_execution = os.environ.get("_HERMES_CRON_EXTERNAL_WORKER")
            os.environ["_HERMES_CRON_EXTERNAL_WORKER"] = execution_id
            try:
                return run_one_job(job, adapters=None, loop=None, verbose=False)
            finally:
                if old_external_execution is None:
                    os.environ.pop("_HERMES_CRON_EXTERNAL_WORKER", None)
                else:
                    os.environ["_HERMES_CRON_EXTERNAL_WORKER"] = old_external_execution
    finally:
        reset_secret_scope(secret_token)
        set_multiplex_active(previous_multiplex)
        reset_hermes_home_override(home_token)


def _notify_provider_jobs_changed() -> None:
    """Best-effort: tell the active scheduler provider the job set changed. Call AFTER a successful
    store mutation so an external provider can re-provision/cancel the one-shot; no-op for the
    built-in. Kept out of cron/jobs.py (import cycle). Never raises."""
    try:
        from cron.scheduler_provider import resolve_cron_scheduler
        resolve_cron_scheduler().on_jobs_changed()
    except Exception as e:
        logger.debug("on_jobs_changed notify failed: %s", e)


class CronSchedulerRegistrationError(RuntimeError):
    """A job was persisted but its first external trigger was not registered."""

    def __init__(self, job: dict, cause: Exception) -> None:
        self.job = job
        self.cause = cause
        super().__init__(
            f"Cron job '{job['id']}' was saved, but its first scheduler "
            f"registration failed ({type(cause).__name__}). Do not create a "
            "duplicate. Pause/resume or update the job to retry registration."
        )

    def user_message(self) -> str:
        """Human-facing variant for chat/CLI surfaces (no exception class name)."""
        label = self.job.get("name") or self.job["id"]
        return (
            f"Saved cron job '{label}', but couldn't register it with the "
            "external scheduler yet. The job is kept — don't re-create it; "
            "pause/resume or edit it (e.g. via /cron) to retry registration."
        )

    def to_dict(self) -> dict:
        """Return the public partial-failure contract without provider details."""
        return {
            "error": str(self),
            "job_id": self.job["id"],
            "job_saved": True,
            "scheduler_registered": False,
            "retry_create": False}


def create_job_with_scheduler_registration(**kwargs) -> dict:
    """Persist one job and register its first trigger with the active provider."""
    from cron.jobs import create_job
    from cron.scheduler_provider import resolve_cron_scheduler

    job = create_job(**kwargs)
    if not job.get("enabled", True):
        return job
    try:
        resolve_cron_scheduler().register_job(job)
    except Exception as exc:
        raise CronSchedulerRegistrationError(job, exc) from exc
    return job


# Dead-owner reap is throttled (opens the executions ledger). Tests may reset
# _last_dead_owner_reap_at to None to force a reap next tick.
# Dead-owner claim reclaim throttle (#86721): recover_interrupted_executions opens the executions ledger, so
# the per-tick reap is rate-limited rather than run on every idle 60s cycle.
_DEAD_OWNER_REAP_INTERVAL_SECONDS = 300.0
_last_dead_owner_reap_at: Optional[float] = None

# Worktree prune throttle: the cron tick is the only reliably periodic process on gateway boxes.
_WORKTREE_MAINTENANCE_INTERVAL_SECONDS = 6 * 3600.0
_last_worktree_maintenance_at: Optional[float] = None
_worktree_maintenance_lock = threading.Lock()


def _worktree_maintenance_repos() -> List[str]:
    """Repos whose ``.worktrees/`` to keep pruned: the hermes checkout plus job workdir repo roots,
    filtered to those that actually have a ``.worktrees/`` dir."""
    repos: set = set()

    # Hermes source checkout (git installs only; wheel installs have no .git).
    with contextlib.suppress(Exception):
        install_root = Path(__file__).resolve().parent.parent
        if (install_root / ".git").exists():
            repos.add(str(install_root))

    with contextlib.suppress(Exception):
        from cron.jobs import load_jobs

        for job in load_jobs():
            workdir = str(job.get("workdir") or "").strip()
            if not workdir or not Path(workdir).is_dir():
                continue
            try:
                probe = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=5, cwd=workdir)
                if probe.returncode == 0 and probe.stdout.strip():
                    repos.add(probe.stdout.strip())
            except Exception:
                continue

    return [r for r in sorted(repos) if (Path(r) / ".worktrees").is_dir()]


def _maybe_run_worktree_maintenance() -> None:
    """Throttled worktree prune from the cron tick, on a daemon thread so the tick never waits on
    git. Same conservative pruner as ``hermes -w`` startup (dirty/unpushed/locked trees untouched).
    Errors never propagate: GC is hygiene, not scheduling."""
    global _last_worktree_maintenance_at
    now = time.monotonic()
    with _worktree_maintenance_lock:
        if (
            _last_worktree_maintenance_at is not None
            and now - _last_worktree_maintenance_at
            < _WORKTREE_MAINTENANCE_INTERVAL_SECONDS
        ):
            return
        _last_worktree_maintenance_at = now

    def _run() -> None:
        try:
            repos = _worktree_maintenance_repos()
            if not repos:
                return
            from cli import _prune_stale_worktrees

            for repo in repos:
                try:
                    _prune_stale_worktrees(repo)
                except Exception:
                    logger.debug("Cron worktree maintenance failed for %s", repo, exc_info=True)
        except Exception:
            logger.debug("Cron worktree maintenance skipped", exc_info=True)

    threading.Thread(target=_run, name="cron-worktree-prune", daemon=True).start()


def _acquire_tick_lock(lock_file):
    """Open + non-blocking lock the tick file (fcntl / msvcrt). Returns the fd, or None on genuine
    contention. A real OSError (esp. EMFILE/ENFILE) must NOT pass as contention — the scheduler
    would look healthy while no job runs — so it is re-raised for the ticker to record a FAILED
    tick."""
    lock_fd = None
    try:
        # Cross-platform file locking: fcntl on Unix, msvcrt on Windows. Only genuine lock contention
        # (another ticker holds the lock) skips the tick silently. A real OSError — most importantly
        # EMFILE/ENFILE from fd exhaustion — must NOT be swallowed as "another instance holds the lock":
        # that previously made the scheduler appear healthy (tick returned 0, heartbeat recorded success)
        # while no job ever ran again (#87644).
        lock_fd = open(lock_file, "w", encoding="utf-8")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        return lock_fd
    except OSError as exc:
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                lock_fd.close()
            if _is_lock_contention_errno(exc):
                logger.debug("Tick skipped — another instance holds the lock")
                return None
        if _is_fd_exhaustion(exc):
            # fd reclamation is the ticker loop's job (scheduler_provider.py); here would double it.
            logger.error(
                "Cron tick could not acquire tick lock: %s — scheduler will "
                "attempt fd reclamation and retry with backoff",
                exc)
        else:
            logger.error("Cron tick could not acquire tick lock: %s", exc)
        raise


def _release_tick_lock(lock_fd) -> None:
    if fcntl:
        with contextlib.suppress((OSError, IOError)):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    elif msvcrt:
        with contextlib.suppress((OSError, IOError)):
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
    lock_fd.close()


def _maybe_reap_dead_owners() -> None:
    """Dead-owner reclaim: a run that died mid-flight would leave its row 'claimed' forever. Only
    rows whose owner process is proved gone are touched (_owner_is_live). Throttled."""
    # Dead-owner claim reclaim (#86721): execution rows carry their owner pid + process start time, but
    # recovery previously ran only at scheduler STARTUP. A one-shot `hermes cron run` that claimed a job and
    # died mid-run (its runner thread lived in the exiting CLI process) left the row 'claimed' forever while
    # the long-lived gateway ticker kept running — blocking every future run of that job. Reap provably-dead
    # owners periodically so stale claims auto-clear without a gateway restart. Throttled so idle 60s ticks
    # don't pay a ledger connection every cycle (#33612).
    global _last_dead_owner_reap_at
    _reap_now = time.monotonic()
    if (
        _last_dead_owner_reap_at is not None
        and _reap_now - _last_dead_owner_reap_at < _DEAD_OWNER_REAP_INTERVAL_SECONDS
    ):
        return
    _last_dead_owner_reap_at = _reap_now
    try:
        from cron.executions import recover_interrupted_executions

        _reclaimed = recover_interrupted_executions()
        if _reclaimed:
            logger.warning(
                "Reclaimed %d cron execution(s) whose owner process died "
                "before reaching a terminal state (marked unknown)",
                _reclaimed)
    except Exception as _reap_exc:
        logger.debug("Dead-owner execution reclaim failed: %s", _reap_exc)


def _sweep_stale_inflight_for_tick(due_jobs: list) -> None:
    """Bound the in-flight set BEFORE the dedup guard so a leaked claim is force-released now
    rather than eating every later fire until restart. Skipped when nothing is in flight."""
    if not _running_job_ids:
        return
    _sweep_jobs = due_jobs
    with contextlib.suppress(Exception):
        _inflight_ids = set(_running_job_ids)
        _due_ids = {j.get("id") for j in due_jobs if isinstance(j, dict)}
        if not _inflight_ids <= _due_ids:
            from cron.jobs import load_jobs as _load_all_jobs

            _sweep_jobs = _load_all_jobs()
    try:
        sweep_stale_inflight(_sweep_jobs)
    except Exception as e:
        logger.warning("Stale in-flight sweep failed: %s", e)


def _resolve_max_parallel_workers() -> Optional[int]:
    """Max workers: env > config.yaml > unbounded (HERMES_CRON_MAX_PARALLEL=1 restores serial)."""
    try:
        _env_par = cron_env_setting("HERMES_CRON_MAX_PARALLEL").strip()
        if _env_par:
            return int(_env_par) or None
    except (ValueError, TypeError):
        logger.warning("Invalid HERMES_CRON_MAX_PARALLEL value; defaulting to unbounded")
    with contextlib.suppress(Exception):
        _ucfg = load_config() or {}
        _cfg_par = (_ucfg.get("cron", {}) if isinstance(_ucfg, dict) else {}).get("max_parallel_jobs")
        if _cfg_par is not None:
            return int(_cfg_par) or None
    return None


def _sweep_mcp_orphans() -> None:
    """Reap MCP stdio orphans (only PIDs flagged by tools.mcp_tool._run_stdio's finally block);
    run AFTER jobs finish so live sessions are never touched."""
    try:
        from tools.mcp_tool_lifecycle import _kill_orphaned_mcp_children
        _kill_orphaned_mcp_children()
    except Exception as _e:
        logger.debug("Post-tick MCP orphan cleanup failed: %s", _e)


def _process_due_job(job: dict, adapters, loop, verbose: bool) -> bool:
    """Run one due job via the shared ``run_one_job`` body."""
    # Claim only when the worker actually starts, so a queued lease can't expire first.
    claimed = claim_job_for_fire(job["id"], return_job=True)
    if not claimed:
        finish_execution(
            job["execution_id"], success=False, error="Fire claim lost; execution was not started.")
        return True
    # CAS returns the persisted record; bool fallback only for older test doubles.
    claimed_job = dict(claimed) if isinstance(claimed, dict) else dict(job)
    claimed_job["execution_id"] = job["execution_id"]
    claimed_job["_scheduled_instant"] = job.get("_scheduled_instant")
    return run_one_job(claimed_job, adapters=adapters, loop=loop, verbose=verbose)


def _submit_with_guard(job: dict, pool: concurrent.futures.ThreadPoolExecutor, process_job):
    """Submit with the in-flight dedup guard; None if a prior tick's run is still in flight.
    Running-set membership is released in the worker's finally."""
    job_id = job["id"]
    job_label = job.get("name", job_id)

    def _clear_run_claim_best_effort() -> None:
        """Best-effort claim cleanup on dispatch-failure paths. Only one-shots carry a run_claim;
        clear_run_claim takes _jobs_lock + full load/save and can raise on degraded paths
        (shutdown, EMFILE) — a claim expiring at TTL beats crashing the tick.

        Only one-shot jobs carry a ``run_claim`` (stamped by get_due_jobs, #59229), so recurring jobs skip
        the call entirely — clear_run_claim acquires _jobs_lock (blocking cross-process flock) and does a
        full load_jobs read, and the dispatch-failure paths fire exactly when the process can least afford N
        pointless lock/read round-trips (interpreter shutdown, EMFILE).  clear_run_claim itself does
        load_jobs/save_jobs file I/O; on those degraded paths it can raise, and these early-exits exist
        precisely to skip cleanly — a stale claim expiring at the TTL is a better outcome than crashing the
        tick (#86522).
        """
        _schedule = job.get("schedule")
        if not (isinstance(_schedule, dict) and _schedule.get("kind") == "once"):
            return
        try:
            clear_run_claim(job_id)
        except Exception as claim_err:
            logger.warning(
                "Could not clear run_claim for job '%s' after dispatch "
                "failure: %s (claim will expire at TTL)",
                job_label, claim_err)

    def _not_dispatched_shutdown() -> None:
        logger.warning("Job '%s' not dispatched — interpreter is shutting down", job_label)

    # During interpreter shutdown pool.submit raises; skip — the job fires on the next tick.
    # If the interpreter is finalizing (gateway SIGTERM / restart / OOM), scheduling any new delivery is
    # futile — asyncio.run and a fresh ThreadPoolExecutor both raise "cannot schedule new futures after
    # interpreter shutdown". Skip gracefully with a warning rather than emitting an ERROR traceback on every
    # restart-race (#58720, #55924).
    # A tick can race gateway teardown: once the interpreter is finalizing, ``pool.submit`` raises "cannot
    # schedule new futures after interpreter shutdown" and crashes the tick. Skip cleanly — the job stays
    # due and will fire on the next healthy tick (#58720, #55924).
    if _interpreter_shutting_down():
        _not_dispatched_shutdown()
        _clear_run_claim_best_effort()
        return None
    if not try_register_running_job(job_id):
        logger.info("Job '%s' already running — skipping", job_label)
        return None
    # Record the attempt before dispatch; recovery marks abandoned rows unknown (no retry).
    try:
        execution = create_execution(
            job_id, source="builtin", scheduled_instant=job.get("_scheduled_instant"))
        dispatched_job = dict(job, execution_id=execution["id"])
        _ctx = contextvars.copy_context()
    except Exception as execution_err:
        # Release the claim so the next tick retries instead of wedging "already running".
        release_running_job(job_id)
        _clear_run_claim_best_effort()
        logger.exception(
            "Job '%s' not dispatched: execution creation failed: %s", job_label, execution_err)
        return None

    def _run_and_release(j=dispatched_job, ctx=_ctx):
        try:
            return ctx.run(process_job, j)
        finally:
            release_running_job(j["id"])

    try:
        fut = pool.submit(_run_and_release)
    except Exception as submit_err:
        release_running_job(job_id)
        _clear_run_claim_best_effort()
        finish_execution(
            execution["id"], success=False, error=f"Executor dispatch failed: {submit_err}")
        if isinstance(submit_err, RuntimeError) and _interpreter_shutting_down(submit_err):
            _not_dispatched_shutdown()
        else:
            logger.error("Job '%s' not dispatched: %s", job_label, submit_err)
        return None

    with _running_lock:
        if job_id in _running_job_ids:
            _running_futures[job_id] = fut
    return fut


def _sweep_mcp_orphans_when_all_done(futures: list) -> None:
    """Async (gateway ticker) mode: sweep via a done-callback after the LAST job completes; sweep
    inline when nothing was dispatched (all skipped / no due jobs)."""
    if not futures:
        _sweep_mcp_orphans()
        return
    _remaining = [len(futures)]

    def _on_done(_f: concurrent.futures.Future) -> None:
        _remaining[0] -= 1
        with contextlib.suppress(Exception):
            _exc = _f.exception()
            if _exc is not None:
                logger.error(
                    "Cron job future failed in async mode: %s", _exc,
                    exc_info=(type(_exc), _exc, _exc.__traceback__))
        if _remaining[0] <= 0:
            _sweep_mcp_orphans()

    for _f in futures:
        _f.add_done_callback(_on_done)


def tick(
    verbose: bool = True, adapters=None, loop=None, sync: bool = True, *, can_dispatch=None):
    """Check and run all due jobs. File-locked so only one tick runs at a time (gateway ticker vs
    standalone daemon / manual tick). ``can_dispatch``: optional gate; false leaves due jobs for the
    next allowed tick. Returns the number of jobs executed (0 if another tick holds the lock)."""
    # Stale-code yield gate — BEFORE the lock race. A process whose checkout was updated under it
    # serves mixed sys.modules (jobs die on ImportErrors); if a fresher gateway holds the runtime
    # lock, ITS ticker dispatches. With no fresh holder (desktop-standalone) the tick proceeds.
    _skew = _should_yield_tick_to_fresh_gateway()
    if _skew is not None:
        _log_tick_yield_once(f"boot={_skew[0]} disk={_skew[1]}")
        raise CronTickYielded(_skew[0], _skew[1])

    lock_dir, lock_file = _get_lock_paths()
    _ensure_cron_dir(lock_dir)
    lock_fd = _acquire_tick_lock(lock_file)
    if lock_fd is None:
        return 0

    try:
        # `hermes pause` ESTOP: skip dispatch, never touch in-flight runs; check_paused logs once.
        with contextlib.suppress(ImportError):
            from agent.estop import check_paused as _estop_check_paused
            if _estop_check_paused("cron", logger):
                return 0

        if can_dispatch is not None and not can_dispatch():
            logger.debug("Cron dispatch paused while gateway drains existing work")
            return 0

        _maybe_reap_dead_owners()
        # Periodic worktree GC (6h, threaded) — the only sweep gateway-only boxes get.
        try:
            _maybe_run_worktree_maintenance()
        except Exception as _wt_exc:
            logger.debug("Worktree maintenance dispatch failed: %s", _wt_exc)

        due_jobs = get_due_jobs()
        _sweep_stale_inflight_for_tick(due_jobs)

        if not due_jobs:
            # Idle tick: skip config load + pool setup, but still reap crashed jobs' MCP orphans.
            if verbose:
                # Idle tick: skip config load + pool partitioning entirely (#33612 — the gateway ticker
                # calls tick(verbose=False) every 60s, so idle ticks previously fell through to
                # load_config()). Still run the post-tick MCP orphan sweep: main intentionally sweeps on
                # idle ticks so orphaned stdio children from crashed jobs are reaped even when nothing is
                # due.
                logger.info("%s - No jobs due", _hermes_now().strftime('%H:%M:%S'))
            _sweep_mcp_orphans()
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", _hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        # Advance next_run_at for recurring jobs FIRST, under the lock, before any execution
        # (at-most-once). Re-advancing running jobs keeps the grace window alive; mark_job_run
        # overwrites it on completion. Composes with the claim-time advance in claim_job_for_fire.
        advance_next_runs([job["id"] for job in due_jobs])

        _max_workers = _resolve_max_parallel_workers()
        if verbose:
            logger.info(
                "Running %d job(s) in parallel (max_workers=%s)",
                len(due_jobs),
                _max_workers if _max_workers else "unbounded")

        def _process_job(job: dict) -> bool:
            return _process_due_job(job, adapters, loop, verbose)

        # Persistent pool, non-blocking dispatch. Already-running jobs are skipped; mark_job_run
        # re-arms next_run_at on completion, so no catch-up queue is needed.
        _results: list = []
        _all_futures: list = []
        pool = _get_parallel_pool(_max_workers)
        for job in due_jobs:
            fut = _submit_with_guard(job, pool, _process_job)
            if fut is None:
                continue
            _all_futures.append(fut)
            if not sync:
                _results.append(True)  # optimistically counted

        if sync:
            for f in concurrent.futures.as_completed(_all_futures):
                try:
                    _results.append(f.result())
                except Exception as exc:
                    logger.error("Cron job future failed: %s", exc)
                    _results.append(False)
            _sweep_mcp_orphans()
            return sum(_results)

        _sweep_mcp_orphans_when_all_done(_all_futures)
        return sum(_results)
    finally:
        _release_tick_lock(lock_fd)


# ---------------------------------------------------------------------------
# Split modules. Imported at the bottom (import cycle: they late-bind ``cron.scheduler`` as
# ``_sched``). Only names this module itself calls; everything else lives in the split module.
# ---------------------------------------------------------------------------
from cron.scheduler_delivery import (  # noqa: E402
    _deliver_result, _delivery_lane_value, _normalize_deliver_value, _resolve_delivery_target,
    _resolve_delivery_targets,
)
from cron.scheduler_script import (  # noqa: E402
    _get_session_db_timeout, _run_job_script_with_claim_heartbeat, _start_heartbeat_thread,
)
from cron.scheduler_prompt import (  # noqa: E402
    _block_and_pause_job, _build_job_prompt, _guard_job_credential_exfil, _parse_wake_gate,
)
from cron.scheduler_preflight import (  # noqa: E402
    BLOCKED_CONFIG_MARKER, BLOCKED_CONFIG_SILENT_MARKER, _cron_preflight_enabled,
    _is_transient_provider_resolve_error, _preflight_job_config,
)


# `python -m cron.scheduler` entry: MUST stay below the split-module imports so the worker /
# tick paths see every name they need.
if __name__ == "__main__":
    if "--external-worker-file" in sys.argv:
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--external-worker-file", type=Path, required=True)
        parser.add_argument("--ack-file", type=Path, required=True)
        args = parser.parse_args()
        # The gateway spawns this worker with stdout/stderr on DEVNULL; without
        # a handler every adoption/ack failure below would be invisible.
        try:
            from hermes_logging import setup_logging

            setup_logging(hermes_home=_get_hermes_home(), mode="cron")
        except Exception:
            pass
        raise SystemExit(
            0 if _run_external_worker_payload(args.external_worker_file, args.ack_file) else 1
        )
    tick(verbose=True)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import asyncio  # noqa: F401,E402
import shutil  # noqa: F401,E402
import signal  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'BOT_CHAT_PLATFORM': ('cron.scheduler_delivery', 'BOT_CHAT_PLATFORM'),
    'SharedRouteAdapters': ('cron.scheduler_preflight', 'SharedRouteAdapters'),
    'cron_delivery_targets': ('cron.scheduler_delivery', 'cron_delivery_targets'),
    'parse_bot_chat_deliver_token': ('cron.scheduler_delivery', 'parse_bot_chat_deliver_token'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
