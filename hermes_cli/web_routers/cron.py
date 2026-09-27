"""Cron dashboard routes.

The ``*_sync`` workers, profile resolution and the threadpool wrapper
(``_run_cron_dashboard_io``) live in web_server_cron and are reached through the
late-binding seam so ``monkeypatch.setattr(web_server_cron, ...)`` keeps working.
"""

import asyncio
import functools
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from hermes_cli.web_deps import late
from hermes_cli.config import cfg_get
from hermes_cli.web_server_cron import (
    _create_cron_job_sync, _cron_optional_text, _cron_string_list, _mutate_cron_for_profile, _normalize_dashboard_cron_script, _raise_if_cron_registration_error, _run_cron_dashboard_io, _validate_dashboard_cron_context_from, _validate_dashboard_cron_effective_job,
)
from hermes_cli.web_models import AutomationBlueprintInstantiate, CronJobCreate, CronJobUpdate
from hermes_cli.web_routers._common import log as _log
from hermes_time import get_timezone as _get_timezone
from hermes_constants import (
    get_hermes_home as _get_hermes_home,
    reset_hermes_home_override as _reset_hermes_home_override,
    set_hermes_home_override as _set_hermes_home_override,
)

router = APIRouter()

_find_cron_job_profile = late("_find_cron_job_profile", "hermes_cli.web_server_cron")
_fire_cron_job_for_profile = late("_fire_cron_job_for_profile", "hermes_cli.web_server_cron")
_forward_cron_fire_to_gateway = late("_forward_cron_fire_to_gateway", "hermes_cli.web_server_cron")
_gateway_intentionally_stopped = late("_gateway_intentionally_stopped", "hermes_cli.web_server_cron")
_notify_cron_provider_for_profile = late("_notify_cron_provider_for_profile", "hermes_cli.web_server_cron")
_call_cron_for_profile = late("_call_cron_for_profile", "hermes_cli.web_server_cron")
load_config = late("load_config", "hermes_cli.config")
_cron_profile_dicts = late("_cron_profile_dicts", "hermes_cli.web_server_cron")
_cron_profile_home = late("_cron_profile_home", "hermes_cli.web_server_cron")
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")

def _job_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Job not found")


def _normalize_dashboard_cron_updates(updates: Dict[str, Any], profile_home: Path) -> Dict[str, Any]:
    """Normalize dashboard JSON into cron.jobs.update_job's storage shape.

    Stays in the dashboard adapter layer on purpose: cron/jobs.py is the source
    of truth for scheduling; this only translates form payloads into shapes the
    core functions already accept.
    """
    normalized = dict(updates or {})
    for key in ("model", "provider", "workdir"):
        if key in normalized:
            normalized[key] = _cron_optional_text(normalized[key])
    if "script" in normalized:
        normalized["script"] = _normalize_dashboard_cron_script(normalized["script"], profile_home)
    if "base_url" in normalized:
        normalized["base_url"] = _cron_optional_text(normalized["base_url"], strip_trailing_slash=True)
    if "deliver" in normalized:
        normalized["deliver"] = _cron_optional_text(normalized["deliver"]) or "local"
    if "failure_deliver" in normalized:
        # Same normalization as deliver, but empty CLEARS the override (failures
        # fall back to deliver) rather than coalescing — the field is optional.
        normalized["failure_deliver"] = _cron_optional_text(normalized["failure_deliver"])
    for key in ("context_from", "enabled_toolsets"):
        if key in normalized:
            normalized[key] = _cron_string_list(normalized[key])
    return normalized


def _job_owner_profile(job_id: str, profile: Optional[str]) -> Optional[str]:
    """Profile that holds ``job_id`` (its jobs.json and the state.db with its run sessions).

    ``profile`` is a caller *hint*, not proof of ownership: the Desktop lists
    jobs cross-profile (``?profile=all``) while its per-item calls carry the
    ambient active profile. Trusting that hint opened a profile that does not
    hold the job, so the ``cron_{job_id}_*`` id-range scan matched nothing and
    answered ``200 {"runs": []}`` — the UI rendered "No runs yet" for a job
    that had run many times (#115345). A hint that does hold the job still
    wins, so deliberately scoped lookups (the same job id in two profiles,
    e.g. a copied jobs.json) keep reading the named profile.
    """
    if profile:
        jobs = _call_cron_for_profile(profile, "list_jobs", True)
        if any(j.get("id") == job_id or j.get("name") == job_id for j in jobs):
            return profile
    return _find_cron_job_profile(job_id)


def _job_profile(job_id: str, profile: Optional[str]) -> str:
    """Owning profile for the get/update/pause/resume/trigger/delete family; 404 when no profile
    holds the job. Same hint validation as the run lookup, so a wrong-profile hint from the
    cross-profile list cannot 404 (or act on the wrong store for) a job the server can locate."""
    selected = _job_owner_profile(job_id, profile)
    if not selected:
        raise _job_not_found()
    return selected


def _found(job):
    if not job:
        raise _job_not_found()
    return job


def _list_cron_jobs_sync(profile: str = "all"):
    requested = (profile or "all").strip()
    if requested.lower() != "all":
        return _call_cron_for_profile(requested, "list_jobs", True)

    # Aggregating across profiles can surface the SAME job id more than once —
    # e.g. a job copied into a second profile's cron/jobs.json during profile
    # creation. Deduplicate by id, deterministically preferring the default
    # profile's copy over per-iteration order (#51721): collect all jobs first,
    # then resolve duplicates by id with default-profile priority, rather than
    # keeping whichever copy happened to be seen first during the profile loop.
    all_jobs: List[Dict[str, Any]] = []
    for item in _cron_profile_dicts():
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            all_jobs.extend(_call_cron_for_profile(name, "list_jobs", True))
        except Exception:
            _log.exception("Failed to list cron jobs for profile %s", name)

    by_id: Dict[str, Dict[str, Any]] = {}
    unkeyed: List[Dict[str, Any]] = []
    for job in all_jobs:
        if not isinstance(job, dict):
            continue
        jid = job.get("id") or job.get("job_id")
        # cron.jobs._normalize_job_record() fills a missing id with the literal
        # sentinel string "unknown" — which is truthy, so a plain `if not jid`
        # check would NOT catch it and two genuinely different id-less jobs from
        # different profiles would collapse into one under this shared sentinel
        # key. Treat the sentinel the same as no id: never deduplicated.
        if not jid or jid == "unknown":
            unkeyed.append(job)
            continue
        existing = by_id.get(jid)
        if existing is None or (
            job.get("is_default_profile") and not existing.get("is_default_profile")
        ):
            by_id[jid] = job
    return list(by_id.values()) + unkeyed


def _get_cron_job_sync(job_id: str, profile: Optional[str] = None):
    return _found(_call_cron_for_profile(_job_profile(job_id, profile), "get_job", job_id))


_CRON_OUTPUT_FILENAME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def _cron_output_runs_dir(profile: Optional[str], job_id: str) -> Path:
    """Output docs live under the job's home — resolve it even without a hint."""
    if profile:
        try:
            _, profile_home = _cron_profile_home(profile)
        except Exception:
            profile_home = _get_hermes_home()
    else:
        profile_home = _get_hermes_home()
    return Path(profile_home) / "cron" / "output" / job_id


@contextmanager
def _owner_home_scope(profile: Optional[str]):
    """Keep reads in the owner's profile context for the whole run-history build.

    Filename stems, the execution ledger and ``hermes_time``'s configured zone all
    resolve through the *current* ``get_hermes_home()``; without this scope a
    cross-profile listing decodes another profile's docs with the dashboard's
    zone (an hours-off timestamp) and reads the dashboard's executions.db. The
    dashboard's own profile needs no override (its home is already current).
    """
    if not profile:
        yield None
        return
    try:
        _, profile_home = _cron_profile_home(profile)
    except Exception:
        yield None
        return
    if Path(profile_home).resolve() == _get_hermes_home().resolve():
        yield None
        return
    token = _set_hermes_home_override(str(profile_home))
    try:
        yield profile_home
    finally:
        _reset_hermes_home_override(token)


def _cron_output_run_timestamp(path: Path) -> Optional[float]:
    """Epoch seconds for an output filename's wall time.

    save_job_output writes the stem with hermes_time.now() — the configured
    IANA timezone (HERMES_TIMEZONE / config), falling back to server-local.
    Read it back with that same ZONE, not a fixed offset snapshot of today's
    local offset: a snapshot is wrong when the configured zone differs from
    the server's, and off by an hour for a historical file across a DST
    transition. With no configured zone, astimezone() on the naive datetime
    resolves the offset in effect on the filename's own date.
    """
    try:
        naive = datetime.strptime(path.stem, _CRON_OUTPUT_FILENAME_FORMAT)
    except ValueError:
        return None
    tz = _get_timezone()
    if tz is not None:
        return naive.replace(tzinfo=tz).timestamp()
    return naive.astimezone().timestamp()


def _cron_output_run_preview(path: Path, max_chars: int = 180) -> str:
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""
    preview = re.sub(r"\s+", " ", raw).strip()
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 1].rstrip() + "…"


def _cron_job_last_run_timestamp(job: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(job, dict):
        return None
    raw = job.get("last_run_at")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _cron_output_status_label(job: Optional[Dict[str, Any]]) -> str:
    if not isinstance(job, dict):
        return ""
    status = str(job.get("last_status") or "").strip()
    if not status:
        return ""
    return status.replace("_", " ").upper()


def _cron_output_run_row(started_at: float, title: str, preview: Optional[str]) -> Dict[str, Any]:
    return {
        "title": title,
        "preview": preview or None,
        "source": "cron_output",
        "started_at": started_at,
        "last_active": started_at,
        "ended_at": started_at,
        "input_tokens": 0,
        "output_tokens": 0,
        "message_count": 0,
        "tool_call_count": 0,
        "model": None,
        "cwd": None,
        "archived": False,
        "is_active": False,
    }


def _iso_to_epoch(text: Any) -> Optional[float]:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return datetime.fromisoformat(text.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _owner_profile_executions(canonical_job_id: str) -> List[Dict[str, Any]]:
    """Terminal execution-ledger rows for the job, newest first.

    Each script-only fire creates exactly one ledger row (claimed → completed /
    failed), so the ledger — not filename proximity to ``last_run_at`` — is the
    per-attempt record that pairs an output doc with its real status. Read inside
    the owner-home scope so it opens the OWNER's cron/executions.db.
    """
    try:
        from cron.executions import list_executions

        rows = list_executions(job_id=canonical_job_id, limit=100)
    except Exception:
        return []
    terminal: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "") not in ("completed", "failed", "unknown"):
            continue
        terminal.append({
            "claimed_at": _iso_to_epoch(row.get("claimed_at")) or 0.0,
            "finished_at": _iso_to_epoch(row.get("finished_at")),
            "status": str(row.get("status") or ""),
            "error": str(row.get("error") or "").strip(),
        })
    terminal.sort(key=lambda r: r["claimed_at"], reverse=True)
    return terminal


def _execution_contains(
    attempt: Dict[str, Any], started_at: float, grace_seconds: float = 300.0,
) -> bool:
    """Whether an output doc's timestamp falls inside a ledger attempt's window.

    ``save_job_output`` runs after the script finishes but before
    ``finish_execution`` closes the attempt, and both clocks are
    ``hermes_time.now()`` in the owner's zone, so a doc written by an attempt
    lands within ``[claimed_at, finished_at + grace]`` of THAT attempt (a still-
    open attempt has no ``finished_at``). Older attempts closed before this doc
    was written, which is what keeps a fast-firing job's status from bleeding
    onto an earlier run's row.
    """
    if started_at + 0.001 < attempt["claimed_at"]:
        return False
    finish = attempt["finished_at"]
    return finish is None or started_at <= finish + grace_seconds


def _execution_status_title(status: str, error: str, fallback: str) -> str:
    label = (status or "").replace("_", " ").upper()
    if label == "UNKNOWN":
        return fallback
    if not label:
        return fallback
    return f"{label} · {error}" if (label == "FAILED" and error) else f"{label} · {fallback}"


def _list_cron_output_runs(
    job: Optional[Dict[str, Any]],
    canonical_job_id: str,
    profile: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """SessionDB-less run history for jobs that never create agent sessions.

    Script-only (no_agent) jobs deliberately skip SessionDB (cron/scheduler.run_job),
    so their completed runs are invisible to the run-history endpoint. Their
    output docs — one .md per fire under cron/output/<job_id>/ — are the only
    per-run record. Rows mirror /api/sessions shape with source='cron_output'
    so the frontend reuses SessionInfo; ids use a cron_output: prefix that can
    never collide with SessionDB cron_{job_id}_* session ids.

    Status comes from the execution ledger (one row per fire), matched to a doc
    by claim window — never from ``last_run_at`` proximity, which mislabels an
    older doc when the newest run's doc is missing. Terminal attempts with no
    surviving doc get their own metadata rows.
    """
    output_dir = _cron_output_runs_dir(profile, canonical_job_id)
    try:
        files = sorted(
            (path for path in output_dir.glob("*.md") if path.is_file()),
            key=lambda path: path.name,
            reverse=True,
        )
    except OSError:
        files = []

    executions = _owner_profile_executions(canonical_job_id)
    represented: set = set()
    runs: List[Dict[str, Any]] = []

    for path in files[:limit]:
        started_at = _cron_output_run_timestamp(path)
        if started_at is None:
            try:
                started_at = path.stat().st_mtime
            except OSError:
                started_at = 0.0
        preview = _cron_output_run_preview(path)
        title = preview or "Script-only run"
        for index, attempt in enumerate(executions):
            if attempt["status"] not in ("completed", "failed"):
                continue
            if _execution_contains(attempt, started_at):
                represented.add(index)
                title = _execution_status_title(attempt["status"], attempt["error"], title)
                break
        runs.append({
            "id": f"cron_output:{canonical_job_id}:{path.stem}",
            **_cron_output_run_row(started_at, title, preview),
        })

    # Terminal attempts whose output doc is gone (pruned, or the run never
    # wrote one — e.g. a failed fire) are still real executions: give each its
    # own metadata row from the ledger instead of letting the latest attempt's
    # status bleed onto an older surviving document.
    for index, attempt in enumerate(executions):
        if index in represented or attempt["status"] not in ("completed", "failed"):
            continue
        if len(runs) >= limit:
            break
        when = attempt["finished_at"] or attempt["claimed_at"]
        error = attempt["error"]
        runs.append({
            "id": f"cron_output:{canonical_job_id}:exec:{index}",
            **_cron_output_run_row(
                when,
                _execution_status_title(attempt["status"], error, "Script-only run"),
                error or None,
            ),
            # Internal: the attempt's claim window, so the caller can drop this
            # row when a session already represents the same execution.
            "_claim_window": (attempt["claimed_at"], attempt["finished_at"]),
        })
    runs.sort(key=lambda r: float(r.get("started_at") or 0), reverse=True)

    if runs:
        return runs

    # No output docs survived (pruned or never written) but the job HAS run:
    # surface one metadata-only row from last_run_at/last_status instead of the
    # bare "No runs" the issue reports.
    latest_ts = _cron_job_last_run_timestamp(job)
    if latest_ts is None:
        return []

    preview = ""
    if isinstance(job, dict):
        preview = str(job.get("last_error") or "").strip()
    title = _cron_output_status_label(job) or "Script-only run"
    if preview:
        title = f"{title} · {preview}"
    return [{
        "id": f"cron_output:{canonical_job_id}:latest",
        **_cron_output_run_row(latest_ts, title, preview),
    }]


def _list_cron_job_runs_sync(job_id: str, profile: Optional[str] = None, limit: int = 20):
    """Run history for a cron job, newest first: agent sessions PLUS script-only fires.

    Agent runs are ordinary sessions with id ``cron_{job_id}_{timestamp}`` (see
    cron/scheduler.run_job); ``source='cron'`` plus the id prefix binds them to
    this job. Backed by ``SessionDB.list_cron_job_runs`` — a bounded id-range
    scan, so cost scales with the requested window, not total cron history.
    Script-only (no_agent) jobs never write sessions; their per-fire output docs
    (``_list_cron_output_runs``) fill the gaps. The two representations are
    reconciled PER EXECUTION, not by branch: a job can switch modes without
    changing its id (update_job supports no_agent on an existing record), so a
    surviving historical agent session must not hide newer script-only fires
    and a recent session must not hide older script output. Rows carrying the
    same wall-clock execution (an agent fire writes BOTH a session and an
    output doc) are de-duplicated by timestamp window; script-only rows that
    match no session are added in. All owner-scoped reads (SessionDB, output
    docs, execution ledger, filename timezone) run inside the owner's profile
    scope so the dashboard's own zone/store is never borrowed cross-profile.
    """
    selected = _job_owner_profile(job_id, profile)
    # job_id may be a human name; resolve to the canonical id used in run-session ids.
    canonical = job_id
    job = None
    if selected:
        job = _call_cron_for_profile(selected, "get_job", job_id)
        if job and job.get("id"):
            canonical = str(job["id"])

    try:
        limit_n = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit_n = 20

    with _owner_home_scope(selected):
        db = _open_session_db_for_profile(selected, read_only=True)
        try:
            session_runs = db.list_cron_job_runs(canonical, limit=limit_n, offset=0)
        finally:
            db.close()

        now = time.time()
        for s in session_runs:
            s["is_active"] = s.get("ended_at") is None and (now - s.get("last_active", s.get("started_at", 0))) < 300
            s["archived"] = bool(s.get("archived"))
            if selected:
                s["profile"] = selected

        doc_runs = _list_cron_output_runs(job, canonical, selected, limit_n)

    return {"runs": _reconcile_cron_runs(session_runs, doc_runs, limit_n), "limit": limit_n}


def _reconcile_cron_runs(
    session_runs: List[Dict[str, Any]],
    doc_runs: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Merge session rows and output-doc rows per execution, newest first.

    An agent fire writes BOTH a session and an output doc for the same
    execution; keeping both would double-render the run. Sessions carry the
    richer record (tokens, message counts, chat navigation), so a doc row
    within ``_SESSION_DOC_MATCH_SECONDS`` of a session row is the same
    execution and the session wins. Doc rows matching no session are distinct
    executions — script-only fires — and are kept. A session row must never be
    dropped in favor of a doc: the doc is the fallback representation.
    """
    _SESSION_DOC_MATCH_SECONDS = 300.0

    merged = list(session_runs)
    for doc in doc_runs:
        doc = {k: v for k, v in doc.items() if not k.startswith("_")}
        doc_ts = float(doc.get("started_at") or 0)
        if any(_doc_matches_session(doc_ts, s, _SESSION_DOC_MATCH_SECONDS) for s in session_runs):
            continue
        merged.append(doc)
    merged.sort(key=lambda r: float(r.get("started_at") or 0), reverse=True)
    return merged[:limit]


def _doc_matches_session(doc_ts: float, session: Dict[str, Any], grace_seconds: float) -> bool:
    """Whether an output doc belongs to a session's run.

    The doc is written when the run FINISHES, so its filename timestamp sits
    at the run's end — compare against the session's full span, not its start:
    ``[started_at - grace, last_active + grace]`` (``last_active`` covers the
    still-open case where ``ended_at`` is None).
    """
    start = float(session.get("started_at") or 0)
    end = float(session.get("ended_at") or session.get("last_active") or start)
    return start - grace_seconds <= doc_ts <= end + grace_seconds


_EXECUTION_FIELDS = {"prompt", "skill", "skills", "script", "no_agent"}


def _update_cron_job_sync(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    selected = _job_profile(job_id, profile)
    try:
        profile_name, profile_home = _cron_profile_home(selected)
        existing = _found(_call_cron_for_profile(profile_name, "get_job", job_id))
        updates = _normalize_dashboard_cron_updates(body.updates, profile_home)
        if "context_from" in updates:
            _validate_dashboard_cron_context_from(updates.get("context_from"), profile_name)
        if _EXECUTION_FIELDS.intersection(updates):
            effective = {**existing, **updates}
            if "skills" in updates and "skill" not in updates:
                effective["skill"] = None
            _validate_dashboard_cron_effective_job(effective)
        job = _mutate_cron_for_profile(profile_name, "update_job", job_id, updates)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _found(job)


def _pause_cron_job_sync(job_id: str, profile: Optional[str] = None):
    return _found(_mutate_cron_for_profile(_job_profile(job_id, profile), "pause_job", job_id))


def _resume_cron_job_sync(job_id: str, profile: Optional[str] = None):
    return _found(_mutate_cron_for_profile(_job_profile(job_id, profile), "resume_job", job_id))


def _trigger_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = _job_profile(job_id, profile)
    job = _found(_call_cron_for_profile(selected, "resolve_job_ref", job_id))
    # Never expose the job as due before claiming it: the built-in ticker and
    # external/manual fire paths share one durable claim, so only one executes
    # this run even racing across processes. Active jobs keep the legacy call
    # shape; paused jobs need the explicit force flag to resume + claim atomically.
    force = not job.get("enabled", True) or job.get("state") == "paused"
    ran = _fire_cron_job_for_profile(selected, job["id"], force=force)
    refreshed = _call_cron_for_profile(selected, "get_job", job["id"])
    if refreshed and refreshed.get("last_run_at") != job.get("last_run_at"):
        return refreshed
    if not ran:
        raise HTTPException(status_code=409, detail="Job is already running or was claimed by another scheduler")
    if refreshed:
        return refreshed
    # A one-shot may remove itself after exhausting repeat=1: keep the response
    # shape without inventing an outcome the store no longer holds; the list
    # refresh removes the completed row.
    return {**job, "enabled": False, "state": "completed"}


def _delete_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = _job_profile(job_id, profile)
    try:
        removed = _mutate_cron_for_profile(selected, "remove_job", job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise _job_not_found()
    return {"ok": True}


# Retry-After (seconds) on retryable cron-fire 503s: sized to clear a
# scale-to-zero wake or gateway restart so a scheduler that honors it spaces its
# next attempt past the outage instead of burning its retry budget in it.
_CRON_FIRE_RETRY_AFTER_SECONDS = 60


@router.get("/api/cron/jobs")
async def list_cron_jobs(profile: str = "all"):
    return await _run_cron_dashboard_io(_list_cron_jobs_sync, profile)


@router.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_get_cron_job_sync, job_id, profile)


@router.get("/api/cron/jobs/{job_id}/runs")
async def list_cron_job_runs(job_id: str, profile: Optional[str] = None, limit: int = 20):
    return await _run_cron_dashboard_io(_list_cron_job_runs_sync, job_id, profile, limit)


@router.post("/api/cron/jobs")
async def create_cron_job(body: CronJobCreate, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_create_cron_job_sync, body, profile)


@router.get("/api/cron/delivery-targets")
async def get_cron_delivery_targets(profile: Optional[str] = None):
    """Delivery targets for the cron dropdown: implicit ``local`` plus the
    configured gateway platforms (a platform without a cron home channel is
    still listed with ``home_target_set: false`` so the UI can say so).

    ``cron_delivery_targets()`` reads each platform's home channel through
    ``get_secret``, which fails closed once this process hosts more than one
    profile home (the dashboard/desktop ``serve`` backend flips multi-profile
    hosting on the first ``?profile=`` request). The read must therefore run
    inside the profile scope, exactly like the sibling cron routes — otherwise
    the poll raises ``UnscopedSecretError`` on every tick and the dropdown
    silently loses every configured platform."""
    targets = [{"id": "local", "name": "Local (save only)", "home_target_set": True, "home_env_var": None}]
    try:
        from cron.scheduler_delivery import cron_delivery_targets

        with _config_profile_scope(profile):
            targets.extend(cron_delivery_targets())
    except Exception:
        _log.exception("GET /api/cron/delivery-targets failed")
    return {"targets": targets}


@router.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_update_cron_job_sync, job_id, body, profile)


@router.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_pause_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_resume_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_trigger_cron_job_sync, job_id, profile)


@router.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_delete_cron_job_sync, job_id, profile)


@router.post("/api/cron/fire")
async def cron_fire_webhook(request: Request):
    """Chronos managed-cron fire webhook (NAS -> agent) — gateway forwarder.

    Gated by the NAS-minted JWT (path is in ``PUBLIC_API_PATHS``), not the
    dashboard cookie. Execution belongs to the GATEWAY process (it owns the live
    platform adapters relay-fronted and E2EE targets need), so the fire is
    forwarded to the gateway api_server's own ``/api/cron/fire`` on loopback
    and its response passed through (the gateway re-verifies the JWT). Gateway
    unreachable -> 503 so NAS retries; deliberately NO local-execution fallback.
    """
    from plugins.cron_providers.chronos.verify import get_fire_verifier

    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else ""

    cfg = await asyncio.to_thread(load_config)
    claims = get_fire_verifier()(
        token=token,
        expected_audience=cfg_get(cfg, "cron", "chronos", "expected_audience", default=""),
        jwks_or_key=cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="") or None,
        issuer=cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None,
    )
    if claims is None:
        return JSONResponse({"error": "invalid fire token"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    job_id = (body or {}).get("job_id") if isinstance(body, dict) else None
    if not job_id:
        return JSONResponse({"error": "missing job_id"}, status_code=400)

    # Walks every profile's job list (file I/O) — off the event loop.
    profile = await _run_cron_dashboard_io(_find_cron_job_profile, job_id)
    if not profile:  # job is gone (cancelled / completed): 200 so NAS does not retry
        return JSONResponse({"status": "gone", "job_id": job_id}, status_code=200)

    forwarded = await _forward_cron_fire_to_gateway(profile, job_id, auth)
    if forwarded is None:
        # Stamp the miss on the job record (last_fire_error) so the dead hop is
        # visible in `cronjob list` / the dashboard. Best-effort: visibility
        # must never break the retry contract below.
        try:
            await _run_cron_dashboard_io(
                _call_cron_for_profile,
                profile,
                "note_fire_forward_failure",
                job_id,
                "scheduled fire could not be forwarded to the gateway "
                "api_server (127.0.0.1 loopback unreachable); the gateway "
                "process may be down or its api_server adapter not bound "
                "(missing API_SERVER_KEY)",
            )
        except Exception:
            _log.debug("could not stamp last_fire_error for %s", job_id, exc_info=True)
        # Split by operator intent: a deliberately stopped gateway (durable
        # desired_state == "stopped") can never be reached by retrying, so drop
        # with 200 + a structured log line — the Chronos provider re-arms every
        # job on the next gateway start. A transient window (wake, restart,
        # crash loop) keeps the retryable 503 with a Retry-After hint.
        if await _run_cron_dashboard_io(_gateway_intentionally_stopped, profile):
            _log.info(
                "cron fire dropped: gateway for profile %r is deliberately "
                "stopped (desired_state=stopped); job %s will resume via "
                "Chronos reconcile on next gateway start",
                profile, job_id,
            )
            return JSONResponse(
                {
                    "status": "gateway_stopped",
                    "detail": "gateway deliberately stopped; fire dropped, jobs re-arm on next gateway start",
                    "job_id": job_id,
                    "profile": profile,
                },
                status_code=200,
            )
        return JSONResponse(
            {"error": "gateway unreachable; retry", "job_id": job_id, "profile": profile},
            status_code=503,
            headers={"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)},
        )
    status_code, gateway_body = forwarded
    if isinstance(gateway_body, dict):
        gateway_body.setdefault("job_id", job_id)
    # The gateway's own 503s (draining, admission failure) are equally transient.
    headers = {"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)} if status_code == 503 else None
    return JSONResponse(gateway_body, status_code=status_code, headers=headers)


@router.get("/api/cron/blueprints")
async def list_cron_blueprints(profile: Optional[str] = None):
    """Blueprint catalog as form schemas; the ``deliver`` slot's options are
    rewritten from the actually configured gateway platforms."""
    try:
        from cron.blueprint_catalog import CATALOG, blueprint_catalog_entry

        deliver_options = None
        try:
            from cron.scheduler_delivery import cron_delivery_targets

            with _config_profile_scope(profile):
                platforms = [t["id"] for t in cron_delivery_targets() if t.get("id")]
            deliver_options = ["origin", "local", *platforms]
        except Exception:
            _log.debug("cron_delivery_targets unavailable; using static deliver options", exc_info=True)

        entries = []
        for r in CATALOG:
            entry = blueprint_catalog_entry(r)
            if deliver_options:
                for f in entry.get("fields", []):
                    if f.get("name") == "deliver":
                        f["options"] = deliver_options
            entries.append(entry)
        return {"blueprints": entries}
    except Exception as e:
        _log.exception("GET /api/cron/blueprints failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cron/blueprints/instantiate")
async def instantiate_blueprint(body: AutomationBlueprintInstantiate, profile: str = "default"):
    """Fill a blueprint's slots and create the cron job (form-submit path)."""
    try:
        from cron.blueprint_catalog import BlueprintFillError, fill_blueprint, get_blueprint

        blueprint = get_blueprint(body.blueprint)
        if blueprint is None:
            raise HTTPException(status_code=404, detail=f"Unknown blueprint: {body.blueprint}")
        try:
            spec = fill_blueprint(blueprint, body.values)
        except BlueprintFillError as exc:  # field-level error — 422 so the form shows it inline
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Blueprint jobs deliver to the dashboard's configured target by default;
        # the form's deliver slot overrides via spec["deliver"].
        spec.pop("origin", None)
        # Off-loop like the siblings; partial keeps **spec keys from colliding
        # with the wrapper's own parameters.
        _create = functools.partial(_call_cron_for_profile, profile, "create_job", **spec)
        created = await _run_cron_dashboard_io(_create)
        # Reconcile the profile-scoped provider (file I/O + NAS calls) off-loop.
        await _run_cron_dashboard_io(_notify_cron_provider_for_profile, profile)
        return created
    except HTTPException:
        raise
    except Exception as e:
        _raise_if_cron_registration_error(e)
        _log.exception("POST /api/cron/blueprints/instantiate failed")
        raise HTTPException(status_code=400, detail=str(e))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
