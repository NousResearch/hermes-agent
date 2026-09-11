"""Cron dashboard routes.

The ``*_sync`` workers, profile resolution and the threadpool wrapper
(``_run_cron_dashboard_io``) live in web_server_cron and are reached through the
late-binding seam so ``monkeypatch.setattr(web_server_cron, ...)`` keeps working.
"""

import asyncio
import functools
import math
import re
import time
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
_public_cron_job = late("_public_cron_job", "hermes_cli.web_server_cron")
_public_cron_job_for_profile = late("_public_cron_job_for_profile", "hermes_cli.web_server_cron")
_public_cron_job_detail = late("_public_cron_job_detail", "hermes_cli.web_server_cron")
_require_concrete_cron_profile = late("_require_concrete_cron_profile", "hermes_cli.web_server_cron")

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


def _job_profile(job_id: str, profile: Optional[str]) -> str:
    """Profile owning ``job_id`` (explicit or discovered); 404 when none."""
    selected = profile or _find_cron_job_profile(job_id)
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
        return [_public_cron_job_for_profile(job, requested)
                for job in _call_cron_for_profile(requested, "list_jobs", True)]

    jobs: List[Dict[str, Any]] = []
    for item in _cron_profile_dicts():
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            jobs.extend(_public_cron_job_for_profile(job, name)
                        for job in _call_cron_for_profile(name, "list_jobs", True))
        except Exception:
            _log.exception("Failed to list cron jobs for profile %s", name)
    return jobs


def _get_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = _job_profile(job_id, profile)
    return _public_cron_job_for_profile(
        _found(_call_cron_for_profile(selected, "get_job", job_id)), selected)


def _get_cron_job_detail_sync(job_id: str, profile: str):
    selected = (profile or "").strip()
    if not selected or selected.lower() == "all":
        raise HTTPException(status_code=400, detail="cron_detail_profile_required")
    return _public_cron_job_detail(
        _found(_call_cron_for_profile(selected, "get_job", job_id)))


_PUBLIC_CRON_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _public_cron_run_time(value: Any) -> Optional[float]:
    if type(value) not in {int, float}:
        return None
    try:
        parsed = float(value)
    except OverflowError:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _public_cron_run(run: Any, *, now: float) -> Optional[Dict[str, Any]]:
    """Restore the bounded run projection; unknown session fields stay private."""
    if type(run) is not dict:
        return None
    run_id = run.get("id")
    if type(run_id) is not str or not _PUBLIC_CRON_RUN_ID_RE.fullmatch(run_id):
        return None
    ended_raw = run.get("ended_at")
    last_active = _public_cron_run_time(run.get("last_active"))
    reason = run.get("end_reason")
    statuses = {"completed": "completed", "success": "completed", "agent_close": "completed",
                "failed": "failed", "error": "failed", "cancelled": "cancelled",
                "interrupted": "cancelled", "timeout": "timeout"}
    status = "running" if ended_raw is None else statuses.get(reason, "ended") if type(reason) is str else "ended"
    archived = run.get("archived")
    return {
        "id": run_id, "status": status,
        "started_at": _public_cron_run_time(run.get("started_at")),
        "ended_at": _public_cron_run_time(ended_raw), "last_active": last_active,
        "is_active": ended_raw is None and last_active is not None and math.isfinite(now)
                     and 0 <= now - last_active < 300,
        "archived": archived is True or (type(archived) is int and archived == 1),
    }


def _list_cron_job_runs_sync(job_id: str, profile: Optional[str] = None, limit: int = 20):
    """Run sessions produced by a cron job, newest first.

    Runs are ordinary sessions with id ``cron_{job_id}_{timestamp}`` (see
    cron/scheduler.run_job); ``source='cron'`` plus the id prefix binds them to
    this job. Session rows are projected onto bounded public run metadata;
    prompts, previews and unknown fields remain private.
    Backed by ``SessionDB.list_cron_job_runs`` — a bounded id-range
    scan, so cost scales with the requested window, not total cron history.
    """
    selected = profile or _find_cron_job_profile(job_id)
    # job_id may be a human name; resolve to the canonical id used in run-session ids.
    canonical = job_id
    if selected:
        job = _call_cron_for_profile(selected, "get_job", job_id)
        if job and job.get("id"):
            canonical = str(job["id"])

    try:
        limit_n = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit_n = 20

    db = _open_session_db_for_profile(selected, read_only=True)
    try:
        rows = db.list_cron_job_runs(canonical, limit=limit_n, offset=0)
        now = time.time()
        runs = [public for row in rows if (public := _public_cron_run(row, now=now)) is not None]
        return {"runs": runs, "limit": limit_n}
    finally:
        db.close()


_EXECUTION_FIELDS = {"prompt", "skill", "skills", "script", "no_agent"}


def _update_cron_job_sync(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    selected = _require_concrete_cron_profile(profile)
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
    return _public_cron_job(_found(job))


def _pause_cron_job_sync(job_id: str, profile: Optional[str] = None):
    return _public_cron_job(_found(_mutate_cron_for_profile(
        _require_concrete_cron_profile(profile), "pause_job", job_id)))


def _resume_cron_job_sync(job_id: str, profile: Optional[str] = None):
    return _public_cron_job(_found(_mutate_cron_for_profile(
        _require_concrete_cron_profile(profile), "resume_job", job_id)))


def _trigger_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = _require_concrete_cron_profile(profile)
    job = _found(_call_cron_for_profile(selected, "resolve_job_ref", job_id))
    # Never expose the job as due before claiming it: the built-in ticker and
    # external/manual fire paths share one durable claim, so only one executes
    # this run even racing across processes. Active jobs keep the legacy call
    # shape; paused jobs need the explicit force flag to resume + claim atomically.
    force = not job.get("enabled", True) or job.get("state") == "paused"
    ran = _fire_cron_job_for_profile(selected, job["id"], force=force)
    refreshed = _call_cron_for_profile(selected, "get_job", job["id"])
    if refreshed and refreshed.get("last_run_at") != job.get("last_run_at"):
        return _public_cron_job(refreshed)
    if not ran:
        raise HTTPException(status_code=409, detail="Job is already running or was claimed by another scheduler")
    if refreshed:
        return _public_cron_job(refreshed)
    # A one-shot may remove itself after exhausting repeat=1: keep the response
    # shape without inventing an outcome the store no longer holds; the list
    # refresh removes the completed row.
    return _public_cron_job({**job, "enabled": False, "state": "completed"})


def _delete_cron_job_sync(job_id: str, profile: Optional[str] = None):
    selected = _require_concrete_cron_profile(profile)
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


def _public_gateway_fire_body(status_code: int, gateway_body: object, job_id: str) -> dict:
    """Receipt-only fire response; gateway diagnostics and profile routing stay private."""
    body = gateway_body if type(gateway_body) is dict else {}
    status = body.get("status")
    if 200 <= status_code < 300 and type(status) is str and status in {"accepted", "duplicate"}:
        return {"status": status, "job_id": job_id}
    error_kind = (
        "invalid_request" if status_code == 400
        else "authentication_failed" if status_code in {401, 403}
        else "claim_conflict" if status_code == 409
        else "gateway_unavailable" if status_code == 503
        else "gateway_fire_failed"
    )
    return {"error": error_kind, "error_kind": error_kind, "job_id": job_id}


@router.get("/api/cron/jobs")
async def list_cron_jobs(profile: str = "all"):
    return await _run_cron_dashboard_io(_list_cron_jobs_sync, profile)


@router.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_get_cron_job_sync, job_id, profile)


@router.get("/api/cron/jobs/{job_id}/detail")
async def get_cron_job_detail(job_id: str, profile: str):
    return await _run_cron_dashboard_io(_get_cron_job_detail_sync, job_id, profile)


@router.get("/api/cron/jobs/{job_id}/runs")
async def list_cron_job_runs(job_id: str, profile: Optional[str] = None, limit: int = 20):
    return await _run_cron_dashboard_io(_list_cron_job_runs_sync, job_id, profile, limit)


@router.post("/api/cron/jobs")
async def create_cron_job(body: CronJobCreate, profile: str):
    return await _run_cron_dashboard_io(_create_cron_job_sync, body, profile)


@router.get("/api/cron/delivery-targets")
async def get_cron_delivery_targets():
    """Delivery targets for the cron dropdown: implicit ``local`` plus the
    configured gateway platforms (a platform without a cron home channel is
    still listed with ``home_target_set: false`` so the UI can say so)."""
    targets = [{"id": "local", "name": "Local (save only)", "home_target_set": True, "home_env_var": None}]
    try:
        from cron.scheduler_delivery import cron_delivery_targets

        targets.extend(cron_delivery_targets())
    except Exception:
        _log.exception("GET /api/cron/delivery-targets failed")
    return {"targets": targets}


@router.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate, profile: str):
    return await _run_cron_dashboard_io(_update_cron_job_sync, job_id, body, profile)


@router.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(job_id: str, profile: str):
    return await _run_cron_dashboard_io(_pause_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(job_id: str, profile: str):
    return await _run_cron_dashboard_io(_resume_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(job_id: str, profile: str):
    return await _run_cron_dashboard_io(_trigger_cron_job_sync, job_id, profile)


@router.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str, profile: str):
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
                    "job_id": job_id,
                },
                status_code=200,
            )
        return JSONResponse(
            {"error": "gateway_unavailable", "error_kind": "gateway_unavailable", "job_id": job_id},
            status_code=503,
            headers={"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)},
        )
    status_code, gateway_body = forwarded
    # The gateway's own 503s (draining, admission failure) are equally transient.
    headers = {"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)} if status_code == 503 else None
    return JSONResponse(
        _public_gateway_fire_body(status_code, gateway_body, job_id),
        status_code=status_code,
        headers=headers,
    )


@router.get("/api/cron/blueprints")
async def list_cron_blueprints():
    """Blueprint catalog as form schemas; the ``deliver`` slot's options are
    rewritten from the actually configured gateway platforms."""
    try:
        from cron.blueprint_catalog import CATALOG, blueprint_catalog_entry

        deliver_options = None
        try:
            from cron.scheduler_delivery import cron_delivery_targets

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
    except Exception:
        _log.exception("GET /api/cron/blueprints failed")
        raise HTTPException(status_code=500, detail="cron_blueprint_list_failed")


@router.post("/api/cron/blueprints/instantiate")
async def instantiate_blueprint(body: AutomationBlueprintInstantiate, profile: str):
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
        selected = _require_concrete_cron_profile(profile)
        _create = functools.partial(_call_cron_for_profile, selected, "create_job", **spec)
        created = await _run_cron_dashboard_io(_create)
        # Reconcile the profile-scoped provider (file I/O + NAS calls) off-loop.
        await _run_cron_dashboard_io(_notify_cron_provider_for_profile, selected)
        return _public_cron_job_for_profile(created, selected)
    except HTTPException:
        raise
    except Exception as e:
        _raise_if_cron_registration_error(e)
        _log.exception("POST /api/cron/blueprints/instantiate failed")
        raise HTTPException(status_code=400, detail="cron_create_failed") from e


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----