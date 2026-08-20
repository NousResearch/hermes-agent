"""Cron dashboard routes (extracted verbatim from web_server.py).

Handler bodies are byte-identical.  The ``*_sync`` workers, profile resolution
and the threadpool wrapper (``_run_cron_dashboard_io``) still live in
web_server — reached via the late-binding seam in :mod:`hermes_cli.web_deps`
so ``monkeypatch.setattr(web_server, ...)`` keeps working (several cron tests
rely on exactly that).
"""

import asyncio  # noqa: F401 — used by handlers
import functools  # noqa: F401
import logging
from typing import Optional  # noqa: F401

from fastapi import APIRouter, HTTPException, Request  # noqa: F401
from fastapi.responses import JSONResponse  # noqa: F401

from hermes_cli.web_deps import late
from hermes_cli.web_models import (
    CronJobCreate,
    CronJobUpdate,
    AutomationBlueprintInstantiate,
)

# Same logger the handlers used before extraction (identical logger object).
_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent — includes config readers so existing
# ``monkeypatch.setattr(web_server, "load_config", ...)`` idioms behave
# identically for these routes).
_run_cron_dashboard_io = late("_run_cron_dashboard_io")
_list_cron_jobs_sync = late("_list_cron_jobs_sync")
_get_cron_job_sync = late("_get_cron_job_sync")
_list_cron_job_runs_sync = late("_list_cron_job_runs_sync")
_create_cron_job_sync = late("_create_cron_job_sync")
_update_cron_job_sync = late("_update_cron_job_sync")
_pause_cron_job_sync = late("_pause_cron_job_sync")
_resume_cron_job_sync = late("_resume_cron_job_sync")
_trigger_cron_job_sync = late("_trigger_cron_job_sync")
_delete_cron_job_sync = late("_delete_cron_job_sync")
_find_cron_job_profile = late("_find_cron_job_profile")
_fire_cron_job_for_profile = late("_fire_cron_job_for_profile")
_forward_cron_fire_to_gateway = late("_forward_cron_fire_to_gateway")
_gateway_intentionally_stopped = late("_gateway_intentionally_stopped")
_notify_cron_provider_for_profile = late("_notify_cron_provider_for_profile")
_call_cron_for_profile = late("_call_cron_for_profile")
_raise_if_cron_registration_error = late("_raise_if_cron_registration_error")
_require_dashboard_admin = late("_require_dashboard_admin")
load_config = late("load_config")
cfg_get = late("cfg_get")

# Retry-After hint (seconds) stamped on retryable cron-fire 503s. Sized to
# clear the common transient windows — a scale-to-zero wake or an s6 gateway
# restart completes well within a minute — so a scheduler that honors it
# spaces its next attempt PAST the outage window instead of burning its whole
# retry budget inside it (OOF-266). Honored by QStash once NAS propagates it;
# harmless (ignored) until then.
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
async def get_cron_delivery_targets():
    """Delivery targets the cron dropdown should offer.

    Always includes the implicit ``local`` option. Beyond that, the list is
    derived dynamically from the configured gateway platforms via
    ``cron.scheduler.cron_delivery_targets()`` — no hardcoded platform list. A
    configured platform that hasn't set its cron home channel is still returned
    with ``home_target_set: false`` so the UI can surface it as "configure a
    home channel first" rather than hiding it.
    """
    targets = [
        {
            "id": "local",
            "name": "Local (save only)",
            "home_target_set": True,
            "home_env_var": None,
        }
    ]
    try:
        from cron.scheduler import cron_delivery_targets

        targets.extend(cron_delivery_targets())
    except Exception:
        _log.exception("GET /api/cron/delivery-targets failed")
    return {"targets": targets}


@router.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    return await _run_cron_dashboard_io(_update_cron_job_sync, job_id, body, profile)


@router.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(request: Request, job_id: str, profile: Optional[str] = None):
    # Mini App token route (required=False): a non-admin paired caller must
    # never reach this. Cookie/desktop and admin-tier Mini App both pass
    # unconditionally, matching this endpoint's existing behavior for the
    # desktop dashboard operator.
    _require_dashboard_admin(request)
    return await _run_cron_dashboard_io(_pause_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(request: Request, job_id: str, profile: Optional[str] = None):
    _require_dashboard_admin(request)
    return await _run_cron_dashboard_io(_resume_cron_job_sync, job_id, profile)


@router.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(request: Request, job_id: str, profile: Optional[str] = None):
    _require_dashboard_admin(request)
    return await _run_cron_dashboard_io(_trigger_cron_job_sync, job_id, profile)


@router.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(request: Request, job_id: str, profile: Optional[str] = None):
    # Mini App token route (required=False), admin-tier only -- same gate as
    # pause/resume/trigger above. This endpoint predates the Mini App and had
    # no per-handler check at all (implicitly desktop-only, gated purely by
    # the cookie/session auth wrapper around every /api/* route); adding the
    # explicit check here is a no-op for the desktop operator (scope is None)
    # and is what makes it safe to register as a Mini App token route below.
    _require_dashboard_admin(request)
    return await _run_cron_dashboard_io(_delete_cron_job_sync, job_id, profile)


@router.post("/api/cron/fire")
async def cron_fire_webhook(request: Request):
    """Chronos managed-cron fire webhook (NAS -> agent) — gateway forwarder.

    Authenticated by a short-lived NAS-minted JWT (verified by the pluggable
    Chronos fire-verifier), NOT the dashboard session cookie — so this path is
    in ``PUBLIC_API_PATHS`` to bypass the dashboard auth gate, and the JWT is
    the real gate.

    The dashboard is only the PUBLIC DOOR here (on hosted deployments the Fly
    proxy exposes exactly one port, the dashboard's). Cron execution belongs
    to the GATEWAY process, which owns the live platform adapters — required
    for relay-fronted logical platforms (their only sender is the live relay
    adapter) and E2EE rooms, neither of which the dashboard's standalone send
    path can serve. So after verifying the JWT this handler FORWARDS the fire
    to the gateway api_server's own ``/api/cron/fire`` on loopback and passes
    the gateway's response through (the gateway re-verifies the JWT — defense
    in depth, no new trust link).

    Gateway unreachable (scale-to-zero wake still booting, restart window,
    api_server disabled) → 503, so NAS retries per the Chronos contract
    (non-2xx = retryable). The store CAS claim de-dupes the eventual double
    fire. Deliberately NO local-execution fallback: delivering from the wrong
    process is worse than a delayed retry.
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

    # _find_cron_job_profile walks every profile and lists its jobs (file
    # I/O per profile) — run it off the event loop like the other cron
    # dashboard endpoints.
    profile = await _run_cron_dashboard_io(_find_cron_job_profile, job_id)
    if not profile:
        # Job is gone (cancelled / completed) — nothing to fire. 200 so NAS
        # does not retry a fire that is intentionally absent.
        return JSONResponse({"status": "gone", "job_id": job_id}, status_code=200)

    forwarded = await _forward_cron_fire_to_gateway(profile, job_id, auth)
    if forwarded is None:
        # Durably stamp the miss on the job record (last_fire_error) so the
        # user and their agent can see "scheduled fire could not reach the
        # runner" in `cronjob list` / the dashboard — without this, a dead
        # 8642 hop is invisible outside gui.log (no execution row is ever
        # created because the claim never happens). Best-effort: visibility
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
        # Gateway unreachable. Split by OPERATOR INTENT (OOF-266):
        #
        # - Deliberately stopped gateway (durable desired_state == "stopped",
        #   written only by the s6 lifecycle commands): retrying can never
        #   succeed until a human starts the gateway again, so the retry
        #   budget is pure waste and the resulting 503→502 storms page the
        #   NAS on-call for a non-incident. Drop with 200 + a structured log
        #   line, mirroring NAS's own instance_stopped drop in the relay.
        #   The fire is NOT lost silently: the log names job and profile,
        #   and the gateway's Chronos provider reconciles + re-arms every
        #   job on its next startup (plugins/cron_providers/chronos
        #   start() -> reconcile()), so fires resume when the operator
        #   starts the gateway.
        #
        # - Transient window (scale-to-zero wake, restart, crash loop —
        #   desired_state is "running" or unknown): keep the retryable 503,
        #   but stamp Retry-After so a scheduler that honors it spaces its
        #   next attempt past the wake/restart window instead of exhausting
        #   the whole retry budget inside it.
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
                    "detail": "gateway deliberately stopped; fire dropped, "
                              "jobs re-arm on next gateway start",
                    "job_id": job_id,
                    "profile": profile,
                },
                status_code=200,
            )
        return JSONResponse(
            {
                "error": "gateway unreachable; retry",
                "job_id": job_id,
                "profile": profile,
            },
            status_code=503,
            headers={"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)},
        )
    status_code, gateway_body = forwarded
    if isinstance(gateway_body, dict):
        gateway_body.setdefault("job_id", job_id)
    headers = (
        # The gateway's own 503s (draining, admission failure) are equally
        # transient — give the scheduler the same spacing hint.
        {"Retry-After": str(_CRON_FIRE_RETRY_AFTER_SECONDS)}
        if status_code == 503
        else None
    )
    return JSONResponse(gateway_body, status_code=status_code, headers=headers)


@router.get("/api/cron/blueprints")
async def list_cron_blueprints():
    """Return the blueprint catalog as form schemas for the dashboard gallery.

    The ``deliver`` slot's options are rewritten from the user's actually
    configured gateway platforms (plus the universal origin/local/all), so the
    form never offers a platform that isn't connected.
    """
    try:
        from cron.blueprint_catalog import CATALOG, blueprint_catalog_entry

        deliver_options = None
        try:
            from cron.scheduler import cron_delivery_targets

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
        from cron.blueprint_catalog import fill_blueprint, get_blueprint, BlueprintFillError

        blueprint = get_blueprint(body.blueprint)
        if blueprint is None:
            raise HTTPException(status_code=404, detail=f"Unknown blueprint: {body.blueprint}")
        try:
            spec = fill_blueprint(blueprint, body.values)
        except BlueprintFillError as exc:
            # Field-level validation error — 422 so the form can show it inline.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Blueprint-created jobs deliver to the dashboard's configured target by
        # default; the form's deliver slot overrides via spec["deliver"].
        spec.pop("origin", None)
        # create_job does per-profile file I/O — keep it off the event loop
        # like the sibling cron endpoints (partial avoids **spec keys ever
        # colliding with the wrapper's own parameters).
        _create = functools.partial(_call_cron_for_profile, profile, "create_job", **spec)
        created = await _run_cron_dashboard_io(_create)
        # Same contract as the other dashboard mutations: reconcile the
        # profile-scoped provider (best-effort; fail-closed for external
        # providers on a multi-profile dashboard). Off the event loop —
        # a Chronos reconcile does file I/O plus NAS network calls.
        await _run_cron_dashboard_io(_notify_cron_provider_for_profile, profile)
        return created
    except HTTPException:
        raise
    except Exception as e:
        _raise_if_cron_registration_error(e)
        _log.exception("POST /api/cron/blueprints/instantiate failed")
        raise HTTPException(status_code=400, detail=str(e))
