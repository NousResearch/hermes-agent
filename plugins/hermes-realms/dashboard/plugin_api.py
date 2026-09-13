"""Mounted by Hermes under its mandatory session-token/OAuth middleware.

Profile is the authenticated routed backend's get_hermes_home(), never a body
field or a mutable focused-session variable. Do not serve this router alone.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from pathlib import Path
import runpy

_load_runtime = runpy.run_path(str(Path(__file__).resolve().parents[1] / "realms/_binding.py"))["load_runtime"]
_integration = _load_runtime("integration")
OwnerError = _integration.OwnerError
get_integration = _integration.get_integration

router = APIRouter()


class SessionOwner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_session_id: str | None = None
    stored_session_id: str | None = None


def resolve_owner(service, identity, *, allow_missing=False):
    try:
        if any(value == "" for value in identity.values()):
            raise OwnerError("Invalid session identity")
        return service.owners.resolve(allow_missing=allow_missing, **identity)
    except OwnerError:
        raise HTTPException(403, "Session ownership mismatch") from None


@router.get("/realms")
def list_realms(
    runtime_session_id: str | None = None, stored_session_id: str | None = None
):
    service = get_integration()
    if runtime_session_id is None and stored_session_id is None:
        return {"mode": service.manager.config.default_mode, "realms": [],
                "setup": _integration.setup_status(driver_executable=service.driver_executable)}
    identity = dict(
        runtime_session_id=runtime_session_id, stored_session_id=stored_session_id
    )
    owner = resolve_owner(service, identity, allow_missing=True)
    if owner is None:
        # Historical sessions may predate trusted ownership registration.
        # A read must neither bind aliases nor inspect another owner's realms.
        return {"mode": service.manager.config.default_mode, "realms": [],
                "setup": _integration.setup_status(driver_executable=service.driver_executable)}
    result = service.status(owner)
    result["setup_job"] = _load_runtime("setup_flow").latest(service, owner)
    for row in result["realms"]:
        row.update(identity)
    return result


class SetupRequest(SessionOwner):
    kind: str


class SetupStart(SetupRequest):
    consent: str


def _setup_call(operation):
    try:
        return operation()
    except (PermissionError, OwnerError):
        raise HTTPException(403, "Setup ownership mismatch") from None
    except FileNotFoundError:
        raise HTTPException(404, "Setup job not found") from None
    except ValueError:
        raise HTTPException(409, "Setup proposal changed, prerequisites are blocked, or a job is active. Prepare again.") from None
    except OSError:
        raise HTTPException(503, "Profile setup storage is unavailable") from None


@router.post("/realms/setup/prepare")
def prepare_setup(request: SetupRequest):
    service = _setup_call(lambda: _load_runtime("setup_plan").proposal_service())
    owner = resolve_owner(service, request.model_dump(exclude={"kind"}))
    return _setup_call(lambda: _load_runtime("setup_flow").prepare(service, owner, request.kind))


@router.post("/realms/setup/start")
def start_setup(request: SetupStart):
    service = get_integration()
    identity = request.model_dump(exclude={"kind", "consent"})
    owner = resolve_owner(service, identity)
    return _setup_call(lambda: _load_runtime("setup_flow").start(service, owner, request.kind, request.consent, identity))


@router.get("/realms/setup/jobs/{job_id}")
def setup_job(job_id: str, runtime_session_id: str | None = None, stored_session_id: str | None = None):
    service = get_integration()
    owner = resolve_owner(service, dict(runtime_session_id=runtime_session_id, stored_session_id=stored_session_id))
    return _setup_call(lambda: _load_runtime("setup_flow").status(service, owner, job_id))


@router.get("/realms/vm/settings")
def vm_settings(check_updates: bool = False):
    """The Desktop → Plugins → Realms "Omarchy VM" block.

    ``check_updates`` is opt-in: the panel must render with no network, and an
    unreachable GitHub reports "unknown" rather than a fabricated verdict.
    """
    service = get_integration()
    return {
        **service.vm.settings(check_updates=check_updates),
        "setup": _integration.vm_setup_status(service.home),
    }


@router.post("/realms/vm/clean")
def vm_clean():
    """Drop stale ISOs and orphaned session disks; never touches a live realm."""
    service = get_integration()
    _cli = _load_runtime("cli")
    return {**_cli.clean(service.vm), "storage": service.vm.storage()}


def require_owned_viewer(service, owner, realm_id, request):
    record = next((r for r in service.manager.list() if r["id"] == realm_id), None)
    if record is None and realm_id.startswith("v-"):
        record = next((r for r in service.vm.list() if r["id"] == realm_id), None)
    if record is None:
        raise HTTPException(404, "Realm is stopped or missing")
    if record["session_id"] != owner:
        raise HTTPException(403, "Realm ownership mismatch")
    if record["status"] != "running":
        raise HTTPException(409, "Realm is not ready")
    import ipaddress

    try:
        local = (
            request.client is not None
            and ipaddress.ip_address(request.client.host).is_loopback
            and request.url.hostname in ("127.0.0.1", "localhost", "::1")
            and not any(
                key in request.headers
                for key in ("forwarded", "x-forwarded-for", "x-forwarded-host")
            )
        )
    except ValueError:
        local = False
    if not local:
        raise HTTPException(
            503,
            "Remote viewing needs an explicit viewer tunnel; use Watch on the backend host",
        )


@router.post("/realms/{realm_id}/watch")
def watch_realm(realm_id: str, identity: SessionOwner, request: Request):
    from fastapi.responses import JSONResponse

    service = get_integration()
    owner = resolve_owner(service, identity.model_dump())
    require_owned_viewer(service, owner, realm_id, request)
    try:
        return JSONResponse(
            service.watch(owner, realm_id),
            headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        )
    except Exception:
        raise HTTPException(
            503, "Viewer bridge unavailable; verify the realm is healthy"
        ) from None


class ViewerRenewal(SessionOwner):
    viewer_token: str


@router.post("/realms/{realm_id}/renew")
def renew_viewer(realm_id: str, identity: ViewerRenewal, request: Request):
    from fastapi.responses import JSONResponse

    service = get_integration()
    owner = resolve_owner(service, identity.model_dump(exclude={"viewer_token"}))
    require_owned_viewer(service, owner, realm_id, request)
    viewer = _load_runtime("bridge").get_profile_viewer(service.home)
    if not viewer.renew(realm_id, identity.viewer_token, ttl=300):
        raise HTTPException(403, "Viewer authorization expired or was revoked")
    return JSONResponse({"renewed": True}, headers={"Cache-Control": "no-store"})
