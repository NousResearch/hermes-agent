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
        return {"mode": service.manager.config.default_mode, "realms": []}
    identity = dict(
        runtime_session_id=runtime_session_id, stored_session_id=stored_session_id
    )
    owner = resolve_owner(service, identity, allow_missing=True)
    if owner is None:
        # Historical sessions may predate trusted ownership registration.
        # A read must neither bind aliases nor inspect another owner's realms.
        return {"mode": service.manager.config.default_mode, "realms": []}
    result = service.status(owner)
    for row in result["realms"]:
        row.update(identity)
    return result


@router.post("/realms/{realm_id}/watch")
def watch_realm(realm_id: str, identity: SessionOwner, request: Request):
    service = get_integration()
    owner = resolve_owner(service, identity.model_dump())
    record = next((r for r in service.manager.list() if r["id"] == realm_id), None)
    if record is None:
        raise HTTPException(404, "Realm is stopped or missing")
    if record["session_id"] != owner:
        raise HTTPException(403, "Realm ownership mismatch")
    if record["status"] != "running":
        raise HTTPException(409, "Realm is not ready")
    from fastapi.responses import JSONResponse
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
    try:
        return JSONResponse(
            service.watch(owner, realm_id),
            headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        )
    except Exception:
        raise HTTPException(
            503, "Viewer bridge unavailable; verify the realm is healthy"
        ) from None
