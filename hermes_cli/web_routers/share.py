"""Owner-side controls for sharing this backend over tailcat.

Mounted on the main listener only: the share listener's gate refuses every
``/api/share/*`` path except pairing, so a paired device can never mint codes,
list, or revoke devices.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from hermes_cli import tailcat_share_store as store
from hermes_cli import web_server_share_runtime as runtime
from hermes_cli.tailcat_share import TailcatUnavailable
from hermes_cli.web_deps import late

router = APIRouter()
_require_token = late("_require_token")


def _status_body() -> dict:
    return {
        **runtime.status(),
        "configured": runtime.configured_mode(),
        "devices": [store.public_device(d) for d in store.load_devices()],
    }


@router.get("/api/share/status")
async def share_status(request: Request):
    _require_token(request)
    return _status_body()


@router.post("/api/share/start")
async def share_start(request: Request):
    _require_token(request)
    try:
        await runtime.start(install=True)
    except TailcatUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_body()


@router.post("/api/share/stop")
async def share_stop(request: Request):
    _require_token(request)
    await runtime.stop()
    return _status_body()


@router.post("/api/share/code")
async def share_code(request: Request):
    _require_token(request)
    current = runtime.status()
    if current["state"] != "ready" or not current["address"]:
        raise HTTPException(status_code=409, detail="Sharing is not ready yet.")
    code = store.mint_code(current["address"], current["port"])
    return {"code": code.render(), "expires_in": store.CODE_TTL_S}


@router.delete("/api/share/devices/{device_id}")
async def share_revoke(device_id: str, request: Request):
    _require_token(request)
    if not store.revoke_device(device_id):
        raise HTTPException(status_code=404, detail="No such device.")
    return _status_body()
