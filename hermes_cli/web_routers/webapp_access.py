"""Web Access routes: Desktop's Web Access pane turns the gateway-served Webapp on and off here.

Thin HTTP over :mod:`hermes_cli.webapp_access`. Every handler acts on the home of the gateway that
serves the requested profile; start and stop write its ``webapp:`` config and restart that gateway,
which reads the config only when it starts. Only the machine's owner reaches them
(:func:`owner_on_this_machine`), never a browser signed in over the network.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from hermes_cli import webapp_access
from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import (
    _profile_scope,
    config_scoped_to_thread,
    config_write_scope,
    destructive_profile,
)

_restart_gateway_after = late("_restart_gateway_after", "hermes_cli.web_server_gateway")


def owner_on_this_machine(request: Request) -> None:
    """Web Access decides what this machine exposes, so only its owner here may use it: the native
    app (a same-user gateway ticket, gated or not) or, on a listener without sign-in, the loopback
    session token that ``auth_middleware`` has already checked. Behind the sign-in gate any other
    caller is a cookie or bearer login, which may come from another device."""
    if getattr(request.state, "native_http_principal", None):
        return
    if getattr(request.app.state, "auth_required", False):
        raise HTTPException(status_code=403, detail="Web Access can only be changed from the Hermes app on this computer.")


router = APIRouter(dependencies=[Depends(owner_on_this_machine)])


class StartBody(BaseModel):
    mode: str
    public_url: Optional[str] = None


@contextmanager
def _gateway_scope(profile: Optional[str], *, config_write: bool):
    """Scope to the home of the gateway serving *profile*; yields that gateway's profile name."""
    from hermes_constants import profile_name_for_home

    with _profile_scope(profile):
        name = profile_name_for_home(webapp_access.gateway_home()) or profile
    with config_write_scope(name) if config_write else _profile_scope(name):
        yield name


async def _scoped(profile: Optional[str], fn: Callable[[Optional[str]], Any], *, config_write: bool = False) -> Any:
    """Run ``fn(gateway profile)`` in the gateway's scope off the loop; access errors become 400s.
    Only a ``config.yaml`` write takes the config lock; registration waits on the portal outside it."""

    def _run():
        with _gateway_scope(profile, config_write=config_write) as gateway_profile:
            return fn(gateway_profile)

    try:
        return await asyncio.to_thread(_run)
    except webapp_access.WebappAccessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _in_gateway_home(gateway: Optional[str], fn: Callable[[], Any]) -> Any:
    """Run ``fn()`` off the loop in the home and secret scope of the *gateway* profile that
    ``_scoped`` resolved, without the skills lock ``_profile_scope`` holds, which every config,
    skills and tools request takes: for the slow steps (the bundle build, Portal registration),
    which never touch skills. Access errors become 400s."""
    try:
        return await config_scoped_to_thread(gateway, fn)
    except webapp_access.WebappAccessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _restart(gateway_profile: Optional[str], what: str) -> dict:
    """The status after asking the gateway to restart onto the new config. A refused restart (a live
    chat) keeps the saved config and reports ``restart_error``."""
    restart = _restart_gateway_after(gateway_profile, what=what, label="Web Access")
    return {**webapp_access.access_status(), **restart}


@router.get("/api/webapp-access/status")
async def webapp_access_status(profile: Optional[str] = None):
    return await _scoped(profile, lambda _gateway: webapp_access.access_status())


@router.post("/api/webapp-access/start")
async def start_webapp_access(body: StartBody, profile: Optional[str] = None):
    profile = destructive_profile(profile, "POST /api/webapp-access/start")
    gateway, (host, public_url, callback) = await _scoped(
        profile, lambda gateway: (gateway, webapp_access.plan(body.mode, body.public_url)))
    # The bundle before anything lasting: a failed build registers, writes and restarts nothing.
    await _in_gateway_home(gateway, webapp_access.build_bundle)
    await _in_gateway_home(gateway, lambda: webapp_access.register(callback))

    def _start(gateway):
        webapp_access.serve(host, public_url)
        return _restart(gateway, "turning on Web Access")

    return await _scoped(profile, _start, config_write=True)


@router.post("/api/webapp-access/stop")
async def stop_webapp_access(profile: Optional[str] = None):
    profile = destructive_profile(profile, "POST /api/webapp-access/stop")

    def _stop(gateway):
        webapp_access.turn_off()
        return _restart(gateway, "turning off Web Access")

    return await _scoped(profile, _stop, config_write=True)


@router.post("/api/webapp-access/restart")
async def restart_webapp_access(profile: Optional[str] = None):
    """Apply the saved config to a gateway that has not restarted onto it yet: restart that gateway
    and nothing else (no Portal registration, no build, no config write)."""
    profile = destructive_profile(profile, "POST /api/webapp-access/restart")
    return await _scoped(profile, lambda gateway: _restart(gateway, "applying Web Access"))
