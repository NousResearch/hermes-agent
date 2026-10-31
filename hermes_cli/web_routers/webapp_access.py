"""Web Access routes: Desktop's Web Access pane turns the gateway-served Webapp on and off here.

Thin HTTP over :mod:`hermes_cli.webapp_access`. Every handler acts on the home of the gateway that
serves the requested profile; start and stop write its ``webapp:`` config and restart that gateway,
which reads the config only when it starts. Start, Stop and Restart run one at a time. Only native
Desktop on this computer reaches them (:func:`owner_on_this_machine`), never a browser, and only on
the gateway's own listener, where its pane calls them (``SurfacePolicy.web_access``; 404 elsewhere).
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
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
_gateway_restart_failure = late("gateway_restart_failure", "hermes_cli.web_server_gateway")
# One Web Access change at a time in this backend (every Desktop window's pane talks to it): a
# Start, Stop or Restart waits for the one in flight, from its build through the restart request,
# so the Portal's callback, the config and the gateway all end on the last change asked for.
_CHANGE_LOCK = threading.Lock()
_CHANGE_POLL_SECONDS = 0.05


def owner_on_this_machine(request: Request) -> None:
    """Web Access decides what this machine exposes, so only native Desktop on this computer may use
    it: a request admitted with a native grant (``dashboard_auth.native_http``), a one-use ticket from
    the gateway's private control socket, which answers only this OS user, sent from loopback with no
    Origin. A browser tab can present neither. Every other credential is refused, the loopback session
    token included: on the Webapp it is the browser credential, which a tab holds even when it reaches
    the gateway from another machine (an SSH tunnel), and behind the sign-in gate a cookie or bearer
    login may come from another device."""
    if not getattr(request.state, "native_http_principal", None):
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


@asynccontextmanager
async def _one_change_at_a_time():
    """Hold :data:`_CHANGE_LOCK` for one change. Polled rather than awaited on a worker thread, so a
    request cancelled while it waits never takes the lock, and the steps run on any event loop."""
    while not _CHANGE_LOCK.acquire(blocking=False):
        await asyncio.sleep(_CHANGE_POLL_SECONDS)
    try:
        yield
    finally:
        _CHANGE_LOCK.release()


async def _restart(gateway: Optional[str], what: str) -> dict:
    """The status after asking the *gateway* profile's gateway to restart onto the saved config. Runs
    after the config write released the config lock, as main's config routes do: the restart spawn
    and the status query (a control-socket round trip) hold up no config, skills or tools request.
    A refused restart (a live chat) keeps the saved config and reports ``restart_error``."""
    restart = await asyncio.to_thread(_restart_gateway_after, gateway, what=what, label="Web Access")
    return {**await _in_gateway_home(gateway, webapp_access.access_status), **restart}


@router.get("/api/webapp-access/status")
async def webapp_access_status(profile: Optional[str] = None):
    """``restart_failure``: the newest restart of this gateway exited without replacing it, and why;
    the pane waiting on that restart stops and says so instead of timing out. The pane polls this,
    so the status (a control-socket round trip) runs outside the skills lock, as in ``_restart``."""
    gateway = await _scoped(profile, lambda gateway: gateway)
    return await _in_gateway_home(gateway, lambda: {
        **webapp_access.access_status(), "restart_failure": _gateway_restart_failure(gateway)})


@router.post("/api/webapp-access/start")
async def start_webapp_access(body: StartBody, profile: Optional[str] = None):
    profile = destructive_profile(profile, "POST /api/webapp-access/start")
    async with _one_change_at_a_time():
        # Writes a standalone profile's first port claim, so it takes the config lock.
        gateway, (host, port, public_url, callback) = await _scoped(
            profile, lambda gateway: (gateway, webapp_access.plan(body.mode, body.public_url)), config_write=True)
        # The bundle before anything else lasting: a failed build registers, restarts and writes nothing
        # but that port claim.
        await _in_gateway_home(gateway, webapp_access.build_bundle)
        await _in_gateway_home(gateway, lambda: webapp_access.register(callback))
        await _scoped(profile, lambda _gateway: webapp_access.serve(host, public_url, port), config_write=True)
        return await _restart(gateway, "turning on Web Access")


@router.post("/api/webapp-access/stop")
async def stop_webapp_access(profile: Optional[str] = None):
    profile = destructive_profile(profile, "POST /api/webapp-access/stop")

    def _stop(gateway):
        webapp_access.turn_off()
        return gateway

    async with _one_change_at_a_time():
        return await _restart(await _scoped(profile, _stop, config_write=True), "turning off Web Access")


@router.post("/api/webapp-access/restart")
async def restart_webapp_access(profile: Optional[str] = None):
    """Apply the saved config to a gateway that has not restarted onto it yet: restart that gateway
    and nothing else (no Portal registration, no build, no config write)."""
    profile = destructive_profile(profile, "POST /api/webapp-access/restart")
    async with _one_change_at_a_time():
        return await _restart(await _scoped(profile, lambda gateway: gateway), "applying Web Access")
