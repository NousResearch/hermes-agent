"""Web Access routes: Desktop's Web Access pane turns the gateway-served Webapp on and off here.

Thin HTTP over :mod:`hermes_agent.hermes_cli.webapp_access`. Every handler acts on the home of the gateway that
serves the requested profile; start and stop write its ``webapp:`` config and restart that gateway,
which reads the config only when it starts. Start, Stop and Restart run one at a time. Only native
Desktop on this computer reaches them (:func:`owner_on_this_machine`), never a browser, and only on
the gateway's own listener, where its pane calls them (``SurfacePolicy.web_access``; 404 elsewhere).
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from hermes_agent.hermes_cli import webapp_access
from hermes_agent.hermes_cli.web_deps import late
from hermes_agent.hermes_cli.web_routers._common import (
    _profile_scope,
    config_scoped_to_thread,
    config_write_scope,
    destructive_profile,
)

_restart_gateway_after = late("_restart_gateway_after", "hermes_agent.hermes_cli.web_server_gateway")
_gateway_restart_failure = late("gateway_restart_failure", "hermes_agent.hermes_cli.web_server_gateway")
# One Web Access change at a time in this backend (every Desktop window's pane talks to it): a
# Start, Stop or Restart waits for the one in flight, from its build through the restart request,
# so the Portal's callback, the config and the gateway all end on the last change asked for.
_CHANGE_LOCK = threading.Lock()
_CHANGE_POLL_SECONDS = 0.05
# The Starts in flight, by the gateway they turn the Webapp on for, oldest first: each one's step and
# when it was asked for. The status the pane polls carries one, because building the bundle can take
# minutes: the pane says what is happening, and a pane opened meanwhile shows the Start running
# instead of offering it. A record lives from the request's arrival (``preparing`` while another
# change holds the lock) until the restart it asked for serves the new config, fails, is replaced
# by a Stop, a Restart or a later Start's restart, or outlasts the pane's own wait for it. Mutated
# on the event loop only.
_STARTS: dict[Optional[str], list[dict]] = {}
#: How long a Start's restart is followed after its request answers: as long as the pane waits.
_RESTART_FOLLOW_SECONDS = 90.0


def _start_in_flight(gateway: Optional[str]) -> Optional[dict]:
    """The Start the status reports for *gateway*: the newest one past ``preparing`` (a queued one
    only waits behind it), else the oldest still waiting. Drops restarts followed for too long."""
    now = time.monotonic()
    records = [r for r in _STARTS.get(gateway, ()) if r.get("follow_until", now) >= now]
    if not records:
        _STARTS.pop(gateway, None)
        return None
    _STARTS[gateway] = records
    running = [r for r in records if r["step"] != "preparing"]
    return running[-1] if running else records[0]


def _reported(record: Optional[dict]) -> Optional[dict]:
    """``start_progress`` as the pane reads it: the step and when the Start was asked for."""
    return None if record is None else {"step": record["step"], "started_at": record["started_at"]}


def _restart_settled(record: dict, status: dict, instance: Optional[tuple]) -> bool:
    """A Start whose request answered ends once a gateway other than the one its restart replaces
    (*instance* answers now) serves the config it wrote, or that restart exited without replacing
    it. The gateway it replaces may serve an earlier Start's config on the same port."""
    if "restart_pid" not in record:
        return False
    failure = status.get("restart_failure") or {}
    if failure.get("pid") == record["restart_pid"]:
        return True
    return instance not in (None, record["replaces"]) and webapp_access.serves_config(status)


def _drop(gateway: Optional[str], record: dict) -> None:
    """Forget *record* and no other Start, so one Start ending never hides another."""
    remaining = [r for r in _STARTS.get(gateway, ()) if r is not record]
    if remaining:
        _STARTS[gateway] = remaining
    else:
        _STARTS.pop(gateway, None)


def _drop_followed(gateway: Optional[str]) -> None:
    """A Stop, a Restart or a later Start's restart replaces the restart a finished Start is
    followed by."""
    for record in [r for r in _STARTS.get(gateway, ()) if "restart_pid" in r]:
        _drop(gateway, record)


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
    the pane waiting on that restart stops and says so instead of timing out. ``start_progress``:
    the Start for this gateway (``step``, ``started_at`` in epoch seconds) from its request's arrival
    until a gateway serves the config it wrote (or its restart failed), else None.
    The pane polls this, so the status (a control-socket round trip) runs outside the skills lock,
    as in ``_restart``."""
    gateway = await _scoped(profile, lambda gateway: gateway)
    # Before the status round trip: a Start that ends during it is still reported, so the pane polls
    # again rather than showing that config change undone until its next refetch.
    start = _start_in_flight(gateway)
    followed = any("restart_pid" in r for r in _STARTS.get(gateway, ()))

    def read():
        # The gateway's identity first: one replaced between the two reads then still counts as old.
        instance = webapp_access.gateway_instance() if followed else None
        return instance, {**webapp_access.access_status(), "restart_failure": _gateway_restart_failure(gateway)}

    instance, status = await _in_gateway_home(gateway, read)
    while start is not None and _restart_settled(start, status, instance):
        _drop(gateway, start)
        start = _start_in_flight(gateway)
    return {**status, "start_progress": _reported(start)}


@router.post("/api/webapp-access/start")
async def start_webapp_access(body: StartBody, profile: Optional[str] = None):
    profile = destructive_profile(profile, "POST /api/webapp-access/start")
    gateway = await _scoped(profile, lambda gateway: gateway)
    # Reported from its arrival: a Start waiting for another change shows as one being prepared, and
    # its clock counts from when it was asked for.
    record = {"step": "preparing", "started_at": time.time()}
    _STARTS.setdefault(gateway, []).append(record)
    try:
        async with _one_change_at_a_time():
            # Writes a standalone profile's first port claim, so it takes the config lock.
            host, port, public_url, callback = await _scoped(
                profile, lambda _gateway: webapp_access.plan(body.mode, body.public_url), config_write=True)
            # The bundle before anything else lasting: a failed build registers, restarts and writes
            # nothing but that port claim.
            record["step"] = "building"
            await _in_gateway_home(gateway, webapp_access.build_bundle)
            record["step"] = "registering"
            await _in_gateway_home(gateway, lambda: webapp_access.register(callback))
            record["step"] = "restarting"
            await _scoped(profile, lambda _gateway: webapp_access.serve(host, public_url, port), config_write=True)
            replaces = await _in_gateway_home(gateway, webapp_access.gateway_instance)
            result = await _restart(gateway, "turning on Web Access")
            if result.get("restart_started") and result.get("restart_pid") is not None:
                # Followed past the request: until the restarted gateway serves this config, a pane
                # opened now shows the Start restarting, not the saved change waiting for a restart.
                _drop_followed(gateway)
                record.update(restart_pid=result["restart_pid"], replaces=replaces,
                              follow_until=time.monotonic() + _RESTART_FOLLOW_SECONDS)
            else:
                _drop(gateway, record)
            # The pane keeps this answer as the status it shows. Where the backend answering is the
            # gateway being restarted, no status read succeeds until its replacement is up, so the
            # answer itself must say the Start is still under way.
            return {**result, "start_progress": _reported(_start_in_flight(gateway))}
    finally:
        if "restart_pid" not in record:
            _drop(gateway, record)


@router.post("/api/webapp-access/stop")
async def stop_webapp_access(profile: Optional[str] = None):
    profile = destructive_profile(profile, "POST /api/webapp-access/stop")

    def _stop(gateway):
        webapp_access.turn_off()
        return gateway

    async with _one_change_at_a_time():
        gateway = await _scoped(profile, _stop, config_write=True)
        _drop_followed(gateway)
        return await _restart(gateway, "turning off Web Access")


@router.post("/api/webapp-access/restart")
async def restart_webapp_access(profile: Optional[str] = None):
    """Apply the saved config to a gateway that has not restarted onto it yet: restart that gateway
    and nothing else (no Portal registration, no build, no config write)."""
    profile = destructive_profile(profile, "POST /api/webapp-access/restart")
    async with _one_change_at_a_time():
        gateway = await _scoped(profile, lambda gateway: gateway)
        _drop_followed(gateway)
        return await _restart(gateway, "applying Web Access")
