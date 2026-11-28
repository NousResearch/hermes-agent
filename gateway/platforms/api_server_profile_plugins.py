"""Per-profile plugin routes on the shared api_server listener (``/p/<profile>/api/plugins/...``).

The listener's root app wires ``ctx.register_platform_handler("api_server", ...)`` factories from the
LAUNCH profile's plugin manager only, so a named profile's own plugin copy had no HTTP surface: its
``/p/<profile>/api/plugins/<name>/...`` requests fell to platform ingress and 404'd.

Here each served named profile gets its own aiohttp sub-app, built lazily on the first request from
THAT profile's plugin manager (resolved inside the profile runtime scope the prefix middleware
already entered) and rebuilt whenever its factory set changes, so a plugin enabled after boot is
wired without hitting the root app's frozen router. A sub-app gets the lifecycle a root route gets:

* ``startup()`` before it serves; construction is single-flight, and an app is owned from the
  moment it is built, so a failed or cancelled startup is still torn down.
* Its own ``on_response_prepare`` for a response it returned unprepared: ``connect()`` installs a
  root relay that fires it with the forwarded request, under the owning profile's scope.
* ``shutdown()``/``cleanup()`` once it can no longer serve: a replaced app retires and is torn down
  when its last request is done (a returned response once it is prepared, a WebSocket once it
  closes); at disconnect, admission stops first and every app is torn down after the listener has
  drained.

Every outer middleware (prefix scope, CORS, body limit, security headers) has already run, and the
plugin's handlers authenticate through the shared adapter's ``_check_auth``, which reads the
profile-scoped ``API_SERVER_KEY``. Nothing here falls back to another profile's plugin manager.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

PLUGIN_ROUTE_PREFIX = "/api/plugins/"
_PLATFORM = "api_server"
# Root-request key: the ``_Served`` record of the sub-app request that answered it.
_SERVED_KEY = "hermes_profile_plugin_served"


class _Entry:
    """One built sub-app and what keeps it alive."""

    __slots__ = ("app", "factories", "inflight", "profile", "retired", "signature", "stopped")

    def __init__(self, profile: str, signature: tuple, app: Any, factories: list):
        self.profile, self.signature, self.app = profile, signature, app
        # The factories ride along so their id()s cannot be reused by a later object while cached.
        self.factories = factories
        self.inflight = 0
        self.retired = self.stopped = False


class _Served:
    """One request a sub-app answered; ``held`` while it still counts toward ``entry.inflight``."""

    __slots__ = ("entry", "forwarded", "held", "profile")

    def __init__(self, entry: _Entry, forwarded: Any, profile: str):
        self.entry, self.forwarded, self.profile = entry, forwarded, profile
        self.held = False


def is_plugin_tail(tail: str) -> bool:
    """True when ``/p/<profile>/<tail>`` addresses a plugin route."""
    return ("/" + (tail or "").lstrip("/")).startswith(PLUGIN_ROUTE_PREFIX)


def _factory_key(factory: Any, plugin_name: str) -> tuple:
    # id() as well as the qualname: a forced re-discovery hands back NEW function objects for the
    # same plugin, and the sub-app must be rebuilt around them rather than keep the evicted ones.
    return (plugin_name, getattr(factory, "__qualname__", None) or repr(factory), id(factory))


def _state(adapter: Any) -> tuple[dict, set]:
    """``(cache, detached)``: the serving entry per profile, and owned entries not in the cache
    (starting, or retired with requests still in flight)."""
    cache = getattr(adapter, "_profile_plugin_apps", None)
    if not isinstance(cache, dict):
        cache = adapter._profile_plugin_apps = {}
    detached = getattr(adapter, "_profile_plugin_detached", None)
    if not isinstance(detached, set):
        detached = adapter._profile_plugin_detached = set()
    return cache, detached


def _build_lock(adapter: Any) -> asyncio.Lock:
    lock = getattr(adapter, "_profile_plugin_apps_lock", None)
    if lock is None:
        lock = adapter._profile_plugin_apps_lock = asyncio.Lock()
    return lock


def _closing(adapter: Any) -> bool:
    return bool(getattr(adapter, "_profile_plugin_closing", False))


def _build_app(adapter: Any, profile: str, factories: list) -> Any:
    from aiohttp import web
    from gateway.platforms.api_server import MAX_REQUEST_BYTES
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    for factory, plugin_name in factories:
        try:
            factory(app, adapter)
            logger.info("[%s] Wired native handlers from plugin '%s' for profile '%s'",
                        adapter.name, plugin_name, profile)
        except Exception as exc:  # one bad plugin must not take the others down
            logger.error("[%s] Plugin '%s' handler factory raised for profile '%s': %s",
                         adapter.name, plugin_name, profile, exc, exc_info=True)
    app.freeze()
    return app


async def _stop(adapter: Any, entry: _Entry) -> None:
    """Shut down and clean up one sub-app, once (``cleanup()`` unwinds a partial startup too). The
    caller holds the owning profile's scope."""
    if entry.stopped:
        return
    entry.stopped = True
    _state(adapter)[1].discard(entry)
    # cleanup() runs even when an on_shutdown callback raises (one may touch state a failed startup
    # never created): it is what releases the cleanup_ctx resources startup already entered.
    try:
        try:
            await entry.app.shutdown()
        finally:
            await entry.app.cleanup()
    except Exception:
        logger.warning("[%s] Plugin routes for profile '%s' failed to clean up", adapter.name,
                       entry.profile, exc_info=True)


async def _release(adapter: Any, served: _Served) -> None:
    """The request no longer needs its app; a retired app goes down with its last request."""
    if not served.held:
        return
    served.held = False
    entry = served.entry
    entry.inflight -= 1
    if entry.retired and not entry.inflight:
        await _stop(adapter, entry)


async def _retire(adapter: Any, entry: _Entry) -> None:
    """Take *entry* out of service; tear it down now, or when its last in-flight request is done."""
    entry.retired = True
    if entry.inflight:
        _state(adapter)[1].add(entry)
    else:
        await _stop(adapter, entry)


async def _current_entry(adapter: Any, profile: str) -> Optional[_Entry]:
    """The started sub-app entry for *profile*, (re)built from the profile's own plugin manager.

    Awaited inside *profile*'s runtime scope (``get_plugin_manager()`` keys on the active home).
    ``None`` when the profile has no ``api_server`` plugin factories, its startup failed, or the
    adapter is disconnecting.
    """
    try:
        from hermes_cli.plugins import get_plugin_manager
        manager = get_plugin_manager()
        factories = list(manager.get_platform_handler_factories(_PLATFORM))
    except Exception as exc:
        logger.warning("[%s] Could not load plugin handler factories for profile '%s': %s",
                       getattr(adapter, "name", "api_server"), profile, exc, exc_info=True)
        return None
    signature = (str(getattr(manager, "scope_key", "") or id(manager)),
                 tuple(_factory_key(f, n) for f, n in factories))
    cache, detached = _state(adapter)
    entry = cache.get(profile)
    if entry is not None and entry.signature == signature:
        return entry
    # Single flight: startup awaits, and a second first-request must not run the factories again.
    async with _build_lock(adapter):
        if _closing(adapter):
            return None
        entry = cache.get(profile)
        if entry is not None and entry.signature == signature:
            return entry
        if entry is not None:
            del cache[profile]
            await _retire(adapter, entry)
        if not factories:
            return None
        entry = _Entry(profile, signature, _build_app(adapter, profile, factories), factories)
        detached.add(entry)  # owned across the startup await: disconnect tears down even a stalled build
        try:
            await entry.app.startup()
        except BaseException as exc:
            # A cancelled build (disconnect timing out the request) unwinds what startup allocated
            # before the cancellation propagates.
            if isinstance(exc, Exception):
                logger.error("[%s] Plugin routes for profile '%s' failed to start", adapter.name,
                             profile, exc_info=True)
            await _stop(adapter, entry)
            if not isinstance(exc, Exception):
                raise
            return None
        detached.discard(entry)
        cache[profile] = entry
        return entry


def begin_closing_profile_plugin_apps(adapter: Any) -> None:
    """Stop admitting requests to (and building) per-profile plugin apps; ``connect()`` re-opens."""
    adapter._profile_plugin_closing = True


def reopen_profile_plugin_apps(adapter: Any) -> None:
    adapter._profile_plugin_closing = False


def attach_profile_plugin_apps(adapter: Any, app: Any) -> None:
    """``connect()``: install the response-prepare relay on the root *app* (before its router
    freezes) and admit per-profile plugin requests again."""
    app.on_response_prepare.append(make_response_prepare_relay(adapter))
    reopen_profile_plugin_apps(adapter)


async def close_profile_plugin_apps(adapter: Any) -> None:
    """Tear down every owned sub-app under its owning profile's scope. Disconnect calls it after the
    listener drained, so no request still runs on one."""
    begin_closing_profile_plugin_apps(adapter)
    cache, detached = _state(adapter)
    async with _build_lock(adapter):  # an in-progress build finishes (or unwinds) first
        entries = [*cache.values(), *detached]
        cache.clear()
    from gateway.run import _profile_runtime_scope
    from hermes_cli.profiles import get_profile_dir
    for entry in entries:
        try:
            scope = _profile_runtime_scope(get_profile_dir(entry.profile), hydrate_secrets=False)
        except Exception:
            logger.warning("[%s] Cannot enter profile '%s' to clean up its plugin routes", adapter.name,
                           entry.profile, exc_info=True)
            continue
        with scope:
            await _stop(adapter, entry)


def make_response_prepare_relay(adapter: Any):
    """Root ``on_response_prepare``: for a response a sub-app request handed back unprepared (a
    returned response, or the error response of a raised exception), fire the sub-app's own signals
    with the forwarded request under the owning profile's scope (the prefix middleware has exited
    it by now), then release the request's hold on the app. A response the handler prepared itself
    fired them already and released on return."""

    async def relay(request: Any, response: Any) -> None:
        served = request.get(_SERVED_KEY)
        if served is None or not served.held:
            return
        from gateway.platforms.api_server import _api_request_profile
        token = _api_request_profile.set(served.profile)
        try:
            with adapter._profile_scope(served.profile):
                try:
                    entry = served.entry
                    if not entry.stopped and entry.app.on_response_prepare:
                        await entry.app.on_response_prepare.send(served.forwarded, response)
                finally:
                    await _release(adapter, served)
        finally:
            _api_request_profile.reset(token)

    return relay


async def dispatch_profile_plugin_route(adapter: Any, profile: Optional[str], tail: str,
                                        request: Any) -> Optional[Any]:
    """Serve ``/p/<profile>/<tail>`` from a NAMED profile's plugin sub-app, or ``None`` when it has
    no route for the path (the caller then falls through to platform ingress, which keeps its parked
    503 / unknown 404 answers). The default profile keeps its root-app routes at the plain path, and
    a parked profile serves nothing. The caller has already entered the profile's runtime scope."""
    from aiohttp import hdrs, web
    from aiohttp.web_urldispatcher import MatchInfoError
    from gateway.run_runtime import profile_parked
    if (not profile or profile == "default" or not is_plugin_tail(tail)
            or profile_parked(adapter.gateway_runner, profile)):
        return None
    if _closing(adapter):
        raise web.HTTPServiceUnavailable(text="API server is stopping; retry")
    entry = await _current_entry(adapter, profile)
    if entry is None:
        return None
    rel_url = request.rel_url.with_path("/" + tail.lstrip("/"), keep_query=True, keep_fragment=True)
    # The root app already answered ``Expect: 100-continue``; the sub-app must not send a second one.
    headers = request.headers.copy()
    headers.popall(hdrs.EXPECT, None)
    forwarded = request.clone(rel_url=rel_url, headers=headers, client_max_size=entry.app._client_max_size)
    match = await entry.app.router.resolve(forwarded)
    if isinstance(match, MatchInfoError) and match.http_exception.status == 404:
        return None
    served = _Served(entry, forwarded, profile)
    request[_SERVED_KEY] = served
    entry.inflight += 1
    served.held = True
    # Until the root request prepares what comes back, the app must stay up: the relay releases it.
    handed_off = False
    try:
        response = await entry.app._handle(forwarded)
        handed_off = not response.prepared
        return response
    except Exception:
        handed_off = True  # aiohttp prepares the error response through the root request
        raise
    finally:
        if not handed_off:  # prepared by the handler (a WebSocket), or cancelled
            await _release(adapter, served)


async def handle_profile_ingress(adapter: Any, request: Any) -> Any:
    """The shared listener's ``/p/{profile}/{tail}`` catch-all: the named profile's own plugin
    routes first, then platform ingress (which keeps its parked 503 / unknown 404 answers)."""
    from gateway.platforms.api_server import _api_request_profile
    from gateway.platforms.shared_ingress import dispatch_profile_ingress
    profile, tail = _api_request_profile.get(), request.match_info.get("tail", "")
    response = await dispatch_profile_plugin_route(adapter, profile, tail, request)
    if response is not None:
        return response
    return await dispatch_profile_ingress(adapter.gateway_runner, profile, tail, request, scoped=True)
