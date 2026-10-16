"""Dashboard plugins of a secondary served profile (PLG-3, #134889).

A gateway serving several profiles imports dashboard plugin code for its launch profile only. A
request scoped to another served profile (its native grant, a ``?profile=`` selection) gets that
profile's own plugins: discovered under its own home and the root, gated by its own ``plugins.enabled`` /
``plugins.disabled``, and run in its own plugin host (bound to its home and secret scope) whatever
its ``plugins.isolation`` says, so another profile's plugin code never enters this process. Bundled
plugins are Hermes code and stay in-process. The host protocol is request/response, so a secondary
profile's plugin websockets are refused by name until it carries them.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from urllib.parse import parse_qsl, urlencode

from fastapi.responses import JSONResponse, Response
from starlette import status
from starlette._utils import get_route_path
from starlette.websockets import WebSocket

from hermes_cli.config import get_process_hermes_home

_log = logging.getLogger("hermes_cli.web_server")

# Valid for every served profile, so a profile's plugin code must never receive them.
_GATEWAY_CREDENTIAL_HEADERS = frozenset({
    "authorization", "cookie", "host", "x-hermes-gateway-ticket", "x-hermes-session-token"})
# Every query credential the gateway accepts (``web_server_chat._ws_auth_ok``, ``_has_valid_query_token``,
# the display bridge, the PTY reattach key) and the profile selector; compared after URL-decoding.
_GATEWAY_QUERY_KEYS = frozenset({"token", "ticket", "internal", "display_ticket", "attach", "profile"})

_profile_plugins: dict[str, list] = {}
_profile_plugins_lock = threading.Lock()

# Requests forwarded to a profile's plugin host wait on that profile's own threads, not the gateway's
# shared ``asyncio.to_thread`` pool: a slow plugin queues behind its own profile's plugin requests
# instead of stalling every route that offloads blocking work.
_HOSTED_REQUEST_THREADS = 4
# How long a forwarded request may take, queueing included, before it answers 504. The host protocol
# has no cancel, so the host may still finish the abandoned handler; the gateway's thread is freed.
HOSTED_REQUEST_TIMEOUT_SECS = 60.0
_hosted_executors: dict[str, ThreadPoolExecutor] = {}
# A name a secondary profile lacks rescans its plugin dirs (it may be newly installed), then not again
# for this long, so 404 probes can't force a disk scan per request. An explicit rescan clears it.
_UNKNOWN_PLUGIN_RESCAN_SECS = 10.0
_unknown_plugins: dict[tuple[str, str], float] = {}


def secondary_profile_home() -> Optional[Path]:
    """The active scope's home when it is a profile other than the launch profile, else None."""
    from hermes_constants import get_hermes_home, hermes_home_key
    home = get_hermes_home()
    return None if hermes_home_key(home) == hermes_home_key(get_process_hermes_home()) else Path(home)


def profile_dashboard_plugins(home: Path, force_rescan: bool = False) -> list:
    """Dashboard plugins of the profile at ``home``, cached per home like the launch profile's list:
    its own ``plugins/``, then the root's (as a standalone ``hermes --profile <name> serve`` scanned
    them; the profile's own copy wins a same-name conflict), then the bundled ones."""
    from hermes_cli.plugins import get_bundled_plugins_dir
    from hermes_cli.profiles import _get_default_hermes_home
    from hermes_cli.web_server_dashboard import _discover_dashboard_plugins
    from hermes_constants import hermes_home_key
    key = hermes_home_key(home)
    with _profile_plugins_lock:
        if force_rescan:
            for unknown in [k for k in _unknown_plugins if k[0] == key]:
                del _unknown_plugins[unknown]
        cached = _profile_plugins.get(key)
        if cached is None or force_rescan or any(not Path(p["_dir"]).is_dir() for p in cached):
            bundled, own = get_bundled_plugins_dir(), Path(home) / "plugins"
            dirs = [own] + [r for r in [_get_default_hermes_home() / "plugins"]
                            if r.resolve(strict=False) != own.resolve(strict=False)]
            cached = _profile_plugins[key] = _discover_dashboard_plugins(
                [(d, "user") for d in dirs] + [(bundled / "memory", "bundled"), (bundled, "bundled")])
        return cached


def _named(plugins: list, name: str) -> Optional[dict]:
    return next((p for p in plugins if p.get("name") == name), None)


def rescanned_plugin(home: Path, name: str) -> Optional[dict]:
    """The profile at ``home``'s entry for ``name`` after a rescan, or None; a name found missing
    within the last :data:`_UNKNOWN_PLUGIN_RESCAN_SECS` is answered without one."""
    from hermes_constants import hermes_home_key
    key, now = (hermes_home_key(home), name), time.monotonic()
    with _profile_plugins_lock:
        if _unknown_plugins.get(key, now) > now:
            return None
    plugin = _named(profile_dashboard_plugins(home, force_rescan=True), name)
    if plugin is None:
        with _profile_plugins_lock:
            for expired in [k for k, until in _unknown_plugins.items() if until <= now]:
                del _unknown_plugins[expired]
            _unknown_plugins[key] = now + _UNKNOWN_PLUGIN_RESCAN_SECS
    return plugin


def plugin_api_path(plugin: Optional[dict]) -> Optional[Path]:
    """The resolved backend file of a discovered plugin entry, or None."""
    if not plugin or not plugin.get("_api_file"):
        return None
    return (Path(plugin["_dir"]) / plugin["_api_file"]).resolve()


def _in_process_api_file(name: str) -> Optional[Path]:
    """The file this process's ``/api/plugins/<name>/`` router runs: the imported module's when one is
    mounted (a hosted forward route has none here), else the copy the first request would mount,
    the launch profile's discovery entry."""
    import sys
    from hermes_cli.web_server import _get_dashboard_plugins, app
    from hermes_cli.web_server_dashboard import _launch_profile_scope, _plugin_api_mounted
    if _plugin_api_mounted(app, name):
        file = getattr(sys.modules.get(f"hermes_dashboard_plugin_{name}"), "__file__", None)
        return Path(file).resolve() if file else None
    with _launch_profile_scope():
        return plugin_api_path(_named(_get_dashboard_plugins(), name))


def _runs_in_process(plugin: Optional[dict]) -> bool:
    """Whether a secondary profile's entry may use this process's router: only its bundled copy,
    and only when that router runs that very file (a launch-profile user plugin can shadow the
    bundled name, or a disabled one hide it)."""
    return (plugin is not None and plugin.get("source") == "bundled"
            and _in_process_api_file(plugin["name"]) == plugin_api_path(plugin))


def secondary_plugin(name: str) -> Optional[dict]:
    """The active secondary profile's own entry for ``name``, or None (also in the launch scope)."""
    home = secondary_profile_home()
    return None if home is None else _named(profile_dashboard_plugins(home), name)


def _secondary_hosted_plugin(name: str) -> Optional[dict]:
    """The active secondary profile's own entry for ``name`` when its plugin host serves it, or None:
    the launch profile's scope, an unknown name, or the bundled copy this process runs."""
    plugin = secondary_plugin(name)
    return None if plugin is None or _runs_in_process(plugin) else plugin


def in_process_router_refused(name: str) -> bool:
    """Whether this process's ``/api/plugins/<name>/`` router (the launch profile's copy) must not
    serve the active scope: a secondary profile runs only its own bundled copy here."""
    return secondary_profile_home() is not None and not _runs_in_process(secondary_plugin(name))


@contextmanager
def plugin_request_scope(scope) -> Iterator[None]:
    """Bind a plugin request's ``?profile=`` selection for its routing decision (the route
    dependency binds it again for the handler). An unknown profile binds nothing: the route
    dependency answers it as for any other route."""
    from fastapi import HTTPException
    from hermes_cli.web_server_profiles import _config_profile_scope, _is_current_profile
    profile = dict(parse_qsl(scope.get("query_string", b"").decode("latin-1"))).get("profile")
    with ExitStack() as stack:
        if not _is_current_profile(profile):
            try:
                stack.enter_context(_config_profile_scope(profile))
            except HTTPException as exc:
                _log.debug("Plugin request for profile %r left unscoped: %s", profile, exc.detail)
        yield


def hosted_response(result: dict[str, Any]) -> Response:
    """A plugin host's buffered ``asgi`` answer as a response."""
    response = Response(content=result["body"], status_code=int(result["status"]))
    for key, value in result["headers"]:  # a list, so repeated headers (Set-Cookie) all survive
        if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
            response.headers.append(key, value)
    return response


def _hosted_executor(home: Path) -> ThreadPoolExecutor:
    from hermes_constants import hermes_home_key
    key = hermes_home_key(home)
    with _profile_plugins_lock:
        executor = _hosted_executors.get(key)
        if executor is None:
            executor = _hosted_executors[key] = ThreadPoolExecutor(
                _HOSTED_REQUEST_THREADS, thread_name_prefix=f"plugin-host-requests-{Path(home).name}")
        return executor


async def run_hosted_request(home: Path, name: str, call: Callable[[float], dict[str, Any]]) -> Response:
    """Run ``call(timeout)``, one plugin host ``asgi`` round trip for plugin ``name`` of the profile at
    ``home``, on that profile's own threads in this request's context; its answer, or 504 after
    :data:`HOSTED_REQUEST_TIMEOUT_SECS`. Other failures propagate."""
    timeout = HOSTED_REQUEST_TIMEOUT_SECS
    deadline, context = time.monotonic() + timeout, contextvars.copy_context()
    job = asyncio.get_running_loop().run_in_executor(
        _hosted_executor(home), context.run, lambda: call(max(deadline - time.monotonic(), 0.01)))
    try:
        return hosted_response(await asyncio.wait_for(job, timeout))
    except TimeoutError:
        _log.warning("Plugin %s: its plugin host did not answer within %gs", name, timeout)
        return JSONResponse({"detail": f"plugin {name!r} did not answer within {timeout:g}s"}, status_code=504)


async def forward_secondary_plugin_request(request, name: str) -> Optional[Response]:
    """Serve ``/api/plugins/<name>/`` for a secondary profile through its own plugin host, or None
    when the in-process router answers (the launch profile, or a bundled plugin). Consent has been
    checked in the request's scope already."""
    plugin, home = _secondary_hosted_plugin(name), secondary_profile_home()
    if plugin is None or home is None:
        return None
    if not plugin.get("_api_file"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    rest = get_route_path(request.scope).split("/", 4)
    path = "/" + (rest[4] if len(rest) > 4 else "")
    query = urlencode([(k, v) for k, v in parse_qsl(request.url.query, keep_blank_values=True)
                       if k not in _GATEWAY_QUERY_KEYS])
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _GATEWAY_CREDENTIAL_HEADERS]
    body = await request.body()

    def call(timeout: float) -> dict[str, Any]:
        from hermes_cli.plugins import get_plugin_manager
        host = get_plugin_manager()._plugin_host()  # the active (secondary) profile's own host
        return host.asgi_request(name, str(plugin["_dir"]), str(plugin["_api_file"]), request.method, path,
                                 query, headers, body, timeout=timeout)

    try:
        return await run_hosted_request(home, name, call)
    except Exception as exc:  # health: allow BLE001 -- the host runs third-party code; any failure is this request's 502
        _log.warning("Plugin %s: its profile's plugin host could not serve %s: %s", name, path, exc)
        return JSONResponse({"detail": f"plugin {name!r} could not run in this profile's plugin host"},
                            status_code=502)


async def refuse_secondary_plugin_socket(scope, receive, send, name: str) -> bool:
    """Answer a secondary profile's socket to a plugin its host serves, and return True; False when
    the in-process router takes it. An authenticated dial is refused by name (HTTP 501 before
    accept, so Desktop's reconnect budget counts it); any other gets the plain policy close."""
    if _secondary_hosted_plugin(name) is None:
        return False
    from hermes_cli.web_server_chat import _ws_auth_ok
    socket = WebSocket(scope, receive, send)
    if not ((scope.get("state") or {}).get("native_http_principal") or _ws_auth_ok(socket)):
        await send({"type": "websocket.close", "code": status.WS_1008_POLICY_VIOLATION})
        return True
    detail = f"plugin {name!r} runs in this profile's plugin host, which does not carry websockets yet"
    _log.info("Plugin %s: refused a websocket for %s: %s", name, secondary_profile_home(), detail)
    if "websocket.http.response" in (scope.get("extensions") or {}):
        await socket.send_denial_response(JSONResponse({"detail": detail}, status_code=501))
    else:
        await send({"type": "websocket.close", "code": status.WS_1008_POLICY_VIOLATION, "reason": detail[:120]})
    return True
