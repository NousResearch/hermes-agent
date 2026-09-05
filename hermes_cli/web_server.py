"""Hermes Agent — Web UI server: FastAPI app assembly, auth/host middleware, ``start_server``.

Route handlers live in ``web_routers/``; their helpers live in the sibling
``web_server_<concern>`` modules and are re-imported here so ``web_server.<name>``
stays the single late-binding seam tests monkeypatch (``web_deps.late``).
Usage: ``python -m hermes_cli.main web [--port 8080]``.
"""

from contextlib import asynccontextmanager

import asyncio
from collections import deque
import hmac
import logging
import os
import re
import secrets
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.parse

from hermes_cli.install_identity import get_install_id as _shared_get_install_id
from hermes_cli.pty_session import run_reaper
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli import __version__
from hermes_cli.config import load_config

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    # First try lazy-installing the dashboard extras. Only the user actually
    # running `hermes dashboard` needs fastapi+uvicorn; lazy install keeps
    # them out of every other install path. After install, re-import.
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.dashboard", prompt=False)
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except Exception:
        raise SystemExit(
            "Web UI requires fastapi and uvicorn.\n"
            f"Install with: {sys.executable} -m pip install 'fastapi' 'uvicorn[standard]'"
        )

WEB_DIST = Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ else Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)


from hermes_cli.web_server_lifecycle import (  # noqa: E402
    PORT_IN_USE_EXIT_CODE,
    _dashboard_forwarded_allow_ips,
    _eager_reconcile_own_session_db,
    _maybe_open_browser,
    _port_bind_conflict,
    _read_bound_port,
    _report_port_in_use,
    _start_parent_death_watchdog,
    _warm_gateway_module,
    _write_dashboard_ready_file,
    _write_machine_sentinel_line,
)


def _start_desktop_cron_ticker(stop_event: "threading.Event", interval: int = 60) -> None:
    """Tick the cron scheduler from inside the desktop dashboard backend.

    The desktop spawns a ``hermes dashboard`` backend, not a gateway, so without
    this a cron created in the app would never fire (no live adapters; delivery
    falls back to the per-platform send path). The primary backend outlives the
    per-profile pool (reaped after ~10 idle minutes), so it ticks EVERY local
    profile's store like a multiplex gateway; external providers keep the
    single-store behavior (registries are not profile-scoped). Cross-process
    safe: the built-in tick takes the per-store ``cron/.tick.lock``.

    Every local profile's store is ticked, not just this backend's own (#69377's desktop sibling): the
    desktop pools per-profile backends and reaps them after ~10 idle minutes, so a secondary profile's
    ticker dies with its backend and that profile's jobs silently stop firing until the user next opens it
    ("tasks on the sleeping profile could be idle" — community report, Aug 2026).
    """
    from cron.scheduler_provider import InProcessCronScheduler, resolve_cron_scheduler

    provider = resolve_cron_scheduler()

    start_kwargs: dict = {"interval": interval}
    if isinstance(provider, InProcessCronScheduler):
        try:
            from hermes_cli.profiles import profiles_to_serve

            profile_homes = list(profiles_to_serve(multiplex=True))
            if len(profile_homes) > 1:
                start_kwargs["profile_homes"] = profile_homes
                # Stand down, per tick, for a profile whose OWN gateway runs:
                # it ticks with live adapters, and the tick-lock race would
                # otherwise deliver through the standalone path (#100489).
                from hermes_cli.profiles import _check_gateway_running

                start_kwargs["profile_gate"] = lambda _name, home: not _check_gateway_running(Path(home))
                from hermes_logging import enable_profile_log_routing

                enable_profile_log_routing(profile_homes)
                _log.info(
                    "Desktop cron scheduler will tick %d profile(s): %s",
                    len(profile_homes),
                    [name for name, _home in profile_homes],
                )
        except Exception:
            # Fail open to the single-store ticker so the active profile keeps firing.
            _log.exception("Desktop cron: profile enumeration failed; ticking active profile only")

    _log.info("Desktop cron scheduler started (provider=%s, interval=%ds)", provider.name, interval)
    provider.start(stop_event, **start_kwargs)


# Desktop `serve` only (start_server(start_mcp_discovery_after_bind=True)):
# seconds after the READY sentinel before the MCP discovery thread starts.
_DESKTOP_MCP_DISCOVERY_DELAY_S = 1.0


@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    app.state.event_channels = {}  # dict[str, set]
    app.state.event_lock = asyncio.Lock()
    app.state.pty_active_session_files = {}  # dict[str, Path]
    # Serializes chat-argv resolution so concurrent /api/pty connections don't
    # overlap ``npm install`` / ``npm run build``. Locks live on app.state (not
    # module globals) so they bind to the running loop, not the import-time one.
    app.state.chat_argv_lock = asyncio.Lock()

    # Bring state.db schema current BEFORE the first session-list poll
    # (#79531/#80037): a store left behind by `hermes update` otherwise 500s
    # every poll while the read-probe heal loses to sibling lock contention.
    # Daemon thread so a locked store never delays the socket (Desktop
    # ready-probe times out at 10s, GH-73083).
    threading.Thread(
        target=_eager_reconcile_own_session_db,
        daemon=True,
        name="statedb-eager-reconcile",
    ).start()

    # Import hermes_cli.gateway *before* the yield: on Windows + 3.11 the
    # import holds the GIL, so run_in_executor still froze the loop 15-22s and
    # the Desktop's 10s ready-probe timed out (GH-73083).
    _warm_gateway_module()

    # Snapshot the checkout revision so lazy-import paths (model picker) can
    # refuse with "restart required" after `hermes update` replaced the code
    # (#86207); the update flow does not reliably restart the dashboard.
    from gateway.code_skew import record_boot_fingerprint

    record_boot_fingerprint()

    # Hosted Bot rooms belong to the backend process. Recovery may need a
    # contended state.db migration, so keep it off the pre-yield path: Group
    # Chat must degrade on its own rather than block every Desktop feature.
    from tui_gateway import methods_groups as _hosted_groups
    import tui_gateway.server  # noqa: F401

    hosted_room_start_cancel = threading.Event()

    def _start_hosted_rooms() -> None:
        try:
            _hosted_groups.start_hosted_room_service()
        except Exception:
            _log.exception("Hosted Group Chat recovery failed during backend startup")
        finally:
            if hosted_room_start_cancel.is_set():
                _hosted_groups.stop_hosted_room_service(timeout=1.0)

    hosted_room_start_thread = threading.Thread(
        target=_start_hosted_rooms,
        daemon=True,
        name="hosted-room-startup",
    )
    hosted_room_start_thread.start()

    # Desktop-spawned backends (HERMES_DESKTOP=1) fire cron jobs themselves,
    # since the app has no gateway running the scheduler. Server `hermes
    # dashboard` is unaffected — it relies on its own gateway.
    cron_stop: "threading.Event | None" = None
    cron_thread: "threading.Thread | None" = None
    if os.getenv("HERMES_DESKTOP") == "1":
        # Reap an orphaned gateway from an abnormal previous exit (reparented to
        # launchd, still holding the platform WebSocket) before forking a fresh
        # one that would race the same credential (#77276). Runs
        # unconditionally; protection of a healthy standalone gateway lives
        # INSIDE the reaper (registration probed with cleanup_stale=False).
        try:
            from hermes_cli.gateway import _reap_unsupervised_gateway_orphans

            _reap_unsupervised_gateway_orphans()
        except Exception:
            _log.exception("Desktop startup: orphan gateway reap failed")

        cron_stop = threading.Event()
        cron_thread = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(cron_stop,),
            daemon=True,
            name="desktop-cron-ticker",
        )
        cron_thread.start()

    # Reap idle/dead keep-alive PTY sessions (30-min TTL).
    pty_reaper_task = asyncio.create_task(run_reaper(PTY_REGISTRY))
    # Periodic authenticated self-test feeding the ``dashboard`` component on /api/status.
    selftest_task = asyncio.create_task(_dashboard_selftest_loop())
    # Live auto-archive timer, independent of list requests.
    auto_archive_task = asyncio.create_task(_auto_archive_ticker_loop())

    # Managed local runtime (local_runtime.enabled): bring llama-server back so a
    # restart doesn't strand a llamacpp main model. Off-thread and best-effort;
    # failure falls back to cloud providers like a cold start. Server only —
    # models load on first inference (an empty router holds no VRAM).
    def _boot_local_runtime():
        try:
            from hermes_cli.config import load_config
            from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

            ensure_local_runtime(load_config())
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("local runtime boot failed: %s", exc)

    threading.Thread(target=_boot_local_runtime, daemon=True, name="local-runtime-boot").start()

    try:
        yield
    finally:
        hosted_room_start_cancel.set()
        _hosted_groups.stop_hosted_room_service(timeout=5.0)
        hosted_room_start_thread.join(timeout=1.0)
        if cron_stop is not None:
            cron_stop.set()
        pty_reaper_task.cancel()
        selftest_task.cancel()
        auto_archive_task.cancel()
        await PTY_REGISTRY.close_all()
        # Stop the managed llama-server with its parent (an orphan pins VRAM).
        try:
            from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime

            shutdown_local_runtime()
        except Exception:  # noqa: BLE001
            pass
        if os.getenv("HERMES_DESKTOP") == "1":
            _terminate_desktop_managed_gateway()


def _app_state_default(app: "FastAPI", name: str, factory):
    """Return ``app.state.<name>``, lazily creating it for non-``with`` TestClient usages.

    The lifespan normally initialises these on the running event loop (an
    asyncio.Lock created at import time binds to whatever loop was active then).
    """
    try:
        return getattr(app.state, name)
    except AttributeError:
        value = factory()
        setattr(app.state, name, value)
        return value


def _get_chat_argv_lock(app: "FastAPI") -> asyncio.Lock:
    return _app_state_default(app, "chat_argv_lock", asyncio.Lock)


def _get_pty_active_session_files(app: "FastAPI") -> dict[str, Path]:
    return _app_state_default(app, "pty_active_session_files", dict)


app = FastAPI(title="Hermes Agent", version=__version__, lifespan=_lifespan)


# Memory-provider OAuth connect routes live in the memory layer, not here.
from hermes_cli.memory_oauth import router as _memory_oauth_router  # noqa: E402

app.include_router(_memory_oauth_router)

# Session token for sensitive endpoints. The desktop shell mints it via
# HERMES_DASHBOARD_SESSION_TOKEN; otherwise fresh per server start. It dies with
# the process and is injected into the SPA HTML so only the web UI can use it.
def _resolve_session_token() -> str:
    return os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or secrets.token_urlsafe(32)


_SESSION_TOKEN = _resolve_session_token()
_SESSION_HEADER_NAME = "X-Hermes-Session-Token"
_SSH_OWNER_NONCE: Optional[str] = None
_SSH_RUNTIME_PURELIB: Optional[Tuple[str, int, int]] = None
_SSH_RUNTIME_MARKER: Optional[str] = None


def _apply_ssh_session_token(token: str) -> None:
    global _SESSION_TOKEN
    if token:
        _SESSION_TOKEN = token


def _apply_ssh_owner_nonce(nonce: Optional[str]) -> None:
    global _SSH_OWNER_NONCE, _SSH_RUNTIME_PURELIB, _SSH_RUNTIME_MARKER
    _SSH_OWNER_NONCE = nonce
    _SSH_RUNTIME_PURELIB = None
    _SSH_RUNTIME_MARKER = None
    if nonce:
        try:
            purelib = sysconfig.get_paths()["purelib"]
        except (KeyError, OSError):
            return
        # Primary identity: a marker FILE in site-packages. A replaced venv
        # loses it deterministically; pip installs leave it. A bare (dev, ino)
        # snapshot alone is NOT enough: ext4 reuses directory inodes at once,
        # so `rm -rf venv && uv venv` can land on the same inode undetected.
        try:
            marker = os.path.join(purelib, f".hermes-ssh-runtime-{nonce}")
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()}\n")
            _SSH_RUNTIME_MARKER = marker
        except OSError:
            pass  # read-only site-packages — fall back to the stat snapshot
        try:
            st = os.stat(purelib)
            _SSH_RUNTIME_PURELIB = (purelib, st.st_dev, st.st_ino)
        except OSError:
            pass


def _ssh_runtime_intact() -> bool:
    if _SSH_RUNTIME_MARKER is not None:
        return os.path.isfile(_SSH_RUNTIME_MARKER)
    # Fallback (read-only site-packages): directory identity snapshot — weaker
    # (inode reuse) but catches cross-device moves and version-bump paths.
    if _SSH_RUNTIME_PURELIB is None:
        return True
    purelib, device, inode = _SSH_RUNTIME_PURELIB
    try:
        st = os.stat(purelib)
    except OSError:
        return False
    return (st.st_dev, st.st_ino) == (device, inode)


# In-browser Chat tab (/chat, /api/pty, /api/ws): always enabled. A module
# constant (not an inlined True) so the WS endpoints and SPA token injection
# share one testable seam.
_DASHBOARD_EMBEDDED_CHAT_ENABLED = True

# Desktop file.attach sends a whole base64 data URL in one JSON-RPC frame;
# uvicorn's 16 MiB default rejects files under the 256 MiB raw attach cap.
_DESKTOP_ATTACHMENT_WS_MAX_BYTES = 384 * 1024 * 1024


# CORS: localhost origins only — allow_origins=["*"] on 0.0.0.0 would let any
# website read/modify config and secrets.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints that do NOT require the session token; everything else under /api/
# is gated below. Shared with the OAuth gate so the two allowlists cannot
# drift (/api/status once 401'd under the OAuth gate, breaking the portal probe).
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS as _PUBLIC_API_PATHS


def _has_valid_session_token(request: Request) -> bool:
    """True if the request carries a valid dashboard session token.

    The dedicated header avoids collisions with reverse proxies that already use
    ``Authorization`` (Caddy ``basic_auth``); the legacy Bearer path stays for
    older dashboard bundles.
    """
    session_header = request.headers.get(_SESSION_HEADER_NAME, "")
    if session_header and hmac.compare_digest(session_header.encode(), _SESSION_TOKEN.encode()):
        return True
    auth = request.headers.get("authorization", "")
    return hmac.compare_digest(auth.encode(), f"Bearer {_SESSION_TOKEN}".encode())


# Routes that may also authenticate via ``?token=`` (download links opened by
# the OS shell / a new tab, where no header can be set). Kept narrow.
_QUERY_TOKEN_API_PATHS: frozenset[str] = frozenset({"/api/files/download"})


def _has_valid_query_token(request: Request, path: str) -> bool:
    if path not in _QUERY_TOKEN_API_PATHS:
        return False
    token = request.query_params.get("token", "")
    return bool(token) and hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode())


def _require_token(request: Request) -> None:
    """Authorize a sensitive endpoint, raising 401 if the caller isn't allowed.

    Loopback mode (``auth_required`` False): validate the SPA-injected
    ``_SESSION_TOKEN``. Gated mode: the token is NOT injected (cookie auth), and
    ``gated_auth_middleware`` already 401'd anything without a verified
    ``request.state.session`` — requiring the absent token here would make every
    ``_require_token`` endpoint unreachable behind the gate, so defer to it.
    """
    if getattr(request.app.state, "auth_required", False):
        ok = getattr(request.state, "session", None) is not None
    else:
        ok = _has_valid_session_token(request)
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized")


# Accepted Host values for loopback binds. DNS rebinding TTL-flips an attacker
# hostname to 127.0.0.1 so the browser treats it as same-origin; validating Host
# at the app layer rejects it. See GHSA-ppp5-vxwm-4cf7.
_LOOPBACK_HOST_VALUES: frozenset = frozenset({"localhost", "127.0.0.1", "::1"})


def _dashboard_public_hosts() -> frozenset[str]:
    """Return the exact hostname declared by ``dashboard.public_url``.

    One source of truth for OAuth redirects, Host and WS Origin validation.
    Malformed or unset values fail closed as an empty set.
    """
    from hermes_cli.dashboard_auth.prefix import resolve_public_url

    public_url = resolve_public_url()
    try:
        hostname = urllib.parse.urlparse(public_url).hostname if public_url else None
    except ValueError:
        hostname = None
    return frozenset({hostname.lower()}) if hostname else frozenset()


def should_require_auth(host: str, allow_public: bool = False) -> bool:
    """True iff the auth gate must be active: any non-loopback bind.

    RFC1918 / CGNAT / link-local are deliberately PUBLIC — a hostile LAN device
    is the threat model. ``allow_public`` (legacy ``--insecure``) is accepted for
    old launch scripts but IGNORED since the June 2026 hermes-0day campaign.
    """
    return host not in _LOOPBACK_HOST_VALUES


def should_require_dashboard_auth(
    host: str,
    trusted_public_hosts: Optional[frozenset[str]] = None,
) -> bool:
    """Gate required for a non-loopback bind OR a non-loopback ``dashboard.public_url``.

    Callers may pass the already-resolved host set so startup and request
    validation share one snapshot.
    """
    if trusted_public_hosts is None:
        trusted_public_hosts = _dashboard_public_hosts()
    return should_require_auth(host) or any(h not in _LOOPBACK_HOST_VALUES for h in trusted_public_hosts)


def _desktop_loopback_auth_exempt(
    host: str,
    ssh_session_token: Optional[str] = None,
    ssh_owner_nonce: Optional[str] = None,
) -> bool:
    """True for a Desktop-owned loopback backend (#96490).

    A non-loopback ``dashboard.public_url`` would otherwise engage the
    ticket-only gate for the private loopback backends Desktop spawns, whose
    per-spawn session token the gate's WS path refuses — Desktop could not boot.
    The public dashboard is a separate non-loopback process that stays gated, so
    this never opens the public surface. Requires ALL of: loopback bind,
    ``HERMES_DESKTOP=1``, and an operator-minted credential (env token, SSH
    session token, or owner nonce).
    """
    return (
        host in _LOOPBACK_HOST_VALUES
        and os.environ.get("HERMES_DESKTOP") == "1"
        and bool(os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or ssh_session_token or ssh_owner_nonce)
    )


def _host_header_hostname(host_header: str) -> str:
    """Return a normalized hostname from a valid HTTP Host authority.

    Host headers are authorities, not full URLs. Reject ambiguous ports,
    malformed IPv6 brackets, and URL syntax so validation always fails closed.
    """
    value = (host_header or "").strip()
    if not value or "://" in value or any(c in value for c in '"\'<> \n\r\t/?#@'):
        return ""

    if value.startswith("["):
        close = value.find("]")
        if close == -1:
            return ""
        hostname = value[1:close]
        # Bracket notation is reserved for IPv6 literals.
        if ":" not in hostname:
            return ""
        suffix = value[close + 1:]
        if suffix and not re.fullmatch(r":\d+", suffix):
            return ""
        return hostname.lower()

    # Unbracketed IPv6 authorities are ambiguous with a port separator.
    if value.count(":") > 1:
        return ""
    if ":" in value:
        hostname, port = value.rsplit(":", 1)
        if not hostname or not port.isdigit():
            return ""
        return hostname.lower()
    return value.lower()


def _is_accepted_host(
    host_header: str,
    bound_host: str,
    trusted_public_hosts: frozenset[str] = frozenset(),
) -> bool:
    """True if the Host header targets the interface we bound to.

    Accepts:
    - Exact bound host (with or without port suffix)
    - Loopback aliases when bound to loopback
    - Exact operator-declared public hosts (with or without port suffix)
    - Any host when bound to 0.0.0.0 (explicit opt-in to non-loopback,
      no protection possible at this layer)
    """
    host_only = _host_header_hostname(host_header)
    if not host_only:
        return False
    # All-interfaces bind: no Host-layer defence is possible; rely on operator
    # network controls.
    if host_only in trusted_public_hosts or bound_host in {"0.0.0.0", "::"}:
        return True
    bound_lc = bound_host.lower()
    if bound_lc in _LOOPBACK_HOST_VALUES:
        return host_only in _LOOPBACK_HOST_VALUES
    return host_only == bound_lc


@app.middleware("http")
async def host_header_middleware(request: Request, call_next):
    """Reject requests whose Host header doesn't match the bound interface (DNS rebinding, GHSA-ppp5-vxwm-4cf7)."""
    # app.state.bound_host is set by start_server() at listen time.
    bound_host = getattr(app.state, "bound_host", None)
    if bound_host and not _is_accepted_host(
        request.headers.get("host", ""), bound_host, getattr(app.state, "trusted_public_hosts", frozenset())
    ):
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    "Invalid Host header. Dashboard requests must use the "
                    "bound hostname or the configured public hostname."
                ),
            },
        )
    return await call_next(request)


@app.middleware("http")
async def _plugin_api_runtime_gate(request: Request, call_next):
    """Block requests to disabled plugin API routes at request time.

    :func:`_mount_plugin_api_routes` gates at import time; a plugin disabled
    while running keeps its router mounted until restart, so enforce on every
    ``/api/plugins/{name}/...`` request. Registered BEFORE the auth middlewares
    (runs AFTER them): an unauthenticated caller must get auth's 401, never this
    404, or the status code becomes a plugin-name oracle.
    """
    path = request.url.path
    # parts: ['', 'api', 'plugins', '<name>', ...]
    parts = path.split("/")
    plugin_name = parts[3] if path.startswith("/api/plugins/") and len(parts) >= 4 else ""
    # Only gate authenticated requests. Unauthenticated ones fall through so
    # auth_middleware / the OAuth gate return 401 first and this route can't
    # be used as a plugin-name oracle.
    if plugin_name and (
        getattr(request.state, "token_authenticated", False)
        or getattr(request.app.state, "auth_required", False)
        or _has_valid_session_token(request)
        or _has_valid_query_token(request, path)
    ):
        try:
            # Gate: only serve user plugins that are in plugins.enabled and not in plugins.disabled. This
            # prevents the frontend from loading JS/CSS from plugins the user has not explicitly activated.
            # (#46435)
            from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
            enabled_set = _get_enabled_set()
            disabled_set = _get_disabled_set()
        except Exception:
            enabled_set = set()
            disabled_set = set()
        # Source from the cached plugin list; unknown => user plugin (safe default — blocks).
        plugin = next((p for p in _get_dashboard_plugins() if p.get("name") == plugin_name), None)
        source = plugin.get("source") if plugin else "user"
        blocked = plugin_name in disabled_set or (source == "user" and plugin_name not in enabled_set)
        if blocked and source in ("user", "bundled"):
            return JSONResponse(status_code=404, content={"detail": "Plugin not found"})
    return await call_next(request)


@app.middleware("http")
async def _dashboard_auth_gate(request: Request, call_next):
    """OAuth gate — active only when start_server flags ``auth_required``; pass-through on loopback.

    Registered between host_header and auth_middleware: host check → cookie auth → token auth.
    """
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware
    return await gated_auth_middleware(request, call_next)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list.

    Skipped for requests the token-auth seam already authenticated
    (``token_authenticated``) and when the OAuth gate is active — cookie auth is
    then authoritative and the loopback-only token path must not override it.
    """
    path = request.url.path
    if (
        not getattr(request.state, "token_authenticated", False)
        and not getattr(request.app.state, "auth_required", False)
        and path.startswith("/api/")
        and path not in _PUBLIC_API_PATHS
        and not path.startswith("/api/mcp/oauth/callback/")
        and not _has_valid_session_token(request)
        and not _has_valid_query_token(request, path)
    ):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.middleware("http")
async def _token_auth_seam(request: Request, call_next):
    """Outermost auth seam: bearer-token auth for opted-in routes (registered LAST = runs FIRST).

    A registered token route is owned here — authenticate, attach the principal
    + ``token_authenticated`` so downstream gates skip enforcement. Non-token
    routes pass through untouched.
    """
    from hermes_cli.dashboard_auth.token_auth import token_auth_middleware
    return await token_auth_middleware(request, call_next)


_DASHBOARD_HEALTH_WINDOW_SECONDS = 300.0


class DashboardHealth:
    """Dashboard-process health: rolling unhandled-error/5xx window + periodic self-test result.

    Feeds ``components`` on the PUBLIC ``/api/status``, so :meth:`snapshot`
    exports counts and enums only — never ``last_error_type``/``last_error_path``.
    """

    def __init__(self, window_seconds: float = _DASHBOARD_HEALTH_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._error_times: "deque[float]" = deque(maxlen=256)
        self.last_error_type: Optional[str] = None
        self.last_error_path: Optional[str] = None  # internal-only, never serialized
        self.last_error_at: Optional[float] = None
        self.selftest_status: str = "unknown"  # unknown | ok | failing
        self.selftest_http_status: Optional[int] = None
        self.selftest_at: Optional[float] = None

    def record_error(self, exc_type: str, path: str) -> None:
        now = time.time()
        self._error_times.append(now)
        self.last_error_type = exc_type
        self.last_error_path = path
        self.last_error_at = now

    def record_selftest(self, passed: bool, http_status: Optional[int]) -> None:
        self.selftest_status = "ok" if passed else "failing"
        self.selftest_http_status = http_status
        self.selftest_at = time.time()

    def recent_error_count(self) -> int:
        cutoff = time.time() - self.window_seconds
        while self._error_times and self._error_times[0] < cutoff:
            self._error_times.popleft()
        return len(self._error_times)

    def snapshot(self) -> Dict[str, Any]:
        """Public component payload: status enum + counts + timestamps only."""
        errors = self.recent_error_count()
        status = "degraded" if (errors or self.selftest_status == "failing") else "ok"
        return {
            "status": status,
            "recent_unhandled_errors": errors,
            "last_error_at": self.last_error_at,
            "selftest": self.selftest_status,
        }


DASHBOARD_HEALTH = DashboardHealth()


@app.middleware("http")
async def _dashboard_health_middleware(request: Request, call_next):
    """Outermost middleware (registered last): count unhandled exceptions and 5xx; re-raises, never alters."""
    try:
        response = await call_next(request)
    except Exception as exc:
        DASHBOARD_HEALTH.record_error(type(exc).__name__, request.url.path)
        raise
    if response.status_code >= 500:
        DASHBOARD_HEALTH.record_error(f"http_{response.status_code}", request.url.path)
    return response


# Authenticated-route self-test: one in-process request per minute against a
# cheap DB-touching route, catching "liveness fine but every authed request 500s".
_DASHBOARD_SELFTEST_INTERVAL_SECONDS = 60.0
_DASHBOARD_SELFTEST_ROUTE = "/api/sessions?limit=1"


async def _dashboard_selftest_once() -> None:
    """Run one authenticated in-process self-test request and record it."""
    try:
        import httpx
    except ImportError:
        return  # optional dependency — leave status "unknown"
    try:
        # Loopback base_url so the Host-header middleware accepts the request.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
            resp = await client.get(_DASHBOARD_SELFTEST_ROUTE, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
        DASHBOARD_HEALTH.record_selftest(resp.status_code == 200, resp.status_code)
    except Exception:
        DASHBOARD_HEALTH.record_selftest(False, None)


async def _dashboard_selftest_loop() -> None:
    """Periodic self-test driver started from the lifespan."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        _log.debug("httpx unavailable — dashboard self-test disabled")
        return
    while True:
        await asyncio.sleep(_DASHBOARD_SELFTEST_INTERVAL_SECONDS)
        # OAuth-gated binds don't honour the session token; the probe would false-alarm 401.
        if getattr(app.state, "auth_required", False):
            continue
        await _dashboard_selftest_once()




# Action registries/spawner are owned by web_server_gateway; routers and tests reach them
# there, so this module reads them through the module too (one patch seam).
from hermes_cli import web_server_gateway as _gateway_mod  # noqa: E402
from hermes_cli.web_server_gateway import _ACTION_LOG_FILES, _terminate_desktop_managed_gateway  # noqa: E402
from hermes_cli.web_server_sessions import _auto_archive_ticker_loop  # noqa: E402
from hermes_cli.web_server_chat import PTY_REGISTRY  # noqa: E402
from hermes_cli.web_server_dashboard import (  # noqa: E402
    _discover_dashboard_plugins, _mount_plugin_api_routes, mount_spa,
)


_GATEWAY_HEALTH_URL = os.getenv("GATEWAY_HEALTH_URL")
_GATEWAY_HEALTH_TIMEOUT_MAX = 1.0
try:
    _GATEWAY_HEALTH_TIMEOUT = float(os.getenv("GATEWAY_HEALTH_TIMEOUT", "1"))
except (ValueError, TypeError):
    _log.warning(
        "Invalid GATEWAY_HEALTH_TIMEOUT value %r — using default 1.0s",
        os.getenv("GATEWAY_HEALTH_TIMEOUT"),
    )
    _GATEWAY_HEALTH_TIMEOUT = 1.0
if _GATEWAY_HEALTH_TIMEOUT <= 0:
    _log.warning(
        "Invalid non-positive GATEWAY_HEALTH_TIMEOUT value %.3fs — using default 1.0s",
        _GATEWAY_HEALTH_TIMEOUT,
    )
    _GATEWAY_HEALTH_TIMEOUT = 1.0
elif _GATEWAY_HEALTH_TIMEOUT > _GATEWAY_HEALTH_TIMEOUT_MAX:
    _log.warning(
        "Capping GATEWAY_HEALTH_TIMEOUT %.3fs to %.3fs for dashboard liveness probes",
        _GATEWAY_HEALTH_TIMEOUT,
        _GATEWAY_HEALTH_TIMEOUT_MAX,
    )
    _GATEWAY_HEALTH_TIMEOUT = _GATEWAY_HEALTH_TIMEOUT_MAX


_MANAGED_FILE_MAX_BYTES = 100 * 1024 * 1024
_FS_DATA_URL_MAX_BYTES = 16 * 1024 * 1024
# Multipart uploads stream to a temp file in fixed chunks and rename into
# place: constant memory, no base64 inflation, no proxy body-size 502s (NS-501).
_UPLOAD_CHUNK_BYTES = 1024 * 1024

# Stable install identity for /api/status: one uuid4 hex per physical install,
# persisted under the ROOT Hermes home (not the profile HERMES_HOME) so every
# profile reports the same id and the desktop can collapse duplicate roster rows
# for one backend. Must never change across restarts, so cached per process.
_INSTALL_ID_CACHE: Dict[str, Optional[str]] = {"root": None, "value": None}


def get_install_id() -> Optional[str]:
    """Process-lifetime-cached stable install id."""
    return _shared_get_install_id(cache=_INSTALL_ID_CACHE)


# Serializes config.yaml read-modify-write cycles for handlers on worker threads
# (asyncio.to_thread): config.py's _CONFIG_LOCK covers each load/save call, not
# the span between them, so two off-loop updates could drop each other's writes.
# RLock so nested helpers that also take it can't self-deadlock.
_CONFIG_MUTATION_LOCK = threading.RLock()

# A finished ``gateway-restart`` child does not mean the gateway is back (it
# exits once the restart is handed off), so in-flight reuse stops coalescing
# exactly when a stale frontend re-fires every few seconds (#89034: 77 restarts,
# state.db corrupted mid-FTS5-write). MAINTAINER DECISION: a fixed window, not
# "until healthy" — a gateway that never returns must not leave the action
# inert. 10s is above the ~3.5s storm spacing and below an operator's retry.
GATEWAY_RESTART_COOLDOWN_SECONDS = 10.0

# ``(monotonic spawn time, Popen, command)`` of the last restart. Deliberately
# NOT read from ``_ACTION_PROCS``: entries there vanish when the child exits.
_LAST_GATEWAY_RESTART: Optional[Tuple[float, subprocess.Popen, Tuple[str, ...]]] = None


def _spawn_gateway_restart(profile: Optional[str] = None) -> Tuple[subprocess.Popen, bool]:
    """Spawn ``hermes gateway restart``, reusing an in-flight or recent restart.

    Concurrent children race each other on the kill-and-start path, so a live
    child is reused; requests within ``GATEWAY_RESTART_COOLDOWN_SECONDS`` for the
    same profile coalesce onto the last spawn too (#89034). Orphaned gateways
    are reaped first so the fresh one doesn't stack a duplicate (#77276).
    Returns ``(proc, reused)``.
    """
    try:
        from hermes_cli.gateway import _reap_unsupervised_gateway_orphans

        _reap_unsupervised_gateway_orphans()
    except Exception:
        pass  # best-effort — don't block the restart on a reap failure

    global _LAST_GATEWAY_RESTART

    subcommand = _gateway_mod._gateway_subcommand(profile, "restart")
    existing = _gateway_mod._ACTION_PROCS.get("gateway-restart")
    if existing is not None and existing.poll() is None:
        existing_command = _gateway_mod._ACTION_COMMANDS.get("gateway-restart")
        if existing_command is None or existing_command == tuple(subcommand):
            return existing, True
        raise RuntimeError("gateway restart already in progress for another profile")

    recent = _LAST_GATEWAY_RESTART
    if recent is not None:
        spawned_at, recent_proc, recent_command = recent
        age = time.monotonic() - spawned_at if recent_command == tuple(subcommand) else None
        if age is not None and age < GATEWAY_RESTART_COOLDOWN_SECONDS:
            _log.info(
                "Coalescing gateway restart: one was started %.1fs ago "
                "(pid %s) and the gateway may still be coming back; not "
                "spawning another (#89034).",
                age,
                getattr(recent_proc, "pid", "?"),
            )
            return recent_proc, True

    proc = _gateway_mod._spawn_hermes_action(subcommand, "gateway-restart")
    _LAST_GATEWAY_RESTART = (time.monotonic(), proc, tuple(subcommand))
    return proc, False


# Collapses repeated identical ElevenLabs voice-list failures (the desktop
# re-polls on every settings focus) to one log line; re-arms on success or a
# changed signature.
_voice_list_last_error: Optional[str] = None


def _voice_list_error_logged_once(signature: Optional[str]) -> bool:
    """True if ``signature`` is new and should be logged now; ``None`` clears the latch."""
    global _voice_list_last_error
    if signature is None:
        _voice_list_last_error = None
        return False
    if signature == _voice_list_last_error:
        return False
    _voice_list_last_error = signature
    return True


_ACTION_LOG_FILES.setdefault("computer-use-grant", "action-computer-use-grant.log")

# Cache discovered plugins per-process (refresh on explicit re-scan).
_dashboard_plugins_cache: Optional[list] = None


def _get_dashboard_plugins(force_rescan: bool = False) -> list:
    global _dashboard_plugins_cache
    stale = _dashboard_plugins_cache is None or force_rescan or any(
        not Path(p["_dir"]).is_dir() for p in _dashboard_plugins_cache
    )
    if stale:
        _dashboard_plugins_cache = _discover_dashboard_plugins()
    return _dashboard_plugins_cache


# Router mounting. ORDER IS ROUTE-MATCHING ORDER: literal paths must land before
# templated siblings (e.g. /api/sessions/bulk-delete before /api/sessions/{id}).
from hermes_cli.web_routers import (  # noqa: E402
    files as _files_routes,
    git as _git_routes,
    local_models as _local_models_routes,
    status as _status_routes,
    actions as _actions_routes,
    audio as _audio_routes,
    sessions as _sessions_routes,
    profiles as _profiles_routes,
    memory_providers as _memory_providers_routes,
    config_env as _config_env_routes,
    models as _models_routes,
    messaging as _messaging_routes,
    oauth as _oauth_routes,
    cron as _cron_routes,
    mcp as _mcp_routes,
    ops as _ops_routes,
    skills as _skills_routes,
    tools as _tools_routes,
    analytics as _analytics_routes,
    chat_ws as _chat_ws_routes,
    dashboard_ui as _dashboard_ui_routes,
)

app.include_router(_files_routes.router)
app.include_router(_git_routes.router)
app.include_router(_local_models_routes.router)
app.include_router(_status_routes.router)
app.include_router(_actions_routes.router)
app.include_router(_audio_routes.router)
app.include_router(_actions_routes.status_router)
app.include_router(_sessions_routes.list_router)
app.include_router(_profiles_routes.sessions_router)
app.include_router(_sessions_routes.search_router)
app.include_router(_memory_providers_routes.router)
app.include_router(_config_env_routes.config_router)
app.include_router(_models_routes.router)
app.include_router(_config_env_routes.router)
app.include_router(_messaging_routes.router)
app.include_router(_oauth_routes.router)
app.include_router(_sessions_routes.manage_router)
app.include_router(_status_routes.logs_router)
app.include_router(_cron_routes.router)
app.include_router(_mcp_routes.router)
app.include_router(_ops_routes.router)
app.include_router(_skills_routes.hub_router)
app.include_router(_profiles_routes.router)
app.include_router(_skills_routes.router)
app.include_router(_tools_routes.router)
app.include_router(_analytics_routes.router)
app.include_router(_chat_ws_routes.router)
app.include_router(_dashboard_ui_routes.router)

# Plugin API routes and the dashboard auth routes (/login, /auth/*, /api/auth/*)
# mount before the SPA catch-all so /{full_path:path} doesn't swallow them. Auth
# routes are always mounted — the gate middleware decides enforcement.
_mount_plugin_api_routes()
from hermes_cli.dashboard_auth.routes import router as _dashboard_auth_router  # noqa: E402

app.include_router(_dashboard_auth_router)
mount_spa(app)


def _no_auth_provider_message(host: str) -> str:
    """Actionable SystemExit text for a gated bind with no registered auth provider.

    Names the exact trigger: on a loopback bind the ONLY trigger is
    dashboard.public_url, so print the offending URL and the remove-it exit.
    Bundled providers expose ``LAST_SKIP_REASON`` so an installed-but-
    unconfigured provider is not reported as merely "no providers".
    """
    skip_reasons: list[str] = []
    try:
        from plugins.dashboard_auth import nous as _nous_plugin

        if _nous_plugin.LAST_SKIP_REASON:
            skip_reasons.append(f"  • nous: {_nous_plugin.LAST_SKIP_REASON}")
    except Exception:
        pass

    if host in _LOOPBACK_HOST_VALUES:
        public_url = ""
        try:
            from hermes_cli.dashboard_auth.prefix import resolve_public_url

            public_url = resolve_public_url()
        except Exception:
            pass
        gate_reason = (
            f"dashboard.public_url is set to "
            f"{public_url or '<a non-loopback URL>'} — an "
            f"operator-declared external URL engages the auth gate "
            f"even on a loopback bind"
        )
        fix_hint = (
            "If this dashboard should be LOCAL-ONLY (no reverse "
            "proxy), remove dashboard.public_url from config.yaml "
            "(and unset HERMES_DASHBOARD_PUBLIC_URL) to restore the "
            "unauthenticated loopback mode.\n"
        )
    else:
        gate_reason = f"the auth gate engages on non-loopback binds ({host})"
        fix_hint = ""

    fix_hint += (
        "Configure an auth provider before exposing the dashboard:\n"
        "  • Password: set dashboard.basic_auth.username + "
        "password_hash in config.yaml\n"
        "    (hash with: python -c \"from "
        "plugins.dashboard_auth.basic import hash_password; "
        "print(hash_password('your-password'))\")\n"
        "  • OAuth: run `hermes dashboard register` (Nous Portal) or "
        "install a DashboardAuthProvider plugin.\n"
        "There is no unauthenticated public-dashboard option. For "
        "local-only use, bind 127.0.0.1 and leave dashboard.public_url "
        "unset; a configured external public URL requires auth even "
        "when a local reverse proxy reaches a loopback backend."
    )
    # Credentials exist but the bundled provider is disabled (#54489). Basic
    # auth needs a username AND a credential; a half-configured block is silent.
    try:
        from hermes_cli.config import load_config as _load_cfg
        from hermes_cli.plugins_cmd import _BASIC_AUTH_PLUGIN_KEYS

        cfg = _load_cfg()
        ba = (cfg.get("dashboard") or {}).get("basic_auth") or {}
        disabled = (cfg.get("plugins") or {}).get("disabled") or []
        has_creds = bool(ba.get("username")) and bool(ba.get("password_hash") or ba.get("password"))
        if has_creds and (set(disabled) & _BASIC_AUTH_PLUGIN_KEYS):
            fix_hint = (
                "The 'basic' dashboard-auth plugin is in "
                "plugins.disabled but dashboard.basic_auth is "
                "configured.\n"
                "Remove 'basic' from plugins.disabled (or run "
                "`hermes plugins enable basic`), then restart the "
                "dashboard.\n\n"
            ) + fix_hint
    except Exception:
        pass
    msg = (
        f"Refusing to bind dashboard to {host} — {gate_reason}, "
        f"but no auth providers are registered.\n\n"
    )
    if skip_reasons:
        msg += "Bundled providers reported these issues:\n" + "\n".join(skip_reasons) + "\n\n"
    return msg + fix_hint


def _configure_auth_gate(
    host: str,
    allow_public: bool,
    ssh_session_token: Optional[str],
    ssh_owner_nonce: Optional[str],
) -> None:
    """Resolve the trusted public hosts + auth-gate flag onto ``app.state``.

    Fails closed (``SystemExit`` with an actionable message) when the gate
    engages but no dashboard auth provider is registered.
    """
    # dashboard.public_url is also the exact Host/Origin trust declaration for
    # reverse-proxy deployments; resolved once so middleware never reloads
    # config. A non-loopback public hostname engages the gate even on a loopback
    # backend, else the SPA's local session token becomes remotely reachable.
    app.state.trusted_public_hosts = _dashboard_public_hosts()
    # auth_required drives middleware, SPA-token injection, WS auth, the
    # startup refusal, the gate-on banner and uvicorn proxy_headers.
    if _desktop_loopback_auth_exempt(host, ssh_session_token, ssh_owner_nonce):
        # public_url describes the operator's PUBLIC deployment, not this
        # Desktop-owned loopback backend (#96490), which authenticates with the
        # per-spawn session token the ticket-only gate would refuse.
        app.state.auth_required = should_require_auth(host)
        _log.info(
            "Desktop-owned loopback backend: dashboard.public_url does not "
            "engage the ticket gate for this process; the public deployment "
            "keeps its own gate.",
        )
    else:
        app.state.auth_required = should_require_dashboard_auth(host, app.state.trusted_public_hosts)

    # ``--insecure`` no longer disables the gate (June 2026 hermes-0day
    # hardening); warn that it is a no-op rather than silently ignore it.
    if allow_public and host not in _LOOPBACK_HOST_VALUES:
        _log.warning(
            "--insecure no longer bypasses dashboard authentication. A "
            "non-loopback bind (%s) now ALWAYS requires an auth provider "
            "(OAuth or the bundled password provider). Configure one — see "
            "below — or bind to 127.0.0.1 and reach it over an SSH tunnel / "
            "Tailscale.", host,
        )

    if app.state.auth_required:
        # No escape hatch serves a gated dashboard without a provider.
        from hermes_cli.dashboard_auth import list_providers
        if not list_providers():
            raise SystemExit(_no_auth_provider_message(host))
        _log.info(
            "Dashboard binding to %s with auth gate enabled. Providers: %s",
            host,
            ", ".join(p.name for p in list_providers()),
        )


def _build_uvicorn_server(host: str, port: int):
    """Build the uvicorn ``Config`` + ``Server`` for this bind (reads ``app.state.auth_required``).

    uvicorn.Server is driven directly (not uvicorn.run) so startup is split from
    the main loop: after startup() the socket is bound and held by uvicorn, so the
    OS-assigned port can be read with no pre-bind-then-close TOCTOU. Explicit
    taken ports are caught by the #93608 preflight probe; uvicorn's own bind
    error stays the fallback for races.
    """
    import uvicorn

    # WS keepalive ping runs ON the agent event loop; a GIL-holding worker call
    # can starve it for minutes, so uvicorn misses the pong and drops a healthy
    # local socket (#53773/#48445/#50005). The ping only detects half-open
    # connections (proxy 524, dropped tunnels), impossible on loopback where a
    # dead client sends a real FIN/RST -> WebSocketDisconnect. So: no ping on
    # loopback; non-loopback sits behind a Cloudflare Tunnel (~100s idle) and
    # keeps a config-driven cadence (dashboard.ws_ping_interval/_timeout,
    # #79635) defaulting to 20/20.
    _is_loopback = host in _LOOPBACK_HOST_VALUES
    try:
        _dash_cfg = load_config().get("dashboard") or {}
    except Exception:
        _dash_cfg = {}

    def _ws_ping_setting(key: str, default: float = 20.0) -> float:
        try:
            return float(_dash_cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning",
        # Off by default so _ws_client_is_allowed sees the real peer, not
        # X-Forwarded-For. Gated mode runs behind a TLS terminator and needs
        # X-Forwarded-Proto for cookie Secure flags.
        proxy_headers=bool(app.state.auth_required),
        # Loopback-only unless the operator trusts a bounded upstream proxy, so
        # spoofed X-Forwarded-* from arbitrary callers is never honoured.
        forwarded_allow_ips=_dashboard_forwarded_allow_ips(_dash_cfg),
        ws_ping_interval=None if _is_loopback else _ws_ping_setting("ws_ping_interval"),
        ws_ping_timeout=None if _is_loopback else _ws_ping_setting("ws_ping_timeout"),
        ws_max_size=_DESKTOP_ATTACHMENT_WS_MAX_BYTES,
    )
    return config, uvicorn.Server(config)


def _best_effort(what: str, fn) -> None:
    """Run a best-effort startup step; any failure (import included) is a debug line."""
    try:
        fn()
    except Exception as exc:
        _log.debug("%s skipped: %s", what, exc)


def _on_server_started(
    server,
    *,
    host: str,
    port: int,
    headless: bool,
    open_browser: bool,
    initial_profile: str,
    start_mcp_discovery_after_bind: bool,
) -> None:
    """Post-bind arming on the serving loop right after ``server.startup()``.

    Reap prior corpses, parent-death watchdog, process identity, READY
    announcement, browser open, deferred MCP discovery, loop-noise filter,
    loop heartbeat.
    """
    # Clear corpses from a previous unclean Desktop exit (crash/SIGKILL/update
    # handoff leaves an orphaned backend + its MCP subtree) before stacking a
    # new tree (EMFILE / missing tabs). The watchdog only protects *this*
    # process going forward.
    def _reap_desktop_serves() -> None:
        from hermes_cli.dashboard_procs import _reap_orphaned_desktop_local_serves

        _reap_orphaned_desktop_local_serves()

    def _reap_mcp_helpers() -> None:
        from hermes_cli.process_identity import reap_orphaned_mcp_helpers

        reap_orphaned_mcp_helpers()

    if os.getenv("HERMES_DESKTOP") == "1":
        _best_effort("orphan desktop-local serve reap", _reap_desktop_serves)
    # Same sweep for stdio MCP helpers (#61514): positive identity only (spawn
    # ledger + spawner provably dead); anything alive or unprovable is untouched.
    _best_effort("orphan MCP helper reap", _reap_mcp_helpers)

    # No-op for standalone `hermes serve` (no HERMES_PARENT_PID).
    _start_parent_death_watchdog()

    actual_port = _read_bound_port(server, fallback=port)
    app.state.bound_port = actual_port

    # Positive process identity in the machine spawn ledger (+ Windows
    # kill-on-close job). Registered AFTER the bind so the entry carries the
    # ACTUAL port — what lets `hermes update` relaunch a manually-started serve
    # on its real endpoint (#63206).
    def _register_identity() -> None:
        from hermes_cli.process_identity import attach_self_to_kill_on_close_job, register_self

        register_self(
            "serve" if headless else "dashboard",
            detail={"host": host, "port": actual_port, "profile": initial_profile or ""},
        )
        attach_self_to_kill_on_close_job()

    _best_effort("process-identity registration", _register_identity)

    _write_dashboard_ready_file(actual_port, _SESSION_TOKEN)
    # Port-discovery sentinel parsed by the Desktop spawn (matches either
    # token). Written to fd 1: tui_gateway.server redirects sys.stdout to
    # stderr at import, and the Desktop watches child.stdout (#96282).
    ready_token = "HERMES_BACKEND_READY" if headless else "HERMES_DASHBOARD_READY"
    _write_machine_sentinel_line(f"{ready_token} port={actual_port}")
    if headless:
        # Auth-gated JSON-RPC/WS only — announce the bind, not a URL. flush:
        # a piped stdout otherwise surfaces this minutes after the sentinel.
        print(f"  Hermes backend listening on {host}:{actual_port}", flush=True)
    else:
        print(f"  Hermes Web UI → http://{host}:{actual_port}")
    _maybe_open_browser(host, actual_port, open_browser, initial_profile)

    if start_mcp_discovery_after_bind:
        # Desktop `serve`: the ~350ms `mcp` SDK import holds the GIL while the
        # renderer does its WS handshake + first hydration reads, so arm it one
        # second later when the shell is painted and idle. An agent build inside
        # that second fires the deferred start itself (wait_for_mcp_discovery).
        try:
            from hermes_cli.mcp_startup import defer_background_mcp_discovery

            defer_background_mcp_discovery(
                logger=_log,
                thread_name="dashboard-mcp-discovery",
                delay=_DESKTOP_MCP_DISCOVERY_DELAY_S,
            )
        except Exception:
            _log.debug("Deferred MCP discovery arm failed", exc_info=True)

    # Collapse the peer-hangup teardown flood (#50005): 50+ identical WinError
    # 10054 tracebacks per Desktop disconnect become one debug line.
    def _install_noise_filter() -> None:
        from tui_gateway.loop_noise import install_loop_noise_filter

        install_loop_noise_filter(asyncio.get_running_loop())

    _best_effort("loop noise filter install", _install_noise_filter)

    # Loop heartbeat watchdog (CF-1): a 2s call_later tick whose drift equals
    # any GIL stall, so a stalled-loop WS drop is diagnosable from the log.
    # call_later (not a task) dies with the loop — nothing to cancel.
    _hb_interval = 2.0
    _hb_stall_threshold = 5.0
    _hb_loop = asyncio.get_running_loop()

    def _loop_heartbeat(expected: float) -> None:
        now = _hb_loop.time()
        drift = now - expected
        if drift > _hb_stall_threshold:
            _log.warning("event loop stalled %.1fs (GIL pressure suspected)", drift)
        _hb_loop.call_later(_hb_interval, _loop_heartbeat, now + _hb_interval)

    _hb_loop.call_later(_hb_interval, _loop_heartbeat, _hb_loop.time() + _hb_interval)


def _run_serve(serve, config, host: str, port: int) -> None:
    """Drive ``serve()`` on the loop uvicorn expects.

    POSIX keeps ``asyncio.run`` (already a SelectorEventLoop / uvloop). On
    Windows ``asyncio.run`` defaults to a ProactorEventLoop, on which uvicorn
    binds a socket that never accepts (#50641), so mirror uvicorn's own runner +
    loop factory there (hand-installed selector policy for uvicorn < 0.36).
    Ctrl+C -> clean return; probe-to-bind port race -> sentinel + exit code.
    """
    runner = asyncio.run
    runner_kwargs: dict = {}
    if sys.platform == "win32":
        # Resolved FIRST; the serve call is outside this try so genuine
        # serve-time errors (port in use) propagate instead of double-running.
        try:
            from uvicorn._compat import asyncio_run as runner

            runner_kwargs = {"loop_factory": config.get_loop_factory()}
        except Exception:
            runner = asyncio.run
            runner_kwargs = {}
            try:
                asyncio.set_event_loop_policy(
                    asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]
                )
            except Exception:
                pass

    # ``capture_signals()`` re-raises the captured signal after graceful
    # shutdown; console Ctrl+C lands as KeyboardInterrupt = clean exit.
    # (Re-raised SIGTERM/SIGBREAK keep their terminate disposition.)
    try:
        runner(serve(), **runner_kwargs)
    except KeyboardInterrupt:
        return
    except SystemExit as exc:
        # Probe-to-bind race (#93608): uvicorn's bind_socket() exits 1 — re-check
        # and translate a confirmed conflict into the sentinel + distinct code.
        if exc.code == 1 and _port_bind_conflict(host, port):
            _report_port_in_use(host, port)
            raise SystemExit(PORT_IN_USE_EXIT_CODE) from None
        raise


def start_server(
    host: str = "127.0.0.1",
    port: int = 9119,
    open_browser: bool = True,
    allow_public: bool = False,
    initial_profile: str = "",
    headless: bool = False,
    ssh_session_token: Optional[str] = None,
    ssh_owner_nonce: Optional[str] = None,
    start_mcp_discovery_after_bind: bool = False,
):
    """Start the web UI server.

    ``initial_profile`` is appended to the auto-opened URL as ``?profile=<name>``
    (profile alias ``<profile> dashboard``). ``headless`` is the ``serve`` path:
    JSON-RPC/WS backend, no UI build, no SPA mount (``HERMES_SERVE_HEADLESS``).
    ``ssh_session_token``/``ssh_owner_nonce`` are process-local Desktop SSH
    bootstrap state, never persisted or exported to children.
    ``start_mcp_discovery_after_bind`` (Desktop ``serve``) defers MCP discovery
    until the ready sentinel is written so its SDK import can't hold the GIL
    against the pre-bind path.
    """
    _apply_ssh_session_token(ssh_session_token or "")
    _apply_ssh_owner_nonce(ssh_owner_nonce)

    # Dashboard-mode starts don't route through main.py's `serve` path, which
    # applies the same RLIMIT_NOFILE floor (policy in resource_limits, #81547).
    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    import uvicorn  # noqa: F401 — fail fast (before any side effects) when the dashboard extra is missing

    try:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive

        start_nous_auth_keepalive()
    except Exception as exc:
        _log.debug("Nous auth keepalive did not start: %s", exc)

    _configure_auth_gate(host, allow_public, ssh_session_token, ssh_owner_nonce)

    # host_header_middleware validates Host against this (DNS rebinding,
    # GHSA-ppp5-vxwm-4cf7).
    app.state.bound_host = host

    config, server = _build_uvicorn_server(host, port)

    # Flush-on-kill guard (#94724): chaining SIGTERM/SIGINT handlers persist
    # in-memory transcripts to state.db before shutdown. Installed BEFORE
    # uvicorn's capture_signals() so uvicorn re-raises into them as the
    # "original" handlers — kills outside the serve window are covered too.
    try:
        from tui_gateway.server import install_exit_flush_signal_handlers

        install_exit_flush_signal_handlers()
    except Exception as exc:
        _log.debug("exit-flush signal handlers not installed: %s", exc)

    # #93608: uvicorn's bind_socket() would exit 1 with a bare ERROR line,
    # indistinguishable from "backend broken". Probe first so a conflict
    # surfaces as the BACKEND_PORT_IN_USE sentinel + distinct exit code.
    # ``--port 0`` is skipped by the probe.
    if _port_bind_conflict(host, port):
        _report_port_in_use(host, port)
        raise SystemExit(PORT_IN_USE_EXIT_CODE)

    async def _serve():
        # startup split from main_loop so the bound (ephemeral) port is readable.
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        with server.capture_signals():
            await server.startup()
            if server.should_exit:
                return

            _on_server_started(
                server,
                host=host,
                port=port,
                headless=headless,
                open_browser=open_browser,
                initial_profile=initial_profile,
                start_mcp_discovery_after_bind=start_mcp_discovery_after_bind,
            )

            await server.main_loop()
            if server.started:
                await server.shutdown()

    _run_serve(_serve, config, host, port)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import List  # noqa: F401,E402
from typing import Literal  # noqa: F401,E402
import atexit  # noqa: F401,E402
import base64  # noqa: F401,E402
import binascii  # noqa: F401,E402
import concurrent.futures  # noqa: F401,E402
import contextlib  # noqa: F401,E402
from contextlib import contextmanager  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from datetime import datetime  # noqa: F401,E402
import functools  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import importlib.util  # noqa: F401,E402
import inspect  # noqa: F401,E402
import ipaddress  # noqa: F401,E402
import json  # noqa: F401,E402
import math  # noqa: F401,E402
import mimetypes  # noqa: F401,E402
import queue  # noqa: F401,E402
import shlex  # noqa: F401,E402
import shutil  # noqa: F401,E402
import stat  # noqa: F401,E402
import tempfile  # noqa: F401,E402
from datetime import timezone  # noqa: F401,E402
import yaml  # noqa: F401,E402
import zipfile  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'AudioTranscriptionRequest': ('hermes_cli.web_models', 'AudioTranscriptionRequest'),
    'AutomationBlueprintInstantiate': ('hermes_cli.web_models', 'AutomationBlueprintInstantiate'),
    'BackupRequest': ('hermes_cli.web_models', 'BackupRequest'),
    'BulkDeleteSessions': ('hermes_cli.web_models', 'BulkDeleteSessions'),
    'CONFIG_SCHEMA': ('hermes_cli.web_server_config', 'CONFIG_SCHEMA'),
    'ChatImageUpload': ('hermes_cli.web_models', 'ChatImageUpload'),
    'ConfigUpdate': ('hermes_cli.web_models', 'ConfigUpdate'),
    'CredentialPoolAdd': ('hermes_cli.web_models', 'CredentialPoolAdd'),
    'CronJobCreate': ('hermes_cli.web_models', 'CronJobCreate'),
    'CronJobUpdate': ('hermes_cli.web_models', 'CronJobUpdate'),
    'CuratorPause': ('hermes_cli.web_models', 'CuratorPause'),
    'CustomEndpointUpdate': ('hermes_cli.web_models', 'CustomEndpointUpdate'),
    'DEFAULT_CONFIG': ('hermes_cli.config', 'DEFAULT_CONFIG'),
    'DebugShareRequest': ('hermes_cli.web_models', 'DebugShareRequest'),
    'EnvVarDelete': ('hermes_cli.web_models', 'EnvVarDelete'),
    'EnvVarReveal': ('hermes_cli.web_models', 'EnvVarReveal'),
    'EnvVarUpdate': ('hermes_cli.web_models', 'EnvVarUpdate'),
    'FontSetBody': ('hermes_cli.web_models', 'FontSetBody'),
    'FsWriteText': ('hermes_cli.web_models', 'FsWriteText'),
    'GitBranchSwitchBody': ('hermes_cli.web_models', 'GitBranchSwitchBody'),
    'GitCommitBody': ('hermes_cli.web_models', 'GitCommitBody'),
    'GitFileBody': ('hermes_cli.web_models', 'GitFileBody'),
    'GitPathBody': ('hermes_cli.web_models', 'GitPathBody'),
    'GitWorktreeAddBody': ('hermes_cli.web_models', 'GitWorktreeAddBody'),
    'GitWorktreeRemoveBody': ('hermes_cli.web_models', 'GitWorktreeRemoveBody'),
    'HookCreate': ('hermes_cli.web_models', 'HookCreate'),
    'HookDelete': ('hermes_cli.web_models', 'HookDelete'),
    'ImportRequest': ('hermes_cli.web_models', 'ImportRequest'),
    'LearningNodeEdit': ('hermes_cli.web_models', 'LearningNodeEdit'),
    'LearningNodeRef': ('hermes_cli.web_models', 'LearningNodeRef'),
    'MCPCatalogInstall': ('hermes_cli.web_models', 'MCPCatalogInstall'),
    'MCPEnabledToggle': ('hermes_cli.web_models', 'MCPEnabledToggle'),
    'MCPServerCreate': ('hermes_cli.web_models', 'MCPServerCreate'),
    'MCPServersReplace': ('hermes_cli.web_models', 'MCPServersReplace'),
    'ManagedDirectoryCreate': ('hermes_cli.web_models', 'ManagedDirectoryCreate'),
    'ManagedFileDelete': ('hermes_cli.web_models', 'ManagedFileDelete'),
    'ManagedFileUpload': ('hermes_cli.web_models', 'ManagedFileUpload'),
    'ManagedFilesPolicy': ('hermes_cli.web_server_files', 'ManagedFilesPolicy'),
    'MemoryProviderConfigUpdate': ('hermes_cli.web_models', 'MemoryProviderConfigUpdate'),
    'MemoryProviderSelect': ('hermes_cli.web_models', 'MemoryProviderSelect'),
    'MemoryProviderSetupRequest': ('hermes_cli.web_models', 'MemoryProviderSetupRequest'),
    'MemoryReset': ('hermes_cli.web_models', 'MemoryReset'),
    'MessagingPlatformUpdate': ('hermes_cli.web_models', 'MessagingPlatformUpdate'),
    'MoaConfigPayload': ('hermes_cli.web_models', 'MoaConfigPayload'),
    'MoaModelSlot': ('hermes_cli.web_models', 'MoaModelSlot'),
    'MoaPresetPayload': ('hermes_cli.web_models', 'MoaPresetPayload'),
    'ModelAssignment': ('hermes_cli.web_models', 'ModelAssignment'),
    'OAuthSubmitBody': ('hermes_cli.web_models', 'OAuthSubmitBody'),
    'OPTIONAL_ENV_VARS': ('hermes_cli.config', 'OPTIONAL_ENV_VARS'),
    'PairingApprove': ('hermes_cli.web_models', 'PairingApprove'),
    'PairingRevoke': ('hermes_cli.web_models', 'PairingRevoke'),
    'ProfileActiveUpdate': ('hermes_cli.web_models', 'ProfileActiveUpdate'),
    'ProfileCreate': ('hermes_cli.web_models', 'ProfileCreate'),
    'ProfileDescribeAuto': ('hermes_cli.web_models', 'ProfileDescribeAuto'),
    'ProfileDescriptionUpdate': ('hermes_cli.web_models', 'ProfileDescriptionUpdate'),
    'ProfileModelUpdate': ('hermes_cli.web_models', 'ProfileModelUpdate'),
    'ProfileRename': ('hermes_cli.web_models', 'ProfileRename'),
    'ProfileSoulUpdate': ('hermes_cli.web_models', 'ProfileSoulUpdate'),
    'ProviderConfigSchema': ('plugins.memory.config_schema', 'ProviderConfigSchema'),
    'ProviderField': ('plugins.memory.config_schema', 'ProviderField'),
    'PtyBridge': ('hermes_cli.pty_bridge', 'PtyBridge'),
    'PtySessionRegistry': ('hermes_cli.pty_session', 'PtySessionRegistry'),
    'PtyUnavailableError': ('hermes_cli.pty_bridge', 'PtyUnavailableError'),
    'RawConfigUpdate': ('hermes_cli.web_models', 'RawConfigUpdate'),
    'RegistryFull': ('hermes_cli.pty_session', 'RegistryFull'),
    'STORAGE_HONCHO_HOST_BLOCK': ('plugins.memory.config_schema', 'STORAGE_HONCHO_HOST_BLOCK'),
    'SessionImport': ('hermes_cli.web_models', 'SessionImport'),
    'SessionPrune': ('hermes_cli.web_models', 'SessionPrune'),
    'SessionRename': ('hermes_cli.web_models', 'SessionRename'),
    'SkillContentUpdate': ('hermes_cli.web_models', 'SkillContentUpdate'),
    'SkillCreate': ('hermes_cli.web_models', 'SkillCreate'),
    'SkillInstallRequest': ('hermes_cli.web_models', 'SkillInstallRequest'),
    'SkillToggle': ('hermes_cli.web_models', 'SkillToggle'),
    'SkillUninstallRequest': ('hermes_cli.web_models', 'SkillUninstallRequest'),
    'SkillsUpdateRequest': ('hermes_cli.web_models', 'SkillsUpdateRequest'),
    'TTSLeaseRequest': ('hermes_cli.web_models', 'TTSLeaseRequest'),
    'TTSSpeakRequest': ('hermes_cli.web_models', 'TTSSpeakRequest'),
    'TelegramOnboardingApply': ('hermes_cli.web_models', 'TelegramOnboardingApply'),
    'TelegramOnboardingStart': ('hermes_cli.web_models', 'TelegramOnboardingStart'),
    'TerminalBackendSelect': ('hermes_cli.web_models', 'TerminalBackendSelect'),
    'ThemeSetBody': ('hermes_cli.web_models', 'ThemeSetBody'),
    'ToolsetEnvUpdate': ('hermes_cli.web_models', 'ToolsetEnvUpdate'),
    'ToolsetModelSelect': ('hermes_cli.web_models', 'ToolsetModelSelect'),
    'ToolsetPostSetup': ('hermes_cli.web_models', 'ToolsetPostSetup'),
    'ToolsetProviderSelect': ('hermes_cli.web_models', 'ToolsetProviderSelect'),
    'ToolsetToggle': ('hermes_cli.web_models', 'ToolsetToggle'),
    'WebhookCreate': ('hermes_cli.web_models', 'WebhookCreate'),
    'WebhookEnabledToggle': ('hermes_cli.web_models', 'WebhookEnabledToggle'),
    'WhatsAppOnboardingApply': ('hermes_cli.web_models', 'WhatsAppOnboardingApply'),
    'WhatsAppOnboardingStart': ('hermes_cli.web_models', 'WhatsAppOnboardingStart'),
    'activate_custom_endpoint': ('hermes_cli.web_routers.config_env', 'activate_custom_endpoint'),
    'add_credential_pool_entry': ('hermes_cli.web_routers.ops', 'add_credential_pool_entry'),
    'add_mcp_server': ('hermes_cli.web_routers.mcp', 'add_mcp_server'),
    'apply_telegram_onboarding': ('hermes_cli.web_routers.messaging', 'apply_telegram_onboarding'),
    'apply_whatsapp_onboarding': ('hermes_cli.web_routers.messaging', 'apply_whatsapp_onboarding'),
    'approve_pairing': ('hermes_cli.web_routers.ops', 'approve_pairing'),
    'auth_mcp_server': ('hermes_cli.web_routers.mcp', 'auth_mcp_server'),
    'build_cron_model_impact': ('hermes_cli.config', 'build_cron_model_impact'),
    'bulk_delete_sessions_endpoint': ('hermes_cli.web_routers.sessions', 'bulk_delete_sessions_endpoint'),
    'cancel_oauth_session': ('hermes_cli.web_routers.oauth', 'cancel_oauth_session'),
    'cancel_telegram_onboarding': ('hermes_cli.web_routers.messaging', 'cancel_telegram_onboarding'),
    'cancel_whatsapp_onboarding': ('hermes_cli.web_routers.messaging', 'cancel_whatsapp_onboarding'),
    'cfg_get': ('hermes_cli.config', 'cfg_get'),
    'check_config_version': ('hermes_cli.config', 'check_config_version'),
    'check_hermes_update': ('hermes_cli.web_routers.actions', 'check_hermes_update'),
    'clear_model_endpoint_credentials': ('hermes_cli.config', 'clear_model_endpoint_credentials'),
    'clear_pending_pairing': ('hermes_cli.web_routers.ops', 'clear_pending_pairing'),
    'coerce_provider_id': ('hermes_cli.config', 'coerce_provider_id'),
    'console_ws': ('hermes_cli.web_routers.chat_ws', 'console_ws'),
    'count_empty_sessions_endpoint': ('hermes_cli.web_routers.sessions', 'count_empty_sessions_endpoint'),
    'create_cron_job': ('hermes_cli.web_routers.cron', 'create_cron_job'),
    'create_hook': ('hermes_cli.web_routers.ops', 'create_hook'),
    'create_managed_directory': ('hermes_cli.web_routers.files', 'create_managed_directory'),
    'create_profile_endpoint': ('hermes_cli.web_routers.profiles', 'create_profile_endpoint'),
    'create_skill': ('hermes_cli.web_routers.skills', 'create_skill'),
    'create_webhook': ('hermes_cli.web_routers.ops', 'create_webhook'),
    'cron_fire_webhook': ('hermes_cli.web_routers.cron', 'cron_fire_webhook'),
    'custom_endpoint_key_env': ('hermes_cli.config', 'custom_endpoint_key_env'),
    'delete_agent_plugin': ('hermes_cli.web_routers.dashboard_ui', 'delete_agent_plugin'),
    'delete_cron_job': ('hermes_cli.web_routers.cron', 'delete_cron_job'),
    'delete_custom_endpoint': ('hermes_cli.web_routers.config_env', 'delete_custom_endpoint'),
    'delete_empty_sessions_endpoint': ('hermes_cli.web_routers.sessions', 'delete_empty_sessions_endpoint'),
    'delete_hook': ('hermes_cli.web_routers.ops', 'delete_hook'),
    'delete_learning_node': ('hermes_cli.web_routers.status', 'delete_learning_node'),
    'delete_managed_file': ('hermes_cli.web_routers.files', 'delete_managed_file'),
    'delete_profile_endpoint': ('hermes_cli.web_routers.profiles', 'delete_profile_endpoint'),
    'delete_session_endpoint': ('hermes_cli.web_routers.sessions', 'delete_session_endpoint'),
    'delete_webhook': ('hermes_cli.web_routers.ops', 'delete_webhook'),
    'derive_gateway_busy': ('gateway.status', 'derive_gateway_busy'),
    'derive_gateway_drainable': ('gateway.status', 'derive_gateway_drainable'),
    'describe_profile_auto_endpoint': ('hermes_cli.web_routers.profiles', 'describe_profile_auto_endpoint'),
    'detect_install_method': ('hermes_cli.config', 'detect_install_method'),
    'disconnect_oauth_provider': ('hermes_cli.web_routers.oauth', 'disconnect_oauth_provider'),
    'download_dashboard_backup': ('hermes_cli.web_routers.ops', 'download_dashboard_backup'),
    'download_managed_file': ('hermes_cli.web_routers.files', 'download_managed_file'),
    'enable_webhooks': ('hermes_cli.web_routers.ops', 'enable_webhooks'),
    'env_var_enabled': ('utils', 'env_var_enabled'),
    'events_ws': ('hermes_cli.web_routers.chat_ws', 'events_ws'),
    'export_session_endpoint': ('hermes_cli.web_routers.sessions', 'export_session_endpoint'),
    'find_provider_entry': ('hermes_cli.config', 'find_provider_entry'),
    'format_docker_update_message': ('hermes_cli.config', 'format_docker_update_message'),
    'fs_default_cwd': ('hermes_cli.web_routers.files', 'fs_default_cwd'),
    'fs_download': ('hermes_cli.web_routers.files', 'fs_download'),
    'fs_git_root': ('hermes_cli.web_routers.files', 'fs_git_root'),
    'fs_list': ('hermes_cli.web_routers.files', 'fs_list'),
    'fs_read_data_url': ('hermes_cli.web_routers.files', 'fs_read_data_url'),
    'fs_read_text': ('hermes_cli.web_routers.files', 'fs_read_text'),
    'fs_write_text': ('hermes_cli.web_routers.files', 'fs_write_text'),
    'gateway_drain': ('hermes_cli.web_routers.actions', 'gateway_drain'),
    'gateway_ws': ('hermes_cli.web_routers.chat_ws', 'gateway_ws'),
    'get_action_status': ('hermes_cli.web_routers.actions', 'get_action_status'),
    'get_active_profile_endpoint': ('hermes_cli.web_routers.profiles', 'get_active_profile_endpoint'),
    'get_auxiliary_models': ('hermes_cli.web_routers.models', 'get_auxiliary_models'),
    'get_client_voice_config': ('hermes_cli.web_routers.audio', 'get_client_voice_config'),
    'get_computer_use_status': ('hermes_cli.web_routers.tools', 'get_computer_use_status'),
    'get_config': ('hermes_cli.web_routers.config_env', 'get_config'),
    'get_config_path': ('hermes_cli.config', 'get_config_path'),
    'get_config_raw': ('hermes_cli.web_routers.analytics', 'get_config_raw'),
    'get_cron_delivery_targets': ('hermes_cli.web_routers.cron', 'get_cron_delivery_targets'),
    'get_cron_job': ('hermes_cli.web_routers.cron', 'get_cron_job'),
    'get_curator_status': ('hermes_cli.web_routers.status', 'get_curator_status'),
    'get_dashboard_font': ('hermes_cli.web_routers.dashboard_ui', 'get_dashboard_font'),
    'get_dashboard_plugins': ('hermes_cli.web_routers.dashboard_ui', 'get_dashboard_plugins'),
    'get_dashboard_themes': ('hermes_cli.web_routers.dashboard_ui', 'get_dashboard_themes'),
    'get_defaults': ('hermes_cli.web_routers.config_env', 'get_defaults'),
    'get_egress_status': ('hermes_cli.web_routers.config_env', 'get_egress_status'),
    'get_elevenlabs_voices': ('hermes_cli.web_routers.audio', 'get_elevenlabs_voices'),
    'get_env_path': ('hermes_cli.config', 'get_env_path'),
    'get_env_vars': ('hermes_cli.web_routers.config_env', 'get_env_vars'),
    'get_health': ('hermes_cli.web_routers.status', 'get_health'),
    'get_hermes_home': ('hermes_cli.config', 'get_hermes_home'),
    'get_learning_graph': ('hermes_cli.web_routers.status', 'get_learning_graph'),
    'get_learning_node': ('hermes_cli.web_routers.status', 'get_learning_node'),
    'get_logs': ('hermes_cli.web_routers.status', 'get_logs'),
    'get_media': ('hermes_cli.web_routers.files', 'get_media'),
    'get_memory_provider_config': ('hermes_cli.web_routers.memory_providers', 'get_memory_provider_config'),
    'get_memory_status': ('hermes_cli.web_routers.ops', 'get_memory_status'),
    'get_messaging_platforms': ('hermes_cli.web_routers.messaging', 'get_messaging_platforms'),
    'get_moa_models': ('hermes_cli.web_routers.models', 'get_moa_models'),
    'get_model_info': ('hermes_cli.web_routers.models', 'get_model_info'),
    'get_model_options': ('hermes_cli.web_routers.models', 'get_model_options'),
    'get_models_analytics': ('hermes_cli.web_routers.analytics', 'get_models_analytics'),
    'get_plugins_hub': ('hermes_cli.web_routers.dashboard_ui', 'get_plugins_hub'),
    'get_portal_status': ('hermes_cli.web_routers.status', 'get_portal_status'),
    'get_process_hermes_home': ('hermes_cli.config', 'get_process_hermes_home'),
    'get_profile_setup_command': ('hermes_cli.web_routers.profiles', 'get_profile_setup_command'),
    'get_profile_soul': ('hermes_cli.web_routers.profiles', 'get_profile_soul'),
    'get_profiles_sessions': ('hermes_cli.web_routers.profiles', 'get_profiles_sessions'),
    'get_profiles_sessions_sidebar': ('hermes_cli.web_routers.profiles', 'get_profiles_sessions_sidebar'),
    'get_provider_config_schema': ('plugins.memory.config_schema', 'get_provider_config_schema'),
    'get_recommended_default_model': ('hermes_cli.web_routers.models', 'get_recommended_default_model'),
    'get_running_pid': ('gateway.status', 'get_running_pid'),
    'get_running_pid_cached': ('gateway.status', 'get_running_pid_cached'),
    'get_runtime_status_running_pid': ('gateway.status', 'get_runtime_status_running_pid'),
    'get_schema': ('hermes_cli.web_routers.config_env', 'get_schema'),
    'get_session_detail': ('hermes_cli.web_routers.sessions', 'get_session_detail'),
    'get_session_latest_descendant': ('hermes_cli.web_routers.sessions', 'get_session_latest_descendant'),
    'get_session_messages': ('hermes_cli.web_routers.sessions', 'get_session_messages'),
    'get_session_stats': ('hermes_cli.web_routers.sessions', 'get_session_stats'),
    'get_sessions': ('hermes_cli.web_routers.sessions', 'get_sessions'),
    'get_skill_content': ('hermes_cli.web_routers.skills', 'get_skill_content'),
    'get_skills': ('hermes_cli.web_routers.skills', 'get_skills'),
    'get_ssh_ownership': ('hermes_cli.web_routers.status', 'get_ssh_ownership'),
    'get_status': ('hermes_cli.web_routers.status', 'get_status'),
    'get_system_stats': ('hermes_cli.web_routers.status', 'get_system_stats'),
    'get_telegram_onboarding_status': ('hermes_cli.web_routers.messaging', 'get_telegram_onboarding_status'),
    'get_terminal_backends': ('hermes_cli.web_routers.tools', 'get_terminal_backends'),
    'get_toolset_config': ('hermes_cli.web_routers.tools', 'get_toolset_config'),
    'get_toolset_models': ('hermes_cli.web_routers.tools', 'get_toolset_models'),
    'get_toolsets': ('hermes_cli.web_routers.tools', 'get_toolsets'),
    'get_update_receipt': ('hermes_cli.web_routers.actions', 'get_update_receipt'),
    'get_usage_analytics': ('hermes_cli.web_routers.analytics', 'get_usage_analytics'),
    'get_whatsapp_onboarding_status': ('hermes_cli.web_routers.messaging', 'get_whatsapp_onboarding_status'),
    'git_base_branches_route': ('hermes_cli.web_routers.git', 'git_base_branches_route'),
    'git_branch_switch_route': ('hermes_cli.web_routers.git', 'git_branch_switch_route'),
    'git_branches_route': ('hermes_cli.web_routers.git', 'git_branches_route'),
    'git_commit_context_route': ('hermes_cli.web_routers.git', 'git_commit_context_route'),
    'git_commit_route': ('hermes_cli.web_routers.git', 'git_commit_route'),
    'git_create_pr_route': ('hermes_cli.web_routers.git', 'git_create_pr_route'),
    'git_file_diff_route': ('hermes_cli.web_routers.git', 'git_file_diff_route'),
    'git_push_route': ('hermes_cli.web_routers.git', 'git_push_route'),
    'git_rev_parse_route': ('hermes_cli.web_routers.git', 'git_rev_parse_route'),
    'git_revert_route': ('hermes_cli.web_routers.git', 'git_revert_route'),
    'git_review_diff_route': ('hermes_cli.web_routers.git', 'git_review_diff_route'),
    'git_review_list_route': ('hermes_cli.web_routers.git', 'git_review_list_route'),
    'git_ship_info_route': ('hermes_cli.web_routers.git', 'git_ship_info_route'),
    'git_stage_route': ('hermes_cli.web_routers.git', 'git_stage_route'),
    'git_status_route': ('hermes_cli.web_routers.git', 'git_status_route'),
    'git_unstage_route': ('hermes_cli.web_routers.git', 'git_unstage_route'),
    'git_worktree_add_route': ('hermes_cli.web_routers.git', 'git_worktree_add_route'),
    'git_worktree_remove_route': ('hermes_cli.web_routers.git', 'git_worktree_remove_route'),
    'git_worktrees_route': ('hermes_cli.web_routers.git', 'git_worktrees_route'),
    'grant_computer_use_permissions': ('hermes_cli.web_routers.tools', 'grant_computer_use_permissions'),
    'import_sessions_endpoint': ('hermes_cli.web_routers.sessions', 'import_sessions_endpoint'),
    'install_mcp_catalog_entry': ('hermes_cli.web_routers.mcp', 'install_mcp_catalog_entry'),
    'install_skill_hub': ('hermes_cli.web_routers.skills', 'install_skill_hub'),
    'instantiate_blueprint': ('hermes_cli.web_routers.cron', 'instantiate_blueprint'),
    'is_nix_install_method': ('hermes_cli.config', 'is_nix_install_method'),
    'list_checkpoints': ('hermes_cli.web_routers.ops', 'list_checkpoints'),
    'list_credential_pool': ('hermes_cli.web_routers.ops', 'list_credential_pool'),
    'list_cron_blueprints': ('hermes_cli.web_routers.cron', 'list_cron_blueprints'),
    'list_cron_job_runs': ('hermes_cli.web_routers.cron', 'list_cron_job_runs'),
    'list_cron_jobs': ('hermes_cli.web_routers.cron', 'list_cron_jobs'),
    'list_custom_endpoints': ('hermes_cli.web_routers.config_env', 'list_custom_endpoints'),
    'list_hooks': ('hermes_cli.web_routers.ops', 'list_hooks'),
    'list_managed_files': ('hermes_cli.web_routers.files', 'list_managed_files'),
    'list_mcp_catalog': ('hermes_cli.web_routers.mcp', 'list_mcp_catalog'),
    'list_mcp_servers': ('hermes_cli.web_routers.mcp', 'list_mcp_servers'),
    'list_oauth_providers': ('hermes_cli.web_routers.oauth', 'list_oauth_providers'),
    'list_pairing': ('hermes_cli.web_routers.ops', 'list_pairing'),
    'list_profiles_endpoint': ('hermes_cli.web_routers.profiles', 'list_profiles_endpoint'),
    'list_skills_hub_sources': ('hermes_cli.web_routers.skills', 'list_skills_hub_sources'),
    'list_webhooks': ('hermes_cli.web_routers.ops', 'list_webhooks'),
    'load_env': ('hermes_cli.config', 'load_env'),
    'mcp_oauth_callback': ('hermes_cli.web_routers.mcp', 'mcp_oauth_callback'),
    'mcp_oauth_flow_status': ('hermes_cli.web_routers.mcp', 'mcp_oauth_flow_status'),
    'normalize_updated_at': ('gateway.status', 'normalize_updated_at'),
    'open_profile_terminal_endpoint': ('hermes_cli.web_routers.profiles', 'open_profile_terminal_endpoint'),
    'parse_active_agents': ('gateway.status', 'parse_active_agents'),
    'pause_cron_job': ('hermes_cli.web_routers.cron', 'pause_cron_job'),
    'poll_oauth_session': ('hermes_cli.web_routers.oauth', 'poll_oauth_session'),
    'post_agent_plugin_disable': ('hermes_cli.web_routers.dashboard_ui', 'post_agent_plugin_disable'),
    'post_agent_plugin_enable': ('hermes_cli.web_routers.dashboard_ui', 'post_agent_plugin_enable'),
    'post_agent_plugin_install': ('hermes_cli.web_routers.dashboard_ui', 'post_agent_plugin_install'),
    'post_agent_plugin_update': ('hermes_cli.web_routers.dashboard_ui', 'post_agent_plugin_update'),
    'post_plugin_visibility': ('hermes_cli.web_routers.dashboard_ui', 'post_plugin_visibility'),
    'preview_skill_hub': ('hermes_cli.web_routers.skills', 'preview_skill_hub'),
    'prune_checkpoints': ('hermes_cli.web_routers.ops', 'prune_checkpoints'),
    'prune_sessions_endpoint': ('hermes_cli.web_routers.sessions', 'prune_sessions_endpoint'),
    'pty_ws': ('hermes_cli.web_routers.chat_ws', 'pty_ws'),
    'pub_ws': ('hermes_cli.web_routers.chat_ws', 'pub_ws'),
    'put_plugin_providers': ('hermes_cli.web_routers.dashboard_ui', 'put_plugin_providers'),
    'read_managed_file': ('hermes_cli.web_routers.files', 'read_managed_file'),
    'read_raw_config': ('hermes_cli.config', 'read_raw_config'),
    'read_runtime_status': ('gateway.status', 'read_runtime_status'),
    'recommended_update_command_for_method': ('hermes_cli.config', 'recommended_update_command_for_method'),
    'redact_key': ('hermes_cli.config', 'redact_key'),
    'remove_credential_pool_entry': ('hermes_cli.web_routers.ops', 'remove_credential_pool_entry'),
    'remove_env_value': ('hermes_cli.config', 'remove_env_value'),
    'remove_env_var': ('hermes_cli.web_routers.config_env', 'remove_env_var'),
    'remove_mcp_server': ('hermes_cli.web_routers.mcp', 'remove_mcp_server'),
    'rename_profile_endpoint': ('hermes_cli.web_routers.profiles', 'rename_profile_endpoint'),
    'rename_session_endpoint': ('hermes_cli.web_routers.sessions', 'rename_session_endpoint'),
    'replace_mcp_servers': ('hermes_cli.web_routers.mcp', 'replace_mcp_servers'),
    'rescan_dashboard_plugins': ('hermes_cli.web_routers.dashboard_ui', 'rescan_dashboard_plugins'),
    'reset_memory': ('hermes_cli.web_routers.ops', 'reset_memory'),
    'resolve_cron_model_drift_defaults': ('hermes_cli.config', 'resolve_cron_model_drift_defaults'),
    'resolve_gateway_liveness': ('gateway.status', 'resolve_gateway_liveness'),
    'restart_gateway': ('hermes_cli.web_routers.actions', 'restart_gateway'),
    'resume_cron_job': ('hermes_cli.web_routers.cron', 'resume_cron_job'),
    'reveal_env_var': ('hermes_cli.web_routers.config_env', 'reveal_env_var'),
    'revoke_pairing': ('hermes_cli.web_routers.ops', 'revoke_pairing'),
    'run_backup': ('hermes_cli.web_routers.ops', 'run_backup'),
    'run_config_migrate': ('hermes_cli.web_routers.status', 'run_config_migrate'),
    'run_curator': ('hermes_cli.web_routers.status', 'run_curator'),
    'run_debug_share_endpoint': ('hermes_cli.web_routers.status', 'run_debug_share_endpoint'),
    'run_doctor': ('hermes_cli.doctor', 'run_doctor'),
    'run_dump': ('hermes_cli.dump', 'run_dump'),
    'run_import': ('hermes_cli.web_routers.ops', 'run_import'),
    'run_import_upload': ('hermes_cli.web_routers.ops', 'run_import_upload'),
    'run_prompt_size': ('hermes_cli.web_routers.status', 'run_prompt_size'),
    'run_security_audit': ('hermes_cli.web_routers.ops', 'run_security_audit'),
    'run_toolset_post_setup': ('hermes_cli.web_routers.tools', 'run_toolset_post_setup'),
    'save_config': ('hermes_cli.config', 'save_config'),
    'save_env_value': ('hermes_cli.config', 'save_env_value'),
    'save_toolset_env': ('hermes_cli.web_routers.tools', 'save_toolset_env'),
    'scan_skill_hub': ('hermes_cli.web_routers.skills', 'scan_skill_hub'),
    'search_sessions': ('hermes_cli.web_routers.sessions', 'search_sessions'),
    'search_skills_hub': ('hermes_cli.web_routers.skills', 'search_skills_hub'),
    'select_terminal_backend': ('hermes_cli.web_routers.tools', 'select_terminal_backend'),
    'select_toolset_model': ('hermes_cli.web_routers.tools', 'select_toolset_model'),
    'select_toolset_provider': ('hermes_cli.web_routers.tools', 'select_toolset_provider'),
    'serve_plugin_asset': ('hermes_cli.web_routers.dashboard_ui', 'serve_plugin_asset'),
    'set_active_profile_endpoint': ('hermes_cli.web_routers.profiles', 'set_active_profile_endpoint'),
    'set_curator_paused': ('hermes_cli.web_routers.status', 'set_curator_paused'),
    'set_dashboard_font': ('hermes_cli.web_routers.dashboard_ui', 'set_dashboard_font'),
    'set_dashboard_theme': ('hermes_cli.web_routers.dashboard_ui', 'set_dashboard_theme'),
    'set_env_var': ('hermes_cli.web_routers.config_env', 'set_env_var'),
    'set_mcp_server_enabled': ('hermes_cli.web_routers.mcp', 'set_mcp_server_enabled'),
    'set_memory_provider': ('hermes_cli.web_routers.ops', 'set_memory_provider'),
    'set_moa_models': ('hermes_cli.web_routers.models', 'set_moa_models'),
    'set_model_assignment': ('hermes_cli.web_routers.models', 'set_model_assignment'),
    'set_webhook_enabled': ('hermes_cli.web_routers.ops', 'set_webhook_enabled'),
    'setup_memory_provider': ('hermes_cli.web_routers.memory_providers', 'setup_memory_provider'),
    'speak_stream_ws': ('hermes_cli.web_routers.audio', 'speak_stream_ws'),
    'speak_text': ('hermes_cli.web_routers.audio', 'speak_text'),
    'start_gateway': ('hermes_cli.web_routers.ops', 'start_gateway'),
    'start_oauth_login': ('hermes_cli.web_routers.oauth', 'start_oauth_login'),
    'start_telegram_onboarding': ('hermes_cli.web_routers.messaging', 'start_telegram_onboarding'),
    'start_whatsapp_onboarding': ('hermes_cli.web_routers.messaging', 'start_whatsapp_onboarding'),
    'stop_gateway': ('hermes_cli.web_routers.ops', 'stop_gateway'),
    'stream_managed_file': ('hermes_cli.web_routers.files', 'stream_managed_file'),
    'submit_oauth_code': ('hermes_cli.web_routers.oauth', 'submit_oauth_code'),
    'test_mcp_server': ('hermes_cli.web_routers.mcp', 'test_mcp_server'),
    'test_messaging_platform': ('hermes_cli.web_routers.messaging', 'test_messaging_platform'),
    'toggle_skill': ('hermes_cli.web_routers.skills', 'toggle_skill'),
    'toggle_toolset': ('hermes_cli.web_routers.tools', 'toggle_toolset'),
    'transcribe_audio_upload': ('hermes_cli.web_routers.audio', 'transcribe_audio_upload'),
    'trigger_cron_job': ('hermes_cli.web_routers.cron', 'trigger_cron_job'),
    'tts_lease': ('hermes_cli.web_routers.audio', 'tts_lease'),
    'uninstall_skill_hub': ('hermes_cli.web_routers.skills', 'uninstall_skill_hub'),
    'update_config': ('hermes_cli.web_routers.config_env', 'update_config'),
    'update_config_raw': ('hermes_cli.web_routers.analytics', 'update_config_raw'),
    'update_cron_job': ('hermes_cli.web_routers.cron', 'update_cron_job'),
    'update_hermes': ('hermes_cli.web_routers.actions', 'update_hermes'),
    'update_learning_node': ('hermes_cli.web_routers.status', 'update_learning_node'),
    'update_memory_provider_config': ('hermes_cli.web_routers.memory_providers', 'update_memory_provider_config'),
    'update_messaging_platform': ('hermes_cli.web_routers.messaging', 'update_messaging_platform'),
    'update_profile_description_endpoint': ('hermes_cli.web_routers.profiles', 'update_profile_description_endpoint'),
    'update_profile_model_endpoint': ('hermes_cli.web_routers.profiles', 'update_profile_model_endpoint'),
    'update_profile_soul': ('hermes_cli.web_routers.profiles', 'update_profile_soul'),
    'update_skill_content': ('hermes_cli.web_routers.skills', 'update_skill_content'),
    'update_skills_hub': ('hermes_cli.web_routers.skills', 'update_skills_hub'),
    'upload_chat_image': ('hermes_cli.web_routers.files', 'upload_chat_image'),
    'upload_managed_file': ('hermes_cli.web_routers.files', 'upload_managed_file'),
    'upload_managed_file_stream': ('hermes_cli.web_routers.files', 'upload_managed_file_stream'),
    'upsert_custom_endpoint': ('hermes_cli.web_routers.config_env', 'upsert_custom_endpoint'),
    'validate_custom_endpoint': ('hermes_cli.web_routers.config_env', 'validate_custom_endpoint'),
    'validate_provider_credential': ('hermes_cli.web_routers.config_env', 'validate_provider_credential'),
    'windows_detach_flags': ('hermes_cli._subprocess_compat', 'windows_detach_flags'),
    'windows_hide_flags': ('hermes_cli._subprocess_compat', 'windows_hide_flags'),
    'write_platform_config_field': ('hermes_cli.config', 'write_platform_config_field'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
