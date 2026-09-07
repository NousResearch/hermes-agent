"""Status dashboard routes: health, /api/status, system stats, curator, learning graph,
portal and diagnostics actions.

Extracted from ``hermes_cli.web_server``; app state and helpers are late-bound through
:mod:`hermes_cli.web_deps` (cycle-safe, monkeypatch-friendly).
"""

import concurrent.futures
import importlib
import logging
import re
import asyncio
import os
import sys
import time
from fastapi import APIRouter
from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_gateway import _display_system_platform
from starlette.concurrency import run_in_threadpool
from fastapi import HTTPException, Request
from gateway.status import derive_gateway_busy, derive_gateway_drainable, normalize_updated_at, parse_active_agents, resolve_gateway_liveness
from hermes_cli import __version__, __release_date__
from hermes_cli.config import get_config_path, get_env_path
from hermes_cli.web_models import CuratorPause, LearningNodeRef, LearningNodeEdit, DebugShareRequest
from hermes_cli.web_routers._common import scoped_to_thread
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()
# Mounted separately by web_server so /api/logs keeps its original route-table position.
logs_router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_collect_profile_gateway_topology_cached = late("_collect_profile_gateway_topology_cached", "hermes_cli.web_server_gateway")
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_dashboard_local_update_managed_externally = late("_dashboard_local_update_managed_externally", "hermes_cli.web_server_files")
_load_configured_gateway_platforms = late("_load_configured_gateway_platforms", "hermes_cli.web_server_gateway")
_probe_gateway_health = late("_probe_gateway_health", "hermes_cli.web_server_gateway")
_require_token = late("_require_token")
_resolve_profile_dir = late("_resolve_profile_dir", "hermes_cli.web_server_profiles")
_resolve_restart_drain_timeout = late("_resolve_restart_drain_timeout", "hermes_cli.web_server_lifecycle")
_spawn_hermes_action = late("_spawn_hermes_action", "hermes_cli.web_server_gateway")
_ssh_runtime_intact = late("_ssh_runtime_intact")
app = LateState("app")  # the FastAPI instance (app.state.*)
check_config_version = late("check_config_version", "hermes_cli.config")
get_hermes_home = late("get_hermes_home", "hermes_cli.config")
get_install_id = late("get_install_id")
get_running_pid_cached = late("get_running_pid_cached", "gateway.status")
get_runtime_status_running_pid = late("get_runtime_status_running_pid", "gateway.status")
load_config = late("load_config", "hermes_cli.config")
read_runtime_status = late("read_runtime_status", "gateway.status")
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")


_STATUS_ACTIVE_SESSIONS_TIMEOUT = 0.75
_GATEWAY_HEALTH_ROUTE_TIMEOUT = 1.0
_HEALTHY_PLATFORM_STATES = {"connected", "running", "ok"}


def _safe_call(mod, fn_name: str, default):
    try:
        fn = getattr(mod, fn_name, None)
        return fn() if callable(fn) else default
    except Exception:
        return default


def _count_status_active_sessions() -> int:
    """Best-effort status garnish. Opens read-only (via the shared stale-schema heal) so
    /api/status never routinely writes to state.db while another Hermes process uses it."""
    from hermes_state import _default_db_path
    # The heal helper bootstraps a missing store; this garnish must not — on a fresh install
    # /api/status polls would otherwise create state.db before the user's first session.
    if not Path(_default_db_path()).exists():
        return 0
    db = _open_session_db_for_profile(None, read_only=True)
    try:
        sessions = db.list_sessions_rich(limit=50, compact_rows=True)
        now = time.time()
        return sum(1 for s in sessions if s.get("ended_at") is None
                   and (now - s.get("last_active", s.get("started_at", 0))) < 300)
    finally:
        db.close()


async def _status_active_sessions() -> int:
    try:
        return await asyncio.wait_for(
            run_in_threadpool(_count_status_active_sessions),
            timeout=_STATUS_ACTIVE_SESSIONS_TIMEOUT)
    except asyncio.TimeoutError:
        _log.debug("/api/status active session count exceeded %.2fs; returning 0",
                   _STATUS_ACTIVE_SESSIONS_TIMEOUT)
    except Exception as exc:
        _log.debug("/api/status active session count unavailable: %s", exc)
    return 0


@router.get("/api/ssh/ownership")
async def get_ssh_ownership(request: Request):
    from hermes_cli.web_server import _SSH_OWNER_NONCE
    _require_token(request)
    if not _SSH_OWNER_NONCE:
        raise HTTPException(status_code=404, detail="SSH ownership is not active")
    return {"ok": True, "sshOwnerNonce": _SSH_OWNER_NONCE, "protocolVersion": 1,
            "runtimeIntact": _ssh_runtime_intact()}


@router.get("/api/health")
async def get_health():
    """Lightweight process liveness for desktop/backend readiness probes."""
    return {"ok": True, "version": __version__,
            "auth_required": bool(getattr(app.state, "auth_required", False))}


# Profile segment mirrors hermes_cli.profiles._PROFILE_ID_RE. Platform segment mirrors the
# Platform enum's normalized values: built-in members plus plugin directory names
# (lowercased), which allow hyphens as well as underscores (e.g. ``reviewer:foo-bar``).
_PROFILE_PLATFORM_STATUS_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}:[a-z0-9][a-z0-9_-]{0,63}$")


def _is_profile_platform_status_key(key: object) -> bool:
    """Accept only the runner's public ``<profile>:<platform>`` key grammar."""
    return isinstance(key, str) and bool(_PROFILE_PLATFORM_STATUS_KEY_RE.fullmatch(key))


def _status_platform_key_allowed(key: object, configured: "set[str] | None") -> bool:
    """Whether a runtime-status platform key may appear publicly: namespaced
    ``<profile>:<platform>`` keys are validated against the grammar *unconditionally* (a
    failed config-set load must not fail open into projecting arbitrary keys from a
    process-local JSON file); plain keys are checked against the configured set when it
    loaded, passed through when it did not."""
    if not isinstance(key, str):
        return False
    if ":" in key:
        return _is_profile_platform_status_key(key)
    return configured is None or key in configured


# Per-entry writer-identity stamps (added by gateway.status.write_runtime_status for the
# aggregation ownership check) are process recon — the same class of detail as the
# auth-gated top-level ``gateway_pid`` — and must not project onto the public endpoint.
_PRIVATE_PLATFORM_ENTRY_KEYS = frozenset({"writer_pid", "writer_start_time"})


def _public_platform_entry(value: Any) -> Any:
    """Strip writer-identity stamps from a platform entry before projection."""
    if not isinstance(value, dict):
        return value
    return {k: v for k, v in value.items() if k not in _PRIVATE_PLATFORM_ENTRY_KEYS}


def _merge_profile_gateway_platforms(gateway_platforms: dict, profile_platforms: dict) -> dict:
    """Merge independent per-profile gateway platform states: hosts running separate gateway
    services per profile (``gateway_mode == "multiple"``) persist each profile's platform
    failures in its own ``gateway_state.json``, invisible to the machine-level probe NAS
    reads unless folded in under the validated ``<profile>:<platform>`` grammar. The active
    profile's own map is skipped (already present); existing keys are never overwritten."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        active = get_active_profile_name()
    except Exception:
        active = "default"
    merged = dict(gateway_platforms)
    for prof, plats in (profile_platforms or {}).items():
        if prof == active or not isinstance(plats, dict):
            continue
        for key, value in plats.items():
            if not isinstance(key, str) or ":" in key or not isinstance(value, dict):
                continue
            namespaced = f"{prof}:{key}"
            if not _is_profile_platform_status_key(namespaced):
                continue
            merged.setdefault(namespaced, _public_platform_entry(value))
    return merged


# --- Gateway liveness detection --- Delegated to the single shared ladder in gateway.status so this
# endpoint and /api/messaging/platforms can never disagree about whether the gateway is up (they used to:
# sidebar "running" while the Channels page rendered "The gateway is not running"). When ?profile=<name> was
# given, scope PID and state reads to that profile's directory — gateway identity files (PID, lock, runtime
# status) are written to the per-profile home, not the process-level HERMES_HOME (see issue #69143). Plain
# /api/status keeps the exact zero-arg call so its behavior (and cache signature) is unchanged. The
# module-level probe references are handed to the resolver so the long-standing
# `monkeypatch.setattr(gateway.status, "get_running_pid_cached", ...)` seam used across the test-suite still
# intercepts them.
def _bounded_health_probe():
    """Health probe with the route's blocking-call budget preserved. The resolver only
    reaches this rung when the local PID probe came up empty, so the timeout is paid at
    most once per request and only in the cross-container case that needs it."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_probe_gateway_health)
        try:
            return future.result(timeout=_GATEWAY_HEALTH_ROUTE_TIMEOUT)
        except concurrent.futures.TimeoutError:
            _log.warning("/api/status gateway health probe exceeded %.2fs; using local status",
                         _GATEWAY_HEALTH_ROUTE_TIMEOUT)
            return False, None
        except Exception:
            return False, None


def _project_gateway_platforms(gateway_platforms: dict, configured: "set[str] | None",
                               gateway_running: bool, gateway_state) -> dict:
    """Public projection of a runtime's platform map (see ``_status_platform_key_allowed``
    for the key rules). A cleanly stopped gateway's platform states are stale noise and are
    cleared so a dead process can't report "connected"; a startup_failed gateway's FATAL
    entries are the diagnosis (credential collisions, auth failures) that the single
    exit_reason string can't express, so they are kept — upstream writer-identity/freshness
    filtering already dropped other processes' entries."""
    platforms = {key: _public_platform_entry(value) for key, value in gateway_platforms.items()
                 if _status_platform_key_allowed(key, configured)}
    if gateway_running:
        return platforms
    if gateway_state == "startup_failed":
        return {key: value for key, value in platforms.items()
                if isinstance(value, dict) and value.get("state") == "fatal"}
    return {}


async def _resolve_gateway_status(profile_dir: Optional[Path], health_url) -> Dict[str, Any]:
    """Liveness + runtime-state readout (running/pid/state/platforms/exit_reason/updated_at
    plus the raw ``runtime`` document).

    Liveness is delegated to the shared ladder in gateway.status so this endpoint and
    /api/messaging/platforms can never disagree. With ``?profile=<name>`` PID/state reads are
    scoped to that profile's directory (gateway identity files live in the per-profile
    home); plain /api/status keeps the exact zero-arg call so its cache signature is
    unchanged. The module-level probe references are handed to the resolver so the
    ``monkeypatch.setattr(gateway.status, "get_running_pid_cached", ...)`` seam still intercepts.
    """
    local_runtime = (read_runtime_status(path=profile_dir / "gateway_state.json")
                     if profile_dir else read_runtime_status())
    liveness = await run_in_threadpool(lambda: resolve_gateway_liveness(
        profile_dir=profile_dir, runtime=local_runtime,
        health_probe=_bounded_health_probe if health_url else None,
        pid_probe=get_running_pid_cached, runtime_reader=read_runtime_status,
        runtime_pid_probe=get_runtime_status_running_pid))
    gateway_running = liveness.running
    remote_health_body: dict | None = liveness.health_body

    try:
        configured = await run_in_threadpool(_load_configured_gateway_platforms)
    except Exception:
        configured = None

    # Prefer the detailed health endpoint response (has full state) when the local runtime
    # status file is absent or stale (cross-container).
    runtime = local_runtime
    if runtime is None and remote_health_body and remote_health_body.get("gateway_state"):
        runtime = remote_health_body

    gateway_state = None
    gateway_platforms: dict = {}
    gateway_exit_reason = None
    gateway_updated_at = None
    if runtime:
        gateway_state = runtime.get("gateway_state")
        if not gateway_running:
            gateway_state = gateway_state if gateway_state in {"stopped", "startup_failed"} else "stopped"
        elif remote_health_body is not None and gateway_state in {None, "stopped"}:
            # The health probe confirmed the gateway is alive, but the local runtime status
            # file may be stale (cross-container): override so the badge is correct.
            gateway_state = "running"
        gateway_platforms = _project_gateway_platforms(
            runtime.get("platforms") or {}, configured, gateway_running, gateway_state)
        gateway_exit_reason = runtime.get("exit_reason")
        # Contract: gateway_updated_at is RFC3339 string | null, never a number. ``runtime``
        # may be the local gateway_state.json (legacy gateways wrote epoch floats; hand
        # edits can inject anything) or a remote /health/detailed body — normalize both.
        gateway_updated_at = normalize_updated_at(runtime.get("updated_at"))

    # No runtime info at all but the health probe confirmed alive (no shared volume).
    if gateway_running and gateway_state is None and remote_health_body is not None:
        gateway_state = "running"

    return {
        "runtime": runtime, "gateway_running": gateway_running, "gateway_pid": liveness.pid,
        "gateway_state": gateway_state, "gateway_platforms": gateway_platforms,
        "gateway_exit_reason": gateway_exit_reason, "gateway_updated_at": gateway_updated_at}


def _auth_gate_status() -> Dict[str, Any]:
    """Dashboard auth gate readout: gate engaged, registered providers, and the RFC 8252
    native-app capability advertisement ``auth_flows`` the desktop reads to pick the
    system-browser + loopback + PKCE flow over the embedded-webview cookie flow. "cookie" is
    always available in gated mode; "native_pkce" when at least one interactive session
    provider is registered (token-only credentials such as drain don't count). Missing
    "native_pkce" ⇒ older gateway ⇒ desktop falls back automatically."""
    auth_required = bool(getattr(app.state, "auth_required", False))
    auth_providers: list[str] = []
    auth_flows: list[str] = []
    try:
        from hermes_cli.dashboard_auth import (
            list_providers as _list_providers, list_session_providers as _list_session_providers)
        auth_providers = [p.name for p in _list_providers()]
        if auth_required:
            auth_flows.append("cookie")
            if _list_session_providers():
                auth_flows.append("native_pkce")
    except Exception:
        # Module not importable yet (early startup) — leave as [].
        pass
    return {"auth_required": auth_required, "auth_providers": auth_providers,
            "auth_flows": auth_flows}


def _nous_session_validity() -> str:
    """Nous bootstrap-session validity for the NAS health sweep: a hosted agent whose Nous
    auth dies terminally looks HEALTHY to every liveness probe yet every inference turn
    fails, and this is the ONLY signal that surfaces it (local auth-store state, no token
    needed). Best-effort: never let auth classification break the probe."""
    try:
        from hermes_cli.auth import get_nous_session_validity
        return get_nous_session_validity()
    except Exception:
        return "unknown"


async def _component_health(gateway: Dict[str, Any]) -> Dict[str, Any]:
    """Component-level health rollup: counts and status enums only (public payload — no
    messages, paths or other detail that could carry secrets). The storage probe reuses the
    gateway readiness state_db check (read-only, 1s-bounded) off-loop."""
    from hermes_cli.web_server import DASHBOARD_HEALTH
    gateway_running, gateway_state = gateway["gateway_running"], gateway["gateway_state"]
    gateway_platforms = gateway["gateway_platforms"]
    components: Dict[str, Any] = {
        "gateway": {
            "status": "ok" if gateway_running and gateway_state in {"running", "draining"} else "degraded",
            "state": gateway_state or ("running" if gateway_running else "stopped")},
        "dashboard": DASHBOARD_HEALTH.snapshot()}
    try:
        from gateway.readiness import _probe_state_db
        storage_check = await run_in_threadpool(_probe_state_db, get_hermes_home())
        components["storage"] = {"status": storage_check.get("status", "degraded")}
    except Exception:
        components["storage"] = {"status": "degraded"}
    platform_states = [str(value.get("state") or value.get("status") or "").lower()
                       for value in gateway_platforms.values() if isinstance(value, dict)]
    connected = sum(1 for state in platform_states if state in _HEALTHY_PLATFORM_STATES)
    components["platforms"] = {"status": "ok" if connected == len(platform_states) else "degraded",
                               "configured": len(gateway_platforms), "connected": connected}
    return components


async def _advisory_pressure(status: Dict[str, Any], home: Path) -> None:
    """Memory / disk pressure rollups + deferred FTS rebuild progress (coarse numbers/enums
    only; public payload). Deliberately NOT folded into components/overall: pressure is
    advisory, not a liveness verdict, and flipping ``overall`` on it would page NAS's
    availability sweep for a condition the valve is already handling. Read-only, never raise.
    """
    for key, mod_name, fn_name in (("memory", "gateway.memory_status", "collect_memory_status"),
                                   ("disk", "gateway.disk_status", "collect_disk_status")):
        try:
            collect = getattr(importlib.import_module(mod_name), fn_name)
            status[key] = await run_in_threadpool(collect, home)
        except Exception:
            status[key] = {"pressure": "unknown"}

    try:
        from hermes_state import SessionDB as _SDB
        from hermes_constants import get_hermes_home as _ghh
        _db_path = _ghh() / "state.db"
        if _db_path.exists():
            _sdb = _SDB(db_path=_db_path, read_only=True)
            try:
                _rebuild = _sdb.fts_rebuild_status()
            finally:
                _sdb.close()
            if _rebuild is not None:
                status["fts_rebuild"] = _rebuild
    except Exception:
        pass


@router.get("/api/status")
async def get_status(profile: Optional[str] = None):
    """Public machine-level liveness probe (``PUBLIC_API_PATHS``): version, gateway state,
    active session count and the auth-gate shape — no bodies, no session content, no secrets.

    ``?profile=`` (dashboard management switcher) uses the config-only contextvar scope, NOT
    _profile_scope: this handler awaits the remote health probe, and _profile_scope swaps
    process-global skills-module attributes a concurrent request would cross-restore.
    """
    from hermes_cli.web_server import _GATEWAY_HEALTH_URL
    status_scope = None
    requested_profile = (profile or "").strip()
    profile_dir: Optional[Path] = None
    if requested_profile and requested_profile.lower() != "current":
        profile_dir = _resolve_profile_dir(requested_profile)
        status_scope = _config_profile_scope(requested_profile)
        status_scope.__enter__()

    try:
        current_ver, latest_ver = check_config_version()
        gateway = await _resolve_gateway_status(profile_dir, _GATEWAY_HEALTH_URL)
        gateway_running, gateway_state = gateway["gateway_running"], gateway["gateway_state"]

        # Topology (cached, TTL 10s) is fetched before the platform rollup so per-profile
        # gateway failures fold into the machine-level view (see
        # _merge_profile_gateway_platforms); a ``?profile=`` request is left unmerged.
        topology = await run_in_threadpool(_collect_profile_gateway_topology_cached)
        if not requested_profile:
            gateway["gateway_platforms"] = _merge_profile_gateway_platforms(
                gateway["gateway_platforms"], topology.get("profile_platforms") or {})

        active_sessions = await _status_active_sessions()

        # Busy/drainable (NAS lifecycle-safety gate) derive from the persisted in-flight turn
        # count + liveness via gateway.status. Liveness keys off gateway_running, NEVER
        # gateway_updated_at — a healthy idle gateway never advances that.
        active_agents = parse_active_agents((gateway["runtime"] or {}).get("active_agents", 0))
        # Off-loop: on a cold Windows install the first import of hermes_cli.gateway blocks
        # 15-30s (.pyc compilation + Defender), exceeding the desktop handshake's 15s timeout.
        restart_drain_timeout = await run_in_threadpool(_resolve_restart_drain_timeout)
        auth = _auth_gate_status()

        status = {
            "version": __version__, "release_date": __release_date__,
            "config_version": current_ver, "latest_config_version": latest_ver,
            "can_update_hermes": not _dashboard_local_update_managed_externally(),
            "gateway_running": gateway_running, "gateway_state": gateway_state,
            "gateway_platforms": gateway["gateway_platforms"],
            "gateway_exit_reason": gateway["gateway_exit_reason"],
            "gateway_updated_at": gateway["gateway_updated_at"],
            "active_agents": active_agents,
            "gateway_busy": derive_gateway_busy(
                gateway_running=gateway_running, gateway_state=gateway_state,
                active_agents=active_agents),
            "gateway_drainable": derive_gateway_drainable(
                gateway_running=gateway_running, gateway_state=gateway_state),
            "restart_drain_timeout": restart_drain_timeout, "active_sessions": active_sessions,
            **auth, "nous_session_valid": _nous_session_validity()}

        # Stable per-install identity (first call may touch disk). Omitted (not null) when
        # unpersistable so older-client behavior and the no-identity fallback stay identical.
        install_id = await run_in_threadpool(get_install_id)
        if install_id:
            status["install_id"] = install_id

        components = await _component_health(gateway)
        status["components"] = components
        status["overall"] = ("ok" if all(item.get("status") == "ok" for item in components.values())
                             else "degraded")
        await _advisory_pressure(status, profile_dir if profile_dir else get_hermes_home())

        # Profile NAMES and ``gateway_mode`` are low-sensitivity product surface (Hermes Cloud
        # renders the profile list over a gated bind) so they survive the auth gate; the
        # per-gateway ``gateways[]`` carries host ports and stays gated below.
        status["profiles"] = topology["profiles"]
        status["gateway_mode"] = topology["gateway_mode"]

        # Host paths, gateway PID, internal health URL and per-gateway ports are deployment
        # recon a liveness probe never needs, and on a gated bind *any* unauthenticated caller
        # reaches this endpoint — surface them only on a loopback / ``--insecure`` bind.
        if not auth["auth_required"]:
            status.update({
                "hermes_home": str(get_hermes_home()), "config_path": str(get_config_path()),
                "env_path": str(get_env_path()), "gateway_pid": gateway["gateway_pid"],
                "gateway_health_url": _GATEWAY_HEALTH_URL, "gateways": topology["gateways"]})

        return status
    finally:
        if status_scope is not None:
            status_scope.__exit__(*sys.exc_info())


@router.get("/api/system/stats")
async def get_system_stats():
    """Host + process system stats for the System page (stdlib identity; psutil CPU/memory/
    disk/uptime when available). Non-sensitive: no env values, no paths beyond hermes home."""
    import platform as _platform

    info: Dict[str, Any] = {
        **_display_system_platform(
            system=_platform.system(), release=_platform.release(), version=_platform.version(),
            platform_label=_platform.platform()),
        "arch": _platform.machine(), "hostname": _platform.node(),
        "python_version": _platform.python_version(),
        "python_impl": _platform.python_implementation(),
        "hermes_version": __version__, "cpu_count": os.cpu_count()}

    def _disk():
        du = psutil.disk_usage(str(get_hermes_home()))
        info["disk"] = {"total": du.total, "used": du.used, "free": du.free, "percent": du.percent}

    def _cpu():
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        la = getattr(psutil, "getloadavg", None)
        if la:
            info["load_avg"] = list(la())

    def _uptime():
        info["uptime_seconds"] = int(time.time() - psutil.boot_time())

    def _process():
        proc = psutil.Process()
        info["process"] = {"pid": proc.pid, "rss": proc.memory_info().rss,
                           "create_time": int(proc.create_time()),
                           "num_threads": proc.num_threads()}

    # psutil enriches the picture when present; every probe below is optional.
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info["memory"] = {"total": vm.total, "available": vm.available, "used": vm.used,
                          "percent": vm.percent}
        for fill in (_disk, _cpu, _uptime, _process):
            try:
                fill()
            except Exception:
                pass
        info["psutil"] = True
    except Exception:
        info["psutil"] = False
        # stdlib-only fallbacks for load average where the kernel exposes it.
        try:
            info["load_avg"] = list(os.getloadavg())
        except (OSError, AttributeError):
            pass

    return info


# Curator — background skill-maintenance status + the pause/resume/run-now controls.


@router.get("/api/curator")
async def get_curator_status():
    try:
        from agent import curator
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Curator unavailable: {exc}")
    state = _safe_call(curator, "load_state", {})
    return {
        "enabled": _safe_call(curator, "is_enabled", True),
        "paused": _safe_call(curator, "is_paused", False),
        "interval_hours": _safe_call(curator, "get_interval_hours", None),
        "last_run_at": state.get("last_run_at"),
        **{key: _safe_call(curator, f"get_{key}", None)
           for key in ("min_idle_hours", "stale_after_days", "archive_after_days")}}


@router.put("/api/curator/paused")
async def set_curator_paused(body: CuratorPause):
    from agent import curator
    curator.set_paused(bool(body.paused))
    return {"ok": True, "paused": bool(body.paused)}


def _spawn_action(argv: list, name: str, prefix: str) -> dict:
    """Spawn a background ``hermes <argv>`` action; a spawn failure is ``500 "<prefix>: <exc>"``."""
    try:
        proc = _spawn_hermes_action(argv, name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{prefix}: {exc}")
    return {"ok": True, "pid": proc.pid, "name": name}


@router.post("/api/curator/run")
async def run_curator():
    """Trigger a curator review now (backgrounded; tail via action status)."""
    return _spawn_action(["curator", "run"], "curator-run", "Failed to run curator")


@router.get("/api/learning/graph")
async def get_learning_graph(profile: Optional[str] = None):
    """Learning graph for the desktop panel: profile-scoped learned skills + memory chunks."""
    def _run():
        from agent.learning_graph import build_learning_graph
        return build_learning_graph()

    try:
        # _profile_scope takes _SKILLS_PROFILE_LOCK and the graph build reads skills/memories
        # from disk — keep it off the event loop.
        return await scoped_to_thread(profile, _run)
    except Exception:
        _log.exception("GET /api/learning/graph failed")
        raise HTTPException(status_code=500, detail="Failed to build learning graph")


async def _learning_mutation(profile: Optional[str], fn, status: int, fallback: str):
    """Run a learning_mutations call under ``_profile_scope`` off-loop; a non-ok result
    becomes ``HTTPException(status, message)``."""
    res = await scoped_to_thread(profile, fn)
    if not res.get("ok"):
        raise HTTPException(status_code=status, detail=res.get("message", fallback))
    return res


@router.get("/api/learning/node")
async def get_learning_node(id: str, profile: Optional[str] = None):
    """Current content of a journey node (skill SKILL.md or memory chunk), for an edit prefill."""
    from agent.learning_mutations import node_detail
    return await _learning_mutation(profile, lambda: node_detail(id), 404, "not found")


@router.delete("/api/learning/node")
async def delete_learning_node(body: LearningNodeRef):
    """Delete a journey node — skills are archived (restorable), memories removed."""
    from agent.learning_mutations import delete_node
    return await _learning_mutation(
        body.profile, lambda: delete_node(body.id), 400, "delete failed")


@router.put("/api/learning/node")
async def update_learning_node(body: LearningNodeEdit):
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    from agent.learning_mutations import edit_node
    return await _learning_mutation(
        body.profile, lambda: edit_node(body.id, body.content), 400, "edit failed")


# Portal — Nous Portal auth + Tool Gateway routing status (read-only).


@router.get("/api/portal")
async def get_portal_status():
    # load_config() + auth/subscription snapshots are disk reads on a polled endpoint —
    # keep them off the event loop.
    return await asyncio.to_thread(_get_portal_status_sync)


def _feature_state(feat) -> str:
    if getattr(feat, "managed_by_nous", False):
        return "via Nous Portal"
    if getattr(feat, "active", False):
        return getattr(feat, "current_provider", None) or "active"
    return "not configured"


def _get_portal_status_sync():
    cfg = load_config() or {}
    auth: Dict[str, Any] = {}
    try:
        from hermes_cli.auth import get_nous_auth_status_local
        # Refresh-free snapshot so polling never performs an OAuth refresh.
        auth = get_nous_auth_status_local() or {}
    except Exception:
        auth = {}

    features = []
    try:
        from hermes_cli.nous_subscription import get_nous_subscription_features
        feats = get_nous_subscription_features(cfg)
        if feats is not None:
            features = [{"label": getattr(feat, "label", ""), "state": _feature_state(feat)}
                        for feat in feats.items()]
    except Exception:
        _log.exception("portal features failed")

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return {
        "logged_in": bool(auth.get("logged_in")), "portal_url": auth.get("portal_base_url"),
        "inference_url": auth.get("inference_base_url"),
        "provider": str((model_cfg or {}).get("provider") or ""),
        "subscription_url": "https://portal.nousresearch.com/manage-subscription",
        "features": features}


# Diagnostics: text-output actions spawned in the background, tailed via /api/actions/<name>.


@router.post("/api/ops/prompt-size")
async def run_prompt_size():
    return _spawn_action(["prompt-size"], "prompt-size", "Failed")


@router.post("/api/ops/dump")
async def run_dump():
    return _spawn_action(["dump"], "dump", "Failed")


@router.post("/api/ops/config-migrate")
async def run_config_migrate():
    return _spawn_action(["config", "migrate"], "config-migrate", "Failed")


@router.post("/api/ops/debug-share")
async def run_debug_share_endpoint(body: DebugShareRequest | None = None):
    """Upload a redacted debug report + full logs and return the paste URLs. Synchronous,
    unlike the other diagnostics actions: the point is the shareable URLs, returned as a
    structured payload the dashboard renders as copyable links."""
    from hermes_cli.debug import build_debug_share
    req = body or DebugShareRequest()
    try:
        result = await asyncio.to_thread(
            build_debug_share, log_lines=max(1, min(int(req.lines), 5000)), redact=bool(req.redact))
    except RuntimeError as exc:
        # Required summary-report upload failed (offline / paste service down).
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}")
    except Exception as exc:
        _log.exception("debug share failed")
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")

    return {"ok": True, "urls": result.urls, "failures": result.failures,
            "redacted": result.redacted, "auto_delete_seconds": result.auto_delete_seconds}


@logs_router.get("/api/logs")
async def get_logs(
    file: str = "agent", lines: int = 100, level: Optional[str] = None,
    component: Optional[str] = None, search: Optional[str] = None):
    from hermes_cli.logs import _read_tail, LOG_FILES
    log_name = LOG_FILES.get(file)
    if not log_name:
        raise HTTPException(status_code=400, detail=f"Unknown log file: {file}")
    log_path = get_hermes_home() / "logs" / log_name
    if not log_path.exists():
        return {"file": file, "lines": []}

    try:
        from hermes_logging import COMPONENT_PREFIXES
    except ImportError:
        COMPONENT_PREFIXES = {}
    # "ALL"/"all"/empty → no filter (None, not (): _matches_filters treats an empty tuple as
    # "must match a prefix", which silently drops every line).
    min_level = level if level and level.upper() != "ALL" else None
    comp_prefixes = None
    if component and component.lower() != "all":
        comp_prefixes = COMPONENT_PREFIXES.get(component)
        if comp_prefixes is None:
            raise HTTPException(status_code=400, detail=f"Unknown component: {component}. "
                                f"Available: {', '.join(sorted(COMPONENT_PREFIXES))}")
    result = _read_tail(
        log_path, min(lines, 500) if not search else 2000,
        has_filters=bool(min_level or comp_prefixes or search),
        min_level=min_level, component_prefixes=comp_prefixes)
    # _read_tail doesn't support free-text search, so post-filter (case-insensitive
    # substring) here and trim to the requested line count afterward.
    if search:
        needle = search.lower()
        result = [l for l in result if needle in l.lower()][-min(lines, 500):]
    return {"file": file, "lines": result}
