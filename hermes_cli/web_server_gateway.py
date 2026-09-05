"""Gateway/process helpers for the dashboard: per-profile gateway topology (+cache), action
subprocess spawning, gateway restart plumbing, system platform display.
"""

import logging
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli._subprocess_compat import windows_detach_flags
from hermes_cli.config import get_hermes_home

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


def _probe_gateway_health() -> tuple[bool, dict | None]:
    """Probe the gateway's HTTP health endpoint (cross-container). Blocking — run in an executor.

    DEPRECATED: driven by the ``GATEWAY_HEALTH_URL`` / ``GATEWAY_HEALTH_TIMEOUT`` env vars,
    to be replaced by a dashboard config key; do not add callers. Accepts a base URL or an
    explicit ``/health`` / ``/health/detailed`` path; tries ``/health/detailed`` first.
    """
    from hermes_cli.web_server import _GATEWAY_HEALTH_TIMEOUT, _GATEWAY_HEALTH_URL
    if not _GATEWAY_HEALTH_URL:
        return False, None
    base = re.sub(r"/health(/detailed)?$", "", _GATEWAY_HEALTH_URL.rstrip("/"))
    for path in (f"{base}/health/detailed", f"{base}/health"):
        try:
            req = urllib.request.Request(path, method="GET")
            with urllib.request.urlopen(req, timeout=_GATEWAY_HEALTH_TIMEOUT) as resp:
                if resp.status == 200:
                    return True, json.loads(resp.read())
        except Exception:
            continue
    return False, None


# ``platform-name -> (config port key, adapter default)`` for port-binding gateway platforms.
# Mirrors PORT_BINDING_PLATFORM_VALUES (gateway/config.py) and each adapter's DEFAULT_PORT /
# DEFAULT_WEBHOOK_PORT. Display-only data for the topology readout, not a bind source.
_PORT_BINDING_PLATFORM_PORTS: Dict[str, Tuple[str, int]] = {
    "webhook": ("port", 8644), "api_server": ("port", 8642), "msgraph_webhook": ("port", 8646),
    "feishu": ("webhook_port", 8765), "wecom_callback": ("port", 8645), "bluebubbles": ("webhook_port", 8645),
    "sms": ("webhook_port", 8080), "whatsapp_cloud": ("webhook_port", 8090), "line": ("port", 8646),
    "teams": ("port", 3978),
}

# Platform states that mean the adapter is NOT serving its port right now.
_PLATFORM_DEAD_STATES = frozenset({"fatal", "disconnected", "stopped", "disabled"})


def _profile_platform_ports(profile_home: Path, runtime: Optional[dict]) -> Dict[str, int]:
    """Best-effort ``platform -> host TCP port`` for one profile's live gateway.

    Ports come from the profile's own config.yaml (``gateway.platforms`` then top-level
    ``platforms`` — later wins, matching load_gateway_config precedence), falling back to the
    adapter default. Env-var overrides (e.g. WEBHOOK_PORT in that profile's .env) are not resolved.
    """
    platforms = (runtime or {}).get("platforms") or {}
    active = [
        name for name, state in platforms.items()
        if name in _PORT_BINDING_PLATFORM_PORTS
        and isinstance(state, dict)
        and state.get("state") not in _PLATFORM_DEAD_STATES]
    if not active:
        return {}

    blocks: Dict[str, dict] = {}
    try:
        # load_config() targets the ACTIVE profile's home; read the probed profile's file raw.
        from hermes_cli.config import read_user_config_raw
        cfg = read_user_config_raw(profile_home / "config.yaml")
        gateway_cfg = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
        for src in ((gateway_cfg or {}).get("platforms"), cfg.get("platforms")):
            if not isinstance(src, dict):
                continue
            for plat_name, plat_block in src.items():
                if isinstance(plat_block, dict):
                    blocks.setdefault(plat_name, {}).update(plat_block)
    except Exception:
        blocks = {}

    ports: Dict[str, int] = {}
    for name in active:
        port_key, default_port = _PORT_BINDING_PLATFORM_PORTS[name]
        block = blocks.get(name) or {}
        extra = block.get("extra") if isinstance(block.get("extra"), dict) else {}
        raw = block.get(port_key, (extra or {}).get(port_key, default_port))
        try:
            ports[name] = int(raw)
        except (TypeError, ValueError):
            ports[name] = default_port
    return ports


def _profile_gateway_writer_identity(profile_home: Path, runtime: Optional[dict]) -> Optional[tuple]:
    """``(pid, start_time)`` of the profile's LIVE gateway, or None.

    Uses the same validated-liveness helper and the same ``_get_process_start_time`` that stamped
    the record, so equality is exact (no unit/clock-source mismatch).
    """
    try:
        from gateway.status import _get_process_start_time, get_runtime_status_running_pid
        pid = get_runtime_status_running_pid(runtime, expected_home=profile_home)
        if pid is None:
            return None
        start_time = _get_process_start_time(pid)
        return None if start_time is None else (pid, start_time)
    except Exception:
        return None


def _owned_profile_platforms(writer_identity: Optional[tuple], platforms: dict) -> dict:
    """Keep only platform entries stamped by the profile's CURRENT gateway process.

    Gateway startup preserves plain platform entries in gateway_state.json across restarts, so the
    raw map can carry fatal state for platforms since disabled/removed. Cross-profile aggregation
    has no config context to filter against, so it demands exact ``(pid, start_time)`` writer
    identity instead. Fail closed: legacy entries without identity, or no live process, yield {} —
    a false "degraded forever" is the worse failure mode.
    """
    if writer_identity is None:
        return {}
    live_pid, live_start = writer_identity
    return {
        key: value for key, value in platforms.items()
        if isinstance(value, dict)
        and value.get("writer_pid") == live_pid
        and value.get("writer_start_time") == live_start}


def _collect_profile_gateway_topology() -> Dict[str, Any]:
    """Enumerate profiles and the gateways serving them for ``/api/status``.

    Returns ``profiles`` (all profile names via the cheap ``profiles_to_serve(True)`` chokepoint),
    ``gateways`` (one ``{"profile", "ports", "served_profiles"?}`` per LIVE gateway; liveness via
    ``_check_gateway_running`` so it agrees with the sidebar), ``gateway_mode``
    (multiplex / single / multiple / none) and ``profile_platforms`` — ownership-filtered runtime
    platform maps per live gateway, an internal aggregation input never exposed directly.
    """
    try:
        from hermes_cli.profiles import _check_gateway_running, profiles_to_serve
        from gateway.status import read_runtime_status
        homes = profiles_to_serve(True)
    except Exception:
        _log.debug("profile/gateway topology enumeration failed", exc_info=True)
        return {"profiles": [], "gateway_mode": "unknown", "gateways": [], "profile_platforms": {}}

    gateways: List[Dict[str, Any]] = []
    profile_platforms: Dict[str, dict] = {}
    multiplex = False
    for name, home in homes:
        try:
            if not _check_gateway_running(home):
                continue
        except Exception:
            continue
        try:
            runtime = read_runtime_status(home / "gateway_state.json")
        except Exception:
            runtime = None
        served = [str(p) for p in ((runtime or {}).get("served_profiles") or [])]
        if name == "default" and len(served) > 1:
            multiplex = True
        plats = (runtime or {}).get("platforms")
        if isinstance(plats, dict) and plats:
            owned = _owned_profile_platforms(_profile_gateway_writer_identity(home, runtime), plats)
            if owned:
                profile_platforms[name] = owned
        entry: Dict[str, Any] = {"profile": name, "ports": _profile_platform_ports(home, runtime)}
        if served:
            entry["served_profiles"] = served
        gateways.append(entry)

    if multiplex:
        mode = "multiplex"
    else:
        mode = {0: "none", 1: "single"}.get(len(gateways), "multiple")
    return {
        "profiles": [name for name, _home in homes],
        "gateway_mode": mode,
        "gateways": gateways,
        "profile_platforms": profile_platforms}


# /api/status is polled ~1/s by the desktop app while it waits for the backend. Each uncached
# collect walks 7+ profile homes (pure-Python yaml + psutil + realpath) in the default executor;
# concurrent polls pile up and hold the GIL for 14-16s, starving the loop so the desktop WS never
# gets gateway.ready. A short TTL cache with a collapse lock keeps the scan to one per window.
# The cache remembers which collector produced the entry: tests monkeypatch
# _collect_profile_gateway_topology per case, and a swapped collector is a miss (no reset hook).
_TOPOLOGY_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "fn": None}
_TOPOLOGY_CACHE_LOCK = threading.Lock()
_TOPOLOGY_CACHE_TTL = 10.0


def _topology_cache_get(fn: Any) -> Optional[Dict[str, Any]]:
    c = _TOPOLOGY_CACHE
    fresh = c["fn"] is fn and time.monotonic() - c["ts"] < _TOPOLOGY_CACHE_TTL
    return c["data"] if fresh and c["data"] is not None else None


def _collect_profile_gateway_topology_cached() -> Dict[str, Any]:
    fn = _collect_profile_gateway_topology
    cached = _topology_cache_get(fn)
    if cached is not None:
        return cached
    with _TOPOLOGY_CACHE_LOCK:
        cached = _topology_cache_get(fn)
        if cached is not None:
            return cached
        data = fn()
        _TOPOLOGY_CACHE.update(data=data, fn=fn, ts=time.monotonic())
        return data


def _load_configured_gateway_platforms() -> set[str]:
    """Connected platform names; synchronous by design — the first ``load_gateway_config()`` does
    platform discovery and can outlast Desktop's WS connect timeout on Windows, so ``get_status``
    runs this in Starlette's worker pool."""
    from gateway.config import load_gateway_config
    return {platform.value for platform in load_gateway_config().get_connected_platforms()}


_WINDOWS_11_MIN_BUILD = 22000


def _windows_build_number(version: str, platform_label: str) -> Optional[int]:
    """Extract the Windows NT build number from stdlib platform strings."""
    for value in (version or "", platform_label or ""):
        match = re.search(r"(?:^|[^\d])10\.0\.(\d{5,})(?:[^\d]|$)", value)
        if match:
            return int(match.group(1))
    return None


def _display_system_platform(*, system: str, release: str, version: str, platform_label: str) -> Dict[str, str]:
    """Host OS fields for display; Windows 10 builds >= 22000 are relabelled Windows 11."""
    if system == "Windows" and release == "10":
        build = _windows_build_number(version, platform_label)
        if build is not None and build >= _WINDOWS_11_MIN_BUILD:
            platform_label = re.sub(r"^Windows-10(?=-)", "Windows-11", platform_label, count=1)
            release = "11"
    return {"os": system, "os_release": release, "os_version": version, "platform": platform_label}


# Gateway + update actions (invoked from the Status page). Spawned detached so the request
# returns immediately; stdin is DEVNULL so stray input() fails fast; stdout/stderr stream to
# ~/.hermes/logs/<action>.log which the dashboard tails.

_ACTION_LOG_DIR: Path = get_hermes_home() / "logs"

# Short ``name`` (from the URL) → log file name under _ACTION_LOG_DIR.
_ACTION_LOG_FILES: Dict[str, str] = {
    "gateway-restart": "gateway-restart.log",
    "gateway-start": "gateway-start.log",
    "gateway-stop": "gateway-stop.log",
    "hermes-update": "hermes-update.log",
    **{name: f"action-{name}.log" for name in (
        "doctor", "security-audit", "backup", "import", "checkpoints-prune", "skills-install",
        "skills-uninstall", "skills-update", "curator-run", "prompt-size", "dump", "config-migrate",
        "tools-post-setup",
    )},
}

# ``name`` → most recent Popen handle / argv / action id, so ``status`` needs no ``ps``.
_ACTION_PROCS: Dict[str, subprocess.Popen] = {}
_ACTION_COMMANDS: Dict[str, Tuple[str, ...]] = {}
_ACTION_IDS: Dict[str, str] = {}
# ``name`` → synthetic result for actions handled without a subprocess (e.g. unsupported Docker updates).
_ACTION_RESULTS: Dict[str, Dict[str, Any]] = {}


def _terminate_desktop_managed_gateway() -> None:
    """Stop a live gateway restart child when its Desktop backend shuts down."""
    proc = _ACTION_PROCS.get("gateway-restart")
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
    except OSError:
        pass  # exited between poll() and terminate()


def _dashboard_spawn_executable() -> str:
    """Interpreter for detached dashboard actions: the install's venv python when it differs
    from ``sys.executable``, else ``sys.executable``.

    Under an SSH remote backend the server runs on the uv BASE interpreter with the venv's
    site-packages injected into sys.path at startup, so ``sys.executable`` is dependency-less and
    a detached child dies on its first third-party import; the venv launcher resolves the same
    dependency set on its own. Paths are compared UNRESOLVED: the venv python is typically a
    symlink to the base interpreter, so resolving would make them compare equal (exactly the
    case this fixes), and pyvenv.cfg discovery keys off argv0's unresolved location. On Windows
    the console python plus ``windows_detach_flags()`` keeps the action invisible without
    pythonw.exe (which makes every console descendant flash its own conhost).

    See #90026.
    Falls back to ``sys.executable`` when no venv interpreter exists next to the install (in-process dev
    runs, exotic layouts). See #54220, #56747.
    """
    from hermes_cli.web_server import PROJECT_ROOT
    exe = Path(sys.executable)
    try:
        for rel in ("venv/bin/python", "venv/Scripts/python.exe"):
            candidate = PROJECT_ROOT / rel
            if candidate.is_file():
                if os.path.normcase(os.path.normpath(str(candidate))) == (
                    os.path.normcase(os.path.normpath(str(exe)))):
                    return sys.executable
                return str(candidate)
    except OSError:
        pass
    return sys.executable


def _spawn_hermes_action(
    subcommand: List[str], name: str, *, env_overrides: Optional[Dict[str, str]] = None
) -> subprocess.Popen:
    """Spawn ``hermes <subcommand>`` detached (via ``hermes_cli.main``) and record the handle."""
    from hermes_cli.web_server import PROJECT_ROOT
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(_ACTION_LOG_DIR / _ACTION_LOG_FILES[name], "ab", buffering=0)
    log_file.write(f"\n=== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())

    cmd = [_dashboard_spawn_executable(), "-m", "hermes_cli.main", *subcommand]
    # The dashboard runs inside the gateway process, so os.environ carries _HERMES_GATEWAY=1;
    # inheriting it trips the child's in-process restart-loop guard (exit 1). Drop it, like
    # the gateway's own restart watcher does.
    # The gateway's own restart watcher already drops it (gateway/run.py); mirror that here (#52470).
    action_env = {**os.environ, "HERMES_NONINTERACTIVE": "1"}
    action_env.pop("_HERMES_GATEWAY", None)
    detach = {"creationflags": windows_detach_flags()} if sys.platform == "win32" else {"start_new_session": True}
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
        env={**action_env, **(env_overrides or {})}, **detach,
    )
    log_file.close()  # child holds its own dup'd fd; keeping ours leaks one per action
    _ACTION_RESULTS.pop(name, None)
    _ACTION_COMMANDS[name] = tuple(subcommand)
    _ACTION_PROCS[name] = proc
    action_id = (env_overrides or {}).get("HERMES_ACTION_ID")
    if action_id:
        _ACTION_IDS[name] = action_id
    else:
        _ACTION_IDS.pop(name, None)
    return proc


def _gateway_subcommand(profile: Optional[str], verb: str) -> List[str]:
    from hermes_cli.web_server_profiles import _profile_cli_args
    return _profile_cli_args(profile) + ["gateway", verb]


def _restart_gateway_after(profile: Optional[str], *, what: str, label: str) -> dict[str, Any]:
    """Best-effort gateway restart after a config change. The save stays authoritative: a failed
    spawn is reported (``restart_started: False`` + ``restart_error``) so the UI can fall back to
    its manual restart banner instead of failing the request."""
    from hermes_cli.web_server import _spawn_gateway_restart
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after %s", what)
        return {"restart_started": False, "restart_error": str(exc)}
    if reused:
        _log.info("%s: reusing in-flight gateway restart (pid %s)", label, proc.pid)
    return {"restart_started": True, "restart_action": "gateway-restart", "restart_pid": proc.pid}


def _split_text_for_speak_stream(text: str, cap: int) -> list:
    """Split *text* into provider-cap-sized pieces on sentence boundaries.

    Deliberately NOT unified with gateway.platforms.helpers' split_text_fence_aware: this
    reflows whitespace (sentences re-joined with single spaces) and has no fence semantics.
    """
    from tools.tts_streaming import SENTENCE_BOUNDARY_RE as _SENTENCE_BOUNDARY_RE
    cap = cap if cap and cap > 0 else 4000
    pieces, buf = [], ""
    for sentence in filter(str.strip, _SENTENCE_BOUNDARY_RE.split(text)):
        while len(sentence) > cap:
            pieces.append(sentence[:cap])
            sentence = sentence[cap:]
        if buf and len(buf) + len(sentence) + 1 > cap:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        pieces.append(buf)
    return pieces


# Per-row fields no session LIST consumer reads but that dominate the payload (``system_prompt``
# is the fully rendered prompt, tens of KB per row — 96% of a 528KB /api/sessions response).
# Detail reads stay complete; list callers that need full rows pass ``?full=1``.
_SESSION_LIST_HEAVY_FIELDS = ("system_prompt", "model_config")


def _strip_session_list_rows(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for s in sessions:
        for key in _SESSION_LIST_HEAVY_FIELDS:
            s.pop(key, None)
    return sessions
