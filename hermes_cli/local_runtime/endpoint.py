"""Endpoint resolution for llamacpp-alias requests (provider integration).

``provider: llamacpp`` with no explicit base_url resolves, in order, to the managed server (state
file), a detected external llama-server, or — during a backend boot race — the managed server once
its state file appears.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import threading
import time
import urllib.request

LLAMACPP_ALIASES = frozenset({"llamacpp", "llama.cpp", "llama-cpp"})

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    """Liveness for the state file's supervisor-child pid: psutil when available, else True
    (optimistic). On Windows ``os.kill(pid, 0)`` TERMINATES the process — never use it as a probe."""
    if not pid or pid < 0:
        return False
    with suppress(Exception):
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    return True


def _state_endpoint() -> dict | None:
    from hermes_cli.local_runtime.supervisor import state_path

    path = state_path()
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    base_url = state.get("base_url", "")
    if not base_url:
        return None
    # Ownership proof: on the stable port a SECOND install (different HERMES_HOME) can own
    # 127.0.0.1:18434 with a different api key while this install's state file still points
    # there. /health is public and answers 200 for ANYONE's server — trusting it sent every
    # request at a server that 401s our key, silently — so the recorded supervisor pid is the
    # ONLY tiebreaker: a live pid is ours (healthy, or STARTING — state is written at spawn, and
    # readiness probes racing the boot must see a configured provider, not missing credentials);
    # a dead pid is a crashed-without-cleanup leftover, ignored so requests don't blackhole.
    if not _pid_alive(int(state.get("pid") or 0)):
        return None
    return {"base_url": base_url, "api_key": state.get("api_key", "")}


def managed_root() -> "tuple[str, str] | None":
    """(base_root, api_key) of the managed router, or None. Resolved through the
    ownership-guarded reader, not a raw state-file read: on the shared stable port a foreign
    install's server answers /health for anyone, and a raw read would attach callers to someone
    else's server."""
    with suppress(Exception):
        state = _state_endpoint()
        if state is None:
            return None
        base = str(state.get("base_url", "")).rsplit("/v1", 1)[0]
        return (base, str(state.get("api_key", ""))) if base else None
    return None


def managed_get_json(base: str, api_key: str, route: str, timeout_s: float) -> object:
    """Authenticated GET against the managed router; raises on any transport/decode failure."""
    req = urllib.request.Request(f"{base}{route}",
                                 headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def resolve_llamacpp_endpoint(config: dict | None = None,
                              wait_for_boot_s: float = 8.0) -> dict | None:
    """Managed-first, detection-second endpoint for llamacpp aliases.

    Boot-race rung: on a fresh backend start there is NO state file yet — the lifespan boot thread
    is still spawning the server (≈1-3 s) while the desktop's readiness probe fires the moment the
    WebSocket connects.
    """
    managed = _state_endpoint()
    if managed:
        return managed

    from hermes_cli.local_runtime.detect import detect_server

    ports = ((config or {}).get("local_runtime") or {}).get("detect_ports") or []
    hit = detect_server(extra_ports=tuple(int(p) for p in ports))
    if hit and not hit.auth_required:
        return {"base_url": hit.base_url, "api_key": ""}

    if wait_for_boot_s > 0 and _boot_in_flight(config):
        _kick_managed_boot(config)
        deadline = time.monotonic() + wait_for_boot_s
        while time.monotonic() < deadline:
            time.sleep(0.25)
            managed = _state_endpoint()
            if managed:
                return managed
    return None


_KICK_LOCK = threading.Lock()


def _load_config_if_none(config: dict | None) -> dict | None:
    if config is not None:
        return config
    from hermes_cli.config import load_config

    return load_config()


def _kick_managed_boot(config: dict | None) -> None:
    """Actively start the managed server when resolution finds it missing — the wait loop assumes
    some OTHER thread is bringing it up, which is true only at backend start."""
    if not _KICK_LOCK.acquire(blocking=False):
        return  # a kick is already in flight

    def _boot() -> None:
        try:
            from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

            ensure_local_runtime(_load_config_if_none(config))
        except Exception:  # noqa: BLE001 — best-effort; resolution falls back
            logger.warning("on-demand managed-server boot failed", exc_info=True)
        finally:
            _KICK_LOCK.release()

    threading.Thread(target=_boot, daemon=True,
                     name="lr-on-demand-boot").start()


def _boot_in_flight(config: dict | None) -> bool:
    """True when the managed runtime is enabled and installed (a verified-manifest scan under
    runtimes_root(), NOT a bare ``server_binary()`` call — that needs an install_dir, and calling
    it bare once made this gate throw-and-return False forever, disabling the boot wait)."""
    with suppress(Exception):
        config = _load_config_if_none(config)
        if not ((config or {}).get("local_runtime") or {}).get("enabled"):
            return False
        from hermes_cli.local_runtime.binaries import manifest_verified, runtimes_root

        return any(manifest_verified(m) for m in runtimes_root().glob("*/*/manifest.json"))
    return False
