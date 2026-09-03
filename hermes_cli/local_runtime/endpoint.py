"""Endpoint resolution for llamacpp-alias requests (provider integration).

The seam between the existing provider mechanism and the managed runtime:
``provider: llamacpp`` with no explicit base_url resolves, in order, to

1. the managed server this Hermes is supervising (state file written by
   LlamaServerSupervisor.start, removed on stop, staleness-checked), or
2. a detected external llama-server.

Returns None when neither exists — the caller falls through to the normal
custom-provider path and its own error reporting.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

LLAMACPP_ALIASES = frozenset({"llamacpp", "llama.cpp", "llama-cpp"})

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    """Liveness for the state file's supervisor-child pid.

    psutil when available; otherwise fall back to True (optimistic) — on
    Windows ``os.kill(pid, 0)`` TERMINATES the process, so it must never be
    used as a probe (windows-git-bash interop pitfall).
    """
    if not pid or pid < 0:
        return False
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
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
    endpoint = {"base_url": base_url, "api_key": state.get("api_key", "")}
    # Ownership proof: the stable port means a SECOND install (different
    # HERMES_HOME — a scratch profile, say) can own 127.0.0.1:18434 with a
    # different api key while this install's state file still points there.
    # /health is a public route, so it answers 200 for ANYONE's server —
    # trusting it alone sent every chat request and the load-progress
    # watcher at a server that 401s our key, silently. The recorded
    # supervisor pid is the tiebreaker: health-200 from a server whose
    # recorded child is DEAD is someone else's server, never a starting one.
    pid_ok = _pid_alive(int(state.get("pid") or 0))
    # Healthy server: done (when it's ours).
    try:
        health = base_url.rsplit("/v1", 1)[0] + "/health"
        with urllib.request.urlopen(health, timeout=3) as r:
            if r.status == 200:
                return endpoint if pid_ok else None
    except (urllib.error.URLError, OSError, TimeoutError):
        pass
    # Not healthy YET: a live supervisor child is a STARTING server (state
    # is written at spawn; llama-server takes seconds to listen). Resolve
    # optimistically so readiness probes racing the boot see a configured
    # provider, not missing credentials. A dead pid is a crashed-without-
    # cleanup leftover — ignore it so requests don't blackhole.
    if pid_ok:
        return endpoint
    return None


def resolve_llamacpp_endpoint(config: dict | None = None,
                              wait_for_boot_s: float = 8.0) -> dict | None:
    """Managed-first, detection-second endpoint for llamacpp aliases.

    Returns {"base_url", "api_key"} or None. api_key is empty for keyless
    external servers (callers substitute the SDK placeholder).

    Boot-race rung: on a fresh backend start there is NO state file yet —
    the lifespan boot thread is still spawning the server (config load +
    preset generation + spawn ≈ 1-3 s) while the desktop's readiness probe
    fires the moment the WebSocket connects. When the runtime is enabled
    and installed, a missing endpoint means BOOTING, not unconfigured:
    poll briefly for the state file instead of failing the probe (twice
    observed as 'no usable credentials' → onboarding on restart).
    """
    managed = _state_endpoint()
    if managed:
        return managed

    from hermes_cli.local_runtime.detect import detect_server

    extra = ()
    if config:
        ports = (config.get("local_runtime") or {}).get("detect_ports") or []
        extra = tuple(int(p) for p in ports)
    hit = detect_server(extra_ports=extra)
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


def _kick_managed_boot(config: dict | None) -> None:
    """Actively start the managed server when resolution finds it missing.

    The wait loop above assumes some OTHER thread is bringing the server
    up — true only at backend start (the lifespan boot thread). A router
    that dies LATER leaves no boot in flight: the backend process was
    killed with the router as part of its tree, or another install took
    the stable port and the ownership guard rightly refused it. In those
    states the wait just expired and agent init failed with 'no provider
    configured', even though the fix is the same idempotent ensure call
    the lifespan makes. Kick it here, off-thread (the resolver's wait
    stays bounded; ensure's own state checks make a concurrent lifespan
    boot harmless) and non-reentrant (racing resolutions kick once).
    """
    if not _KICK_LOCK.acquire(blocking=False):
        return  # a kick is already in flight

    def _boot() -> None:
        try:
            cfg = config
            if cfg is None:
                from hermes_cli.config import load_config

                cfg = load_config()
            from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

            ensure_local_runtime(cfg)
        except Exception:  # noqa: BLE001 — best-effort; resolution falls back
            logger.warning("on-demand managed-server boot failed", exc_info=True)
        finally:
            _KICK_LOCK.release()

    threading.Thread(target=_boot, daemon=True,
                     name="lr-on-demand-boot").start()


def _boot_in_flight(config: dict | None) -> bool:
    """True when the managed runtime is enabled and installed — the state
    a lifespan boot thread is (or is about to be) bringing up.

    Installed-ness is a verified-manifest scan under runtimes_root(), NOT a
    server_binary() call — that helper requires an install_dir argument, and
    calling it bare made this gate throw-and-return-False forever, silently
    disabling the boot wait (the regression
    test had monkeypatched this function instead of exercising it).
    """
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        if not ((config or {}).get("local_runtime") or {}).get("enabled"):
            return False
        import json as _json

        from hermes_cli.local_runtime.binaries import runtimes_root

        for manifest in runtimes_root().glob("*/*/manifest.json"):
            try:
                if _json.loads(manifest.read_text(encoding="utf-8")).get("verified_version"):
                    return True
            except (ValueError, OSError):
                continue
        return False
    except Exception:  # noqa: BLE001
        return False
