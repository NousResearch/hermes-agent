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
import time
import urllib.error
import urllib.request

LLAMACPP_ALIASES = frozenset({"llamacpp", "llama.cpp", "llama-cpp"})


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
    # Healthy server: done.
    try:
        health = base_url.rsplit("/v1", 1)[0] + "/health"
        with urllib.request.urlopen(health, timeout=3) as r:
            if r.status == 200:
                return endpoint
    except (urllib.error.URLError, OSError, TimeoutError):
        pass
    # Not healthy YET: a live supervisor child is a STARTING server (state
    # is written at spawn; llama-server takes seconds to listen). Resolve
    # optimistically so readiness probes racing the boot see a configured
    # provider, not missing credentials. A dead pid is a crashed-without-
    # cleanup leftover — ignore it so requests don't blackhole.
    if _pid_alive(int(state.get("pid") or 0)):
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
        deadline = time.monotonic() + wait_for_boot_s
        while time.monotonic() < deadline:
            time.sleep(0.25)
            managed = _state_endpoint()
            if managed:
                return managed
    return None


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
