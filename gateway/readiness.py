"""Bounded, non-destructive readiness probes for authenticated health surfaces."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from utils import load_yaml_file_readonly


_DISK_DEGRADED_PERCENT = 90.0
_CONNECTED_STATES = {"connected", "running", "ok"}


def _check(status: str, detail: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"status": status, **({"detail": detail} if detail else {}), **extra}


def _probe_state_db(home: Path) -> dict[str, Any]:
    """Read-only schema probe plus the process-wide corruption latch (``hermes_state_health``).

    The schema read only catches an unreadable header or schema; damage deeper in the file
    surfaces when a reader or writer touches it, and those publish into the latch. Reporting
    the latch here is what makes readiness and ``/api/status`` agree with the session list
    (#72046). ``detail="corrupt"`` is the one reason string consumers key off."""
    from hermes_state_health import STORAGE_CORRUPT, note_storage_error, storage_state

    path = home / "state.db"
    if not path.exists():
        return _check("ok", "not initialized")
    if storage_state(path) == STORAGE_CORRUPT:
        return _check("degraded", STORAGE_CORRUPT)
    try:
        # Read-only schema query: catches unreadable/corrupt DBs without competing with
        # writers. ``closing`` is required — sqlite3's context manager only commits/rolls
        # back, never closes, so a bare ``with connect()`` leaks a connection per poll.
        with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=1.0)) as conn:
            # A readiness probe must never compete with normal state writers. See #69567, #69678.
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        return _check("ok")
    except Exception as exc:
        if note_storage_error(path, exc):
            return _check("degraded", STORAGE_CORRUPT)
        return _check("degraded", type(exc).__name__)


def _probe_config(home: Path) -> dict[str, Any]:
    path = home / "config.yaml"
    if not path.exists():
        return _check("ok", "using defaults")
    try:
        raw = load_yaml_file_readonly(path)
    except Exception as exc:
        return _check("degraded", f"invalid config ({type(exc).__name__})")
    return _check("ok") if raw is None or isinstance(raw, dict) else _check("degraded", "top level is not a mapping")


def _probe_disk(home: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(home)
    except Exception as exc:
        return _check("degraded", type(exc).__name__)
    used_pct = round((usage.used / usage.total) * 100, 1) if usage.total else 0.0
    return _check("degraded" if used_pct >= _DISK_DEGRADED_PERCENT else "ok", used_percent=used_pct, free_bytes=usage.free)


def _probe_gateway(runtime_status: dict[str, Any]) -> dict[str, Any]:
    state = str(runtime_status.get("gateway_state") or "unknown")
    platforms = runtime_status.get("platforms")
    platforms = platforms if isinstance(platforms, dict) else {}
    connected = sum(
        isinstance(v, dict) and str(v.get("state") or v.get("status") or "").lower() in _CONNECTED_STATES
        for v in platforms.values()
    )
    return _check("ok" if state in {"running", "draining"} else "degraded", state=state,
                  connected_platforms=connected, platforms=len(platforms))


def _probe_session_store(runtime_status: dict[str, Any], state_db_probe: dict[str, Any]) -> dict[str, Any]:
    """Report the running gateway cache state, not an independent reopen.  A corrupt store is
    unavailable whatever the cache says: an open handle on a damaged file is not a working one."""
    if state_db_probe.get("detail") == "corrupt":
        return _check("unavailable", "corrupt")
    runtime_store = runtime_status.get("session_store")
    state = str(runtime_store.get("status") or "unknown") if isinstance(runtime_store, dict) else ""
    if state in {"ok", "unavailable", "retrying"}:
        return _check(state)
    # Older gateways publish no cache state: fall back to the state_db probe.
    return _check("ok" if state_db_probe.get("status") == "ok" else "unavailable")


# The local import path a turn needs: the provider client stack (compiled extensions
# included) plus each enabled platform's adapter module. Checked here rather than by calling
# a provider, so a probe can never spend money, read a credential, or touch the network.
_CLIENT_STACK_MODULES = ("openai", "pydantic_core", "pydantic", "httpx")


def _import_for_probe(name: str) -> Any:
    """The one import path every dependency probe uses.

    Imports run in-process on purpose: this process's ``sys.path`` and extension ABI are
    exactly what a request would use, so a subprocess would answer for the wrong interpreter.
    """
    import importlib

    return importlib.import_module(name)


def _import_failures(modules: Iterable[str]) -> dict[str, str]:
    """{module: exception class} for modules that cannot import here.

    Exception *text* is deliberately dropped: it can embed local paths and DLL names, and this
    payload is served over an authenticated HTTP surface.
    """
    failures: dict[str, str] = {}
    for name in modules:
        try:
            _import_for_probe(name)
        except Exception as exc:  # noqa: BLE001 - any import error is the finding
            failures[name] = type(exc).__name__
    return failures


def _adapter_modules(platforms: Iterable[str]) -> dict[str, str]:
    """{platform: module} for platforms the gateway serves through a builtin adapter.

    Read from the gateway's own adapter table so the probe checks the module the gateway
    would import for that platform, never a second list that can drift.
    """
    try:
        from gateway.run import _BUILTIN_ADAPTERS
    except Exception:  # noqa: BLE001 - a gateway that will not import is reported by other checks
        return {}
    by_value = {platform.value: spec[0] for platform, spec in _BUILTIN_ADAPTERS.items()}
    return {str(name): f"gateway.platforms.{by_value[str(name)]}" for name in platforms if str(name) in by_value}


def _probe_provider_client() -> dict[str, Any]:
    failures = _import_failures(_CLIENT_STACK_MODULES)
    if failures:
        return _check("degraded", ", ".join(f"{name}: {error}" for name, error in sorted(failures.items())))
    return _check("ok")


def _probe_adapter_imports(platforms: Iterable[str]) -> dict[str, Any]:
    modules = _adapter_modules(platforms)
    failures = _import_failures(modules.values())
    if failures:
        broken = sorted(platform for platform, module in modules.items() if module in failures)
        detail = ", ".join(f"{name}: {failures[modules[name]]}" for name in broken)
        return _check("degraded", detail, unimportable=broken)
    return _check("ok")


def _probe_environment(project_root: Path) -> dict[str, Any]:
    """Identity of the running interpreter and of the dependency generation it selected.

    Reports booleans and version strings only — never an environment path.
    """
    identity: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "abi": sys.implementation.cache_tag,
    }
    try:
        # Import inside the guard: a broken dependency-generation module must degrade the probe,
        # never propagate out and 500 the authenticated ``/health/detailed`` surface.
        from pm.environments import committed_venv, running_from_selected_environment

        committed = committed_venv(project_root) is not None
        on_selected = running_from_selected_environment(project_root)
    except Exception as exc:  # noqa: BLE001 - an unreadable/unimportable record is itself the finding
        return _check("degraded", f"dependency environment probe failed ({type(exc).__name__})", **identity)
    identity.update({"committed_generation": committed, "on_selected_environment": on_selected})
    if committed and not on_selected:
        return _check("degraded", "not running from the committed dependency environment", **identity)
    return _check("ok", **identity)


def collect_dependency_readiness(
    *, platforms: Iterable[str] = (), project_root: Path | None = None,
) -> dict[str, Any]:
    """Can this interpreter import what a served turn needs?  Side-effect free.

    Imports and reads only: no provider call, no credential read, no user config read, no
    socket. A process can answer ``/health`` while failing every request, so callers that
    want to know whether a gateway can actually serve a turn must ask this instead of the
    liveness route (verified incident: a bound API server whose ``pydantic_core`` import
    raised under the running interpreter).
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    checks = {
        "provider_client": _probe_provider_client(),
        "adapters": _probe_adapter_imports(platforms),
        "environment": _probe_environment(root),
    }
    return {"status": "ok" if all(c["status"] == "ok" for c in checks.values()) else "degraded", "checks": checks}


def collect_runtime_readiness(
    *, configured_model: str, runtime_status: dict[str, Any] | None, active_api_runs: int = 0,
    process_completion_queue_depth: int = 0, active_delegations: int = 0,
    platforms: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Bounded readiness diagnostics, no runtime mutation.  Even authenticated, probes
    expose status and counts only: never config values, credentials, paths, payloads.

    ``platforms`` is the enabled/configured platform set to check adapter imports for; it
    defaults to whatever the runtime status reports as running platforms.
    """
    home = get_hermes_home()
    runtime = runtime_status if isinstance(runtime_status, dict) else {}
    if platforms is None:
        reported = runtime.get("platforms")
        platforms = tuple(reported) if isinstance(reported, dict) else ()
    state_db_probe = _probe_state_db(home)
    checks = {
        "state_db": state_db_probe,
        "session_store": _probe_session_store(runtime, state_db_probe),
        "config": _probe_config(home),
        "model": _check("ok" if str(configured_model or "").strip() else "degraded"),
        "disk": _probe_disk(home),
        "gateway": _probe_gateway(runtime),
        "background_queues": _check(
            "ok", active_api_runs=max(0, int(active_api_runs)),
            process_completions=max(0, int(process_completion_queue_depth)),
            active_delegations=max(0, int(active_delegations)),
        ),
        "dependencies": collect_dependency_readiness(platforms=platforms),
    }
    return {"status": "ok" if all(c.get("status") == "ok" for c in checks.values()) else "degraded", "checks": checks}


__all__ = ["collect_dependency_readiness", "collect_runtime_readiness"]
