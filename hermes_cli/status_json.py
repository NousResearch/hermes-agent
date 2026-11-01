"""``hermes status --json``: the status facts as one stable, documented JSON document.

Shape (``schema_version`` 1) is documented in website/docs/reference/cli-commands.md
§ ``hermes status``. Secrets never appear: API keys are reported by provider name only.
A probe that fails degrades only its own section to ``null`` and records why in ``errors``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def _estop(_config: dict) -> dict[str, Any]:
    from agent.estop import get_state

    state = get_state() or {}
    return {"paused": bool(state), "reason": state.get("reason"), "timestamp": state.get("timestamp")}


def _gateway(_config: dict) -> dict[str, Any]:
    from hermes_cli.gateway import get_gateway_runtime_snapshot, named_profile_served_by_running_multiplexer

    snapshot = get_gateway_runtime_snapshot()
    via_multiplexer = not snapshot.running and named_profile_served_by_running_multiplexer()
    return {
        "running": bool(snapshot.running or via_multiplexer),
        "via_default_multiplexer": bool(via_multiplexer),
        "manager": snapshot.manager,
        "pids": list(snapshot.gateway_pids or []),
        "service_installed": bool(snapshot.service_installed),
        "service_running": bool(snapshot.service_running),
    }


def _terminal(config: dict) -> dict[str, Any]:
    terminal_cfg = config.get("terminal")
    backend = terminal_cfg.get("backend", "local") if isinstance(terminal_cfg, dict) else "local"
    return {"backend": os.getenv("TERMINAL_ENV", "") or backend,
            "sudo": os.getenv("SUDO_PASSWORD", "") != ""}


def _auth(_config: dict) -> dict[str, Any]:
    from hermes_cli import auth
    from hermes_cli.status import _first_env_value
    from hermes_cli.status_auth import _API_KEYS, _OAUTH_BLOCKS

    api_keys = [name for name, env in _API_KEYS.items() if _first_env_value(env)]
    if auth.get_anthropic_key():
        api_keys.append("Anthropic")
    oauth = {}
    for name, getter_name, _hint, _rows in _OAUTH_BLOCKS:
        st = getattr(auth, getter_name)() or {}
        logged_in = bool(st.get("logged_in"))
        oauth[name] = {"logged_in": logged_in, "last_refresh": st.get("last_refresh"),
                       "error": None if logged_in else st.get("error")}
    return {"api_keys": api_keys, "oauth": oauth}


def _platforms(_config: dict) -> dict[str, Any]:
    """Same verdict as the text renderer: the gateway's ``get_connected_platforms``."""
    from gateway.config import load_gateway_config
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()  # plugin platforms register their labels here
    cfg = load_gateway_config()
    connected = {p.value for p in cfg.get_connected_platforms()}
    homes = {p.value: pc.home_channel for p, pc in cfg.platforms.items() if pc.home_channel}
    return {
        entry.name: {"label": entry.label, "configured": entry.name in connected,
                     "home_channel": homes[entry.name].chat_id if entry.name in homes else None}
        for entry in platform_registry.plugin_entries()
    }


def _cron(_config: dict) -> dict[str, Any]:
    from hermes_constants import get_hermes_home

    jobs_file = get_hermes_home() / "cron" / "jobs.json"
    if not jobs_file.exists():
        return {"jobs": 0, "enabled": 0}
    # utf-8-sig: same dialect as cron/jobs.load_jobs (Windows editors leave a BOM).
    jobs = json.loads(jobs_file.read_text(encoding="utf-8-sig")).get("jobs", [])
    return {"jobs": len(jobs), "enabled": sum(1 for j in jobs if j.get("enabled", True))}


def _session_store(_config: dict) -> dict[str, Any]:
    """Counts and usage from this profile's state.db, opened read-only (never creates it)."""
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    if not (get_hermes_home() / "state.db").exists():
        return {"total": 0, "gateway_active": 0, "last_activity": None, "recent": [],
                "usage": {"tokens": 0, "cost_usd": 0.0}}
    db = SessionDB(read_only=True)
    try:
        gateway_rows = db.list_gateway_sessions(active_only=True) or []
        freshest = max((float(r.get("last_active") or 0) for r in gateway_rows), default=0.0)
        recent = [{k: r.get(k) for k in ("id", "title", "source", "model", "started_at", "last_active")}
                  for r in db.list_recent_sessions_bounded(limit=10) or []]
        return {"total": db.session_count(), "gateway_active": len(gateway_rows),
                "last_activity": freshest or None, "recent": recent, "usage": db.usage_totals()}
    finally:
        db.close()


def _profile_homes() -> list[tuple[str, Path]]:
    """Every live profile home (leases are written to the profile that owns the session),
    plus the current home when it sits outside the root (a custom ``HERMES_HOME``)."""
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import get_default_hermes_root, get_hermes_home, named_profile_is_live

    root = get_default_hermes_root()
    homes = [("default", root)]
    profiles_dir = root / "profiles"
    if profiles_dir.is_dir():
        homes.extend((p.name, p) for p in sorted(profiles_dir.iterdir()) if named_profile_is_live(p))
    current = get_hermes_home()
    if current.resolve() not in {home.resolve() for _name, home in homes}:
        homes.append((get_active_profile_name(), current))
    return homes


def _row_for(home: Path, session_id: str) -> dict[str, Any]:
    if not (home / "state.db").exists():
        return {}
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        return db.get_session(session_id) or {}
    finally:
        db.close()


def live_sessions() -> dict[str, Any]:
    """Live sessions on this host from the lease registry, across every profile.

    A lease is held by the owning process (CLI, TUI/Desktop for the session's lifetime,
    a gateway for the turn in flight) and reads dead the moment that process exits, so a
    crashed CLI never shows here. ``unreadable`` lists profiles whose registry is corrupt.
    """
    from hermes_cli.active_sessions import ActiveSessionRegistryError, read_live_session_leases

    sessions: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for profile, home in _profile_homes():
        try:
            leases = read_live_session_leases(home)
        except (ActiveSessionRegistryError, OSError):
            unreadable.append(profile)
            continue
        for lease in leases:
            sid = str(lease.get("session_id") or "")
            row = _row_for(home, sid)
            sessions.append({
                "session_id": sid, "profile": profile, "surface": lease.get("surface"),
                "pid": lease.get("pid"), "started_at": lease.get("started_at"),
                "source": row.get("source"), "model": row.get("model"), "title": row.get("title"),
            })
    sessions.sort(key=lambda s: s["started_at"] or 0)
    return {"sessions": sessions, "unreadable": unreadable}


def _slots(config: dict) -> dict[str, Any] | None:
    from hermes_cli.active_sessions import active_session_registry_snapshot, resolve_max_concurrent_sessions

    cap = resolve_max_concurrent_sessions(config)
    return None if cap is None else {"in_use": len(active_session_registry_snapshot()), "cap": cap}


_SECTIONS: dict[str, Callable[[dict], Any]] = {
    "estop": _estop, "gateway": _gateway, "terminal": _terminal, "auth": _auth,
    "platforms": _platforms, "cron": _cron, "session_store": _session_store,
    "live": lambda _config: live_sessions(), "slots": _slots,
}


def get_status_data() -> dict[str, Any]:
    """Collect every status fact as a JSON-serializable dict (see module docstring)."""
    import hermes_cli
    from hermes_cli.config import get_env_path, load_config
    from hermes_cli.profiles import get_active_profile_name
    from hermes_cli.status import PROJECT_ROOT, _configured_model_label, _effective_provider_label
    from hermes_constants import get_hermes_home

    config = load_config()
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "version": getattr(hermes_cli, "__version__", "unknown"),
        "project_root": str(PROJECT_ROOT),
        "python_version": sys.version.split()[0],
        "hermes_home": str(get_hermes_home()),
        "profile": get_active_profile_name(),
        "env_file_exists": get_env_path().exists(),
        "model": _configured_model_label(config),
        "provider": _effective_provider_label(),
    }
    errors: dict[str, str] = {}
    for name, probe in _SECTIONS.items():
        try:
            data[name] = probe(config)
        except Exception as exc:  # health: allow BLE001 -- per-section boundary; reason lands in "errors"
            logger.debug("status --json probe %s failed", name, exc_info=True)
            data[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
    store = data.pop("session_store") or {}
    live = data.pop("live") or {}
    data["sessions"] = {
        "live": live.get("sessions", []), "registry_unreadable": live.get("unreadable", []),
        "slots": data.pop("slots"), "gateway_active": store.get("gateway_active"),
        "total": store.get("total"), "last_activity": store.get("last_activity"),
        "recent": store.get("recent", []),
    }
    data["usage"] = store.get("usage")
    data["errors"] = errors
    return data


def print_status_json() -> int:
    print(json.dumps(get_status_data(), indent=2, default=str))
    return 0
