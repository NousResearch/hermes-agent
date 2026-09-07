"""Profile-scoped helpers: profile discovery fallback, profile dir/MCP-server writes,
the profile/config scope context managers, skills-hub and tools/analytics catalog helpers.
"""

import logging
import hashlib
import os
import re
import sys
import threading
from contextlib import contextmanager
from fastapi import HTTPException
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from hermes_cli.config import DEFAULT_CONFIG, get_process_hermes_home
from hermes_cli.web_models import MCPServerCreate
from hermes_cli.web_server_gateway import _ACTION_LOG_FILES
from hermes_cli.web_server_mcp import _normalize_mcp_server_create

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


def _safe(callable_: Callable[[], Any], default: Any) -> Any:
    try:
        return callable_()
    except Exception:
        return default


def _is_current_profile(profile: Optional[str]) -> bool:
    """None/""/"current" all mean the dashboard's own profile."""
    requested = (profile or "").strip()
    return not requested or requested.lower() == "current"


@contextmanager
def _hermes_home_scope(path) -> Any:
    """Scope ``load_config``/``save_config`` (anything resolving ``get_hermes_home()`` at call
    time) to ``path`` for the block via the context-local HERMES_HOME override."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(str(path))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _is_other_profile(profile: Optional[str]) -> bool:
    """True when ``profile`` names a profile other than this process's own."""
    if _is_current_profile(profile):
        return False
    try:
        target = _resolve_profile_dir(profile.strip())
    except HTTPException:
        return True
    return target.resolve() != get_process_hermes_home().resolve()


def _approval_mode_of(config: Dict[str, Any]) -> str:
    """Normalize approvals.mode from an in-memory config document. Both sides of the
    broadcast comparison use in-memory documents: re-reading through the config cache after
    a save can serve the pre-save document when the replacement file collides on the
    (mtime_ns, size) cache key, suppressing the broadcast exactly when the mode changed."""
    from tools.approval_context import _normalize_approval_mode
    approvals = config.get("approvals")
    default_mode = (DEFAULT_CONFIG.get("approvals") or {}).get("mode", "manual")
    mode = approvals.get("mode", default_mode) if isinstance(approvals, dict) else default_mode
    return _normalize_approval_mode(mode)


def _broadcast_gateway_session_info() -> None:
    """Broadcast session.info on the in-process gateway when it's loaded (``sys.modules``
    guard, not an import: gateway never imported means no live sessions to notify)."""
    server = sys.modules.get("tui_gateway.server")
    if server is None:
        return
    try:
        server.broadcast_session_info()
    except Exception:
        _log.exception("session.info broadcast after config save failed")


def _parse_model_ids(resp: "Any") -> List[str]:
    """Model ids from an OpenAI-compatible ``/v1/models`` response: ``{"data": [{"id": ..}]}``
    or a bare ``{"data": ["id", ..]}``. ``[]`` on any parse/HTTP error so a slightly
    non-standard endpoint never hard-blocks."""
    try:
        if not resp.is_success:
            return []
        payload = resp.json()
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    ids = [str((item.get("id") if isinstance(item, dict) else item) or "").strip() for item in data]
    return [mid for mid in ids if mid]


def _fallback_profile_entry(profiles_mod, name: str, home: Path, *, is_default: bool,
                            has_env: bool, gateway_running: Callable[[], bool]) -> Dict[str, Any]:
    model, provider = _safe(lambda: profiles_mod._read_config_model(home), (None, None))
    meta = lambda key, default: _safe(  # noqa: E731
        lambda: profiles_mod.read_profile_meta(home).get(key, default), default)
    return {
        "name": name, "path": str(home), "is_default": is_default, "model": model,
        "provider": provider, "has_env": has_env,
        "skill_count": _safe(lambda: profiles_mod._count_skills(home), 0),
        "gateway_running": _safe(gateway_running, False),
        "description": meta("description", ""), "description_auto": meta("description_auto", False),
        "distribution_name": None, "distribution_version": None, "distribution_source": None,
        "has_alias": False}


def _fallback_profile_dicts(profiles_mod) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    default_home = profiles_mod._get_default_hermes_home()
    if default_home.is_dir():
        profiles.append(_fallback_profile_entry(
            profiles_mod, "default", default_home, is_default=True,
            has_env=(default_home / ".env").exists(),
            gateway_running=lambda: profiles_mod._check_gateway_running(default_home)))

    profiles_root = profiles_mod._get_profiles_root()
    if profiles_root.is_dir():
        # os.scandir (context-managed) rather than Path.iterdir: an exception mid-iteration
        # must not leak the directory fd — the sidebar polls every few seconds, so a leak
        # exhausts RLIMIT_NOFILE within days.
        with os.scandir(profiles_root) as scan:
            entries = sorted(scan, key=lambda e: e.name)
        for entry in entries:
            home = Path(entry.path)
            if not entry.is_dir() or not profiles_mod._PROFILE_ID_RE.match(entry.name):
                continue
            profiles.append(_fallback_profile_entry(
                profiles_mod, entry.name, home, is_default=False,
                has_env=_safe(lambda: (home / ".env").exists(), False),
                gateway_running=lambda home=home, name=entry.name: (
                    profiles_mod._check_gateway_running(home)
                    or profiles_mod._served_by_running_multiplexer(name))))
    return profiles


def _resolve_profile_dir(name: str) -> Path:
    """Validate ``name`` and resolve to its directory or raise an HTTPException."""
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.validate_profile_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' does not exist.")
    return profiles_mod.get_profile_dir(name)


def _write_profile_mcp_servers(profile_dir: Path, servers: List["MCPServerCreate"]) -> int:
    """Write MCP server entries into ``profile_dir``'s config.yaml (HERMES_HOME-scoped).

    Mirrors the per-server shape ``POST /api/mcp/servers`` builds, batched so the whole
    profile-create write is one config save. Returns the number of servers written.
    """
    from hermes_cli.config import load_config, save_config
    from hermes_cli.mcp_config import _save_bearer_auth_token
    written = 0
    with _hermes_home_scope(profile_dir):
        cfg = load_config()
        mcp = cfg.setdefault("mcp_servers", {})
        for server in servers:
            try:
                name, entry, bearer_token = _normalize_mcp_server_create(server)
            except ValueError as exc:
                display_name = (server.name or "").strip() or "<unnamed>"
                _log.warning("Profile-create: skipping MCP server '%s': %s", display_name, exc)
                continue
            if bearer_token is not None:
                entry["headers"] = _save_bearer_auth_token(name, bearer_token)
            mcp[name] = entry
            written += 1
        if written:
            save_config(cfg)
        elif not mcp:
            # Don't leave the stray empty key we just created in the new profile's config.
            cfg.pop("mcp_servers", None)
            save_config(cfg)
    return written


# Skills & Tools endpoints accept an optional ``profile`` query param so the dashboard can
# manage ANY profile's skills/toolsets ("Set as active" only flips the sticky active_profile
# file for FUTURE invocations, which misled users into thinking toggles landed there).


_SKILLS_PROFILE_LOCK = threading.RLock()


@contextmanager
def _profile_scope(profile: Optional[str]):
    """Scope config + skill-directory resolution to ``profile`` for one request.

    Two seams: (1) ``load_config``/``save_config`` resolve ``get_hermes_home()`` at call
    time, so the contextvar override reaches them; (2) ``tools.skills_tool`` /
    ``tools.skill_manager_tool`` bind ``SKILLS_DIR`` at import time, so both are retargeted
    under a lock and restored after. For the dashboard's own profile config resolution is
    untouched, but the skill-module globals are still retargeted to the *current*
    ``get_hermes_home()`` so writes land in the live home even when the import-time binding
    is stale (test isolation, late HERMES_HOME override). Yields the profile dir for a named
    profile, None for the current one.

    ``tools.skills_sync`` (reset/diff/list-modified/opt-in/opt-out/ repair-official) needs NO retargeting:
    since #65828 its directory lookups resolve at call time through the same contextvar override set in step
    1.
    """
    from hermes_constants import get_hermes_home
    from tools import skills_tool as _skills_tool
    from tools import skill_manager_tool as _skill_mgr
    with _config_profile_scope(profile) as scoped:
        profile_dir = get_hermes_home() if scoped is None else scoped
        modules = (_skills_tool, _skill_mgr)
        with _SKILLS_PROFILE_LOCK:
            saved = [(m.HERMES_HOME, m.SKILLS_DIR) for m in modules]
            for m in modules:
                m.HERMES_HOME, m.SKILLS_DIR = profile_dir, profile_dir / "skills"
            try:
                yield scoped
            finally:
                for m, (home, skills_dir) in zip(modules, saved):
                    m.HERMES_HOME, m.SKILLS_DIR = home, skills_dir


@contextmanager
def _config_profile_scope(profile: Optional[str]):
    """Await-safe, config-only profile scope: touches ONLY the task-local HERMES_HOME
    contextvar, never the process-global skills-module attributes ``_profile_scope`` swaps
    (holding those across an ``await`` lets a concurrent request restore THIS request's dir
    on its ``finally``). None/""/"current" = no override."""
    if _is_current_profile(profile):
        yield None
        return
    profile_dir = _resolve_profile_dir(profile.strip())
    with _hermes_home_scope(profile_dir):
        yield profile_dir


# Terminal backend picker rows — GUI counterpart of terminal.backend. Keep in sync with
# tools/terminal_tool.py::_create_environment and the terminal.backend enum.

# --------------------------------------------------------------------------- Terminal execution backend
# picker — the GUI counterpart of terminal.backend in config.yaml. Each row carries a fast, defensive health
# probe (Docker daemon reachable, SSH host configured, Modal/Daytona credentials present) so the
# Capabilities panel can render Ready / Needs setup guidance instead of a bare enum (issues #57738 /
# #63783). Probes must never raise — a probe failure renders as a status, not a 500.
# ---------------------------------------------------------------------------
_TERMINAL_BACKENDS: List[Dict[str, str]] = [
    dict(zip(("name", "label", "description"), row)) for row in (
        ("local", "Local", "Run commands directly on this machine. No isolation."),
        ("docker", "Docker",
         "Run commands in an isolated Docker container with a persistent workspace."),
        ("singularity", "Singularity / Apptainer",
         "Run commands in a Singularity/Apptainer container (HPC-friendly, rootless)."),
        ("modal", "Modal", "Run commands in a Modal cloud sandbox."),
        ("daytona", "Daytona", "Run commands in a Daytona cloud sandbox."),
        ("ssh", "SSH", "Run commands on a remote host over SSH."))]


def _plugin_terminal_backend_rows() -> List[Dict[str, str]]:
    """Picker rows for plugin-registered terminal backends (fail-soft)."""
    rows: List[Dict[str, str]] = []
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent — plugin state may not be loaded yet
    except Exception:
        pass
    try:
        from agent.terminal_env_registry import list_providers
        for provider in list_providers():
            try:
                rows.append({"name": provider.name.strip().lower(), "label": provider.display_name,
                             "description": provider.description})
            except Exception:
                continue
    except Exception:
        return rows
    return rows


# Token / cost analytics helpers.
_AUX_COUNTERS = ("input_tokens", "output_tokens", "estimated_cost", "api_calls")


def _token_volume(row: Dict[str, Any]) -> Any:
    return (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)


def _aux_usage_rows(db, cutoff: float) -> List[Dict[str, Any]]:
    """Per-(model, task) auxiliary usage within the window: the task-dimension rows
    (task != '') record_auxiliary_usage writes into session_model_usage. [] when the
    table predates the task column (older DB opened read-only by newer code).

    See #23270.
    """
    try:
        cur = db._conn.execute("""
            SELECT u.model,
                   u.task,
                   u.billing_provider,
                   SUM(u.input_tokens) as input_tokens,
                   SUM(u.output_tokens) as output_tokens,
                   SUM(u.cache_read_tokens) as cache_read_tokens,
                   SUM(u.reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(u.estimated_cost_usd), 0) as estimated_cost,
                   COUNT(DISTINCT u.session_id) as sessions,
                   SUM(COALESCE(u.api_call_count, 0)) as api_calls,
                   MAX(u.last_seen) as last_used_at
            FROM session_model_usage u
            JOIN sessions s ON s.id = u.session_id
            WHERE s.started_at > ? AND u.task != ''
            GROUP BY u.model, u.task, u.billing_provider
            ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC
        """, (cutoff,))
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


def _merge_aux_into_by_model(
    by_model: List[Dict[str, Any]], aux_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold aux usage rows into the sessions-derived per-model list. Aux usage lives only in
    session_model_usage (never in the sessions counters), so this cannot double-count;
    models that ONLY appear via aux calls (e.g. a vision model) get their own entry."""
    if not aux_rows:
        return by_model
    merged: Dict[str, Dict[str, Any]] = {row.get("model") or "unknown": row for row in by_model}
    for aux in aux_rows:
        model = aux.get("model") or "unknown"
        target = merged.setdefault(model, {"model": model, "input_tokens": 0, "output_tokens": 0,
                                           "estimated_cost": 0, "sessions": 0, "api_calls": 0})
        for key in _AUX_COUNTERS:
            target[key] = (target.get(key) or 0) + (aux.get(key) or 0)
        target.setdefault("aux_tasks", []).append(
            {"task": aux.get("task") or "", **{key: aux.get(key) or 0 for key in _AUX_COUNTERS}})
    return sorted(merged.values(), key=_token_volume, reverse=True)


def _aux_task_summary(aux_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate aux usage rows across models into a per-task summary."""
    by_task: Dict[str, Dict[str, Any]] = {}
    for aux in aux_rows:
        task = aux.get("task") or ""
        d = by_task.setdefault(task, {"task": task, "input_tokens": 0, "output_tokens": 0,
                                      "estimated_cost": 0, "api_calls": 0, "models": []})
        for key in _AUX_COUNTERS:
            d[key] += aux.get(key) or 0
        model = aux.get("model") or "unknown"
        if model not in d["models"]:
            d["models"].append(model)
    return sorted(by_task.values(), key=_token_volume, reverse=True)


def _profile_cli_args(profile: Optional[str]) -> List[str]:
    """``["-p", <name>]`` for a validated non-default profile, else ``[]``. Hub actions run in
    a fresh ``hermes`` subprocess whose ``_apply_profile_override()`` reads ``-p`` from argv —
    the only mechanism that reaches import-time-bound globals like ``skills_hub.SKILLS_DIR``."""
    requested = (profile or "").strip()
    if not requested or requested.lower() in {"current", "default"}:
        return []
    from hermes_cli import profiles as profiles_mod
    _resolve_profile_dir(requested)
    return ["-p", profiles_mod.normalize_profile_name(requested)]


def _hub_action_name(verb: str, key: str) -> str:
    """Unique per-skill hub action name (+ registered log file): ``_spawn_hermes_action``
    tracks one process/log per name, so a shared "skills-install" would make concurrent
    row-level actions overwrite each other. Slug (readable) + hash (collision-proof)."""
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")[:48] or "skill"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    name = f"skills-{verb}-{slug}-{digest}"
    _ACTION_LOG_FILES.setdefault(name, f"action-{name}.log")
    return name


def _installed_hub_identifiers(profile: Optional[str] = None) -> dict:
    """identifier -> installed lock entry for hub-installed skills (UI marks installed search
    results). Scoped to ``profile``'s skills/.hub/lock.json when given — HubLockFile takes an
    explicit path, sidestepping the import-time LOCK_FILE binding. {} when unreadable."""
    try:
        from tools.skills_hub import HubLockFile
        if _is_current_profile(profile):
            lock = HubLockFile()
        else:
            profile_dir = _resolve_profile_dir(profile.strip())
            lock = HubLockFile(profile_dir / "skills" / ".hub" / "lock.json")
        keys = ("name", "trust_level", "scan_verdict")
        return {entry["identifier"]: {k: entry.get(k) for k in keys}
                for entry in lock.list_installed() if entry.get("identifier")}
    except Exception:
        return {}
