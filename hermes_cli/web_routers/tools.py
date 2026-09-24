"""Toolset / terminal-backend dashboard routes.

The toolset/terminal catalogs and helpers stay in web_server (some are defined
*after* this router's mount point) — reached via the late-binding seam so
monkeypatching on web_server stays authoritative.
"""

import asyncio
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import late
from hermes_cli.web_server_profiles import _plugin_terminal_backend_rows
from starlette.concurrency import run_in_threadpool
from hermes_cli.web_models import (
    TerminalBackendSelect, ToolsetEnvUpdate, ToolsetModelSelect, ToolsetPostSetup,
    ToolsetProviderSelect, ToolsetToggle)
from hermes_cli.web_routers._common import (
    _CONFIG_MUTATION_LOCK, _profile_cli_args, _profile_scope, _spawn_hermes_action,
    config_write_scope, log as _log, scoped_to_thread, spawn_profile_action)

router = APIRouter()

load_config = late("load_config", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")


def _env_value(name: str) -> str:
    """``get_env_value`` that never raises (empty string on any failure)."""
    try:
        from hermes_cli.config import get_env_value

        return get_env_value(name) or ""
    except Exception:
        return ""


def _terminal_cfg_value(terminal_cfg: dict, key: str, env_var: str) -> str:
    """Read a terminal.* setting from config.yaml, falling back to its env var."""
    value = terminal_cfg.get(key)
    if value is not None and str(value).strip():
        return str(value).strip()
    return _env_value(env_var).strip()


def _terminal_backend_rows() -> List[Dict[str, str]]:
    """Built-in picker rows plus plugin-registered backends, computed per request
    so a plugin installed after server start still shows up."""
    from hermes_cli.web_server_profiles import _TERMINAL_BACKENDS
    return [*_TERMINAL_BACKENDS, *_plugin_terminal_backend_rows()]


def _probe_docker_backend(_cfg) -> tuple:
    if not shutil.which("docker"):
        return ("needs_setup", "Docker CLI not found — install Docker Desktop or docker-ce.")
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2)
        if proc.returncode == 0:
            return ("ready", "")
        return ("needs_setup", "Docker daemon not reachable — start Docker and retry.")
    except subprocess.TimeoutExpired:
        return ("needs_setup", "Docker daemon not responding (timed out).")
    except Exception as exc:
        return ("unavailable", f"Docker probe failed: {exc}")


def _probe_singularity_backend(_cfg) -> tuple:
    if shutil.which("singularity") or shutil.which("apptainer"):
        return ("ready", "")
    return ("needs_setup", "Neither singularity nor apptainer found on PATH.")


def _probe_ssh_backend(terminal_cfg: dict) -> tuple:
    host = _terminal_cfg_value(terminal_cfg, "ssh_host", "TERMINAL_SSH_HOST")
    user = _terminal_cfg_value(terminal_cfg, "ssh_user", "TERMINAL_SSH_USER")
    missing = [k for k, v in (("terminal.ssh_host", host), ("terminal.ssh_user", user)) if not v]
    if missing:
        return (
            "needs_setup",
            f"Set {' and '.join(missing)} in config.yaml (or the matching TERMINAL_SSH_* env vars).",
        )
    return ("ready", f"{user}@{host}")


def _probe_modal_backend(_cfg) -> tuple:
    try:
        from tools.tool_backend_helpers import has_direct_modal_credentials

        if has_direct_modal_credentials():
            return ("ready", "")
    except Exception:
        pass
    if _env_value("MODAL_TOKEN_ID") and _env_value("MODAL_TOKEN_SECRET"):
        return ("ready", "")
    return (
        "needs_setup",
        "Modal credentials not found — set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (or run `modal setup`).",
    )


def _probe_daytona_backend(_cfg) -> tuple:
    if _env_value("DAYTONA_API_KEY"):
        return ("ready", "")
    return ("needs_setup", "Set DAYTONA_API_KEY to use the Daytona backend.")


_BACKEND_PROBES = {
    "local": lambda _cfg: ("ready", ""),
    "docker": _probe_docker_backend,
    "singularity": _probe_singularity_backend,
    "ssh": _probe_ssh_backend,
    "modal": _probe_modal_backend,
    "daytona": _probe_daytona_backend,
}


def _probe_terminal_backend(name: str, terminal_cfg: dict) -> tuple:
    """Return ``(status, detail)`` for one backend. Never raises."""
    try:
        probe = _BACKEND_PROBES.get(name)
        if probe is not None:
            return probe(terminal_cfg)
        try:
            from agent.terminal_env_registry import get_provider

            provider = get_provider(name)
            if provider is not None:
                return provider.probe()
        except Exception:
            pass
        return ("unavailable", f"Unknown backend: {name}")
    except Exception as exc:  # pragma: no cover — belt-and-braces guard
        return ("unavailable", f"Probe failed: {exc}")


# Toolsets whose backends carry a selectable model catalog, mapped to the
# config.yaml section their `model` key lives in. Mirrors the CLI's
# post-selection model pickers in tools_config.py.
_MODEL_CATALOG_TOOLSETS = {"image_gen": "image_gen", "video_gen": "video_gen"}


def _resolve_toolset_model_plugin(ts_key: str, provider_row: dict) -> Optional[str]:
    """Map a provider picker row to its model-catalog plugin name.

    Plugin-backed rows carry ``image_gen_plugin_name`` / ``video_gen_plugin_name``;
    the managed "Nous Subscription" image row instead carries the legacy
    ``imagegen_backend: "fal"`` marker (same underlying FAL catalog).
    """
    if ts_key == "image_gen":
        return provider_row.get("image_gen_plugin_name") or (
            "fal" if provider_row.get("imagegen_backend") else None)
    if ts_key == "video_gen":
        return provider_row.get("video_gen_plugin_name")
    return None


def _toolset_model_catalog(ts_key: str, plugin_name: str):
    """Return ``(catalog_dict, default_model)`` for a toolset's plugin backend."""
    from hermes_cli.tools_config import _plugin_image_gen_catalog, _plugin_video_gen_catalog

    if ts_key == "image_gen":
        return _plugin_image_gen_catalog(plugin_name)
    return _plugin_video_gen_catalog(plugin_name)


def _category_providers(ts_key: str, config: dict) -> list:
    """Visible provider rows for a toolset's category (fresh entitlement read)."""
    from hermes_cli.tools_config import TOOL_CATEGORIES, _visible_providers

    cat = TOOL_CATEGORIES.get(ts_key)
    return _visible_providers(cat, config, force_fresh=True) if cat else []


def _find_toolset_provider_row(
    ts_key: str, config: dict, provider: Optional[str]) -> Optional[dict]:
    """Resolve a provider picker row by name, or the active row when omitted."""
    from hermes_cli.tools_config import _is_provider_active

    rows = _category_providers(ts_key, config)
    if provider:
        return next((p for p in rows if p.get("name") == provider), None)
    return next((p for p in rows if _is_provider_active(p, config, force_fresh=True)), None)


def _require_known_toolset(name: str) -> None:
    """400 for toolset keys outside the effective configurable set."""
    from hermes_cli.tools_config import _get_effective_configurable_toolsets

    if name not in {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}:
        raise _bad_request(f"Unknown toolset: {name}")


def _dict_section(config: dict, key: str) -> dict:
    """``config[key]`` as a dict, replacing a non-dict value in place."""
    section = config.setdefault(key, {})
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _no_models(name: str) -> dict:
    return {"name": name, "has_models": False, "models": [], "current": None, "default": None}


@router.get("/api/tools/toolsets")
async def get_toolsets(profile: Optional[str] = None):
    from hermes_cli.tools_config import (
        _CONFIG_ONLY_TOOLSETS, _get_effective_configurable_toolsets, _get_platform_tools,
        _toolset_configuration_platform, _toolset_has_keys, get_nous_subscription_features,
        gui_toolset_label)
    from hermes_cli.platforms import platform_label
    from toolsets import resolve_toolset
    from utils import is_truthy_value

    def _read():
        with _profile_scope(profile):
            config = load_config()
            toolset_rows = _get_effective_configurable_toolsets()
            target_platforms = {
                _toolset_configuration_platform(name) for name, _, _ in toolset_rows}
            enabled_by_platform = {
                platform: _get_platform_tools(config, platform, include_default_mcp_servers=False)
                for platform in target_platforms}
            features = get_nous_subscription_features(config)
        return config, toolset_rows, enabled_by_platform, features

    config, toolset_rows, enabled_by_platform, features = await run_in_threadpool(_read)
    result = []
    for name, label, desc in toolset_rows:
        try:
            tools = sorted(set(resolve_toolset(name)))
        except Exception:
            tools = []
        target_platform = _toolset_configuration_platform(name)
        if name in _CONFIG_ONLY_TOOLSETS:
            # Config-only capabilities (stt) have no per-platform toolset —
            # their switch is their own config section (e.g. stt.enabled).
            section = config.get(name)
            section = section if isinstance(section, dict) else {}
            is_enabled = is_truthy_value(section.get("enabled", True), default=True)
        else:
            is_enabled = name in enabled_by_platform[target_platform]
        result.append({
            "name": name, "label": gui_toolset_label(label), "description": desc,
            "platform": target_platform,
            "platform_label": gui_toolset_label(platform_label(target_platform, target_platform)),
            "enabled": is_enabled, "available": is_enabled,
            "configured": _toolset_has_keys(name, config, features=features), "tools": tools})
    return result


@router.put("/api/tools/toolsets/{name}")
async def toggle_toolset(name: str, body: ToolsetToggle, profile: Optional[str] = None):
    """Enable/disable a configurable toolset for its configuration platform
    (``platform_toolsets.cli`` for most; platform-restricted toolsets target
    their own platform) via the same ``_save_platform_tools`` the CLI uses."""
    from hermes_cli.tools_config import (
        _CONFIG_ONLY_TOOLSETS, _get_platform_tools, _save_platform_tools,
        _toolset_configuration_platform)

    _require_known_toolset(name)
    target_platform = _toolset_configuration_platform(name)
    scope_profile = body.profile or profile

    def _run():
        with config_write_scope(scope_profile):
            config = load_config()
            if name in _CONFIG_ONLY_TOOLSETS:
                # Config-only capabilities (stt) toggle their own section's
                # ``enabled`` flag — there is no platform_toolsets entry.
                _dict_section(config, name)["enabled"] = bool(body.enabled)
                save_config(config)
                return
            enabled = set(
                _get_platform_tools(config, target_platform, include_default_mcp_servers=False))
            if body.enabled:
                enabled.add(name)
            else:
                enabled.discard(name)
            _save_platform_tools(config, target_platform, enabled)

    await asyncio.to_thread(_run)

    # Install-on-enable: a provider whose post_setup install predicate is
    # UNSATISFIED (cua-driver binary missing, etc.) gets the same background
    # install `hermes tools` runs — otherwise the toggle "saves" but the tool
    # never appears.  Best-effort: a spawn failure never fails the toggle.
    post_setup_started: Optional[str] = None
    if body.enabled and name not in _CONFIG_ONLY_TOOLSETS:
        def _pending_install_key() -> Optional[str]:
            from hermes_cli.tools_config import (
                TOOL_CATEGORIES, _post_setup_already_installed, _visible_providers)

            cat = TOOL_CATEGORIES.get(name)
            if not cat:
                return None
            with _profile_scope(scope_profile):
                config = load_config()
                keys = [prov.get("post_setup") for prov in _visible_providers(cat, config)]
                return next((k for k in keys if k and not _post_setup_already_installed(k)), None)

        try:
            pending_key = await asyncio.to_thread(_pending_install_key)
            if pending_key:
                _spawn_hermes_action(
                    _profile_cli_args(scope_profile) + ["tools", "post-setup", pending_key],
                    "tools-post-setup")
                post_setup_started = pending_key
        except Exception:
            _log.exception("install-on-enable post-setup spawn failed for %s", name)

    return {
        "ok": True, "name": name, "platform": target_platform, "enabled": body.enabled,
        "post_setup_started": post_setup_started}


@router.get("/api/tools/toolsets/{name}/config")
async def get_toolset_config(name: str, profile: Optional[str] = None):
    """Provider matrix + key status for a toolset's config panel (the CLI picker
    rows, each env var annotated ``is_set``); no category -> ``has_category: false``."""
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES, _is_provider_active, _visible_providers, provider_readiness_status,
        web_provider_capabilities)
    from hermes_cli.config import get_env_value
    from hermes_cli.nous_subscription import get_nous_subscription_features

    _require_known_toolset(name)

    def _read():
        with _profile_scope(profile):
            config = load_config()
            cat = TOOL_CATEGORIES.get(name)
            providers = []
            active_provider = None
            if cat:
                # Entitlement state fetched once for the whole matrix.
                features = get_nous_subscription_features(config, force_fresh=True)
                for prov in _visible_providers(cat, config, force_fresh=True):
                    env_vars = [
                        {
                            "key": e["key"], "prompt": e.get("prompt", e["key"]),
                            "url": e.get("url"), "default": e.get("default"),
                            "is_set": bool(get_env_value(e["key"]))}
                        for e in prov.get("env_vars", [])]
                    # Same active-provider determination as the CLI picker, so the
                    # GUI highlights the provider actually written to config.
                    is_active = _is_provider_active(prov, config, force_fresh=True)
                    if is_active and active_provider is None:
                        active_provider = prov["name"]
                    row = {
                        "name": prov["name"],
                        "badge": prov.get("badge", ""),
                        "tag": prov.get("tag", ""),
                        "env_vars": env_vars,
                        "post_setup": prov.get("post_setup"),
                        "requires_nous_auth": bool(prov.get("requires_nous_auth")),
                        "is_active": is_active,
                        # Server-side readiness: zero-env-var rows are NOT
                        # automatically ready (logged-out Nous rows, never-run
                        # post_setup installs).
                        "status": provider_readiness_status(
                            prov, config, features=features, is_active=is_active)}
                    if name == "web" and prov.get("web_backend"):
                        # web is two capabilities (search/extract); surface each
                        # row's backend key + capabilities for per-capability selection.
                        row["web_backend"] = prov["web_backend"]
                        row["capabilities"] = web_provider_capabilities(prov["web_backend"])
                    if name == "tts" and prov.get("tts_provider"):
                        # Key written to tts.provider; doubles as the config section
                        # (tts.<key>.*) holding the provider's voice/model settings.
                        row["tts_provider"] = prov["tts_provider"]
                    providers.append(row)
            payload = {
                "name": name, "has_category": cat is not None, "providers": providers,
                "active_provider": active_provider}
            if name == "web":
                # Resolve active backends exactly as the web_search/web_extract
                # dispatchers do, so badges reflect what a call would hit now.
                try:
                    from tools.web_tools import _get_extract_backend, _get_search_backend

                    search_backend = _get_search_backend()
                    extract_backend = _get_extract_backend()
                except Exception:
                    search_backend = extract_backend = None
                payload["active_search_backend"] = search_backend
                payload["active_extract_backend"] = extract_backend
        return payload

    return await asyncio.to_thread(_read)


@router.get("/api/tools/toolsets/{name}/models")
async def get_toolset_models(
    name: str, provider: Optional[str] = None, profile: Optional[str] = None):
    """Model catalog for a toolset backend (image/video gen) — the GUI
    counterpart of the CLI model picker.  ``provider`` names a picker row
    (default: the active provider); no catalog -> ``has_models: false``."""
    section = _MODEL_CATALOG_TOOLSETS.get(name)
    if section is None:
        return _no_models(name)

    def _read():
        with _profile_scope(profile):
            config = load_config()
            row = _find_toolset_provider_row(name, config, provider)
            plugin = _resolve_toolset_model_plugin(name, row) if row else None
            if not plugin:
                return None

            catalog, default_model = _toolset_model_catalog(name, plugin)
            section_cfg = config.get(section)
            current = None
            if isinstance(section_cfg, dict):
                raw = section_cfg.get("model")
                if isinstance(raw, str) and raw.strip():
                    current = raw.strip()
            if current not in catalog:
                current = default_model if default_model in catalog else None
        models = [
            {
                "id": model_id, "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""), "strengths": meta.get("strengths", ""),
                "price": meta.get("price", "")}
            for model_id, meta in catalog.items()]
        return {
            "name": name, "has_models": bool(models), "provider": row.get("name"),
            "plugin": plugin, "models": models, "current": current, "default": default_model}

    return await asyncio.to_thread(_read) or _no_models(name)


@router.put("/api/tools/toolsets/{name}/model")
async def select_toolset_model(
    name: str, body: ToolsetModelSelect, profile: Optional[str] = None):
    """Persist a backend model selection (``image_gen.model`` /
    ``video_gen.model``), validated against the resolved backend's catalog."""
    section = _MODEL_CATALOG_TOOLSETS.get(name)
    if section is None:
        raise _bad_request(f"Toolset has no model catalog: {name}")
    model_id = (body.model or "").strip()
    if not model_id:
        raise _bad_request("model is required")

    def _run():
        with config_write_scope(body.profile or profile):
            config = load_config()
            row = _find_toolset_provider_row(name, config, body.provider)
            plugin = _resolve_toolset_model_plugin(name, row) if row else None
            if not plugin:
                raise _bad_request(f"No model-capable backend is active for {name}")

            catalog, _default = _toolset_model_catalog(name, plugin)
            if model_id not in catalog:
                raise _bad_request(f"Unknown model {model_id!r} for backend {plugin!r}")

            _dict_section(config, section)["model"] = model_id
            save_config(config)
        return plugin

    plugin = await asyncio.to_thread(_run)
    return {"ok": True, "name": name, "model": model_id, "plugin": plugin}


@router.put("/api/tools/toolsets/{name}/provider")
async def select_toolset_provider(
    name: str, body: ToolsetProviderSelect, profile: Optional[str] = None):
    """Persist a provider selection via ``apply_provider_selection`` (shared with
    ``hermes tools``, so both write identical keys).

    ``web`` only: ``capability`` ('search' | 'extract') writes
    ``web.<capability>_backend`` (the override the dispatchers resolve first);
    omitted -> legacy ``web.backend``.  Managed Nous rows report Portal
    entitlement (``needs_nous_auth`` + ``feature``): the GUI has no inline
    login, so an unentitled selection would write config and never activate.
    """
    from hermes_cli.tools_config import apply_provider_selection, web_provider_capabilities
    from hermes_cli.nous_subscription import (
        MANAGED_FEATURE_COVERAGE_CATEGORY, get_nous_subscription_features)

    _require_known_toolset(name)

    if body.capability is not None:
        if name != "web":
            raise _bad_request("capability selection is only supported for the web toolset")
        if body.capability not in ("search", "extract"):
            raise _bad_request(
                f"Unknown capability: {body.capability!r} (expected 'search' or 'extract')")

    def _provider_row(config):
        return next(
            (p for p in _category_providers(name, config) if p.get("name") == body.provider), None)

    def _run():
        with _profile_scope(body.profile or profile):
            with _CONFIG_MUTATION_LOCK:
                config = load_config()
                if body.capability is not None:
                    # Per-capability path writes web.<capability>_backend only —
                    # web.backend is untouched so the other capability keeps
                    # resolving through the shared fallback chain.
                    prov = _provider_row(config)
                    if prov is None:
                        raise _bad_request(
                            f"Unknown provider {body.provider!r} for toolset {name!r}")
                    backend = prov.get("web_backend")
                    if not backend:
                        raise _bad_request(f"Provider {body.provider!r} has no web backend key")
                    if body.capability not in web_provider_capabilities(backend):
                        raise _bad_request(f"{body.provider} does not support {body.capability}")
                    _dict_section(config, "web")[f"{body.capability}_backend"] = backend
                else:
                    try:
                        apply_provider_selection(name, body.provider, config)
                    except KeyError as exc:
                        raise _bad_request(str(exc).strip('"'))
                save_config(config)
                response: Dict[str, Any] = {"ok": True, "name": name, "provider": body.provider}
                if body.capability is not None:
                    response["capability"] = body.capability

            # Entitlement check for managed Nous rows (mirrors the CLI's
            # ensure_nous_portal_access gate).  Hits the Portal, so it runs AFTER
            # releasing the mutation lock — still in the worker thread + scope.
            row = _provider_row(config)
            managed_feature = (row or {}).get("managed_nous_feature")
            if managed_feature:
                features = get_nous_subscription_features(config, force_fresh=True)
                acct = features.account_info
                category = MANAGED_FEATURE_COVERAGE_CATEGORY.get(managed_feature)
                entitled = bool(
                    acct
                    and acct.logged_in
                    and (
                        acct.tool_gateway_entitled_for(category)
                        if category
                        else acct.tool_gateway_entitled))
                if not entitled:
                    response["needs_nous_auth"] = True
                    response["feature"] = managed_feature
        return response

    return await asyncio.to_thread(_run)


@router.put("/api/tools/toolsets/{name}/env")
async def save_toolset_env(name: str, body: ToolsetEnvUpdate, profile: Optional[str] = None):
    """Persist API keys to ``.env`` via ``save_env_value``.  Keys are validated
    against the union of the category's visible-provider ``env_vars`` so this
    can't write arbitrary env vars; a blank value means "leave unchanged"."""
    from hermes_cli.config import get_env_value, save_env_value

    _require_known_toolset(name)

    def _run():
        with _profile_scope(body.profile or profile):
            config = load_config()
            allowed: set[str] = {
                e["key"]
                for prov in _category_providers(name, config)
                for e in prov.get("env_vars", [])}

            unknown = [k for k in body.env if k not in allowed]
            if unknown:
                raise _bad_request(
                    f"Unknown env var(s) for toolset {name}: {', '.join(sorted(unknown))}")

            saved: List[str] = []
            skipped: List[str] = []
            for key, value in body.env.items():
                if value and value.strip():
                    try:
                        save_env_value(key, value.strip())
                    except ValueError as exc:
                        raise _bad_request(str(exc))
                    saved.append(key)
                else:
                    skipped.append(key)

            status = {k: bool(get_env_value(k)) for k in allowed}
        return saved, skipped, status

    saved, skipped, status = await asyncio.to_thread(_run)
    return {"ok": True, "name": name, "saved": saved, "skipped": skipped, "is_set": status}


@router.post("/api/tools/toolsets/{name}/post-setup")
async def run_toolset_post_setup(
    name: str, body: ToolsetPostSetup, profile: Optional[str] = None):
    """Spawn ``hermes tools post-setup <key>`` (long-running installs) as a
    background action tailed via ``GET /api/actions/tools-post-setup/status``;
    ``profile`` is threaded so hooks see the drawer's HERMES_HOME."""
    from hermes_cli.tools_config import valid_post_setup_keys

    _require_known_toolset(name)
    if body.key not in valid_post_setup_keys():
        raise _bad_request(f"Unknown post-setup key: {body.key}")

    result = spawn_profile_action(
        body.profile or profile, ["tools", "post-setup", body.key], "tools-post-setup",
        log_msg="Failed to spawn tools post-setup", prefix="Failed to run post-setup")
    result["key"] = body.key
    return result


@router.get("/api/tools/terminal/backends")
async def get_terminal_backends(profile: Optional[str] = None):
    """Terminal backend rows with health probes: ``status`` is ``ready`` /
    ``needs_setup`` / ``unavailable``; a probe failure is a status, never an
    error response."""
    def _read():
        with _profile_scope(profile):
            config = load_config()
            terminal_cfg = config.get("terminal")
            if not isinstance(terminal_cfg, dict):
                terminal_cfg = {}
            rows = _terminal_backend_rows()
            active = str(terminal_cfg.get("backend") or "local").strip().lower()
            if active not in {row["name"] for row in rows}:
                active = "local"

            backends = []
            for row in rows:
                status, detail = _probe_terminal_backend(row["name"], terminal_cfg)
                backends.append({
                    "name": row["name"], "label": row["label"], "description": row["description"],
                    "active": row["name"] == active, "status": status, "detail": detail})
        return {"active": active, "backends": backends}

    return await asyncio.to_thread(_read)


@router.put("/api/tools/terminal/backend")
async def select_terminal_backend(
    body: TerminalBackendSelect, profile: Optional[str] = None):
    """Persist ``terminal.backend``.  A backend that still needs setup is
    allowed — the picker shows guidance instead of blocking, like the CLI."""
    backend = (body.backend or "").strip().lower()
    valid_names = {row["name"] for row in _terminal_backend_rows()}
    if backend not in valid_names:
        raise _bad_request(
            f"Unknown terminal backend: {body.backend!r}. "
            f"Use one of: {', '.join(sorted(valid_names))}")

    def _run():
        with config_write_scope(body.profile or profile):
            config = load_config()
            _dict_section(config, "terminal")["backend"] = backend
            save_config(config)

    await asyncio.to_thread(_run)
    return {"ok": True, "backend": backend}


@router.get("/api/tools/computer-use/status")
async def get_computer_use_status(profile: Optional[str] = None):
    """Computer Use readiness for the desktop card (payload shape: see
    ``tools.computer_use.permissions.computer_use_status``)."""
    from tools.computer_use.permissions import computer_use_status

    return await scoped_to_thread(profile, computer_use_status)


@router.post("/api/tools/computer-use/permissions/grant")
async def grant_computer_use_permissions(profile: Optional[str] = None):
    """Spawn ``hermes computer-use permissions grant`` (macOS-only: launches
    CuaDriver via LaunchServices so the TCC dialog is attributed correctly).
    The frontend polls ``GET /api/actions/computer-use-grant/status``."""
    if sys.platform != "darwin":
        raise _bad_request("Computer Use permission grants are a macOS concept.")
    return spawn_profile_action(
        profile, ["computer-use", "permissions", "grant"], "computer-use-grant",
        log_msg="Failed to spawn computer-use permissions grant",
        prefix="Failed to request permissions")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'LateState': ('hermes_cli.web_deps', 'LateState'),
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
