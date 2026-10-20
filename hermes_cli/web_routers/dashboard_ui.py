"""Dashboard theme/font and dashboard-plugin (discovery, hub, install/enable, asset serving) routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are resolved late at call time (cycle-safe).
"""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from hermes_cli.web_deps import LateState, late
from hermes_cli.config import cfg_get
from hermes_cli.web_routers._common import config_scoped_to_thread
from hermes_cli.web_server_dashboard import (
    _BUILTIN_DASHBOARD_THEMES, _discover_user_themes, _invalidate_plugins_hub_cache, _merged_plugins_hub,
    _plugin_consent_withheld,
)
from hermes_cli.web_server_memory import _normalize_memory_provider_name, _require_memory_provider_ready
from hermes_cli.web_models import (
    FontSetBody, ThemeSetBody, _AgentPluginInstallBody, _PluginProvidersPutBody, _PluginVisibilityBody,
)

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_get_dashboard_plugins = late("_get_dashboard_plugins")
_require_token = late("_require_token")
load_config = late("load_config", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")
# Home + secret scope for one request. NOT ``_profile_scope``: the bodies below clone/pull over the
# network and hold the scope for their whole duration, and ``_profile_scope`` also holds the
# process-global skills lock.
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_CONFIG_MUTATION_LOCK = LateState("_CONFIG_MUTATION_LOCK")


def _set_dashboard_key(key: str, value, profile: Optional[str] = None) -> None:
    """Write ``dashboard.<key>`` to the profile's config.yaml under the config mutation lock."""
    with _config_profile_scope(profile), _CONFIG_MUTATION_LOCK:
        config = load_config()
        if "dashboard" not in config:
            config["dashboard"] = {}
        config["dashboard"][key] = value
        save_config(config)


@router.get("/api/dashboard/themes")
async def get_dashboard_themes(profile: Optional[str] = None):
    """Available themes + the active one. Built-ins ship name/label/description only
    (the frontend owns their definitions in `web/src/themes/presets.ts`); user themes
    from `~/.hermes/dashboard-themes/*.yaml` ship their normalised `definition`."""
    def _run():
        config = load_config()
        active = cfg_get(config, "dashboard", "theme", default="default")
        themes = list(_BUILTIN_DASHBOARD_THEMES)
        seen = {t["name"] for t in themes}
        for t in _discover_user_themes():
            if t["name"] in seen:
                continue
            themes.append({"name": t["name"], "label": t["label"], "description": t["description"], "definition": t})
            seen.add(t["name"])
        return {"themes": themes, "active": active}

    return await config_scoped_to_thread(profile, _run)


@router.put("/api/dashboard/theme")
async def set_dashboard_theme(body: ThemeSetBody, profile: Optional[str] = None):
    """Set the active dashboard theme (persists to config.yaml)."""
    await asyncio.to_thread(_set_dashboard_key, "theme", body.name, profile)
    return {"ok": True, "theme": body.name}


# Curated font-override ids, kept in sync with FONT_CHOICES in web/src/themes/fonts.ts. The
# frontend owns the stacks + webfont URLs; the backend only needs the id allow-list to reject
# anything unvetted (the webfont URL is injected as a <link>, so never accept arbitrary ids/URLs).
_FONT_DEFAULT_ID = "theme"
_FONT_CHOICES = frozenset({
    "system-sans", "system-serif", "system-mono",
    "inter", "ibm-plex-sans", "work-sans", "atkinson-hyperlegible", "dm-sans",
    "spectral", "fraunces", "source-serif",
    "jetbrains-mono", "ibm-plex-mono", "space-mono",
})


@router.get("/api/dashboard/font")
async def get_dashboard_font(profile: Optional[str] = None):
    """Return the active font override (``"theme"`` = use the theme's font)."""
    def _run():
        font = cfg_get(load_config(), "dashboard", "font", default=_FONT_DEFAULT_ID)
        return {"font": font if font in _FONT_CHOICES else _FONT_DEFAULT_ID}

    return await config_scoped_to_thread(profile, _run)


@router.put("/api/dashboard/font")
async def set_dashboard_font(body: FontSetBody, profile: Optional[str] = None):
    """Set the font override (config.yaml). Unknown ids coerce to ``"theme"`` rather than
    400 so a stale client can't wedge the picker."""
    font = body.font if body.font in _FONT_CHOICES else _FONT_DEFAULT_ID
    await asyncio.to_thread(_set_dashboard_key, "font", font, profile)
    return {"ok": True, "font": font}


@router.get("/api/dashboard/plugins")
async def get_dashboard_plugins(profile: Optional[str] = None):
    """Return discovered dashboard plugins (excludes user-hidden and non-enabled ones)."""
    def _run():
        hidden: list = cfg_get(load_config(), "dashboard", "hidden_plugins", default=[]) or []
        # The consent the asset route and the API gates apply, so no listed tab fails to load.
        return [p for p in _get_dashboard_plugins()
                if p.get("name", "") not in hidden and not _plugin_consent_withheld(p)]

    # Strip internal fields before sending to frontend.
    return [{k: v for k, v in p.items() if not k.startswith("_")}
            for p in await config_scoped_to_thread(profile, _run)]


@router.get("/api/dashboard/plugins/rescan")
async def rescan_dashboard_plugins(profile: Optional[str] = None):
    """Force re-scan of the selected profile's dashboard plugins (its own cache, PLG-3)."""
    plugins = await config_scoped_to_thread(profile, lambda: _get_dashboard_plugins(force_rescan=True))
    return {"ok": True, "count": len(plugins)}


@router.get("/api/dashboard/plugins/hub")
async def get_plugins_hub(request: Request, profile: Optional[str] = None):
    """Unified agent plugins + dashboard extension metadata (session protected)."""
    _require_token(request)
    try:
        return await config_scoped_to_thread(profile, _merged_plugins_hub)
    except HTTPException:
        raise  # an unknown ?profile= is the scope's 404, not a hub failure
    except Exception as exc:
        _log.warning("plugins/hub failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build plugins hub.") from exc


def _plugin_action(result: dict, fallback_error: str, *, rescan: bool,
                   client_decides: tuple[str, ...] = ("consent_required",)) -> dict:
    """Common tail of agent-plugin mutations: 400 on ``ok=False``, then invalidate caches
    (rescanning discovery when files changed on disk). An answer flagged by one of *client_decides*
    is not a failure: nothing changed, the client shows it (the capability delta, the scan findings)
    and retries with the user's decision."""
    if any(result.get(flag) for flag in client_decides):
        return result
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or fallback_error)
    if rescan:
        _get_dashboard_plugins(force_rescan=True)
    _invalidate_plugins_hub_cache()
    return result


@router.get("/api/dashboard/plugins/catalog")
async def get_plugins_catalog(request: Request):
    """Curated plugin catalog merged with installed state (session protected)."""
    _require_token(request)

    def _run():
        from hermes_cli.plugins_cmd import _discover_all_plugins, _get_disabled_set, _get_enabled_set
        from hermes_cli.plugins_cmd_catalog import installed_catalog_state
        from hermes_cli.web_server_dashboard import _plugin_runtime_status
        enabled, disabled = _get_enabled_set(), _get_disabled_set()
        installed = {}
        for name, _v, _d, _s, dir_str, key in _discover_all_plugins():
            aliases = {name, key} - {""}
            info = {"dir": dir_str, "runtime_status": _plugin_runtime_status(aliases, enabled, disabled)}
            installed.update({alias: info for alias in aliases})
        return installed_catalog_state(installed)

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        _log.warning("plugins/catalog failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build plugins catalog.") from exc


async def _plugin_mutation(fn: Callable[[], dict]) -> dict:
    """Run a dashboard plugin mutation inside the dashboard profile's home + secret scope, off the
    event loop.

    Scope: the git-credential path reaches ``get_secret("GITHUB_TOKEN")``, which fails closed once
    the process serves more than its launch profile — an unhandled ``UnscopedSecretError`` was the
    bare 500 on install/update (#115256). Off-loop: the bodies clone/pull and rescan the plugins
    directory. ``_config_profile_scope`` not ``_profile_scope``: the latter holds the process-wide
    skills lock for the whole (network-bound) body.
    """
    def _run():
        with _config_profile_scope(None):
            return fn()

    return await asyncio.to_thread(_run)


@router.post("/api/dashboard/agent-plugins/install")
async def post_agent_plugin_install(request: Request, body: _AgentPluginInstallBody):
    _require_token(request)
    from hermes_cli.plugins_cmd import dashboard_install_plugin

    catalog_name = (body.catalog_name or "").strip()
    identifier = body.identifier.strip()
    if not identifier and not catalog_name:
        raise HTTPException(status_code=400, detail="Provide an identifier or a catalog_name.")
    result = await _plugin_mutation(lambda: _plugin_action(
        dashboard_install_plugin(
            identifier, force=body.force, enable=body.enable, catalog_name=catalog_name or None,
            ref=body.ref),
        "Install failed.", rescan=True))
    # Strip internal paths from the response
    result.pop("after_install_path", None)
    return result


def _validate_plugin_name(name: str) -> str:
    """Reject path-traversal attempts in plugin name URL parameters."""
    name = name.strip("/")
    if not name or ".." in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid plugin name.")
    return name


async def _named_plugin_action(request: Request, name: str, action: Callable[[str], dict], fallback_error: str, *,
                               rescan: bool, client_decides: tuple[str, ...] = ("consent_required",)) -> dict:
    _require_token(request)
    return await _plugin_mutation(lambda: _plugin_action(
        action(_validate_plugin_name(name)), fallback_error, rescan=rescan, client_decides=client_decides))


@router.post("/api/dashboard/agent-plugins/{name:path}/enable")
async def post_agent_plugin_enable(request: Request, name: str):
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled
    return await _named_plugin_action(request, name, lambda n: dashboard_set_agent_plugin_enabled(n, enabled=True),
                                      "Enable failed.", rescan=False)


@router.post("/api/dashboard/agent-plugins/{name:path}/disable")
async def post_agent_plugin_disable(request: Request, name: str):
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled
    return await _named_plugin_action(request, name, lambda n: dashboard_set_agent_plugin_enabled(n, enabled=False),
                                      "Disable failed.", rescan=False)


@router.post("/api/dashboard/agent-plugins/{name:path}/update")
async def post_agent_plugin_update(request: Request, name: str):
    from hermes_cli.plugins_cmd import dashboard_update_user_plugin
    # Body is optional: ``{"accept_capabilities": true}`` applies a re-pin the user confirmed after a
    # ``consent_required`` answer; ``{"force": true}`` applies an update whose scan returned caution
    # after the user saw the ``scan_findings``.
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    accept, force = body.get("accept_capabilities") is True, body.get("force") is True
    return await _named_plugin_action(
        request, name, lambda n: dashboard_update_user_plugin(n, accept_capabilities=accept, force=force),
        "Update failed.", rescan=True, client_decides=("consent_required", "scan_blocked"))


@router.delete("/api/dashboard/agent-plugins/{name:path}")
async def delete_agent_plugin(request: Request, name: str):
    from hermes_cli.plugins_cmd import dashboard_remove_user_plugin
    return await _named_plugin_action(request, name, dashboard_remove_user_plugin, "Remove failed.", rescan=True)


@router.post("/api/dashboard/agent-plugins/activate")
async def post_agent_plugin_activate(request: Request):
    """``hermes plugins install`` / ``enable`` in another process asks this backend to load the plugin
    for ``home`` and hand its MCP servers and skills to that profile's open chats
    (``hermes_cli.plugins_activation.load_and_go_live``). ``home`` must be a profile this host serves."""
    _require_token(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = _validate_plugin_name(str((body or {}).get("name") or ""))
    from hermes_constants import get_hermes_home, hermes_home_key, profile_name_for_home
    home = Path(str((body or {}).get("home") or "")).expanduser()
    profile = profile_name_for_home(home) if str(home) not in ("", ".") else None
    if profile is None:
        raise HTTPException(status_code=400, detail="Not a Hermes profile home.")
    from hermes_cli.plugins_activation import load_and_go_live

    def _run():
        with _config_profile_scope(None if profile == "default" else profile):
            if hermes_home_key(get_hermes_home()) != hermes_home_key(home):
                raise HTTPException(status_code=400, detail="Home is not served by this backend.")
            return {"ok": True, "activation": load_and_go_live(name)}

    return await asyncio.to_thread(_run)


@router.put("/api/dashboard/plugin-providers")
async def put_plugin_providers(request: Request, body: _PluginProvidersPutBody,
                               profile: Optional[str] = None):
    """Persist memory provider / context engine selection (writes config.yaml)."""
    _require_token(request)
    from hermes_cli.plugins_cmd import _save_context_engine, _save_memory_provider

    def _run():
        # ``_save_memory_provider``/``_save_context_engine`` are functools.partial over
        # ``_write_config_value`` -> save_config: the write is one hop away and lands in
        # whatever home the scope names. Unlike plugin INSTALLATION (host venv, pinned to the
        # serving profile), these two keys are per-profile settings — `memory.provider` is the
        # same key PUT /api/memory/provider scopes — so they follow ``?profile=``.
        with _config_profile_scope(profile), _CONFIG_MUTATION_LOCK:
            if body.memory_provider is not None:
                memory_provider = _normalize_memory_provider_name(body.memory_provider)
                # Readiness resolves through load_config(); inside the scope so the answer is
                # about the profile being written, not the launch profile.
                _require_memory_provider_ready(memory_provider)
                _save_memory_provider(memory_provider)
            if body.context_engine is not None:
                _save_context_engine(body.context_engine)
        _invalidate_plugins_hub_cache()
        return {"ok": True}

    return await asyncio.to_thread(_run)


@router.post("/api/dashboard/plugins/{name:path}/visibility")
async def post_plugin_visibility(request: Request, name: str, body: _PluginVisibilityBody,
                                 profile: Optional[str] = None):
    """Toggle a plugin's sidebar visibility (persists to config.yaml dashboard.hidden_plugins)."""
    _require_token(request)
    name = _validate_plugin_name(name)

    def _run():
        with _config_profile_scope(profile), _CONFIG_MUTATION_LOCK:
            config = load_config()
            if "dashboard" not in config or not isinstance(config.get("dashboard"), dict):
                config["dashboard"] = {}
            hidden_list: list = config["dashboard"].get("hidden_plugins") or []
            if not isinstance(hidden_list, list):
                hidden_list = []
            if body.hidden and name not in hidden_list:
                hidden_list.append(name)
            elif not body.hidden and name in hidden_list:
                hidden_list.remove(name)
            config["dashboard"]["hidden_plugins"] = hidden_list
            save_config(config)
        _invalidate_plugins_hub_cache()
        return {"ok": True, "name": name, "hidden": body.hidden}

    return await asyncio.to_thread(_run)


# Browser-asset suffix allowlist. Everything else is 404'd so we never leak ``.py`` backend
# sources, READMEs, ``.env.example`` etc. Extend deliberately; do NOT add a fallback.
_PLUGIN_ASSET_CONTENT_TYPES = {
    ".js": "application/javascript", ".mjs": "application/javascript", ".css": "text/css",
    ".json": "application/json", ".html": "text/html", ".svg": "image/svg+xml",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".ico": "image/x-icon", ".woff2": "font/woff2", ".woff": "font/woff",
    ".ttf": "font/ttf", ".otf": "font/otf", ".map": "application/json",
}


@router.get("/dashboard-plugins/@{profile}/{plugin_name}/{file_path:path}")  # before the unscoped route
async def serve_profile_plugin_asset(profile: str, plugin_name: str, file_path: str):
    """``serve_plugin_asset`` with the profile as a path segment (PLG-3): URLs relative to a stylesheet
    or bundle (``url("./icon.svg")``, chunk imports) drop a query string but keep this prefix."""
    return await serve_plugin_asset(plugin_name, file_path, profile)


@router.get("/dashboard-plugins/{plugin_name}/{file_path:path}")
async def serve_plugin_asset(plugin_name: str, file_path: str, profile: Optional[str] = None):
    """Serve static assets from a dashboard plugin's ``dashboard/`` directory.

    Unauthenticated on purpose: the SPA loads plugin JS via ``<script src>`` and CSS
    via ``<link href>``, which cannot attach an auth header. Hence the suffix
    allowlist — user plugins ship a ``plugin_api.py`` backend the browser never
    fetches, and without it anyone on the loopback port could curl a private
    plugin's source. Path traversal is blocked via ``resolve().is_relative_to()``;
    user plugins must be enabled (bundled ones not disabled) (GHSA-mcfc-hp25-cjv7) in the profile
    whose copy these files are: a native ticket's, the ``?profile=`` the browser's plugin list used
    (PLG-3), else the launch profile. An unknown profile answers as an unknown plugin.

    See #46435.
    """
    def _consented() -> Optional[dict]:
        plugin = next((p for p in _get_dashboard_plugins() if p["name"] == plugin_name), None)
        return None if not plugin or _plugin_consent_withheld(plugin) else plugin

    try:
        plugin = await config_scoped_to_thread(profile, _consented)
    except HTTPException:
        plugin = None
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")

    base = Path(plugin["_dir"])
    target = (base / file_path).resolve()

    if not target.is_relative_to(base.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = _PLUGIN_ASSET_CONTENT_TYPES.get(target.suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})
