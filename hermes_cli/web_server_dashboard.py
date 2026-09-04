"""Dashboard UI assets: SPA mount, theme normalisation/bootstrap CSS, dashboard-plugin discovery and the plugins-hub merge.
"""

import logging
import importlib.util
import json
import os
import sys
import threading
import time
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Any, Dict, List, Optional
from hermes_cli.config import cfg_get, get_process_hermes_home
from utils import env_var_enabled

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


def _normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header value (single source of truth lives in
    ``hermes_cli.dashboard_auth.prefix`` so gate, OAuth, cookies and SPA mount agree)."""
    from hermes_cli.dashboard_auth.prefix import normalise_prefix
    return normalise_prefix(raw)


def _layer_hex(palette: Dict[str, Any], key: str, default: str) -> str:
    layer = palette.get(key) or {}
    return layer.get("hex", default) if isinstance(layer, dict) else default


def _render_active_theme_bootstrap_css() -> str:
    """Critical-CSS ``<style>`` shim for the active *user* theme, so the first paint uses the
    target palette instead of flashing the bundle's default Hermes Teal until
    ``ThemeProvider.applyTheme()`` runs. Built-in themes return "" (the bundle owns them).

    Variable names MUST match what the bundle consumes (``layerVars()`` /
    ``typographyVars()`` in ``web/src/themes/context.tsx``). The ``html,body`` rule
    references the variables rather than literals so runtime theme switches stay live:
    ``applyTheme()`` writes inline styles on ``documentElement`` which outrank this block.
    """
    from hermes_cli.config import load_config
    try:
        active = cfg_get(load_config(), "dashboard", "theme", default="default")
        if not active or not isinstance(active, str):
            return ""
        if any(b["name"] == active for b in _BUILTIN_DASHBOARD_THEMES):
            return ""
        for theme in _discover_user_themes():
            if theme.get("name") != active:
                continue
            palette = theme.get("palette") or {}
            typo = theme.get("typography") or {}
            font_sans = typo.get("fontSans") or _THEME_DEFAULT_TYPOGRAPHY["fontSans"]
            base_size = typo.get("baseSize") or _THEME_DEFAULT_TYPOGRAPHY["baseSize"]

            def _esc(s: str) -> str:  # defensive ``</style>`` escape
                return str(s).replace("</", "<\\/")
            return (
                '<style id="hermes-theme-bootstrap">'
                ":root{"
                f"--background-base:{_esc(_layer_hex(palette, 'background', '#0a0a0a'))};"
                f"--midground-base:{_esc(_layer_hex(palette, 'midground', '#e5e5e5'))};"
                f"--theme-font-sans:{_esc(font_sans)};"
                f"--theme-base-size:{_esc(base_size)};"
                "}"
                "html,body{background-color:var(--background-base);"
                "color:var(--midground-base);"
                "font-family:var(--theme-font-sans);"
                "font-size:var(--theme-base-size);}"
                "</style>"
            )
        return ""
    except Exception:
        _log.debug("theme bootstrap render failed", exc_info=True)
        return ""


# Hashed bundle assets are immutable by construction (content hash in the filename; index.html
# is served ``no-store`` and always references the current hashes).
_IMMUTABLE_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"
_NO_STORE = {"Cache-Control": "no-store, no-cache, must-revalidate"}
_HEADLESS_MSG = (
    "Headless backend (hermes serve): web UI disabled — use "
    "`hermes dashboard` for the browser UI."
)


def mount_spa(application: FastAPI):
    """Mount the built SPA; unmatched paths fall back to index.html for client-side routing.

    The session token is injected into index.html via a ``<script>`` tag so the SPA can
    authenticate without a separate token-dispensing endpoint. Behind a path-prefix reverse
    proxy (``X-Forwarded-Prefix: /hermes``) the served index.html is rewritten so absolute
    asset URLs and the runtime ``__HERMES_BASE_PATH__`` honour that prefix without a rebuild.

    A missing WEB_DIST is deliberately NOT a mount-time terminal state: every route copes
    with a missing dist per-request (404 JSON / ``check_dir=False``), so a long-lived
    ``--skip-build`` process recovers the moment a build appears on disk — no restart.
    """
    from hermes_cli.web_server import WEB_DIST, _DASHBOARD_EMBEDDED_CHAT_ENABLED, _SESSION_TOKEN, app

    # `hermes serve` is the headless backend: it must NEVER serve the browser SPA, even if a
    # dist is lying around, so only the JSON-RPC/WS/API surface is reachable.
    if os.environ.get("HERMES_SERVE_HEADLESS") == "1":

        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            # Desktop token handshake: the Electron shell boots by fetching `/` and reading
            # ``window.__HERMES_SESSION_TOKEN__`` for /api/ws auth. When headless 404'd every
            # path, a renderer whose spawn token no longer matched (e.g. after `hermes update`)
            # white-screened. Serve a token-only page at the exact root, but ONLY when the auth
            # gate is off: on a gated serve the token must never be readable without auth.
            # See #94227, #95575.
            gated = bool(getattr(application.state, "auth_required", False))
            if full_path == "" and not gated:
                return HTMLResponse(
                    "<!doctype html><html><head><script>"
                    f"window.__HERMES_SESSION_TOKEN__={json.dumps(_SESSION_TOKEN)};"
                    "window.__HERMES_AUTH_REQUIRED__=false;"
                    f"</script></head><body>{_HEADLESS_MSG}</body></html>",
                    headers=_NO_STORE,
                )
            return JSONResponse({"error": _HEADLESS_MSG}, status_code=404)
        return

    # A missing WEB_DIST is deliberately NOT a mount-time terminal state (#82614): a long-lived `hermes
    # dashboard --skip-build` process that survives a `git pull` (or starts before the first build) used to
    # install a permanent no_frontend catch-all here and could never recover — every route answered 404
    # "Frontend not built" until the process was restarted, even after `npm run build` completed. The SPA
    # routes below all cope with a missing dist per-request (`_serve_index` returns the same 404 JSON when
    # index.html is unreadable; the asset mounts use check_dir=False and 404 on missing files), so mounting
    # them unconditionally makes the dashboard recover the moment a build appears on disk — no restart
    # needed.
    def _serve_index(prefix: str = ""):
        """index.html with the session token + base-path injected.

        When the OAuth auth gate is active (``app.state.auth_required``), the legacy
        ``_SESSION_TOKEN`` is NOT injected — the SPA reads identity from ``/api/auth/me`` over
        cookie auth; ``__HERMES_AUTH_REQUIRED__`` tells it which scheme to use for /api/pty
        and /api/ws (ticket vs token).
        """
        try:
            html = (WEB_DIST / "index.html").read_text(encoding="utf-8")
        except OSError:
            # Partial build / wiped dist / permissions: same JSON 404 as a fully-missing dist.
            return JSONResponse({"error": "Frontend not built. Run: cd web && npm run build"}, status_code=404)
        chat_js = "true" if _DASHBOARD_EMBEDDED_CHAT_ENABLED else "false"
        gated = bool(getattr(app.state, "auth_required", False))
        token_js = "" if gated else f'window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";'
        bootstrap_script = (
            f"<script>{token_js}"
            f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
            f'window.__HERMES_BASE_PATH__="{prefix}";'
            f"window.__HERMES_AUTH_REQUIRED__={'true' if gated else 'false'};"
            f"</script>"
        )
        if prefix:
            # Rewrite absolute asset URLs baked into the Vite build to go through the proxy.
            for attr in ('href="/assets/', 'src="/assets/', 'href="/favicon.ico"', 'href="/fonts/',
                         'href="/ds-assets/', 'src="/ds-assets/'):
                html = html.replace(attr, attr.replace('"/', f'"{prefix}/', 1))
        theme_bootstrap = _render_active_theme_bootstrap_css()
        if theme_bootstrap:
            html = html.replace("</head>", f"{theme_bootstrap}</head>", 1)
        html = html.replace("</head>", f"{bootstrap_script}</head>", 1)
        return HTMLResponse(html, headers=_NO_STORE)

    # Built CSS contains absolute ``url(/fonts/...)`` / ``url(/ds-assets/...)`` references that
    # browsers resolve against the document origin — wrong under a proxy prefix. Intercept CSS
    # BEFORE the StaticFiles mount and rewrite when a prefix is in play.
    @application.get("/assets/{filename}.css")
    async def serve_css(filename: str, request: Request):
        css_path = WEB_DIST / "assets" / f"{filename}.css"
        if not css_path.is_file() or not css_path.resolve().is_relative_to(WEB_DIST.resolve()):
            return JSONResponse({"error": "not found"}, status_code=404)
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        css = css_path.read_text(encoding="utf-8")
        if prefix:
            for asset_dir in ("/fonts/", "/fonts-terminal/", "/ds-assets/", "/assets/"):
                for quote in ("", '"', "'"):
                    css = css.replace(f"url({quote}{asset_dir}", f"url({quote}{prefix}{asset_dir}")
        return Response(
            content=css, media_type="text/css", headers={"Cache-Control": _IMMUTABLE_ASSET_CACHE_CONTROL}
        )

    class _ImmutableAssetFiles(StaticFiles):
        """StaticFiles that marks hashed bundle assets immutable so reloads skip revalidation."""

        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            if response.status_code == 200:
                response.headers["Cache-Control"] = _IMMUTABLE_ASSET_CACHE_CONTROL
            return response

    # check_dir=False: the dist may not exist yet; StaticFiles 404s per-request until it does.
    application.mount(
        "/assets", _ImmutableAssetFiles(directory=WEB_DIST / "assets", check_dir=False), name="assets"
    )

    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        # An unmatched /api/* path is a missing endpoint, not a client-side route: return a
        # real 404 JSON instead of index.html (which breaks JSON clients with a SyntaxError).
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse({"detail": f"No such API endpoint: /{full_path}"}, status_code=404)
        file_path = WEB_DIST / full_path
        # Prevent path traversal via url-encoded sequences (%2e%2e/)
        if (
            full_path
            and file_path.resolve().is_relative_to(WEB_DIST.resolve())
            and file_path.exists()
            and file_path.is_file()
        ):
            return FileResponse(file_path)
        return _serve_index(prefix)


# ---------------------------------------------------------------------------
# Dashboard themes
# ---------------------------------------------------------------------------

# Built-in themes — label + description only; colors live in web/src/themes/presets.ts.
_BUILTIN_DASHBOARD_THEMES = [
    {"name": "default",       "label": "Hermes Teal",         "description": "Classic dark teal — the canonical Hermes look"},
    {"name": "default-large", "label": "Hermes Teal (Large)", "description": "Hermes Teal with bigger fonts and roomier spacing"},
    {"name": "nous-blue",     "label": "Nous Blue",           "description": "Light mode — vivid Nous-blue accents on cream canvas"},
    {"name": "midnight",      "label": "Midnight",            "description": "Deep blue-violet with cool accents"},
    {"name": "ember",     "label": "Ember",          "description": "Warm crimson and bronze — forge vibes"},
    {"name": "mono",      "label": "Mono",           "description": "Clean grayscale — minimal and focused"},
    {"name": "cyberpunk", "label": "Cyberpunk",      "description": "Neon green on black — matrix terminal"},
    {"name": "rose",      "label": "Rosé",           "description": "Soft pink and warm ivory — easy on the eyes"},
]


def _parse_theme_layer(value: Any, default_hex: str, default_alpha: float = 1.0) -> Optional[Dict[str, Any]]:
    """Normalise a theme layer spec (bare hex shorthand or ``{hex, alpha}`` dict); ``None`` on
    garbage so the caller falls back to a built-in default."""
    if value is None:
        return {"hex": default_hex, "alpha": default_alpha}
    if isinstance(value, str):
        return {"hex": value, "alpha": default_alpha}
    if not isinstance(value, dict):
        return None
    hex_val = value.get("hex", default_hex)
    if not isinstance(hex_val, str):
        return None
    try:
        alpha_f = float(value.get("alpha", default_alpha))
    except (TypeError, ValueError):
        alpha_f = default_alpha
    return {"hex": hex_val, "alpha": max(0.0, min(1.0, alpha_f))}


_THEME_DEFAULT_TYPOGRAPHY: Dict[str, str] = {
    "fontSans": 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    "fontMono": 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace',
    "baseSize": "15px",
    "lineHeight": "1.55",
    "letterSpacing": "0",
}
_THEME_DEFAULT_LAYOUT: Dict[str, str] = {
    "radius": "0.5rem", "density": "comfortable"
}
_THEME_OVERRIDE_KEYS = {
    "card", "cardForeground", "popover", "popoverForeground",
    "primary", "primaryForeground", "secondary", "secondaryForeground",
    "muted", "mutedForeground", "accent", "accentForeground",
    "destructive", "destructiveForeground", "success", "warning",
    "border", "input", "ring",
}

# Named asset slots; other keys under ``assets.custom`` become ``--theme-asset-custom-<key>``.
_THEME_NAMED_ASSET_KEYS = {"bg", "hero", "logo", "crest", "sidebar", "header"}

# Component-style buckets: each camelCase property under a bucket emits
# ``--component-<bucket>-<kebab-property>`` on :root, consumed by shell components.
_THEME_COMPONENT_BUCKETS = {
    "card", "header", "footer", "sidebar", "tab", "progress", "badge", "backdrop", "page"
}
_THEME_LAYOUT_VARIANTS = {"standard", "cockpit", "tiled"}

# customCSS cap so an oversized theme YAML can't blow up the payload or <style> tag.
_THEME_CUSTOM_CSS_MAX = 32 * 1024


def _dict_field(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _css_ident(key: Any) -> bool:
    return isinstance(key, str) and key.replace("-", "").replace("_", "").isalnum()


def _normalise_theme_definition(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise a user theme YAML into the wire format ``ThemeProvider`` expects; ``None`` if
    unusable. Accepts the full schema and a loose form (top-level ``colors``, bare hex).

    customCSS is clipped but intentionally NOT sanitised — themes are user-authored YAML in
    ~/.hermes/, the same trust level as config.yaml. Empty asset values are dropped so a
    theme can explicitly clear a slot.
    """
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not _nonempty_str(name):
        return None

    palette_src = _dict_field(data, "palette")
    colors_src = _dict_field(data, "colors")

    def _layer(key: str, default_hex: str, default_alpha: float = 1.0) -> Dict[str, Any]:
        parsed = _parse_theme_layer(palette_src.get(key, colors_src.get(key)), default_hex, default_alpha)
        return parsed if parsed is not None else {"hex": default_hex, "alpha": default_alpha}

    raw_noise = palette_src.get("noiseOpacity", data.get("noiseOpacity"))
    try:
        noise = float(raw_noise) if raw_noise is not None else 1.0
    except (TypeError, ValueError):
        noise = 1.0
    palette = {
        "background": _layer("background", "#041c1c", 1.0),
        "midground": _layer("midground", "#ffe6cb", 1.0),
        "foreground": _layer("foreground", "#ffffff", 0.0),
        "warmGlow": palette_src.get("warmGlow") or data.get("warmGlow") or "rgba(255, 189, 56, 0.35)",
        "noiseOpacity": noise,
    }

    typo_src = _dict_field(data, "typography")
    typography = dict(_THEME_DEFAULT_TYPOGRAPHY)
    for key in ("fontSans", "fontMono", "fontDisplay", "fontUrl", "baseSize", "lineHeight", "letterSpacing"):
        if _nonempty_str(typo_src.get(key)):
            typography[key] = typo_src[key]

    layout_src = _dict_field(data, "layout")
    layout = dict(_THEME_DEFAULT_LAYOUT)
    if _nonempty_str(layout_src.get("radius")):
        layout["radius"] = layout_src["radius"]
    density = layout_src.get("density")
    if isinstance(density, str) and density in {"compact", "comfortable", "spacious"}:
        layout["density"] = density

    color_overrides = {
        k: v for k, v in _dict_field(data, "colorOverrides").items()
        if k in _THEME_OVERRIDE_KEYS and _nonempty_str(v)
    }

    assets_src = _dict_field(data, "assets")
    assets_out: Dict[str, Any] = {k: assets_src[k] for k in _THEME_NAMED_ASSET_KEYS if _nonempty_str(assets_src.get(k))}
    custom_assets = {k: v for k, v in _dict_field(assets_src, "custom").items() if _css_ident(k) and _nonempty_str(v)}
    if custom_assets:
        assets_out["custom"] = custom_assets

    custom_css_val = data.get("customCSS")
    custom_css = custom_css_val[:_THEME_CUSTOM_CSS_MAX] if _nonempty_str(custom_css_val) else None

    component_styles: Dict[str, Dict[str, str]] = {}
    for bucket, props in _dict_field(data, "componentStyles").items():
        if bucket not in _THEME_COMPONENT_BUCKETS or not isinstance(props, dict):
            continue
        clean = {
            prop: str(value) for prop, value in props.items()
            if _css_ident(prop) and isinstance(value, (str, int, float)) and str(value).strip()
        }
        if clean:
            component_styles[bucket] = clean

    layout_variant = data.get("layoutVariant")
    if not (isinstance(layout_variant, str) and layout_variant in _THEME_LAYOUT_VARIANTS):
        layout_variant = "standard"

    result: Dict[str, Any] = {
        "name": name,
        "label": data.get("label") or name,
        "description": data.get("description", ""),
        "palette": palette,
        "typography": typography,
        "layout": layout,
        "layoutVariant": layout_variant,
    }
    if color_overrides:
        result["colorOverrides"] = color_overrides
    if assets_out:
        result["assets"] = assets_out
    if custom_css is not None:
        result["customCSS"] = custom_css
    if component_styles:
        result["componentStyles"] = component_styles
    return result


def _discover_user_themes() -> list:
    """Fully-normalised user themes from ``<launch home>/dashboard-themes/*.yaml``.

    Uses the process launch home, not ``get_hermes_home()``, so a transient profile override
    from embedded chat does not hide themes under the server's own ``HERMES_HOME``.
    """
    themes_dir = get_process_hermes_home() / "dashboard-themes"
    if not themes_dir.is_dir():
        return []
    result = []
    for f in sorted(themes_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        normalised = _normalise_theme_definition(data)
        if normalised is not None:
            result.append(normalised)
    return result


# ---------------------------------------------------------------------------
# Dashboard plugin system
# ---------------------------------------------------------------------------

def _safe_plugin_api_relpath(api_field: Any, *, dashboard_dir: Path) -> Optional[str]:
    """Validate the manifest's ``api`` field (later imported as a Python module — arbitrary
    code execution by design).

    An absolute path would swallow the plugin dir (``Path('safe') / '/tmp/evil.py'`` ->
    ``/tmp/evil.py``) and ``../`` could climb out of it (GHSA-5qr3-c538-wm9j). Returns the
    original string when the resolved path stays under ``dashboard_dir``, else ``None`` so
    the plugin still loads its static JS/CSS but its backend ``api`` is rejected.

    The web server later imports this file as a Python module via ``importlib.util.spec_from_file_location``
    (arbitrary code execution by design — that's how plugins extend the backend). Pre-#29156 the field was
    used as-is, which meant:
    """
    if not isinstance(api_field, str) or not api_field.strip():
        return None
    candidate = Path(api_field)
    if candidate.is_absolute():
        return None
    try:
        (dashboard_dir / candidate).resolve().relative_to(dashboard_dir.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return api_field


def _dashboard_plugin_search_dirs() -> List[tuple]:
    """``(root, source)`` pairs to scan, in priority order (first name wins).

    User dashboard plugins are a dashboard-owned asset (like theme YAML): resolved from the
    process launch home so they don't vanish when a request is scoped to another profile.
    When the process itself is profile-scoped (``HERMES_HOME=<root>/profiles/<name>``) the
    launch home has no ``plugins/`` — user plugins live in the hermes root — so the default
    root is scanned too; profile-local plugins stay authoritative over same-named root ones.
    The project source is gated on shared truthy semantics (``1``/``true``/``yes``/``on``):
    a bare non-empty check let ``=0``/``=false`` silently enable it (GHSA-5qr3-c538-wm9j).
    """
    from hermes_cli.plugins import get_bundled_plugins_dir
    from hermes_constants import get_default_hermes_root

    bundled_root = get_bundled_plugins_dir()
    # User dashboard plugins are a dashboard-owned asset (same category as theme YAML): resolve them from
    # the process launch home so they don't vanish when a request is scoped to another profile via a
    # context-local HERMES_HOME override (e.g. embedded /chat under --open-profile). #87197: when the
    # process itself is profile-scoped (``--profile <name>`` sets ``HERMES_HOME=<root>/profiles/<name>``),
    # the launch home is the profile directory, which has no ``plugins/`` — user plugins are installed in
    # the hermes root (``~/.hermes/plugins``). Scan the default root as well (``get_default_hermes_root()``
    # unwraps ``<root>/profiles/<name>`` → ``<root>`` and returns a custom ``HERMES_HOME`` unchanged when it
    # *is* the root), mirroring how ``hermes_cli.plugins`` resolves plugin install locations. The
    # ``seen_names`` dedupe below keeps profile-local plugins (if any) authoritative over same-named root
    # plugins.
    user_plugin_roots = [get_process_hermes_home() / "plugins"]
    root_plugins = get_default_hermes_root() / "plugins"
    if root_plugins.resolve(strict=False) != user_plugin_roots[0].resolve(strict=False):
        user_plugin_roots.append(root_plugins)
    search_dirs = [(d, "user") for d in user_plugin_roots]
    search_dirs += [(bundled_root / "memory", "bundled"), (bundled_root, "bundled")]
    # GHSA-5qr3-c538-wm9j (#29156): the previous ``os.environ.get(...)`` check treated *any* non-empty
    # string as truthy, so ``=0``, ``=false``, and ``=no`` — all of which the agent loader and operators
    # correctly read as "disabled" — silently *enabled* the untrusted project source in the web server.
    # Combined with the absolute-path RCE primitive on the manifest's ``api`` field (now patched below),
    # this turned the opt-in into a sticky always-on switch. Use the shared truthy semantics (``1`` /
    # ``true`` / ``yes`` / ``on``) so the gate matches ``hermes_cli/plugins.py`` and the documented user
    # contract.
    if env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        search_dirs.append((Path.cwd() / ".hermes" / "plugins", "project"))
    return search_dirs


def _dashboard_plugin_entry(data: Dict[str, Any], name: str, dashboard_dir: Path, source: str) -> Dict[str, Any]:
    # Tab options: ``path`` + ``position`` for a new tab, optional ``override`` to replace a
    # built-in route, and ``hidden`` to register component/slots without adding a tab.
    raw_tab = data.get("tab", {}) if isinstance(data.get("tab"), dict) else {}
    tab_info = {"path": raw_tab.get("path", f"/{name}"), "position": raw_tab.get("position", "end")}
    override_path = raw_tab.get("override")
    if isinstance(override_path, str) and override_path.startswith("/"):
        tab_info["override"] = override_path
    if bool(raw_tab.get("hidden")):
        tab_info["hidden"] = True
    # Slots the plugin populates via ``window.registerSlot(pluginName, slotName, Component)``.
    slots_src = data.get("slots")
    slots = [s for s in slots_src if isinstance(s, str) and s] if isinstance(slots_src, list) else []
    # Validate ``api`` at discovery time so the cached value is already safe for the importer.
    raw_api = data.get("api")
    safe_api = _safe_plugin_api_relpath(raw_api, dashboard_dir=dashboard_dir)
    if raw_api and safe_api is None:
        _log.warning(
            "Plugin %s: refusing unsafe api path %r (must be a "
            "relative file inside the plugin's dashboard/ "
            "directory); backend routes from this plugin will "
            "not be mounted",
            name, raw_api,
        )
    return {
        "name": name,
        "label": data.get("label", name),
        "description": data.get("description", ""),
        "icon": data.get("icon", "Puzzle"),
        "version": data.get("version", "0.0.0"),
        "tab": tab_info,
        "slots": slots,
        "entry": data.get("entry", "dist/index.js"),
        "css": data.get("css"),
        "has_api": bool(safe_api),
        "source": source,
        "_dir": str(dashboard_dir),
        "_api_file": safe_api,
    }


def _discover_dashboard_plugins() -> list:
    """Scan ``<plugins root>/*/dashboard/manifest.json`` across user, bundled and (opt-in)
    project plugin sources — same three sources as ``hermes_cli.plugins``."""
    plugins = []
    seen_names: set = set()
    for plugins_root, source in _dashboard_plugin_search_dirs():
        if not plugins_root.is_dir():
            continue
        with os.scandir(plugins_root) as scan:
            children = sorted((Path(e.path) for e in scan), key=lambda p: p.name)
        for child in children:
            manifest_file = child / "dashboard" / "manifest.json"
            if not child.is_dir() or not manifest_file.exists():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                name = data.get("name", child.name)
                if name in seen_names:
                    continue
                seen_names.add(name)
                plugins.append(_dashboard_plugin_entry(data, name, child / "dashboard", source))
            except Exception as exc:
                _log.warning("Bad dashboard plugin manifest %s: %s", manifest_file, exc)
                continue
    return plugins


def _strip_dashboard_manifest(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in p.items() if not k.startswith("_")}


_PLUGINS_HUB_CACHE_TTL_SECONDS = 5.0
_plugins_hub_cache: Optional[Dict[str, Any]] = None
_plugins_hub_cache_expires_at = 0.0
_plugins_hub_cache_lock = threading.Lock()


def _invalidate_plugins_hub_cache() -> None:
    global _plugins_hub_cache, _plugins_hub_cache_expires_at
    with _plugins_hub_cache_lock:
        _plugins_hub_cache = None
        _plugins_hub_cache_expires_at = 0.0


_plugins_hub_probe_inflight: set = set()
_plugins_hub_probe_lock = threading.Lock()


def _schedule_check_fn_probe(fn) -> Optional[threading.Thread]:
    """Warm a cold ``check_fn`` verdict off the request path.

    The hub read path only consumes cached availability; the only other warmer is the
    tool-schema build, which a dashboard-only session never runs — so a cold cache would
    report ``auth_required=False`` forever. Daemon-thread probe, deduplicated per function;
    the short hub TTL surfaces the verdict on the next fetch. Returns ``None`` when a probe
    for *fn* is already in flight.
    """
    with _plugins_hub_probe_lock:
        if fn in _plugins_hub_probe_inflight:
            return None
        _plugins_hub_probe_inflight.add(fn)

    def _probe():
        try:
            from tools.registry import _check_fn_cached

            _check_fn_cached(fn)
        except Exception:
            pass
        finally:
            with _plugins_hub_probe_lock:
                _plugins_hub_probe_inflight.discard(fn)

    thread = threading.Thread(target=_probe, name="plugins-hub-checkfn-probe", daemon=True)
    thread.start()
    return thread


def _plugin_auth_hint(name: str, provides_tools: list) -> tuple:
    """``(auth_required, auth_command)`` from last-known cached tool availability only.

    A missing cache entry is "unknown": schedule a background probe rather than probing
    inline (which would starve the root event loop), so the short hub TTL picks it up.
    """
    try:
        from tools.registry import get_cached_check_fn_result, registry
        for tname in provides_tools:
            entry = registry.get_entry(tname)
            if not entry or not entry.check_fn:
                continue
            cached_result = get_cached_check_fn_result(entry.check_fn)
            if cached_result is None:
                _schedule_check_fn_probe(entry.check_fn)
            elif cached_result is False:
                return True, f"hermes auth {name}"
    except Exception:
        pass
    return False, ""


def _merged_plugins_hub(force_refresh: bool = False) -> Dict[str, Any]:
    """Agent discovery + dashboard manifests + provider picker metadata.

    IMPORTANT: powers a dashboard request path, so it must stay read-only and cheap — never
    execute tool ``check_fn`` probes here (imports, auth/network checks would starve the root
    event loop). Only cached availability is consumed and the payload is memoized briefly to
    collapse the dashboard's bursty duplicate fetches.
    """
    from hermes_cli.web_server_memory import _discover_memory_provider_statuses, _normalize_memory_provider_name
    from hermes_cli.web_server import _get_dashboard_plugins
    from hermes_cli.config import get_hermes_home, load_config
    global _plugins_hub_cache, _plugins_hub_cache_expires_at
    now = time.monotonic()
    if not force_refresh:
        with _plugins_hub_cache_lock:
            if _plugins_hub_cache is not None and now < _plugins_hub_cache_expires_at:
                return _plugins_hub_cache

    started_at = time.monotonic()
    from hermes_cli.plugins_cmd import (
        _discover_all_plugins,
        _get_current_context_engine,
        _get_current_memory_provider,
        _discover_context_engines,
        _get_disabled_set,
        _get_enabled_set,
        _read_manifest as _read_plugin_manifest_at,
    )

    dashboard_list = _get_dashboard_plugins()
    dash_by_name = {str(p["name"]): p for p in dashboard_list}
    disabled_set = _get_disabled_set()
    enabled_set = _get_enabled_set()
    hidden_plugins: list = cfg_get(load_config(), "dashboard", "hidden_plugins", default=[]) or []
    plugins_root_resolved = (get_hermes_home() / "plugins").resolve()
    rows: List[Dict[str, Any]] = []

    for name, version, description, source, dir_str, key in _discover_all_plugins():
        # Both the path-derived key (nested category plugins) and the bare manifest name
        # count for enabled/disabled state, matching the runtime loader's back-compat lookup.
        aliases = {name, key} if key else {name}
        if aliases & disabled_set:
            runtime_status = "disabled"
        elif aliases & enabled_set:
            runtime_status = "enabled"
        else:
            runtime_status = "inactive"

        dir_path = Path(dir_str)
        dm = dash_by_name.get(name)
        try:
            dir_path.resolve().relative_to(plugins_root_resolved)
            under_user_tree = True
        except ValueError:
            under_user_tree = False
        can_remove_update = source in {"user", "git"} and under_user_tree and dir_path.is_dir()

        provides_tools = _read_plugin_manifest_at(dir_path).get("provides_tools") or []
        auth_required, auth_command = _plugin_auth_hint(name, provides_tools) if provides_tools else (False, "")

        rows.append({
            "name": name,
            "version": version or "",
            "description": description or "",
            "source": source,
            "runtime_status": runtime_status,
            "has_dashboard_manifest": dm is not None or (dir_path / "dashboard" / "manifest.json").exists(),
            "dashboard_manifest": _strip_dashboard_manifest(dm) if dm else None,
            "path": dir_str,
            "can_remove": can_remove_update,
            "can_update_git": can_remove_update and (dir_path / ".git").exists(),
            "auth_required": auth_required,
            "auth_command": auth_command,
            "user_hidden": name in hidden_plugins,
        })

    agent_names = {r["name"] for r in rows}
    orphan_dashboard = [_strip_dashboard_manifest(p) for p in dashboard_list if str(p["name"]) not in agent_names]
    memory_providers = _discover_memory_provider_statuses()
    try:
        context_engines = [{"name": n, "description": desc} for n, desc in _discover_context_engines()]
    except Exception:
        context_engines = []

    payload = {
        "plugins": rows,
        "orphan_dashboard_plugins": orphan_dashboard,
        "providers": {
            "memory_provider": _normalize_memory_provider_name(_get_current_memory_provider()),
            "memory_options": memory_providers,
            "context_engine": _get_current_context_engine(),
            "context_options": context_engines,
        },
    }
    duration = time.monotonic() - started_at
    if duration >= 0.25:
        _log.info(
            "plugins/hub rebuilt in %.3fs (plugins=%d memory_options=%d)", duration, len(rows), len(memory_providers)
        )
    with _plugins_hub_cache_lock:
        _plugins_hub_cache = payload
        _plugins_hub_cache_expires_at = time.monotonic() + _PLUGINS_HUB_CACHE_TTL_SECONDS
    return payload


def _plugin_api_mount_skip_reason(plugin: Dict[str, Any], enabled_set: set, disabled_set: set) -> Optional[str]:
    """Why a plugin's backend ``api`` must NOT be imported, or None when it may be.

    User plugins must be in ``plugins.enabled`` and not ``plugins.disabled`` before their
    Python runs (GHSA-mcfc-hp25-cjv7); bundled plugins are trusted but respect an explicit
    disable; project plugins (``./.hermes/plugins/``) ship with the CWD and are
    attacker-controlled when opening a malicious repo — never auto-imported (GHSA-5qr3-c538-wm9j).
    """
    source, plugin_name = plugin.get("source"), plugin.get("name", "")
    if source in ("user", "bundled") and plugin_name in disabled_set:
        return "explicitly disabled"
    if source == "user" and plugin_name not in enabled_set:
        return "not in plugins.enabled"
    return None


def _mount_plugin_api_routes():
    """Import and mount backend API routes from plugins that declare them.

    Each plugin's ``api`` file must expose a ``router`` (FastAPI APIRouter), mounted under
    ``/api/plugins/<name>/``. See ``_plugin_api_mount_skip_reason`` for the trust gates.

    Backend import is restricted to ``bundled`` and ``user`` sources. Project plugins
    (``./.hermes/plugins/``) ship with the CWD and are therefore attacker-controlled in any threat model
    where the user opens a malicious repo; they can extend the dashboard UI via static JS/CSS but their
    Python ``api`` file is never auto-imported by the web server. See GHSA-5qr3-c538-wm9j (#29156).
    Additionally, user plugins must be explicitly enabled via the ``plugins.enabled`` allow-list in
    config.yaml before their backend code is imported. Without this gate, an installed-but-not-enabled
    plugin's Python code would execute at dashboard startup — a code execution vector that bypasses the
    user's intent. (#46435, GHSA-mcfc-hp25-cjv7)
    """
    from hermes_cli.web_server import _get_dashboard_plugins, app
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()

    for plugin in _get_dashboard_plugins():
        api_file_name = plugin.get("_api_file")
        if not api_file_name:
            continue
        skip = _plugin_api_mount_skip_reason(plugin, enabled_set, disabled_set)
        if skip:
            _log.debug("Plugin %s: skipping API mount (%s)", plugin.get("name", ""), skip)
            continue
        if plugin.get("source") == "project":
            _log.warning(
                "Plugin %s: ignoring backend api=%s (project plugins may "
                "not auto-import Python code; move the plugin to "
                "~/.hermes/plugins/ if you trust it)",
                plugin["name"], api_file_name,
            )
            continue
        dashboard_dir = Path(plugin["_dir"])
        api_path = dashboard_dir / api_file_name
        try:
            api_path.resolve().relative_to(dashboard_dir.resolve())
        except (OSError, RuntimeError, ValueError):
            # Discovery already filters this; defence in depth in case ``_dir`` was tampered
            # with after caching or a future caller bypasses the validator.
            _log.warning(
                "Plugin %s: refusing to import api file outside its "
                "dashboard directory (%s)", plugin["name"], api_path,
            )
            continue
        if not api_path.exists():
            _log.warning("Plugin %s declares api=%s but file not found", plugin["name"], api_file_name)
            continue
        try:
            module_name = f"hermes_dashboard_plugin_{plugin['name']}"
            spec = importlib.util.spec_from_file_location(module_name, api_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            # Register in sys.modules BEFORE exec_module so pydantic/FastAPI can resolve
            # string annotations (``from __future__ import annotations``) by module name.
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            router = getattr(mod, "router", None)
            if router is None:
                _log.warning("Plugin %s api file has no 'router' attribute", plugin["name"])
                continue
            app.include_router(router, prefix=f"/api/plugins/{plugin['name']}")
            _log.info("Mounted plugin API routes: /api/plugins/%s/", plugin["name"])
        except Exception as exc:
            _log.warning("Failed to load plugin %s API routes: %s", plugin["name"], exc)
