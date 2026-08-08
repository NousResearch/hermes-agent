"""SPA mount + critical-CSS shim helpers for the Hermes dashboard (god-file extraction).

Lifted verbatim from ``hermes_cli/web_server.py`` (shard s5, cluster c17 —
``spa_mount_mixin``): ``mount_spa`` (with its nested route handlers), the
active-theme bootstrap-CSS renderer, prefix normalisation, and the immutable
asset-cache header constant.  ``web_server`` re-exports the public names so
existing call sites and tests keep working.

Import discipline (mirrors ``cli_commands_mixin.py``): ``web_server``-internal
symbols (``app``, ``WEB_DIST``, ``_SESSION_TOKEN``,
``_DASHBOARD_EMBEDDED_CHAT_ENABLED``, ``_log``, ``cfg_get``, ``load_config``,
``_BUILTIN_DASHBOARD_THEMES``, ``_discover_user_themes``,
``_THEME_DEFAULT_TYPOGRAPHY``) are imported LAZILY inside the functions that
use them — resolved at call time, so tests that monkeypatch
``web_server.load_config`` / ``web_server.WEB_DIST`` keep working.
"""

import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles



def _normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header value.

    Thin re-export of :func:`hermes_cli.dashboard_auth.prefix.normalise_prefix`
    — the single source of truth lives in the dashboard_auth package so
    the gate middleware, the OAuth routes, the cookie helpers, and the
    SPA mount all agree on validation rules.
    """
    from hermes_cli.dashboard_auth.prefix import normalise_prefix
    return normalise_prefix(raw)


def _render_active_theme_bootstrap_css() -> str:
    """Critical-CSS shim for the active user theme.

    Returns a ``<style>`` block with the ``:root`` CSS variables that
    ``ThemeProvider.applyTheme()`` installs once the
    ``/api/dashboard/themes`` round-trip completes.  The goal is to
    eliminate the green flash where the first paint shows the bundle's
    default Hermes Teal canvas before the SPA flips the configured user
    theme into place.

    Built-in themes return an empty string — their full definitions live
    in ``web/src/themes/presets.ts`` and are applied by the bundle
    before paint, so no shim is needed for them.
    """
    from hermes_cli.web_server import (
        _BUILTIN_DASHBOARD_THEMES,
        _THEME_DEFAULT_TYPOGRAPHY,
        _discover_user_themes,
        _log,
        cfg_get,
        load_config,
    )
    try:
        config = load_config()
        active = cfg_get(config, "dashboard", "theme", default="default")
        if not active or not isinstance(active, str):
            return ""
        # Built-in: the bundle already owns the definition, no flash.
        if any(b["name"] == active for b in _BUILTIN_DASHBOARD_THEMES):
            return ""
        for theme in _discover_user_themes():
            if theme.get("name") != active:
                continue
            palette = theme.get("palette") or {}
            bg = palette.get("background") or {}
            mg = palette.get("midground") or {}
            bg_hex = bg.get("hex", "#0a0a0a") if isinstance(bg, dict) else "#0a0a0a"
            mg_hex = mg.get("hex", "#e5e5e5") if isinstance(mg, dict) else "#e5e5e5"
            typo = theme.get("typography") or {}
            font_sans = typo.get("fontSans") or _THEME_DEFAULT_TYPOGRAPHY["fontSans"]
            base_size = typo.get("baseSize") or _THEME_DEFAULT_TYPOGRAPHY["baseSize"]
            # Defensive ``</style>`` escape — current values are well-known
            # hex/font strings, but this keeps the helper safe if it is
            # later extended to ship user-authored CSS literals.
            def _esc(s: str) -> str:
                return str(s).replace("</", "<\\/")
            # Variable names MUST match what the bundle actually consumes:
            #   - ``--background-base`` / ``--midground-base`` come from
            #     ``layerVars()`` in ``web/src/themes/context.tsx``.
            #   - ``--theme-font-sans`` / ``--theme-base-size`` come from
            #     ``typographyVars()`` there, and ``index.css`` applies them
            #     via ``html{font-family:var(--theme-font-sans);
            #     font-size:var(--theme-base-size)}``.
            # The ``html,body`` canvas rule references the SAME variables
            # instead of literal values so runtime theme switches stay
            # live: ``applyTheme()`` writes these vars as inline styles on
            # ``documentElement``, which outrank this stylesheet block in
            # the cascade — the rule below re-resolves automatically and
            # never goes stale when the user picks a different theme.
            return (
                '<style id="hermes-theme-bootstrap">'
                ":root{"
                f"--background-base:{_esc(bg_hex)};"
                f"--midground-base:{_esc(mg_hex)};"
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


# Hashed bundle assets (``/assets/<name>-<contenthash>.<ext>``) are immutable
# by construction: any content change produces a new filename, and the entry
# point (index.html) is served ``no-store`` so it always references the
# current hashes. A year-long immutable cache lets browsers skip even the
# revalidation round-trip on every dashboard load.
_IMMUTABLE_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"


def mount_spa(application: FastAPI):
    """Mount the built SPA. Falls back to index.html for client-side routing.

    The session token is injected into index.html via a ``<script>`` tag so
    the SPA can authenticate against protected API endpoints without a
    separate (unauthenticated) token-dispensing endpoint.

    When served behind a path-prefix reverse proxy (e.g.
    ``mission-control.tilos.com/hermes/*`` -> local Caddy -> :9119), the
    proxy injects ``X-Forwarded-Prefix: /hermes`` on every request. We
    rewrite the served ``index.html`` so absolute asset URLs (``/assets/...``)
    and the SPA's runtime ``__HERMES_BASE_PATH__`` honour that prefix
    without rebuilding the bundle.
    """
    from hermes_cli.web_server import (
        WEB_DIST,
        _DASHBOARD_EMBEDDED_CHAT_ENABLED,
        _SESSION_TOKEN,
        app,
    )
    # `hermes serve` is the headless backend: it must NEVER serve the browser
    # SPA, even if a dist is lying around from a prior `dashboard`/build. Take
    # the no-frontend path so only the JSON-RPC/WS/API surface is reachable.
    _headless = os.environ.get("HERMES_SERVE_HEADLESS") == "1"
    if _headless or not WEB_DIST.exists():
        _msg = (
            "Headless backend (hermes serve): web UI disabled — use "
            "`hermes dashboard` for the browser UI."
            if _headless
            else "Frontend not built. Run: cd web && npm run build"
        )

        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            return JSONResponse({"error": _msg}, status_code=404)
        return

    _index_path = WEB_DIST / "index.html"

    def _serve_index(prefix: str = ""):
        """Return index.html with the session token + base-path injected.

        ``prefix`` is the normalised ``X-Forwarded-Prefix`` (e.g. ``/hermes``)
        or empty string when served at root.

        When the OAuth auth gate is active (``app.state.auth_required``),
        the legacy ``_SESSION_TOKEN`` is NOT injected — the SPA reads
        identity from ``/api/auth/me`` over cookie auth instead.  The
        ``__HERMES_AUTH_REQUIRED__`` flag lets the SPA pick the right
        auth scheme for /api/pty and /api/ws (ticket vs token).
        """
        try:
            html = _index_path.read_text(encoding="utf-8")
        except OSError:
            # The dist dir existed at mount time but index.html is missing or
            # unreadable now (partial build, wiped dist, permissions). Without
            # this guard every request raises FileNotFoundError (500). Return
            # the same JSON 404 payload mount_spa uses for a fully-missing
            # dist so clients get a clear, consistent signal.
            return JSONResponse(
                {"error": "Frontend not built. Run: cd web && npm run build"},
                status_code=404,
            )
        chat_js = "true" if _DASHBOARD_EMBEDDED_CHAT_ENABLED else "false"
        gated = bool(getattr(app.state, "auth_required", False))
        gated_js = "true" if gated else "false"
        if gated:
            bootstrap_script = (
                f"<script>"
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        else:
            bootstrap_script = (
                f'<script>window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";'
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        if prefix:
            # Rewrite absolute asset URLs baked into the Vite build so the
            # browser fetches them through the same proxy prefix.
            html = html.replace('href="/assets/', f'href="{prefix}/assets/')
            html = html.replace('src="/assets/', f'src="{prefix}/assets/')
            html = html.replace('href="/favicon.ico"', f'href="{prefix}/favicon.ico"')
            html = html.replace('href="/fonts/', f'href="{prefix}/fonts/')
            html = html.replace('href="/ds-assets/', f'href="{prefix}/ds-assets/')
            html = html.replace('src="/ds-assets/', f'src="{prefix}/ds-assets/')
        # Theme flash mitigation: when the active theme is a user theme
        # (``HERMES_HOME/dashboard-themes/<name>.yaml``), inject a minimal
        # critical-CSS block so the first paint uses the target palette.
        # Without this the SPA paints the default Hermes Teal canvas, then
        # ``ThemeProvider`` flips the CSS variables once
        # ``/api/dashboard/themes`` resolves.  Built-in themes are already
        # in the bundle's ``presets.ts`` so no shim is needed for them.
        theme_bootstrap = _render_active_theme_bootstrap_css()
        if theme_bootstrap:
            html = html.replace("</head>", f"{theme_bootstrap}</head>", 1)
        html = html.replace("</head>", f"{bootstrap_script}</head>", 1)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # When served behind a path-prefix proxy, the built CSS contains
    # absolute ``url(/fonts/...)`` and ``url(/ds-assets/...)`` references.
    # Browsers resolve those against the document origin, which means
    # under ``/hermes`` they'd hit ``mission-control.tilos.com/fonts/...``
    # (the MC Pages app), not the Hermes backend. Intercept CSS asset
    # requests BEFORE the StaticFiles mount and rewrite the absolute paths
    # when a prefix is in play.
    @application.get("/assets/{filename}.css")
    async def serve_css(filename: str, request: Request):
        css_path = WEB_DIST / "assets" / f"{filename}.css"
        if not css_path.is_file() or not css_path.resolve().is_relative_to(
            WEB_DIST.resolve()
        ):
            return JSONResponse({"error": "not found"}, status_code=404)
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        css = css_path.read_text(encoding="utf-8")
        if prefix:
            for asset_dir in ("/fonts/", "/fonts-terminal/", "/ds-assets/", "/assets/"):
                css = css.replace(f"url({asset_dir}", f"url({prefix}{asset_dir}")
                css = css.replace(f"url(\"{asset_dir}", f"url(\"{prefix}{asset_dir}")
                css = css.replace(f"url('{asset_dir}", f"url('{prefix}{asset_dir}")
        return Response(
            content=css,
            media_type="text/css",
            headers={"Cache-Control": _IMMUTABLE_ASSET_CACHE_CONTROL},
        )

    class _ImmutableAssetFiles(StaticFiles):
        """StaticFiles that marks hashed bundle assets immutable.

        Everything under ``/assets/`` carries a Vite content hash in its
        filename, so a given URL's bytes can never change — a rebuild
        produces a NEW filename referenced by a fresh (``no-store``)
        index.html. Without this header every dashboard load re-validated
        each chunk; with it the browser serves reloads straight from its
        HTTP cache.
        """

        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            if response.status_code == 200:
                response.headers["Cache-Control"] = _IMMUTABLE_ASSET_CACHE_CONTROL
            return response

    application.mount(
        "/assets", _ImmutableAssetFiles(directory=WEB_DIST / "assets"), name="assets"
    )

    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        # An unmatched /api/* path is a missing/renamed endpoint, NOT a
        # client-side route. Falling through to index.html here returns
        # `<!doctype html>` with status 200, which makes JSON clients (the
        # desktop app's fetchJson, dashboard fetch wrappers) blow up with an
        # opaque `SyntaxError: Unexpected token '<'`. Return a real 404 JSON
        # so the caller sees a clear "no such endpoint" instead.
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse(
                {"detail": f"No such API endpoint: /{full_path}"},
                status_code=404,
            )
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
