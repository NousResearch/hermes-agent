"""Regression coverage for #28987 — dashboard JS bundles must be served
with a JavaScript MIME type, not ``text/plain``.

The bug surfaced only on native Windows where ``HKEY_CLASSES_ROOT\\.js``
can be set to ``Content Type = text/plain`` by various installers.
Python's ``mimetypes.init()`` reads that registry value, Starlette's
``FileResponse`` / ``StaticFiles`` then propagate it to the
``Content-Type`` header, and browsers refuse to execute the Vite
ESM bundle under strict MIME checking — leaving the dashboard blank.

We can't easily run the test on Windows in CI, but we *can* reproduce
the failure mode portably by **poisoning the strict ``mimetypes`` map
before importing the web server** and asserting that the module's
startup normalisation undoes the damage.  This catches both the
Windows-registry case and any future regression where someone adds a
``mimetypes.add_type('text/plain', '.js')`` call elsewhere.
"""

from __future__ import annotations

import importlib
import mimetypes
import sys
from pathlib import Path

import pytest


# JavaScript module scripts only execute when ``Content-Type`` matches
# the HTML spec's "JavaScript MIME type" essence — which is exactly
# ``text/javascript``, ``application/javascript``, or any of the
# obsolete-but-still-allowed variants.  RFC 9239 (May 2022) makes
# ``text/javascript`` the only IANA-blessed value; we accept either
# spelling so a future Starlette / Python change doesn't flake the test.
JAVASCRIPT_MIME_PREFIXES = ("text/javascript", "application/javascript")


def _is_javascript_mime(content_type: str) -> bool:
    """True if ``content_type`` is a JavaScript-MIME-type-essence match."""
    if not content_type:
        return False
    essence = content_type.split(";", 1)[0].strip().lower()
    return essence in JAVASCRIPT_MIME_PREFIXES


@pytest.fixture
def poisoned_mimetypes(monkeypatch):
    """Save the live ``mimetypes`` strict map, poison ``.js`` / ``.mjs``
    with ``text/plain`` to simulate the Windows registry pollution,
    then restore on teardown so other tests aren't affected.
    """
    original = mimetypes.types_map.copy()
    yield_target = mimetypes.types_map
    yield_target[".js"] = "text/plain"
    yield_target[".mjs"] = "text/plain"
    try:
        yield
    finally:
        mimetypes.types_map.clear()
        mimetypes.types_map.update(original)


class TestMimeNormalizationFunction:
    """Unit-level coverage for ``_normalize_web_asset_mime_types``."""

    def test_overrides_text_plain_on_dot_js(self, poisoned_mimetypes):
        # Sanity: the fixture really did poison the map.  Without this
        # assert, a future change that silently un-poisons ``.js``
        # would make the rest of the test pass for the wrong reason.
        assert mimetypes.guess_type("a.js")[0] == "text/plain"

        from hermes_cli.web_server import _normalize_web_asset_mime_types
        _normalize_web_asset_mime_types()

        assert _is_javascript_mime(mimetypes.guess_type("a.js")[0]), (
            "Module bundles served as text/plain are rejected by browsers "
            "under strict MIME checking (#28987)."
        )

    def test_overrides_text_plain_on_dot_mjs(self, poisoned_mimetypes):
        # ``.mjs`` is what Vite/Rollup uses for ESM-only builds; even on
        # vanilla Python ``mimetypes`` doesn't ship a mapping for it,
        # so this codifies the explicit override.
        from hermes_cli.web_server import _normalize_web_asset_mime_types
        _normalize_web_asset_mime_types()

        assert _is_javascript_mime(mimetypes.guess_type("a.mjs")[0])

    def test_idempotent(self):
        # Importing the module once at process start is enough — but
        # calling the helper again must not break anything.
        from hermes_cli.web_server import _normalize_web_asset_mime_types
        _normalize_web_asset_mime_types()
        _normalize_web_asset_mime_types()
        _normalize_web_asset_mime_types()

        assert _is_javascript_mime(mimetypes.guess_type("a.js")[0])

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("bundle.js", JAVASCRIPT_MIME_PREFIXES),
            ("worker.mjs", JAVASCRIPT_MIME_PREFIXES),
            ("legacy.cjs", JAVASCRIPT_MIME_PREFIXES),
            ("styles.css", ("text/css",)),
            ("logo.svg", ("image/svg+xml",)),
            ("font.woff2", ("font/woff2",)),
            ("font.woff", ("font/woff",)),
            ("icon.ico", ("image/x-icon",)),
            ("bundle.js.map", ("application/json",)),
            ("module.wasm", ("application/wasm",)),
            ("manifest.webmanifest", ("application/manifest+json",)),
        ],
    )
    def test_all_overrides_apply(self, filename, expected):
        # Every extension in ``_WEB_ASSET_MIME_OVERRIDES`` should resolve
        # to its intended value.  Parametrising one row per asset means
        # the test name pinpoints the failing extension when something
        # regresses.
        from hermes_cli.web_server import _normalize_web_asset_mime_types
        _normalize_web_asset_mime_types()

        actual = mimetypes.guess_type(filename)[0]
        assert actual in expected, (
            f"{filename}: expected one of {expected!r}, got {actual!r}"
        )


class TestMimeNormalizationAtImport:
    """Catch the realistic Windows scenario: the registry pollutes the
    map BEFORE Python imports ``hermes_cli.web_server``.  After the
    import completes, ``mimetypes.guess_type('.js')`` must return a
    JavaScript MIME type.  This is the exact ordering the bug report
    describes."""

    def test_import_normalizes_poisoned_map(self):
        # Reload pattern: remove the cached module so re-import runs
        # the module body (which calls ``_normalize_web_asset_mime_types``).
        original_types_map = mimetypes.types_map.copy()
        had_module = "hermes_cli.web_server" in sys.modules
        cached_module = sys.modules.get("hermes_cli.web_server")
        try:
            mimetypes.types_map[".js"] = "text/plain"
            mimetypes.types_map[".mjs"] = "text/plain"
            assert mimetypes.guess_type("a.js")[0] == "text/plain"

            sys.modules.pop("hermes_cli.web_server", None)
            importlib.import_module("hermes_cli.web_server")

            assert _is_javascript_mime(mimetypes.guess_type("a.js")[0])
            assert _is_javascript_mime(mimetypes.guess_type("a.mjs")[0])
        finally:
            mimetypes.types_map.clear()
            mimetypes.types_map.update(original_types_map)
            # Re-apply the normalization so other tests that import
            # web_server see the corrected map (the original test
            # cache may have been from before any poison).
            if cached_module is not None:
                sys.modules["hermes_cli.web_server"] = cached_module
            elif not had_module:
                sys.modules.pop("hermes_cli.web_server", None)


class TestStaticFileServingEndToEnd:
    """Drive a real Starlette ``TestClient`` against a synthetic build
    directory.  Catches regressions in the wire-up between
    ``mount_spa()``, ``StaticFiles``, and the corrected MIME map."""

    @pytest.fixture
    def fake_dashboard(self, tmp_path, monkeypatch, poisoned_mimetypes):
        """Build a minimal ``WEB_DIST`` that looks like a real Vite
        build (index.html + hashed asset bundles) and point the web
        server at it for the duration of one test.

        The ``poisoned_mimetypes`` fixture simulates the Windows-
        registry pollution that motivated #28987.  We then re-apply
        ``_normalize_web_asset_mime_types`` to mirror what happens at
        ``import hermes_cli.web_server`` time — without that line,
        the test would just be exercising whatever map state happened
        to be cached when an earlier test imported the module.
        """
        try:
            from starlette.testclient import TestClient
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("fastapi / starlette not installed")

        # Reapply the production-startup normalisation *after* the
        # fixture's poison has run, exactly like the module-level
        # call does at process start.
        from hermes_cli.web_server import _normalize_web_asset_mime_types
        _normalize_web_asset_mime_types()

        dist = tmp_path / "web_dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text(
            '<!doctype html><html><head>'
            '<script type="module" src="/assets/index-abc.js"></script>'
            '<link rel="stylesheet" href="/assets/index-abc.css">'
            '</head><body><div id="root"></div></body></html>'
        )
        (dist / "assets" / "index-abc.js").write_text(
            "export const HERMES_DASHBOARD = 'ok';\n"
        )
        (dist / "assets" / "index-abc.mjs").write_text(
            "export const ESM_VARIANT = true;\n"
        )
        (dist / "assets" / "index-abc.js.map").write_text('{"version":3}')
        (dist / "assets" / "logo.svg").write_text("<svg/>")
        (dist / "favicon.ico").write_bytes(b"\x00\x00")

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "WEB_DIST", dist)

        # Build a fresh app so we can re-mount with the synthetic dist
        # (the module-level ``app`` already has the real (possibly
        # missing) build mounted at import time).
        application = FastAPI()
        ws.mount_spa(application)
        client = TestClient(application)
        return client

    def test_js_bundle_is_served_as_javascript(self, fake_dashboard):
        """The exact assertion from #28987's reproduction recipe:
        ``Content-Type`` on a built JS bundle must be a JavaScript
        MIME type, not ``text/plain``."""
        resp = fake_dashboard.get("/assets/index-abc.js")
        assert resp.status_code == 200, resp.text
        assert _is_javascript_mime(resp.headers["content-type"]), (
            f"Expected JavaScript MIME, got: {resp.headers['content-type']!r}"
        )
        assert b"HERMES_DASHBOARD" in resp.content

    def test_mjs_bundle_is_served_as_javascript(self, fake_dashboard):
        resp = fake_dashboard.get("/assets/index-abc.mjs")
        assert resp.status_code == 200, resp.text
        assert _is_javascript_mime(resp.headers["content-type"]), (
            f"Expected JavaScript MIME, got: {resp.headers['content-type']!r}"
        )

    def test_source_map_is_served_as_json(self, fake_dashboard):
        # Source maps aren't strictly required by browsers, but devtools
        # complain loudly when they come back as ``text/plain``.
        resp = fake_dashboard.get("/assets/index-abc.js.map")
        assert resp.status_code == 200
        essence = resp.headers["content-type"].split(";", 1)[0].strip()
        assert essence == "application/json"

    def test_svg_is_served_as_image(self, fake_dashboard):
        # Inline ``<img src="…svg">`` and CSS ``url(...svg)`` both
        # require a real ``image/`` MIME — ``text/plain`` would break
        # rendering on the dashboard's brand assets.
        resp = fake_dashboard.get("/assets/logo.svg")
        assert resp.status_code == 200
        essence = resp.headers["content-type"].split(";", 1)[0].strip()
        assert essence == "image/svg+xml"

    def test_index_html_still_served_as_html(self, fake_dashboard):
        # Make sure the MIME normalisation didn't accidentally
        # downgrade the SPA shell itself.
        resp = fake_dashboard.get("/")
        assert resp.status_code == 200
        essence = resp.headers["content-type"].split(";", 1)[0].strip()
        assert essence == "text/html"
        assert "id=\"root\"" in resp.text


class TestMimeOverrideRegistryShape:
    """The override table is a public-ish surface (other modules might
    eventually want to extend it).  Lock its invariants down so a
    future drive-by edit doesn't silently break things."""

    def test_js_and_mjs_both_resolve_to_javascript(self):
        from hermes_cli.web_server import _WEB_ASSET_MIME_OVERRIDES
        assert _is_javascript_mime(_WEB_ASSET_MIME_OVERRIDES[".js"])
        assert _is_javascript_mime(_WEB_ASSET_MIME_OVERRIDES[".mjs"])

    def test_every_value_has_a_slash(self):
        """Sanity guard against typos like ``"text"`` or ``"javascript"``
        — every value must be a real ``type/subtype`` pair."""
        from hermes_cli.web_server import _WEB_ASSET_MIME_OVERRIDES
        for ext, mime in _WEB_ASSET_MIME_OVERRIDES.items():
            assert "/" in mime, f"{ext!r} maps to malformed MIME {mime!r}"
            assert not mime.startswith(" "), f"{ext!r} value has leading space"

    def test_every_extension_starts_with_dot(self):
        """``mimetypes.add_type`` requires extensions to include the
        leading dot — guard against ``js`` (no dot) sneaking in."""
        from hermes_cli.web_server import _WEB_ASSET_MIME_OVERRIDES
        for ext in _WEB_ASSET_MIME_OVERRIDES:
            assert ext.startswith("."), f"Extension {ext!r} missing leading dot"
