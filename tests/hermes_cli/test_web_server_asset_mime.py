"""Dashboard asset MIME types are platform-independent (#28987).

On Windows, ``mimetypes.init()`` reads the registry, where ``.js`` can be
mapped to ``text/plain`` (user-scoped HKCU entries suffice). Starlette's
``StaticFiles``/``FileResponse`` then serve the dashboard bundle as
``text/plain``, browsers reject the ESM module script under strict MIME
checking, and the SPA stays blank.

These tests poison the process-wide ``mimetypes`` map the same way the
Windows registry does, then assert the serving layer pins a JavaScript
``Content-Type`` for ``.js`` assets regardless of the host map.
"""

import mimetypes

import pytest

JAVASCRIPT_MIME_ESSENCES = ("text/javascript", "application/javascript")


def _essence(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _is_javascript(content_type: "str | None") -> bool:
    if not content_type:
        return False
    return _essence(content_type) in JAVASCRIPT_MIME_ESSENCES


@pytest.fixture
def poisoned_mimetypes():
    """Simulate the Windows registry poisoning described in #28987.

    Writes through ``mimetypes.add_type`` so the poison lands in the same
    table ``guess_type`` consults (on Python >= 3.13 that is the lazily
    created ``_db`` instance, not the module-level ``types_map`` copy).
    Restores the exact prior entries on teardown so other tests are unaffected.
    """
    mimetypes.init()
    poisoned = {".js": "text/plain", ".mjs": "text/plain", ".css": "text/plain"}
    saved = {ext: mimetypes.types_map.get(ext) for ext in poisoned}
    for ext, mime in poisoned.items():
        mimetypes.add_type(mime, ext, strict=True)
    # Sanity inside the fixture: the poison really is in effect before the fix runs.
    assert mimetypes.guess_type("a.js")[0] == "text/plain"
    try:
        yield
    finally:
        for ext, mime in saved.items():
            if mime is None:
                mimetypes.types_map.pop(ext, None)
            else:
                mimetypes.add_type(mime, ext, strict=True)


class TestServedJsAssetContentType:
    """End-to-end: a .js asset served through the SPA mount keeps a JS Content-Type."""

    @pytest.fixture
    def spa_client(self, tmp_path, monkeypatch, poisoned_mimetypes):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import hermes_cli.web_server as ws
        import hermes_cli.web_server_dashboard as wsd  # imports pin the MIME map

        dist = tmp_path / "web_dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text(
            "<html><head><title>t</title></head><body>SPA</body></html>",
            encoding="utf-8",
        )
        (dist / "assets" / "index-CqUa8pQa.js").write_text(
            "export const boot = () => {};", encoding="utf-8"
        )
        (dist / "assets" / "worker-Bd0k2mZx.mjs").write_text(
            "export const worker = true;", encoding="utf-8"
        )
        # The registry poison (fixture) predates the dashboard process; mirror
        # that by re-applying the startup normalization NOW, as import of
        # ``web_server_dashboard`` did at process start (and keeps the test
        # meaningful even when an earlier test already imported the module).
        from hermes_cli.web_asset_mime_types import normalize_web_asset_mime_types

        normalize_web_asset_mime_types()
        monkeypatch.setattr(ws, "WEB_DIST", dist)
        monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
        app = FastAPI()
        wsd.mount_spa(app)
        return TestClient(app)

    def test_js_asset_served_as_javascript_despite_host_map(self, spa_client):
        # The host mimetypes table was poisoned to text/plain (Windows registry
        # case) before the serving layer applied its explicit mappings.
        resp = spa_client.get("/assets/index-CqUa8pQa.js")
        assert resp.status_code == 200
        assert _is_javascript(resp.headers["content-type"]), (
            f"expected a JavaScript Content-Type, got {resp.headers['content-type']!r}"
        )

    def test_mjs_asset_served_as_javascript_despite_host_map(self, spa_client):
        resp = spa_client.get("/assets/worker-Bd0k2mZx.mjs")
        assert resp.status_code == 200
        assert _is_javascript(resp.headers["content-type"])


class TestNormalizeWebAssetMimeTypes:
    """Unit: the normalization overrides the poisoned host map."""

    def test_reapplies_over_poisoned_map(self, poisoned_mimetypes):
        from hermes_cli.web_asset_mime_types import normalize_web_asset_mime_types

        assert mimetypes.guess_type("a.js")[0] == "text/plain"
        normalize_web_asset_mime_types()
        assert _is_javascript(mimetypes.guess_type("a.js")[0])
        assert _is_javascript(mimetypes.guess_type("a.mjs")[0])

    def test_idempotent(self):
        from hermes_cli.web_asset_mime_types import normalize_web_asset_mime_types

        normalize_web_asset_mime_types()
        normalize_web_asset_mime_types()
        assert _is_javascript(mimetypes.guess_type("a.js")[0])

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("bundle.js", JAVASCRIPT_MIME_ESSENCES),
            ("worker.mjs", JAVASCRIPT_MIME_ESSENCES),
            ("legacy.cjs", JAVASCRIPT_MIME_ESSENCES),
            ("styles.css", ("text/css",)),
            ("logo.svg", ("image/svg+xml",)),
            ("font.woff2", ("font/woff2",)),
            ("icon.ico", ("image/x-icon",)),
            ("bundle.js.map", ("application/json",)),
            ("module.wasm", ("application/wasm",)),
            ("manifest.webmanifest", ("application/manifest+json",)),
        ],
    )
    def test_pinned_extensions_resolve(self, filename, expected):
        from hermes_cli.web_asset_mime_types import normalize_web_asset_mime_types

        normalize_web_asset_mime_types()
        assert mimetypes.guess_type(filename)[0] in expected

    def test_table_is_complete_for_served_assets(self):
        from hermes_cli.web_asset_mime_types import WEB_ASSET_MIME_TYPES

        # Every extension we pin must be present in the strict map after import
        # of the dashboard module (module-level call already ran).
        import hermes_cli.web_server_dashboard  # noqa: F401

        for ext, mime in WEB_ASSET_MIME_TYPES.items():
            assert mimetypes.guess_type(f"file{ext}")[0] == mime
