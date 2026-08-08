"""Regression tests for the shard-s5 wave-1 extraction: spa_mount_mixin.

The SPA mount + theme-bootstrap helpers were moved verbatim from
``hermes_cli/web_server.py`` into ``hermes_cli/spa_mount_mixin.py``
(god-file decomposition, cluster c17).  ``web_server`` re-exports them so
``web_server.mount_spa`` / ``web_server._render_active_theme_bootstrap_css``
call sites and tests keep working.

``mount_spa`` resolves ``WEB_DIST``/``app``/``_SESSION_TOKEN`` lazily from
``web_server`` at call time, so tests that monkeypatch
``web_server.WEB_DIST`` / ``web_server.load_config`` (as the existing
``tests/hermes_cli/test_web_server.py`` does) keep working unchanged.
"""

import pytest

from hermes_cli import web_server
from hermes_cli import spa_mount_mixin as mixin


class TestReExportSeam:
    @pytest.mark.parametrize(
        "name", ["mount_spa", "_normalise_prefix", "_render_active_theme_bootstrap_css"]
    )
    def test_reexported_identity(self, name):
        assert getattr(web_server, name) is getattr(mixin, name)


class TestThemeBootstrapCssMixin:
    def test_builtin_theme_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            web_server, "load_config", lambda: {"dashboard": {"theme": "default"}}
        )
        assert mixin._render_active_theme_bootstrap_css() == ""

    def test_user_theme_renders_bundle_vars(self, monkeypatch):
        monkeypatch.setattr(
            web_server, "load_config", lambda: {"dashboard": {"theme": "ocean"}}
        )
        monkeypatch.setattr(
            web_server,
            "_discover_user_themes",
            lambda: [
                {
                    "name": "ocean",
                    "palette": {
                        "background": {"hex": "#0a1628"},
                        "midground": {"hex": "#dbe4f0"},
                    },
                    "typography": {"fontSans": "Inter, sans-serif", "baseSize": "17px"},
                }
            ],
        )
        css = mixin._render_active_theme_bootstrap_css()
        assert css.startswith('<style id="hermes-theme-bootstrap">')
        assert "--background-base:#0a1628;" in css
        assert "--midground-base:#dbe4f0;" in css
        assert "--theme-font-sans:Inter, sans-serif;" in css
        assert "--theme-base-size:17px;" in css

    def test_invalid_theme_returns_empty(self, monkeypatch):
        monkeypatch.setattr(web_server, "load_config", lambda: {"dashboard": {}})
        assert mixin._render_active_theme_bootstrap_css() == ""


class TestNormalisePrefix:
    def test_none_and_empty(self):
        assert mixin._normalise_prefix(None) == ""
        assert mixin._normalise_prefix("") == ""

    def test_leading_slash_added(self):
        assert mixin._normalise_prefix("hermes") == "/hermes"

    def test_trailing_slash_stripped(self):
        assert mixin._normalise_prefix("/hermes/") == "/hermes"


class TestMountSpaMixin:
    @staticmethod
    def _mount_client(tmp_path, monkeypatch):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
        dist = tmp_path / "web_dist"
        (dist / "assets").mkdir(parents=True, exist_ok=True)
        (dist / "index.html").write_text(
            "<html><head><title>t</title></head><body>SPA</body></html>",
            encoding="utf-8",
        )
        monkeypatch.setattr(web_server, "WEB_DIST", dist)
        spa_app = FastAPI()
        mixin.mount_spa(spa_app)
        return TestClient(spa_app)

    def test_serves_index_with_token_injected(self, tmp_path, monkeypatch):
        prev = getattr(web_server.app.state, "auth_required", None)
        try:
            web_server.app.state.auth_required = False
            client = self._mount_client(tmp_path, monkeypatch)
            resp = client.get("/")
            assert resp.status_code == 200
            assert "SPA" in resp.text
            assert "__HERMES_SESSION_TOKEN__" in resp.text
        finally:
            web_server.app.state.auth_required = prev

    def test_api_miss_returns_json_404(self, tmp_path, monkeypatch):
        client = self._mount_client(tmp_path, monkeypatch)
        resp = client.get("/api/definitely-not-a-route")
        assert resp.status_code == 404
        assert resp.headers["content-type"].startswith("application/json")

    def test_css_served_with_immutable_cache(self, tmp_path, monkeypatch):
        (tmp_path / "web_dist" / "assets").mkdir(parents=True)
        (tmp_path / "web_dist" / "assets" / "app.css").write_text(
            "body{color:red}", encoding="utf-8"
        )
        client = self._mount_client(tmp_path, monkeypatch)
        resp = client.get("/assets/app.css")
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"
