"""The dashboard's language-pack routes: ``/api/i18n/languages`` and ``/api/i18n/catalog``.

They are the HTTP twin of the ``i18n.languages`` / ``i18n.catalog`` JSON-RPC pair Desktop uses: a pack
or a user overlay that ships ``<lang>.web.yaml`` reaches the web dashboard without rebuilding it, and a
language the dashboard never bundled (``vi``) becomes selectable. Both routes sit behind the
session-token gate like every ``/api/`` route.
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest
from fastapi.testclient import TestClient

from agent import i18n, i18n_layers
from hermes_cli import web_server


@pytest.fixture
def home(tmp_path, monkeypatch):
    i18n_layers._reset_registry_for_tests()
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Discovery is the real loader's job; here packs are registered directly.
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)
    i18n.reset_language_cache()
    yield home
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


def _client(token: bool = True) -> TestClient:
    client = TestClient(web_server.app, raise_server_exceptions=False)
    if token:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def test_web_is_a_language_pack_surface():
    assert "web" in i18n_layers.SURFACES


def test_a_pack_language_with_web_strings_is_listed_with_its_web_surface(home):
    i18n_layers.register_pack("vi", "web", {"common.save": "Lưu"}, source="plugin:hermes-lang-vi",
                              endonym="Tiếng Việt")
    i18n.reset_language_cache()

    rows = _client().get("/api/i18n/languages").json()["languages"]
    vi = next(row for row in rows if row["id"] == "vi")

    assert vi["endonym"] == "Tiếng Việt"
    assert "web" in vi["surfaces"]
    assert rows[0]["id"] == "en"


def test_the_catalog_serves_the_web_layer_pack_over_overlay(home):
    (home / "locales" / "vi.web.yaml").write_text(
        yaml.safe_dump({"common.save": "Lưu (overlay)", "common.cancel": "Huỷ"}, allow_unicode=True),
        encoding="utf-8")
    i18n_layers.register_pack("vi", "web", {"common.save": "Lưu"}, source="plugin:hermes-lang-vi")
    i18n.reset_language_cache()

    body = _client().get("/api/i18n/catalog", params={"lang": "VI"}).json()

    assert body["lang"] == "vi" and body["surface"] == "web"
    assert body["messages"]["common.save"] == "Lưu"         # plugin pack wins over the overlay
    assert body["messages"]["common.cancel"] == "Huỷ"       # partial layers merge key by key


def test_other_surfaces_do_not_leak_into_the_web_catalog(home):
    i18n_layers.register_pack("vi", "desktop", {"common.save": "Lưu (desktop)"}, source="plugin:x")
    i18n.reset_language_cache()

    assert _client().get("/api/i18n/catalog", params={"lang": "vi"}).json()["messages"] == {}


def test_an_unknown_surface_is_a_400(home):
    assert _client().get("/api/i18n/catalog", params={"lang": "vi", "surface": "webz"}).status_code == 400


def test_an_unknown_language_resolves_to_english_with_an_empty_layer(home):
    body = _client().get("/api/i18n/catalog", params={"lang": "xx-nope"}).json()

    assert body == {"lang": "en", "surface": "web", "messages": {}}


@pytest.mark.parametrize("path", ["/api/i18n/languages", "/api/i18n/catalog?lang=vi"])
def test_the_routes_sit_behind_the_session_token_gate(home, path):
    assert _client(token=False).get(path).status_code == 401
