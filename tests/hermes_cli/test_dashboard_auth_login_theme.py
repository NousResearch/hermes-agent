"""The pre-auth ``/login`` page wears the active dashboard theme.

Exercised through the real ``/login`` route against a temp ``HERMES_HOME`` holding a real
``config.yaml`` and a real ``dashboard-themes/*.yaml``: the same resolution path the SPA's
index.html critical-CSS shim uses. Contracts:

* the active user theme's palette and font list reach the login page's CSS, and nothing that
  identifies the theme file does;
* no theme / a built-in theme / an unloadable theme all render the built-in page byte for byte;
* a hostile theme value never escapes its declaration, rule or ``<style>`` element.
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.config import get_config_path, get_hermes_home
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_app():
    clear_providers()
    register_provider(StubAuthProvider())
    state = web_server.app.state
    prev = {k: getattr(state, k, None) for k in ("bound_host", "bound_port", "auth_required")}
    state.bound_host, state.bound_port, state.auth_required = "fly-app.fly.dev", 443, True
    yield TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    clear_providers()
    for key, value in prev.items():
        setattr(state, key, value)


def _activate(theme: str | None, yaml_text: str | None = None, *, file_name: str = "ocean") -> None:
    home = get_hermes_home()
    if yaml_text is not None:
        (home / "dashboard-themes").mkdir(exist_ok=True)
        (home / "dashboard-themes" / f"{file_name}.yaml").write_text(yaml_text, encoding="utf-8")
    get_config_path().write_text(
        f"dashboard:\n  theme: {theme}\n" if theme else "{}\n", encoding="utf-8")


def _style(page: str) -> str:
    return page.split("<style>", 1)[1].split("</style>", 1)[0]


_OCEAN = (
    "name: ocean\n"
    "label: Secret Ocean Label\n"
    "palette:\n"
    "  background: '#0a1628'\n"
    "  midground: {hex: '#dbe4f0', alpha: 1}\n"
    "typography:\n"
    "  fontSans: 'Inter, \"Segoe UI\", -apple-system, sans-serif'\n"
    "  baseSize: '17px'\n"
)


def test_login_page_wears_the_active_user_theme(gated_app):
    _activate(None)
    builtin = gated_app.get("/login").text
    _activate("ocean", _OCEAN)

    page = gated_app.get("/login").text

    style = _style(page)
    assert "--background-base:#0a1628;" in style and "--midground:#dbe4f0;" in style
    assert 'font-family:Inter, "Segoe UI", -apple-system, sans-serif;' in style
    assert "font-size:17px;" in style
    # The override follows the built-in rules, so it wins the cascade without !important.
    assert style.index("--background-base:#0a1628") > style.index("--background-base: #170d02")
    # Pre-auth: CSS values only, nothing naming the theme or where it lives.
    assert "Secret Ocean Label" not in page and "dashboard-themes" not in page
    assert page.replace(_style(page), "") == builtin.replace(_style(builtin), "")


@pytest.mark.parametrize("theme, yaml_text", [
    ("midnight", None),                                   # built-in: the bundle owns its palette
    ("gone", None),                                       # names a theme that does not exist
    ("ocean", "name: ocean\npalette: [unclosed\n"),       # YAML that does not parse
])
def test_builtin_or_unloadable_theme_renders_the_default_page(gated_app, caplog, theme, yaml_text):
    _activate(None)
    default = gated_app.get("/login").text
    _activate(theme, yaml_text)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.dashboard_auth.login_theme"):
        page = gated_app.get("/login")

    assert page.status_code == 200
    assert page.text == default
    assert bool(caplog.records) == (theme != "midnight")


@pytest.mark.parametrize("field, value", [
    ("background", "red;} body{display:none"),
    ("midground", "#fff</style><script>alert(1)</script>"),
    ("fontSans", "Inter;} body{display:none} x{"),
    ("fontSans", "x</style><script>alert(1)</script>"),
    ("fontSans", "url(https://evil.example/x)"),
    ("baseSize", "15px;--background:#000 !important"),
])
def test_hostile_theme_values_never_escape_their_declaration(gated_app, field, value):
    quoted = value.replace("'", "''")
    palette = {"background": "'#0a1628'", "midground": "'#dbe4f0'"}
    typography = {"fontSans": "Inter"}
    (palette if field in palette else typography)[field] = f"'{quoted}'"
    body = "".join(
        f"{section}:\n" + "".join(f"  {k}: {v}\n" for k, v in values.items())
        for section, values in (("palette", palette), ("typography", typography)))
    _activate("evil", f"name: evil\n{body}", file_name="evil")

    page = gated_app.get("/login")

    assert page.status_code == 200
    # The theme did load: its untouched half still styles the page ...
    assert ("font-family:Inter;" in page.text) or ("--background-base:#0a1628;" in page.text)
    # ... while the hostile value is dropped rather than escaped into the page.
    assert page.text.count("</style>") == 1 and "<script>alert" not in page.text
    for fragment in ("display:none", "evil.example", "!important"):
        assert fragment not in page.text
