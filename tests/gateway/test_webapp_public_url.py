"""The Webapp's own public URL, ``webapp.public_url``, and never ``dashboard.public_url``.

A user who reaches the gateway's Webapp through a reverse proxy or tunnel sets ``webapp.public_url``.
On the Webapp listener it engages sign-in even on a loopback bind, it is the one public Host the
listener accepts, and sign-in and MCP OAuth callbacks are built on it. ``dashboard.public_url``
describes ``hermes dashboard``: the gateway's Webapp ignores it, and the standalone dashboard keeps
reading it as before. The live cases run a real gateway listener in a child process
(``fixtures/webapp_fallback_peer.py``) with its own ``HERMES_HOME``.
"""
from types import SimpleNamespace

import pytest

from gateway import webapp_listener as webapp
from tests.gateway.fixtures.webapp_fallback_peer import free_port, run

WEBAPP_URL = "https://webapp.example.net"
DASHBOARD_URL = "https://dash.example.net"


def _home(tmp_path, config):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


def test_the_webapp_has_no_public_url_unless_configured(tmp_path):
    assert webapp.webapp_settings(_home(tmp_path, "webapp:\n  enabled: true\n")).public_url == ""


@pytest.mark.parametrize(("value", "url"), [
    ("https://webapp.example.net", "https://webapp.example.net"),
    ("https://webapp.example.net/", "https://webapp.example.net"),
    ("https://webapp.example.net:8443/hermes/", "https://webapp.example.net:8443/hermes"),
    ("http://hermes.lan:9120", "http://hermes.lan:9120"),
    ("''", ""),
])
def test_the_webapp_reads_its_public_url(tmp_path, value, url):
    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  public_url: {value}\n"))
    assert (settings.enabled, settings.public_url) == (True, url)


@pytest.mark.parametrize("value", [
    "webapp.example.net",  # no scheme
    "ftp://webapp.example.net",
    "https://",
    "7",
    "https://user:secret@webapp.example.net",
    "https://webapp.example.net/?next=x",
    "'https://webapp.example.net/#top'",
    "'https://webapp.example.net/<script>'",
])
def test_a_malformed_webapp_public_url_keeps_the_webapp_off(tmp_path, value):
    settings = webapp.webapp_settings(_home(tmp_path, f"webapp:\n  enabled: true\n  public_url: {value}\n"))
    assert (settings.enabled, settings.bind) == (False, ("127.0.0.1", 0))


def test_a_listener_that_declares_its_public_url_never_reads_the_dashboards(monkeypatch):
    from hermes_cli.dashboard_auth.prefix import resolve_public_url

    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", DASHBOARD_URL)
    assert resolve_public_url(SimpleNamespace(listener_public_url=WEBAPP_URL)) == WEBAPP_URL
    assert resolve_public_url(SimpleNamespace(listener_public_url="")) == ""
    # The standalone dashboard declares nothing and reads dashboard.public_url as before.
    assert resolve_public_url(SimpleNamespace()) == DASHBOARD_URL
    assert resolve_public_url() == DASHBOARD_URL


def _request(state, headers):
    from starlette.requests import Request

    app = SimpleNamespace(state=state)
    return Request({"type": "http", "method": "POST", "path": "/api/config", "raw_path": b"/api/config",
                    "query_string": b"", "scheme": "http", "server": ("127.0.0.1", 9120), "root_path": "",
                    "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()], "app": app})


def test_the_webapps_mcp_oauth_callback_is_built_on_its_own_public_url(monkeypatch):
    from hermes_cli.web_routers.mcp import _mcp_oauth_callback_url

    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", DASHBOARD_URL)
    state = SimpleNamespace(listener_public_url=WEBAPP_URL)
    request = _request(state, {"host": "webapp.example.net"})
    assert _mcp_oauth_callback_url(request, "srv") == f"{WEBAPP_URL}/api/mcp/oauth/callback/srv"


def test_a_browser_without_fetch_metadata_writes_from_the_webapps_public_origin(monkeypatch):
    # Without Sec-Fetch-Site the Origin is compared with this listener's own public URL.
    from hermes_cli.dashboard_auth.request_utils import cookie_origin_is_allowed

    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", DASHBOARD_URL)
    state = SimpleNamespace(listener_public_url=WEBAPP_URL, bound_host="127.0.0.1",
                            trusted_public_hosts=frozenset({"webapp.example.net"}))
    own = _request(state, {"host": "webapp.example.net", "origin": WEBAPP_URL})
    dashboard = _request(state, {"host": "webapp.example.net", "origin": DASHBOARD_URL})
    assert cookie_origin_is_allowed(own) is True
    assert cookie_origin_is_allowed(dashboard) is False


@pytest.fixture(scope="module")
def own_url(tmp_path_factory):
    # The "my own URL" shape: a loopback Webapp behind a proxy at webapp.public_url, with sign-in.
    spec = {"webapp": {"enabled": True, "port": free_port(), "public_url": WEBAPP_URL}, "basic_auth": True,
            "dashboard_public_url": DASHBOARD_URL, "probe_host": "webapp.example.net"}
    return run(tmp_path_factory.mktemp("own-url"), spec)


def test_the_webapps_public_url_engages_sign_in_on_a_loopback_bind(own_url):
    assert own_url["start_error"] is None, own_url.get("start_traceback")
    assert own_url["status"]["state"] == "serving"
    assert own_url["state"]["auth_required"] is True
    assert own_url["launch"]["sign_in"] is True


def test_hermes_webapp_opens_the_webapps_public_url_to_sign_in(own_url):
    # The sign-in callback is on the public URL, so the browser starts there too; native clients
    # keep the loopback address.
    assert own_url["launch"]["url"] == f"{WEBAPP_URL}/"
    assert own_url["api_origin"].startswith("http://127.0.0.1:")


def test_the_webapp_trusts_only_its_own_public_host(own_url):
    assert own_url["state"]["trusted_public_hosts"] == ["webapp.example.net"]
    assert own_url["probe_host_status_code"] == 200


def test_sign_in_callbacks_are_built_on_the_webapps_public_url(own_url):
    assert own_url["redirect_uri_for_probe_host"] == f"{WEBAPP_URL}/auth/callback"


@pytest.fixture(scope="module")
def dashboard_url_only(tmp_path_factory):
    # dashboard.public_url set for `hermes dashboard`, no sign-in provider, a loopback Webapp.
    spec = {"webapp": {"enabled": True, "port": free_port()}, "dashboard_public_url": DASHBOARD_URL,
            "probe_host": "dash.example.net"}
    return run(tmp_path_factory.mktemp("dashboard-url"), spec)


def test_the_dashboards_public_url_does_not_gate_the_webapp(dashboard_url_only):
    assert dashboard_url_only["start_error"] is None, dashboard_url_only.get("start_traceback")
    assert dashboard_url_only["status"]["state"] == "serving"
    assert dashboard_url_only["state"]["auth_required"] is False
    assert dashboard_url_only["launch"].get("sign_in") is None


def test_the_webapp_refuses_the_dashboards_public_host(dashboard_url_only):
    assert dashboard_url_only["state"]["trusted_public_hosts"] == []
    assert dashboard_url_only["probe_host_status_code"] == 400


def test_a_webapp_public_url_without_sign_in_leaves_native_clients_a_listener(tmp_path):
    spec = {"webapp": {"enabled": True, "port": free_port(), "public_url": WEBAPP_URL}}
    report = run(tmp_path, spec)
    assert report["start_error"] is None, report.get("start_traceback")
    assert report["native_session_list"] is True
    assert report["status"]["state"] == "unavailable"
    assert "webapp.public_url" in report["status"]["reason"]
    assert "dashboard.public_url" not in report["status"]["reason"]

