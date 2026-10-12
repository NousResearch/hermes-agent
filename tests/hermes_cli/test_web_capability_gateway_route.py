"""Each web capability picks its own route: the user's own key or the Nous Tool Gateway.

"Use for Extract" on the Nous Subscription row must send extract through the gateway while search
keeps the user's own Firecrawl key, and the toolset-level Nous pick must take both back. Before the
fix the managed pick wrote "firecrawl", which reads as the BYOK key, so the gateway was unreachable
per capability (#100513).
"""

import types

import pytest


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    starlette = pytest.importorskip("starlette.testclient")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    test_client = starlette.TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_web_capability_picks_choose_own_key_or_nous_gateway(client, monkeypatch):
    import plugins.web.firecrawl.provider as fc
    import tools.web_tools as wt
    from hermes_cli.config import load_config

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-own-key")
    monkeypatch.setattr(fc._gateway, "resolve_managed_tool_gateway", lambda *a, **k: types.SimpleNamespace(
        nous_user_token="nous-tok", gateway_origin="https://firecrawl-gateway.example"))
    monkeypatch.setattr(fc, "Firecrawl", lambda **kw: types.SimpleNamespace(kwargs=kw), raising=False)
    monkeypatch.setattr(wt, "_firecrawl_client", None, raising=False)

    def put(**body):
        assert client.put("/api/tools/toolsets/web/provider", json=body).status_code == 200
        return client.get("/api/tools/toolsets/web/config").json()

    put(provider="Firecrawl", capability="search")
    data = put(provider="Nous Subscription", capability="extract")
    assert (data["search_via_nous"], data["extract_via_nous"]) == (False, True)
    assert fc._get_firecrawl_client("search").kwargs["api_key"] == "fc-own-key"
    assert fc._get_firecrawl_client("extract").kwargs["api_key"] == "nous-tok"

    data = put(provider="Nous Subscription")
    web = load_config()["web"]
    assert not web.get("search_backend") and not web.get("extract_backend")
    assert (data["search_via_nous"], data["extract_via_nous"]) == (True, True)
