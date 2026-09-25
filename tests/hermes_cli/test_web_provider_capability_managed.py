"""Per-capability web provider picks through the dashboard PUT endpoint.

The runtime reads ``web.search_backend`` / ``web.extract_backend`` as DIRECT
vendor selections ("a stored vendor selection never is" the managed route, see
``tools.web_tools._managed_web_search``). A managed Nous row therefore cannot
be expressed as a per-capability vendor pin: writing the row's servicing
vendor name demoted the managed route to a direct keyless call (observed live:
picking "Nous Subscription" for search wrote ``web.search_backend:
"firecrawl"``, which then failed keyless with HTTP 403). A managed pick must
promote the toolset-level selection (``web.backend: nous``) and clear both
per-capability overrides; vendor picks keep writing their capability key.
"""

import pytest


@pytest.fixture()
def client_and_home(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    # The endpoint's post-save entitlement check hits the Portal; the selection
    # itself must not depend on network state.
    import hermes_cli.nous_subscription as nous_subscription

    features = nous_subscription.get_nous_subscription_features({}, force_fresh=True)
    monkeypatch.setattr(
        nous_subscription, "get_nous_subscription_features", lambda *a, **kw: features
    )
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, get_hermes_home()


def _pick(client, provider, capability=None):
    body = {"provider": provider}
    if capability is not None:
        body["capability"] = capability
    return client.put("/api/tools/toolsets/web/provider", json=body)


def test_managed_capability_pick_promotes_the_shared_selection(client_and_home):
    client, home = client_and_home

    resp = _pick(client, "Nous Subscription", capability="search")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["managed"] is True

    import yaml

    web = (yaml.safe_load((home / "config.yaml").read_text()) or {}).get("web", {})
    assert web["backend"] == "nous"
    # Both per-capability overrides are gone: either one would be read as a
    # direct vendor selection and bypass the managed route.
    assert "search_backend" not in web
    assert "extract_backend" not in web

    from tools.web_tools import _get_search_backend

    assert _get_search_backend() == "perplexity"


def test_vendor_capability_pick_still_writes_only_its_capability_key(client_and_home):
    client, home = client_and_home

    resp = _pick(client, "Firecrawl Self-Hosted", capability="search")
    assert resp.status_code == 200
    assert resp.json()["managed"] is False

    import yaml

    web = (yaml.safe_load((home / "config.yaml").read_text()) or {}).get("web", {})
    assert web["search_backend"] == "firecrawl"
    assert "backend" not in web
    assert "extract_backend" not in web
