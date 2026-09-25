"""Per-capability web provider picks through the dashboard PUT endpoint.

The runtime reads ``web.search_backend`` / ``web.extract_backend`` as DIRECT
vendor selections ("a stored vendor selection never is" the managed route, see
``tools.web_tools._managed_web_search``). A managed Nous row therefore cannot
be expressed as a per-capability vendor pin: writing the row's servicing
vendor name demoted the managed route to a direct keyless call (observed live:
picking "Nous Subscription" for search wrote ``web.search_backend:
"firecrawl"``, which then failed keyless with HTTP 403). A managed pick must
promote the toolset-level selection (``web.backend: nous``). Search and
extract picks are independent surfaces, so the managed pick clears only the
chosen capability's pin and preserves the other one — pinning it to the
previous shared vendor when it had none, so the promotion does not silently
flip the other capability. Vendor picks keep writing their capability key.
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


def _seed_web(home, web_cfg):
    """Write a ``web`` config shape the endpoint must start from (the live-bug
    regressions start from a NON-empty web section)."""
    import yaml

    path = home / "config.yaml"
    cfg = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    cfg["web"] = {**(cfg.get("web") or {}), **web_cfg}
    path.write_text(yaml.safe_dump(cfg))


def _web_section(home):
    import yaml

    cfg = yaml.safe_load((home / "config.yaml").read_text()) or {}
    return cfg.get("web") or {}


def _pick(client, provider, capability=None):
    body = {"provider": provider}
    if capability is not None:
        body["capability"] = capability
    return client.put("/api/tools/toolsets/web/provider", json=body)


def test_managed_search_pick_clears_the_live_firecrawl_pin(client_and_home):
    """The observed live bug: web.search_backend: firecrawl from an earlier
    pick. A managed search pick must remove that pin and promote the shared
    selection, leaving the runtime on Perplexity Fast."""
    client, home = client_and_home
    _seed_web(home, {"search_backend": "firecrawl", "use_gateway": False})

    resp = _pick(client, "Nous Subscription", capability="search")
    assert resp.status_code == 200
    assert resp.json()["managed"] is True

    web = _web_section(home)
    assert web["backend"] == "nous"
    assert "search_backend" not in web
    assert "use_gateway" not in web

    from tools.web_tools import _get_search_backend

    assert _get_search_backend() == "perplexity"


def test_managed_pick_keeps_the_other_capability_vendor_pin(client_and_home):
    """Search and extract picks are independent: an existing extract vendor pin
    survives a managed search pick (and vice versa)."""
    client, home = client_and_home
    _seed_web(home, {"search_backend": "firecrawl", "extract_backend": "exa"})

    resp = _pick(client, "Nous Subscription", capability="search")
    assert resp.status_code == 200

    web = _web_section(home)
    assert web["backend"] == "nous"
    assert "search_backend" not in web
    assert web["extract_backend"] == "exa"

    from tools.web_tools import _get_extract_backend

    assert _get_extract_backend() == "exa"


def test_managed_pick_pins_the_previous_shared_vendor_for_the_other_capability(client_and_home):
    """When the other capability had no pin of its own but was riding a real
    shared vendor (e.g. web.backend: tavily), the promotion must not silently
    flip that capability onto the managed route."""
    client, home = client_and_home
    _seed_web(home, {"backend": "tavily", "search_backend": "firecrawl"})

    resp = _pick(client, "Nous Subscription", capability="search")
    assert resp.status_code == 200

    web = _web_section(home)
    assert web["backend"] == "nous"
    assert "search_backend" not in web
    assert web["extract_backend"] == "tavily"


def test_managed_extract_pick_clears_the_live_pin_and_keeps_search_vendor(client_and_home):
    """Same promotion in the extract direction: the firecrawl extract pin is
    cleared, a deliberate search vendor pin survives, and the managed extract
    runtime still resolves to Firecrawl (gateway-serviced)."""
    client, home = client_and_home
    _seed_web(home, {"extract_backend": "firecrawl", "search_backend": "perplexity"})

    resp = _pick(client, "Nous Subscription", capability="extract")
    assert resp.status_code == 200
    assert resp.json()["managed"] is True

    web = _web_section(home)
    assert web["backend"] == "nous"
    assert "extract_backend" not in web
    assert web["search_backend"] == "perplexity"

    from tools.web_tools import _get_extract_backend, _get_search_backend

    assert _get_extract_backend() == "firecrawl"
    assert _get_search_backend() == "perplexity"


def test_vendor_capability_pick_still_writes_only_its_capability_key(client_and_home):
    client, home = client_and_home

    resp = _pick(client, "Firecrawl Self-Hosted", capability="search")
    assert resp.status_code == 200
    assert resp.json()["managed"] is False

    web = _web_section(home)
    assert web["search_backend"] == "firecrawl"
    assert "backend" not in web
    assert "extract_backend" not in web
