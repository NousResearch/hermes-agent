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
previous shared vendor when it had none, but only when that vendor can serve
the capability (a search-only vendor such as brave-free stays unpinned and
falls through the registry ladder). A toolset-level managed pick (no
``capability``) governs the whole toolset and clears BOTH pins — that is the
repair path for configs corrupted by the pre-fix write. Vendor picks keep
writing their capability key.
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
    """Write a ``web`` config shape through the real config writer (the live-bug
    regressions start from a NON-empty web section)."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["web"] = {**(cfg.get("web") or {}), **web_cfg}
    save_config(cfg)


def _web_section(home):
    from hermes_cli.config import load_config

    return load_config().get("web") or {}


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
    assert not web.get("search_backend")
    assert not web.get("use_gateway")

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
    assert not web.get("search_backend")
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
    assert not web.get("search_backend")
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
    assert not web.get("extract_backend")
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
    assert not web.get("backend")
    assert not web.get("extract_backend")


def test_toolset_level_managed_pick_clears_stale_pins(client_and_home):
    """The corrupted shape this PR exists to repair, healed through the
    toolset-level pick: a managed selection without ``capability`` routes
    through the shared ``_write_provider_config`` writer, which must clear BOTH
    per-capability pins — the dispatchers resolve a pin FIRST and read it as a
    DIRECT vendor selection, so a stale pin keeps outranking the ``nous``
    selection just written and the user never reaches the managed route."""
    client, home = client_and_home
    _seed_web(home, {"search_backend": "firecrawl", "extract_backend": "firecrawl"})

    resp = _pick(client, "Nous Subscription")
    assert resp.status_code == 200

    web = _web_section(home)
    assert web["backend"] == "nous"
    assert not web.get("search_backend")
    assert not web.get("extract_backend")

    from tools.web_tools import _get_search_backend, _managed_web_search

    assert _managed_web_search() is True
    assert _get_search_backend() == "perplexity"


def test_managed_pick_does_not_pin_a_search_only_vendor_for_extract(client_and_home):
    """The sibling-preservation pin must respect capabilities: brave-free is
    search-only, so a managed search pick riding ``web.backend: brave-free``
    must NOT write ``extract_backend: brave-free`` — the GUI badge would name a
    vendor the registry falls through. Leaving it unpinned lets extract resolve
    through the managed route (gateway-serviced Firecrawl) honestly."""
    client, home = client_and_home
    _seed_web(home, {"backend": "brave-free"})

    resp = _pick(client, "Nous Subscription", capability="search")
    assert resp.status_code == 200

    web = _web_section(home)
    assert web["backend"] == "nous"
    assert not web.get("extract_backend")

    from tools.web_tools import _get_extract_backend

    assert _get_extract_backend() == "firecrawl"


def test_alternating_managed_picks_settle_both_capabilities_on_the_managed_route(client_and_home):
    """A capability pin is a PRESERVATION mechanism for a capability the user
    has not explicitly picked — not a preference that survives an explicit
    managed pick of that capability. Walk both directions: pick 1 preserves
    extract on the shared vendor; pick 2 is an explicit managed extract pick,
    so extract joins search on the managed route and nothing resurrects a pin."""
    client, home = client_and_home
    _seed_web(home, {"backend": "tavily", "search_backend": "firecrawl"})

    resp = _pick(client, "Nous Subscription", capability="search")
    assert resp.status_code == 200
    web = _web_section(home)
    assert web["extract_backend"] == "tavily"

    resp = _pick(client, "Nous Subscription", capability="extract")
    assert resp.status_code == 200
    web = _web_section(home)
    assert web["backend"] == "nous"
    assert not web.get("extract_backend")
    assert not web.get("search_backend")

    from tools.web_tools import _get_extract_backend, _get_search_backend

    assert _get_search_backend() == "perplexity"
    assert _get_extract_backend() == "firecrawl"
