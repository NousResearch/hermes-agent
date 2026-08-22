"""Off-catalog connector discovery — the trust boundary, not the transport.

The registry client's job is to answer "what exists, and how much should you
trust it" without ever handing the renderer something that would run code on
the user's machine. These tests pin the two rules that make the tier safe:

- package-only entries never surface (installing one is arbitrary code
  execution; connecting to a URL is not)
- ``verified`` means the publisher's registry namespace owns the domain
  serving the endpoint — nothing weaker, and no way to spoof it

Everything hits the parser and the classifier directly; no network.
"""

import pytest

from hermes_cli import mcp_registry as registry


@pytest.fixture(autouse=True)
def _clear_cache():
    registry.clear_cache()
    yield
    registry.clear_cache()


def _server(name, remotes=None, packages=None, status="active", **extra):
    payload = {"server": {"name": name, "description": "", **extra}}
    if remotes is not None:
        payload["server"]["remotes"] = remotes
    if packages is not None:
        payload["server"]["packages"] = packages
    if status:
        payload["_meta"] = {"io.modelcontextprotocol.registry/official": {"status": status}}
    return payload


def _http(url, type_="streamable-http", **extra):
    return {"type": type_, "url": url, **extra}


# ─── Namespace → domain ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "registry_name,domain",
    [
        ("com.notion/mcp", "notion.com"),
        ("app.linear/linear", "linear.app"),
        ("com.paypal.mcp/mcp", "mcp.paypal.com"),
        ("io.github.someone/thing", "someone.github.io"),
        # Not reverse-DNS shaped — nothing is being asserted.
        ("single/thing", ""),
        ("", ""),
    ],
)
def test_namespace_domain(registry_name, domain):
    assert registry.namespace_domain(registry_name) == domain


# ─── Trust classification ────────────────────────────────────────────────────


def test_publisher_serving_its_own_domain_is_verified():
    trust, publisher = registry.classify("com.notion/mcp", "https://mcp.notion.com/mcp")

    assert trust == registry.TRUST_VERIFIED
    assert publisher == "notion.com"


def test_apex_host_matches_its_own_namespace():
    trust, _ = registry.classify("com.paypal.mcp/mcp", "https://mcp.paypal.com/mcp")
    assert trust == registry.TRUST_VERIFIED


@pytest.mark.parametrize(
    "registry_name,url",
    [
        # Suffix matching is on a dot boundary, so a lookalike registrable
        # domain cannot pose as the real one.
        ("com.notion/mcp", "https://mcp.notion.com.evil.test/mcp"),
        ("com.notion/mcp", "https://notnotion.com/mcp"),
        # Namespace and endpoint simply disagree.
        ("io.github.someone/linear-broker", "https://linear-broker.vercel.app/api/mcp"),
        # Malformed authority must never read as verified.
        ("com.notion/mcp", "https://mcp.notion.com:notaport/mcp"),
        # No namespace to verify against.
        ("plain/thing", "https://plain.example/mcp"),
    ],
)
def test_unowned_or_unparseable_endpoints_are_community(registry_name, url):
    trust, _ = registry.classify(registry_name, url)
    assert trust == registry.TRUST_COMMUNITY


@pytest.mark.parametrize(
    "host",
    [
        "abc.trycloudflare.com",
        "my-thing.vercel.app",
        "someone.github.io",
        "proj.pages.dev",
        "api.workers.dev",
    ],
)
def test_shared_subdomain_hosts_can_never_be_verified(host):
    """Anyone can claim a subdomain there, so owning one proves nothing —
    even though the namespace and the endpoint host match perfectly."""
    namespace = ".".join(reversed(host.split(".")))

    trust, _ = registry.classify(f"{namespace}/thing", f"https://{host}/mcp")

    assert trust == registry.TRUST_COMMUNITY
    assert registry.is_shared_subdomain(host)


# ─── Entry parsing ───────────────────────────────────────────────────────────


def test_package_only_entries_never_surface(monkeypatch):
    """The code-execution class is filtered in the backend, so a renderer
    cannot ask for one even by crafting the request."""
    _stub_search(
        monkeypatch,
        [
            _server("io.github.someone/local-tool", packages=[{"registryType": "npm"}]),
            _server("com.notion/mcp", remotes=[_http("https://mcp.notion.com/mcp")]),
        ],
    )

    results = registry.search("thing")

    assert [entry.name for entry in results] == ["notion"]


def test_plaintext_http_endpoints_are_dropped(monkeypatch):
    _stub_search(monkeypatch, [_server("com.example/thing", remotes=[_http("http://example.com/mcp")])])

    assert registry.search("thing") == []


def test_deleted_entries_are_dropped(monkeypatch):
    _stub_search(
        monkeypatch,
        [_server("com.notion/mcp", remotes=[_http("https://mcp.notion.com/mcp")], status="deleted")],
    )

    assert registry.search("notion") == []


def test_streamable_http_wins_over_sse(monkeypatch):
    _stub_search(
        monkeypatch,
        [
            _server(
                "com.notion/mcp",
                remotes=[
                    _http("https://mcp.notion.com/sse", "sse"),
                    _http("https://mcp.notion.com/mcp"),
                ],
            )
        ],
    )

    entry = registry.search("notion")[0]

    assert entry.transport == "streamable-http"
    assert entry.url.endswith("/mcp")


@pytest.mark.parametrize(
    "registry_name,expected",
    [
        # A generic base falls back to the most specific real namespace label.
        ("com.notion/mcp", "notion"),
        ("com.paypal.mcp/mcp", "paypal"),
        ("com.cloudflare.mcp/server", "cloudflare"),
        # A distinctive base is kept as-is.
        ("app.linear/linear", "linear"),
        ("ai.smithery/smithery-notion", "smithery-notion"),
    ],
)
def test_config_name_prefers_the_vendor_over_the_protocol(monkeypatch, registry_name, expected):
    _stub_search(monkeypatch, [_server(registry_name, remotes=[_http("https://host.example/mcp")])])

    assert registry.search("thing")[0].name == expected


def test_header_specs_carry_names_and_secrecy_never_values(monkeypatch):
    _stub_search(
        monkeypatch,
        [
            _server(
                "com.example/thing",
                remotes=[
                    _http(
                        "https://mcp.example.com/mcp",
                        headers=[{"name": "X-Api-Key", "isRequired": True, "isSecret": True}],
                    )
                ],
            )
        ],
    )

    header = registry.search("thing")[0].headers[0]

    assert header == {"name": "X-Api-Key", "description": "", "required": True, "secret": True}


# ─── Ranking and resilience ──────────────────────────────────────────────────


def test_verified_outranks_community_and_exact_name_outranks_the_rest(monkeypatch):
    _stub_search(
        monkeypatch,
        [
            _server("io.github.a/notion-extra", remotes=[_http("https://a.example.com/mcp")]),
            _server("com.notionish/notion-helper", remotes=[_http("https://mcp.notionish.com/mcp")]),
            _server("com.notion/mcp", remotes=[_http("https://mcp.notion.com/mcp")]),
        ],
    )

    results = registry.search("notion")

    assert results[0].name == "notion"
    assert results[0].trust == registry.TRUST_VERIFIED
    assert results[-1].trust == registry.TRUST_COMMUNITY


def test_allow_unverified_false_hard_limits_to_verified(monkeypatch):
    monkeypatch.setattr(
        registry,
        "registry_settings",
        lambda: {**_settings(), "allow_unverified": False},
    )
    _stub_search(
        monkeypatch,
        [
            _server("io.github.a/thing", remotes=[_http("https://a.example.com/mcp")]),
            _server("com.notion/mcp", remotes=[_http("https://mcp.notion.com/mcp")]),
        ],
        patch_settings=False,
    )

    assert [entry.trust for entry in registry.search("thing")] == [registry.TRUST_VERIFIED]


def test_disabled_registry_returns_nothing(monkeypatch):
    monkeypatch.setattr(registry, "registry_settings", lambda: {**_settings(), "enabled": False})

    assert registry.search("notion") == []


def test_short_queries_do_not_hit_the_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("should not have queried the registry")

    monkeypatch.setattr(registry, "registry_settings", lambda: _settings())
    monkeypatch.setitem(__import__("sys").modules, "httpx", _FailingHttpx(explode))

    assert registry.search("n") == []


def test_unreachable_registry_degrades_to_empty_never_raises(monkeypatch):
    monkeypatch.setattr(registry, "registry_settings", lambda: _settings())
    monkeypatch.setitem(
        __import__("sys").modules,
        "httpx",
        _FailingHttpx(lambda *a, **k: (_ for _ in ()).throw(OSError("no route to host"))),
    )

    assert registry.search("notion") == []


def test_results_are_cached_within_the_ttl(monkeypatch):
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return _Response({"servers": [_server("com.notion/mcp", remotes=[_http("https://mcp.notion.com/mcp")])]})

    monkeypatch.setattr(registry, "registry_settings", lambda: _settings())
    monkeypatch.setitem(__import__("sys").modules, "httpx", _FailingHttpx(counted))

    assert registry.search("notion")[0].name == "notion"
    assert registry.search("notion")[0].name == "notion"
    assert len(calls) == 1


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _settings():
    return {
        "enabled": True,
        "url": registry.DEFAULT_REGISTRY_URL,
        "timeout_seconds": 1,
        "cache_ttl_minutes": 30,
        "allow_unverified": True,
    }


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FailingHttpx:
    """Minimal httpx stand-in: the module only ever calls ``get``."""

    def __init__(self, get):
        self.get = get


def _stub_search(monkeypatch, servers, patch_settings=True):
    if patch_settings:
        monkeypatch.setattr(registry, "registry_settings", lambda: _settings())

    monkeypatch.setitem(
        __import__("sys").modules,
        "httpx",
        _FailingHttpx(lambda *a, **k: _Response({"servers": servers})),
    )
