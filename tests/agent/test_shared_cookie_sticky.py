"""Unit + live-server tests for shared-cookie-jar sticky routing
(``providers.<name>.cookie_jar: true``).

Hermes builds a fresh httpx/OpenAI client per request (see
``AIAgent._create_request_openai_client`` and the #10933 closed-transport
invariants). Behind cookie-sticky load balancers (nginx ``sticky cookie``,
Cloudflare in front) each rebuild therefore lands as a brand-new LB session —
prompt-cache locality dies and ``cached_tokens`` collapses. These tests pin:

  * config resolution — ``cookie_jar`` opt-in matches by route, jar identity
    is stable across calls, no opt-in → no jar
  * transport behavior — Set-Cookie extraction on request 1 and Cookie
    replay on request 2 across SEPARATE client builds (the LB invariant),
    plus jar isolation between independent agents
"""
import http.server
import threading
from http.cookiejar import CookieJar

import pytest

from agent.shared_cookie_transport import build_shared_cookie_http_client
from hermes_cli.config_providers import (
    _normalize_custom_provider_entry,
    get_custom_provider_cookie_jar,
)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _entry(base_url, **extra):
    entry = {"name": "test", "base_url": base_url, "key_env": "X_TEST_KEY"}
    entry.update(extra)
    return entry


class TestGetCustomProviderCookieJar:
    def test_opt_in_returns_jar(self):
        url = "https://example.com/v1"
        entries = [_normalize_custom_provider_entry(_entry(url, cookie_jar=True))]
        assert get_custom_provider_cookie_jar(url, entries) is not None

    def test_no_opt_in_returns_none(self):
        url = "https://example.com/v1"
        entries = [_normalize_custom_provider_entry(_entry(url))]
        assert get_custom_provider_cookie_jar(url, entries) is None

    def test_jar_is_shared_across_calls(self):
        url = "https://shared.example.com/v1"
        entries = [_normalize_custom_provider_entry(_entry(url, cookie_jar=True))]
        jar1 = get_custom_provider_cookie_jar(url, entries)
        jar2 = get_custom_provider_cookie_jar(url, entries)
        assert jar1 is not None and jar1 is jar2

    def test_no_match_returns_none(self):
        entries = [_normalize_custom_provider_entry(
            _entry("https://other.com/v1", cookie_jar=True))]
        assert get_custom_provider_cookie_jar("https://nowhere.com/v1", entries) is None

    def test_trailing_slash_insensitive(self):
        entries = [_normalize_custom_provider_entry(
            _entry("https://slashy.com/v1/", cookie_jar=True))]
        assert get_custom_provider_cookie_jar("https://slashy.com/v1", entries) is not None


# ---------------------------------------------------------------------------
# Transport behavior
# ---------------------------------------------------------------------------

class _StickyHandler(http.server.BaseHTTPRequestHandler):
    """Issues ``route=sticky42`` on the first cookieless request; logs Cookie headers."""

    received_cookies = []

    def do_POST(self):
        _StickyHandler.received_cookies.append(self.headers.get("Cookie"))
        body = b'{"ok": true}'
        self.send_response(200)
        if not self.headers.get("Cookie"):
            self.send_header("Set-Cookie", "route=sticky42; Path=/")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def sticky_server():
    _StickyHandler.received_cookies = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _StickyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


class TestSharedCookieTransportReplay:
    def test_cookie_replayed_across_client_rebuilds(self, sticky_server):
        import openai

        jar = CookieJar()

        def make_client():
            client = build_shared_cookie_http_client(jar=jar, limits=None, timeout=5.0)
            return openai.OpenAI(
                api_key="test", base_url=sticky_server, http_client=client,
            )

        oc1 = make_client()
        oc1.post("/chat/completions", body={}, cast_to=object)
        oc1.close()

        # Rebuild: fresh httpx.Client (like _create_request_openai_client),
        # same shared jar.
        oc2 = make_client()
        oc2.post("/chat/completions", body={}, cast_to=object)
        oc2.close()

        assert _StickyHandler.received_cookies[0] is None
        assert "route=sticky42" in (_StickyHandler.received_cookies[1] or "")

    def test_independent_jars_do_not_leak(self, sticky_server):
        import openai

        jar_a, jar_b = CookieJar(), CookieJar()

        def make_client(jar):
            client = build_shared_cookie_http_client(jar=jar, limits=None, timeout=5.0)
            return openai.OpenAI(
                api_key="test", base_url=sticky_server, http_client=client,
            )

        oc1 = make_client(jar_a)
        oc1.post("/chat/completions", body={}, cast_to=object)
        oc1.close()
        oc2 = make_client(jar_b)
        oc2.post("/chat/completions", body={}, cast_to=object)
        oc2.close()

        # Second request went through a jar that never saw the Set-Cookie.
        assert _StickyHandler.received_cookies[1] is None


# ---------------------------------------------------------------------------
# Concurrency (PR review feedback): one lock per JAR, not per transport
# ---------------------------------------------------------------------------

class TestSharedJarLocking:
    def test_bundles_from_registry_share_one_lock(self):
        from agent.shared_cookie_transport import SharedCookieJar
        from hermes_cli.config_providers import _SHARED_COOKIE_JARS

        url = "https://lockcheck.example.com/v1"
        entries = [_normalize_custom_provider_entry(_entry(url, cookie_jar=True))]
        b1 = get_custom_provider_cookie_jar(url, entries)
        b2 = get_custom_provider_cookie_jar(url, entries)
        assert isinstance(b1, SharedCookieJar)
        assert b1 is b2 and b1.lock is b2.lock
        # Cleanup so route registries stay isolated between tests.
        route = url.rstrip("/").lower()
        _SHARED_COOKIE_JARS.pop(route, None)

    def test_plain_jar_wraps_into_bundle_with_lock(self):
        from agent.shared_cookie_transport import SharedCookieJar

        jar = CookieJar()
        bundle1 = SharedCookieJar._from(jar)
        bundle2 = SharedCookieJar._from(jar)
        # Same jar object, independently minted locks (single-client case).
        assert bundle1.jar is jar and bundle2.jar is jar
        assert bundle1.lock is not bundle2.lock

    def test_concurrent_transports_share_lock(self, sticky_server):
        """Two clients built from the same registry bundle serialize on one lock."""
        import httpx
        import openai
        from agent.shared_cookie_transport import SharedCookieJar

        bundle = SharedCookieJar()
        lock_checks = []

        def make_client():
            client = build_shared_cookie_http_client(
                jar=bundle, limits=None, timeout=5.0)
            return openai.OpenAI(
                api_key="test", base_url=sticky_server, http_client=client,
            ), client

        oc1, raw1 = make_client()
        oc2, raw2 = make_client()
        # Both clients' https transports must share the bundle's lock.
        for raw in (raw1, raw2):
            request = raw.build_request("POST", sticky_server + "/chat/completions")
            transport = raw._transport_for_url(request.url)
            lock_checks.append(getattr(transport, "_bundle", None) is bundle)
        oc1.post("/chat/completions", body={}, cast_to=object)
        oc2.post("/chat/completions", body={}, cast_to=object)
        oc1.close()
        oc2.close()
        assert all(lock_checks), f"transports not sharing the bundle lock: {lock_checks}"

    def test_async_mode_returns_async_client(self):
        import httpx

        client = build_shared_cookie_http_client(
            jar=CookieJar(), async_mode=True, limits=None, timeout=5.0)
        assert isinstance(client, httpx.AsyncClient)

    def test_sync_mode_returns_sync_client(self):
        import httpx

        client = build_shared_cookie_http_client(
            jar=CookieJar(), limits=None, timeout=5.0)
        assert isinstance(client, httpx.Client)
