"""Tests for plugins.platforms.telegram.telegram_network – fallback transport layer.

Background
----------
api.telegram.org resolves to an IP (e.g. 149.154.166.110) that is unreachable
from some networks.  The workaround: route TCP through a different IP in the
same Telegram-owned 149.154.160.0/20 block (e.g. 149.154.167.220) while
keeping TLS SNI and the Host header as api.telegram.org so Telegram's edge
servers still accept the request.  This is the programmatic equivalent of:

    curl --resolve api.telegram.org:443:149.154.167.220 https://api.telegram.org/bot<token>/getMe

The TelegramFallbackTransport implements this: try known IPv4 Telegram API
IPs first (so a blackholed IPv6 AAAA for the hostname cannot hang
initialize — #87015), then fall through to the dual-stack hostname last,
and "stick" to whichever path works.
"""

import httpx
import pytest

import plugins.platforms.telegram.telegram_network as tnet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTransport(httpx.AsyncBaseTransport):
    """Records calls and raises / returns based on a host→action mapping."""

    def __init__(self, calls, behavior):
        self.calls = calls
        self.behavior = behavior
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            {
                "url_host": request.url.host,
                "host_header": request.headers.get("host"),
                "sni_hostname": request.extensions.get("sni_hostname"),
                "path": request.url.path,
            }
        )
        action = self.behavior.get(request.url.host, "ok")
        if action == "timeout":
            raise httpx.ConnectTimeout("timed out")
        if action == "connect_error":
            raise httpx.ConnectError("connect error")
        if isinstance(action, Exception):
            raise action
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        self.closed = True


def _fake_transport_factory(calls, behavior):
    """Returns a factory that creates FakeTransport instances."""
    instances = []

    def factory(**kwargs):
        t = FakeTransport(calls, behavior)
        instances.append(t)
        return t

    factory.instances = instances
    return factory


def _telegram_request(path="/botTOKEN/getMe"):
    return httpx.Request("GET", f"https://api.telegram.org{path}")


# ═══════════════════════════════════════════════════════════════════════════
# IP parsing & validation
# ═══════════════════════════════════════════════════════════════════════════

class TestParseFallbackIpEnv:
    def test_filters_invalid_and_ipv6(self, caplog):
        ips = tnet.parse_fallback_ip_env("149.154.167.220, bad, 2001:67c:4e8:f004::9,149.154.167.220")
        assert ips == ["149.154.167.220", "149.154.167.220"]
        assert "Ignoring invalid Telegram fallback IP" in caplog.text
        assert "Ignoring non-IPv4 Telegram fallback IP" in caplog.text

    def test_none_returns_empty(self):
        assert tnet.parse_fallback_ip_env(None) == []


class TestNormalizeFallbackIps:
    def test_deduplication_happens_at_transport_level(self):
        """_normalize does not dedup; TelegramFallbackTransport.__init__ does."""
        raw = ["149.154.167.220", "149.154.167.220"]
        assert tnet._normalize_fallback_ips(raw) == ["149.154.167.220", "149.154.167.220"]


# ═══════════════════════════════════════════════════════════════════════════
# Request rewriting
# ═══════════════════════════════════════════════════════════════════════════

class TestRewriteRequestForIp:
    def test_preserves_host_and_sni(self):
        request = _telegram_request()
        rewritten = tnet._rewrite_request_for_ip(request, "149.154.167.220")

        assert rewritten.url.host == "149.154.167.220"
        assert rewritten.headers["host"] == "api.telegram.org"
        assert rewritten.extensions["sni_hostname"] == "api.telegram.org"
        assert rewritten.url.path == "/botTOKEN/getMe"


# ═══════════════════════════════════════════════════════════════════════════
# Fallback transport – core behavior
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackTransport:
    """IPv4 literals first → hostname last → stick to whichever works."""

    @pytest.mark.asyncio
    async def test_ipv4_literal_tried_before_hostname_and_becomes_sticky(self, monkeypatch):
        calls = []
        behavior = {"api.telegram.org": "timeout", "149.154.167.220": "ok"}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
        resp = await transport.handle_async_request(_telegram_request())

        assert resp.status_code == 200
        assert transport._sticky_ip == "149.154.167.220"
        assert [c["url_host"] for c in calls] == ["149.154.167.220"]
        assert calls[0]["host_header"] == "api.telegram.org"
        assert calls[0]["sni_hostname"] == "api.telegram.org"

        # Second request goes straight to sticky IP
        calls.clear()
        resp2 = await transport.handle_async_request(_telegram_request())
        assert resp2.status_code == 200
        assert calls[0]["url_host"] == "149.154.167.220"


    @pytest.mark.asyncio
    async def test_sticky_ip_tried_first_but_falls_through_if_stale(self, monkeypatch):
        """If the sticky IP stops working, the transport retries others."""
        calls = []
        behavior = {
            "api.telegram.org": "timeout",
            "149.154.167.220": "ok",
            "149.154.167.221": "ok",
        }
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220", "149.154.167.221"])

        # First request: .220 works immediately (IPv4-first) → becomes sticky
        await transport.handle_async_request(_telegram_request())
        assert transport._sticky_ip == "149.154.167.220"

        # Now .220 goes bad too
        calls.clear()
        behavior["149.154.167.220"] = "timeout"

        resp = await transport.handle_async_request(_telegram_request())
        assert resp.status_code == 200
        # Sticky .220 fails → remaining IPv4 .221 works. Hostname is last
        # and is never needed.
        assert [c["url_host"] for c in calls] == ["149.154.167.220", "149.154.167.221"]
        assert transport._sticky_ip == "149.154.167.221"

    @pytest.mark.asyncio
    async def test_blackholed_hostname_never_reached_when_ipv4_works(self, monkeypatch):
        """#87015: a hostname connect that never returns must not run first."""
        calls = []

        class _HangOnHostname(FakeTransport):
            async def handle_async_request(self, request):
                if request.url.host == "api.telegram.org":
                    raise AssertionError(
                        "dual-stack hostname was tried before a working IPv4 literal"
                    )
                return await super().handle_async_request(request)

        def factory(**kwargs):
            return _HangOnHostname(calls, {"149.154.167.220": "ok"})

        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)
        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
        resp = await transport.handle_async_request(_telegram_request())
        assert resp.status_code == 200
        assert calls[0]["url_host"] == "149.154.167.220"

    def test_attempt_order_is_ipv4_then_hostname(self):
        transport = tnet.TelegramFallbackTransport(["149.154.167.220", "149.154.166.110"])
        assert transport._attempt_order() == [
            "149.154.167.220",
            "149.154.166.110",
            None,
        ]

    @pytest.mark.asyncio
    async def test_hostname_tried_last_when_ipv4_fails(self, monkeypatch):
        """IPv6-only / seed-IP-blocked hosts still reach the hostname last."""
        calls = []
        behavior = {"149.154.167.220": "timeout", "api.telegram.org": "ok"}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))
        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
        resp = await transport.handle_async_request(_telegram_request())
        assert resp.status_code == 200
        assert [c["url_host"] for c in calls] == ["149.154.167.220", "api.telegram.org"]
        assert transport._sticky_ip is None


class TestFallbackTransportPassthrough:
    """Requests that don't need fallback behavior."""

    @pytest.mark.asyncio
    async def test_non_telegram_host_bypasses_fallback(self, monkeypatch):
        calls = []
        behavior = {}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
        request = httpx.Request("GET", "https://example.com/path")
        resp = await transport.handle_async_request(request)

        assert resp.status_code == 200
        assert calls[0]["url_host"] == "example.com"
        assert transport._sticky_ip is tnet._UNSET


class TestFallbackTransportInit:


    def test_uses_proxy_env_for_primary_and_fallback_transports(self, monkeypatch):
        seen_kwargs = []

        def factory(**kwargs):
            seen_kwargs.append(kwargs.copy())
            return FakeTransport([], {})

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])

        assert transport._fallback_ips == ["149.154.167.220"]
        # Fallback pools are now built lazily (#63311), so __init__ constructs
        # only the primary transport. Force the fallback pool to materialize to
        # observe its kwargs.
        import asyncio

        asyncio.run(transport._get_fallback("149.154.167.220"))
        assert len(seen_kwargs) == 2
        assert all(kwargs["proxy"] == "http://proxy.example:8080" for kwargs in seen_kwargs)

    def test_no_proxy_bypasses_fallback_ip_cidr(self, monkeypatch):
        seen_kwargs = []

        def factory(**kwargs):
            seen_kwargs.append(kwargs.copy())
            return FakeTransport([], {})

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("NO_PROXY", "149.154.160.0/20")
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])

        assert transport._fallback_ips == ["149.154.167.220"]
        # Lazy fallback build (#63311): materialize the fallback pool.
        import asyncio

        asyncio.run(transport._get_fallback("149.154.167.220"))
        assert len(seen_kwargs) == 2
        assert all("proxy" not in kwargs for kwargs in seen_kwargs)

    def test_forwards_limits_to_inner_transports(self, monkeypatch):
        """Verify that caller-supplied limits reach the inner
        AsyncHTTPTransport instances (#58790).  httpx ignores the
        client-level limits kwarg when a custom transport is
        supplied, so the limits must be forwarded via transport_kwargs.
        """
        seen_kwargs = []

        def factory(**kwargs):
            seen_kwargs.append(kwargs.copy())
            return FakeTransport([], {})

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        custom_limits = httpx.Limits(
            max_connections=42,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
        )
        transport = tnet.TelegramFallbackTransport(
            ["149.154.167.220"], limits=custom_limits
        )

        # Lazy fallback build (#63311): __init__ builds only the primary; the
        # fallback pool is constructed on demand. Materialize it so both the
        # primary and the fallback are observed.
        import asyncio

        asyncio.run(transport._get_fallback("149.154.167.220"))
        # 1 primary + 1 fallback = 2 AsyncHTTPTransport instances
        assert len(seen_kwargs) == 2
        for kw in seen_kwargs:
            assert "limits" in kw
            # Caller-supplied limits must win over the setdefault default.
            assert kw["limits"] is custom_limits


class TestFallbackTransportClose:
    @pytest.mark.asyncio
    async def test_aclose_closes_all_transports(self, monkeypatch):
        factory = _fake_transport_factory([], {})
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        transport = tnet.TelegramFallbackTransport(["149.154.167.220", "149.154.167.221"])
        # Lazy fallback build (#63311): materialize both fallback pools so
        # aclose() has something to tear down.
        await transport._get_fallback("149.154.167.220")
        await transport._get_fallback("149.154.167.221")
        await transport.aclose()

        # 1 primary + 2 fallback transports
        assert len(factory.instances) == 3
        assert all(t.closed for t in factory.instances)


# ═══════════════════════════════════════════════════════════════════════════
# Config layer – TELEGRAM_FALLBACK_IPS env → config.extra
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigFallbackIps:
    def test_env_var_populates_config_extra(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides

        monkeypatch.setenv("TELEGRAM_FALLBACK_IPS", "149.154.167.220,149.154.167.221")
        config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")})
        _apply_env_overrides(config)

        assert config.platforms[Platform.TELEGRAM].extra["fallback_ips"] == [
            "149.154.167.220", "149.154.167.221",
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Adapter layer – _fallback_ips() reads config correctly
# ═══════════════════════════════════════════════════════════════════════════

class TestAdapterFallbackIps:
    def _make_adapter(self, extra=None):
        import sys
        from unittest.mock import MagicMock

        # Ensure telegram mock is in place
        if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "__file__"):
            mod = MagicMock()
            mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
            mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
            mod.constants.ChatType.GROUP = "group"
            mod.constants.ChatType.SUPERGROUP = "supergroup"
            mod.constants.ChatType.CHANNEL = "channel"
            mod.constants.ChatType.PRIVATE = "private"
            for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
                sys.modules.setdefault(name, mod)

        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        config = PlatformConfig(enabled=True, token="test-token")
        if extra:
            config.extra.update(extra)
        return TelegramAdapter(config)

    def test_list_in_extra(self):
        adapter = self._make_adapter(extra={"fallback_ips": ["149.154.167.220"]})
        assert adapter._fallback_ips() == ["149.154.167.220"]

    def test_csv_string_in_extra(self):
        adapter = self._make_adapter(extra={"fallback_ips": "149.154.167.220,149.154.167.221"})
        assert adapter._fallback_ips() == ["149.154.167.220", "149.154.167.221"]


# ═══════════════════════════════════════════════════════════════════════════
# DoH auto-discovery
# ═══════════════════════════════════════════════════════════════════════════

def _doh_answer(*ips: str) -> dict:
    """Build a minimal DoH JSON response with A records."""
    return {"Answer": [{"type": 1, "data": ip} for ip in ips]}


class FakeDoHClient:
    """Mock httpx.AsyncClient for DoH queries."""

    def __init__(self, responses: dict):
        # responses: URL prefix → (status, json_body) | Exception
        self._responses = responses
        self.requests_made: list[dict] = []

    @staticmethod
    def _make_response(status, body, url):
        """Build an httpx.Response with a request attached (needed for raise_for_status)."""
        request = httpx.Request("GET", url)
        return httpx.Response(status, json=body, request=request)

    async def get(self, url, *, params=None, headers=None, **kwargs):
        self.requests_made.append({"url": url, "params": params, "headers": headers})
        for prefix, action in self._responses.items():
            if url.startswith(prefix):
                if isinstance(action, Exception):
                    raise action
                status, body = action
                return self._make_response(status, body, url)
        return self._make_response(200, {}, url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestDiscoverFallbackIps:
    """Tests for discover_fallback_ips() — DoH-based auto-discovery."""

    def _patch_doh(self, monkeypatch, responses, system_dns_ips=None):
        """Wire up fake DoH client and system DNS."""
        client = FakeDoHClient(responses)
        monkeypatch.setattr(tnet.httpx, "AsyncClient", lambda **kw: client)

        if system_dns_ips is not None:
            addrs = [(None, None, None, None, (ip, 443)) for ip in system_dns_ips]
            monkeypatch.setattr(tnet.socket, "getaddrinfo", lambda *a, **kw: addrs)
        else:
            def _fail(*a, **kw):
                raise OSError("dns failed")
            monkeypatch.setattr(tnet.socket, "getaddrinfo", _fail)
        return client

    @pytest.mark.asyncio
    async def test_google_and_cloudflare_ips_collected(self, monkeypatch):
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.167.221")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert "149.154.167.220" in ips
        assert "149.154.167.221" in ips

    @pytest.mark.asyncio
    async def test_system_dns_ip_kept_when_doh_confirms(self, monkeypatch):
        """DoH-confirmed IPs are kept even when they match system DNS (#14520).

        The system-DNS IP is often the most reliable path; including it as a
        fallback lets the IP-rewrite retry recover from transient primary-path
        failures instead of jumping straight to the hardcoded seed list.
        """
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.166.110", "149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.166.110")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert ips == ["149.154.166.110", "149.154.167.220"]

    @pytest.mark.asyncio
    async def test_doh_results_deduplicated(self, monkeypatch):
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.167.220")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert ips == ["149.154.167.220"]


    @pytest.mark.asyncio
    async def test_all_doh_ips_same_as_system_dns_kept(self, monkeypatch):
        """DoH agrees with system DNS — keep that IP instead of seed list (#14520).

        Previous behavior fell through to ``_SEED_FALLBACK_IPS`` here, but the
        seed addresses are not routable on every network.  When DoH confirms
        the system IP, that IP is the best candidate we have and should be
        used as the fallback target.
        """
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.166.110")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.166.110")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert ips == ["149.154.166.110"]


    @pytest.mark.asyncio
    async def test_hung_system_dns_does_not_gate_doh_results(self, monkeypatch):
        """#63309: socket.getaddrinfo has no timeout of its own — a wedged OS
        resolver must not stall discovery. DoH answers must come back promptly
        even while the system-DNS worker thread is still hanging."""
        import time as _time

        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer()),
        }, system_dns_ips=["149.154.166.110"])
        monkeypatch.setattr(tnet, "_DOH_TIMEOUT", 0.2)

        def _hung_getaddrinfo(*a, **kw):
            _time.sleep(0.2)  # far beyond the discovery bound
            raise OSError("resolver wedged")

        monkeypatch.setattr(tnet.socket, "getaddrinfo", _hung_getaddrinfo)

        start = _time.monotonic()
        ips = await tnet.discover_fallback_ips()
        elapsed = _time.monotonic() - start

        assert ips == ["149.154.167.220"]
        assert elapsed < 1.4, f"discovery gated on hung system DNS ({elapsed:.2f}s)"

