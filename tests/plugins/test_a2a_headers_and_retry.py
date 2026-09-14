"""Focused coverage for per-peer headers, User-Agent, 524 indeterminate
handling, and destination-bound secret forwarding (PR #86322).

Extracted from test_a2a_plugin.py when credential forwarding became
origin-bound and 524 responses became a typed indeterminate outcome
(no automatic replay), per upstream review: keep this surface in its own
module under the 2K-line test-file invariant.
"""

from __future__ import annotations

import pytest
import urllib.error as urlerr
import urllib.request

from plugins.platforms.a2a import protocol, tools


# ---------------------------------------------------------------------------
# Header propagation and precedence
# ---------------------------------------------------------------------------

class TestPerPeerHeaders:
    def test_resolve_peer_carries_headers(self, monkeypatch):
        cfg = {"a2a_agents": {"mac": {
            "url": "http://localhost:9999",
            "headers": {"CF-Access-Client-Id": "cf-id"},
        }}}
        monkeypatch.setattr(tools, "_load_config", lambda: cfg)
        peer = tools._resolve_peer("mac")
        assert peer is not None
        assert peer["headers"] == {"CF-Access-Client-Id": "cf-id"}

    def test_resolve_peer_defaults_headers_to_empty(self, monkeypatch):
        cfg = {"a2a_agents": {"mac": {"url": "http://localhost:9999"}}}
        monkeypatch.setattr(tools, "_load_config", lambda: cfg)
        peer = tools._resolve_peer("mac")
        assert peer is not None
        assert peer["headers"] == {}

    def test_call_sends_auth_and_custom_headers(self, monkeypatch):
        cfg = {"a2a_agents": {"mac": {
            "url": "http://localhost:9999",
            "auth": {"type": "bearer", "token": "tok-123"},
            "headers": {"CF-Access-Client-Id": "cf-id", "CF-Access-Client-Secret": "cf-sec"},
        }}}
        monkeypatch.setattr(tools, "_load_config", lambda: cfg)
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: None)
        captured = {}

        def fake_post(url, body, headers, timeout, retry_524=False, allowed_origins=()):
            captured.update(headers)
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t", "c1", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        tools.a2a_call({"agent": "mac", "message": "ping"})
        assert captured["Authorization"] == "Bearer tok-123"
        assert captured["CF-Access-Client-Id"] == "cf-id"
        assert captured["CF-Access-Client-Secret"] == "cf-sec"

    def test_custom_headers_take_precedence(self, monkeypatch):
        """Peer config headers win over the derived Authorization header on
        name collision (a proxy may require a different auth scheme)."""
        cfg = {"a2a_agents": {"mac": {
            "url": "http://localhost:9999",
            "auth": {"type": "bearer", "token": "tok-123"},
            "headers": {"Authorization": "***"},
        }}}
        monkeypatch.setattr(tools, "_load_config", lambda: cfg)
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: None)
        captured = {}

        def fake_post(url, body, headers, timeout, retry_524=False, allowed_origins=()):
            captured.update(headers)
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t", "c1", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        tools.a2a_call({"agent": "mac", "message": "ping"})
        assert captured["Authorization"] == "***"

    def test_orchestrate_fanout_sends_custom_headers(self, monkeypatch):
        cfg = {"a2a_agents": {
            "a": {"url": "http://localhost:9001", "capabilities": ["x"], "headers": {"X-Tenant": "t1"}},
            "b": {"url": "http://localhost:9002", "capabilities": ["x"], "headers": {"X-Tenant": "t2"}},
        }}
        monkeypatch.setattr(tools, "_load_config", lambda: cfg)
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: None)
        seen = []

        def fake_post(url, body, headers, timeout, retry_524=False, allowed_origins=()):
            seen.append((url, headers.get("X-Tenant")))
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t", "c1", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        tools.a2a_orchestrate({"capability": "x", "message": "m"})
        assert len(seen) == 2
        assert {t for _, t in seen} == {"t1", "t2"}


class TestClientHttpEdgeCases:
    def test_get_json_sends_user_agent_and_custom_headers(self, monkeypatch):
        seen = {}

        class _Resp:
            def read(self):
                return b'{"name": "peer"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_open(req, timeout=None):
            seen["headers"] = dict(req.headers)
            return _Resp()

        monkeypatch.setattr(tools, "_open_url_no_redirect_leak",
                            lambda req, timeout, allowed=(): fake_open(req, timeout))
        out = tools._http_get_json("http://peer/.well-known/agent-card.json",
                                   {"CF-Access-Client-Id": "cf-id"}, 30)
        assert out == {"name": "peer"}
        lowered = {k.lower(): v for k, v in seen["headers"].items()}
        assert lowered["user-agent"] == "Hermes-A2A/1.0"
        assert lowered["cf-access-client-id"] == "cf-id"


# ---------------------------------------------------------------------------
# 524 handling: typed indeterminate, never auto-retried
# ---------------------------------------------------------------------------

class Test524Indeterminate:
    def test_524_raises_typed_indeterminate_error(self, monkeypatch):
        """A 524 is indeterminate (origin may have completed the task) and is
        NEVER auto-retried: it raises _A2aIndeterminateError so callers can
        distinguish 'failed safely' from 'may have executed'."""
        attempts = {"n": 0}

        def fake_open(req, timeout, allowed=()):
            attempts["n"] += 1
            raise urlerr.HTTPError(req.full_url, 524, "Origin Time-out", {}, None)

        monkeypatch.setattr(tools, "_open_url_no_redirect_leak",
                            lambda req, timeout, allowed=(): fake_open(req, timeout))
        with pytest.raises(tools._A2aIndeterminateError):
            tools._http_post_json("http://peer/", {"x": 1}, {}, 30)
        assert attempts["n"] == 1

    def test_other_errors_still_raise_httperror(self, monkeypatch):
        def fake_open(req, timeout, allowed=()):
            raise urlerr.HTTPError(req.full_url, 502, "Bad Gateway", {}, None)

        monkeypatch.setattr(tools, "_open_url_no_redirect_leak",
                            lambda req, timeout, allowed=(): fake_open(req, timeout))
        with pytest.raises(urlerr.HTTPError):
            tools._http_post_json("http://peer/", {"x": 1}, {}, 30)

    def test_a2a_call_surfaces_indeterminate_advice(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: {"a2a_agents": {"r": {
            "url": "http://localhost:9999"}}})

        def fake_send(*a, **k):
            raise tools._A2aIndeterminateError("peer origin timed out (HTTP 524)")

        monkeypatch.setattr(tools, "_send_task", fake_send)
        out = tools.a2a_call({"agent": "r", "message": "hello"})
        assert "INDETERMINATE" in out
        assert "Do not blindly retry" in out

    def test_524_single_attempt_no_sleep(self, monkeypatch):
        """No retry ladder, no backoff sleeps — exactly one request."""
        sleeps = []
        monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
        attempts = {"n": 0}

        def fake_open(req, timeout, allowed=()):
            attempts["n"] += 1
            raise urlerr.HTTPError(req.full_url, 524, "Origin Time-out", {}, None)

        monkeypatch.setattr(tools, "_open_url_no_redirect_leak",
                            lambda req, timeout, allowed=(): fake_open(req, timeout))
        with pytest.raises(tools._A2aIndeterminateError):
            tools._http_post_json("http://peer/", {"x": 1}, {}, 30)
        assert attempts["n"] == 1 and sleeps == []


class TestPostUserAgent:
    def test_post_sends_hermes_user_agent(self, monkeypatch):
        """The Hermes-A2A/1.0 UA must be on real POSTs (proxy filtering);
        asserted at the opener seam where the Request is actually built."""
        captured = {}

        def fake_open(req, timeout, allowed=()):
            captured["ua"] = req.get_header("User-agent")
            return _Ctxlike()

        class _Ctxlike:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"ok": true}'

        monkeypatch.setattr(tools, "_open_url_no_redirect_leak", fake_open)
        out = tools._http_post_json("http://peer/", {"x": 1}, {}, 30)
        assert out == {"ok": True}
        assert captured["ua"] == "Hermes-A2A/1.0"


# ---------------------------------------------------------------------------
# Redirect policy: credentials never follow a cross-origin 3xx
# ---------------------------------------------------------------------------

class TestRedirectPolicy:
    def _redirect_handler(self, allowed=()):
        from plugins.platforms.a2a.tools import _NoCredentialRedirectHandler
        return _NoCredentialRedirectHandler(tuple(allowed))

    def test_same_origin_redirect_followed(self):
        h = self._redirect_handler()
        req = urllib.request.Request("https://configured.example/rpc")
        new = h.redirect_request(req, None, 302, "Found", {},
                                 "https://configured.example/rpc/v2")
        assert new is not None  # followed: same origin, credentials stay on-service

    def test_cross_origin_redirect_refused(self):
        """THE exfiltration vector: a 3xx from the (trusted, same-origin)
        endpoint pointing at a foreign host must be refused, not followed.
        Under the old default opener the foreign host received the full
        credential map (empirically reproduced in review run t_2ee74a0d)."""
        h = self._redirect_handler()
        req = urllib.request.Request("https://configured.example/rpc")
        with pytest.raises(urllib.error.HTTPError) as ei:
            h.redirect_request(req, None, 302, "Found", {},
                               "https://evil.example/collect")
        assert "refused" in str(ei.value.reason)

    def test_cross_origin_redirect_allowed_via_pinned_origin(self):
        h = self._redirect_handler(allowed=["https://trusted.example/rpc"])
        req = urllib.request.Request("https://configured.example/rpc")
        new = h.redirect_request(req, None, 302, "Found", {},
                                 "https://trusted.example/other/path")
        assert new is not None

    def test_scheme_change_is_cross_origin(self):
        h = self._redirect_handler()
        req = urllib.request.Request("https://configured.example/rpc")
        with pytest.raises(urllib.error.HTTPError):
            h.redirect_request(req, None, 302, "Found", {},
                               "http://configured.example/rpc")  # https->http downgrade refused

    def test_port_change_is_cross_origin(self):
        h = self._redirect_handler()
        req = urllib.request.Request("https://configured.example/rpc")
        with pytest.raises(urllib.error.HTTPError):
            h.redirect_request(req, None, 302, "Found", {},
                               "https://configured.example:8443/rpc")

    def test_end_to_end_post_refuses_cross_origin_redirect(self, monkeypatch):
        """Full POST path through the guarded opener: the redirecting server
        never gets a second request, the foreign target is never contacted,
        and the caller sees the refusal."""
        import http.server
        import threading
        hits = {"redirector": 0, "foreign": 0}

        class Foreign(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                hits["foreign"] += 1
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        class Redirector(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                hits["redirector"] += 1
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{foreign_port}/x")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        fs = http.server.HTTPServer(("127.0.0.1", 0), Foreign)
        foreign_port = fs.server_address[1]
        rs = http.server.HTTPServer(("127.0.0.1", 0), Redirector)
        threading.Thread(target=fs.serve_forever, daemon=True).start()
        threading.Thread(target=rs.serve_forever, daemon=True).start()
        try:
            with pytest.raises(urllib.error.HTTPError) as ei:
                tools._http_post_json(f"http://127.0.0.1:{rs.server_address[1]}/",
                                      {"x": 1}, {"Authorization": "Bearer tok"}, 10)
            assert hits == {"redirector": 1, "foreign": 0}
            assert "refused" in str(ei.value.reason)
        finally:
            fs.shutdown()
            rs.shutdown()


# ---------------------------------------------------------------------------
# End-to-end: allowed_rpc_origins governs the task POST redirect path
# ---------------------------------------------------------------------------

class TestSendTaskRedirectAllowlist:
    """Real _send_task against live loopback servers: a configured-origin
    POST that 302s to a PINNED origin must be followed with credentials;
    the same redirect to an UNPINNED origin must be refused with nothing
    delivered to the foreign host. Pins the wiring the maintainer flagged
    (the allowlist reached the card fetch but not the task POST)."""

    def _spin(self, handler):
        import http.server
        import threading
        s = http.server.HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        return s, f"http://127.0.0.1:{s.server_address[1]}"

    def test_pinned_origin_redirect_followed_with_credentials(self):
        """Through the REAL _send_task path (wiring under test): the peer's
        configured origin 302s the task POST to a PINNED origin. The pinned
        server must receive the credential-bearing request. Fails if
        _send_task drops allowed_origins on the POST call."""
        import http.server
        import json as _json

        seen = {}

        def rpc_ok(req_id):
            return _json.dumps(protocol.jsonrpc_result(
                req_id,
                protocol.build_task("t1", "c1", protocol.STATE_COMPLETED, "ok"))).encode()

        class Pinned(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen["pinned_path"] = self.path
                seen["pinned_auth"] = self.headers.get("Authorization")
                seen["pinned_cf"] = self.headers.get("CF-Access-Client-Id")
                b = rpc_ok(1)
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

        class Redirector(http.server.BaseHTTPRequestHandler):
            pinned_url = None

            def do_GET(self):
                b = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b)

            def log_message(self, *a):
                pass

            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", self.pinned_url + "/rpc")
                self.send_header("Content-Length", "0")
                self.end_headers()

        ps, pinned = self._spin(Pinned)
        Redirector.pinned_url = pinned
        rs, redir = self._spin(Redirector)
        peer = {"url": redir, "timeout": 10,
                "auth": {"type": "bearer", "token": "tok-pin"},
                "headers": {"CF-Access-Client-Id": "cf-pin"},
                "allowed_rpc_origins": [pinned]}
        reply, _ctx, _state = tools._send_task("pin-peer", peer, "hello", "")
        assert "ok" in (reply or "")
        assert seen.get("pinned_auth") == "Bearer tok-pin", "credentials must reach pinned origin"
        assert seen.get("pinned_cf") == "cf-pin"
        ps.shutdown()
        rs.shutdown()

    def test_full_send_task_unpinned_redirect_refused_no_delivery(self):
        import http.server

        foreign_hits = {"n": 0, "auth": None}

        class Foreign(http.server.BaseHTTPRequestHandler):
            def _r(self):
                foreign_hits["n"] += 1
                foreign_hits["auth"] = self.headers.get("Authorization")
                b = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b)

            do_GET = do_POST = _r

            def log_message(self, *a):
                pass

        class Redirector(http.server.BaseHTTPRequestHandler):
            foreign_url = None

            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", self.foreign_url + "/steal")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        fs, foreign = self._spin(Foreign)
        Redirector.foreign_url = foreign
        rs, redir = self._spin(Redirector)
        peer = {"url": redir, "timeout": 10,
                "auth": {"type": "bearer", "token": "tok-x"},
                "headers": {"CF-Access-Client-Id": "cf-x"}}
        with pytest.raises(urllib.error.HTTPError) as ei:
            tools._send_task("redir-peer", peer, "hello", "")
        assert foreign_hits["n"] == 0, "foreign host must never be contacted"
        assert "refused" in str(ei.value.reason)
        fs.shutdown()
        rs.shutdown()


# ---------------------------------------------------------------------------
# Origin-level allowlist matching
# ---------------------------------------------------------------------------

class TestOriginLevelAllowlist:
    def test_allowlist_entry_matches_any_path(self):
        peer = {"url": "https://configured.example",
                "allowed_rpc_origins": ["https://trusted.example"]}
        assert tools._origin_allowed("https://trusted.example/rpc", peer)
        assert tools._origin_allowed("https://trusted.example/", peer)
        assert tools._origin_allowed("https://trusted.example", peer)

    def test_allowlist_is_origin_scoped_not_string_scoped(self):
        peer = {"url": "https://configured.example",
                "allowed_rpc_origins": ["https://trusted.example"]}
        assert not tools._origin_allowed("https://trusted.evil.example/rpc", peer)
        assert not tools._origin_allowed("https://trusted.example:8443/rpc", peer)
        assert not tools._origin_allowed("http://trusted.example/rpc", peer)

    def test_port0_is_not_silently_defaulted(self):
        assert tools._url_origin("https://configured.example:0/rpc") == ("https", "configured.example:0")
        assert not tools._url_same_origin("https://configured.example:0/rpc",
                                          "https://configured.example/rpc")


class TestAuthCollisionWarning:
    def test_authorization_override_logs_warning(self, monkeypatch, caplog):
        import logging as _logging
        peer = {"url": "http://localhost:9999",
                "auth": {"type": "bearer", "token": "t"},
                "headers": {"Authorization": "ProxyAuth xyz"}}
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: None)

        def fake_post(url, body, headers, timeout, retry_524=False, allowed_origins=()):
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t", "c1", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        with caplog.at_level(_logging.WARNING, logger="plugins.platforms.a2a.tools"):
            tools._send_task("p", peer, "hi", "")
        assert any("override the derived Authorization" in r.message for r in caplog.records)

    def test_distinct_headers_do_not_warn(self, monkeypatch, caplog):
        import logging as _logging
        monkeypatch.setattr(tools, "_load_config", lambda: {"a2a_agents": {"p": {
            "url": "http://localhost:9999",
            "auth": {"type": "bearer", "token": "t"},
            "headers": {"CF-Access-Client-Id": "x"},
        }}})

        def fake_send(*a, **k):
            return ("reply", "ctx", protocol.STATE_COMPLETED)

        monkeypatch.setattr(tools, "_send_task", fake_send)
        with caplog.at_level(_logging.WARNING, logger="plugins.platforms.a2a.tools"):
            tools.a2a_call({"agent": "p", "message": "hi"})
        assert not any("override the derived Authorization" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Destination-bound secret forwarding
# ---------------------------------------------------------------------------

class TestRpcOriginBinding:
    def _peer(self, **extra):
        return {"url": "https://configured.example", **extra}

    def test_same_origin_card_url_used(self):
        assert tools._url_same_origin("https://configured.example/rpc",
                                      "https://configured.example")
        assert not tools._url_same_origin("https://other.example/rpc",
                                          "https://configured.example")

    def test_port_and_scheme_are_part_of_origin(self):
        assert not tools._url_same_origin("http://configured.example/rpc",
                                          "https://configured.example")
        assert not tools._url_same_origin("https://configured.example:8443/rpc",
                                          "https://configured.example")

    def test_cross_origin_rpc_rejected_without_allowlist(self, monkeypatch):
        """A card advertising a foreign RPC origin must NOT receive the
        credential-bearing header map: the send falls back to the
        configured origin."""
        card = {"supportedInterfaces": [{"protocolBinding": "JSONRPC",
                                         "url": "https://evil.example/rpc"}]}
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: card)
        posted = {}

        def fake_post(url, body, headers, timeout, retry_524=False, allowed_origins=()):
            posted["url"] = url
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t", "c1", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        import logging as _logging
        monkeypatch.setattr(tools, "logger", _logging.getLogger("test-a2a"))
        reply, ctx, state = tools._send_task(
            "peer-x", self._peer(auth={"type": "bearer", "token": "tok"}), "hi", "")
        assert posted["url"] == "https://configured.example"

    def test_cross_origin_rpc_allowed_via_allowlist(self, monkeypatch):
        card = {"supportedInterfaces": [{"protocolBinding": "JSONRPC",
                                         "url": "https://trusted.example/rpc"}]}
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: card)
        posted = {}

        def fake_post(url, body, headers, timeout, retry_524=False, allowed_origins=()):
            posted["url"] = url
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t", "c1", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        peer = self._peer(allowed_rpc_origins=["https://trusted.example/rpc"])
        tools._send_task("peer-x", peer, "hi", "")
        assert posted["url"] == "https://trusted.example/rpc"

    def test_card_fetch_carries_credentials_to_configured_origin(self, monkeypatch):
        """The card fetch goes to the CONFIGURED origin, so it carries the
        credential map (a Cloudflare-Access-fronted peer otherwise 403s the
        card, losing streaming/capability discovery). The egress bound is on
        the RPC destination, proven by the cross-origin tests above."""
        fetched = {}

        def fake_fetch(url, headers, timeout, allowed_origins=()):
            fetched["url"] = url
            fetched["headers"] = dict(headers)
            return None

        monkeypatch.setattr(tools, "_fetch_card", fake_fetch)

        def fake_post(url, body, headers, timeout, retry_524=False, allowed_origins=()):
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t", "c1", protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        peer = self._peer(auth={"type": "bearer", "token": "tok"},
                          headers={"CF-Access-Client-Secret": "sec"})
        tools._send_task("peer-x", peer, "hi", "")
        assert fetched["url"].startswith("https://configured.example")
        assert fetched["headers"].get("Authorization") == "Bearer tok"
        assert fetched["headers"].get("CF-Access-Client-Secret") == "sec"
