"""Tests for explicit outbound A2A User-Agent headers and HTTP 403 diagnostics (#161003)."""

from __future__ import annotations

import io
import urllib.error

from plugins.platforms.a2a import security, tools


class TestA2AOutboundUserAgent:
    def test_http_json_sets_explicit_user_agent_and_respects_caller_override(self, monkeypatch):
        """Outbound A2A requests must not send Python-urllib's default User-Agent."""
        seen_headers = []

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"ok": true}'

        def fake_urlopen(req, timeout=0):
            seen_headers.append(dict(req.header_items()))
            return _Resp()

        monkeypatch.setattr(tools.urllib.request, "urlopen", fake_urlopen)

        caller_headers = {"Authorization": "Bearer tok"}
        assert tools._http_json("https://peer.example/card", caller_headers, 10, "GET") == {"ok": True}
        assert "User-Agent" not in caller_headers
        ua_default = next(v for k, v in seen_headers[0].items() if k.lower() == "user-agent")
        assert ua_default.startswith("hermes-agent-a2a/")

        tools._http_json("https://peer.example/card", {"user-agent": "CustomClient/1.0"}, 10, "GET")
        ua_override = next(v for k, v in seen_headers[1].items() if k.lower() == "user-agent")
        assert ua_override == "CustomClient/1.0"

    def test_post_callback_sets_explicit_user_agent(self, monkeypatch):
        """Push notification callbacks also carry an explicit Hermes User-Agent."""
        captured = {}

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def stream(self, method, url, content=None, headers=None):
                captured["headers"] = dict(headers or {})
                return type("_R", (), {
                    "status_code": 200,
                    "__enter__": lambda s: s,
                    "__exit__": lambda s, *a: False,
                })()

        monkeypatch.setattr("httpx.Client", lambda **kw: _FakeClient())
        caller_headers = {"Content-Type": "application/json"}
        assert security.post_callback("http://127.0.0.1:9999/cb", b"{}", caller_headers, localhost_mode=True) == 200
        assert "User-Agent" not in caller_headers
        assert captured["headers"]["User-Agent"].startswith("hermes-agent-a2a/")

    def test_403_surfaces_response_body_detail_for_proxy_blocks(self, monkeypatch):
        err_discover = urllib.error.HTTPError(
            "https://peer.example/.well-known/agent-card.json",
            403,
            "Forbidden",
            {},
            io.BytesIO(b"error code: 1010"),
        )
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **kw: (_ for _ in ()).throw(err_discover))
        out_discover = tools.a2a_discover({"url": "https://peer.example"})
        assert "HTTP 403" in out_discover
        assert "error code: 1010" in out_discover

        err_call = urllib.error.HTTPError(
            "https://peer.example/",
            403,
            "Forbidden",
            {},
            io.BytesIO(b"error code: 1010"),
        )
        monkeypatch.setattr(tools, "_resolve_peer", lambda name: {"url": "https://peer.example", "auth": {}, "timeout": 5})
        monkeypatch.setattr(tools, "_send_task", lambda *a, **kw: (_ for _ in ()).throw(err_call))
        out_call = tools.a2a_call({"agent": "cf-peer", "message": "hi"})
        assert "HTTP 403" in out_call
        assert "error code: 1010" in out_call
        assert "proxy/WAF" in out_call
