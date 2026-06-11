"""Keyless Parallel search via the free hosted Search MCP.

Covers the transport added in ``plugins/web/parallel/provider.py`` that lets
``web_search`` work with no ``PARALLEL_API_KEY``:

- ``_mcp_headers``  — Bearer attached only when a key is held
- ``_decode_mcp_envelope`` — plain-JSON and SSE (``data:``) response bodies
- ``_mcp_payload`` — structuredContent preferred, text-block JSON fallback, errors
- ``_mcp_web_search`` — full handshake (mocked transport) → standard search shape
- ``ParallelWebSearchProvider.search`` — keyless path routes to the MCP
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

import plugins.web.parallel.provider as pp


# ─── _mcp_headers ──────────────────────────────────────────────────────────

class TestMcpHeaders:
    def test_anonymous_has_no_authorization(self):
        h = pp._mcp_headers(session_id=None, api_key=None)
        assert "Authorization" not in h
        assert h["Accept"] == "application/json, text/event-stream"
        assert "Mcp-Session-Id" not in h

    def test_user_agent_is_generic_not_hermes(self):
        # Telemetry policy: no third-party usage attribution without opt-in.
        # The UA must be set (not python-httpx default) but must not name
        # hermes, on both the anonymous and keyed paths.
        for ua in (
            pp._mcp_headers(session_id=None, api_key=None)["User-Agent"],
            pp._mcp_headers(session_id="sid", api_key="pk-live")["User-Agent"],
        ):
            assert ua == f"{pp._MCP_CLIENT_NAME}/{pp._MCP_CLIENT_VERSION}"
            assert "hermes" not in ua.lower()

    def test_session_id_and_bearer_when_present(self):
        h = pp._mcp_headers(session_id="sid-123", api_key="pk-live")
        assert h["Mcp-Session-Id"] == "sid-123"
        assert h["Authorization"] == "Bearer pk-live"


# ─── SSE / JSON-RPC parsing ──────────────────────────────────────────────────

class TestMcpResponseParsing:
    def test_plain_json_matched_by_id(self):
        body = '{"jsonrpc":"2.0","id":"abc","result":{"ok":true}}'
        assert pp._mcp_response_envelope(body, "abc")["result"]["ok"] is True

    def test_sse_selects_response_for_request_id_skipping_notifications(self):
        # A progress notification (no id) precedes the real result; an unrelated
        # response id is also present. We must pick the one matching our id.
        body = (
            'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress","params":{"p":1}}\n\n'
            'event: message\ndata: {"jsonrpc":"2.0","id":"other","result":{"ok":false}}\n\n'
            'event: message\ndata: {"jsonrpc":"2.0","id":"req-1","result":{"ok":true}}\n\n'
        )
        env = pp._mcp_response_envelope(body, "req-1")
        assert env["result"]["ok"] is True

    def test_sse_multiline_data_concatenated(self):
        body = 'data: {"jsonrpc":"2.0","id":"x",\ndata: "result":{"n":42}}\n\n'
        assert pp._mcp_response_envelope(body, "x")["result"]["n"] == 42

    def test_falls_back_to_last_result_when_id_absent(self):
        body = '{"jsonrpc":"2.0","id":"server-chose","result":{"ok":true}}'
        # request id doesn't match, but there's a single result → use it
        assert pp._mcp_response_envelope(body, "mismatch")["result"]["ok"] is True

    def test_empty_body(self):
        assert pp._mcp_response_envelope("", "x") == {}
        assert pp._mcp_response_envelope("   ", "x") == {}

    def test_batched_json_array_flattened(self):
        # Streamable HTTP may batch messages into a JSON array.
        body = ('[{"jsonrpc":"2.0","method":"notifications/progress"},'
                '{"jsonrpc":"2.0","id":"req-9","result":{"ok":true}}]')
        assert pp._mcp_response_envelope(body, "req-9")["result"]["ok"] is True

    def test_batched_sse_data_array_flattened(self):
        body = 'data: [{"jsonrpc":"2.0","id":"a","result":{"n":1}}]\n\n'
        assert pp._mcp_response_envelope(body, "a")["result"]["n"] == 1


# ─── _mcp_payload ────────────────────────────────────────────────────────────

class TestMcpPayload:
    def test_prefers_structured_content(self):
        env = {"result": {"structuredContent": {"results": [{"url": "u"}]},
                          "content": [{"type": "text", "text": "ignored"}]}}
        assert pp._mcp_payload(env) == {"results": [{"url": "u"}]}

    def test_parses_text_block_json(self):
        inner = {"search_id": "s1", "results": [{"url": "u", "title": "t"}]}
        env = {"result": {"content": [{"type": "text", "text": json.dumps(inner)}]}}
        assert pp._mcp_payload(env)["search_id"] == "s1"

    def test_raises_on_jsonrpc_error(self):
        with pytest.raises(RuntimeError, match="Parallel MCP error"):
            pp._mcp_payload({"error": {"code": -32000, "message": "boom"}})

    def test_raises_on_tool_iserror(self):
        with pytest.raises(RuntimeError, match="Parallel MCP tool error"):
            pp._mcp_payload({"result": {"isError": True, "content": []}})


# ─── _mcp_web_search (mocked transport) ──────────────────────────────────────

class _FakeResponse:
    def __init__(self, *, text="", headers=None):
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        return None


class _FakeClient:
    """Stands in for httpx.Client: replays init → ack → tools/call."""

    def __init__(self, search_payload, init_session_id="server-sid"):
        self._search_payload = search_payload
        self._init_session_id = init_session_id
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.calls.append({"headers": headers, "json": json})
        req = json or {}
        method = req.get("method")
        req_id = req.get("id")
        if method == "initialize":
            # Echo the request id, as the real server does.
            return _FakeResponse(
                text=json_dumps({"jsonrpc": "2.0", "id": req_id,
                                 "result": {"protocolVersion": "2099-01-01"}}),
                headers=(
                    {"mcp-session-id": self._init_session_id}
                    if self._init_session_id is not None
                    else {}
                ),
            )
        if method == "notifications/initialized":
            return _FakeResponse(text="")
        # tools/call
        envelope = {"jsonrpc": "2.0", "id": req_id, "result": {
            "content": [{"type": "text", "text": json_dumps(self._search_payload)}],
        }}
        return _FakeResponse(text=json_dumps(envelope))


def json_dumps(obj):
    return json.dumps(obj)


class TestMcpWebSearch:
    def _payload(self, n):
        return {"search_id": "s", "results": [
            {"url": f"https://ex/{i}", "title": f"t{i}",
             "excerpts": [f"a{i}", f"b{i}"]}
            for i in range(n)
        ]}

    def test_returns_standard_shape_and_handshake(self):
        fake = _FakeClient(self._payload(3))
        with patch.object(pp.httpx, "Client", return_value=fake):
            out = pp._mcp_web_search("hello", limit=5, api_key=None)

        assert out["success"] is True
        # Free-tier results credit Parallel.
        assert "Parallel" in out["attribution"]
        web = out["data"]["web"]
        assert [r["position"] for r in web] == [1, 2, 3]
        assert web[0]["url"] == "https://ex/0"
        assert web[0]["description"] == "a0 b0"  # excerpts joined
        # handshake order
        methods = [c["json"].get("method") for c in fake.calls]
        assert methods == ["initialize", "notifications/initialized", "tools/call"]
        # session id from the initialize response header is reused
        assert fake.calls[-1]["headers"]["Mcp-Session-Id"] == "server-sid"

    def test_stateless_server_no_session_header_not_invented(self):
        # A stateless Streamable-HTTP server may omit mcp-session-id on
        # initialize; we must NOT invent one (sending an unissued session id can
        # get follow-up requests rejected). The follow-ups carry no header.
        fake = _FakeClient(self._payload(1), init_session_id=None)
        with patch.object(pp.httpx, "Client", return_value=fake):
            out = pp._mcp_web_search("hello", limit=5, api_key=None)
        assert out["success"] is True
        follow_ups = [c for c in fake.calls if c["json"].get("method") != "initialize"]
        assert follow_ups, "expected notifications/initialized + tools/call"
        assert all("Mcp-Session-Id" not in c["headers"] for c in follow_ups)
        # anonymous → no Authorization on any call
        assert all("Authorization" not in c["headers"] for c in fake.calls)
        # tools/call mirrors query into objective + search_queries
        args = fake.calls[-1]["json"]["params"]["arguments"]
        assert args["objective"] == "hello"
        assert args["search_queries"] == ["hello"]

    def test_limit_is_applied_client_side(self):
        fake = _FakeClient(self._payload(10))
        with patch.object(pp.httpx, "Client", return_value=fake):
            out = pp._mcp_web_search("q", limit=2, api_key=None)
        assert len(out["data"]["web"]) == 2

    def test_bearer_attached_when_key_present(self):
        fake = _FakeClient(self._payload(1))
        with patch.object(pp.httpx, "Client", return_value=fake):
            pp._mcp_web_search("q", limit=1, api_key="pk-live")
        assert all(c["headers"]["Authorization"] == "Bearer pk-live" for c in fake.calls)

    def test_negotiated_protocol_version_echoed_post_init(self):
        fake = _FakeClient(self._payload(1))
        with patch.object(pp.httpx, "Client", return_value=fake):
            pp._mcp_web_search("q", limit=1, api_key=None)
        # initialize request doesn't carry the (not-yet-negotiated) version...
        assert "MCP-Protocol-Version" not in fake.calls[0]["headers"]
        # ...but notifications/initialized and tools/call echo the negotiated one.
        assert fake.calls[1]["headers"]["MCP-Protocol-Version"] == "2099-01-01"
        assert fake.calls[-1]["headers"]["MCP-Protocol-Version"] == "2099-01-01"


# ─── provider.search keyless routing ─────────────────────────────────────────

class TestProviderKeylessSearch:
    def test_search_without_key_uses_mcp(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        captured = {}

        def _fake(query, limit, api_key):
            captured.update(query=query, limit=limit, api_key=api_key)
            return {"success": True, "data": {"web": []}}

        monkeypatch.setattr(pp, "_mcp_web_search", _fake)
        out = pp.ParallelWebSearchProvider().search("kittens", limit=4)
        assert out["success"] is True
        assert captured == {"query": "kittens", "limit": 4, "api_key": None}

    def test_is_available_reflects_key(self, monkeypatch):
        # is_available() gates the registry's active-provider walk + picker, so
        # it's key-based (keyless dispatch is handled by _get_backend, not this).
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        assert pp.ParallelWebSearchProvider().is_available() is False
        monkeypatch.setenv("PARALLEL_API_KEY", "k")
        assert pp.ParallelWebSearchProvider().is_available() is True


# ─── web_fetch (keyless extract) ─────────────────────────────────────────────

class TestMcpWebFetch:
    def _payload(self, urls):
        return {"extract_id": "e1", "results": [
            {"url": u, "title": f"T{i}", "publish_date": None,
             "excerpts": [f"chunk-a-{i}", f"chunk-b-{i}"]}
            for i, u in enumerate(urls)
        ]}

    def test_maps_to_extract_shape(self):
        urls = ["https://a.test", "https://b.test"]
        fake = _FakeClient(self._payload(urls))
        with patch.object(pp.httpx, "Client", return_value=fake):
            out = pp._mcp_web_fetch(urls, api_key=None)
        assert [r["url"] for r in out] == urls
        assert out[0]["content"] == "chunk-a-0\n\nchunk-b-0"
        assert out[0]["raw_content"] == out[0]["content"]
        assert out[0]["metadata"] == {"sourceURL": "https://a.test", "title": "T0"}
        # tools/call targeted web_fetch, requesting full page bodies.
        args = fake.calls[-1]["json"]["params"]
        assert args["name"] == "web_fetch"
        assert args["arguments"]["urls"] == urls
        assert args["arguments"]["full_content"] is True
        assert args["arguments"]["session_id"].startswith(f"{pp._MCP_CLIENT_NAME}-")

    def test_prefers_full_content_over_excerpts(self):
        payload = {"results": [
            {"url": "https://a.test", "title": "T",
             "excerpts": ["snippet"], "full_content": "the entire page body"},
        ]}
        fake = _FakeClient(payload)
        with patch.object(pp.httpx, "Client", return_value=fake):
            out = pp._mcp_web_fetch(["https://a.test"], api_key=None)
        assert out[0]["content"] == "the entire page body"

    def test_missing_url_becomes_error_entry(self):
        # Server returns only one of the two requested URLs.
        fake = _FakeClient(self._payload(["https://a.test"]))
        with patch.object(pp.httpx, "Client", return_value=fake):
            out = pp._mcp_web_fetch(["https://a.test", "https://missing.test"], api_key=None)
        assert len(out) == 2
        missing = [r for r in out if r["url"] == "https://missing.test"][0]
        assert "error" in missing
        assert missing["content"] == ""

    def test_preserves_order_and_duplicate_inputs(self):
        # MCP returns each unique URL once; output must still be one row per
        # input, in order, including the duplicate.
        fake = _FakeClient(self._payload(["https://a.test", "https://b.test"]))
        urls = ["https://b.test", "https://a.test", "https://b.test"]
        with patch.object(pp.httpx, "Client", return_value=fake):
            out = pp._mcp_web_fetch(urls, api_key=None)
        assert [r["url"] for r in out] == urls  # one row per input, in order
        assert all("error" not in r for r in out)  # all three resolved

    def test_extract_without_key_uses_web_fetch(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        captured = {}

        def _fake(urls, api_key):
            captured.update(urls=list(urls), api_key=api_key)
            return [{"url": urls[0], "title": "", "content": "x",
                     "raw_content": "x", "metadata": {}}]

        monkeypatch.setattr(pp, "_mcp_web_fetch", _fake)
        out = asyncio.run(pp.ParallelWebSearchProvider().extract(["https://x.test"]))
        assert out[0]["content"] == "x"
        assert captured == {"urls": ["https://x.test"], "api_key": None}


# ─── keyed v1 REST search ────────────────────────────────────────────────────

class TestKeyedV1Search:
    def test_passes_max_results_and_omits_branding(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-live")
        monkeypatch.delenv("PARALLEL_SEARCH_MODE", raising=False)
        captured = {}

        class _Res:
            def __init__(self, url):
                self.url, self.title, self.excerpts = url, "T", ["x"]

        class _Resp:
            results = [_Res(f"https://r/{i}") for i in range(10)]

        class _Client:
            def search(self, **kw):
                captured.update(kw)
                return _Resp()

        monkeypatch.setattr(pp, "_get_sync_client", lambda: _Client())
        out = pp.ParallelWebSearchProvider().search("q", limit=7)

        assert out["success"] is True
        # honors the caller's limit via advanced_settings.max_results
        assert captured["advanced_settings"] == {"max_results": 7}
        assert captured["mode"] == "advanced"            # v1 default
        assert captured["session_id"].startswith(f"{pp._MCP_CLIENT_NAME}-")  # per-call id
        assert len(out["data"]["web"]) == 7              # client-side slice
        # paid path: no free-tier attribution, no [Parallel] label signal
        assert "attribution" not in out
        assert "provider" not in out


# ─── v1 search mode mapping ──────────────────────────────────────────────────

class TestResolveSearchMode:
    @pytest.mark.parametrize("env,expected", [
        (None, "advanced"),        # default
        ("advanced", "advanced"),
        ("basic", "basic"),
        ("fast", "basic"),         # legacy → basic
        ("one-shot", "basic"),     # legacy → basic
        ("agentic", "advanced"),   # legacy → advanced
        ("garbage", "advanced"),   # invalid → default
        ("BASIC", "basic"),        # case-insensitive
    ])
    def test_mode_mapping(self, monkeypatch, env, expected):
        if env is None:
            monkeypatch.delenv("PARALLEL_SEARCH_MODE", raising=False)
        else:
            monkeypatch.setenv("PARALLEL_SEARCH_MODE", env)
        assert pp._resolve_search_mode() == expected
