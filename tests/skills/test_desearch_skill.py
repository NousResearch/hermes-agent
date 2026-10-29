"""Offline tests for optional-skills/research/desearch/scripts/desearch.py.

The HTTP opener is replaced with a stub, so no network call is made and no
real API key is needed.
"""
from __future__ import annotations

import io
import json
import runpy
import urllib.error
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/research/desearch/scripts/desearch.py"
)


@pytest.fixture
def mod():
    return runpy.run_path(str(SCRIPT))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(body, captured):
    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        raw = body if isinstance(body, str) else json.dumps(body)
        return _Response(raw.encode())
    return opener


def _tool_result(text, is_error=False):
    return {"jsonrpc": "2.0", "id": 1,
            "result": {"content": [{"type": "text", "text": text}], "isError": is_error}}


def test_missing_key_is_reported(mod, monkeypatch):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    with pytest.raises(mod["DesearchError"], match="console.desearch.ai/api-keys"):
        mod["call_tool"]("web-search", {"query": "q"}, opener=_opener({}, {}))


def test_call_uses_bearer_and_hosted_endpoint(mod, monkeypatch):
    monkeypatch.setenv("DESEARCH_API_KEY", "test-only-key")
    seen = {}
    out = mod["call_tool"]("web-search", {"query": "q"},
                           opener=_opener(_tool_result('{"data": []}'), seen))
    assert out == '{"data": []}'
    assert seen["url"] == "https://mcp.desearch.ai/mcp"
    assert seen["headers"]["authorization"] == "Bearer test-only-key"
    assert "x-api-key" not in seen["headers"]
    assert seen["body"]["method"] == "tools/call"
    assert seen["body"]["params"] == {"name": "web-search", "arguments": {"query": "q"}}


def test_event_stream_body_is_parsed(mod, monkeypatch):
    monkeypatch.setenv("DESEARCH_API_KEY", "k")
    sse = "event: message\ndata: " + json.dumps(_tool_result("ok")) + "\n\n"
    assert mod["call_tool"]("web-search", {"query": "q"}, opener=_opener(sse, {})) == "ok"


def test_tool_error_and_rpc_error_raise(mod, monkeypatch):
    monkeypatch.setenv("DESEARCH_API_KEY", "k")
    with pytest.raises(mod["DesearchError"], match="web-search failed"):
        mod["call_tool"]("web-search", {"query": "q"},
                         opener=_opener(_tool_result("bad", is_error=True), {}))
    rpc_error = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Invalid params"}}
    with pytest.raises(mod["DesearchError"], match="-32602"):
        mod["call_tool"]("web-search", {}, opener=_opener(rpc_error, {}))


def test_http_401_points_at_the_key(mod, monkeypatch):
    monkeypatch.setenv("DESEARCH_API_KEY", "k")

    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"no"))

    with pytest.raises(mod["DesearchError"], match="DESEARCH_API_KEY"):
        mod["call_tool"]("web-search", {"query": "q"}, opener=opener)


@pytest.mark.parametrize("argv,tool,arguments", [
    (["web", "q", "--start", "10"], "web-search", {"query": "q", "start": 10}),
    (["ai", "p", "--sources", "reddit, hackernews", "--date-filter", "PAST_WEEK", "--links-only"],
     "ai-search", {"prompt": "p", "tools": ["reddit", "hackernews"],
                   "date_filter": "PAST_WEEK", "result_type": "ONLY_LINKS"}),
    (["x", "q", "--user", "u", "--since", "2026-01-01"], "x-search",
     {"query": "q", "count": 10, "user": "u", "start_date": "2026-01-01"}),
    (["links", "p", "--count", "20"], "web-links-search", {"prompt": "p", "count": 20}),
    (["extract", "https://example.com"], "extract", {"url": "https://example.com", "format": "text"}),
    (["call", "x-trends", '{"woeid": 1}'], "x-trends", {"woeid": 1}),
])
def test_subcommands_map_to_mcp_tools(mod, argv, tool, arguments):
    assert mod["build_call"](mod["parser"]().parse_args(argv)) == (tool, arguments)


def test_call_rejects_non_object_json(mod):
    args = mod["parser"]().parse_args(["call", "x-trends", "[1]"])
    with pytest.raises(mod["DesearchError"], match="JSON object"):
        mod["build_call"](args)


def test_main_exits_1_without_key(mod, monkeypatch, capsys):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    assert mod["main"](["web", "q"]) == 1
    assert "DESEARCH_API_KEY" in capsys.readouterr().err
