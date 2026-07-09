"""Unit tests for the gbrain document leg (Phase 2b) — mapping, fail-open,
flag-off-is-inert, token caching, and the QMD-replacement gate derivation.

All HTTP is mocked; no live serve or mem0 needed. Run from the repo root:
  venv/bin/python -m pytest plugins/memory/mem0/test_gbrain_recall.py -v -o addopts=""
"""
import json
import threading
import time

import pytest

from plugins.memory.mem0 import gbrain_recall, qmd_recall
from plugins.memory.mem0 import Mem0MemoryProvider


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mcp_payload(rows):
    """Wrap gbrain `search` rows in the MCP tools/call response envelope."""
    return {
        "result": {
            "content": [{"type": "text", "text": json.dumps(rows)}],
        }
    }


_ROW = {
    "slug": "ai/dora-doorbell/dora-ai-doorbell",
    "page_id": 4214,
    "title": "Dora Ai Doorbell",
    "score": 0.8589794749628127,
    "chunk_text": "SECRET BODY MUST NOT LEAK",
}


@pytest.fixture(autouse=True)
def _clear_token_cache():
    with gbrain_recall._token_lock:
        gbrain_recall._token_cache.clear()
    yield
    with gbrain_recall._token_lock:
        gbrain_recall._token_cache.clear()


# ---------------------------------------------------------------------------
# parse_gbrain_results: mapping to the exact QMD pointer shape
# ---------------------------------------------------------------------------

def test_mapping_exact_pointer_shape():
    out = gbrain_recall.parse_gbrain_results(_mcp_payload([_ROW]), min_score=0.5)
    assert out == [{
        "file": "ai/dora-doorbell/dora-ai-doorbell",
        "title": "Dora Ai Doorbell",
        "score": 0.859,
        "line": 0,
        "docid": "gbrain:4214",
    }]
    # pointer-only contract: exactly the five QMD keys, no chunk_text leak
    assert set(out[0].keys()) == {"file", "title", "score", "line", "docid"}


def test_mapping_min_score_floor():
    low = dict(_ROW, score=0.2)
    assert gbrain_recall.parse_gbrain_results(_mcp_payload([low]), min_score=0.5) == []
    assert len(gbrain_recall.parse_gbrain_results(_mcp_payload([low]), min_score=0.1)) == 1


def test_mapping_drops_malformed_rows():
    rows = [
        "not-a-dict",
        {"slug": "", "score": 0.9},          # empty slug dropped
        {"slug": "ok/page", "score": "bad"},  # unparsable score -> 0.0 -> floored
        dict(_ROW, page_id=None),             # missing page_id -> "gbrain:"
    ]
    out = gbrain_recall.parse_gbrain_results(_mcp_payload(rows), min_score=0.5)
    assert len(out) == 1
    assert out[0]["docid"] == "gbrain:"


def test_mapping_error_and_garbage_payloads():
    assert gbrain_recall.parse_gbrain_results(None, 0.5) == []
    assert gbrain_recall.parse_gbrain_results({}, 0.5) == []
    assert gbrain_recall.parse_gbrain_results({"result": {"isError": True, "content": [{"text": "boom"}]}}, 0.5) == []
    assert gbrain_recall.parse_gbrain_results({"result": {"content": [{"text": "{not json"}]}}, 0.5) == []
    assert gbrain_recall.parse_gbrain_results({"result": {"content": [{"text": json.dumps({"a": 1})}]}}, 0.5) == []


def test_render_block_and_empty():
    hits = gbrain_recall.parse_gbrain_results(_mcp_payload([_ROW]), 0.5)
    block = gbrain_recall.render_gbrain_block(hits)
    assert block.startswith("## Local Docs (gbrain)")
    assert "ai/dora-doorbell/dora-ai-doorbell" in block and "86%" in block
    assert gbrain_recall.render_gbrain_block([]) == ""


# ---------------------------------------------------------------------------
# gbrain_search: fail-open + token caching (HTTP mocked)
# ---------------------------------------------------------------------------

def _mock_http(monkeypatch, handler, creds_ok=True):
    """Install an HTTP mock. handler(path, body_dict_or_str) -> (status, body_str)."""
    calls = {"token": 0, "mcp": 0}

    def fake_post(base_url, path, body, headers, timeout_s, state=None, lock=None):
        if path == "/token":
            calls["token"] += 1
        elif path == "/mcp":
            calls["mcp"] += 1
        return handler(path, body)

    monkeypatch.setattr(gbrain_recall, "_http_post", fake_post)
    if creds_ok:
        monkeypatch.setattr(
            gbrain_recall, "_read_client_creds",
            lambda p: {"client_id": "cid", "client_secret": "sec"},
        )
    else:
        monkeypatch.setattr(gbrain_recall, "_read_client_creds", lambda p: None)
    return calls


def _ok_handler(path, body):
    if path == "/token":
        return 200, json.dumps({"access_token": "tok-1", "expires_in": 3600})
    return 200, json.dumps(_mcp_payload([_ROW]))


def test_search_happy_path(monkeypatch):
    _mock_http(monkeypatch, _ok_handler)
    out = gbrain_recall.gbrain_search("doorbell", limit=3, min_score=0.5, deadline_s=4.0)
    assert out and out[0]["file"] == _ROW["slug"]


def test_token_cached_one_auth_per_process(monkeypatch):
    calls = _mock_http(monkeypatch, _ok_handler)
    for _ in range(3):
        gbrain_recall.gbrain_search("q", limit=3, min_score=0.5, deadline_s=4.0)
    assert calls["token"] == 1          # one mint, three searches
    assert calls["mcp"] == 3


def test_401_invalidates_and_retries_once(monkeypatch):
    seq = {"n": 0}

    def handler(path, body):
        if path == "/token":
            return 200, json.dumps({"access_token": f"tok-{seq['n']}", "expires_in": 3600})
        seq["n"] += 1
        if seq["n"] == 1:
            return 401, "expired"
        return 200, json.dumps(_mcp_payload([_ROW]))

    calls = _mock_http(monkeypatch, handler)
    out = gbrain_recall.gbrain_search("q", limit=3, min_score=0.5, deadline_s=4.0)
    assert out                     # recovered within the same call
    assert calls["token"] == 2     # re-minted exactly once


# fail-OPEN matrix: every failure mode returns [] and never raises
@pytest.mark.parametrize("handler", [
    lambda p, b: (500, "server error"),
    lambda p, b: (200, "not json at all"),
    lambda p, b: (_ for _ in ()).throw(ConnectionRefusedError("down")),
    lambda p, b: (200, json.dumps({"error": {"code": -32000, "message": "boom"}})),
])
def test_fail_open_on_any_error(monkeypatch, handler):
    def wrapped(path, body):
        if path == "/token":
            return 200, json.dumps({"access_token": "t", "expires_in": 3600})
        return handler(path, body)
    _mock_http(monkeypatch, wrapped)
    assert gbrain_recall.gbrain_search("q", limit=3, min_score=0.5, deadline_s=4.0) == []


def test_fail_open_no_creds(monkeypatch):
    calls = _mock_http(monkeypatch, _ok_handler, creds_ok=False)
    assert gbrain_recall.gbrain_search("q", limit=3, min_score=0.5, deadline_s=4.0) == []
    assert calls["mcp"] == 0  # never reached the tool call


def test_fail_open_token_mint_fails(monkeypatch):
    _mock_http(monkeypatch, lambda p, b: (403, "denied"))
    assert gbrain_recall.gbrain_search("q", limit=3, min_score=0.5, deadline_s=4.0) == []


def test_deadline_honored_slow_backend():
    """A backend that hangs longer than the deadline must return [] promptly —
    bounded by the deadline, not by the hang. Uses a REAL localhost socket so the
    watchdog's SHUT_RDWR actually fires (a monkeypatched _http_post can't be
    interrupted, which made the old assertion vacuous — Greptile #248)."""
    import socket
    import threading as _th

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    port = srv.getsockname()[1]
    stop = _th.Event()

    def _serve():
        while not stop.is_set():
            try:
                srv.settimeout(0.2)
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                req = conn.recv(65536).decode("utf-8", "replace")
                if "POST /token" in req:
                    body = json.dumps({"access_token": "t", "expires_in": 3600})
                    conn.sendall(
                        f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                    )
                    conn.close()
                else:
                    # Hang: never respond. The client-side watchdog must cut this.
                    stop.wait(10.0)
                    conn.close()
            except Exception:
                pass

    th = _th.Thread(target=_serve, daemon=True)
    th.start()
    try:
        t0 = time.monotonic()
        out = gbrain_recall.gbrain_search(
            "q", limit=3, min_score=0.5, deadline_s=0.5,
            url=f"http://127.0.0.1:{port}",
            creds_path=_write_creds_tmp(),
        )
        elapsed = time.monotonic() - t0
        assert out == []                      # fail-open, no partial junk
        assert elapsed < 2.0, f"deadline not enforced: {elapsed:.2f}s"  # 0.5s deadline + slack, NOT the 10s hang
    finally:
        stop.set()
        try:
            srv.close()
        except Exception:
            pass


def _write_creds_tmp():
    import tempfile
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
    f.write("GBRAIN_RAIL_CLIENT_ID=test\nGBRAIN_RAIL_CLIENT_SECRET=test\n")
    f.close()
    return f.name


def test_creds_file_parsing(tmp_path):
    f = tmp_path / "rail-client.env"
    f.write_text("# comment\nGBRAIN_RAIL_CLIENT_ID=abc\nGBRAIN_RAIL_CLIENT_SECRET='s3c'\n")
    creds = gbrain_recall._read_client_creds(str(f))
    assert creds == {"client_id": "abc", "client_secret": "s3c"}
    assert gbrain_recall._read_client_creds(str(tmp_path / "missing.env")) is None
    g = tmp_path / "partial.env"
    g.write_text("GBRAIN_RAIL_CLIENT_ID=abc\n")
    assert gbrain_recall._read_client_creds(str(g)) is None


# ---------------------------------------------------------------------------
# load_gbrain_config + provider gate derivation (flag off => inert)
# ---------------------------------------------------------------------------

def test_config_defaults_flag_off():
    cfg = gbrain_recall.load_gbrain_config(None)
    assert cfg["enabled"] is False
    cfg2 = gbrain_recall.load_gbrain_config({"unknown_key": 1, "min_score": None})
    assert cfg2 == gbrain_recall.GBRAIN_DEFAULTS  # unknown keys ignored, None ignored


def _provider(qmd_enabled=False, gbrain_enabled=False, mem0_rows=None):
    """Minimal provider with initialize()'s gate derivation replicated via the
    REAL initialize() config path (config dict only; network calls stubbed)."""
    p = Mem0MemoryProvider()
    p._config = {}
    p._rerank = False
    p._keyword_search = None
    p._temporal_search = False
    p._consecutive_failures = 0
    p._breaker_open_until = 0
    p._qmd_cfg = qmd_recall.load_qmd_config({"enabled": qmd_enabled})
    p._qmd_enabled = qmd_enabled
    p._qmd_prefetch_enabled = qmd_enabled
    p._qmd_search_enabled = qmd_enabled
    p._gbrain_cfg = gbrain_recall.load_gbrain_config({"enabled": gbrain_enabled})
    p._gbrain_enabled = gbrain_enabled
    p._gbrain_prefetch_enabled = gbrain_enabled
    p._gbrain_search_enabled = gbrain_enabled
    if p._gbrain_prefetch_enabled:
        p._qmd_prefetch_enabled = False
    if p._gbrain_search_enabled:
        p._qmd_search_enabled = False

    class _Stub:
        def search(self, **kw):
            return list(mem0_rows or [])
    p._get_client = lambda: _Stub()
    p._drop_forgotten = lambda rows: rows
    p._read_filters = lambda: {}
    return p


def _run_prefetch(p, query):
    p.queue_prefetch(query)
    if p._prefetch_future:
        p._prefetch_future.result(timeout=5)
    return p.prefetch(query)


def test_flag_off_is_inert_no_gbrain_call(monkeypatch):
    """gbrain disabled (default): _gbrain_pointers is a no-op, gbrain_search is
    never invoked, and prefetch output is byte-identical to today."""
    called = {"n": 0}
    def spy(*a, **k):
        called["n"] += 1
        return [{"file": "x", "title": "x", "score": 1.0, "line": 0, "docid": "gbrain:1"}]
    monkeypatch.setattr(gbrain_recall, "gbrain_search", spy)

    p = _provider(qmd_enabled=False, gbrain_enabled=False,
                  mem0_rows=[{"memory": "fact one"}])
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert called["n"] == 0
    assert out == "## Mem0 Memory\n- fact one"  # exactly the legacy render
    # explicit tool path also inert
    assert p._gbrain_pointers("q", limit=3, deadline_s=1.0) == []
    reply = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert "docs" not in reply


def test_gbrain_replaces_qmd_prefetch(monkeypatch):
    """gbrain on + qmd on: only the gbrain leg fires (one retrieval leg per turn)."""
    qmd_calls = {"n": 0}
    monkeypatch.setattr(qmd_recall, "qmd_query",
                        lambda *a, **k: qmd_calls.__setitem__("n", qmd_calls["n"] + 1) or [])
    monkeypatch.setattr(
        gbrain_recall, "gbrain_search",
        lambda *a, **k: [{"file": "ai/dora-doorbell/dora-ai-doorbell",
                          "title": "Dora Ai Doorbell", "score": 0.859,
                          "line": 0, "docid": "gbrain:4214"}],
    )
    p = _provider(qmd_enabled=True, gbrain_enabled=True,
                  mem0_rows=[{"memory": "fact one"}])
    assert p._qmd_prefetch_enabled is False   # gate derivation: QMD lane superseded
    assert p._qmd_search_enabled is False
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert qmd_calls["n"] == 0                 # QMD never called
    assert "## Local Docs (gbrain)" in out
    assert "ai/dora-doorbell/dora-ai-doorbell" in out
    assert "## Mem0 Memory\n- fact one" in out
    assert out.index("## Mem0 Memory") < out.index("## Local Docs")  # mem0 first


def test_gbrain_search_docs_key(monkeypatch):
    monkeypatch.setattr(
        gbrain_recall, "gbrain_search",
        lambda *a, **k: [{"file": "s/p", "title": "T", "score": 0.9, "line": 0,
                          "docid": "gbrain:7"}],
    )
    p = _provider(gbrain_enabled=True, mem0_rows=[{"memory": "m", "score": 0.9}])
    reply = json.loads(p.handle_tool_call("mem0_search", {"query": "local dns split"}))
    assert reply["docs"][0]["file"] == "s/p"
    assert reply["results"][0]["memory"] == "m"


def test_gbrain_down_keeps_mem0_block(monkeypatch):
    """Fail-open at the provider level: gbrain returning [] never drops mem0."""
    monkeypatch.setattr(gbrain_recall, "gbrain_search", lambda *a, **k: [])
    p = _provider(gbrain_enabled=True, mem0_rows=[{"memory": "fact one"}])
    out = _run_prefetch(p, "where did we decide the local dns split")
    assert out == "## Mem0 Memory\n- fact one"


def test_non_lookup_skips_gbrain(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(gbrain_recall, "gbrain_search",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [])
    p = _provider(gbrain_enabled=True, mem0_rows=[{"memory": "fact one"}])
    _run_prefetch(p, "ship it")
    assert called["n"] == 0
