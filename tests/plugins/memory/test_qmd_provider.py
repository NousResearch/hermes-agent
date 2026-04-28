"""Tests for the QMD memory provider plugin.

Uses ``httpx.MockTransport`` (no external mocking lib required) by
patching the provider's ``_get_client`` factory to return a client
backed by the test transport.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import pytest

from plugins.memory.qmd import (
    DEFAULT_BASE_URL,
    DEFAULT_INDEX,
    QMDMemoryProvider,
    SEARCH_SCHEMA,
    STATUS_SCHEMA,
    _BREAKER_COOLDOWN_SECS,
    _BREAKER_THRESHOLD,
    _collection_from_file,
    _load_config,
    _truncate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_payload(results=None):
    return {
        "ok": True,
        "results": results if results is not None else [
            {
                "docid": "doc-1",
                "file": "qmd://session-logs/2025/01/15.md",
                "title": "January planning",
                "score": 0.91,
                "externalRerankScore": 0.88,
                "context": "Plans for January.",
                "snippet": "We agreed to ship the QMD plugin upstream.",
            },
        ],
    }


def _health_payload():
    return {
        "ok": True,
        "uptime_s": 12345,
        "indexes": {
            "default": {"docs": 4321, "dim": 2560},
        },
        "rerank_url": "http://localhost:8010/v1/rerank",
    }


def _make_provider(monkeypatch, tmp_path, *, search_handler=None,
                   health_handler=None):
    """Build a QMDMemoryProvider whose httpx.Client uses a MockTransport."""
    # Isolate $HERMES_HOME
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("QMD_REMOTE_API_TOKEN", "test-token")
    monkeypatch.setenv("QMD_REMOTE_API_BASE_URL", "http://localhost:18181")
    monkeypatch.delenv("QMD_DEFAULT_INDEX", raising=False)
    monkeypatch.delenv("QMD_TIMEOUT", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search" and request.method == "POST":
            return search_handler(request) if search_handler \
                else httpx.Response(200, json=_search_payload())
        if request.url.path == "/health" and request.method == "GET":
            return health_handler(request) if health_handler \
                else httpx.Response(200, json=_health_payload())
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    p = QMDMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    # Inject the mock transport: replace cached client construction.
    real_get_client = p._get_client
    cached: Dict[str, httpx.Client] = {}

    def patched_get_client():
        if "c" not in cached:
            cached["c"] = httpx.Client(
                base_url=p._base_url,
                timeout=p._timeout,
                transport=transport,
            )
        p._client = cached["c"]
        return cached["c"]

    p._get_client = patched_get_client  # type: ignore
    return p


@pytest.fixture
def provider(monkeypatch, tmp_path):
    p = _make_provider(monkeypatch, tmp_path)
    yield p
    p.shutdown()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_truncate_short_string_unchanged():
    assert _truncate("hi", 10) == "hi"


def test_truncate_long_string_appends_ellipsis():
    out = _truncate("a" * 50, 10)
    assert out.endswith("…")
    assert len(out) == 10


def test_collection_from_file_extracts_segment():
    assert _collection_from_file("qmd://session-logs/foo/bar.md") == "session-logs"


def test_collection_from_file_returns_none_on_non_qmd():
    assert _collection_from_file("file:///etc/hosts") is None
    assert _collection_from_file("") is None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_load_config_env_only(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("QMD_REMOTE_API_TOKEN", "tok-from-env")
    monkeypatch.setenv("QMD_REMOTE_API_BASE_URL", "http://example.test:9999")
    monkeypatch.delenv("QMD_DEFAULT_INDEX", raising=False)
    cfg = _load_config()
    assert cfg["api_token"] == "tok-from-env"
    assert cfg["base_url"] == "http://example.test:9999"
    assert cfg["default_index"] == DEFAULT_INDEX


def test_load_config_json_overrides_non_secret(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("QMD_REMOTE_API_TOKEN", "tok-from-env")
    (tmp_path / "qmd.json").write_text(json.dumps({
        "base_url": "http://from-json:1234",
        "default_index": "json-index",
        "api_token": "MUST-BE-IGNORED",
    }))
    cfg = _load_config()
    assert cfg["api_token"] == "tok-from-env"   # never from JSON
    assert cfg["base_url"] == "http://from-json:1234"
    assert cfg["default_index"] == "json-index"


def test_save_config_drops_secret(tmp_path):
    p = QMDMemoryProvider()
    p.save_config(
        {"api_token": "secret", "base_url": "http://x", "timeout": 99},
        str(tmp_path),
    )
    saved = json.loads((tmp_path / "qmd.json").read_text())
    assert "api_token" not in saved
    assert saved["base_url"] == "http://x"
    assert saved["timeout"] == 99


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_is_available_false_without_token(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("QMD_REMOTE_API_TOKEN", raising=False)
    assert QMDMemoryProvider().is_available() is False


def test_is_available_true_with_token(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("QMD_REMOTE_API_TOKEN", "x")
    assert QMDMemoryProvider().is_available() is True


# ---------------------------------------------------------------------------
# Search — happy path / error / breaker
# ---------------------------------------------------------------------------

def test_qmd_search_happy_path(provider):
    raw = provider.handle_tool_call("qmd_search", {"query": "January plans"})
    out = json.loads(raw)
    assert out["count"] == 1
    assert out["results"][0]["title"] == "January planning"
    assert out["results"][0]["rerank_score"] == 0.88
    assert out["query"] == "January plans"


def test_qmd_search_collection_filter(monkeypatch, tmp_path):
    payload = _search_payload(results=[
        {"docid": "1", "file": "qmd://session-logs/a.md", "title": "A",
         "score": 0.9, "snippet": "x", "context": "x"},
        {"docid": "2", "file": "qmd://global-facts/b.md", "title": "B",
         "score": 0.8, "snippet": "y", "context": "y"},
    ])

    def search_handler(req):
        return httpx.Response(200, json=payload)

    p = _make_provider(monkeypatch, tmp_path, search_handler=search_handler)
    try:
        raw = p.handle_tool_call(
            "qmd_search",
            {"query": "...", "collection_filter": "session-logs"},
        )
        out = json.loads(raw)
        assert out["count"] == 1
        assert out["results"][0]["docid"] == "1"
    finally:
        p.shutdown()


def test_qmd_search_4xx_returns_error(monkeypatch, tmp_path):
    def search_handler(req):
        return httpx.Response(401, text="bad token")

    p = _make_provider(monkeypatch, tmp_path, search_handler=search_handler)
    try:
        raw = p.handle_tool_call("qmd_search", {"query": "x"})
        out = json.loads(raw)
        assert "error" in out
    finally:
        p.shutdown()


def test_qmd_search_circuit_breaker_trips(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def search_handler(req):
        call_count["n"] += 1
        return httpx.Response(503, text="down")

    p = _make_provider(monkeypatch, tmp_path, search_handler=search_handler)
    try:
        for _ in range(_BREAKER_THRESHOLD):
            p.handle_tool_call("qmd_search", {"query": "x"})
        # Breaker should now be open. Next call short-circuits.
        before = call_count["n"]
        raw = p.handle_tool_call("qmd_search", {"query": "x"})
        out = json.loads(raw)
        assert "error" in out
        assert "temporarily unavailable" in out["error"]
        assert call_count["n"] == before  # no extra HTTP request
    finally:
        p.shutdown()


def test_qmd_search_missing_query(provider):
    out = json.loads(provider.handle_tool_call("qmd_search", {}))
    assert "error" in out


def test_handle_tool_call_unknown_tool_errors(provider):
    out = json.loads(provider.handle_tool_call("qmd_nope", {}))
    assert "error" in out


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_qmd_status_happy_path(provider):
    raw = provider.handle_tool_call("qmd_status", {})
    out = json.loads(raw)
    assert out["uptime_s"] == 12345
    assert "default" in out["indexes"]


def test_qmd_status_5xx_returns_error(monkeypatch, tmp_path):
    def health_handler(req):
        return httpx.Response(500, text="boom")

    p = _make_provider(monkeypatch, tmp_path, health_handler=health_handler)
    try:
        out = json.loads(p.handle_tool_call("qmd_status", {}))
        assert "error" in out
    finally:
        p.shutdown()


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------

def test_queue_prefetch_populates_result(provider):
    provider.queue_prefetch("a long enough query to trigger prefetch")
    # join the background thread
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=2)
    block = provider.prefetch("anything")
    assert "QMD Memory" in block
    assert "January planning" in block


def test_queue_prefetch_skips_short_query(provider):
    provider.queue_prefetch("hi")
    assert provider._prefetch_thread is None


def test_queue_prefetch_skips_when_thread_alive(provider):
    """Overlapping prefetches must not spawn a second thread."""
    started = threading.Event()
    release = threading.Event()

    def _slow_search(*_, **__):
        started.set()
        release.wait(timeout=2)
        return []

    provider._search_remote = _slow_search    # type: ignore
    provider.queue_prefetch("a long enough query to trigger prefetch")
    assert started.wait(timeout=1)
    first = provider._prefetch_thread

    provider.queue_prefetch("another long enough query to trigger prefetch")
    assert provider._prefetch_thread is first   # second call no-ops

    release.set()
    if first:
        first.join(timeout=2)


# ---------------------------------------------------------------------------
# Schema completeness
# ---------------------------------------------------------------------------

def test_search_schema_required_fields():
    assert SEARCH_SCHEMA["name"] == "qmd_search"
    assert "query" in SEARCH_SCHEMA["parameters"]["required"]
    props = SEARCH_SCHEMA["parameters"]["properties"]
    for key in ("query", "top_k", "index", "collection_filter"):
        assert key in props, f"missing property: {key}"


def test_status_schema_no_params():
    assert STATUS_SCHEMA["name"] == "qmd_status"
    assert STATUS_SCHEMA["parameters"]["required"] == []


def test_get_tool_schemas_lists_both(provider):
    names = {s["name"] for s in provider.get_tool_schemas()}
    assert names == {"qmd_search", "qmd_status"}


def test_get_config_schema_has_token_marked_secret():
    p = QMDMemoryProvider()
    schema = p.get_config_schema()
    keys = {f["key"] for f in schema}
    assert "api_token" in keys
    token_field = next(f for f in schema if f["key"] == "api_token")
    assert token_field.get("secret") is True
    assert token_field.get("required") is True


# ---------------------------------------------------------------------------
# System prompt block
# ---------------------------------------------------------------------------

def test_system_prompt_block_mentions_index_and_host(provider):
    block = provider.system_prompt_block()
    assert provider._default_index in block
    assert provider._base_url in block
    assert "Read-only" in block


# ---------------------------------------------------------------------------
# sync_turn — read-only no-op
# ---------------------------------------------------------------------------

def test_sync_turn_is_noop(provider):
    # Should not raise, should not record anything.
    provider.sync_turn("user msg", "assistant reply", session_id="s1")


# ---------------------------------------------------------------------------
# Shutdown idempotence
# ---------------------------------------------------------------------------

def test_shutdown_is_idempotent(provider):
    provider.shutdown()
    provider.shutdown()    # second call must not raise
