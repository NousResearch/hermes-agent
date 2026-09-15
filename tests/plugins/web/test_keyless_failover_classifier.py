"""Regression tests for the keyless ring failover classifier.

Covers the 2026-08-31 live failure: Exa's anonymous MCP endpoint sometimes
answers HTTP 200 with a body that doesn't parse as JSON-RPC (rate-limit
pages, HTML error pages). The vendor wrapper turns that into the error
string "Keyless Exa search failed: Unrecognized MCP response shape. ...",
which carries none of the classic rate-limit markers — so the ring used to
treat it as a hard, vendor-independent failure and never tried the next
free provider. These tests pin the fix: shape failures are *retryable
vendor failures*, so the walk advances; auth/config errors keep failing
fast.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]

# Load by file path: the runtime env's editable-install finder claims the
# top-level "plugins" name, so `from plugins.web import keyless_mcp` is
# unreliable under pytest. Direct loading sidesteps package resolution.
_spec = importlib.util.spec_from_file_location(
    "keyless_mcp_under_test", REPO / "plugins" / "web" / "keyless_mcp.py"
)
assert _spec is not None and _spec.loader is not None
km = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(km)


def _make_ring(monkeypatch, behavior):
    """Replace both vendor registries with scripted fakes.

    *behavior*: {vendor: callable(arg) -> result} keyed by vendor name;
    unscripted vendors return success. Returns the call log — every vendor
    invocation is recorded in wrapper order.
    """
    calls: list = []

    def fake_searcher(vendor):
        def _search(query, limit=5):
            calls.append(vendor)
            fn = behavior.get(vendor)
            if fn is None:
                return {"success": True, "data": {"web": []}}
            return fn(query)

        return _search

    def fake_extractor(vendor):
        def _extract(urls):
            calls.append(vendor)
            fn = behavior.get(vendor)
            if fn is None:
                return [{"url": u, "title": "t", "content": "c"} for u in urls]
            return fn(urls)

        return _extract

    searchers = {v: fake_searcher(v) for v in km._KEYLESS_RING}
    extractors = {v: fake_extractor(v) for v in km._KEYLESS_RING}
    monkeypatch.setattr(km, "_KEYLESS_SEARCHERS", searchers)
    monkeypatch.setattr(km, "_KEYLESS_EXTRACTORS", extractors)
    return calls


@pytest.fixture(autouse=True)
def _stable_ring(monkeypatch):
    """Pin the walk to ring order starting at exa; keep config out."""
    monkeypatch.setattr(km, "_ring_order", lambda name=None: list(km._KEYLESS_RING))
    yield


SHAPE_ERROR = (
    "Keyless Exa search failed: Unrecognized MCP response shape. "
    "Set EXA_API_KEY (https://exa.ai) or another web backend via "
    "`hermes tools` for reliable service."
)


class TestSearchFailover:
    def test_masked_429_shape_failure_advances_to_next_vendor(self, monkeypatch):
        """THE regression: exa returns a masked rate-limit (shape failure);
        the ring must move on to parallel instead of returning the error."""
        def exa_fail(query):
            return {"success": False, "error": SHAPE_ERROR}

        def parallel_ok(query):
            return {"success": True, "data": {"web": [{"url": "https://x", "title": "t"}]}}

        calls = _make_ring(monkeypatch, {"exa": exa_fail, "parallel": parallel_ok})

        result = km.search_with_failover("exa", "obscura rust cdp")

        assert calls == ["exa", "parallel"], "ring did not advance past exa"
        assert result["success"] is True
        assert result["data"]["served_by"] == "parallel"

    def test_explicit_rate_limit_still_fails_over(self, monkeypatch):
        """Classic 429s advance the ring; if every vendor throttles, the
        walk exhausts with the aggregate error."""
        def throttled(query):
            return {"success": False, "error": "search failed: HTTP 429: slow down"}

        calls = _make_ring(monkeypatch, {v: throttled for v in km._KEYLESS_RING})

        result = km.search_with_failover("exa", "q")

        assert calls == list(km._KEYLESS_RING), "walk did not cover the ring"
        assert result["success"] is False
        assert "all keyless vendors throttled" in result["error"]

    def test_rate_limit_failover_serves_from_next_vendor(self, monkeypatch):
        def exa_429(query):
            return {"success": False, "error": "Keyless Exa search failed: HTTP 429: slow down"}

        calls = _make_ring(monkeypatch, {"exa": exa_429})

        result = km.search_with_failover("exa", "q")

        assert calls == ["exa", "parallel"]
        assert result["success"] is True
        assert result["data"]["served_by"] == "parallel"

    def test_auth_error_still_fails_fast(self, monkeypatch):
        def exa_auth(query):
            return {"success": False, "error": "Keyless Exa search failed: HTTP 401: unauthorized"}

        calls = _make_ring(monkeypatch, {"exa": exa_auth})

        result = km.search_with_failover("exa", "q")

        assert calls == ["exa"], "auth error must not advance the ring"
        assert result["success"] is False
        assert "401" in result["error"]


class TestExtractFailover:
    def test_masked_429_shape_failure_batch_advances(self, monkeypatch):
        def exa_fail(urls):
            return [
                {"url": u, "title": "", "content": "", "error": SHAPE_ERROR}
                for u in urls
            ]

        def parallel_ok(urls):
            return [{"url": u, "title": "t", "content": "c"} for u in urls]

        calls = _make_ring(monkeypatch, {"exa": exa_fail, "parallel": parallel_ok})

        results = km.extract_with_failover(
            "exa", ["https://raw.githubusercontent.com/x/y/main/README.md"]
        )

        assert calls == ["exa", "parallel"], "extract ring did not advance"
        assert all(r.get("error") is None for r in results)

    def test_extract_mixed_batch_retries_only_retryable_urls(self, monkeypatch):
        """THE second blind spot (live 2026-08-31): a mixed batch where one
        URL masked-throttles and another succeeds must re-fetch ONLY the
        retryable URL on the next vendor, and return results in input
        order with the hard error preserved verbatim."""
        shape_url = "https://raw.githubusercontent.com/x/y/main/README.md"
        ok_url = "https://example.com"

        def exa_mixed(urls):
            out = []
            for u in urls:
                if u == shape_url:
                    out.append({"url": u, "title": "", "content": "", "error": SHAPE_ERROR})
                else:
                    out.append({"url": u, "title": "Example", "content": "body"})
            return out

        def parallel_rescues(urls):
            assert urls == [shape_url], "only the retryable URL should be re-fetched"
            return [{"url": u, "title": "README", "content": "x" * 50} for u in urls]

        calls = _make_ring(monkeypatch, {"exa": exa_mixed, "parallel": parallel_rescues})

        results = km.extract_with_failover("exa", [shape_url, ok_url])

        assert calls == ["exa", "parallel"]
        assert len(results) == 2 and results[0]["url"] == shape_url  # order kept
        assert results[0].get("error") is None and results[0]["served_by"] == "parallel"
        assert results[1]["content"] == "body" and "served_by" not in results[1]

    def test_extract_retryable_url_exhausting_all_vendors_keeps_last_error(self, monkeypatch):
        def throttled(urls):
            return [{"url": u, "title": "", "content": "", "error": "HTTP 429: slow down"} for u in urls]

        calls = _make_ring(monkeypatch, {v: throttled for v in km._KEYLESS_RING})

        results = km.extract_with_failover("exa", ["https://a.example"])

        assert calls == list(km._KEYLESS_RING)
        assert len(results) == 1 and results[0]["error"] == "HTTP 429: slow down"

    def test_extract_notes_served_by_on_failover(self, monkeypatch):
        """Observability: search results carry served_by on failover; extract
        results must too, or the fallback is silent."""

        def exa_fail(urls):
            return [
                {"url": u, "title": "", "content": "", "error": SHAPE_ERROR}
                for u in urls
            ]

        def parallel_ok(urls):
            return [{"url": u, "title": "t", "content": "c"} for u in urls]

        _make_ring(monkeypatch, {"exa": exa_fail, "parallel": parallel_ok})

        results = km.extract_with_failover("exa", ["https://example.com"])

        assert results and all(r.get("served_by") == "parallel" for r in results)
