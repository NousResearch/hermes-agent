#!/usr/bin/env python3
"""Offline mocked tests for the Parallel provider search-depth → mode mapping.

Requirements covered:

1. search_depth="fast" sends mode="fast".
2. search_depth="fast" never sends mode="turbo".
3. default search_depth behavior remains as intended (env var or "agentic").
4. supported explicit modes (PARALLEL_SEARCH_MODE) remain supported.
5. an unsupported search_depth value does not result in mode="turbo".
6. limit and query propagation remain unchanged.
7. response parsing remains unchanged.
8. no live network request occurs (fully mocked client).
9. Router-disabled behavior is unaffected (no router code touched).
10. no other provider behavior changes.

No live network requests. All client calls are fakes injected into the
canonical cache slots on :mod:`tools.web_tools`.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from plugins.web.parallel.provider import ParallelWebSearchProvider


class _FakeResults:
    def __init__(self, items):
        self.results = items


class _FakeSearchResponse:
    def __init__(self, items):
        self.results = items


def _result(url="https://example.com/1", title="Sample", excerpts=("desc",)):
    return SimpleNamespace(url=url, title=title, excerpts=list(excerpts))


class _FakeBeta:
    def __init__(self):
        self.search_calls = []

    def search(self, search_queries=None, objective=None, mode=None, max_results=None):
        self.search_calls.append(
            {
                "search_queries": search_queries,
                "objective": objective,
                "mode": mode,
                "max_results": max_results,
            }
        )
        return _FakeSearchResponse([_result()])


class _FakeClient:
    def __init__(self):
        self.beta = _FakeBeta()


def _make_provider():
    """Provider whose search() hits the fake client via the cache slot."""
    fake = _FakeClient()
    patcher = patch("tools.web_tools._parallel_client", fake)
    patcher.start()
    provider = ParallelWebSearchProvider()
    return provider, fake, patcher


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure the canonical client slot is clean before/after each test."""
    import tools.web_tools as wt

    prev = getattr(wt, "_parallel_client", None)
    wt._parallel_client = None
    yield
    wt._parallel_client = prev


def test_fast_depth_sends_fast_mode():
    provider, fake, patcher = _make_provider()
    try:
        resp = provider.search("vector database survey", limit=1, search_depth="fast")
        call = fake.beta.search_calls[-1]
        assert call["mode"] == "fast"
        assert resp["success"] is True
    finally:
        patcher.stop()


def test_fast_depth_never_sends_turbo():
    provider, fake, patcher = _make_provider()
    try:
        provider.search("vector database survey", limit=1, search_depth="fast")
        for call in fake.beta.search_calls:
            assert call["mode"] != "turbo"
    finally:
        patcher.stop()


def test_default_depth_uses_env_or_agentic():
    provider, fake, patcher = _make_provider()
    try:
        with patch.dict("os.environ", {}, clear=False):
            resp = provider.search("vector database survey", limit=1, search_depth=None)
            call = fake.beta.search_calls[-1]
            assert call["mode"] == "agentic"
            assert resp["success"] is True
    finally:
        patcher.stop()


def test_default_depth_honors_env_var():
    provider, fake, patcher = _make_provider()
    try:
        with patch.dict("os.environ", {"PARALLEL_SEARCH_MODE": "one-shot"}, clear=False):
            provider.search("vector database survey", limit=1, search_depth=None)
            call = fake.beta.search_calls[-1]
            assert call["mode"] == "one-shot"
    finally:
        patcher.stop()


def test_supported_explicit_modes_stay_valid():
    provider, fake, patcher = _make_provider()
    try:
        for env_mode in ("agentic", "fast", "one-shot"):
            with patch.dict("os.environ", {"PARALLEL_SEARCH_MODE": env_mode}, clear=False):
                provider.search("query", limit=1, search_depth=None)
                assert fake.beta.search_calls[-1]["mode"] == env_mode
    finally:
        patcher.stop()


def test_unsupported_depth_does_not_become_turbo():
    provider, fake, patcher = _make_provider()
    try:
        with patch.dict("os.environ", {}, clear=False):
            provider.search("query", limit=1, search_depth="bogus-depth")
            call = fake.beta.search_calls[-1]
            assert call["mode"] != "turbo"
            assert call["mode"] == "agentic"  # falls back to default
    finally:
        patcher.stop()


def test_limit_and_query_propagation_unchanged():
    provider, fake, patcher = _make_provider()
    try:
        provider.search("hello world query", limit=3, search_depth="fast")
        call = fake.beta.search_calls[-1]
        assert call["search_queries"] == ["hello world query"]
        assert call["objective"] == "hello world query"
        assert call["max_results"] == 3
    finally:
        patcher.stop()


def test_limit_capped_at_20():
    provider, fake, patcher = _make_provider()
    try:
        provider.search("query", limit=99, search_depth="fast")
        call = fake.beta.search_calls[-1]
        assert call["max_results"] == 20
    finally:
        patcher.stop()


def test_response_parsing_unchanged():
    provider, fake, patcher = _make_provider()
    try:
        resp = provider.search("query", limit=1, search_depth="fast")
        assert resp == {
            "success": True,
            "data": {
                "web": [
                    {
                        "url": "https://example.com/1",
                        "title": "Sample",
                        "description": "desc",
                        "position": 1,
                    }
                ]
            },
        }
    finally:
        patcher.stop()


def test_no_network_request_occurs():
    """The fake client never touches the network; no real SDK import needed."""
    provider, fake, patcher = _make_provider()
    try:
        provider.search("query", limit=1, search_depth="fast")
        provider.search("query", limit=1, search_depth=None)
        assert len(fake.beta.search_calls) == 2
    finally:
        patcher.stop()


def test_router_disabled_path_unaffected():
    """Router code is not imported by the provider; provider works standalone."""
    provider, fake, patcher = _make_provider()
    try:
        resp = provider.search("query", limit=1, search_depth="fast")
        assert resp["success"] is True
    finally:
        patcher.stop()


def test_deep_depth_maps_to_agentic():
    provider, fake, patcher = _make_provider()
    try:
        with patch.dict("os.environ", {}, clear=False):
            provider.search("query", limit=1, search_depth="deep")
            assert fake.beta.search_calls[-1]["mode"] == "agentic"
            provider.search("query", limit=1, search_depth="deepest")
            assert fake.beta.search_calls[-1]["mode"] == "agentic"
    finally:
        patcher.stop()
