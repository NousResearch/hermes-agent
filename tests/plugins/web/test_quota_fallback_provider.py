"""Tests for the quota-fallback web search provider.

Covers:
- First-success short-circuit (Tavily returns results → return).
- Quota/billing errors → fallback to next provider.
- Empty results + ``empty_results_fallback=true`` → fallback.
- Rate limit on Exa → fallback to Baidu.
- All providers fail → fallback to SearXNG.
- All providers fail → aggregated error with per-provider details.
- Cooldown state skips exhausted providers.
- Corrupted state file does not crash.
- Empty query returns error directly.
- CJK detection and language-aware routing.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from agent.web_search_provider import WebSearchProvider
from plugins.web.quota_fallback.provider import (
    QuotaFallbackWebSearchProvider,
    _DEFAULT_CJK,
    _DEFAULT_LATIN,
    _has_cjk,
)
from plugins.web.quota_fallback import provider as fb_provider

# ---------------------------------------------------------------------------
# Mock child providers
# ---------------------------------------------------------------------------


class MockProvider:
    """Duck-typed child provider for test use."""

    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        supports_search: bool = True,
        search_result: Dict[str, Any] = None,
    ):
        self._name = name
        self._available = available
        self._supports_search = supports_search
        self._search_result = search_result or {"success": True, "data": {"web": []}}

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def supports_search(self) -> bool:
        return self._supports_search

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        return self._search_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the state file to a temp path and ensure it's clean."""
    state_file = tmp_path / "web-fallback-state.json"
    state_file.write_text("{}")
    monkeypatch.setattr(
        "plugins.web.quota_fallback.provider._state_path",
        lambda: state_file,
    )


@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch: pytest.MonkeyPatch) -> Dict[str, MockProvider]:
    """Replace ``get_provider()`` with a lookup into a local dict.

    Patches at the ``agent.web_search_registry`` level — the provider
    module imports ``get_provider`` from there via its lazy call-site import.
    Returns the dict so tests can add/remove providers.
    """
    providers: Dict[str, MockProvider] = {}

    def fake_get_provider(name: str):
        return providers.get(name)

    monkeypatch.setattr(
        "agent.web_search_registry.get_provider",
        fake_get_provider,
    )
    return providers


@pytest.fixture
def provider() -> QuotaFallbackWebSearchProvider:
    return QuotaFallbackWebSearchProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK_RESULT = {"success": True, "data": {"web": [{"title": "R", "url": "https://r", "description": "", "position": 1}]}}
_EMPTY_RESULT = {"success": True, "data": {"web": []}}
_QUOTA_ERROR = {"success": False, "error": "402 Payment Required — quota exceeded"}
_RATE_LIMIT_ERROR = {"success": False, "error": "429 Too Many Requests — rate limit exceeded"}
_NETWORK_ERROR = {"success": False, "error": "Connection error: timeout"}


def _with_providers(providers_dict: Dict[str, MockProvider], **kwargs) -> Dict[str, MockProvider]:
    """Add mock providers to the registry dict. Returns the dict for chaining."""
    for name, cfg in kwargs.items():
        providers_dict[name] = MockProvider(name=name, **cfg)
    return providers_dict


# ===================================================================
# Tests
# ===================================================================


class TestSearchSuccess:
    """Happy path — first provider succeeds."""

    def test_tavily_success_returns_immediately(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _OK_RESULT},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("test query")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["title"] == "R"

    def test_success_from_second_provider_when_first_unavailable(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"available": False},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("test")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1


class TestQuotaFallback:
    """Quota/billing errors trigger fallback to the next provider."""

    def test_tavily_quota_falls_to_exa(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("test")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1

    def test_rate_limit_falls_through(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _RATE_LIMIT_ERROR},
                "exa": {"search_result": _RATE_LIMIT_ERROR},
                "baidu": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("test")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1

    def test_all_quota_errors_return_aggregated(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _QUOTA_ERROR},
                "baidu": {"search_result": _QUOTA_ERROR},
                "searxng": {"search_result": _QUOTA_ERROR},
            },
        )
        result = provider.search("test")
        assert result["success"] is False
        assert "All search providers failed" in result["error"]
        for name in ("tavily", "exa", "baidu", "searxng"):
            assert name in result["error"]


class TestEmptyResultsFallback:
    """Empty results can optionally trigger a fallback."""

    def test_empty_falls_through_by_default(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _EMPTY_RESULT},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        # default has empty_results_fallback=true
        result = provider.search("test")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1  # From exa

    def test_empty_results_fallback_disabled(self, provider, _patch_registry, monkeypatch):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _EMPTY_RESULT},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        monkeypatch.setattr(
            "plugins.web.quota_fallback.provider.QuotaFallbackWebSearchProvider._read_config",
            lambda self: {"empty_results_fallback": False},
        )
        result = provider.search("test")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 0  # Stops at tavily

    def test_empty_triggers_fallback_when_enabled(self, provider, _patch_registry, monkeypatch):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _EMPTY_RESULT},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        monkeypatch.setattr(
            "plugins.web.quota_fallback.provider.QuotaFallbackWebSearchProvider._read_config",
            lambda self: {"empty_results_fallback": True},
        )
        result = provider.search("test")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1


class TestCooldown:
    """Cooldown state prevents repeated calls to exhausted providers."""

    def test_quota_exhausted_provider_is_skipped(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        # First call — tavily fails and gets cooldowned
        r1 = provider.search("test")
        assert r1["success"] is True
        assert r1["data"]["web"][0]["title"] == "R"

        # Second call — tavily should be skipped due to cooldown
        result2 = provider.search("test")
        assert result2["success"] is True

    def test_state_file_corruption_does_not_crash(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _OK_RESULT},
            },
        )
        # Write garbage to state file
        fb_provider._state_path().write_text("{invalid json!!!")
        # Should not crash
        result = provider.search("test")
        assert result["success"] is True


class TestAllProvidersFail:
    """When every provider fails, return aggregated error."""

    def test_aggregated_error_contains_all_reasons(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _RATE_LIMIT_ERROR},
                "baidu": {"search_result": _NETWORK_ERROR},
                "searxng": {"search_result": _QUOTA_ERROR},
            },
        )
        result = provider.search("test")
        assert result["success"] is False
        error = result["error"]
        assert "tavily" in error
        assert "exa" in error
        assert "baidu" in error
        assert "searxng" in error

    def test_no_providers_registered(self, provider, _patch_registry):
        result = provider.search("test")
        assert result["success"] is False
        assert "All search providers failed" in result["error"]

    def test_empty_query(self, provider):
        result = provider.search("")
        assert result["success"] is False
        assert "Query is empty" in result["error"]
        result = provider.search("   ")
        assert result["success"] is False


class TestProviderProperties:
    """ABC contract tests."""

    def test_name(self, provider):
        assert provider.name == "quota-fallback"

    def test_display_name(self, provider):
        assert provider.display_name == "Quota Fallback Search"

    def test_is_available(self, provider):
        assert provider.is_available() is True

    def test_supports_search(self, provider):
        assert provider.supports_search() is True

    def test_supports_extract(self, provider):
        assert provider.supports_extract() is False


class TestConfigCustomization:
    """Custom order via config.yaml."""

    def test_custom_order(self, provider, _patch_registry, monkeypatch):
        providers = _with_providers(
            _patch_registry,
            **{
                "searxng": {"search_result": _OK_RESULT},
                "tavily": {"search_result": _QUOTA_ERROR},
            },
        )
        monkeypatch.setattr(
            "plugins.web.quota_fallback.provider.QuotaFallbackWebSearchProvider._read_config",
            lambda self: {"order": ["searxng", "tavily"]},
        )
        result = provider.search("test")
        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "R"

    def test_custom_cooldown_minutes(self, provider, _patch_registry, monkeypatch):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        monkeypatch.setattr(
            "plugins.web.quota_fallback.provider.QuotaFallbackWebSearchProvider._read_config",
            lambda self: {"cooldown_minutes": {"quota": 1}},
        )
        result = provider.search("test")
        assert result["success"] is True

        # Verify state file was written with cooldown
        state = json.loads(fb_provider._state_path().read_text())
        assert "cooldowns" in state
        assert "tavily" in state["cooldowns"]
        assert state["cooldowns"]["tavily"]["type"] == "quota"


class TestSearXNGLastResort:
    """When all providers fail, SearXNG is the last resort."""

    def test_searxng_is_last_resort(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _QUOTA_ERROR},
                "baidu": {"search_result": _QUOTA_ERROR},
                "searxng": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("test")
        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "R"


class TestExceptionSafety:
    """Child provider exceptions are caught and treated as failures."""

    def test_child_exception_caught(self, provider, _patch_registry):
        class ExplodingProvider(MockProvider):
            def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
                raise RuntimeError("Kaboom!")

        _patch_registry["tavily"] = ExplodingProvider(
            name="tavily", search_result=_OK_RESULT
        )
        _patch_registry["exa"] = MockProvider(
            name="exa", search_result=_OK_RESULT
        )
        result = provider.search("test")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1


class TestNetworkErrorFallback:
    """Network errors trigger fallback."""

    def test_network_error_falls_through(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _NETWORK_ERROR},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("test")
        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "R"

    def test_connection_refused_falls_through(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": {"success": False, "error": "Connection refused"}},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("test")
        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "R"


class TestCJKDetection:
    """CJK character detection for language-aware routing."""

    def test_ascii_query_no_cjk(self):
        assert _has_cjk("hello world") is False
        assert _has_cjk("OpenAI Codex CLI") is False
        assert _has_cjk("12345!@#$%") is False
        assert _has_cjk("") is False

    def test_chinese_query_detected(self):
        assert _has_cjk("今日新闻") is True
        assert _has_cjk("人工智能 GPT") is True
        assert _has_cjk("你好 world") is True

    def test_japanese_detected(self):
        assert _has_cjk("こんにちは") is True
        assert _has_cjk("東京") is True

    def test_korean_detected(self):
        assert _has_cjk("안녕하세요") is True
        assert _has_cjk("서울") is True


class TestLanguageRouting:
    """CJK queries route to baidu first; Latin queries to tavily."""

    def test_chinese_query_baidu_first(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "baidu": {"search_result": _OK_RESULT},
                "tavily": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("你好世界")
        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "R"

    def test_chinese_baidu_falls_to_others(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "baidu": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("你好世界")
        assert result["success"] is True

    def test_latin_query_tavily_first(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _OK_RESULT},
                "baidu": {"search_result": _OK_RESULT},
            },
        )
        result = provider.search("hello world")
        assert result["success"] is True

    def test_latin_query_order_respected(self, provider, _patch_registry):
        providers = _with_providers(
            _patch_registry,
            **{
                "tavily": {"search_result": _QUOTA_ERROR},
                "exa": {"search_result": _OK_RESULT},
                "baidu": {"search_result": _OK_RESULT},
            },
        )
        # Latin: tavily → exa (succeeds before baidu)
        result = provider.search("hello world")
        assert result["success"] is True
