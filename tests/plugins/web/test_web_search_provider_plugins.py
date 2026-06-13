"""Plugin-side tests for the web search provider migration (PR #25182).

Covers:

- All seven bundled plugins (brave-free, ddgs, searxng, exa, parallel,
  tavily, firecrawl) instantiate and self-report the expected
  capabilities + ABC-derived defaults.
- Each plugin's ``is_available()`` correctly reflects env-var presence.
- The web_search_registry resolves an active provider in the documented
  scenarios (explicit config wins ignoring availability, fallback walks
  legacy preference filtered by availability, unknown name falls back).
- Plugin response shapes match the legacy bit-for-bit contract.

Per the dev skill: these tests use *real* imports from the plugin
modules — no mocking of provider classes themselves — so the test
catches drift in the ABC interface, the registry, and the plugin
glue layer simultaneously.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_web_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every web-provider env var so is_available() returns False."""
    for k in (
        "BRAVE_SEARCH_API_KEY",
        "SEARXNG_URL",
        "TAVILY_API_KEY",
        "TAVILY_BASE_URL",
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "PARALLEL_SEARCH_MODE",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)


def _ensure_plugins_loaded() -> None:
    """Idempotently load plugins so the registry is populated."""
    from hermes_cli.plugins import _ensure_plugins_discovered

    _ensure_plugins_discovered()


# ---------------------------------------------------------------------------
# Per-plugin discovery + capability flags
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean web-provider env."""
    _clear_web_env(monkeypatch)


class TestBundledPluginsRegister:
    """All seven bundled web plugins discover and register correctly."""

    def test_all_seven_plugins_present_in_registry(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import list_providers

        names = sorted(p.name for p in list_providers())
        assert names == [
            "brave-free",
            "ddgs",
            "exa",
            "firecrawl",
            "parallel",
            "searxng",
            "tavily",
        ]

    @pytest.mark.parametrize(
        "plugin_name,expected_search,expected_extract,expected_crawl",
        [
            ("brave-free", True, False, False),
            ("ddgs", True, False, False),
            ("searxng", True, False, False),
            ("exa", True, True, False),
            ("parallel", True, True, False),
            ("tavily", True, True, True),
            # firecrawl: search + extract + crawl. Crawl was originally
            # disabled in the migration (fell through to a legacy inline
            # path); the follow-up commit enabled it natively.
            ("firecrawl", True, True, True),
        ],
    )
    def test_capability_flags_match_spec(
        self,
        plugin_name: str,
        expected_search: bool,
        expected_extract: bool,
        expected_crawl: bool,
    ) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None, f"plugin {plugin_name!r} not registered"
        assert provider.supports_search() is expected_search
        assert provider.supports_extract() is expected_extract
        assert provider.supports_crawl() is expected_crawl

    @pytest.mark.parametrize(
        "plugin_name",
        ["brave-free", "ddgs", "searxng", "exa", "parallel", "tavily", "firecrawl"],
    )
    def test_each_plugin_has_name_and_display_name(self, plugin_name: str) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None
        assert provider.name == plugin_name
        assert provider.display_name  # any non-empty string

    @pytest.mark.parametrize(
        "plugin_name",
        ["brave-free", "ddgs", "searxng", "exa", "parallel", "tavily", "firecrawl"],
    )
    def test_each_plugin_has_setup_schema(self, plugin_name: str) -> None:
        """``get_setup_schema()`` returns a dict the picker can consume."""
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None
        schema = provider.get_setup_schema()
        assert isinstance(schema, dict)
        assert "name" in schema
        assert "env_vars" in schema


# ---------------------------------------------------------------------------
# is_available() behavior
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """Each plugin's ``is_available()`` returns False without env config."""

    def test_brave_free_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("brave-free")
        assert p is not None
        assert p.is_available() is False  # no BRAVE_SEARCH_API_KEY
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "real")
        assert p.is_available() is True

    def test_searxng_requires_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("searxng")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
        assert p.is_available() is True

    def test_tavily_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("tavily")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("TAVILY_API_KEY", "real")
        assert p.is_available() is True

    def test_exa_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("exa")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("EXA_API_KEY", "real")
        assert p.is_available() is True

    def test_parallel_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("parallel")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("PARALLEL_API_KEY", "real")
        assert p.is_available() is True

    def test_firecrawl_requires_either_key_or_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        assert p.is_available() is False

        # Either FIRECRAWL_API_KEY or FIRECRAWL_API_URL lights it up.
        monkeypatch.setenv("FIRECRAWL_API_KEY", "real")
        assert p.is_available() is True
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        assert p.is_available() is True

    def test_ddgs_always_available_when_package_importable(self) -> None:
        """DDGS is the always-on fallback — no API key required.

        It may report unavailable if the ``ddgs`` package itself isn't
        installed in the env (legitimate — the plugin's post_setup hook
        triggers pip install on first selection). We only assert that
        is_available() doesn't raise.
        """
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("ddgs")
        assert p is not None
        # Truthy or falsy, just must not raise.
        _ = bool(p.is_available())


# ---------------------------------------------------------------------------
# Registry resolution semantics (Option B — conservative smart fallback)
# ---------------------------------------------------------------------------


class TestRegistryResolution:
    """``_resolve()`` follows explicit-config + availability-filtered fallback."""

    def test_explicit_configured_provider_returned_even_when_unavailable(
        self,
    ) -> None:
        """Explicit ``web.search_backend`` wins regardless of is_available().

        Without availability filtering on the explicit path, the dispatcher
        would silently switch backends; with this check the dispatcher
        surfaces a precise "FOO_API_KEY is not set" error instead.
        """
        _ensure_plugins_loaded()
        from agent.web_search_registry import _resolve, get_provider

        # No BRAVE_SEARCH_API_KEY (fixture cleared it).
        result = _resolve("brave-free", capability="search")
        assert result is not None
        assert result.name == "brave-free"
        # Confirm it's the unavailable one — dispatcher will surface
        # a typed credential-missing error to the caller.
        assert result.is_available() is False

    def test_unknown_configured_name_falls_back_to_available_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typo / uninstalled plugin → walk legacy preference, pick available."""
        _ensure_plugins_loaded()
        from agent.web_search_registry import _resolve

        monkeypatch.setenv("EXA_API_KEY", "real")
        result = _resolve("not-a-real-provider", capability="search")
        # Either ddgs (no-key fallback) or exa (the only available
        # premium provider) — both are valid. The point is the unknown
        # name shouldn't return None when SOMETHING is available.
        assert result is not None
        assert result.is_available() is True

    def test_explicit_search_only_provider_for_extract_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asking for extract via a search-only backend → fall back.

        ``brave-free`` is search-only (``supports_extract() is False``).
        When the registry resolves it for an extract capability, the
        explicit-config branch rejects it as capability-incompatible
        and the fallback walk picks an extract-capable provider.
        """
        _ensure_plugins_loaded()
        from agent.web_search_registry import _resolve

        monkeypatch.setenv("EXA_API_KEY", "real")
        result = _resolve("brave-free", capability="extract")
        # Should land on exa (only extract-capable available provider).
        assert result is not None
        assert result.supports_extract() is True
        assert result.is_available() is True

    def test_no_config_no_credentials_returns_none(
        self,
    ) -> None:
        """No backend configured AND no available providers → typically None.

        ``ddgs`` is the no-credential fallback; if its ``ddgs`` Python
        package is installed in the test env, ddgs will be picked.
        Otherwise the resolver returns None. Either outcome is correct.
        """
        _ensure_plugins_loaded()
        from agent.web_search_registry import _resolve

        result = _resolve(None, capability="search")
        if result is not None:
            # The only no-credential provider is ddgs; anything else
            # means an env var leaked in.
            assert result.is_available() is True


# ---------------------------------------------------------------------------
# Sync-vs-async extract detection
# ---------------------------------------------------------------------------


class TestAsyncExtractDispatch:
    """The dispatcher detects async vs sync extract methods correctly."""

    def test_parallel_extract_is_async(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("parallel")
        assert p is not None
        assert inspect.iscoroutinefunction(p.extract) is True

    def test_firecrawl_extract_is_async(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        assert inspect.iscoroutinefunction(p.extract) is True

    def test_exa_extract_is_sync(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("exa")
        assert p is not None
        assert inspect.iscoroutinefunction(p.extract) is False

    def test_tavily_extract_is_sync(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("tavily")
        assert p is not None
        assert inspect.iscoroutinefunction(p.extract) is False


# ---------------------------------------------------------------------------
# Error response shape (preserved bit-for-bit from legacy)
# ---------------------------------------------------------------------------


class TestErrorResponseShapes:
    """When credentials are missing, plugins return typed errors, not raises."""

    def test_brave_free_returns_error_dict_when_unconfigured(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("brave-free")
        assert p is not None
        result = p.search("test", limit=5)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_searxng_returns_error_dict_when_unconfigured(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("searxng")
        assert p is not None
        result = p.search("test", limit=5)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_exa_returns_error_dict_when_unconfigured(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("exa")
        assert p is not None
        result = p.search("test", limit=5)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_tavily_returns_error_dict_when_unconfigured(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("tavily")
        assert p is not None
        result = p.search("test", limit=5)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result

    def test_parallel_extract_returns_per_url_errors_when_unconfigured(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("parallel")
        assert p is not None
        result = asyncio.run(p.extract(["https://example.com"]))
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        assert result[0]["url"] == "https://example.com"

    def test_firecrawl_extract_returns_per_url_errors_when_unconfigured(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        # firecrawl extract returns [] when the website-policy gate rejects
        # the URL, or a per-URL error dict when the gate passes but the
        # firecrawl client fails. Use a URL the policy allows to make sure
        # we hit the credential-missing path.
        result = asyncio.run(p.extract(["https://example.com"]))
        assert isinstance(result, list)
        if result:  # if anything came back, it should be an error entry
            assert "error" in result[0]

    def test_tavily_crawl_returns_error_dict_when_unconfigured(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("tavily")
        assert p is not None
        result = p.crawl("https://example.com")
        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)
        if result["results"]:
            assert "error" in result["results"][0]

    def test_firecrawl_crawl_returns_error_dict_when_unconfigured(self) -> None:
        """firecrawl crawl is async (wraps SDK in to_thread); error must be
        surfaced via the per-page result shape, not raised."""
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        assert inspect.iscoroutinefunction(p.crawl)
        result = asyncio.run(p.crawl("https://example.com"))
        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)
        # Without FIRECRAWL_API_KEY, the plugin's _get_firecrawl_client()
        # raises ValueError which is caught and returned as a per-page error.
        assert len(result["results"]) >= 1
        assert "error" in result["results"][0]
        assert result["results"][0]["url"] == "https://example.com"
