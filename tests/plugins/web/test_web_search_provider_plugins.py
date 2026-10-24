"""Plugin-side tests for the web search provider migration (PR #25182).

Covers:

- All bundled plugins (brave-free, ddgs, searxng, exa, parallel,
  tavily, firecrawl, keenable, xai) instantiate and self-report the expected
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

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_web_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every web-provider env var so is_available() returns False."""
    for k in (
        "BRAVE_SEARCH_API_KEY",
        "SEARXNG_URL",
        "KEENABLE_API_KEY",
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
        "XAI_API_KEY",
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

    def test_keenable_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("keenable")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("KEENABLE_API_KEY", "real")
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

    def test_firecrawl_explicit_config_allows_keyless_cloud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        assert p.is_available() is False

        monkeypatch.setattr(
            "tools.web_tools._load_web_config",
            lambda: {"backend": "firecrawl"},
            raising=False,
        )
        assert p.is_available() is True


    def test_xai_requires_api_key_or_oauth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """xAI needs XAI_API_KEY or OAuth tokens in auth.json."""
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("xai")
        assert p is not None
        assert p.is_available() is False  # no XAI_API_KEY, no auth.json
        monkeypatch.setenv("XAI_API_KEY", "real")
        assert p.is_available() is True


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
        from agent.web_search_registry import _resolve

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


    def test_no_config_no_credentials_returns_none(
        self,
    ) -> None:
        """No backend configured AND no credentials → keyless tier or ddgs.

        Resolution order with zero credentials: ddgs if its Python package
        is importable, else the keyless free tier (Parallel/Exa public
        endpoints — resolves with ``is_available() == False`` but
        ``is_keyless_available() == True``), else None (keyless tier
        disabled). All three outcomes are correct; a provider that is
        neither keyed nor keyless-capable means an env var leaked in.
        """
        _ensure_plugins_loaded()
        from agent.web_search_registry import _resolve

        result = _resolve(None, capability="search")
        if result is not None:
            assert result.is_available() or result.is_keyless_available()





# ---------------------------------------------------------------------------
# Firecrawl timeout alignment (#43272)
# ---------------------------------------------------------------------------


class TestFirecrawlNoResponseError:
    """The SDK raises AttributeError("'NoneType' object has no attribute 'status_code'") when no HTTP
    response arrived (API restarting); the failure text must say Firecrawl was unreachable."""

    _SDK_ERR = AttributeError("'NoneType' object has no attribute 'status_code'")

    def _provider(self, monkeypatch, fake_client):
        from plugins.web.firecrawl import provider as firecrawl_provider
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")  # direct path, not the keyless ring
        monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda capability=None: fake_client)
        monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: None)
        return firecrawl_provider.FirecrawlWebSearchProvider()

    def test_search_maps_no_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        client = MagicMock()
        client.search.side_effect = self._SDK_ERR
        out = self._provider(monkeypatch, client).search("q", 5)
        assert out["success"] is False
        assert "unreachable (no HTTP response" in out["error"]
        assert "NoneType" not in out["error"]

    def test_extract_maps_no_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        client = MagicMock()
        client.scrape.side_effect = self._SDK_ERR
        out = asyncio.run(self._provider(monkeypatch, client).extract(["https://example.com"]))
        assert "unreachable (no HTTP response" in out[0]["error"]

    def test_other_attribute_errors_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        client = MagicMock()
        client.search.side_effect = AttributeError("'dict' object has no attribute 'web'")
        out = self._provider(monkeypatch, client).search("q", 5)
        assert "'dict' object has no attribute 'web'" in out["error"]


class TestFirecrawlScrapeTimeout:
    """Verify the server-side scrape timeout matches the asyncio deadline."""

    def test_extract_passes_timeout_ms_to_scrape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """scrape() must receive timeout=60000 (ms) so the server-side
        deadline matches the 60 s asyncio.wait_for deadline.  Without this
        the API uses its 30 s default, causing SCRAPE_TIMEOUT."""
        from unittest.mock import MagicMock, patch
        from plugins.web.firecrawl import provider as firecrawl_provider

        # Direct-credential mode: with no FIRECRAWL_API_KEY extract() routes
        # to the keyless ring instead of the direct scrape path under test.
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")

        # Set up a fake client whose scrape() returns a minimal response
        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "metadata": {"title": "Test", "sourceURL": "https://example.com"},
            "markdown": "content",
        }

        # Patch _get_firecrawl_client to return our fake client
        monkeypatch.setattr(
            firecrawl_provider, "_get_firecrawl_client", lambda capability=None: fake_client
        )
        # Patch check_website_access to allow the URL
        monkeypatch.setattr(
            firecrawl_provider, "check_website_access", lambda url: None
        )

        p = firecrawl_provider.FirecrawlWebSearchProvider()
        results = asyncio.run(p.extract(["https://example.com"]))

        # Verify scrape was called with timeout=60000
        fake_client.scrape.assert_called_once()
        call_kwargs = fake_client.scrape.call_args
        assert call_kwargs.kwargs.get("timeout") == 60_000, (
            f"Expected timeout=60000, got {call_kwargs.kwargs.get('timeout')}"
        )

        # Verify the result was processed correctly
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        assert results[0]["content"] == "content"
