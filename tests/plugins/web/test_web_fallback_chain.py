"""Tests for the runtime fallback chain in ``agent.web_search_registry``.

Covers:

- ``_is_provider_failure()`` detection rules for dict-shaped and
  list-shaped results, including partial success (not a failure).
- ``resolve_fallback_chain()`` ordering: explicit config first, then
  legacy preference order, then remaining alphabetically. Includes
  availability filtering and capability filtering.
- The "silent empty" bug from ZsZolee's report: a provider returning
  ``{"success": true, "data": {"web": []}}`` is correctly detected as
  a failure, so the chain advances to the next provider.

These tests use a lightweight fake provider registered directly into
the registry (no plugin discovery required) so they run fast and don't
depend on which bundled plugins are installed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from agent.web_search_provider import WebSearchProvider
from agent.web_search_registry import (
    _is_provider_failure,
    _reset_for_tests,
    register_provider,
    resolve_fallback_chain,
)


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------


class _FakeProvider(WebSearchProvider):
    """Minimal provider with controllable capabilities and availability."""

    def __init__(
        self,
        name: str,
        *,
        search: bool = True,
        extract: bool = True,
        available: bool = True,
        display_name: Optional[str] = None,
    ) -> None:
        self._name = name
        self._search = search
        self._extract = extract
        self._available = available
        self._display_name = display_name or name

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display_name

    def supports_search(self) -> bool:
        return self._search

    def supports_extract(self) -> bool:
        return self._extract

    def is_available(self) -> bool:
        return self._available

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        return {"success": True, "data": {"web": []}}

    def extract(self, urls: List[str], **kwargs: Any) -> Any:
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty registry."""
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no web.backend config keys leak in."""
    # Patch _read_config_key to always return None unless a test
    # explicitly monkeypatches it.
    import agent.web_search_registry as reg

    monkeypatch.setattr(reg, "_read_config_key", lambda *path: None)


# ---------------------------------------------------------------------------
# _is_provider_failure
# ---------------------------------------------------------------------------


class TestIsProviderFailure:
    """``_is_provider_failure()`` correctly classifies results."""

    def test_empty_dict_is_failure(self) -> None:
        assert _is_provider_failure({}) is True

    def test_none_is_failure(self) -> None:
        assert _is_provider_failure(None) is True

    def test_success_false_is_failure(self) -> None:
        assert _is_provider_failure({"success": False, "error": "rate limited"}) is True

    def test_success_true_with_results_is_not_failure(self) -> None:
        result = {"success": True, "data": {"web": [{"title": "ok", "url": "https://example.com"}]}}
        assert _is_provider_failure(result) is False

    def test_success_true_with_empty_web_is_failure(self) -> None:
        """The ZsZolee bug: silent empty results must be detected."""
        result = {"success": True, "data": {"web": []}}
        assert _is_provider_failure(result) is True

    def test_success_true_with_no_data_key_is_not_failure(self) -> None:
        """If data is missing entirely, we can't classify -- not a failure."""
        result = {"success": True}
        assert _is_provider_failure(result) is False

    def test_empty_list_is_failure(self) -> None:
        assert _is_provider_failure([]) is True

    def test_list_all_errored_is_failure(self) -> None:
        result = [
            {"url": "https://a.com", "error": "403"},
            {"url": "https://b.com", "error": "timeout"},
        ]
        assert _is_provider_failure(result) is True

    def test_list_partial_success_is_not_failure(self) -> None:
        """At least one non-error entry means partial success."""
        result = [
            {"url": "https://a.com", "content": "hello"},
            {"url": "https://b.com", "error": "403"},
        ]
        assert _is_provider_failure(result) is False

    def test_list_all_success_is_not_failure(self) -> None:
        result = [
            {"url": "https://a.com", "content": "hello"},
            {"url": "https://b.com", "content": "world"},
        ]
        assert _is_provider_failure(result) is False

    def test_string_is_not_failure(self) -> None:
        """A bare string is an unexpected shape -- don't classify as failure."""
        assert _is_provider_failure("some text") is False


# ---------------------------------------------------------------------------
# resolve_fallback_chain
# ---------------------------------------------------------------------------


class TestResolveFallbackChain:
    """``resolve_fallback_chain()`` returns correctly ordered providers."""

    def test_empty_registry_returns_empty_chain(self) -> None:
        chain = resolve_fallback_chain(capability="search")
        assert chain == []

    def test_single_available_provider(self) -> None:
        p = _FakeProvider("exa", search=True, extract=False, available=True)
        register_provider(p)
        chain = resolve_fallback_chain(capability="search")
        assert len(chain) == 1
        assert chain[0].name == "exa"

    def test_unavailable_providers_excluded_from_chain(self) -> None:
        """Providers where is_available() is False are not in the chain
        (except the explicitly configured one)."""
        p1 = _FakeProvider("firecrawl", available=False)
        p2 = _FakeProvider("ddgs", available=True)
        register_provider(p1)
        register_provider(p2)
        chain = resolve_fallback_chain(capability="search")
        assert len(chain) == 1
        assert chain[0].name == "ddgs"

    def test_search_capability_filters_extract_only_providers(self) -> None:
        """A provider that only supports extract is excluded from search chain."""
        p = _FakeProvider("exa", search=False, extract=True, available=True)
        register_provider(p)
        chain = resolve_fallback_chain(capability="search")
        assert chain == []

    def test_extract_capability_filters_search_only_providers(self) -> None:
        """A provider that only supports search is excluded from extract chain."""
        p = _FakeProvider("brave-free", search=True, extract=False, available=True)
        register_provider(p)
        chain = resolve_fallback_chain(capability="extract")
        assert chain == []

    def test_legacy_preference_order_respected(self) -> None:
        """Chain follows _LEGACY_PREFERENCE order, not registration order."""
        # Register in reverse of legacy order
        for name in reversed(["firecrawl", "parallel", "tavily", "exa", "searxng", "brave-free", "ddgs"]):
            register_provider(_FakeProvider(name))
        chain = resolve_fallback_chain(capability="search")
        names = [p.name for p in chain]
        # Legacy order: firecrawl, parallel, tavily, exa, searxng, brave-free, ddgs
        assert names == ["firecrawl", "parallel", "tavily", "exa", "searxng", "brave-free", "ddgs"]

    def test_unknown_providers_appended_alphabetically(self) -> None:
        """Providers not in _LEGACY_PREFERENCE are appended alphabetically."""
        register_provider(_FakeProvider("zebra"))
        register_provider(_FakeProvider("alpha"))
        register_provider(_FakeProvider("ddgs"))  # known, comes first
        chain = resolve_fallback_chain(capability="search")
        names = [p.name for p in chain]
        # ddgs (legacy) first, then alpha, zebra alphabetically
        assert names == ["ddgs", "alpha", "zebra"]

    def test_configured_provider_first_even_if_not_in_legacy_order(self) -> None:
        """Explicitly configured provider goes first, then legacy order."""
        import agent.web_search_registry as reg

        # Monkeypatch _read_config_key to return "ddgs" for search
        original = reg._read_config_key

        def _mock_read(*path: str) -> Optional[str]:
            if path == ("web", "search_backend"):
                return "ddgs"
            return None

        reg._read_config_key = _mock_read
        try:
            register_provider(_FakeProvider("firecrawl"))
            register_provider(_FakeProvider("ddgs"))
            chain = resolve_fallback_chain(capability="search")
            assert chain[0].name == "ddgs"
            assert chain[1].name == "firecrawl"
        finally:
            reg._read_config_key = original

    def test_configured_unavailable_provider_excluded_from_chain(self) -> None:
        """The explicitly configured provider is NOT in the chain when
        is_available() is False. The preflight resolver surfaces the typed
        credential error; the chain is for runtime failures only."""
        import agent.web_search_registry as reg

        original = reg._read_config_key

        def _mock_read(*path: str) -> Optional[str]:
            if path == ("web", "search_backend"):
                return "firecrawl"
            return None

        reg._read_config_key = _mock_read
        try:
            register_provider(_FakeProvider("firecrawl", available=False))
            register_provider(_FakeProvider("ddgs", available=True))
            chain = resolve_fallback_chain(capability="search")
            names = [p.name for p in chain]
            # firecrawl excluded (unavailable); only ddgs in chain
            assert "firecrawl" not in names
            assert names == ["ddgs"]
        finally:
            reg._read_config_key = original

    def test_no_duplicates_in_chain(self) -> None:
        """A provider that's both configured and in legacy order appears once."""
        import agent.web_search_registry as reg

        original = reg._read_config_key

        def _mock_read(*path: str) -> Optional[str]:
            if path == ("web", "search_backend"):
                return "firecrawl"
            return None

        reg._read_config_key = _mock_read
        try:
            register_provider(_FakeProvider("firecrawl"))
            register_provider(_FakeProvider("ddgs"))
            chain = resolve_fallback_chain(capability="search")
            names = [p.name for p in chain]
            assert names.count("firecrawl") == 1
            assert names == ["firecrawl", "ddgs"]
        finally:
            reg._read_config_key = original

    def test_shared_web_backend_config_used_when_capability_specific_absent(self) -> None:
        """``web.backend`` is used as fallback when ``web.search_backend`` is unset."""
        import agent.web_search_registry as reg

        original = reg._read_config_key

        def _mock_read(*path: str) -> Optional[str]:
            if path == ("web", "backend"):
                return "tavily"
            return None

        reg._read_config_key = _mock_read
        try:
            register_provider(_FakeProvider("firecrawl"))
            register_provider(_FakeProvider("tavily"))
            chain = resolve_fallback_chain(capability="search")
            assert chain[0].name == "tavily"
            assert chain[1].name == "firecrawl"
        finally:
            reg._read_config_key = original

# ---------------------------------------------------------------------------
# Regression test for extract chain exception handling
# ---------------------------------------------------------------------------


class TestExtractChainExceptionHandling:
    """Regression: when provider.extract() raises an exception, the chain
    must advance to the next provider, not break.

    Bug found by independent code review: the original inline list check
    ``results and all(...)`` short-circuited on empty list (falsy), so
    exceptions (which set results=[]) fell through to ``break`` instead
    of ``continue``, stopping the fallback chain.
    """

    def test_extract_exception_advances_to_next_provider(self) -> None:
        """_is_provider_failure correctly detects empty list from exception."""
        # Simulate what happens after an exception: results = []
        assert _is_provider_failure([]) is True

    def test_extract_all_errored_advances_to_next_provider(self) -> None:
        """_is_provider_failure detects all-errored list."""
        result = [
            {"url": "https://a.com", "error": "403"},
            {"url": "https://b.com", "error": "timeout"},
        ]
        assert _is_provider_failure(result) is True

    def test_extract_partial_success_does_not_trigger_fallback(self) -> None:
        """_is_provider_failure does NOT flag partial success as failure."""
        result = [
            {"url": "https://a.com", "content": "hello"},
            {"url": "https://b.com", "error": "403"},
        ]
        assert _is_provider_failure(result) is False


# ---------------------------------------------------------------------------
# Contentless-row detection (teknium1 review comment)
# ---------------------------------------------------------------------------


class TestContentlessRowDetection:
    """Regression: a list of rows with no ``error`` key but also no
    ``content`` (e.g. Firecrawl returning empty markdown/HTML) must be
    detected as a failure so the chain advances to the next provider.

    teknium1 pointed out that Firecrawl can produce a result like::

        [{"url": "https://a.com", "title": "a", "content": ""}]

    -- no ``error`` key, structurally "successful", but useless to the user.
    The old ``all(r.get("error"))`` check returned False, the loop broke,
    and the fallback chain never tried the next provider.
    """

    def test_all_rows_contentless_no_error_is_failure(self) -> None:
        """Every row has empty content and no error -> failure."""
        result = [
            {"url": "https://a.com", "title": "a", "content": ""},
            {"url": "https://b.com", "title": "b", "content": ""},
        ]
        assert _is_provider_failure(result) is True

    def test_all_rows_missing_content_key_is_failure(self) -> None:
        """Every row lacks a content key entirely -> failure."""
        result = [
            {"url": "https://a.com", "title": "a"},
            {"url": "https://b.com", "title": "b"},
        ]
        assert _is_provider_failure(result) is True

    def test_mixed_contentless_and_content_is_not_failure(self) -> None:
        """One row has content, one doesn't -> partial success, not failure."""
        result = [
            {"url": "https://a.com", "title": "a", "content": ""},
            {"url": "https://b.com", "title": "b", "content": "hello"},
        ]
        assert _is_provider_failure(result) is False

    def test_mixed_error_and_contentless_is_failure(self) -> None:
        """One row errored, one contentless (no error) -> failure.
        Not all have error, but all lack content -> failure via content check."""
        result = [
            {"url": "https://a.com", "error": "403"},
            {"url": "https://b.com", "content": ""},
        ]
        assert _is_provider_failure(result) is True

    def test_single_row_empty_content_is_failure(self) -> None:
        """Single row with no error but empty content -> failure."""
        result = [{"url": "https://a.com", "content": ""}]
        assert _is_provider_failure(result) is True

    def test_all_rows_error_and_contentless_is_failure(self) -> None:
        """All rows have both error and empty content -> failure (via error check)."""
        result = [
            {"url": "https://a.com", "error": "403", "content": ""},
            {"url": "https://b.com", "error": "timeout", "content": ""},
        ]
        assert _is_provider_failure(result) is True
