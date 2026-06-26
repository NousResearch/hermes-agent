"""Unit tests for the CloakBrowser web-search provider plugin."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_web_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CLOAKBROWSER_PROXY",
        "CLOAKBROWSER_HEADLESS",
        "CLOAKBROWSER_HUMANIZE",
        "CLOAKBROWSER_GEOIP",
    ):
        monkeypatch.delenv(key, raising=False)


class TestCloakBrowserProvider:
    def test_registers_in_registry(self) -> None:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        from agent.web_search_registry import get_provider

        provider = get_provider("cloakbrowser")
        assert provider is not None
        assert provider.name == "cloakbrowser"
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    def test_is_available_false_without_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        from agent.web_search_registry import get_provider

        provider = get_provider("cloakbrowser")
        assert provider is not None

        def _raise_import(*_a, **_k):
            raise ImportError("no cloakbrowser")

        monkeypatch.setattr(
            "plugins.web.cloakbrowser.provider._ensure_cloakbrowser",
            _raise_import,
        )
        assert provider.is_available() is False

    def test_search_success_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.cloakbrowser.provider import CloakBrowserWebSearchProvider

        provider = CloakBrowserWebSearchProvider()
        sample = [
            {
                "title": "Example",
                "url": "https://example.com",
                "description": "An example",
                "position": 1,
            }
        ]
        monkeypatch.setattr(
            "plugins.web.cloakbrowser.provider._ensure_cloakbrowser",
            lambda: None,
        )
        monkeypatch.setattr(
            "plugins.web.cloakbrowser.provider.search_duckduckgo_sync",
            lambda q, limit: sample,
        )
        result = provider.search("example query", limit=3)
        assert result["success"] is True
        assert result["data"]["web"] == sample

    def test_search_import_error_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.cloakbrowser.provider import CloakBrowserWebSearchProvider

        provider = CloakBrowserWebSearchProvider()

        def _raise(*_a, **_k):
            raise ImportError("missing cloakbrowser")

        monkeypatch.setattr(
            "plugins.web.cloakbrowser.provider._ensure_cloakbrowser",
            _raise,
        )
        result = provider.search("q", limit=1)
        assert result["success"] is False
        assert "cloakbrowser" in result["error"].lower()

    def test_extract_delegates_to_sync_helper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plugins.web.cloakbrowser.provider import CloakBrowserWebSearchProvider

        provider = CloakBrowserWebSearchProvider()
        monkeypatch.setattr(
            "plugins.web.cloakbrowser.provider._ensure_cloakbrowser",
            lambda: None,
        )
        monkeypatch.setattr(
            "plugins.web.cloakbrowser.provider.extract_urls_sync",
            lambda urls, format=None: [
                {"url": urls[0], "title": "T", "content": "body", "raw_content": "body"}
            ],
        )
        rows = asyncio.run(
            provider.extract(["https://example.com"], format="markdown")
        )
        assert rows[0]["url"] == "https://example.com"
        assert rows[0]["content"] == "body"

    def test_launch_options_defaults(self) -> None:
        from plugins.web.cloakbrowser.session import launch_options

        opts = launch_options()
        assert opts["headless"] is True
        assert opts["humanize"] is False
        assert opts["geoip"] is False
        assert "proxy" not in opts

    def test_launch_options_proxy_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.cloakbrowser.session import launch_options

        monkeypatch.setenv("CLOAKBROWSER_PROXY", "http://user:pass@proxy:8080")
        monkeypatch.setenv("CLOAKBROWSER_HUMANIZE", "1")
        opts = launch_options()
        assert opts["proxy"] == "http://user:pass@proxy:8080"
        assert opts["humanize"] is True

    def test_parse_ddg_html_results(self) -> None:
        from plugins.web.cloakbrowser.session import _parse_ddg_html_results

        page = MagicMock()
        row = MagicMock()

        link_first = MagicMock()
        link_first.count.return_value = 1
        link_first.get_attribute.return_value = "https://example.com"
        link_first.inner_text.return_value = "Example Domain"
        link = MagicMock()
        link.first = link_first

        snippet_first = MagicMock()
        snippet_first.count.return_value = 1
        snippet_first.inner_text.return_value = "Example snippet"
        snippet = MagicMock()
        snippet.first = snippet_first

        row.locator.side_effect = lambda sel: link if "result__a" in sel else snippet

        rows = MagicMock()
        rows.count.return_value = 1
        rows.nth.return_value = row
        page.locator.return_value = rows

        parsed = _parse_ddg_html_results(page, 5)
        assert len(parsed) == 1
        assert parsed[0]["url"] == "https://example.com"
        assert parsed[0]["title"] == "Example Domain"
