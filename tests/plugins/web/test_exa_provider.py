"""Tests for plugins/web/exa/provider.py — coverage-focused.

Covers the Exa web search + extract provider, keyless variant, SDK client
construction, and response shape helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_exa_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "EXA_API_KEY",
        "XAI_API_KEY",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_USER_TOKEN",
        "FIRECRAWL_API_KEY",
        "BRAVE_SEARCH_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_exa_env(monkeypatch)


# ---------------------------------------------------------------------------
# _get_exa_client — cached SDK client construction
# ---------------------------------------------------------------------------


class TestGetExaClient:
    """Cover _get_exa_client — lazy Exa SDK client with caching."""

    def test_missing_key_raises_valueerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        import plugins.web.exa.provider as p

        with pytest.raises(ValueError, match="EXA_API_KEY"):
            p._get_exa_client()

    def test_sdk_mode_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key-123")
        import plugins.web.exa.provider as p

        fake_exa = MagicMock()
        fake_exa_cls = MagicMock(return_value=fake_exa)
        fake_exa.return_value = MagicMock()

        with patch.dict("sys.modules", {"exa_py": MagicMock(Exa=fake_exa_cls)}), \
             patch("plugins.web._common.lazy_ensure"):
            import tools.web_tools as _wt

            _wt._exa_client = None
            client = p._get_exa_client()
            assert client is not None
            fake_exa_cls.assert_called_once_with(api_key="test-key-123")

    def test_cached_client_returned_on_repeat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        import plugins.web.exa.provider as p

        fake_exa = MagicMock()
        fake_exa_cls = MagicMock(return_value=fake_exa)
        # _factory does: from exa_py import Exa; client = Exa(api_key=...)
        # So Exa(...) calls fake_exa_cls(...) which returns fake_exa (the client)
        fake_exa_cls.return_value = fake_exa

        with patch.dict("sys.modules", {"exa_py": MagicMock(Exa=fake_exa_cls)}), \
             patch("plugins.web._common.lazy_ensure"):
            import tools.web_tools as _wt

            _wt._exa_client = None
            client1 = p._get_exa_client()
            # Don't reset _exa_client — just call again, should return cached
            client2 = p._get_exa_client()
            assert client1 is client2
            assert client1 is fake_exa
            assert client1 is not None


# ---------------------------------------------------------------------------
# ExaWebSearchProvider
# ---------------------------------------------------------------------------


class TestExaWebSearchProvider:
    """Cover ExaWebSearchProvider.search and .extract paths."""

    def test_name_and_display_name(self) -> None:
        from plugins.web.exa.provider import ExaWebSearchProvider

        provider = ExaWebSearchProvider()
        assert provider.name == "exa"
        assert provider.display_name == "Exa"

    def test_supports_search_and_extract(self) -> None:
        from plugins.web.exa.provider import ExaWebSearchProvider

        provider = ExaWebSearchProvider()
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    def test_get_setup_schema_returns_shape(self) -> None:
        from plugins.web.exa.provider import ExaWebSearchProvider

        schema = ExaWebSearchProvider().get_setup_schema()
        assert schema["name"].startswith("Exa")
        assert "badge" in schema
        assert schema["tag"] is not None

    def test_search_returns_keyless_result_when_ring_enabled(self) -> None:
        with patch(
            "plugins.web.exa.provider.use_keyless", return_value=True
        ), patch(
            "plugins.web.exa.provider.keyless_search",
            return_value={"success": True, "data": {"web": []}},
        ):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.search("test query", limit=5)
            assert result["success"] is True

    def test_search_returns_error_on_interrupt(self) -> None:
        with patch("tools.interrupt.is_interrupted", return_value=True):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.search("query", limit=5)
            assert result["success"] is False
            assert result["error"] == "Interrupted"

    def test_search_returns_ok_with_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        import plugins.web.exa.provider as p

        fake_exa = MagicMock()
        fake_response = MagicMock()
        fake_response.results = [
            MagicMock(url="http://a.com", title="Title A", highlights=["highlight A"]),
            MagicMock(url="http://b.com", title="Title B", highlights=["highlight B"]),
        ]
        fake_exa.search.return_value = fake_response

        with patch.object(p, "_get_exa_client", return_value=fake_exa):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.search("test query", limit=2)
            assert result["success"] is True
            assert len(result["data"]["web"]) == 2
            assert result["data"]["web"][0]["url"] == "http://a.com"
            assert result["data"]["web"][0]["title"] == "Title A"

    def test_search_returns_error_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        import plugins.web.exa.provider as p

        fake_exa = MagicMock()
        fake_exa.search.side_effect = RuntimeError("Exa API error")

        with patch.object(p, "_get_exa_client", return_value=fake_exa):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.search("query", limit=5)
            assert result["success"] is False
            assert "Exa" in result["error"]

    def test_extract_returns_keyless_result_when_ring_enabled(self) -> None:
        with patch(
            "plugins.web.exa.provider.use_keyless", return_value=True
        ), patch(
            "plugins.web.exa.provider.keyless_extract",
            return_value=[{"url": "http://a.com", "content": "keyless content"}],
        ):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.extract(["http://a.com"])
            assert result[0]["url"] == "http://a.com"
            assert result[0]["content"] == "keyless content"

    def test_extract_returns_error_on_interrupt(self) -> None:
        with patch("tools.interrupt.is_interrupted", return_value=True):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.extract(["http://a.com"])
            assert len(result) == 1
            assert result[0]["url"] == "http://a.com"
            assert result[0]["error"] == "Interrupted"

    def test_extract_returns_documents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        import plugins.web.exa.provider as p

        fake_exa = MagicMock()
        fake_response = MagicMock()
        fake_response.results = [
            MagicMock(url="http://a.com", title="Title A", text="Content A"),
            MagicMock(url="http://b.com", title="Title B", text="Content B"),
        ]
        fake_exa.get_contents.return_value = fake_response

        with patch.object(p, "_get_exa_client", return_value=fake_exa):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.extract(["http://a.com", "http://b.com"])
            assert len(result) == 2
            assert result[0]["url"] == "http://a.com"
            assert result[0]["title"] == "Title A"
            assert result[0]["content"] == "Content A"

    def test_extract_returns_error_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        import plugins.web.exa.provider as p

        fake_exa = MagicMock()
        fake_exa.get_contents.side_effect = RuntimeError("Exa extract error")

        with patch.object(p, "_get_exa_client", return_value=fake_exa):
            from plugins.web.exa.provider import ExaWebSearchProvider

            provider = ExaWebSearchProvider()
            result = provider.extract(["http://a.com"])
            assert len(result) == 1
            assert result[0]["url"] == "http://a.com"
            assert "Exa" in result[0]["error"]

    def test_is_available_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "real-key")
        from plugins.web.exa.provider import ExaWebSearchProvider

        provider = ExaWebSearchProvider()
        assert provider.is_available() is True

    def test_is_available_false_without_key(self) -> None:
        from plugins.web.exa.provider import ExaWebSearchProvider

        provider = ExaWebSearchProvider()
        assert provider.is_available() is False
