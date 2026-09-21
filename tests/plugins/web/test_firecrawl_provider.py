"""Tests for plugins/web/firecrawl/provider.py — coverage-focused.

Covers the lazy SDK proxy, dual-auth client construction (direct / gateway /
keyless), response-shape normalizers, and the extract() per-URL policy gate
path that the existing plugin-level tests don't reach.

Design notes:
* Patching targets the SOURCE modules that the provider imports from locally
  (e.g. patch("tools.lazy_deps.ensure") not patch("provider._lazy_ensure")),
  because the provider uses local imports inside many functions.
* Module-level imports (is_safe_url, check_website_access) can be patched on
  the provider module directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

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
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


def _ensure_plugins_loaded() -> None:
    """Idempotently load plugins so the registry is populated."""
    from hermes_cli.plugins import _ensure_plugins_discovered

    _ensure_plugins_discovered()


def _reset_firecrawl_client() -> None:
    """Drop the cached Firecrawl client so tests can re-instantiate cleanly."""
    import tools.web_tools as _wt

    _wt._firecrawl_client = None
    _wt._firecrawl_client_config = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean web-provider env."""
    _clear_web_env(monkeypatch)
    _reset_firecrawl_client()


# ---------------------------------------------------------------------------
# Lazy SDK proxy — _FirecrawlProxy
# ---------------------------------------------------------------------------


class TestFirecrawlProxy:
    """Cover the lazy proxy object (__call__, __instancecheck__, __repr__)."""

    def test_proxy_repr(self) -> None:
        from plugins.web.firecrawl.provider import Firecrawl

        assert repr(Firecrawl) == "<lazy firecrawl.Firecrawl proxy>"

    def test_proxy_is_callable(self) -> None:
        from plugins.web.firecrawl.provider import Firecrawl

        assert callable(Firecrawl)

    def test_proxy_instancecheck_without_sdk(self) -> None:
        """__instancecheck__ delegates to _load_firecrawl_cls()."""
        import plugins.web.firecrawl.provider as p

        with patch.object(p, "_load_firecrawl_cls") as mock_load:
            mock_load.side_effect = ImportError("no firecrawl SDK")
            with pytest.raises(ImportError, match="no firecrawl SDK"):
                isinstance(MagicMock(), p.Firecrawl)

    def test_proxy_instancecheck_with_fake_sdk(self) -> None:
        """When the SDK is "available", isinstance works."""
        import plugins.web.firecrawl.provider as p

        fake_cls = type("FakeFirecrawl", (), {})
        with patch.object(p, "_load_firecrawl_cls", return_value=fake_cls):
            assert isinstance(MagicMock(spec=fake_cls), p.Firecrawl)
            assert not isinstance(MagicMock(), p.Firecrawl)


# ---------------------------------------------------------------------------
# _load_firecrawl_cls — import + cache path
# ---------------------------------------------------------------------------


class TestLoadFirecrawlCls:
    """Cover _load_firecrawl_cls — the lazy import + global cache."""

    def test_cache_hit_returns_cached_cls(self) -> None:
        """When _FIRECRAWL_CLS_CACHE is already set, returns it."""
        import plugins.web.firecrawl.provider as p

        fake_cls = type("CachedFake", (), {})
        with patch.object(p, "_FIRECRAWL_CLS_CACHE", fake_cls):
            assert p._load_firecrawl_cls() is fake_cls

    def test_import_error_on_lazy_ensure_failure(self) -> None:
        """ImportError from lazy_ensure is caught and re-raised."""
        import plugins.web.firecrawl.provider as p

        with patch.object(p, "lazy_ensure",
            side_effect=ImportError("search.firecrawl not installed"),
        ):
            with pytest.raises(ImportError, match="search.firecrawl"):
                p._load_firecrawl_cls()

    def test_other_exception_from_lazy_ensure_becomes_import_error(self) -> None:
        """Non-ImportError exceptions from lazy_ensure are wrapped in ImportError."""
        import plugins.web.firecrawl.provider as p

        with patch.object(p, "lazy_ensure",
            side_effect=ImportError("something weird"),
        ):
            with pytest.raises(ImportError, match="something weird"):
                p._load_firecrawl_cls()


# ---------------------------------------------------------------------------
# _get_direct_firecrawl_config
# ---------------------------------------------------------------------------


class TestGetDirectFirecrawlConfig:
    """Cover _get_direct_firecrawl_config — direct vs keyless vs None."""

    def test_no_key_no_url_no_selection_returns_none(self) -> None:
        """Without FIRECRAWL_API_KEY, FIRECRAWL_API_URL, and without explicit
        firecrawl selection, returns None."""
        import plugins.web.firecrawl.provider as p

        assert p._get_direct_firecrawl_config() is None

    def test_keyless_when_explicit_selection_and_no_credentials(self) -> None:
        """Explicit firecrawl selection with no credentials → keyless tuple."""
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            result = p._get_direct_firecrawl_config()
        assert result is not None
        mode, kwargs, cache_key = result
        assert mode == "keyless"
        assert kwargs == {"api_url": "https://api.firecrawl.dev"}
        assert cache_key[0] == "direct-keyless"

    def test_sdk_mode_with_api_key_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FIRECRAWL_API_KEY set (no URL) → sdk mode."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key-123")
        import plugins.web.firecrawl.provider as p

        result = p._get_direct_firecrawl_config()
        assert result is not None
        mode, kwargs, cache_key = result
        assert mode == "sdk"
        assert kwargs == {"api_key": "test-key-123"}
        assert cache_key == ("direct", None, "test-key-123")

    def test_sdk_mode_with_api_url_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FIRECRAWL_API_URL set (no key) → sdk mode."""
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        import plugins.web.firecrawl.provider as p

        result = p._get_direct_firecrawl_config()
        assert result is not None
        mode, kwargs, cache_key = result
        assert mode == "sdk"
        assert kwargs == {"api_url": "http://localhost:3002"}
        assert cache_key == ("direct", "http://localhost:3002", None)

    def test_sdk_mode_with_both_key_and_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both FIRECRAWL_API_KEY and FIRECRAWL_API_URL → sdk with both."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "my-key")
        monkeypatch.setenv("FIRECRAWL_API_URL", "https://custom.firecrawl.dev")
        import plugins.web.firecrawl.provider as p

        result = p._get_direct_firecrawl_config()
        assert result is not None
        mode, kwargs, cache_key = result
        assert mode == "sdk"
        assert kwargs == {"api_key": "my-key", "api_url": "https://custom.firecrawl.dev"}
        assert cache_key == ("direct", "https://custom.firecrawl.dev", "my-key")

    def test_api_url_trailing_slash_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Trailing slashes on FIRECRAWL_API_URL are stripped."""
        monkeypatch.setenv("FIRECRAWL_API_URL", "https://example.com/")
        import plugins.web.firecrawl.provider as p

        result = p._get_direct_firecrawl_config()
        _, kwargs, _ = result
        assert kwargs["api_url"] == "https://example.com"


# ---------------------------------------------------------------------------
# _is_explicit_firecrawl_selection
# ---------------------------------------------------------------------------


class TestIsExplicitFirecrawlSelection:
    """Cover _is_explicit_firecrawl_selection."""

    def test_returns_true_for_backend_firecrawl(self) -> None:
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            assert p._is_explicit_firecrawl_selection() is True

    def test_returns_true_for_search_backend_firecrawl(self) -> None:
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={"search_backend": "firecrawl"}):
            assert p._is_explicit_firecrawl_selection() is True

    def test_returns_true_for_extract_backend_firecrawl(self) -> None:
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={"extract_backend": "firecrawl"}):
            assert p._is_explicit_firecrawl_selection() is True

    def test_returns_false_for_other_backend(self) -> None:
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={"backend": "exa"}):
            assert p._is_explicit_firecrawl_selection() is False

    def test_returns_false_for_empty_config(self) -> None:
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={}):
            assert p._is_explicit_firecrawl_selection() is False


# ---------------------------------------------------------------------------
# _use_keyless_ring
# ---------------------------------------------------------------------------


class TestUseKeylessRing:
    """Cover _use_keyless_ring — ring dispatch eligibility."""

    def test_false_when_api_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "some-key")
        import plugins.web.firecrawl.provider as p

        assert p._use_keyless_ring() is False

    def test_false_when_api_url_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        import plugins.web.firecrawl.provider as p

        assert p._use_keyless_ring() is False

    def test_false_when_nous_managed_selected(self) -> None:
        """When read_selection returns NOUS_MANAGED_PROVIDER, ring is disabled."""
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
        import plugins.web.firecrawl.provider as p

        with patch(
            "tools.tool_backend_helpers.read_selection",
            return_value=NOUS_MANAGED_PROVIDER,
        ):
            assert p._use_keyless_ring() is False

    def test_false_when_tool_gateway_ready_and_not_explicitly_selected(self) -> None:
        """Gateway ready + no explicit firecrawl selection → no ring."""
        import plugins.web.firecrawl.provider as p

        with patch(
            "tools.tool_backend_helpers.read_selection", return_value=None
        ), patch(
            "plugins.web.firecrawl.provider._is_explicit_firecrawl_selection",
            return_value=False,
        ), patch.object(p, "_is_tool_gateway_ready", return_value=True
        ):
            assert p._use_keyless_ring() is False

    def test_true_when_keyless_mcp_use_keyless_returns_true(self) -> None:
        """When all disqualifiers are absent, use_keyless drives ring decision."""
        import plugins.web.firecrawl.provider as p

        with patch(
            "tools.tool_backend_helpers.read_selection", return_value=None
        ), patch(
            "plugins.web.firecrawl.provider._is_explicit_firecrawl_selection",
            return_value=False,
        ), patch.object(p, "_is_tool_gateway_ready", return_value=False
        ), patch(
            "plugins.web.keyless_mcp.use_keyless", return_value=True
        ):
            assert p._use_keyless_ring() is True


# ---------------------------------------------------------------------------
# _KeylessFirecrawlClient
# ---------------------------------------------------------------------------


class TestKeylessFirecrawlClient:
    """Cover _KeylessFirecrawlClient — keyless REST client."""

    def test_default_api_url(self) -> None:
        from plugins.web.firecrawl.provider import _KeylessFirecrawlClient

        client = _KeylessFirecrawlClient()
        assert client.api_url == "https://api.firecrawl.dev"

    def test_custom_api_url(self) -> None:
        from plugins.web.firecrawl.provider import _KeylessFirecrawlClient

        client = _KeylessFirecrawlClient(api_url="http://localhost:9999")
        assert client.api_url == "http://localhost:9999"

    def test_api_url_trailing_slash_stripped(self) -> None:
        from plugins.web.firecrawl.provider import _KeylessFirecrawlClient

        client = _KeylessFirecrawlClient(api_url="http://localhost:9999/")
        assert client.api_url == "http://localhost:9999"

    def test_search_builds_correct_payload(self) -> None:
        """search() calls _post with the right path and payload."""
        from plugins.web.firecrawl.provider import _KeylessFirecrawlClient

        client = _KeylessFirecrawlClient(api_url="http://mock")
        with patch("httpx.post", return_value=MagicMock(json=lambda: {"results": []})) as mock_post:
            result = client.search(query="test query", limit=10)
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://mock/v2/search"
            assert call_args[1]["json"] == {"query": "test query", "limit": 10}
            assert result == {"results": []}

    def test_scrape_builds_correct_payload(self) -> None:
        """scrape() calls _post with the right path and payload."""
        from plugins.web.firecrawl.provider import _KeylessFirecrawlClient

        client = _KeylessFirecrawlClient(api_url="http://mock")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = client.scrape(url="http://example.com", formats=["markdown"])
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://mock/v2/scrape"
            assert call_args[1]["json"] == {
                "url": "http://example.com",
                "formats": ["markdown"],
            }
            assert result == {"data": {}}


# ---------------------------------------------------------------------------
# _get_firecrawl_gateway_url
# ---------------------------------------------------------------------------


class TestGetFirecrawlGatewayUrl:
    """Cover _get_firecrawl_gateway_url."""

    def test_delegates(self) -> None:
        import tools.managed_tool_gateway as gw
        import plugins.web.firecrawl.provider as p

        with patch.object(gw, "build_vendor_gateway_url", return_value="https://gateway.example.com"):
            assert p._get_firecrawl_gateway_url() == "https://gateway.example.com"


# ---------------------------------------------------------------------------
# _is_tool_gateway_ready
# ---------------------------------------------------------------------------


class TestIsToolGatewayReady:
    """Cover _is_tool_gateway_ready."""

    def test_returns_true_when_gateway_resolves(self) -> None:
        """When resolve_managed_tool_gateway returns a non-None value."""
        import tools.managed_tool_gateway as gw
        import plugins.web.firecrawl.provider as p

        fake_gateway = MagicMock()
        with patch.object(
            gw, "resolve_managed_tool_gateway", return_value=fake_gateway
        ):
            assert p._is_tool_gateway_ready() is True

    def test_returns_false_when_gateway_is_none(self) -> None:
        """When resolve_managed_tool_gateway returns None."""
        import tools.managed_tool_gateway as gw
        import plugins.web.firecrawl.provider as p

        with patch.object(gw, "resolve_managed_tool_gateway", return_value=None):
            assert p._is_tool_gateway_ready() is False


# ---------------------------------------------------------------------------
# _has_direct_firecrawl_config
# ---------------------------------------------------------------------------


class TestHasDirectFirecrawlConfig:
    """Cover _get_direct_firecrawl_config — direct vs keyless vs None."""

    def test_true_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "key")
        import plugins.web.firecrawl.provider as p

        result = p._get_direct_firecrawl_config()
        assert result is not None
        mode, kwargs, cache_key = result
        assert mode == "sdk"
        assert kwargs == {"api_key": "key"}

    def test_true_when_url_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        import plugins.web.firecrawl.provider as p

        result = p._get_direct_firecrawl_config()
        assert result is not None
        mode, kwargs, cache_key = result
        assert mode == "sdk"
        assert kwargs == {"api_url": "http://localhost:3002"}

    def test_false_when_neither_present(self) -> None:
        import plugins.web.firecrawl.provider as p

        assert p._get_direct_firecrawl_config() is None


# ---------------------------------------------------------------------------
# check_firecrawl_api_key
# ---------------------------------------------------------------------------


class TestCheckFirecrawlApiKey:
    """Cover check_firecrawl_api_key — the availability checker."""

    def test_true_with_direct_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "real-key")
        import plugins.web.firecrawl.provider as p

        assert p.check_firecrawl_api_key() is True

    def test_true_with_direct_api_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        import plugins.web.firecrawl.provider as p

        assert p.check_firecrawl_api_key() is True

    def test_true_when_gateway_ready_no_direct(self) -> None:
        """When no direct config but gateway is ready → True (legacy fallback)."""
        import tools.managed_tool_gateway as gw
        import tools.tool_backend_helpers as tbh
        import plugins.web.firecrawl.provider as p

        fake_gateway = MagicMock()
        with patch.object(gw, "resolve_managed_tool_gateway", return_value=fake_gateway), \
             patch.object(tbh, "read_selection", return_value=None):
            assert p.check_firecrawl_api_key() is True

    def test_false_when_no_direct_and_no_gateway(self) -> None:
        """No direct config and no gateway → False."""
        import tools.managed_tool_gateway as gw
        import tools.tool_backend_helpers as tbh
        import plugins.web.firecrawl.provider as p

        with patch.object(gw, "resolve_managed_tool_gateway", return_value=None), \
             patch.object(tbh, "read_selection", return_value=None):
            assert p.check_firecrawl_api_key() is False

    def test_true_when_nous_selected_and_gateway_ready(self) -> None:
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
        import tools.managed_tool_gateway as gw
        import plugins.web.firecrawl.provider as p

        fake_gateway = MagicMock()
        with patch.object(gw, "resolve_managed_tool_gateway", return_value=fake_gateway), \
             patch(
            "tools.tool_backend_helpers.read_selection",
            return_value=NOUS_MANAGED_PROVIDER,
        ):
            assert p.check_firecrawl_api_key() is True

    def test_false_when_nous_selected_but_no_gateway(self) -> None:
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
        import tools.managed_tool_gateway as gw
        import plugins.web.firecrawl.provider as p

        with patch.object(gw, "resolve_managed_tool_gateway", return_value=None), \
             patch(
            "tools.tool_backend_helpers.read_selection",
            return_value=NOUS_MANAGED_PROVIDER,
        ):
            assert p.check_firecrawl_api_key() is False


# ---------------------------------------------------------------------------
# _firecrawl_backend_help_suffix
# ---------------------------------------------------------------------------


class TestFirecrawlBackendHelpSuffix:
    """Cover _firecrawl_backend_help_suffix — help text guidance."""

    def test_empty_when_disabled(self) -> None:
        import tools.tool_backend_helpers as tbh
        import plugins.web.firecrawl.provider as p

        with patch.object(tbh, "managed_nous_tools_enabled", return_value=False):
            assert p._firecrawl_backend_help_suffix() == ""

    def test_suffix_when_enabled(self) -> None:
        import tools.tool_backend_helpers as tbh
        import plugins.web.firecrawl.provider as p

        with patch.object(tbh, "managed_nous_tools_enabled", return_value=True):
            s = p._firecrawl_backend_help_suffix()
            assert "Nous Tool Gateway" in s


# ---------------------------------------------------------------------------
# _raise_web_backend_configuration_error
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# _get_firecrawl_client — full client construction
# ---------------------------------------------------------------------------


class TestGetFirecrawlClient:
    """Cover _get_firecrawl_client — the central client factory."""

    def test_cached_client_returned_when_config_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When _firecrawl_client_config matches, cached client is returned."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch(
            "tools.tool_backend_helpers.read_selection", return_value=None
        ):
            result = p._get_firecrawl_client()
            assert result is fake_client

    def test_direct_sdk_mode_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FIRECRAWL_API_KEY → sdk client via Firecrawl proxy."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "my-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_firecrawl = MagicMock()
        fake_firecrawl.return_value = MagicMock()
        with patch.object(p, "Firecrawl", fake_firecrawl):
            _reset_firecrawl_client()
            with patch(
                "tools.tool_backend_helpers.read_selection", return_value=None
            ):
                client = p._get_firecrawl_client()
                assert client is not None
                fake_firecrawl.assert_called_once_with(api_key="my-key")

    def test_direct_sdk_mode_with_api_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FIRECRAWL_API_URL → sdk client with api_url."""
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_firecrawl = MagicMock()
        fake_firecrawl.return_value = MagicMock()
        with patch.object(p, "Firecrawl", fake_firecrawl):
            _reset_firecrawl_client()
            with patch(
                "tools.tool_backend_helpers.read_selection", return_value=None
            ):
                client = p._get_firecrawl_client()
                assert client is not None
                fake_firecrawl.assert_called_once_with(api_url="http://localhost:3002")

    def test_keyless_mode_creates_keyless_client(self) -> None:
        """Explicit firecrawl selection with no credentials → _KeylessFirecrawlClient."""
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            _reset_firecrawl_client()
            with patch(
                "tools.tool_backend_helpers.read_selection", return_value=None
            ):
                client = p._get_firecrawl_client()
                from plugins.web.firecrawl.provider import _KeylessFirecrawlClient

                assert isinstance(client, _KeylessFirecrawlClient)
                assert client.api_url == "https://api.firecrawl.dev"

    def test_nous_selected_with_gateway_raises_when_unavailable(self) -> None:
        """When NOUS_MANAGED_PROVIDER selected and gateway unavailable → ValueError."""
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
        import plugins.web.firecrawl.provider as p

        with patch(
            "tools.tool_backend_helpers.read_selection",
            return_value=NOUS_MANAGED_PROVIDER,
        ), patch(
            "tools.web_tools.resolve_managed_tool_gateway",
            return_value=None,
        ):
            _reset_firecrawl_client()
            with pytest.raises(ValueError, match="Nous Tool Gateway is not available"):
                p._get_firecrawl_client()

    def test_gateway_client_with_nous_token(self) -> None:
        """When gateway is available, client is constructed with gateway creds."""
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER
        import tools.web_tools as _wt
        import tools.managed_tool_gateway as gw
        import plugins.web.firecrawl.provider as p

        fake_gateway = MagicMock()
        fake_gateway.gateway_origin = "https://gateway.firecrawl.dev"
        fake_gateway.nous_user_token = "nous-token-123"
        fake_client = MagicMock()
        with patch.object(p, "Firecrawl", return_value=fake_client), \
             patch.object(gw, "resolve_managed_tool_gateway", return_value=fake_gateway), \
             patch.object(gw, "read_nous_access_token", return_value="nous-token-123"), \
             patch("tools.tool_backend_helpers.read_selection", return_value=NOUS_MANAGED_PROVIDER):
            _reset_firecrawl_client()
            client = p._get_firecrawl_client()
            assert client is fake_client

    def test_unselected_no_direct_config_falls_back_to_gateway(self) -> None:
        """Never-configured web section: legacy fallback to gateway."""
        import tools.web_tools as _wt
        import tools.managed_tool_gateway as gw
        import plugins.web.firecrawl.provider as p

        fake_gateway = MagicMock()
        fake_gateway.gateway_origin = "https://gateway.firecrawl.dev"
        fake_gateway.nous_user_token = "fallback-token"
        fake_client = MagicMock()
        with patch.object(p, "Firecrawl", return_value=fake_client), \
             patch.object(gw, "resolve_managed_tool_gateway", return_value=fake_gateway), \
             patch.object(gw, "read_nous_access_token", return_value="fallback-token"), \
             patch("tools.tool_backend_helpers.read_selection", return_value=None):
            _reset_firecrawl_client()
            client = p._get_firecrawl_client()
            assert client is fake_client

    def test_unselected_no_gateway_raises_configuration_error(self) -> None:
        """Never-configured, no direct, no gateway → ValueError."""
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "resolve_managed_tool_gateway", return_value=None), patch(
            "tools.tool_backend_helpers.read_selection", return_value=None
        ):
            _reset_firecrawl_client()
            with pytest.raises(ValueError, match="not configured"):
                p._get_firecrawl_client()

    def test_direct_selected_no_credentials_raises(self) -> None:
        """Direct firecrawl selection with no credentials → keyless client (not ValueError)."""
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        with patch.object(_wt, "_load_web_config", return_value={"backend": "firecrawl"}):
            _reset_firecrawl_client()
            with patch(
                "tools.tool_backend_helpers.read_selection", return_value=None
            ):
                client = p._get_firecrawl_client()
                from plugins.web.firecrawl.provider import _KeylessFirecrawlClient

                assert isinstance(client, _KeylessFirecrawlClient)
                assert client.api_url == "https://api.firecrawl.dev"


# ---------------------------------------------------------------------------
# _to_plain_object — response normalization
# ---------------------------------------------------------------------------


class TestToPlainObject:
    """Cover _to_plain_object — SDK object → plain Python."""

    def test_none_returns_none(self) -> None:
        from plugins.web.firecrawl.provider import _to_plain_object

        assert _to_plain_object(None) is None

    def test_primitive_passes_through(self) -> None:
        from plugins.web.firecrawl.provider import _to_plain_object

        assert _to_plain_object(42) == 42
        assert _to_plain_object("hello") == "hello"
        assert _to_plain_object(3.14) == 3.14
        assert _to_plain_object(True) is True
        assert _to_plain_object(False) is False

    def test_dict_passes_through(self) -> None:
        from plugins.web.firecrawl.provider import _to_plain_object

        assert _to_plain_object({"a": 1}) == {"a": 1}

    def test_list_passes_through(self) -> None:
        from plugins.web.firecrawl.provider import _to_plain_object

        assert _to_plain_object([1, 2, 3]) == [1, 2, 3]

    def test_model_dump_object(self) -> None:
        """Objects with model_dump() are converted via it."""
        from plugins.web.firecrawl.provider import _to_plain_object

        class FakeModel:
            def model_dump(self) -> Dict[str, Any]:
                return {"x": 1}

        assert _to_plain_object(FakeModel()) == {"x": 1}

    def test_dict_via___dict__(self) -> None:
        """Objects with __dict__ but no model_dump use __dict__."""
        from plugins.web.firecrawl.provider import _to_plain_object

        class FakeObj:
            pass

        obj = FakeObj()
        obj.visible = 42
        obj._hidden = "skip"

        assert _to_plain_object(obj) == {"visible": 42}

    def test_unknown_type_returned_as_is(self) -> None:
        """Objects that are not dict/list/str/int/float/bool and have no
        model_dump or __dict__ are returned unchanged."""
        from plugins.web.firecrawl.provider import _to_plain_object

        obj = object()
        assert _to_plain_object(obj) is obj


# ---------------------------------------------------------------------------
# _normalize_result_list
# ---------------------------------------------------------------------------


class TestNormalizeResultList:
    """Cover _normalize_result_list — mixed SDK payloads → list of dicts."""

    def test_non_list_returns_empty(self) -> None:
        from plugins.web.firecrawl.provider import _normalize_result_list

        assert _normalize_result_list("not a list") == []
        assert _normalize_result_list(None) == []
        assert _normalize_result_list(42) == []

    def test_empty_list(self) -> None:
        from plugins.web.firecrawl.provider import _normalize_result_list

        assert _normalize_result_list([]) == []

    def test_plain_dicts_passed_through(self) -> None:
        from plugins.web.firecrawl.provider import _normalize_result_list

        result = _normalize_result_list([{"url": "http://a"}, {"url": "http://b"}])
        assert result == [{"url": "http://a"}, {"url": "http://b"}]

    def test_sdk_objects_normalized_via_model_dump(self) -> None:
        from plugins.web.firecrawl.provider import _normalize_result_list

        class FakeResult:
            def model_dump(self) -> Dict[str, Any]:
                return {"url": "http://sdk.example"}

        result = _normalize_result_list([FakeResult()])
        assert result == [{"url": "http://sdk.example"}]

    def test_non_dict_items_skipped(self) -> None:
        from plugins.web.firecrawl.provider import _normalize_result_list

        result = _normalize_result_list(["string", 42, None])
        assert result == []


# ---------------------------------------------------------------------------
# _extract_web_search_results
# ---------------------------------------------------------------------------


class TestExtractWebSearchResults:
    """Cover _extract_web_search_results — response shape variants."""

    def test_dict_with_data_list(self) -> None:
        """response = {'data': [{...}, ...]} → returns data list."""
        from plugins.web.firecrawl.provider import _extract_web_search_results

        response = {"data": [{"url": "http://a"}, {"url": "http://b"}]}
        result = _extract_web_search_results(response)
        assert len(result) == 2
        assert result[0]["url"] == "http://a"

    def test_dict_with_data_web_list(self) -> None:
        """response = {'data': {'web': [{...}]}} → returns web list."""
        from plugins.web.firecrawl.provider import _extract_web_search_results

        response = {"data": {"web": [{"url": "http://x"}]}}
        result = _extract_web_search_results(response)
        assert len(result) == 1
        assert result[0]["url"] == "http://x"

    def test_dict_with_data_results_list(self) -> None:
        """response = {'data': {'results': [{...}]}} → returns results list."""
        from plugins.web.firecrawl.provider import _extract_web_search_results

        response = {"data": {"results": [{"url": "http://y"}]}}
        result = _extract_web_search_results(response)
        assert len(result) == 1
        assert result[0]["url"] == "http://y"

    def test_dict_with_top_level_web(self) -> None:
        """response = {'web': [{...}]} → returns web list."""
        from plugins.web.firecrawl.provider import _extract_web_search_results

        response = {"web": [{"url": "http://top"}]}
        result = _extract_web_search_results(response)
        assert len(result) == 1
        assert result[0]["url"] == "http://top"

    def test_dict_with_top_level_results(self) -> None:
        """response = {'results': [{...}]} → returns results list."""
        from plugins.web.firecrawl.provider import _extract_web_search_results

        response = {"results": [{"url": "http://r"}]}
        result = _extract_web_search_results(response)
        assert len(result) == 1
        assert result[0]["url"] == "http://r"

    def test_sdk_object_with_web_attribute(self) -> None:
        """SDK response object with .web attribute → returns web list."""
        from plugins.web.firecrawl.provider import _extract_web_search_results

        class FakeSDKResponse:
            web = [{"url": "http://sdk"}]

        result = _extract_web_search_results(FakeSDKResponse())
        assert len(result) == 1
        assert result[0]["url"] == "http://sdk"

    def test_empty_response_returns_empty_list(self) -> None:
        from plugins.web.firecrawl.provider import _extract_web_search_results

        assert _extract_web_search_results({}) == []
        assert _extract_web_search_results(None) == []


# ---------------------------------------------------------------------------
# _extract_scrape_payload
# ---------------------------------------------------------------------------


class TestExtractScrapePayload:
    """Cover _extract_scrape_payload — scrape response normalization."""

    def test_dict_with_data_key(self) -> None:
        """{'data': {...}} → returns the data dict."""
        from plugins.web.firecrawl.provider import _extract_scrape_payload

        payload = {"data": {"markdown": "hello", "metadata": {}}}
        result = _extract_scrape_payload(payload)
        assert result == {"markdown": "hello", "metadata": {}}

    def test_dict_without_data_key_returned_as_is(self) -> None:
        """{'markdown': '...', ...} → returns the whole dict."""
        from plugins.web.firecrawl.provider import _extract_scrape_payload

        payload = {"markdown": "hello", "metadata": {}}
        result = _extract_scrape_payload(payload)
        assert result == {"markdown": "hello", "metadata": {}}

    def test_non_dict_returns_empty(self) -> None:
        from plugins.web.firecrawl.provider import _extract_scrape_payload

        assert _extract_scrape_payload("not a dict") == {}
        assert _extract_scrape_payload(None) == {}

    def test_sdk_object_with_data_attribute(self) -> None:
        """SDK object with .data attribute → _to_plain_object flattens to dict with data key."""
        from plugins.web.firecrawl.provider import _extract_scrape_payload

        class FakeScrapeResult:
            class Data:
                markdown = "sdk content"

            def __init__(self):
                self.data = self.Data()

        result = _extract_scrape_payload(FakeScrapeResult())
        assert isinstance(result, dict)
        assert "data" in result
        assert result["data"].markdown == "sdk content"


# ---------------------------------------------------------------------------
# FirecrawlWebSearchProvider — search + extract paths
# ---------------------------------------------------------------------------


class TestFirecrawlWebSearchProvider:
    """Cover FirecrawlWebSearchProvider.search and .extract paths."""

    def test_name_and_display_name(self) -> None:
        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        provider = FirecrawlWebSearchProvider()
        assert provider.name == "firecrawl"
        assert provider.display_name == "Firecrawl"

    def test_supports_search_and_extract(self) -> None:
        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        provider = FirecrawlWebSearchProvider()
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    def test_is_available_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_available() returns True when FIRECRAWL_API_KEY is set."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "real-key")
        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        provider = FirecrawlWebSearchProvider()
        assert provider.is_available() is True

    def test_is_available_with_api_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_available() returns True when FIRECRAWL_API_URL is set."""
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        provider = FirecrawlWebSearchProvider()
        assert provider.is_available() is True

    def test_is_available_false_without_config(self) -> None:
        """is_available() returns False when no config is present."""
        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        provider = FirecrawlWebSearchProvider()
        assert provider.is_available() is False

    def test_search_returns_keyless_result_when_ring_enabled(self) -> None:
        """When _use_keyless_ring returns True, search delegates to
        search_with_failover."""
        import plugins.web.firecrawl.provider as p

        with patch.object(p, "_use_keyless_ring", return_value=True), patch(
            "plugins.web.keyless_mcp.search_with_failover",
            return_value={"success": True, "data": {"web": []}},
        ) as mock_failover:
            from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

            provider = p.FirecrawlWebSearchProvider()
            result = provider.search("test query", limit=5)
            mock_failover.assert_called_once_with("firecrawl", "test query", 5)
            assert result == {"success": True, "data": {"web": []}}

    def test_search_returns_error_on_interrupt(self) -> None:
        """When is_interrupted() is True, search returns error dict."""
        import plugins.web.firecrawl.provider as p

        with patch("tools.interrupt.is_interrupted", return_value=True):
            from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

            provider = p.FirecrawlWebSearchProvider()
            result = provider.search("query", limit=5)
            assert result == {"success": False, "error": "Interrupted"}

    def test_search_returns_success_with_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: search returns results from _extract_web_search_results."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.search.return_value = {"data": [{"url": "http://a.com"}]}
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch(
            "tools.tool_backend_helpers.read_selection", return_value=None
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = provider.search("test query", limit=3)
            assert result["success"] is True
            assert result["data"]["web"] == [{"url": "http://a.com"}]
            fake_client.search.assert_called_once_with(query="test query", limit=3)

    def test_search_returns_error_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the client throws, search returns an error dict."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.search.side_effect = RuntimeError("network error")
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch(
            "tools.tool_backend_helpers.read_selection", return_value=None
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = provider.search("query", limit=5)
            assert result["success"] is False
            assert "Firecrawl search failed" in result["error"]

    def test_extract_returns_error_on_interrupt(self) -> None:
        """When is_interrupted() is True, extract returns error items."""
        import plugins.web.firecrawl.provider as p

        with patch("tools.interrupt.is_interrupted", return_value=True):
            from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://example.com"]))
            assert len(result) == 1
            assert result[0]["url"] == "http://example.com"
            assert result[0]["error"] == "Interrupted"

    def test_extract_keyless_delegates_to_failover(self) -> None:
        """When _use_keyless_ring is True, extract uses extract_with_failover."""
        import plugins.web.firecrawl.provider as p

        with patch.object(p, "_use_keyless_ring", return_value=True), patch(
            "plugins.web.keyless_mcp.extract_with_failover",
            return_value=[{"url": "http://x", "content": "ok"}],
        ) as mock_failover:
            from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://a.com", "http://b.com"]))
            mock_failover.assert_called_once_with("firecrawl", ["http://a.com", "http://b.com"])
            assert len(result) == 1
            assert result[0]["url"] == "http://x"

    def test_extract_website_policy_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When check_website_access blocks a URL, extract returns blocked result."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value={
            "host": "blocked.example", "rule": "block-rule",
            "source": "policy", "message": "Blocked by policy",
        }):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://blocked.example"]))
            assert len(result) == 1
            assert result[0]["blocked_by_policy"] is not None
            assert result[0]["error"] == "Blocked by policy"

    def test_extract_format_markdown_selects_markdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """format='markdown' → markdown content is chosen."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "data": {
                "markdown": "## Markdown content",
                "html": "<html></html>",
                "metadata": {"title": "Test", "sourceURL": "http://example.com"},
            }
        }
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None), patch.object(
            p, "is_safe_url", return_value=True
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://example.com"], format="markdown"))
            assert len(result) == 1
            assert result[0]["content"] == "## Markdown content"

    def test_extract_format_html_selects_html(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """format='html' → html content is chosen when available."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "data": {
                "markdown": "## MD",
                "html": "<html>body</html>",
                "metadata": {"title": "Test", "sourceURL": "http://example.com"},
            }
        }
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None), patch.object(
            p, "is_safe_url", return_value=True
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://example.com"], format="html"))
            assert len(result) == 1
            assert result[0]["content"] == "<html>body</html>"

    def test_extract_defaults_to_markdown_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No format specified → markdown preferred when available."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "data": {
                "markdown": "## Markdown",
                "html": "<html></html>",
                "metadata": {"title": "Test", "sourceURL": "http://example.com"},
            }
        }
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None), patch.object(
            p, "is_safe_url", return_value=True
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://example.com"]))
            assert len(result) == 1
            assert result[0]["content"] == "## Markdown"

    def test_extract_falls_back_to_html_when_no_markdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No format, no markdown → html is used."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "data": {
                "html": "<html>only</html>",
                "metadata": {"title": "Test", "sourceURL": "http://example.com"},
            }
        }
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None), patch.object(
            p, "is_safe_url", return_value=True
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://example.com"]))
            assert len(result) == 1
            assert result[0]["content"] == "<html>only</html>"

    def test_extract_ssrf_blocked_after_redirect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When final URL (post-redirect) is unsafe, extract blocks the result."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "data": {
                "markdown": "content",
                "metadata": {
                    "title": "Test",
                    "sourceURL": "http://169.254.169.254/latest/meta-data/",
                },
            }
        }
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None), patch.object(
            p, "is_safe_url", return_value=False
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://example.com"]))
            assert len(result) == 1
            assert "Blocked" in result[0]["error"]
            assert "private or internal" in result[0]["error"]

    def test_extract_redirect_re_checked_by_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After redirect, final URL is re-checked against website policy."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "data": {
                "markdown": "content",
                "metadata": {
                    "title": "Redirected",
                    "sourceURL": "http://redirected.example.com",
                },
            }
        }
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", side_effect=[
            None,
            {"host": "redirected.example.com", "rule": "block",
             "source": "policy", "message": "Blocked after redirect"},
        ]), patch.object(p, "is_safe_url", return_value=True):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://original.com"]))
            assert len(result) == 1
            assert result[0]["blocked_by_policy"] is not None
            assert "Blocked after redirect" in result[0]["error"]

    def test_extract_timeout_returns_timeout_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When scrape exceeds 60s, a timeout error is returned."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.side_effect = asyncio.TimeoutError()
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://slow.example.com"]))
            assert len(result) == 1
            assert "timed out" in result[0]["error"]
            assert "60s" in result[0]["error"]

    def test_extract_scrape_exception_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When scrape raises a non-timeout exception, an error item is returned."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.side_effect = RuntimeError("scrape failed")
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://example.com"]))
            assert len(result) == 1
            assert result[0]["error"] == "scrape failed"

    def test_extract_multiple_urls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple URLs are scraped in sequence."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
        import tools.web_tools as _wt
        import plugins.web.firecrawl.provider as p

        fake_client = MagicMock()
        fake_client.scrape.return_value = {
            "data": {
                "markdown": "content",
                "metadata": {"title": "Test", "sourceURL": "http://example.com"},
            }
        }
        _wt._firecrawl_client = fake_client
        _wt._firecrawl_client_config = ("direct", None, "test-key")

        with patch.object(p, "check_website_access", return_value=None), patch.object(
            p, "is_safe_url", return_value=True
        ):
            provider = p.FirecrawlWebSearchProvider()
            result = asyncio.run(provider.extract(["http://a.com", "http://b.com"]))
            assert len(result) == 2

    def test_get_setup_schema(self) -> None:
        """get_setup_schema returns the expected shape."""
        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        provider = FirecrawlWebSearchProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "Firecrawl"
        assert "badge" in schema
        assert "tag" in schema
        assert any(e["key"] == "FIRECRAWL_API_KEY" for e in schema["env_vars"])


# ---------------------------------------------------------------------------
# is_keyless_available
# ---------------------------------------------------------------------------


class TestIsKeylessAvailable:
    """Cover FirecrawlWebSearchProvider.is_keyless_available."""

    def test_true_when_keyless_enabled_and_not_paid(self) -> None:
        with patch(
            "plugins.web.keyless_mcp.keyless_enabled", return_value=True
        ), patch(
            "plugins.web.keyless_mcp.provider_tier", return_value="free"
        ):
            from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

            provider = FirecrawlWebSearchProvider()
            assert provider.is_keyless_available() is True

    def test_false_when_keyless_disabled(self) -> None:
        with patch(
            "plugins.web.keyless_mcp.keyless_enabled", return_value=False
        ):
            from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

            provider = FirecrawlWebSearchProvider()
            assert provider.is_keyless_available() is False

    def test_false_when_provider_tier_paid(self) -> None:
        with patch(
            "plugins.web.keyless_mcp.keyless_enabled", return_value=True
        ), patch(
            "plugins.web.keyless_mcp.provider_tier", return_value="paid"
        ):
            from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

            provider = FirecrawlWebSearchProvider()
            assert provider.is_keyless_available() is False
