"""Regression tests for direct-credential fallback when the tool gateway is
unavailable (#79628) — browser-use provider.

When ``use_gateway: true`` is set but the Nous Tool Gateway cannot
authenticate (expired Portal session), the browser-use provider used to
return ``None`` (→ caller raises) even though a valid direct
``BROWSER_USE_API_KEY`` was present. The fix falls back to the direct
credential (mirroring the Krea/FAL pattern).
"""

from unittest.mock import patch

from plugins.browser.browser_use import provider as browser_use_provider


class TestBrowserUseGatewayFallback:
    def test_direct_key_fallback_when_gateway_unavailable(self, monkeypatch):
        """Direct key + use_gateway:true + dead gateway → direct config."""
        provider = browser_use_provider.BrowserUseBrowserProvider()

        monkeypatch.setattr(
            "tools.managed_tool_gateway.resolve_managed_tool_gateway",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "tools.managed_tool_gateway.peek_nous_access_token",
            lambda: None,
        )
        monkeypatch.setattr(
            "tools.tool_backend_helpers.prefers_gateway",
            lambda *a: True,
        )

        with patch.object(
            browser_use_provider, "get_secret", return_value="bu-direct-key"
        ):
            config = provider._get_config_or_none()

        assert config == {
            "api_key": "bu-direct-key",
            "base_url": browser_use_provider._BASE_URL,
            "managed_mode": False,
        }

    def test_no_direct_key_still_returns_none(self, monkeypatch):
        """No direct key + dead gateway → None (caller raises, unchanged)."""
        provider = browser_use_provider.BrowserUseBrowserProvider()

        monkeypatch.setattr(
            "tools.managed_tool_gateway.resolve_managed_tool_gateway",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "tools.managed_tool_gateway.peek_nous_access_token",
            lambda: None,
        )
        monkeypatch.setattr(
            "tools.tool_backend_helpers.prefers_gateway",
            lambda *a: True,
        )

        with patch.object(
            browser_use_provider, "get_secret", return_value=""
        ):
            assert provider._get_config_or_none() is None

    def test_gateway_wins_when_resolvable(self, monkeypatch):
        """Gateway resolvable → managed config used (no direct fallback)."""
        provider = browser_use_provider.BrowserUseBrowserProvider()

        managed = type(
            "Managed",
            (),
            {"nous_user_token": "nous-token", "gateway_origin": "https://gw.example.com/"},
        )()

        monkeypatch.setattr(
            "tools.managed_tool_gateway.resolve_managed_tool_gateway",
            lambda *a, **k: managed,
        )
        monkeypatch.setattr(
            "tools.tool_backend_helpers.prefers_gateway",
            lambda *a: True,
        )

        with patch.object(
            browser_use_provider, "get_secret", return_value="bu-direct-key"
        ):
            config = provider._get_config_or_none()

        assert config == {
            "api_key": "nous-token",
            "base_url": "https://gw.example.com",
            "managed_mode": True,
        }
