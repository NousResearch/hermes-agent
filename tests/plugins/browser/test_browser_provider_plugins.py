"""Plugin-side tests for the browser provider migration (PR #25214).

Covers:

- All bundled plugins (browserbase, browser-use, firecrawl, remote_browser)
  instantiate and self-report the expected ABC defaults.
- Each plugin's ``is_available()`` correctly reflects env-var presence.
- The browser_registry resolves an active provider in the documented
  scenarios:
    * explicit config wins ignoring availability (so dispatcher surfaces
      a typed credentials error)
    * legacy preference walk: browser-use → browserbase (filtered by
      availability)
    * firecrawl is NOT in the legacy walk — explicit-only
    * unknown name falls through to auto-detect
    * ``local`` short-circuits to None

These tests use *real* imports from the plugin modules — no mocking of
provider classes themselves — so the test catches drift in the ABC
interface, the registry, and the plugin glue layer simultaneously.
Mirrors ``tests/plugins/web/test_web_search_provider_plugins.py`` from
PR #25182.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_browser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every browser-provider env var so is_available() returns False."""
    for k in (
        "BROWSERBASE_API_KEY",
        "BROWSERBASE_PROJECT_ID",
        "BROWSERBASE_BASE_URL",
        "BROWSER_USE_API_KEY",
        "BROWSER_USE_GATEWAY_URL",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_BROWSER_TTL",
        "REMOTE_BROWSER_API_KEY",
        "REMOTE_BROWSER_BASE_URL",
        "REMOTE_BROWSER_CREATE_PATH",
        "REMOTE_BROWSER_STATUS_PATH_TEMPLATE",
        "REMOTE_BROWSER_TERMINATE_PATH_TEMPLATE",
        "REMOTE_BROWSER_TIMEOUT_MINUTES",
        "REMOTE_BROWSER_POLL_TIMEOUT_SECONDS",
        "REMOTE_BROWSER_POLL_INTERVAL_SECONDS",
        "REMOTE_BROWSER_READY_GRACE_SECONDS",
        "REMOTE_BROWSER_RESOLUTION",
        "REMOTE_BROWSER_REGION",
        "REMOTE_BROWSER_PROFILE_ID",
        "REMOTE_BROWSER_PROFILE_NAME",
        "REMOTE_BROWSER_RECORDING",
        "REMOTE_BROWSER_RECORDING_RETENTION_DAYS",
        "REMOTE_BROWSER_PROXY_TYPE",
        "REMOTE_BROWSER_PROXY_URL",
        "REMOTE_BROWSER_LAUNCH_ARGUMENTS",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_USER_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)


def _ensure_plugins_loaded() -> None:
    """Idempotently load plugins so the registry is populated."""
    from hermes_cli.plugins import _ensure_plugins_discovered

    _ensure_plugins_discovered()


# ---------------------------------------------------------------------------
# Per-test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean browser-provider env."""
    _clear_browser_env(monkeypatch)


# ---------------------------------------------------------------------------
# Bundled plugins register
# ---------------------------------------------------------------------------


class TestBundledPluginsRegister:
    """All bundled browser plugins discover and register correctly."""

    def test_all_plugins_present_in_registry(self) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import list_providers

        names = sorted(p.name for p in list_providers())
        assert names == ["browser-use", "browserbase", "firecrawl", "remote_browser"]

    @pytest.mark.parametrize(
        "plugin_name,expected_display",
        [
            ("browserbase", "Browserbase"),
            ("browser-use", "Browser Use"),
            ("firecrawl", "Firecrawl"),
            ("remote_browser", "Remote Browser"),
        ],
    )
    def test_each_plugin_has_name_and_display_name(
        self, plugin_name: str, expected_display: str
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None, f"plugin {plugin_name!r} not registered"
        assert provider.name == plugin_name
        assert provider.display_name == expected_display


# ---------------------------------------------------------------------------
# is_available() behavior
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """Each plugin's ``is_available()`` reflects env-var presence accurately."""

    def test_browserbase_requires_both_api_key_and_project_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("browserbase")
        assert p is not None
        assert p.is_available() is False

        # API key alone is insufficient.
        monkeypatch.setenv("BROWSERBASE_API_KEY", "key")
        assert p.is_available() is False

        # Both env vars set → available.
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj")
        assert p.is_available() is True


    def test_browser_use_satisfied_by_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("browser-use")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("BROWSER_USE_API_KEY", "key")
        assert p.is_available() is True

    def test_firecrawl_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("firecrawl")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("FIRECRAWL_API_KEY", "key")
        assert p.is_available() is True

    def test_remote_browser_requires_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("remote_browser")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("REMOTE_BROWSER_API_KEY", "key")
        assert p.is_available() is True


class TestRemoteBrowserProviderConfig:
    """Remote Browser keeps secrets in env and behavioral settings in config."""

    def test_reads_remote_browser_config_section(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hermes_cli import config as hermes_config
        from plugins.browser.remote_browser.provider import RemoteBrowserProvider

        monkeypatch.setenv("REMOTE_BROWSER_API_KEY", "rb_secret")
        monkeypatch.setattr(
            hermes_config,
            "read_raw_config",
            lambda: {
                "browser": {
                    "remote_browser": {
                        "base_url": "https://example.test/",
                        "timeout_minutes": 9,
                        "poll_interval_seconds": 3,
                        "recording_enabled": False,
                        "launch_arguments": ["--one", "--two"],
                    }
                }
            },
        )

        config = RemoteBrowserProvider()._get_config()
        assert config["api_key"] == "rb_secret"
        assert config["base_url"] == "https://example.test"
        assert config["timeout_minutes"] == 9
        assert config["poll_interval_seconds"] == 3
        assert config["recording_enabled"] is False
        assert config["launch_arguments"] == ["--one", "--two"]

    def test_cdp_url_uses_path_api_key_not_query_token(self) -> None:
        from plugins.browser.remote_browser.provider import RemoteBrowserProvider

        provider = RemoteBrowserProvider()
        cdp_url = provider._apply_api_key_to_cdp_url(
            "wss://brapi.example/cdp/rb_123?token=old&keep=yes",
            "rb_secret/with space",
        )

        assert cdp_url == (
            "wss://brapi.example/cdp/rb_123/api-key/"
            "rb_secret%2Fwith%20space?keep=yes"
        )


# ---------------------------------------------------------------------------
# Registry resolution semantics
# ---------------------------------------------------------------------------


class TestRegistryResolution:
    """``_resolve()`` implements the documented three-rule precedence."""

    def test_resolve_none_with_no_creds_returns_none(self) -> None:
        """No config, no env → local mode (None)."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        assert _resolve(None) is None

    def test_explicit_local_returns_none(self) -> None:
        """``cloud_provider: local`` is a positive choice; short-circuits to None."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        assert _resolve("local") is None


    def test_legacy_walk_prefers_browser_use_over_browserbase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule 3: walk order is browser-use → browserbase."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        # Both available — browser-use should win.
        monkeypatch.setenv("BROWSER_USE_API_KEY", "k1")
        monkeypatch.setenv("BROWSERBASE_API_KEY", "k2")
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "p")

        provider = _resolve(None)
        assert provider is not None
        assert provider.name == "browser-use"


# ---------------------------------------------------------------------------
# Legacy ABC backward-compat aliases (is_configured / provider_name)
# ---------------------------------------------------------------------------


class TestLegacyAbcAliases:
    """is_configured() and provider_name() delegate to the new API."""

    @pytest.mark.parametrize(
        "plugin_name",
        ["browserbase", "browser-use", "firecrawl", "remote_browser"],
    )
    def test_is_configured_delegates_to_is_available(self, plugin_name: str) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider(plugin_name)
        assert p is not None
        assert p.is_configured() is p.is_available()

    @pytest.mark.parametrize(
        "plugin_name,expected_label",
        [
            ("browserbase", "Browserbase"),
            ("browser-use", "Browser Use"),
            ("firecrawl", "Firecrawl"),
            ("remote_browser", "Remote Browser"),
        ],
    )
    def test_provider_name_returns_display_name(
        self, plugin_name: str, expected_label: str
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider(plugin_name)
        assert p is not None
        assert p.provider_name() == expected_label


# ---------------------------------------------------------------------------
# Picker integration
# ---------------------------------------------------------------------------


class TestPickerIntegration:
    """`_plugin_browser_providers()` exposes all plugins as picker rows."""

    def test_picker_rows_match_registered_plugins(self) -> None:
        _ensure_plugins_loaded()
        from hermes_cli.tools_config import _plugin_browser_providers

        rows = _plugin_browser_providers()
        names = sorted(r.get("browser_provider") for r in rows)
        assert names == ["browser-use", "browserbase", "firecrawl", "remote_browser"]
