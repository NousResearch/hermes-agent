"""Plugin-side tests for the browser provider migration (PR #25214).

Covers:

- All three bundled plugins (browserbase, browser-use, firecrawl)
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
    """All three bundled browser plugins discover and register correctly."""

    def test_all_three_plugins_present_in_registry(self) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import list_providers

        names = sorted(p.name for p in list_providers())
        assert names == ["browser-use", "browserbase", "firecrawl"]

    @pytest.mark.parametrize(
        "plugin_name,expected_display",
        [
            ("browserbase", "Browserbase"),
            ("browser-use", "Browser Use"),
            ("firecrawl", "Firecrawl"),
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

    @pytest.mark.parametrize(
        "plugin_name",
        ["browserbase", "browser-use", "firecrawl"],
    )
    def test_each_plugin_has_setup_schema(self, plugin_name: str) -> None:
        """``get_setup_schema()`` returns a dict the picker can consume."""
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None
        schema = provider.get_setup_schema()
        assert isinstance(schema, dict)
        assert "name" in schema
        assert "env_vars" in schema
        # Every cloud-browser plugin needs the agent-browser post-setup hook
        # so the picker auto-installs the CLI on selection.
        assert schema.get("post_setup") == "agent_browser"

    @pytest.mark.parametrize(
        "plugin_name",
        ["browserbase", "browser-use", "firecrawl"],
    )
    def test_each_plugin_implements_full_lifecycle(self, plugin_name: str) -> None:
        """The ABC's three lifecycle methods are all overridden."""
        _ensure_plugins_loaded()
        from agent.browser_provider import BrowserProvider
        from agent.browser_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None
        # Each method must be a real override, not the ABC's NotImplementedError
        # default — we check by comparing the function reference.
        assert type(provider).create_session is not BrowserProvider.create_session
        assert type(provider).close_session is not BrowserProvider.close_session
        assert (
            type(provider).emergency_cleanup is not BrowserProvider.emergency_cleanup
        )


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

    def test_browserbase_project_id_alone_insufficient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_plugins_loaded()
        from agent.browser_registry import get_provider

        p = get_provider("browserbase")
        assert p is not None
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj")
        assert p.is_available() is False

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

    def test_explicit_browserbase_returns_provider_even_when_unavailable(self) -> None:
        """Rule 1: explicit-config wins even when credentials are missing.

        This is critical — the dispatcher needs to surface a typed
        credentials error rather than silently switching backends.
        """
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        provider = _resolve("browserbase")
        assert provider is not None
        assert provider.name == "browserbase"
        assert provider.is_available() is False  # confirms "ignoring availability"

    def test_explicit_firecrawl_returns_provider_even_when_unavailable(self) -> None:
        """Firecrawl behaves the same as browserbase under explicit config."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        provider = _resolve("firecrawl")
        assert provider is not None
        assert provider.name == "firecrawl"

    def test_explicit_unknown_falls_back_to_auto_detect(self) -> None:
        """Rule 1 miss: unknown name → fall through to legacy walk."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        # With no credentials anywhere, auto-detect should also fail.
        assert _resolve("not-a-real-provider") is None

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

    def test_legacy_walk_falls_through_to_browserbase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule 3: browser-use unavailable → browserbase picked."""
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        monkeypatch.setenv("BROWSERBASE_API_KEY", "k")
        monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "p")

        provider = _resolve(None)
        assert provider is not None
        assert provider.name == "browserbase"

    def test_firecrawl_not_in_legacy_walk_even_when_only_one_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: firecrawl is NEVER auto-selected even when single-eligible.

        Pre-PR-#25214, the dispatcher only auto-detected between Browser Use
        and Browserbase; firecrawl was reachable solely via explicit
        config. We preserve that gate because FIRECRAWL_API_KEY is shared
        with the *web* firecrawl plugin — auto-routing a web-extract user
        to a paid cloud browser would be a real behaviour regression.
        """
        _ensure_plugins_loaded()
        from agent.browser_registry import _resolve

        monkeypatch.setenv("FIRECRAWL_API_KEY", "k")

        # Only firecrawl is_available() — but it's not in the legacy walk.
        assert _resolve(None) is None


# ---------------------------------------------------------------------------
# Legacy ABC backward-compat aliases (is_configured / provider_name)
# ---------------------------------------------------------------------------


class TestLegacyAbcAliases:
    """is_configured() and provider_name() delegate to the new API."""

    @pytest.mark.parametrize(
        "plugin_name",
        ["browserbase", "browser-use", "firecrawl"],
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
    """`_plugin_browser_providers()` exposes all three plugins as picker rows."""

    def test_picker_rows_match_registered_plugins(self) -> None:
        _ensure_plugins_loaded()
        from hermes_cli.tools_config import _plugin_browser_providers

        rows = _plugin_browser_providers()
        names = sorted(r.get("browser_provider") for r in rows)
        assert names == ["browser-use", "browserbase", "firecrawl"]

    def test_picker_rows_carry_post_setup_hook(self) -> None:
        """Every browser plugin row has post_setup='agent_browser' so
        selecting it triggers the agent-browser CLI install."""
        _ensure_plugins_loaded()
        from hermes_cli.tools_config import _plugin_browser_providers

        for row in _plugin_browser_providers():
            assert row.get("post_setup") == "agent_browser", (
                f"plugin row {row['browser_provider']!r} missing post_setup hook"
            )

    def test_picker_rows_carry_browser_plugin_name_marker(self) -> None:
        """`browser_plugin_name` matches `browser_provider` so downstream
        code can route through the registry when it wants to."""
        _ensure_plugins_loaded()
        from hermes_cli.tools_config import _plugin_browser_providers

        for row in _plugin_browser_providers():
            assert row.get("browser_plugin_name") == row.get("browser_provider")
