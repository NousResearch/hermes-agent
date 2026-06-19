"""Plugin-side tests for the web search provider migration (PR #25182).

Covers:

- All nine bundled plugins (brave-free, camofox, ddgs, searxng, exa,
  parallel, tavily, firecrawl, xai) instantiate and self-report the expected
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
        "CAMOFOX_URL",
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
    """All nine bundled web plugins discover and register correctly."""

    def test_all_bundled_plugins_present_in_registry(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import list_providers

        names = sorted(p.name for p in list_providers())
        assert names == [
            "brave-free",
            "camofox",
            "ddgs",
            "exa",
            "firecrawl",
            "parallel",
            "searxng",
            "tavily",
            "xai",
        ]

    @pytest.mark.parametrize(
        "plugin_name,expected_search,expected_extract",
        [
            ("brave-free", True, False),
            ("camofox", True, True),
            ("ddgs", True, False),
            ("searxng", True, False),
            ("exa", True, True),
            ("parallel", True, True),
            ("tavily", True, True),
            ("firecrawl", True, True),
            # xai: search-only via Grok's agentic web_search tool.
            ("xai", True, False),
        ],
    )
    def test_capability_flags_match_spec(
        self,
        plugin_name: str,
        expected_search: bool,
        expected_extract: bool,
    ) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        provider = get_provider(plugin_name)
        assert provider is not None, f"plugin {plugin_name!r} not registered"
        assert provider.supports_search() is expected_search
        assert provider.supports_extract() is expected_extract

    @pytest.mark.parametrize(
        "plugin_name",
        ["brave-free", "camofox", "ddgs", "searxng", "exa", "parallel", "tavily", "firecrawl", "xai"],
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
        ["brave-free", "camofox", "ddgs", "searxng", "exa", "parallel", "tavily", "firecrawl", "xai"],
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

    def test_xai_requires_api_key_or_oauth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """xAI needs XAI_API_KEY or OAuth tokens in auth.json."""
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("xai")
        assert p is not None
        assert p.is_available() is False  # no XAI_API_KEY, no auth.json
        monkeypatch.setenv("XAI_API_KEY", "real")
        assert p.is_available() is True

    def test_camofox_requires_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("camofox")
        assert p is not None
        assert p.is_available() is False
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        assert p.is_available() is True

    def test_camofox_setup_schema_reuses_camofox_install_flow(self) -> None:
        """Selecting Web Camofox should run the same setup hook as Browser Camofox."""
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("camofox")
        assert p is not None
        schema = p.get_setup_schema()
        assert schema["post_setup"] == "camofox"
        assert schema["env_vars"] == [
            {
                "key": "CAMOFOX_URL",
                "prompt": "Camofox server URL",
                "default": "http://localhost:9377",
                "url": "https://github.com/jo-inc/camofox-browser",
            }
        ]


class TestCamofoxProviderUsesSharedBrowserLayer:
    def test_search_uses_shared_tab_helpers_and_closes_tab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        calls: List[Any] = []

        def fake_create_tab(user_id: str, session_key: str, url: str | None = None, **kwargs: Any) -> Dict[str, Any]:
            calls.append(("create", user_id, session_key, url, kwargs))
            return {"tabId": "tab-web"}

        def fake_navigate(user_id: str, tab_id: str, url: str, **kwargs: Any) -> Dict[str, Any]:
            calls.append(("navigate", user_id, tab_id, url, kwargs))
            return {"url": url}

        def fake_snapshot(user_id: str, tab_id: str, **kwargs: Any) -> Dict[str, Any]:
            calls.append(("snapshot", user_id, tab_id, kwargs))
            return {
                "snapshot": (
                    '- heading "Example Result" [level=2]\n'
                    '  - link "Example"\n'
                    '    /url: https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com\n'
                    '  - text: Useful summary'
                )
            }

        def fake_close_tab(user_id: str, tab_id: str, **kwargs: Any) -> None:
            calls.append(("close", user_id, tab_id, kwargs))

        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", fake_create_tab)
        monkeypatch.setattr(mod, "camofox_tab_navigate", fake_navigate)
        monkeypatch.setattr(
            mod,
            "camofox_tab_evaluate",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("old server")),
        )
        monkeypatch.setattr(mod, "camofox_tab_snapshot", fake_snapshot)
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", fake_close_tab)

        result = CamofoxWebSearchProvider().search("example", limit=1)

        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://example.com"
        assert [call[0] for call in calls] == ["create", "navigate", "snapshot", "close"]
        assert calls[0][2] == "web_search"
        assert calls[0][3] is None

    def test_search_closes_tab_when_navigation_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        closed: List[tuple[str, str]] = []

        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", lambda *args, **kwargs: {"tabId": "tab-web"})

        def fail_navigate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("Camofox request failed")

        monkeypatch.setattr(mod, "camofox_tab_navigate", fail_navigate)
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda user_id, tab_id, **kwargs: closed.append((user_id, tab_id)))

        result = CamofoxWebSearchProvider().search("example", limit=1)

        assert result["success"] is False
        assert "Camofox search failed" in result["error"]
        assert len(closed) == 1
        assert closed[0][1] == "tab-web"

    def test_extract_returns_error_when_tab_creation_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        def fail_create_tab(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("Camofox did not return a tabId")

        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", fail_create_tab)
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda *args, **kwargs: pytest.fail("nothing to close"))

        result = CamofoxWebSearchProvider().extract(["https://example.com"])

        assert result == [
            {
                "url": "https://example.com",
                "title": "",
                "content": "",
                "raw_content": "",
                "error": "Camofox extract failed: Camofox did not return a tabId",
            }
        ]


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

    def test_no_config_no_credentials_returns_legacy_firecrawl_default(
        self,
    ) -> None:
        """No backend configured and no available providers → Firecrawl default.

        This preserves legacy ``tools.web_tools._get_backend()`` behavior: the
        dispatcher still selects Firecrawl so missing-credential calls surface
        Firecrawl's precise setup error instead of a generic "no provider".
        """
        _ensure_plugins_loaded()
        from agent.web_search_registry import _resolve

        result = _resolve(None, capability="search")
        assert result is not None
        assert result.name == "firecrawl"
        assert result.is_available() is False


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

    def test_firecrawl_config_error_points_paid_users_to_nous_subscription(self, monkeypatch):
        from plugins.web.firecrawl import provider as firecrawl_provider

        monkeypatch.setattr(
            "tools.web_tools.managed_nous_tools_enabled",
            lambda: True,
            raising=False,
        )

        with pytest.raises(ValueError) as exc_info:
            firecrawl_provider._raise_web_backend_configuration_error()

        message = str(exc_info.value)
        assert "With your Nous subscription you can also use the Tool Gateway" in message
        assert "select Nous Subscription as the web provider" in message
        assert "managed Firecrawl web tools is unavailable" not in message

    def test_firecrawl_config_error_uses_entitlement_message_when_not_paid(self, monkeypatch):
        from plugins.web.firecrawl import provider as firecrawl_provider

        monkeypatch.setattr(
            "tools.web_tools.managed_nous_tools_enabled",
            lambda: False,
            raising=False,
        )
        monkeypatch.setattr(
            "tools.web_tools.nous_tool_gateway_unavailable_message",
            lambda capability: f"{capability} denied by test entitlement.",
            raising=False,
        )

        with pytest.raises(ValueError) as exc_info:
            firecrawl_provider._raise_web_backend_configuration_error()

        assert "managed Firecrawl web tools denied by test entitlement" in str(exc_info.value)

    def test_xai_search_returns_error_dict_when_unconfigured(self) -> None:
        """xAI returns a typed error dict (no XAI_API_KEY)."""
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        p = get_provider("xai")
        assert p is not None
        result = p.search("test", limit=5)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
