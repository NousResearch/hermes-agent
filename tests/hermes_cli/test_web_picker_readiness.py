"""Web picker readiness must reflect registry availability, not just credential presence (#132526).

A keyless web row (empty ``env_vars``) used to ride the unconditional "ready" verdict, so the
Desktop panel advertised free-ring / native backends that dispatch can never reach. These tests
pin the row shape ``_plugin_provider_rows`` produces and the ``provider_readiness_status``
branch that consults the live registry.
"""

from __future__ import annotations

import pytest

from hermes_cli.tools_config import provider_readiness_status

_ALL_WEB_KEYS = (
    "EXA_API_KEY",
    "TAVILY_API_KEY",
    "FIRECRAWL_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "PERPLEXITY_API_KEY",
    "KEENABLE_API_KEY",
    "PARALLEL_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ALL_WEB_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def web_registry_with_bundled(monkeypatch):
    """Register the bundled exa + openai-native providers, restore the registry afterwards."""
    from agent.web_search_registry import _reset_for_tests, register_provider
    from plugins.web.exa.provider import ExaWebSearchProvider
    from plugins.web.openai_native.provider import OpenAINativeWebSearchProvider

    _reset_for_tests()
    register_provider(ExaWebSearchProvider())
    register_provider(OpenAINativeWebSearchProvider())
    yield
    _reset_for_tests()


def _free_row(backend: str, *, name: str = None) -> dict:
    """A keyless free-tier picker row, shaped like ``_plugin_provider_rows`` output."""
    return {
        "name": name or f"{backend.title()} · Free (keyless)",
        "badge": "free · no key",
        "tag": f"{backend} keyless ring",
        "env_vars": [],
        "web_tier": "free",
        "web_backend": backend,
        "web_search_plugin_name": backend,
    }


def _keyless_ring(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr(
        "agent.web_search_registry._keyless_tier_enabled", lambda: enabled
    )


class TestKeylessRowReadiness:
    def test_free_tier_row_needs_setup_when_ring_off_and_unkeyed(
        self, web_registry_with_bundled, monkeypatch
    ):
        _keyless_ring(monkeypatch, enabled=False)
        assert provider_readiness_status(_free_row("exa"), {}) == "needs_setup"

    def test_free_tier_row_ready_when_keyless_ring_on(
        self, web_registry_with_bundled, monkeypatch
    ):
        _keyless_ring(monkeypatch, enabled=True)
        assert provider_readiness_status(_free_row("exa"), {}) == "ready"

    def test_free_tier_row_ready_when_vendor_keyed(
        self, web_registry_with_bundled, monkeypatch
    ):
        _keyless_ring(monkeypatch, enabled=False)
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        assert provider_readiness_status(_free_row("exa"), {}) == "ready"

    def test_openai_native_row_needs_setup_without_codex_credentials(
        self, web_registry_with_bundled, monkeypatch
    ):
        _keyless_ring(monkeypatch, enabled=True)
        monkeypatch.setattr(
            "plugins.web.openai_native.provider.has_codex_credentials", lambda: False
        )
        row = {
            "name": "OpenAI Native Web Search (Codex Responses)",
            "badge": "native",
            "tag": "",
            "env_vars": [],
            "web_backend": "openai-native",
            "web_search_plugin_name": "openai-native",
        }
        assert provider_readiness_status(row, {}) == "needs_setup"

    def test_openai_native_row_ready_with_codex_credentials(
        self, web_registry_with_bundled, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.web.openai_native.provider.has_codex_credentials", lambda: True
        )
        row = {
            "name": "OpenAI Native Web Search (Codex Responses)",
            "badge": "native",
            "tag": "",
            "env_vars": [],
            "web_backend": "openai-native",
            "web_search_plugin_name": "openai-native",
        }
        assert provider_readiness_status(row, {}) == "ready"

    def test_unknown_backend_row_keeps_legacy_ready(
        self, web_registry_with_bundled, monkeypatch
    ):
        _keyless_ring(monkeypatch, enabled=False)
        assert provider_readiness_status(_free_row("no-such-backend"), {}) == "ready"

    def test_registry_failure_stays_ready(self, web_registry_with_bundled, monkeypatch):
        _keyless_ring(monkeypatch, enabled=True)
        monkeypatch.setattr(
            "plugins.web.exa.provider.ExaWebSearchProvider.is_keyless_available",
            lambda self: (_ for _ in ()).throw(RuntimeError("probe blew up")),
        )
        monkeypatch.setattr(
            "plugins.web.exa.provider.ExaWebSearchProvider.is_available",
            lambda self: (_ for _ in ()).throw(RuntimeError("probe blew up")),
        )
        assert provider_readiness_status(_free_row("exa"), {}) == "ready"


class TestPostSetupAndKeyedRowsUnaffected:
    def test_post_setup_row_uses_install_check_not_registry(
        self, web_registry_with_bundled, monkeypatch
    ):
        # Registry says the backend is reachable (keyed), but the post_setup install check is the
        # verdict for rows carrying a hook — the registry branch must not pre-empt it either way.
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        monkeypatch.setattr(
            "hermes_cli.tools_config_post_setup._module_installed", lambda module: False
        )
        row = _free_row("exa", name="Exa (bundled install)")
        row["post_setup"] = "ddgs"
        assert provider_readiness_status(row, {}) == "needs_setup"
        monkeypatch.setattr(
            "hermes_cli.tools_config_post_setup._module_installed", lambda module: True
        )
        assert provider_readiness_status(row, {}) == "ready"

    def test_keyed_row_env_branch_still_wins(self, monkeypatch):
        # env_vars non-empty: the credential branch answers before the registry is consulted.
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        row = {
            "name": "Tavily",
            "badge": "",
            "tag": "",
            "env_vars": [{"key": "TAVILY_API_KEY"}],
            "web_backend": "tavily",
            "web_search_plugin_name": "tavily",
        }
        assert provider_readiness_status(row, {}) == "ready"


class TestPluginRowShape:
    def test_bundled_free_rows_carry_web_backend_and_empty_env_vars(
        self, web_registry_with_bundled
    ):
        from hermes_cli.tools_config_providers import _plugin_rows_for

        rows = {row["name"]: row for row in _plugin_rows_for("web")}
        free = rows["Exa · Free (keyless)"]
        assert free["web_backend"] == "exa"
        assert free["web_tier"] == "free"
        assert free["env_vars"] == []
        paid = rows["Exa · Paid (API key)"]
        assert [e["key"] for e in paid["env_vars"]] == ["EXA_API_KEY"]
