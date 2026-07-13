"""Tests for the TTS plugin picker surface in hermes_cli/tools_config.py (issue #30398).

Covers ``_plugin_tts_providers()`` and the ``_visible_providers()``
integration that injects plugin rows into the Text-to-Speech category.

Mirrors the structure of existing image_gen / browser picker tests.
"""

from __future__ import annotations

import pytest

from agent import tts_registry
from agent.tts_provider import TTSProvider
from hermes_cli import tools_config


class _FakeTTSProvider(TTSProvider):
    def __init__(self, name: str, schema: dict | None = None):
        self._name = name
        self._schema = schema

    @property
    def name(self) -> str:
        return self._name

    def synthesize(self, text, output_path, **kw):
        return output_path

    def get_setup_schema(self):
        if self._schema is not None:
            return self._schema
        return super().get_setup_schema()


@pytest.fixture(autouse=True)
def _reset_registry():
    tts_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()


class TestPluginTTSProviders:
    """``_plugin_tts_providers()`` returns picker-row dicts."""

    def test_empty_when_no_plugins(self):
        assert tools_config._plugin_tts_providers() == []

    def test_returns_row_for_registered_plugin(self):
        tts_registry.register_provider(
            _FakeTTSProvider(
                name="cartesia",
                schema={
                    "name": "Cartesia",
                    "badge": "paid",
                    "tag": "Ultra-low-latency streaming",
                    "env_vars": [
                        {"key": "CARTESIA_API_KEY", "prompt": "Cartesia API key",
                         "url": "https://play.cartesia.ai/console"},
                    ],
                },
            )
        )
        rows = tools_config._plugin_tts_providers()
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "Cartesia"
        assert row["badge"] == "paid"
        assert row["tag"] == "Ultra-low-latency streaming"
        assert row["env_vars"][0]["key"] == "CARTESIA_API_KEY"
        # Selecting this row writes ``tts.provider: cartesia`` — same
        # write path as a hardcoded row.
        assert row["tts_provider"] == "cartesia"
        assert row["tts_plugin_name"] == "cartesia"

    def test_filters_builtin_shadow_defensively(self):
        """Even if a plugin slipped past the registry's built-in check
        (e.g. via direct ``agent.tts_registry.register_provider`` rather
        than the ``ctx.register_tts_provider`` hook), the picker layer
        filters it out so the picker invariant holds."""
        # Use lower-level call to bypass the warning + skip in
        # register_provider (the registry's built-in guard).
        # Note: this is intentionally pathological — production code
        # paths go through the hook which catches this first.
        provider = _FakeTTSProvider(name="edge")
        tts_registry._providers["edge"] = provider  # type: ignore[index]
        try:
            rows = tools_config._plugin_tts_providers()
            assert rows == [], (
                "Picker must filter built-in name shadows even when the "
                "registry has been bypassed."
            )
        finally:
            tts_registry._providers.pop("edge", None)  # type: ignore[arg-type]

    def test_skips_providers_with_no_name(self):
        """Defense in depth: a provider with no .name attribute is skipped
        rather than crashing the picker."""

        class _NoName:
            display_name = "Bogus"
            def get_setup_schema(self):
                return {"name": "Bogus"}

        tts_registry._providers["bogus"] = _NoName()  # type: ignore[assignment]
        try:
            rows = tools_config._plugin_tts_providers()
            # Provider has no .name so the picker filters it out
            assert all(r.get("tts_plugin_name") != "bogus" for r in rows)
        finally:
            tts_registry._providers.pop("bogus", None)  # type: ignore[arg-type]

    def test_skips_providers_whose_schema_raises(self):
        class _ExplodingSchema(_FakeTTSProvider):
            def get_setup_schema(self):
                raise RuntimeError("boom")

        tts_registry.register_provider(_ExplodingSchema(name="exploding"))
        tts_registry.register_provider(_FakeTTSProvider(name="working"))
        rows = tools_config._plugin_tts_providers()
        assert [r["tts_plugin_name"] for r in rows] == ["working"]

    def test_minimal_schema_uses_display_name(self):
        """A provider with no setup_schema override gets a row built from
        ``display_name`` and ``name`` only."""
        tts_registry.register_provider(_FakeTTSProvider(name="minimal"))
        rows = tools_config._plugin_tts_providers()
        assert len(rows) == 1
        assert rows[0]["name"] == "Minimal"  # display_name default
        assert rows[0]["tts_provider"] == "minimal"
        assert rows[0]["env_vars"] == []

    def test_post_setup_passthrough(self):
        tts_registry.register_provider(
            _FakeTTSProvider(
                name="my-tts",
                schema={
                    "name": "My TTS",
                    "post_setup": "my_post_install_hook",
                    "env_vars": [],
                },
            )
        )
        rows = tools_config._plugin_tts_providers()
        assert rows[0].get("post_setup") == "my_post_install_hook"


class TestVisibleProvidersInjectsTTSPlugins:
    """``_visible_providers()`` injects plugin rows into the Text-to-Speech
    category alongside the hardcoded built-in rows."""

    def test_tts_category_includes_plugin_rows(self):
        tts_registry.register_provider(_FakeTTSProvider(name="cartesia"))

        tts_cat = tools_config.TOOL_CATEGORIES["tts"]
        visible = tools_config._visible_providers(tts_cat, config={})

        names = [row.get("name") for row in visible]
        # Hardcoded rows (sample — check at least one is present)
        assert "Microsoft Edge TTS" in names
        # Plugin row injected at the end
        assert "Cartesia" in names

        # Plugin row has tts_provider key for write-path compat
        plugin_rows = [r for r in visible if r.get("tts_plugin_name")]
        assert len(plugin_rows) == 1
        assert plugin_rows[0]["tts_provider"] == "cartesia"

    def test_other_categories_unaffected_by_tts_plugins(self):
        """Registering a TTS plugin must not leak into the Image Generation
        or Browser pickers."""
        tts_registry.register_provider(_FakeTTSProvider(name="cartesia"))

        img_cat = tools_config.TOOL_CATEGORIES["image_gen"]
        visible = tools_config._visible_providers(img_cat, config={})
        names = [row.get("name") for row in visible]
        assert "Cartesia" not in names

    def test_tts_category_without_plugins_only_hardcoded(self):
        """No plugins → picker shows exactly the hardcoded rows."""
        tts_cat = tools_config.TOOL_CATEGORIES["tts"]
        visible = tools_config._visible_providers(tts_cat, config={})
        names = [row.get("name") for row in visible]
        # No row has the plugin marker
        assert all(not row.get("tts_plugin_name") for row in visible)
        # Hardcoded rows still present (sample one of the always-visible ones)
        assert "Microsoft Edge TTS" in names


class TestBuiltinTTSSurfaceParity:
    """Every built-in TTS provider must be reachable from both CLI selection
    surfaces — the interactive ``hermes setup tts`` menu and the ``hermes
    tools`` picker (issue #35439).

    ``tests/agent/test_tts_registry.py::TestBuiltinSync`` already guards the
    two runtime-internal lists (``_BUILTIN_NAMES`` vs
    ``BUILTIN_TTS_PROVIDERS``); these tests extend the same guarantee to the
    user-facing menus, which had drifted: ``piper`` was missing from
    ``setup tts`` and ``minimax`` / ``neutts`` were missing from the picker.
    """

    @staticmethod
    def _picker_builtin_providers() -> set:
        return {
            row["tts_provider"]
            for row in tools_config.TOOL_CATEGORIES["tts"]["providers"]
            if row.get("tts_provider")
        }

    def test_setup_menu_offers_every_builtin_provider(self):
        from hermes_cli.setup import SETUP_TTS_BUILTIN_PROVIDERS
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        menu = set(SETUP_TTS_BUILTIN_PROVIDERS)
        assert menu == BUILTIN_TTS_PROVIDERS, (
            "`hermes setup tts` menu drifted from BUILTIN_TTS_PROVIDERS:\n"
            f"  missing from menu: {sorted(BUILTIN_TTS_PROVIDERS - menu)}\n"
            f"  extra in menu:     {sorted(menu - BUILTIN_TTS_PROVIDERS)}"
        )

    def test_setup_menu_has_no_duplicates(self):
        from hermes_cli.setup import SETUP_TTS_BUILTIN_PROVIDERS

        assert len(SETUP_TTS_BUILTIN_PROVIDERS) == len(set(SETUP_TTS_BUILTIN_PROVIDERS))

    def test_tools_picker_offers_every_builtin_provider(self):
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        missing = BUILTIN_TTS_PROVIDERS - self._picker_builtin_providers()
        assert not missing, (
            "`hermes tools` TTS picker is missing built-in providers: "
            f"{sorted(missing)}"
        )
