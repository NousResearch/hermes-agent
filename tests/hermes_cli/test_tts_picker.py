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


    def test_minimal_schema_uses_display_name(self):
        """A provider with no setup_schema override gets a row built from
        ``display_name`` and ``name`` only."""
        tts_registry.register_provider(_FakeTTSProvider(name="minimal"))
        rows = tools_config._plugin_tts_providers()
        # Bundled TTS backends (e.g. plugins/tts/mimo) may also surface
        # here; pin only the row this test registers.
        minimal_rows = [r for r in rows if r.get("tts_provider") == "minimal"]
        assert len(minimal_rows) == 1
        assert minimal_rows[0]["name"] == "Minimal"  # display_name default
        assert minimal_rows[0]["env_vars"] == []



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
        cartesia_rows = [r for r in plugin_rows if r.get("tts_plugin_name") == "cartesia"]
        assert len(cartesia_rows) == 1
        assert cartesia_rows[0]["tts_provider"] == "cartesia"

    def test_other_categories_unaffected_by_tts_plugins(self):
        """Registering a TTS plugin must not leak into the Image Generation
        or Browser pickers."""
        tts_registry.register_provider(_FakeTTSProvider(name="cartesia"))

        img_cat = tools_config.TOOL_CATEGORIES["image_gen"]
        visible = tools_config._visible_providers(img_cat, config={})
        names = [row.get("name") for row in visible]
        assert "Cartesia" not in names

