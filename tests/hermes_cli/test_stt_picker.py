"""Tests for the STT plugin picker surface in hermes_cli/tools_config.py.

Covers ``_plugin_stt_providers()`` and the ``_visible_providers()``
integration that injects plugin rows (e.g. soniox-stt) into the
Speech-to-Text category.

Mirrors tests/hermes_cli/test_tts_picker.py (issue #30398 pattern).
"""

from __future__ import annotations

import pytest

from agent import transcription_registry
from agent.transcription_provider import TranscriptionProvider
from hermes_cli import tools_config


class _FakeSTTProvider(TranscriptionProvider):
    def __init__(self, name: str, schema: dict | None = None, streaming: bool = False):
        self._name = name
        self._schema = schema
        self._streaming = streaming

    @property
    def name(self) -> str:
        return self._name

    @property
    def streaming_capable(self) -> bool:
        return self._streaming

    def transcribe(self, file_path: str, **kw):
        return {"success": True, "transcript": "", "provider": self._name}

    def get_setup_schema(self):
        if self._schema is not None:
            return self._schema
        return super().get_setup_schema()


@pytest.fixture(autouse=True)
def _reset_registry():
    transcription_registry._reset_for_tests()
    yield
    transcription_registry._reset_for_tests()


class TestPluginSTTProviders:
    """``_plugin_stt_providers()`` returns picker-row dicts."""

    def test_minimal_schema_uses_display_name(self):
        """A provider with no setup_schema override gets a row built from
        ``display_name`` and ``name`` only."""
        transcription_registry.register_provider(_FakeSTTProvider(name="minimal"))
        rows = tools_config._plugin_stt_providers()
        assert len(rows) == 1
        assert rows[0]["name"] == "Minimal"  # display_name default
        assert rows[0]["stt_provider"] == "minimal"
        assert rows[0]["env_vars"] == []

    def test_schema_env_vars_and_streaming_row(self):
        """A provider with a full setup schema (env var prompt) maps into the
        picker row that the generic selection path persists."""
        provider = _FakeSTTProvider(
            name="soniox",
            streaming=True,
            schema={
                "name": "Soniox",
                "badge": "paid",
                "tag": "live-streaming STT",
                "env_vars": [
                    {"key": "SONIOX_API_KEY", "prompt": "Soniox API key", "url": "https://console.soniox.com"},
                ],
            },
        )
        transcription_registry.register_provider(provider)
        rows = tools_config._plugin_stt_providers()
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "Soniox"
        assert row["badge"] == "paid"
        assert row["stt_provider"] == "soniox"
        assert row["env_vars"][0]["key"] == "SONIOX_API_KEY"

    def test_builtin_shadowing_filtered(self):
        """A plugin named like a built-in STT provider never reaches the picker."""
        transcription_registry.register_provider(_FakeSTTProvider(name="openai"))
        assert tools_config._plugin_stt_providers() == []

    def test_skips_providers_with_no_name(self):
        class _NoName:
            display_name = "Bogus"

            def get_setup_schema(self):
                return {"name": "Bogus"}

        transcription_registry._providers["bogus"] = _NoName()  # type: ignore[assignment]
        try:
            rows = tools_config._plugin_stt_providers()
            assert all(r.get("stt_plugin_name") != "bogus" for r in rows)
        finally:
            transcription_registry._providers.pop("bogus", None)  # type: ignore[arg-type]


class TestVisibleProvidersInjectsSTTPlugins:
    """``_visible_providers()`` injects plugin rows into the Speech-to-Text
    category alongside the hardcoded built-ins."""

    def test_stt_category_contains_plugin_rows(self):
        transcription_registry.register_provider(_FakeSTTProvider(name="soniox"))
        cat = tools_config.TOOL_CATEGORIES["stt"]
        visible = tools_config._visible_providers(cat, config={})
        names = [row.get("name") for row in visible]
        # Hardcoded rows still present.
        assert "Local Whisper" in names
        # Plugin row injected with the stt_provider write-path key.
        plugin_rows = [row for row in visible if row.get("stt_plugin_name")]
        assert len(plugin_rows) == 1
        assert plugin_rows[0]["stt_provider"] == "soniox"
