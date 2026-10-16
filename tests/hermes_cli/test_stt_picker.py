"""Tests for the Speech-to-Text category in `hermes tools` (tools_config).

Covers the STT provider picker rows, config writes (stt.provider /
use_gateway), the model picker catalog, config-only checklist exclusion,
and the faster_whisper post-setup readiness hook.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.tools_config import (
    STT_MODEL_CATALOG,
    TOOL_CATEGORIES,
    _checklist_toolset_keys,
    _configure_stt_model,
    _is_provider_active,
    _visible_providers,
    _write_provider_config,
    apply_provider_selection,
)

import pytest

from agent import transcription_registry
from agent.transcription_provider import TranscriptionProvider
from hermes_cli import tools_config

def _stt_cat():
    return TOOL_CATEGORIES["stt"]

def _stt_provider_named(name):
    return next(p for p in _stt_cat()["providers"] if p["name"] == name)

class TestConfigWrites:
    def test_write_provider_config_sets_stt_provider(self):
        config = {"stt": {"use_gateway": True}}
        prov = _stt_provider_named("Groq")
        _write_provider_config(prov, config, managed_feature=None)
        assert config["stt"]["provider"] == "groq"
        # Legacy key is popped so the read-time shim can't override the pick.
        assert "use_gateway" not in config["stt"]

    def test_apply_provider_selection_stt(self):
        config = {}
        with patch(
            "hermes_cli.tools_config.get_nous_subscription_features"
        ) as feats:
            feats.return_value = MagicMock(
                nous_auth_present=False, account_info=None
            )
            apply_provider_selection("stt", "OpenAI", config)
        assert config["stt"]["provider"] == "openai"

class TestActiveDetection:
    def test_active_matches_config(self):
        config = {"stt": {"provider": "groq"}}
        assert _is_provider_active(_stt_provider_named("Groq"), config)
        assert not _is_provider_active(_stt_provider_named("OpenAI"), config)

    def test_unset_provider_defaults_to_local(self):
        assert _is_provider_active(_stt_provider_named("Local Whisper"), {})

class TestModelPicker:

    def test_every_model_picker_lists_the_runtime_catalog(self):
        """Dashboard selects and Desktop suggestions offer exactly STT_MODEL_CATALOG per provider."""
        import re

        from hermes_cli.web_server_config import _SCHEMA_OVERRIDES
        from tools.transcription_common import BUILTIN_STT_PROVIDERS, STT_MODEL_CONFIG_KEY

        desktop = (Path(__file__).resolve().parents[2] / "apps/desktop/src/app/settings/constants.ts").read_text()
        for provider, models in STT_MODEL_CATALOG.items():
            key = f"stt.{provider}.{STT_MODEL_CONFIG_KEY.get(provider, 'model')}"
            listed = re.search(rf"'{re.escape(key)}': \[([^\]]*)\]", desktop)
            assert listed, key
            desktop_models = re.findall(r"'([^']+)'", listed.group(1))
            # Local sizes are listed smallest-first in the UIs; the catalog puts the default first.
            same = set if provider == "local" else list
            assert same(_SCHEMA_OVERRIDES[key]["options"]) == same(models), key
            assert same(desktop_models) == same(models), key
        assert set(_SCHEMA_OVERRIDES["stt.provider"]["options"]) == BUILTIN_STT_PROVIDERS - {"local_command"}

    def test_configure_stt_model_defaults_to_current(self):
        config = {"stt": {"openai": {"model": "gpt-transcribe"}}}
        with patch(
            "hermes_cli.tools_config._prompt_choice", return_value=0
        ) as pc:
            _configure_stt_model("openai", config)
        # default index should point at the currently configured model
        args = pc.call_args[0]
        assert args[2] == STT_MODEL_CATALOG["openai"].index("gpt-transcribe")

class TestConfigOnlyExclusion:

    def test_stt_excluded_from_checklist_universe(self):
        assert "stt" not in _checklist_toolset_keys("cli")
        # sanity: tts (a real toolset) stays in
        assert "tts" in _checklist_toolset_keys("cli")


# --- Plugin-registered STT providers in the picker (issue #133223) ---
#
# A plugin provider registered via ``PluginContext.register_transcription_provider``
# is fully functional at runtime (``stt.provider: <name>`` dispatches to it), so the
# picker owes it the same row a built-in provider gets. Mirrors
# ``tests/hermes_cli/test_tts_picker.py`` on the transcription side.


class _FakeSTTPluginProvider(TranscriptionProvider):
    def __init__(self, name, schema=None):
        self._name = name
        self._schema = schema

    @property
    def name(self):
        return self._name

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        return {"success": True, "transcript": "", "provider": self._name}

    def get_setup_schema(self):
        if self._schema is not None:
            return self._schema
        return super().get_setup_schema()


@pytest.fixture(autouse=True)
def _clean_transcription_registry():
    transcription_registry._reset_for_tests()
    yield
    transcription_registry._reset_for_tests()


def _plugin_stt_rows():
    # Attribute access (not a top-level ``from ... import``) so the pre-fix tree fails
    # the plugin-row cases per test instead of erroring the whole module at collection.
    return tools_config._plugin_stt_providers()


class TestPluginSTTProviderRows:

    def test_setup_schema_becomes_a_row_with_stt_markers(self):
        schema = {
            "name": "Bifrost",
            "badge": "plugin",
            "tag": "self-hosted Whisper",
            "env_vars": [{"key": "BIFROST_API_KEY", "prompt": "Bifrost API key", "url": "https://example.invalid"}],
        }
        transcription_registry.register_provider(_FakeSTTPluginProvider(name="bifrost", schema=schema))

        rows = _plugin_stt_rows()

        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "Bifrost"
        assert row["badge"] == "plugin"
        assert row["tag"] == "self-hosted Whisper"
        assert row["env_vars"] == schema["env_vars"]
        # Both markers: selecting the row writes ``stt.provider: bifrost``, and the
        # plugin-name marker routes config writes through the registry.
        assert row["stt_provider"] == "bifrost"
        assert row["stt_plugin_name"] == "bifrost"

    def test_minimal_schema_uses_display_name(self):
        """A provider with no setup_schema override gets a row built from
        ``display_name`` and ``name`` only."""
        transcription_registry.register_provider(_FakeSTTPluginProvider(name="minimal"))

        rows = _plugin_stt_rows()

        assert len(rows) == 1
        assert rows[0]["name"] == "Minimal"
        assert rows[0]["stt_provider"] == "minimal"
        assert rows[0]["env_vars"] == []

    def test_builtin_names_are_not_offered_as_plugin_rows(self):
        """Built-in STT names have their own hardcoded rows; a plugin row of the
        same name would show the provider twice."""
        transcription_registry._providers["groq"] = _FakeSTTPluginProvider(name="groq")
        try:
            assert all(r.get("stt_provider") != "groq" for r in _plugin_stt_rows())
        finally:
            transcription_registry._providers.pop("groq", None)

    def test_skips_providers_with_no_name(self):
        """Defense in depth: a provider with no .name attribute is skipped
        rather than crashing the picker."""

        class _NoName:
            display_name = "Bogus"

            def get_setup_schema(self):
                return {"name": "Bogus"}

        transcription_registry._providers["bogus"] = _NoName()
        try:
            assert all(r.get("stt_provider") != "bogus" for r in _plugin_stt_rows())
        finally:
            transcription_registry._providers.pop("bogus", None)


class TestPluginSTTProvidersInThePicker:

    def test_stt_category_lists_the_plugin_row(self):
        transcription_registry.register_provider(_FakeSTTPluginProvider(name="bifrost"))

        visible = _visible_stt_providers(config={})

        assert "Bifrost" in [row.get("name") for row in visible]
        plugin_rows = [r for r in visible if r.get("stt_plugin_name")]
        assert len(plugin_rows) == 1
        assert plugin_rows[0]["stt_provider"] == "bifrost"

    def test_plugin_row_is_selectable_and_reported_active(self):
        """The marker written by the row is the one active detection reads."""
        transcription_registry.register_provider(_FakeSTTPluginProvider(name="bifrost"))

        config = {"stt": {"provider": "bifrost"}}
        plugin_rows = [r for r in _visible_stt_providers(config=config) if r.get("stt_plugin_name")]

        assert len(plugin_rows) == 1
        assert _is_provider_active(plugin_rows[0], config)

    def test_plugin_rows_do_not_leak_into_other_categories(self):
        transcription_registry.register_provider(_FakeSTTPluginProvider(name="bifrost"))

        for key in ("tts", "image_gen"):
            names = [row.get("name") for row in _visible_providers(TOOL_CATEGORIES[key], config={})]
            assert "Bifrost" not in names


def _visible_stt_providers(config):
    return _visible_providers(_stt_cat(), config)
