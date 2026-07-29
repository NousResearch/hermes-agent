"""Tests for the Speech-to-Text category in `hermes tools` (tools_config).

Covers the STT provider picker rows, config writes (stt.provider /
use_gateway), the model picker catalog, config-only checklist exclusion,
and the faster_whisper post-setup readiness hook.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.tools_config import (  # noqa: E402
    _CONFIG_ONLY_TOOLSETS,
    CONFIGURABLE_TOOLSETS,
    STT_MODEL_CATALOG,
    TOOL_CATEGORIES,
    _checklist_toolset_keys,
    _configure_stt_model,
    _is_provider_active,
    _write_provider_config,
    apply_provider_selection,
)


def _stt_cat():
    return TOOL_CATEGORIES["stt"]


def _stt_provider_named(name):
    return next(p for p in _stt_cat()["providers"] if p["name"] == name)


class TestSttCategory:
    def test_stt_category_exists(self):
        cat = _stt_cat()
        assert cat["name"] == "Speech-to-Text"
        assert len(cat["providers"]) >= 5

    def test_stt_in_configurable_toolsets(self):
        keys = {k for k, _, _ in CONFIGURABLE_TOOLSETS}
        assert "stt" in keys

    def test_every_row_writes_stt_provider(self):
        for prov in _stt_cat()["providers"]:
            assert prov.get("stt_provider"), f"row {prov['name']} missing stt_provider"

    def test_provider_keys_cover_runtime_dispatch(self):
        """Every picker row's stt_provider must be a runtime built-in."""
        from agent.transcription_registry import _BUILTIN_NAMES

        for prov in _stt_cat()["providers"]:
            assert prov["stt_provider"] in _BUILTIN_NAMES

    def test_managed_row_shares_tts_coverage_category(self):
        from hermes_cli.nous_subscription import MANAGED_FEATURE_COVERAGE_CATEGORY

        managed = [p for p in _stt_cat()["providers"] if p.get("managed_nous_feature")]
        assert managed, "expected a Nous Subscription row"
        for p in managed:
            assert p["managed_nous_feature"] == "stt"
        assert MANAGED_FEATURE_COVERAGE_CATEGORY["stt"] == "openai-audio"


class TestConfigWrites:
    def test_write_provider_config_sets_stt_provider(self):
        config = {}
        prov = _stt_provider_named("Groq")
        _write_provider_config(prov, config, managed_feature=None)
        assert config["stt"]["provider"] == "groq"
        assert config["stt"]["use_gateway"] is False

    def test_write_provider_config_managed_sets_gateway(self):
        config = {}
        prov = _stt_provider_named("Nous Subscription")
        _write_provider_config(prov, config, managed_feature="stt")
        assert config["stt"]["provider"] == "openai"
        assert config["stt"]["use_gateway"] is True
        # Managed stt must not create a bogus top-level section via the
        # generic use_gateway fallback (it's in the exclusion set).
        assert set(config.keys()) == {"stt"}

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

    def test_byok_pick_clears_stale_gateway_flag(self):
        config = {"stt": {"provider": "openai", "use_gateway": True}}
        prov = _stt_provider_named("Groq")
        _write_provider_config(prov, config, managed_feature=None)
        assert config["stt"]["use_gateway"] is False


class TestActiveDetection:
    def test_active_matches_config(self):
        config = {"stt": {"provider": "groq"}}
        assert _is_provider_active(_stt_provider_named("Groq"), config)
        assert not _is_provider_active(_stt_provider_named("OpenAI"), config)

    def test_unset_provider_defaults_to_local(self):
        assert _is_provider_active(_stt_provider_named("Local Whisper"), {})


class TestModelPicker:
    def test_catalog_includes_gpt_transcribe(self):
        assert "gpt-transcribe" in STT_MODEL_CATALOG["openai"]

    def test_catalog_matches_runtime_model_sets(self):
        from tools.transcription_tools import GROQ_MODELS, OPENAI_MODELS

        assert set(STT_MODEL_CATALOG["openai"]) == OPENAI_MODELS
        assert set(STT_MODEL_CATALOG["groq"]) == GROQ_MODELS

    def test_configure_stt_model_writes_model(self):
        config = {}
        with patch("hermes_cli.tools_config._prompt_choice", return_value=3):
            _configure_stt_model("openai", config)
        assert config["stt"]["openai"]["model"] == STT_MODEL_CATALOG["openai"][3]

    def test_configure_stt_model_elevenlabs_uses_model_id(self):
        config = {}
        with patch("hermes_cli.tools_config._prompt_choice", return_value=0):
            _configure_stt_model("elevenlabs", config)
        assert config["stt"]["elevenlabs"]["model_id"] == "scribe_v2"
        assert "model" not in config["stt"]["elevenlabs"]

    def test_configure_stt_model_skips_uncataloged_provider(self):
        config = {}
        with patch("hermes_cli.tools_config._prompt_choice") as pc:
            _configure_stt_model("xai", config)
            _configure_stt_model("deepinfra", config)
        pc.assert_not_called()
        assert config == {}

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
    def test_stt_is_config_only(self):
        assert "stt" in _CONFIG_ONLY_TOOLSETS

    def test_stt_excluded_from_checklist_universe(self):
        assert "stt" not in _checklist_toolset_keys("cli")
        # sanity: tts (a real toolset) stays in
        assert "tts" in _checklist_toolset_keys("cli")


class TestPostSetup:
    def test_faster_whisper_in_post_setup_ready(self):
        from hermes_cli.tools_config import _POST_SETUP_READY

        assert "faster_whisper" in _POST_SETUP_READY
