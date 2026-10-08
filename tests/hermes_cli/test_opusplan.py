"""Provider-aware ``opusplan``: plan/exec pair resolution, ``/model opusplan`` and launch-time routing.

``opusplan`` runs the main session on the active provider's plan model and delegation on its exec model.
The pair comes from ``providers.<name>.opusplan`` first, the provider plugin's ``opus``/``sonnet`` aliases
second, and is otherwise an error naming the provider (never a guess).
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hermes_cli.model_switch import model_selection_config_updates, switch_model
from hermes_cli.opusplan import (
    OpusplanError, ROLE_EXEC, ROLE_PLAN, is_opusplan, resolve_model_in_config, resolve_opusplan_model,
    resolve_opusplan_pair, resolve_startup_model, worker_pin)
from providers.base import ProviderProfile

ROUTER_URL = "http://192.168.10.13:4000/v1"
ROUTER = {"base_url": ROUTER_URL, "api_key": "sk-x",
          "opusplan": {"plan": "GLM-5.3-Flash-850K", "exec": "Qwen3.8FlashNext"}}
CLAUDE_LIKE = ProviderProfile(
    name="claude-like", auth_type="external_process", base_url="process://claude-like",
    fallback_models=("claude-opus-5-5[1m]", "claude-sonnet-5-5[1m]"),
    model_aliases={"opus": "claude-opus-5-5[1m]", "sonnet": "claude-sonnet-5-5[1m]", "haiku": "claude-haiku-5-5[1m]"})
_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _write_config(home, cfg):
    (home / "config.yaml").write_text(json.dumps(cfg))  # JSON is valid YAML


class TestResolvePair:
    def test_config_block_wins_for_a_custom_provider(self):
        assert resolve_opusplan_pair(["ecc-router"], user_providers={"ecc-router": ROUTER}) == (
            "GLM-5.3-Flash-850K", "Qwen3.8FlashNext")

    @pytest.mark.parametrize("name", ["custom:ecc-router", "ECC-Router", "auto"])
    def test_provider_identity_forms_find_the_same_entry(self, name):
        names = [name, "ecc-router"] if name == "auto" else [name]
        assert resolve_opusplan_pair(names, user_providers={"ecc-router": ROUTER}) == (
            "GLM-5.3-Flash-850K", "Qwen3.8FlashNext")

    def test_endpoint_matches_the_entry_when_the_session_only_knows_bare_custom(self):
        assert resolve_opusplan_pair(["custom"], base_url=ROUTER_URL + "/", user_providers={"ecc-router": ROUTER}) == (
            "GLM-5.3-Flash-850K", "Qwen3.8FlashNext")

    def test_config_block_outranks_plugin_aliases(self):
        override = {"claude-like": {"opusplan": {"plan": "big", "exec": "small"}}}
        with patch("providers.get_provider_profile", return_value=CLAUDE_LIKE):
            assert resolve_opusplan_pair(["claude-like"], user_providers=override) == ("big", "small")

    def test_plugin_aliases_are_the_zero_config_fallback(self):
        with patch("providers.get_provider_profile", return_value=CLAUDE_LIKE):
            assert resolve_opusplan_pair(["claude-like"], user_providers={}) == (
                "claude-opus-5-5[1m]", "claude-sonnet-5-5[1m]")

    def test_plugin_with_only_one_alias_is_not_a_pair(self):
        half = ProviderProfile(name="half", model_aliases={"opus": "o"})
        with patch("providers.get_provider_profile", return_value=half), pytest.raises(OpusplanError, match="'half'"):
            resolve_opusplan_pair(["half"], user_providers={})

    def test_unknown_provider_errors_naming_it_and_the_config_block(self):
        with patch("providers.get_provider_profile", return_value=None), pytest.raises(OpusplanError) as err:
            resolve_opusplan_pair(["acme-gw"], user_providers={})
        message = str(err.value)
        assert "'acme-gw'" in message and "providers" in message and "opusplan" in message
        assert "plan:" in message and "exec:" in message

    @pytest.mark.parametrize("block", [{"plan": "only-plan"}, {"exec": "only-exec"}, {}, "glm"])
    def test_incomplete_block_is_an_error_not_a_guess(self, block):
        with pytest.raises(OpusplanError, match="providers.router.opusplan must set both"):
            resolve_opusplan_pair(["router"], user_providers={"router": {"base_url": ROUTER_URL, "opusplan": block}})

    def test_role_selector(self):
        providers = {"ecc-router": ROUTER}
        assert resolve_opusplan_model(ROLE_PLAN, ["ecc-router"], user_providers=providers) == "GLM-5.3-Flash-850K"
        assert resolve_opusplan_model(ROLE_EXEC, ["ecc-router"], user_providers=providers) == "Qwen3.8FlashNext"
        with pytest.raises(ValueError):
            resolve_opusplan_model("boss", ["ecc-router"], user_providers=providers)

    def test_keyword_detection(self):
        assert is_opusplan("opusplan") and is_opusplan(" OpusPlan ")
        assert not is_opusplan("opus") and not is_opusplan(None) and not is_opusplan("opusplan-v2")


class TestConfigHelpers:
    CFG = {"model": {"default": "opusplan", "provider": "ecc-router"}, "providers": {"ecc-router": ROUTER}}

    def test_plain_model_passes_through_untouched(self):
        assert resolve_model_in_config("gpt-5", ROLE_PLAN, self.CFG) == "gpt-5"
        assert resolve_model_in_config("", ROLE_PLAN, self.CFG) == ""

    def test_default_provider_comes_from_model_section(self):
        assert resolve_model_in_config("opusplan", ROLE_PLAN, self.CFG) == "GLM-5.3-Flash-850K"
        assert resolve_model_in_config("opusplan", ROLE_EXEC, self.CFG) == "Qwen3.8FlashNext"

    def test_requested_provider_beats_the_config_provider(self):
        cfg = {**self.CFG, "providers": {**self.CFG["providers"], "other": {
            "base_url": "http://x/v1", "opusplan": {"plan": "p2", "exec": "e2"}}}}
        assert resolve_model_in_config("opusplan", ROLE_EXEC, cfg, "other") == "e2"

    def test_startup_model_reports_whether_opusplan_was_used(self):
        assert resolve_startup_model("opusplan", ROLE_PLAN, "ecc-router", cfg=self.CFG) == ("GLM-5.3-Flash-850K", True)
        assert resolve_startup_model("qwen", ROLE_PLAN, "ecc-router", cfg=self.CFG) == ("qwen", False)

    def test_worker_pin_only_for_opusplan_profiles(self):
        assert worker_pin(self.CFG) == ("Qwen3.8FlashNext", "ecc-router")
        assert worker_pin({"model": {"default": "gpt-5", "provider": "openai"}}) is None
        auto = {**self.CFG, "model": {"default": "opusplan", "provider": "auto"}}
        with patch("hermes_cli.auth.resolve_provider", return_value="ecc-router"):
            assert worker_pin(auto) == ("Qwen3.8FlashNext", "")


def _switch(raw, *, provider, providers, profile=None, base_url=ROUTER_URL):
    with patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("providers.get_provider_profile", return_value=profile), \
         patch("hermes_cli.model_switch.get_authenticated_provider_slugs", return_value=[provider]), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "k", "base_url": base_url, "api_mode": "chat_completions"}):
        return switch_model(raw_input=raw, current_provider=provider, current_model="old", current_base_url=base_url,
                            user_providers=providers, custom_providers=[])


class TestSlashModel:
    def test_model_opusplan_selects_the_plan_model_of_a_custom_provider(self):
        result = _switch("opusplan", provider="ecc-router", providers={"ecc-router": ROUTER})
        assert result.success, result.error_message
        assert (result.new_model, result.opusplan) == ("GLM-5.3-Flash-850K", True)

    def test_model_opusplan_uses_plugin_aliases_on_a_process_provider(self):
        result = _switch("opusplan", provider="claude-like", providers={}, profile=CLAUDE_LIKE,
                         base_url="process://claude-like")
        assert result.success, result.error_message
        assert (result.new_model, result.opusplan) == ("claude-opus-5-5[1m]", True)

    def test_explicit_provider_flag_picks_that_providers_pair(self):
        other = {"ecc-router": ROUTER, "lab": {"base_url": "http://lab/v1", "api_key": "k",
                                               "opusplan": {"plan": "lab-big", "exec": "lab-small"}}}
        assert _switch("opusplan", provider="ecc-router", providers=other).new_model == "GLM-5.3-Flash-850K"
        with patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
             patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
             patch("hermes_cli.model_switch.get_model_info", return_value=None), \
             patch("hermes_cli.model_switch.get_model_capabilities", return_value=None):
            explicit = switch_model(raw_input="opusplan", current_provider="ecc-router", current_model="x",
                                    current_base_url=ROUTER_URL, explicit_provider="lab", user_providers=other,
                                    custom_providers=[])
        assert explicit.success, explicit.error_message
        assert (explicit.new_model, explicit.target_provider) == ("lab-big", "lab")

    def test_explicit_provider_without_a_pair_does_not_borrow_the_current_one(self):
        no_pair = {"ecc-router": ROUTER, "lab": {"base_url": "http://lab/v1", "api_key": "k"}}
        with patch("providers.get_provider_profile", return_value=None):
            result = switch_model(raw_input="opusplan", current_provider="ecc-router", current_model="x",
                                  current_base_url=ROUTER_URL, explicit_provider="lab", user_providers=no_pair,
                                  custom_providers=[])
        assert not result.success and "'lab'" in result.error_message

    def test_provider_without_a_pair_fails_cleanly_naming_it(self):
        result = _switch("opusplan", provider="acme-gw", providers={"acme-gw": {"base_url": "http://acme/v1"}})
        assert not result.success
        assert "'acme-gw'" in result.error_message and "opusplan" in result.error_message

    def test_ordinary_models_are_not_flagged(self):
        result = _switch("qwen3", provider="ecc-router", providers={"ecc-router": ROUTER})
        assert result.opusplan is False

    def test_global_persist_keeps_the_keyword_so_the_split_survives_restart(self):
        result = _switch("opusplan", provider="ecc-router", providers={"ecc-router": ROUTER})
        updates = model_selection_config_updates(result, {"default": "old", "provider": "ecc-router"})
        assert updates["default"] == "opusplan"
        plain = _switch("qwen3", provider="ecc-router", providers={"ecc-router": ROUTER})
        assert model_selection_config_updates(plain, {})["default"] == "qwen3"


class TestLaunchRouting:
    CFG = {"model": {"default": "opusplan", "provider": "ecc-router"}, "providers": {"ecc-router": ROUTER}}

    def test_gateway_default_model_is_the_plan_model(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model(self.CFG) == "GLM-5.3-Flash-850K"
        assert _resolve_gateway_model({"model": "qwen"}) == "qwen"

    def test_tui_default_model_is_the_plan_model(self, tmp_path, monkeypatch):
        import tui_gateway.server as server
        monkeypatch.delenv("HERMES_TUI_MODEL", raising=False)
        monkeypatch.setattr(server, "_load_cfg", lambda: self.CFG)
        monkeypatch.setattr(server, "_env_model_seed", lambda: "")
        assert server._resolve_model() == "GLM-5.3-Flash-850K"

    def test_oneshot_flag_and_config_both_resolve_to_the_plan_model(self):
        from hermes_cli.oneshot import _resolve_model_and_provider
        assert _resolve_model_and_provider(self.CFG, None, None).model == "GLM-5.3-Flash-850K"
        assert _resolve_model_and_provider(self.CFG, "opusplan", None).model == "GLM-5.3-Flash-850K"

    def test_auxiliary_main_model_reader_never_sees_the_keyword(self, tmp_path):
        from agent import auxiliary_client
        from hermes_constants import get_hermes_home
        _write_config(get_hermes_home(), self.CFG)
        with patch.object(auxiliary_client, "_runtime_main_value", return_value=None):
            assert auxiliary_client._read_main_model() == "GLM-5.3-Flash-850K"

    def test_provider_block_with_opusplan_does_not_trip_the_unknown_key_warning(self, caplog):
        from hermes_cli.config_providers import _normalize_custom_provider_entry as normalize_provider_entry
        with caplog.at_level("WARNING"):
            normalize_provider_entry(ROUTER, provider_key="ecc-router")
        assert "unknown config keys" not in caplog.text
