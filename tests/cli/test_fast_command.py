"""Tests for the /fast CLI command and service-tier config handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestParseServiceTierConfig(unittest.TestCase):
    def _parse(self, raw):
        cli_mod = _import_cli()
        return cli_mod._parse_service_tier_config(raw)

    def test_fast_maps_to_priority(self):
        self.assertEqual(self._parse("fast"), "priority")
        self.assertEqual(self._parse("priority"), "priority")

    def test_flex_is_accepted(self):
        """agent.service_tier: flex was rejected as unknown and silently dropped."""
        self.assertEqual(self._parse("flex"), "flex")
        self.assertEqual(self._parse("  FLEX  "), "flex")

    def test_normal_and_unknown_still_disable(self):
        for raw in ["", "normal", "off", "standard", "bogus"]:
            self.assertIsNone(self._parse(raw), f"{raw!r} should not enable a tier")



class TestHandleFastCommand(unittest.TestCase):
    def _make_cli(self, service_tier=None):
        return SimpleNamespace(
            service_tier=service_tier,
            provider="openai-codex",
            requested_provider="openai-codex",
            model="gpt-5.4",
            _fast_command_available=lambda: True,
            agent=MagicMock(),
        )

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier=None)
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        # Bare /fast shows status, does not change config
        mock_save.assert_not_called()
        # Should have printed the status line
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("normal", printed)


    def test_normal_argument_clears_service_tier(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier="priority")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast normal")

        # Session-scoped by default: no config write.
        mock_save.assert_not_called()
        self.assertIsNone(stub.service_tier)
        self.assertIsNone(stub.agent)


    def test_unsupported_model_does_not_expose_fast(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            service_tier=None,
            provider="openai-codex",
            requested_provider="openai-codex",
            model="gpt-5.3-codex",
            _fast_command_available=lambda: False,
            agent=MagicMock(),
        )

        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        mock_save.assert_not_called()
        self.assertTrue(mock_cprint.called)


class TestPriorityProcessingModels(unittest.TestCase):
    """Verify the expanded Priority Processing model registry."""

    def test_all_documented_models_supported(self):
        from hermes_cli.models import model_supports_fast_mode

        # All OpenAI flagship models support Priority Processing — including
        # future releases (gpt-5.5, 5.6...) via pattern matching.
        supported = [
            "gpt-5.5", "gpt-5.5-mini",
            "gpt-5.4", "gpt-5.4-mini", "gpt-5.2",
            "gpt-5.1", "gpt-5", "gpt-5-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-4o-mini",
            "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
        ]
        for model in supported:
            assert model_supports_fast_mode(model), f"{model} should support fast mode"


    def test_codex_models_excluded(self):
        """Codex models route through Responses API and don't accept service_tier."""
        from hermes_cli.models import model_supports_fast_mode

        for model in ["gpt-5-codex", "gpt-5.2-codex", "gpt-5.3-codex", "gpt-5.1-codex-max"]:
            assert not model_supports_fast_mode(model), f"{model} is codex — should not expose /fast"



    def test_grok_46_supports_priority_processing(self):
        from hermes_cli.models import (
            model_supports_fast_mode,
            resolve_fast_mode_overrides,
        )

        assert model_supports_fast_mode("grok-4.6") is True
        assert model_supports_fast_mode("x-ai/grok-4.6-latest") is True
        assert model_supports_fast_mode("grok-4.5") is False
        assert resolve_fast_mode_overrides("grok-4.6") == {"service_tier": "priority"}

    def test_resolve_overrides_returns_service_tier(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        result = resolve_fast_mode_overrides("gpt-5.4")
        assert result == {"service_tier": "priority"}

        result = resolve_fast_mode_overrides("gpt-4.1")
        assert result == {"service_tier": "priority"}



class TestTwoLevelModelPaths(unittest.TestCase):
    """Aggregators route models as ``aggregator/vendor/model``.

    Only the aggregator segment was stripped, so ``openrouter/openai/gpt-4.1``
    reduced to ``openai/gpt-4.1`` and matched no prefix — service_tier was
    silently dropped for every OpenRouter-routed model.
    """

    def test_openrouter_openai_path_supports_fast_mode(self):
        from hermes_cli.models import model_supports_fast_mode

        assert model_supports_fast_mode("openrouter/openai/gpt-4.1")


class TestGoogleServiceTier(unittest.TestCase):
    """Gemini accepts a top-level ``service_tier`` on generateContent.

    See https://ai.google.dev/gemini-api/docs/flex-inference (flex) and
    https://ai.google.dev/gemini-api/docs/generate-content/priority-inference
    (priority).
    """

    def test_gemini_models_detected(self):
        from hermes_cli.models import _is_google_service_tier_model

        for model in [
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-pro",
            "google/gemini-3.5-flash",
            "openrouter/google/gemini-3.5-flash",
        ]:
            assert _is_google_service_tier_model(model), f"{model} should be tier-eligible"

    def test_non_gemini_models_not_detected(self):
        from hermes_cli.models import _is_google_service_tier_model

        for model in ["gpt-5.4", "claude-opus-4-6", "openrouter/openai/gpt-4.1"]:
            assert not _is_google_service_tier_model(model), f"{model} is not a Gemini model"


class TestFlexTier(unittest.TestCase):
    """``agent.service_tier: flex`` must survive all the way to the override dict."""

    def test_openai_flex_override(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gpt-5.4", tier="flex") == {"service_tier": "flex"}

    def test_gemini_flex_override(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides(
            "openrouter/google/gemini-3.5-flash", tier="flex"
        ) == {"service_tier": "flex"}

    def test_gemini_priority_override(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gemini-3.6-flash", tier="priority") == {
            "service_tier": "priority"
        }

    def test_anthropic_flex_returns_none(self):
        """Anthropic has no flex equivalent — speed=fast is a priority-only knob."""
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("claude-opus-4-6", tier="flex") is None

    def test_anthropic_priority_still_returns_speed(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("claude-opus-4-6", tier="priority") == {
            "speed": "fast"
        }

    def test_default_tier_is_priority(self):
        """Existing callers pass no tier and must keep priority semantics."""
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gpt-5.4") == {"service_tier": "priority"}
        assert resolve_fast_mode_overrides("claude-opus-4-6") == {"speed": "fast"}


class TestFastModeRouting(unittest.TestCase):
    def test_fast_command_exposed_for_model_even_when_provider_is_auto(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(provider="auto", requested_provider="auto", model="gpt-5.4", agent=None)

        assert cli_mod.HermesCLI._fast_command_available(stub) is True


    def test_turn_route_injects_overrides_without_provider_switch(self):
        """Fast mode should add request_overrides but NOT change the provider/runtime."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="gpt-5.4",
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            service_tier="priority",
        )

        route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        # Provider should NOT have changed
        assert route["runtime"]["provider"] == "openrouter"
        assert route["runtime"]["api_mode"] == "chat_completions"
        # But request_overrides should be set
        assert route["request_overrides"] == {"service_tier": "priority"}

    def test_turn_route_keeps_primary_runtime_when_model_has_no_fast_backend(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="gpt-5.3-codex",
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            service_tier="priority",
        )

        route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        assert route["runtime"]["provider"] == "openrouter"
        assert route.get("request_overrides") is None


class TestAnthropicFastMode(unittest.TestCase):
    """Verify Anthropic Fast Mode model support and override resolution."""

    def test_anthropic_opus_supported(self):
        from hermes_cli.models import model_supports_fast_mode

        # Native Anthropic format (hyphens)
        assert model_supports_fast_mode("claude-opus-4-6") is True
        # OpenRouter format (dots)
        assert model_supports_fast_mode("claude-opus-4.6") is True
        # With vendor prefix
        assert model_supports_fast_mode("anthropic/claude-opus-4-6") is True
        assert model_supports_fast_mode("anthropic/claude-opus-4.6") is True

    def test_anthropic_non_opus46_models_excluded(self):
        """The speed=fast parameter is gated to Opus 4.6 — others excluded.

        Per https://platform.claude.com/docs/en/build-with-claude/fast-mode,
        sending speed=fast to Opus 4.7, Sonnet, or Haiku returns HTTP 400.
        Opus 4.8 uses a separate ``…-fast`` model id, not this parameter.
        """
        from hermes_cli.models import model_supports_fast_mode

        assert model_supports_fast_mode("claude-sonnet-4-6") is False
        assert model_supports_fast_mode("claude-sonnet-4.6") is False
        assert model_supports_fast_mode("claude-haiku-4-5") is False
        assert model_supports_fast_mode("claude-opus-4-7") is False
        assert model_supports_fast_mode("claude-opus-4-8") is False
        assert model_supports_fast_mode("anthropic/claude-sonnet-4.6") is False
        assert model_supports_fast_mode("anthropic/claude-opus-4-7") is False



    def test_resolve_overrides_returns_speed_for_anthropic(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        result = resolve_fast_mode_overrides("claude-opus-4-6")
        assert result == {"speed": "fast"}

        result = resolve_fast_mode_overrides("anthropic/claude-opus-4.6")
        assert result == {"speed": "fast"}





    def test_fast_command_hidden_for_anthropic_sonnet(self):
        """Sonnet doesn't support fast mode (Opus 4.6 only) — /fast must be hidden."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-sonnet-4-6", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is False



    def test_turn_route_injects_speed_for_anthropic(self):
        """Anthropic models should get speed:'fast' override, not service_tier."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            api_mode="anthropic_messages",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            service_tier="priority",
        )

        route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        assert route["runtime"]["provider"] == "anthropic"
        assert route["request_overrides"] == {"speed": "fast"}


class TestAnthropicFastModeAdapter(unittest.TestCase):
    """Verify build_anthropic_kwargs handles fast_mode parameter."""

    def test_fast_mode_adds_speed_and_beta(self):
        from agent.anthropic_adapter import build_anthropic_kwargs, _FAST_MODE_BETA

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
        )
        assert kwargs.get("extra_body", {}).get("speed") == "fast"
        assert "speed" not in kwargs
        assert "extra_headers" in kwargs
        assert _FAST_MODE_BETA in kwargs["extra_headers"].get("anthropic-beta", "")

    def test_fast_mode_off_no_speed(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=False,
        )
        assert kwargs.get("extra_body", {}).get("speed") is None
        assert "speed" not in kwargs
        assert "extra_headers" not in kwargs

    def test_fast_mode_skipped_for_third_party_endpoint(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
            base_url="https://api.minimax.io/anthropic/v1",
        )
        # Third-party endpoints should NOT get speed or fast-mode beta
        assert kwargs.get("extra_body", {}).get("speed") is None
        assert "speed" not in kwargs
        assert "extra_headers" not in kwargs



class TestConfigDefault(unittest.TestCase):
    def test_default_config_has_service_tier(self):
        from hermes_cli.config import DEFAULT_CONFIG

        agent = DEFAULT_CONFIG.get("agent", {})
        self.assertIn("service_tier", agent)
        self.assertEqual(agent["service_tier"], "")


class TestGeminiVersionGate(unittest.TestCase):
    """Google lists service tiers for gemini-2.5+ only.

    Gemini's native REST rejects the whole request on unexpected body fields,
    so sending ``service_tier`` to gemini-1.5/2.0 would hard-fail every call
    for a user who set ``flex`` globally.
    """

    def test_supported_versions_detected(self):
        from hermes_cli.models import _is_google_service_tier_model

        for model in [
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
            "gemini-3-flash-preview", "gemini-3.1-pro-preview",
            "gemini-3.5-flash", "gemini-3.6-flash",
            "openrouter/google/gemini-3.5-flash",
        ]:
            assert _is_google_service_tier_model(model), f"{model} should be eligible"

    def test_pre_2_5_models_excluded(self):
        from hermes_cli.models import _is_google_service_tier_model

        for model in [
            "gemini-1.5-pro", "gemini-1.5-flash",
            "gemini-2.0-flash", "google/gemini-2.0-flash-lite",
        ]:
            assert not _is_google_service_tier_model(model), f"{model} predates tiers"

    def test_non_gemini_google_models_excluded(self):
        from hermes_cli.models import _is_google_service_tier_model

        for model in ["gemma-3-27b", "google/gemma-2-9b", "lyria-2"]:
            assert not _is_google_service_tier_model(model), f"{model} is not Gemini"


class TestTierWhitelist(unittest.TestCase):
    """Only documented tier values may reach a provider.

    The parsers normalize today, but that invariant lives in four separate
    places; an unnormalized value must not become `service_tier: fast` on the
    wire (OpenAI 400s on that).
    """

    def test_unknown_tier_returns_none(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        for tier in ["fast", "on", "standard", "scale", "bogus"]:
            assert resolve_fast_mode_overrides("gpt-5.4", tier=tier) is None, tier

    def test_documented_tiers_still_pass(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gpt-5.4", tier="priority") == {"service_tier": "priority"}
        assert resolve_fast_mode_overrides("gpt-5.4", tier="flex") == {"service_tier": "flex"}
