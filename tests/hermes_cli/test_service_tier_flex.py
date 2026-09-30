"""Flex service-tier coverage for the classic CLI surface (PR #37059).

Kept in its own file (not appended to test_fast_command.py) so routine upstream
growth of that file cannot conflict with these tests.
"""

import unittest


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


class TestParseServiceTierFlex(unittest.TestCase):
    def _parse(self, raw):
        cli_mod = _import_cli()
        return cli_mod._parse_service_tier_config(raw)

    def test_flex_is_accepted(self):
        """agent.service_tier: flex was rejected as unknown and silently dropped."""
        self.assertEqual(self._parse("flex"), "flex")
        self.assertEqual(self._parse("  FLEX  "), "flex")

    def test_normal_and_unknown_still_disable(self):
        for raw in ["", "normal", "off", "standard", "bogus"]:
            self.assertIsNone(self._parse(raw), f"{raw!r} should not enable a tier")


class TestTwoLevelModelPaths(unittest.TestCase):
    """Aggregators route models as ``aggregator/vendor/model`` — deliberately NOT eligible.

    The route gate keeps tier params off aggregator routes regardless, so matching
    two-level ids would only expose a ``/fast`` toggle on the CLI/gateway (which check
    the model alone) that confirms FAST and then sends nothing. A single vendor prefix
    (``openai/gpt-4.1``) still matches — that is the first-party form.
    """

    def test_aggregator_two_level_ids_are_not_eligible(self):
        from hermes_cli.models import model_supports_fast_mode

        assert not model_supports_fast_mode("openrouter/openai/gpt-4.1")

    def test_single_vendor_prefix_still_matches(self):
        from hermes_cli.models import model_supports_fast_mode

        assert model_supports_fast_mode("openai/gpt-4.1")


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
        ]:
            assert _is_google_service_tier_model(model), f"{model} should be tier-eligible"

    def test_non_gemini_models_not_detected(self):
        from hermes_cli.models import _is_google_service_tier_model

        # Two-level aggregator ids are deliberately ineligible (route-gated anyway;
        # matching them would only expose a lying /fast toggle).
        for model in [
            "gpt-5.4", "claude-opus-4-6",
            "openrouter/openai/gpt-4.1", "openrouter/google/gemini-3.5-flash",
        ]:
            assert not _is_google_service_tier_model(model), f"{model} must not be tier-eligible"


class TestFlexTier(unittest.TestCase):
    """``agent.service_tier: flex`` must survive all the way to the override dict."""

    def test_openai_flex_override(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gpt-5.4", tier="flex") == {"service_tier": "flex"}

    def test_gemini_flex_override(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides(
            "google/gemini-3.5-flash", tier="flex"
        ) == {"service_tier": "flex"}

    def test_gemini_priority_override(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gemini-3.6-flash", tier="priority") == {
            "service_tier": "priority"
        }

    def test_anthropic_flex_returns_none(self):
        """Anthropic has no flex equivalent — speed=fast is a priority-only knob."""
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("claude-opus-4-8", tier="flex") is None

    def test_anthropic_priority_still_returns_speed(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("claude-opus-4-8", tier="priority") == {
            "speed": "fast"
        }

    def test_default_tier_is_priority(self):
        """Existing callers pass no tier and must keep priority semantics."""
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gpt-5.4") == {"service_tier": "priority"}
        assert resolve_fast_mode_overrides("claude-opus-4-8") == {"speed": "fast"}



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
            "google/gemini-3.5-flash",
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

    Since #128307 the invariant lives at the parse layer: every surface goes
    through ``agent.fast_mode.parse_service_tier`` (one word table), so an
    unnormalized word like ``fast`` never reaches the resolver as a tier.
    """

    def test_parser_normalizes_every_word(self):
        from agent.fast_mode import SERVICE_TIER_WORDS, parse_service_tier

        assert parse_service_tier("fast") == "priority"
        assert parse_service_tier("flex") == "flex"
        assert parse_service_tier("bogus") is None
        # flex is a static per-request tier alongside priority/ultrafast.
        from agent.fast_mode import STATIC_TIERS

        assert "flex" in STATIC_TIERS and "flex" in SERVICE_TIER_WORDS.values()

    def test_documented_tiers_still_pass(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gpt-5.4", tier="priority") == {"service_tier": "priority"}
        assert resolve_fast_mode_overrides("gpt-5.4", tier="flex") == {"service_tier": "flex"}
