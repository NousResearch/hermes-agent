"""#94881: determine_api_mode() must be model-aware for Copilot.

Copilot is dual-wire like Nous: GPT-5+ models (except gpt-5-mini) are only
reachable via the Responses API, everything else via /chat/completions. The
model-less transport lookup stamped chat_completions into config during the
switch flow, silently pinning every Responses-only model out of the catalog
with an error that read like an entitlement problem.
"""
from __future__ import annotations

import pytest

from hermes_cli.providers import determine_api_mode

COPILOT_HOST = "https://api.githubcopilot.com"


class TestCopilotDualWire:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5.6-sol", "codex_responses"),
            ("gpt-5.6-luna-900k", "codex_responses"),
            ("gpt-5.6-terra", "codex_responses"),
            ("gpt-5.5", "codex_responses"),
            ("gpt-5.4-mini", "codex_responses"),
            ("gpt-5.3-codex", "codex_responses"),
            ("gpt-5-mini", "chat_completions"),
            ("gpt-4o", "chat_completions"),
            ("claude-sonnet-5", "chat_completions"),
            ("claude-opus-5", "chat_completions"),
            ("gemini-3-pro", "chat_completions"),
        ],
    )
    def test_model_derives_the_wire(self, model, expected):
        assert determine_api_mode("copilot", COPILOT_HOST, model) == expected

    def test_unknown_model_stays_on_chat_completions(self):
        # Callers that don't yet know the model keep the safer
        # OpenAI-compatible path — mirrors the Nous carve-out default.
        assert determine_api_mode("copilot", COPILOT_HOST, "") == "chat_completions"
        assert determine_api_mode("copilot", COPILOT_HOST) == "chat_completions"

    @pytest.mark.parametrize("alias", ["Copilot", "github-copilot", "github_copilot"])
    def test_provider_aliases_covered(self, alias):
        assert (
            determine_api_mode(alias, COPILOT_HOST, "gpt-5.6-sol")
            == "codex_responses"
        )

    @pytest.mark.parametrize(
        "model",
        ["copilot/gpt-5.6-sol", "github-copilot/gpt-5.5"],
    )
    def test_qualified_form_normalized_before_pattern_check(self, model):
        # The switch flow can derive api_mode before its own id resolution
        # strips the provider prefix; the raw prefix must not make the
        # pattern check miss and stamp chat_completions.
        assert determine_api_mode("copilot", COPILOT_HOST, model) == "codex_responses"

    def test_nous_carve_out_unchanged(self):
        assert (
            determine_api_mode("nous", "", "anthropic/claude-sonnet-4")
            == "anthropic_messages"
        )
        assert determine_api_mode("nous", "", "gpt/anything") == "chat_completions"

    def test_non_copilot_provider_unaffected(self):
        # An unrelated provider on an unmandated host still goes through the
        # transport lookup, not the Copilot carve-out.
        assert (
            determine_api_mode("some-other-provider", "https://example.invalid", "gpt-5.6")
            == "chat_completions"
        )
