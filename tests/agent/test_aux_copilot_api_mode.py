"""Auxiliary transport selection must follow the MODEL, not a stale api_mode.

Regression contract for the goal-judge outage (2026-08-10):

``model.api_mode: codex_responses`` in a profile's config.yaml was written for
a GPT-5 main model.  When that profile later switched its model to
``claude-opus-5``, main chat kept working (it resolves the wire per model) but
every auxiliary call inherited the stale literal, got wrapped in
``CodexAuxiliaryClient``, and died with::

    HTTP 400 unsupported_api_for_model
    model claude-opus-5 does not support Responses API

Because the goal judge runs on the auxiliary lane, the user-visible symptom was
``kanban: goal completion rejected by judge`` — a transport bug wearing the
costume of a review verdict.  These tests assert the invariant that prevents
that whole failure class: for Copilot, the model decides the wire.
"""

import pytest

from hermes_cli.models import _should_use_copilot_responses_api


class TestCopilotResponsesCapability:
    """The capability predicate the auxiliary router must defer to."""

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.4", "gpt-5.3-codex"])
    def test_gpt5_family_uses_responses_api(self, model):
        assert _should_use_copilot_responses_api(model) is True

    @pytest.mark.parametrize(
        "model", ["claude-opus-5", "claude-sonnet-5", "gemini-3.6-flash", "gpt-5-mini"]
    )
    def test_non_responses_models_use_chat_completions(self, model):
        assert _should_use_copilot_responses_api(model) is False


class TestStaleApiModeIsNotHonoured:
    """A stale ``codex_responses`` must not wrap a Chat-Completions model.

    These call the real ``resolve_provider_client`` — the exact path the goal
    judge takes — rather than asserting on a reimplementation of the rule.
    """

    def _resolve(self, monkeypatch, model):
        from agent import auxiliary_client as aux

        monkeypatch.setattr(
            aux, "_resolve_provider_api_key", lambda *a, **k: "test-token", raising=False
        )
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "test-token")
        return aux.resolve_provider_client(
            "copilot",
            model,
            api_mode="codex_responses",  # the stale literal
        )

    def test_claude_model_is_not_codex_wrapped(self, monkeypatch):
        from agent.auxiliary_client import CodexAuxiliaryClient

        client, resolved = self._resolve(monkeypatch, "claude-opus-5")
        if client is None:
            pytest.skip("no copilot credentials in this environment")
        assert not isinstance(client, CodexAuxiliaryClient), (
            "claude-opus-5 was wrapped for the Responses API despite Copilot "
            "serving it over chat.completions — this is the 400 "
            "unsupported_api_for_model regression"
        )
        assert resolved == "claude-opus-5"

    def test_gpt5_model_still_codex_wrapped(self, monkeypatch):
        """The fix must not over-correct: GPT-5 genuinely needs Responses."""
        from agent.auxiliary_client import CodexAuxiliaryClient

        client, _ = self._resolve(monkeypatch, "gpt-5.6-sol")
        if client is None:
            pytest.skip("no copilot credentials in this environment")
        assert isinstance(client, CodexAuxiliaryClient), (
            "gpt-5.6-sol must keep the Responses-API wrapper"
        )
