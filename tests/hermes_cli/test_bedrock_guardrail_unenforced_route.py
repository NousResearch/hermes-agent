"""bedrock.guardrail vs. the resolved Bedrock route (issue #52179 follow-up).

Bedrock applies a guardrail two ways: ``guardrailConfig`` in the Converse body, and the
``X-Amzn-Bedrock-Guardrail*`` InvokeModel headers used by the Claude route. The Mantle
Responses endpoint that serves ``openai.gpt-5.x`` has neither, so a configured
``bedrock.guardrail`` is not applied there. Warn instead of running unguarded in silence.
"""

import logging
from unittest.mock import patch

import pytest

GUARDRAIL_CFG = {
    "bedrock": {
        "region": "us-east-2",
        "guardrail": {"guardrail_identifier": "gr-abc123", "guardrail_version": "3"},
    }
}
NO_GUARDRAIL_CFG = {"bedrock": {"region": "us-east-2"}}

MANTLE_MODEL = "openai.gpt-5.6-sol"
CLAUDE_MODEL = "us.anthropic.claude-sonnet-5"
CONVERSE_MODEL = "us.meta.llama4-maverick-17b-instruct-v1:0"

WARN_MARKER = "runs unguarded"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)


def _resolve(model, config):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with patch("hermes_cli.config.load_config", return_value=config), \
         patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
         patch("hermes_cli.runtime_provider._get_model_config",
               return_value={"provider": "bedrock", "default": model}):
        return resolve_runtime_provider(requested="bedrock")


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and WARN_MARKER in r.getMessage()]


class TestGuardrailUnenforcedRouteWarning:
    def test_mantle_route_with_guardrail_warns(self, caplog):
        """openai.gpt-5.x resolves to Mantle Responses, which carries no guardrail."""
        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider_backends"):
            runtime = _resolve(MANTLE_MODEL, GUARDRAIL_CFG)
        assert runtime["api_mode"] == "codex_responses"
        assert runtime["bedrock_openai"] is True
        warned = _warnings(caplog)
        assert len(warned) == 1, warned
        assert MANTLE_MODEL in warned[0]
        assert "bedrock.guardrail" in warned[0]

    def test_mantle_route_without_guardrail_is_silent(self, caplog):
        """No guardrail configured: nothing to warn about."""
        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider_backends"):
            runtime = _resolve(MANTLE_MODEL, NO_GUARDRAIL_CFG)
        assert runtime["api_mode"] == "codex_responses"
        assert "guardrail_config" not in runtime
        assert _warnings(caplog) == []

    def test_converse_route_with_guardrail_is_silent(self, caplog):
        """Converse attaches guardrailConfig, so the guardrail is enforced — no warning."""
        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider_backends"):
            runtime = _resolve(CONVERSE_MODEL, GUARDRAIL_CFG)
        assert runtime["api_mode"] == "bedrock_converse"
        assert runtime["guardrail_config"] == {"guardrailIdentifier": "gr-abc123", "guardrailVersion": "3"}
        assert _warnings(caplog) == []

    def test_claude_route_with_guardrail_is_silent(self, caplog):
        """The Claude route enforces the guardrail through InvokeModel headers — no warning."""
        with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_provider_backends"):
            runtime = _resolve(CLAUDE_MODEL, GUARDRAIL_CFG)
        assert runtime["api_mode"] == "anthropic_messages"
        assert runtime["bedrock_anthropic"] is True
        assert _warnings(caplog) == []
