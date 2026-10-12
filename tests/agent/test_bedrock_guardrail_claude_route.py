"""``bedrock.guardrail`` must be enforced on the Claude route, not only Converse. The AnthropicBedrock
SDK speaks InvokeModel, whose body has no ``guardrailConfig``; Bedrock reads the guardrail from
``X-Amzn-Bedrock-Guardrail*`` headers. Blocks come back with ``stop_reason=end_turn`` and the
guardrail's canned text, flagged only by ``amazon-bedrock-guardrailAction`` in the body."""

from types import SimpleNamespace
from unittest.mock import patch

CFG = {"bedrock": {"guardrail": {"guardrail_identifier": "gr-1", "guardrail_version": "DRAFT", "trace": "enabled"}}}


def test_anthropic_bedrock_client_carries_configured_guardrail_headers():
    with patch("hermes_cli.config.load_config_readonly", return_value=CFG):
        from agent.anthropic_adapter import build_anthropic_bedrock_client
        client = build_anthropic_bedrock_client("us-east-2")
    headers = {k.lower(): v for k, v in client.default_headers.items()}
    assert headers["x-amzn-bedrock-guardrailidentifier"] == "gr-1"
    assert headers["x-amzn-bedrock-guardrailversion"] == "DRAFT"
    assert headers["x-amzn-bedrock-trace"] == "ENABLED"
    assert "anthropic-beta" in headers, "guardrail headers must not displace the beta header"


def test_guardrail_intervened_invokemodel_reply_is_content_filter_not_model_text():
    from agent.transports.anthropic import AnthropicTransport
    transport = AnthropicTransport()
    blocked = SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text="BLOCKED_BY_GUARDRAIL_INPUT")],
        model_extra={"amazon-bedrock-guardrailAction": "INTERVENED"},
    )
    clean = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="hi")], model_extra={})
    assert transport.response_finish_reason(blocked) == "content_filter"
    assert transport.response_finish_reason(clean) == "stop"
