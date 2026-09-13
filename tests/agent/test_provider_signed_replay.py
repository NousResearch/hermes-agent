"""Signed Bedrock history cannot be rewritten or exempt arbitrary new secrets."""
import copy

import pytest

from agent.provider_redaction import redact_provider_api_kwargs, redact_provider_message_values
from agent.secret_scope import reset_secret_scope, set_secret_scope


@pytest.fixture
def active_secret(monkeypatch):
    secret = "signed-replay-secret-77487"
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"REPLAY_TOKEN": secret})
    try:
        yield secret
    finally:
        reset_secret_scope(token)


def signed_request(secret, location=None):
    return {
        "modelId": "fixture",
        "system": [{"text": secret if location == "system" else "safe system"}],
        "messages": [
            {"role": "user", "content": [{"text": secret if location == "prior" else "safe prompt"}]},
            {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {
                    "text": secret if location == "reasoning" else "safe reasoning",
                    "signature": "issuer-signed-fixture",
                }}},
                {"text": "safe answer"},
            ]},
            {"role": "user", "content": [{"text": secret}]},
        ],
    }


@pytest.mark.parametrize("location", ["system", "prior", "reasoning"])
def test_signed_prefix_collision_rejects_without_mutation(active_secret, location):
    request = signed_request(active_secret, location)
    original = copy.deepcopy(request)
    with pytest.raises(ValueError, match="signed Bedrock history") as exc:
        redact_provider_api_kwargs(request)
    assert active_secret not in str(exc.value)
    assert request == original


def test_signed_prefix_without_collision_stays_exact_and_suffix_is_masked(active_secret):
    request = signed_request(active_secret)
    original = copy.deepcopy(request)
    result = redact_provider_api_kwargs(request)
    assert result["messages"][:2] == original["messages"][:2]
    assert result["system"] == original["system"]
    assert result["messages"][2]["content"] == [{"text": "***"}]
    assert request == original


def test_normalized_bedrock_sidecar_does_not_bypass_collision_check(active_secret):
    messages = signed_request(active_secret, "prior")["messages"]
    messages[1]["bedrock_content_blocks"] = messages[1].pop("content")
    with pytest.raises(ValueError, match="signed Bedrock history"):
        redact_provider_message_values(messages)


def test_native_bedrock_builder_masks_visible_values_and_preserves_arguments(active_secret):
    from agent.bedrock_adapter import build_converse_kwargs
    messages = [{"role": "user", "content": active_secret}, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "fixture", "type": "function", "function": {"name": "execute", "arguments": '{"value":"' + active_secret + '"}'}}
    ]}]
    original = copy.deepcopy(messages)
    result = build_converse_kwargs("fixture", messages)
    assert result["messages"][0]["content"] == [{"text": "***"}]
    assert result["messages"][1]["content"][0]["toolUse"]["input"] == {"value": active_secret}
    assert messages == original
