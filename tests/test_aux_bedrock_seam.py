"""Regression tests for the R1-C11 Bedrock adapter-family extraction.

The Bedrock auxiliary-client family (BedrockAuxiliaryClient and friends)
moved from agent/auxiliary_client.py to agent/bedrock_completions.py.
agent/auxiliary_client.py re-exports the six names lazily via module
``__getattr__``; these tests pin that identity seam and the constructor
wiring so the extraction stays byte-faithful on future slices.
"""

import pytest

from agent import auxiliary_client as aux
from agent import bedrock_completions

REEXPORTED_NAMES = (
    "AsyncBedrockAuxiliaryClient",
    "BedrockAuxiliaryClient",
    "_AsyncBedrockChatShim",
    "_AsyncBedrockCompletionsAdapter",
    "_BedrockChatShim",
    "_BedrockCompletionsAdapter",
)


@pytest.mark.parametrize("name", REEXPORTED_NAMES)
def test_seam_identity(name):
    """Every re-exported name resolves to the same object as the module's."""
    assert getattr(aux, name) is getattr(bedrock_completions, name)


def test_seam_unknown_name_raises_attribute_error():
    """Module __getattr__ must not mask real AttributeErrors."""
    with pytest.raises(AttributeError):
        getattr(aux, "BedrockAuxiliaryClientDoesNotExist")


def test_bedrock_client_constructor_wiring():
    client = aux.BedrockAuxiliaryClient("us-east-1", "openai.gpt-oss-20b-1:0")
    assert client._region == "us-east-1"
    assert client._model == "openai.gpt-oss-20b-1:0"
    assert client.api_key == "aws-sdk"
    assert client.base_url == "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert isinstance(client.chat, aux._BedrockChatShim)
    assert isinstance(client.chat.completions, aux._BedrockCompletionsAdapter)
    assert client.chat.completions._region == "us-east-1"
    assert client.chat.completions._model == "openai.gpt-oss-20b-1:0"


def test_async_bedrock_client_constructor_wiring():
    sync = aux.BedrockAuxiliaryClient("eu-west-1", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    async_client = aux.AsyncBedrockAuxiliaryClient(sync)
    assert async_client.api_key == sync.api_key
    assert async_client.base_url == sync.base_url
    assert isinstance(async_client.chat, aux._AsyncBedrockChatShim)
    assert isinstance(
        async_client.chat.completions, aux._AsyncBedrockCompletionsAdapter
    )
    assert async_client.chat.completions._sync is sync.chat.completions


def test_bedrock_create_preserves_lazy_call_converse_seam(monkeypatch):
    """create() still reaches call_converse via its function-local lazy import."""
    captured = {}

    def fake_call_converse(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("agent.bedrock_adapter.call_converse", fake_call_converse)
    client = aux.BedrockAuxiliaryClient("us-east-1", "openai.gpt-oss-20b-1:0")
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        stop="END",
    )
    assert resp is not None
    assert captured["region"] == "us-east-1"
    assert captured["model"] == "openai.gpt-oss-20b-1:0"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    # Default max_tokens when none passed; str stop normalized to a list.
    assert captured["max_tokens"] == 4096
    assert captured["stop_sequences"] == ["END"]
