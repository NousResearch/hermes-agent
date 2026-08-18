"""Test: _moa_prepared_request does not leak to native OpenAI clients (#78382).

After a client replacement (credential rotation / fallback / dead-connection
cleanup), agent.client may become a native OpenAI client while agent.provider
stays "moa".  The dispatch must strip the MoA-internal key so the native SDK
does not reject it.
"""
import types
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request


class _FakeNativeClient:
    """Mimics a native OpenAI client whose create() rejects unknown kwargs."""

    def __init__(self):
        self.chat = types.SimpleNamespace()
        self.chat.completions = types.SimpleNamespace()
        self.chat.completions.create = MagicMock(return_value="native-response")


def _make_agent(provider="moa"):
    agent = MagicMock()
    agent.provider = provider
    agent.client = _FakeNativeClient()
    agent.api_mode = "chat_completions"
    return agent


def test_moa_key_stripped_from_native_client():
    """_moa_prepared_request must not reach a native OpenAI client."""
    agent = _make_agent(provider="moa")
    api_kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "_moa_prepared_request": {"messages": [], "model": "x"},
    }

    # Patch make_client to return a factory that returns our fake client
    def _make_client(label=None):
        return agent.client

    _dispatch_nonstreaming_api_request(agent, api_kwargs, make_client=_make_client)

    # The native client should NOT have received the MoA-internal key.
    call_kwargs = agent.client.chat.completions.create.call_args[1]
    assert "_moa_prepared_request" not in call_kwargs, (
        f"_moa_prepared_request leaked to native client: {call_kwargs.keys()}"
    )


def test_no_moa_key_when_absent():
    """Normal non-MoA call should work without the key present."""
    agent = _make_agent(provider="openai")
    api_kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
    }

    def _make_client(label=None):
        return agent.client

    _dispatch_nonstreaming_api_request(agent, api_kwargs, make_client=_make_client)
    call_kwargs = agent.client.chat.completions.create.call_args[1]
    assert "_moa_prepared_request" not in call_kwargs
    assert call_kwargs["model"] == "gpt-4"
