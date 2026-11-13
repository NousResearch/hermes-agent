"""``_AnthropicCompletionsAdapter`` forwards the caller's request timeout to the Messages call.

Without it every auxiliary request on an anthropic_messages route ran on the client's 900 s read
default instead of its task budget (compression, titles, ``hermes fallback check --live``).
"""
import types

import pytest

from hermes_agent.agent import auxiliary_client as aux


def _send(monkeypatch, **kwargs):
    sent: dict = {}
    monkeypatch.setattr("hermes_agent.agent.anthropic_adapter.create_anthropic_message",
                        lambda client, api_kwargs, **_: sent.update(api_kwargs) or types.SimpleNamespace(content=[]))
    monkeypatch.setattr("hermes_agent.agent.transports.get_transport", lambda name: types.SimpleNamespace(
        normalize_response=lambda r, **_: types.SimpleNamespace(content="OK", tool_calls=None, reasoning=None,
                                                               finish_reason="stop")))
    adapter = aux._AnthropicCompletionsAdapter(object(), "claude-x", base_url="http://h")
    adapter.create(model="claude-x", messages=[{"role": "user", "content": "hi"}], max_tokens=8, **kwargs)
    return sent


def test_positive_timeout_rides_on_the_messages_call(monkeypatch):
    assert _send(monkeypatch, timeout=12.5)["timeout"] == 12.5


@pytest.mark.parametrize("timeout", [None, 0, -1, "30"])
def test_unusable_timeout_keeps_the_client_default(monkeypatch, timeout):
    assert "timeout" not in _send(monkeypatch, timeout=timeout)
