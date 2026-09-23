"""Managed routes cannot escape through transports without final-wire validation."""
from types import SimpleNamespace

import pytest

from agent.model_selection_types import RoutingBlocked


@pytest.mark.parametrize("provider,api_mode", [("bedrock", "bedrock_converse"), ("moa", "chat_completions")])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("managed", [False, True])
def test_unsupported_agent_transport_never_sends_managed(monkeypatch, provider, api_mode, streaming, managed):
    import agent.chat_completion_helpers as helpers

    sends = []

    def send(*args, **kwargs):
        sends.append(kwargs)
        return "legacy-result"

    monkeypatch.setattr(helpers, "_bedrock_converse_call", send)
    monkeypatch.setattr(helpers, "_BedrockStream", lambda *args: SimpleNamespace(run=send))
    monkeypatch.setattr(helpers, "_StreamingCall", lambda *args: SimpleNamespace(run=send))
    monkeypatch.setattr(helpers, "_check_stale_giveup", lambda agent: None)
    agent = SimpleNamespace(
        provider=provider, api_mode=api_mode, _interrupt_requested=False,
        _managed_routing_receipt_id="receipt" if managed else None,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send))),
    )

    def execute():
        if streaming:
            return helpers.interruptible_streaming_api_call(agent, {})
        return helpers._dispatch_nonstreaming_api_request(agent, {}, make_client=lambda *a, **k: agent.client)

    if managed:
        with pytest.raises(RoutingBlocked) as blocked:
            execute()
        assert blocked.value.reason == "unsupported_executor"
        assert sends == []
    else:
        assert execute() == "legacy-result"
        assert len(sends) == 1


@pytest.mark.parametrize("managed", [False, True])
def test_auxiliary_bedrock_never_sends_managed(monkeypatch, managed):
    from agent.auxiliary_client import BedrockAuxiliaryClient
    from agent import managed_route_aux_wire as wire
    import agent.bedrock_adapter as bedrock

    sends = []

    def send(**kwargs):
        sends.append(kwargs)
        return "legacy-result"

    monkeypatch.setattr(bedrock, "call_converse", send)
    client = BedrockAuxiliaryClient("us-east-1", "fixture-model")
    token = wire._current.set(wire._WireContext(None, "receipt", "bedrock", client.base_url) if managed else None)
    try:
        if managed:
            with pytest.raises(RoutingBlocked) as blocked:
                client.chat.completions.create(messages=[{"role": "user", "content": "fixture"}])
            assert blocked.value.reason == "unsupported_executor"
            assert sends == []
        else:
            assert client.chat.completions.create(messages=[]) == "legacy-result"
            assert len(sends) == 1
    finally:
        wire._current.reset(token)
