"""Provider replay carriers and raw reasoning guards survive display dedup."""
import json

import httpx
import pytest
from openai import OpenAI

from providers.base import ProviderProfile

pytest_plugins = ("tests.agent.test_reasoning_delivery_dedup",)

CARRIER_URL = "https://carrier.example/v1"


class _CarrierProfile(ProviderProfile):
    def replay_message_carriers(self, values, *, model, base_url):
        return dict(values)


@pytest.fixture
def declared_carriers(request):
    """A profile that declares the opaque/text carrier pair (GitHub Copilot's shape lives in a plugin now)."""
    import providers

    profile = _CarrierProfile(name="carrier-fake", hostnames=("carrier.example",),
                              message_carrier_opaque="reasoning_opaque", message_carrier_text="reasoning_text")
    providers.list_providers()  # discovery first, so the fixture registration is not shadowed
    providers.register_provider(profile)

    def _cleanup():
        providers._REGISTRY.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)


def stream_response(agent, monkeypatch, deltas):
    """Real SDK and main collector with a bounded, entirely synthetic HTTP stream."""
    chunks = [{"id": "fixture", "created": 1, "object": "chat.completion.chunk",
               "model": "fixture", "choices": [{"index": 0, "delta": delta,
                                                  "finish_reason": None}]} for delta in deltas]
    chunks.append({"id": "fixture", "created": 1, "object": "chat.completion.chunk",
                   "model": "fixture", "choices": [{"index": 0, "delta": {},
                                                      "finish_reason": "stop"}]})
    body = ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=body, request=req))
    with OpenAI(api_key="synthetic", base_url="https://fixture.invalid/v1", max_retries=0,
                http_client=httpx.Client(transport=transport)) as client:
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
        return agent._interruptible_streaming_api_call({"model": "fixture",
            "messages": [{"role": "user", "content": "fixture"}]})


def test_stream_declared_readable_and_opaque_carriers_survive_replay(agent, monkeypatch, declared_carriers):
    from agent.chat_completion_helpers import build_assistant_message
    from agent.reasoning_carriers import shape_wire_carriers

    callbacks = []
    agent.reasoning_callback = callbacks.append
    response = stream_response(agent, monkeypatch, [
        {"reasoning_text": "Plan ", "reasoning_opaque": "earlier-opaque"},
        {"reasoning_text": "complete", "reasoning_opaque": "latest-opaque"},
        {"content": "Answer"},
    ])
    message = response.choices[0].message
    assert message.reasoning_text == "Plan complete"
    assert message.reasoning_opaque == "latest-opaque"
    assert "".join(callbacks) == "Plan complete"
    stored = build_assistant_message(agent, message, "stop")
    wire = shape_wire_carriers([stored], model="fixture", base_url=CARRIER_URL)[0]
    assert (wire["reasoning_text"], wire["reasoning_opaque"]) == ("Plan complete", "latest-opaque")


def test_stream_gemini_signature_only_delta_survives_replay(agent, monkeypatch):
    from agent.chat_completion_helpers import build_assistant_message
    from agent.reasoning_carriers import shape_wire_carriers

    response = stream_response(agent, monkeypatch, [
        {"content": "Answer"}, {"extra_content": {"google": {"thought_signature": "synthetic-signature"}}},
    ])
    message = response.choices[0].message
    assert message.extra_content == {"google": {"thought_signature": "synthetic-signature"}}
    stored = build_assistant_message(agent, message, "stop")
    wire = shape_wire_carriers([stored], model="gemini-3.6-flash",
                              base_url="https://generativelanguage.googleapis.com/v1beta")[0]
    assert wire["extra_content"] == message.extra_content


@pytest.mark.parametrize("channel", ["ordinary", "mirrored", "ordinary-behind-details", "details-behind-ordinary"])
def test_raw_repetition_guard_is_independent_of_display_dedup(agent, monkeypatch, channel):
    loop = "A repetitive synthetic reasoning sentence which must eventually be stopped. "
    deltas = []
    for index in range(500):
        ordinary = f"Different ordinary step {index}. " if channel == "details-behind-ordinary" else loop
        delta = {"reasoning_content": ordinary}
        if channel != "ordinary":
            detail = f"Different detailed step {index}. " if channel == "ordinary-behind-details" else loop
            delta["reasoning_details"] = [{"type": "reasoning.text", "text": detail, "index": 0}]
        deltas.append(delta)
    deltas.append({"content": "Must not reach this terminal answer"})
    response = stream_response(agent, monkeypatch, deltas)
    assert getattr(response, "_runaway_repetition", False)
    assert response.choices[0].finish_reason == "length"
    assert not response.choices[0].message.content
