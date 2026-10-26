"""Generic endpoint seams (``hostnames`` via ``providers.profile_for_host`` + the L3 fields/hooks) through a fake relay profile.

The relay is not any shipped provider: the contract is that a profile claiming an endpoint controls its
Responses item-id replay, cache-key routing, issuer stamp, reasoning shape and message-level reasoning
carriers, and that nothing leaks onto endpoints it does not claim.
"""

from types import SimpleNamespace

import pytest

from providers.base import ProviderProfile

RELAY = "https://relay.example/v1"
OTHER = "https://other.example/v1"


class _RelayProfile(ProviderProfile):
    def reasoning_payload(self, model, reasoning_config, *, base_url):
        return {"effort": "low"}

    def replay_message_carriers(self, values, *, model, base_url):
        return {"blob": values["blob"]} if values.get("blob") else {}


@pytest.fixture
def relay(request):
    import providers

    profile = _RelayProfile(
        name="relay-fake", hostnames=("relay.example",), responses_replay_item_ids=False, responses_prompt_cache_key=False,
        responses_issuer_kind="relay_kind", responses_trusts_completed_status=False,
        responses_profile_reasoning=True, message_carrier_opaque="blob", message_carrier_text="blob_text",
    )
    providers.list_providers()  # discovery first, so the fixture registration is not shadowed
    providers.register_provider(profile)

    def _cleanup():
        providers._REGISTRY.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    return profile


_HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "pong", "codex_message_items": [{
        "type": "message", "role": "assistant", "status": "completed", "id": "msg_short",
        "content": [{"type": "output_text", "text": "pong"}],
    }]},
    {"role": "user", "content": "again"},
]


def _build(base_url):
    from agent.transports.codex import ResponsesApiTransport

    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        "gpt-5.4", _HISTORY, None, base_url=base_url, session_id="s1",
        reasoning_config={"enabled": True, "effort": "high"}, profile_reasoning={"effort": "low"},
    )
    return transport, kwargs


def test_claimed_endpoint_controls_responses_wire(relay):
    transport, kw = _build(RELAY)
    message = next(i for i in kw["input"] if i.get("role") == "assistant")
    assert "id" not in message
    assert "prompt_cache_key" not in kw
    assert kw["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "include" not in kw
    assert transport._last_issuer_kind == "relay_kind"


def test_unclaimed_endpoint_keeps_generic_wire(relay):
    transport, kw = _build(OTHER)
    message = next(i for i in kw["input"] if i.get("role") == "assistant")
    assert message["id"] == "msg_short"
    assert kw["prompt_cache_key"]
    assert kw["include"] == ["reasoning.encrypted_content"]
    assert str(transport._last_issuer_kind).startswith("other:")


def test_declared_issuer_untrusted_completed_status_continues(relay):
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(status="completed", incomplete_details=None, output=[SimpleNamespace(
        type="reasoning", id="rs_1", encrypted_content="enc", summary=[SimpleNamespace(type="summary_text", text="t")],
    )])
    assert _normalize_codex_response(response, issuer_kind="relay_kind")[1] == "incomplete"
    assert _normalize_codex_response(response, issuer_kind="other:https://x")[1] == "stop"


def test_declared_carriers_are_captured_and_replayed_only_on_the_claimed_endpoint(relay):
    from agent.reasoning_carriers import CARRIER_TYPE, carrier_record, shape_wire_carriers

    record = carrier_record(SimpleNamespace(blob="B", blob_text="T"))
    assert record == {"type": CARRIER_TYPE, "blob": "B", "blob_text": "T"}
    assistant = {"role": "assistant", "content": "x", "blob": "B", "reasoning_details": [record]}
    on_relay = shape_wire_carriers([assistant], model="m", base_url=RELAY)[0]
    elsewhere = shape_wire_carriers([assistant], model="m", base_url=OTHER)[0]
    assert on_relay["blob"] == "B" and "blob_text" not in on_relay and "reasoning_details" not in on_relay
    assert "blob" not in elsewhere and "reasoning_details" not in elsewhere


def test_persisted_carrier_keys_of_an_uninstalled_plugin_never_reach_the_wire():
    """No registered profile declares ``blob`` (its plugin is gone): the record's own keys are still stripped."""
    from agent.reasoning_carriers import CARRIER_TYPE, shape_wire_carriers

    record = {"type": CARRIER_TYPE, "blob": "B"}
    assistant = {"role": "assistant", "content": "x", "blob": "B", "reasoning_details": [record]}
    wire = shape_wire_carriers([assistant], model="m", base_url=RELAY)[0]
    assert wire == {"role": "assistant", "content": "x"}
    assert assistant["reasoning_details"] == [record]  # history is never mutated
