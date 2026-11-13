"""A managed local model is never sent a reasoning effort its chat template refuses.

Regression for #136324: Qwen3.8's template raises on ``minimal`` and ``max``, llama-server answers
HTTP 500, and every retry repeats the same request. The catalog lists the levels each template
accepts; the request and the picker's "sends X" hint both clamp onto them.
"""

import pytest

from hermes_agent.agent.reasoning_effort import EFFORT_LADDER, effort_display_label
from hermes_cli.local_runtime import catalog, growth

MANAGED = "http://127.0.0.1:18434/v1"


@pytest.fixture
def managed_endpoint(monkeypatch):
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda url: url == MANAGED)


def _model(entry_id):
    return catalog.catalog_by_id()[entry_id].variants[0].model_id


def _wire_effort(model_id, reasoning_config, base_url=MANAGED):
    from hermes_agent.agent.transports.chat_completions import ChatCompletionsTransport
    from providers import get_provider_profile

    wire = ChatCompletionsTransport().build_kwargs(
        model=model_id, messages=[{"role": "user", "content": "Hello"}],
        provider_profile=get_provider_profile("llamacpp"), base_url=base_url,
        reasoning_config=reasoning_config)
    return wire.get("reasoning_effort")


@pytest.mark.parametrize("asked,sent", [
    ("max", "xhigh"), ("ultra", "xhigh"), ("minimal", "low"),
    ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"),
])
def test_qwen38_gets_the_nearest_level_its_template_accepts(managed_endpoint, asked, sent):
    assert _wire_effort(_model("qwen3.8-27b"), {"enabled": True, "effort": asked}) == sent


@pytest.mark.parametrize("off", [{"enabled": False}, {"enabled": True, "effort": "none"}])
def test_turning_thinking_off_still_sends_none(managed_endpoint, off):
    assert _wire_effort(_model("qwen3.8-27b"), off) == "none"


def test_a_background_call_to_the_same_model_is_clamped_too(managed_endpoint):
    from hermes_agent.agent.auxiliary_client import _build_call_kwargs

    kwargs = _build_call_kwargs(
        "custom", _model("qwen3.8-27b"), [{"role": "user", "content": "Hello"}],
        reasoning_config={"enabled": True, "effort": "max"}, base_url=MANAGED)
    assert kwargs["reasoning_effort"] == "xhigh"


def test_a_model_whose_template_takes_any_level_keeps_the_request(managed_endpoint):
    for model_id in (_model("nemotron-3.5-bolt-74b-a9b"), "a-sideloaded-model"):
        assert _wire_effort(model_id, {"enabled": True, "effort": "max"}) == "max"


def test_another_server_serving_the_same_model_name_keeps_the_request(managed_endpoint):
    sent = _wire_effort(_model("qwen3.8-27b"), {"enabled": True, "effort": "max"},
                        base_url="http://192.168.1.20:8080/v1")
    assert sent == "max"


def test_the_picker_hint_names_the_level_actually_sent(managed_endpoint):
    from hermes_agent.gateway.session_local import effort_fields

    model_id = _model("qwen3.8-27b")
    info = effort_fields({"enabled": True, "effort": "max"}, "", model_id, "custom", MANAGED)
    assert (info["reasoning_effort"], info["reasoning_effort_wire"]) == ("max", "xhigh")
    assert effort_display_label("max", "llamacpp", model_id) == "max (sends xhigh on this route)"
    assert effort_display_label("high", "llamacpp", model_id) == "high"
    assert effort_display_label("none", "llamacpp", model_id) == "none"


def test_catalog_effort_levels_are_levels_hermes_can_send():
    sendable = set(EFFORT_LADDER) - {"none", "ultra"}
    for entry in catalog.CATALOG:
        assert set(entry.reasoning_efforts) <= sendable, entry.id
