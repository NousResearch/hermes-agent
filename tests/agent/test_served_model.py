"""agent.served_model — per-response served-model capture behind routing proxies (#54864)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.served_model import (
    capture_served_model,
    install_served_model_capture,
    result_model_fields,
    served_model_from_response,
)


class _Agent:
    model = "hermes-router"
    _fallback_activated = False
    _primary_runtime: dict = {}


def test_httpx_hook_captures_litellm_header_and_clears_when_absent():
    served = {"value": "gpt-4o-2024-11-20"}

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"x-litellm-model-id": served["value"]} if served["value"] else {}
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "hermes-router",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }, headers=headers)

    client = openai.OpenAI(api_key="x", base_url="http://proxy.test/v1",
                           http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    agent = _Agent()
    install_served_model_capture(agent, client)
    install_served_model_capture(agent, client)  # idempotent: one hook per client
    assert len(client._client.event_hooks["response"]) == 1

    client.chat.completions.create(model="hermes-router", messages=[{"role": "user", "content": "hi"}])
    assert result_model_fields(agent) == {"requested_model": "hermes-router", "served_model": "gpt-4o-2024-11-20"}

    served["value"] = ""  # next response has no routing header: the stale value must not survive
    client.chat.completions.create(model="hermes-router", messages=[{"role": "user", "content": "hi"}])
    assert result_model_fields(agent) == {"requested_model": "hermes-router", "served_model": None}

    # Hermes' own fallback route surfaces the same way when no proxy header is present.
    agent._fallback_activated = True
    agent._primary_runtime = {"model": "gpt-5.6-sol"}
    agent.model = "qwen/qwen3.8-max"
    assert result_model_fields(agent) == {"requested_model": "gpt-5.6-sol", "served_model": "qwen/qwen3.8-max"}


# ── Source (c): the served model named in the response BODY ────────────────────────────────


def test_served_model_from_response_reads_dict_and_object_forms():
    assert served_model_from_response({"model": "zai-org/glm-5.3-flash"}) == "zai-org/glm-5.3-flash"
    assert served_model_from_response(SimpleNamespace(model="zai-org/glm-5.3-flash")) == "zai-org/glm-5.3-flash"


@pytest.mark.parametrize("bad", [
    {},                                    # no model field at all
    {"model": None},                       # explicitly absent
    {"model": ""},                         # empty string
    {"model": "   "},                      # whitespace only
    {"model": 123},                        # wrong type
    None,                                  # no response object
    SimpleNamespace(),                     # object without the attribute
])
def test_served_model_from_response_never_fabricates_a_value(bad):
    assert served_model_from_response(bad) is None


def test_capture_from_body_covers_a_gateway_that_sends_no_header():
    """A gateway echoing the served model in the body (no x-litellm-* header) is captured,
    so the footer can show ``alias → served`` behind such a proxy instead of the alias alone."""
    agent = _Agent()
    agent.last_served_model = None

    capture_served_model(agent, SimpleNamespace(model="zai-org/glm-5.3-flash"))

    assert result_model_fields(agent) == {
        "requested_model": "hermes-router", "served_model": "zai-org/glm-5.3-flash",
    }


def test_header_value_wins_over_body_echo_for_the_same_response():
    """A LiteLLM proxy puts the alias in the body and the deployment in the header: the more
    specific header must not be overwritten by the body's echo."""
    agent = _Agent()
    agent.last_served_model = "gpt-4o-2024-11-20"  # stamped by the httpx hook

    capture_served_model(agent, SimpleNamespace(model="hermes-router"))

    assert result_model_fields(agent)["served_model"] == "gpt-4o-2024-11-20"


def test_body_echo_equal_to_requested_model_is_not_reported_as_served():
    """A provider that echoes the request back must not turn into a fake ``a → a`` footer."""
    agent = _Agent()
    agent.last_served_model = None

    capture_served_model(agent, SimpleNamespace(model="hermes-router"))

    # Captured, but result_model_fields keeps it: the footer's renderer skips an equal pair.
    assert served_model_from_response(SimpleNamespace(model="hermes-router")) == "hermes-router"


def test_body_capture_is_fail_open_on_a_hostile_response():
    """Never raise on a hot path: an object whose ``model`` accessor explodes is just ignored."""
    class _Exploding:
        @property
        def model(self):
            raise RuntimeError("boom")

    agent = _Agent()
    agent.last_served_model = None
    capture_served_model(agent, _Exploding())  # must not raise
    assert result_model_fields(agent)["served_model"] is None
