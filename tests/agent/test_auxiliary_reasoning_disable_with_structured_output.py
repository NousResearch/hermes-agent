"""Thinking-off + structured-output combination rejection in ``agent.auxiliary_client``.

The title lane disables reasoning (``reasoning_config={"enabled": False}``) and concurrently
asks for a strict ``json_schema`` (``agent/title_generator.py``). A thinking-trained model
behind an OpenAI-compatible gateway (Shanghai AI Lab's discovery gateway, Atria-Dawn /
Intern-S2) answers ``reasoning_effort: "none"`` happily and ``response_format: json_schema``
happily, but not both at once: the grammar-constrained decode cannot be built with reasoning
disabled, and the rejection is a bare, field-less
``422 upstream_request_rejected / Unprocessable Entity`` — no ``param``, no field name, no
wording the field-keyed classifiers can match. Before the fix, no parameter rung fired and
the title died on the first exchange of every session on that route.

The recovery lifts the thinking-off encoding to the floor effort (``low``) — the same
reaction as the reasoning-required class — but only while the request is
grammar-constrained, so the thinking-off encoding is untouched for every other route
(Ollama / vLLM / GLM / DeepSeek take ``none`` as intended, #112781's guarantee). The route
is remembered so the next thinking-off *and* grammar-constrained aux call starts at the
floor without the guaranteed 422.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_reasoning_floor
from agent.auxiliary_client import call_llm

# Shanghai AI Lab discovery gateway, reproduced live: each accepted alone, combination 422.
_GATEWAY_422 = (
    "Error code: 422 - {'error': {'code': 'upstream_request_rejected', "
    "'message': 'Unprocessable Entity', 'trace_id': '9c6ab000efe3cb2e7621333bac3da473', "
    "'type': 'invalid_request_error'}, "
    "'request_id': 'req_62e377ea-cc8e-45c3-937c-dffc67534f4b'}"
)


@pytest.fixture(autouse=True)
def _fresh_memo():
    auxiliary_reasoning_floor._FLOORED_ROUTES.clear()
    yield
    auxiliary_reasoning_floor._FLOORED_ROUTES.clear()


def _call(client, **kw):
    call_kwargs = dict(
        task="title_generation", messages=[{"role": "user", "content": "hi"}],
        extra_body={"response_format": {"type": "json_schema"}}, reasoning_config={"enabled": False},
    )
    call_kwargs.update(kw)
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "Atria-Dawn", "https://discovery-api.intern-ai.org.cn/v1",
                            "sk-x", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "Atria-Dawn")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_wk: resp),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=None),
    ):
        return call_llm(**call_kwargs)


def test_silent_422_lifts_thinking_off_to_the_floor_and_remembers_the_route():
    """First call: ``none`` + json_schema → 422 → retry at the floor with everything else intact.
    Second call on the same route+model: the floor goes out up front, no 422 round-trip."""
    client = MagicMock()
    client.base_url = "https://discovery-api.intern-ai.org.cn/v1"
    client.chat.completions.create.side_effect = [RuntimeError(_GATEWAY_422), {"ok": True}, {"ok": True}]

    assert _call(client) == {"ok": True}
    first, retry = (c.kwargs for c in client.chat.completions.create.call_args_list[:2])
    assert first["reasoning_effort"] == "none"  # the deliberate thinking-off encoding still goes out first
    assert retry["reasoning_effort"] == auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT
    assert retry["extra_body"]["response_format"] == {"type": "json_schema"}  # the constraint survives
    assert retry["model"] == first["model"]

    assert _call(client) == {"ok": True}  # second call starts at the floor
    assert client.chat.completions.create.call_count == 3
    assert ("discovery-api.intern-ai.org.cn", "Atria-Dawn") in auxiliary_reasoning_floor._FLOORED_ROUTES


def test_nameless_422_without_a_response_format_does_not_lift_the_disable():
    """The conjunction is the whole fix: the same bare 422 on a request with NO structured-output
    constraint must not touch the thinking-off encoding — that is the un-fixable-class guard for
    every route that takes ``none`` as intended (#112781)."""
    client = MagicMock()
    client.base_url = "https://discovery-api.intern-ai.org.cn/v1"
    client.chat.completions.create.side_effect = [RuntimeError(_GATEWAY_422)]

    with pytest.raises(RuntimeError, match="upstream_request_rejected"):
        _call(client, extra_body={})
    assert client.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "none"


def test_422_that_names_a_field_takes_its_precise_rung_not_the_silent_one():
    """An error naming ``reasoning_effort`` is the strip class (#112781); the silent rung must not
    shadow it. Regression guard on rung ordering: field-keyed rungs run first."""
    client = MagicMock()
    client.base_url = "https://relay.example/v1"
    named = ("Error code: 400 - {'error': {'message': 'Unrecognized request argument supplied: "
             "reasoning_effort', 'type': 'invalid_request_error', 'param': '', 'code': None}}")
    client.chat.completions.create.side_effect = [RuntimeError(named), {"ok": True}]

    assert _call(client) == {"ok": True}
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 2
    assert "reasoning_effort" not in calls[1].kwargs  # stripped, not floored
    assert calls[1].kwargs["extra_body"]["response_format"] == {"type": "json_schema"}


def test_non_422_error_does_not_fire_the_silent_rung():
    """The rung's status guard rejects a 500 carrying the same wording: server error, not a
    parameter rejection (mirroring ``_is_reasoning_field_rejection``'s own guard). A status-less
    error with the same text is the silent class, so the guard is the only thing separating them."""
    from agent.auxiliary_client import _is_reasoning_disable_with_structured_output_rejection

    server_error = RuntimeError("Error code: 500 - {'error': {'code': 'upstream_request_rejected', "
                                "'message': 'Unprocessable Entity'}}")
    server_error.status_code = 500
    assert _is_reasoning_disable_with_structured_output_rejection(server_error) is False

    silent_422 = RuntimeError(_GATEWAY_422)
    silent_422.status_code = 422
    assert _is_reasoning_disable_with_structured_output_rejection(silent_422) is True

    statusless = RuntimeError(_GATEWAY_422)
    assert _is_reasoning_disable_with_structured_output_rejection(statusless) is True
