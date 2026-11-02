"""Reasoning-required rejection → step the aux effort up to the floor and remember the route.

The title lane disables reasoning (``reasoning_config={"enabled": False}``), which the custom profile
encodes as top-level ``reasoning_effort: "none"``. Endpoints that understand the field but refuse the
disable (Nous Portal on gpt-6-astra: 400 "Reasoning is mandatory for this endpoint and cannot be
disabled") used to fail the title outright — the strip rung (#112781) never matched this wording.
The recovery is a step UP (``low``), memoised per (route, model) so the next thinking-off aux call
on that route starts at the floor without the guaranteed 400.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_reasoning_floor
from agent.auxiliary_client import call_llm

_PORTAL_400 = (
    "Error code: 400 - {'status': 400, 'message': 'This request is not valid. Check the model name and "
    "other parameters. Additional info: Reasoning is mandatory for this endpoint and cannot be disabled.'}"
)


@pytest.fixture(autouse=True)
def _fresh_memo():
    auxiliary_reasoning_floor._FLOORED_ROUTES.clear()
    yield
    auxiliary_reasoning_floor._FLOORED_ROUTES.clear()


def _call(client):
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "openai/gpt-6-astra", "http://127.0.0.1:8765/v1", "sk-x", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "openai/gpt-6-astra")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=None),
    ):
        return call_llm(
            task="title_generation", messages=[{"role": "user", "content": "hi"}],
            extra_body={"response_format": {"type": "json_object"}}, reasoning_config={"enabled": False},
        )


def test_reasoning_required_400_steps_effort_up_to_the_floor_and_remembers_the_route():
    """First call: ``none`` → 400 → retry at the floor with everything else intact. Second call on the
    same route+model: the floor goes out up front, no 400 round-trip."""
    client = MagicMock()
    client.base_url = "http://127.0.0.1:8765/v1"
    client.chat.completions.create.side_effect = [RuntimeError(_PORTAL_400), {"ok": True}, {"ok": True}]

    assert _call(client) == {"ok": True}
    first, retry = (c.kwargs for c in client.chat.completions.create.call_args_list[:2])
    assert first["reasoning_effort"] == "none"
    assert retry["reasoning_effort"] == auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT
    assert retry["extra_body"]["response_format"] == {"type": "json_object"}
    assert retry["model"] == first["model"]

    assert _call(client) == {"ok": True}
    assert client.chat.completions.create.call_count == 3
    upfront = client.chat.completions.create.call_args_list[2].kwargs
    assert upfront["reasoning_effort"] == auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT


def test_field_rejection_still_strips_instead_of_stepping_up():
    """A relay that does not know the field at all keeps the #112781 behaviour: the field is dropped,
    nothing is remembered as a floor."""
    client = MagicMock()
    client.base_url = "https://relay.example/v1"
    client.chat.completions.create.side_effect = [
        RuntimeError("Error code: 400 - Unrecognized request argument supplied: reasoning_effort"), {"ok": True},
    ]
    assert _call(client) == {"ok": True}
    retry = client.chat.completions.create.call_args_list[1].kwargs
    assert "reasoning_effort" not in retry
    assert not auxiliary_reasoning_floor._FLOORED_ROUTES


# Live body from the firstVDS qwencloud route (Alibaba token-plan, deepseek-v4.1-flash, 2026-09-21):
# a closed level set that starts at ``minimal``, so the thinking-off encoding ``none`` is refused by
# value rather than by field name.
_TOKENPLAN_400 = (
    'Error code: 400 - {\'error\':{\'code\':\'invalid_parameter_error\',\'param\':None,'
    '\'message\':"\'reasoning_effort\' must be one of: \'minimal\', \'low\', \'medium\', \'high\', '
    '\'xhigh\', \'max\', \'ultra\'",\'type\':\'invalid_request_error\'}}'
)


def test_tokenplan_level_rejection_is_a_reasoning_rejection_not_a_field_rejection():
    """The closed-set wording names the field but blames its value: it must classify as a *required*
    rejection (floor step-up), not as a field rejection (strip) — the endpoint needs the field on the
    wire with an accepted level."""
    from agent.error_classifier import is_reasoning_field_rejection, is_reasoning_required_rejection

    assert is_reasoning_required_rejection(_TOKENPLAN_400)
    assert not is_reasoning_field_rejection(_TOKENPLAN_400)


def test_tokenplan_level_rejection_steps_up_to_the_floor_and_remembers_the_route():
    """Alibaba token-plan (qwencloud) on deepseek-v4.1-flash: ``none`` → 400 → retry at ``low`` → ok,
    and the next thinking-off aux call on that route starts at the floor."""
    client = MagicMock()
    client.base_url = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    client.chat.completions.create.side_effect = [
        RuntimeError(_TOKENPLAN_400), {"ok": True}, {"ok": True},
    ]

    assert _call(client) == {"ok": True}
    first, retry = (c.kwargs for c in client.chat.completions.create.call_args_list[:2])
    assert first["reasoning_effort"] == "none"
    assert retry["reasoning_effort"] == auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT
    assert retry["extra_body"]["response_format"] == {"type": "json_object"}

    assert _call(client) == {"ok": True}
    upfront = client.chat.completions.create.call_args_list[2].kwargs
    assert upfront["reasoning_effort"] == auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT
