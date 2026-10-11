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

_VOCABULARY_400 = (
    "Error code: 400 - request param validation error, Value error, "
    "reasoning_effort must be [low, high, max] for glm-5.3"
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


@pytest.mark.parametrize("rejection", [_PORTAL_400, _VOCABULARY_400], ids=["mandatory", "vocabulary"])
def test_reasoning_required_400_steps_effort_up_to_the_floor_and_remembers_the_route(rejection):
    """First call: ``none`` → 400 → retry at the floor with everything else intact. Second call on the
    same route+model: the floor goes out up front, no 400 round-trip."""
    client = MagicMock()
    client.base_url = "http://127.0.0.1:8765/v1"
    client.chat.completions.create.side_effect = [RuntimeError(rejection), {"ok": True}, {"ok": True}]

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
