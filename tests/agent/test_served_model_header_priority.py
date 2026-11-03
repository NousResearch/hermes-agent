"""The turn-seam served-model stamp must not overwrite a proxy deployment id.

A LiteLLM-style proxy reports the deployment that actually ran the call in a
response header. The body's model echo is coarser: it names the logged model,
not the deployment. When both describe the same response the header wins.
"""

from types import SimpleNamespace

import agent.turn_usage as tu

MODEL = "openai/gpt-6-astra"
WIRE = "gpt-6-astra"


def _fields(stamp, own=None, served_model=MODEL):
    agent = SimpleNamespace(last_served_model=stamp, model=MODEL)
    if own is not None:
        agent._turn_served_model = own
    return tu._served_model_fields(agent, SimpleNamespace(model=served_model), WIRE), agent


def test_header_deployment_id_survives_the_turn_seam():
    """The hook stamped the deployment; the body echo must not clobber it."""
    _, agent = _fields("gpt-4o-2024-11-20")
    assert agent.last_served_model == "gpt-4o-2024-11-20"


def test_body_model_is_captured_when_no_header_stamped_one():
    value, agent = _fields(None)
    assert value == MODEL
    assert agent.last_served_model == MODEL


def test_own_previous_stamp_is_replaced_by_the_new_one():
    """A stamp this function made is not a header stamp, so it may be updated."""
    value, agent = _fields("openai/older-echo", own="openai/older-echo")
    assert value == MODEL
    assert agent.last_served_model == MODEL


def test_own_stamp_is_cleared_when_the_response_has_no_served_model():
    _, agent = _fields(MODEL, own=MODEL, served_model=WIRE)
    assert agent.last_served_model is None, "a stale stamp must not persist"


def test_a_response_that_matches_the_wire_model_stamps_nothing():
    value, agent = _fields(None, served_model=WIRE)
    assert value is None
    assert agent.last_served_model is None


def test_an_absent_model_field_stamps_nothing():
    value, agent = _fields(None, served_model=None)
    assert value is None
    assert agent.last_served_model is None
