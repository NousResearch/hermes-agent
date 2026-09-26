"""Providers that return thinking as a structured ``reasoning`` JSON field.

The OpenAI SDK parses unknown response fields into ``model_extra``. Nous'
inference API returns some models' chain-of-thought this way (verified live:
z-ai/glm-5.3-flash on inference-api.nousresearch.com returns
``message.reasoning`` populated while ``content`` carries only the answer).
``extract_reasoning`` read attributes and ``model_extra["reasoning_content"]``
but never ``model_extra["reasoning"]``, so the chain-of-thought was silently
dropped: the stored assistant message carried ``reasoning: ""`` and
batch_runner's no-reasoning discard deleted whole trajectories that had
perfectly good answers in them.
"""
import types

from agent.agent_runtime_helpers import extract_reasoning


def _assistant_message(**extra):
    return types.SimpleNamespace(
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        content="The answer is 391.",
        model_extra=dict(extra),
    )


def test_reasoning_field_surviving_in_model_extra_is_extracted():
    msg = _assistant_message(reasoning="17*23 = 391, via 17*20 + 17*3.")
    agent = types.SimpleNamespace()
    assert extract_reasoning(agent, msg) == "17*23 = 391, via 17*20 + 17*3."


def test_reasoning_content_in_model_extra_still_extracted():
    msg = _assistant_message(reasoning_content="checking arithmetic")
    agent = types.SimpleNamespace()
    assert extract_reasoning(agent, msg) == "checking arithmetic"


def test_no_reasoning_anywhere_returns_none():
    msg = _assistant_message()
    agent = types.SimpleNamespace()
    assert extract_reasoning(agent, msg) is None
