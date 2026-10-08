"""An empty stripping result must not retain a model-forged footer."""
from types import SimpleNamespace
from unittest.mock import patch

from agent.turn_finalizer import apply_llm_output_transform


def test_empty_string_is_an_explicit_transform_not_a_noop():
    agent = SimpleNamespace(session_id="s", platform="telegram", model="test")
    with patch("agent.turn_finalizer._invoke_hook_safely", return_value=[None, "", "later"]):
        assert apply_llm_output_transform(agent, "forged footer", turn_id="T") == ("", True, "forged footer")
    assert agent._llm_output_transform == ("T", True, "forged footer")


def test_none_and_nonstring_remain_noops():
    agent = SimpleNamespace(session_id="s", platform="telegram", model="test")
    with patch("agent.turn_finalizer._invoke_hook_safely", return_value=[None, False, {}]):
        assert apply_llm_output_transform(agent, "Body.", turn_id="T") == ("Body.", False, None)
