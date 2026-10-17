"""Summarizer language rule (#130117): key on the user's messages, never on assistant drift."""
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _prompt(has_user_turn: bool) -> str:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True)
    return compressor._build_summary_prompt("x", 500, None, "", has_user_turn)  # the rule is static template text


def test_user_turn_prompt_keys_language_on_user_messages_and_forbids_translation():
    prompt = _prompt(True)
    assert "language of the user's own messages (not assistant replies" in prompt
    assert "do not translate or switch to English" in prompt


def test_no_user_turn_prompt_grounds_language_in_source_turns_only():
    prompt = _prompt(False)
    assert "dominant natural language of the source turns being summarized" in prompt
    assert "language of the user's own messages" not in prompt
