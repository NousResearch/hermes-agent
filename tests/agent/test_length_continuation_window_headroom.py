"""Length continuation stops when the PROMPT filled the context window (#106120).

``finish_reason='length'`` with ``usage.prompt_tokens`` ≈ context length means there was
no room to generate, not that the answer was long. Continuing appends a fragment + nudge
— strictly more prompt — so every retry is worse. The turn must end on the first
truncation and name the context window as the cause; a real output-cap truncation (plenty
of headroom) keeps continuing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        a.context_compressor.context_length = 32768
        return a


def _length_response(content: str, prompt_tokens: int):
    from tests.run_agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id="resp",
        model="test/model",
        choices=[SimpleNamespace(
            index=0, message=_mock_assistant_msg(content=content), finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=40,
                              total_tokens=prompt_tokens + 40),
    )


def _run(agent, message):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def test_prompt_filling_the_window_ends_the_turn_on_first_truncation(loop_agent):
    # Reporter's live numbers: 32,638-token prompt in a 32,768 window (~130 tokens of room).
    loop_agent.client.chat.completions.create.side_effect = [
        _length_response(f"part {i} ", prompt_tokens=32638 + 47 * i) for i in range(4)
    ]
    result = _run(loop_agent, "summarize everything so far")

    assert loop_agent.client.chat.completions.create.call_count == 1
    assert result["partial"] is True
    assert "part 0" in result["final_response"]
    assert "context window" in result["final_response"].lower()
    assert "32,638" in result["final_response"] and "32,768" in result["final_response"]
    assert "continuation attempts" not in (result.get("error") or "")
    # No continuation trail is left behind for the next turn.
    assert not any(
        m.get("_length_continuation_fragment") or m.get("_length_continuation_nudge")
        for m in result["messages"] if isinstance(m, dict)
    )


def test_output_cap_truncation_with_headroom_still_continues(loop_agent):
    loop_agent.client.chat.completions.create.side_effect = [
        _length_response(f"part {i} ", prompt_tokens=4_000 + 500 * i) for i in range(4)
    ]
    result = _run(loop_agent, "write me a long report")

    assert loop_agent.client.chat.completions.create.call_count == 4
    assert "truncated after 4 continuation attempts" in (result.get("error") or "")
    assert "part 3" in result["final_response"]
