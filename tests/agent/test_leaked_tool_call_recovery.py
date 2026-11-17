"""Regression tests for tool calls leaked as plain text (``chat/completions``).

Some OpenAI-compatible servers — every local one we tried (LM Studio, Ollama,
vLLM windows, and local inference proxies; observed ~2026-11 on a Qwen3 build)
— occasionally write the tool call into the assistant *content* instead of
shipping a structured ``tool_calls`` entry:

    Bien sûr, let me check the files first.
    {"cmd": "ls -la ~"}

The wire response is then a perfectly ordinary ``finish_reason="stop"`` turn
with no tool calls, so the loop took it as the final answer, persisted the
turn as complete, and **executed nothing** — with no warning anywhere. On a
hosted model this is rare; on small local models it is the dominant failure
mode, and it is silent: the session ends cleanly and the user believes the
task ran.

``codex_responses`` already recovers this shape through
``_leaked_tool_call_text`` (Harmony ``to=functions.…`` calls and shell JSON
with an action lead-in). This suite covers the same recovery on the
``chat/completions`` path, plus the two properties that matter most: a
legitimate code block must NOT be re-prompted, and the retry budget must be
bounded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Two forms the detector recognizes. Both are unambiguous leaks: a model that
# wanted to *say* this would have phrased it as prose.
# NOTE: the lead-in must be English — `_SHELL_JSON_LEAK_LEADIN_PATTERN` is anchored
# on English action verbs. A French lead-in ("Bien sûr, let me check…") is NOT
# detected; that limit is pinned by `test_non_english_leadin_is_not_detected`.
LEAKED_SHELL = 'Sure, let me check the files first.\n\n{"cmd": "ls -la ~"}'
LEAKED_SHELL_FR = 'Bien sûr, let me check the files first.\n\n{"cmd": "ls -la ~"}'
LEAKED_HARMONY = 'Let me look.\n<|channel|>commentary to=functions.terminal {"command": "ls ~"}'

# A code block is NOT a leak: users ask for code all the time, and
# re-prompting on it would break ordinary use. Deliberately out of scope.
LEGITIMATE_CODE = (
    "Voici le script :\n\n```python\n"
    "from pathlib import Path\n"
    "Path('rapport.html').write_text('<html></html>')\n```"
)


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client, mirroring
    ``test_dropped_tool_call_recovery.py``."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


class TestLeakedToolCallRecovery:
    def test_leaked_shell_call_reprompts_instead_of_exiting(self, loop_agent):
        """finish_reason=stop with a shell call in the text must re-prompt
        rather than exit the loop with the text as the final answer."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=LEAKED_SHELL, finish_reason="stop"),
            _mock_response(content="Il y a 42 fichiers.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("combien de fichiers ?")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "A leaked tool call must trigger a re-prompt (second API call), "
            "not exit the loop after one call with nothing executed."
        )
        second_call = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call.kwargs.get("messages") or second_call.args[0].get("messages")
        last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
        assert last_user is not None
        assert "tool call" in (last_user.get("content") or "").lower(), (
            "The nudge must explicitly ask the model to issue the tool call."
        )
        assert "42 fichiers" in result["final_response"]

    def test_leaked_harmony_call_reprompts(self, loop_agent):
        """The Harmony ``to=functions.…`` form leaks the same way and gets the
        same recovery."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=LEAKED_HARMONY, finish_reason="stop"),
            _mock_response(content="done", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("liste les fichiers")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "The Harmony leak form must be recovered too."
        )

    def test_non_english_leadin_is_not_detected(self, loop_agent):
        """KNOWN LIMIT, pinned deliberately rather than left implicit.

        ``_SHELL_JSON_LEAK_LEADIN_PATTERN`` is anchored on English action verbs
        ("let me check", "I'll create", …), so a lead-in in another language is
        not recognized and the turn ends as a final answer. Widening the pattern
        is a separate change with its own false-positive risk; this test exists
        so the gap is a documented decision, not a surprise.
        """
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=LEAKED_SHELL_FR, finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("combien de fichiers ?")

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "Documented limit: a non-English lead-in is not detected by the "
            "current shell-JSON detector."
        )

    def test_legitimate_code_block_is_not_reprompted(self, loop_agent):
        """A code block in the answer is frequently what the user asked for.
        Re-prompting on it would break ordinary use, so it must pass through."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=LEGITIMATE_CODE, finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("montre-moi un exemple de script")

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A legitimate code block must not be treated as a leaked tool call."
        )
        assert "Path" in result["final_response"]

    def test_clean_stop_text_turn_is_unaffected(self, loop_agent):
        """Ordinary text answers keep working, obviously."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Here is your answer.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert "Here is your answer." in result["final_response"]

    def test_persistent_leaks_are_bounded(self, loop_agent):
        """A model that never stops leaking must not loop forever: give up
        after a bounded number of consecutive re-prompts."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=LEAKED_SHELL, finish_reason="stop") for _ in range(9)
        ] + [_mock_response(content="done", finish_reason="stop")]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("fais le travail")

        # 1 initial call + at most 3 bounded re-prompts = 4 total. It must NOT
        # consume all nine staged leaks.
        assert loop_agent.client.chat.completions.create.call_count <= 4, (
            "Consecutive leaked tool calls must be bounded (no infinite loop)."
        )
        assert result is not None

    def test_nudge_pair_is_ephemeral_scaffolding(self, loop_agent):
        """The re-prompt pair must never reach the durable transcript, so a
        resumed session cannot replay the internal nudge as user context."""
        from agent.session_persistence import _is_ephemeral_scaffolding
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=LEAKED_SHELL, finish_reason="stop"),
            _mock_response(content="fini", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("fais le travail")

        leftover = [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("_dropped_toolcall_nudge")
        ]
        assert not leftover, (
            "The re-prompt pair must be stripped at finalization, not kept in "
            "the returned transcript."
        )
        assert _is_ephemeral_scaffolding(
            {"role": "user", "content": "nudge", "_dropped_toolcall_nudge": True}
        )
