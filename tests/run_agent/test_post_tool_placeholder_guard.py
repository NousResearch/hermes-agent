"""Tests for the post-tool placeholder guard (issue #42503).

After a tool call completes, some models return a short progress/status
string ("writing...", "working on it", "好的，我来处理") instead of a
substantive final answer.  The guard detects these and appends a
continuation nudge so the agent loop doesn't stop prematurely.

Coverage:
  Layer 1 — regex unit tests (pattern matching only)
  Layer 2 — run_conversation integration tests covering:
    - tool → placeholder → nudge → final answer
    - tool → real concise answer → finalize normally
    - tool → definitive decision ("I'll leave it as is.") → finalize normally
    - placeholder after nudge (no infinite loop)
    - synthetic message cleanup on successful follow-up
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Pull the regex out of the source so we can unit-test it independently.
# This is the same pattern used in conversation_loop.py.
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r'^('
    # English: bare acknowledgements
    r'(ok(ay)?|sure|got it|understood|alright|noted)[.,!]?\s*'
    # English: forward-looking intent with progress verbs only
    r'|(i\'?ll|let me|i will|i\'?m going to|i\'?m)\s+'
    r'(do|handle|process|check|look|work|fix|update|write|analyze'
    r'|review|take care|get|start|begin|go ahead|proceed'
    r'|working on|going to|look into|looking)\b.{0,60}'
    r'|working on (it|this|that)\.{0,3}'
    r'|writing\.{0,3}'
    r'|processing\.{0,3}'
    r'|on it\.{0,3}'
    r'|just a (moment|sec|second)\.{0,3}'
    r'|one (moment|sec|second)\.{0,3}'
    # Chinese: acknowledgements and progress markers
    r'|好的[，,。！]?\s*(我|让我|来|下面|接下来|继续)?.{0,40}'
    r'|收到[。！]?\s*'
    r'|正在(处理|写|分析|查看|执行).{0,30}'
    r'|继续(处理|写|分析|执行)?.{0,30}'
    r')$',
    re.IGNORECASE | re.DOTALL,
)


def _is_placeholder(text: str) -> bool:
    return len(text) < 120 and bool(_PLACEHOLDER_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# Layer 1: regex unit tests
# ---------------------------------------------------------------------------

class TestPlaceholderRegex:

    # --- should match (placeholders) ---

    def test_writing_ellipsis(self):
        assert _is_placeholder("writing...")

    def test_writing_no_ellipsis(self):
        assert _is_placeholder("writing")

    def test_working_on_it(self):
        assert _is_placeholder("working on it")

    def test_working_on_it_ellipsis(self):
        assert _is_placeholder("working on it...")

    def test_working_on_this(self):
        assert _is_placeholder("working on this")

    def test_processing_ellipsis(self):
        assert _is_placeholder("processing...")

    def test_ok(self):
        assert _is_placeholder("ok")

    def test_okay(self):
        assert _is_placeholder("okay.")

    def test_sure(self):
        assert _is_placeholder("sure!")

    def test_got_it(self):
        assert _is_placeholder("got it")

    def test_understood(self):
        assert _is_placeholder("understood.")

    def test_noted(self):
        assert _is_placeholder("noted")

    def test_let_me_process(self):
        assert _is_placeholder("let me process that")

    def test_let_me_check(self):
        assert _is_placeholder("Let me check the results.")

    def test_ill_handle(self):
        assert _is_placeholder("I'll handle that now.")

    def test_im_working(self):
        assert _is_placeholder("I'm working on it.")

    def test_just_a_moment(self):
        assert _is_placeholder("just a moment...")

    def test_one_second(self):
        assert _is_placeholder("one second...")

    def test_on_it(self):
        assert _is_placeholder("on it.")

    def test_chinese_hao_de(self):
        assert _is_placeholder("好的，我来处理")

    def test_chinese_hao_de_simple(self):
        assert _is_placeholder("好的")

    def test_chinese_shou_dao(self):
        assert _is_placeholder("收到")

    def test_chinese_zheng_zai(self):
        assert _is_placeholder("正在处理")

    def test_chinese_ji_xu(self):
        assert _is_placeholder("继续处理")

    def test_chinese_hao_de_with_continuation(self):
        assert _is_placeholder("好的，让我来分析一下")

    # --- should NOT match (real answers) ---

    def test_done(self):
        assert not _is_placeholder("Done.")

    def test_fixed(self):
        assert not _is_placeholder("Fixed.")

    def test_file_updated(self):
        assert not _is_placeholder("The file has been updated.")

    def test_number_answer(self):
        assert not _is_placeholder("42")

    def test_yes_it_works(self):
        assert not _is_placeholder("Yes, it works.")

    def test_result_sentence(self):
        assert not _is_placeholder("The function returned 3 items.")

    def test_ill_leave_as_is(self):
        # "I'll leave it as is." is a definitive final decision, not progress-only.
        assert not _is_placeholder("I'll leave it as is.")

    def test_ill_keep_current(self):
        assert not _is_placeholder("I'll keep the current implementation.")

    def test_ill_skip_that(self):
        assert not _is_placeholder("I'll skip that one.")

    def test_let_me_know(self):
        assert not _is_placeholder("Let me know if you need anything else.")

    def test_long_response_not_placeholder(self):
        long = "I have processed the tool results. " * 10
        assert not _is_placeholder(long)

    def test_code_block_not_placeholder(self):
        assert not _is_placeholder("```python\nprint('hello')\n```")

    def test_chinese_done(self):
        assert not _is_placeholder("处理完成。")

    def test_chinese_result(self):
        assert not _is_placeholder("文件已更新，共修改了 3 处。")


# ---------------------------------------------------------------------------
# Layer 2: run_conversation integration tests
# ---------------------------------------------------------------------------

def _tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(*, content, finish_reason, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent_for_integration():
    """Create a fully wired AIAgent for run_conversation tests."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("bash")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"bash"}
    agent.client = MagicMock()
    return agent


class TestPlaceholderGuardIntegration:

    def test_placeholder_triggers_nudge_and_returns_final_answer(self):
        """tool_call → placeholder → nudge → final answer.

        The guard should detect the placeholder, nudge the model, and
        return the real answer from the third API call.
        """
        agent = _make_agent_for_integration()
        agent.client.chat.completions.create.side_effect = [
            # 1: tool call
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("bash", "call_1")],
            ),
            # 2: placeholder after tool (triggers guard)
            _response(content="writing...", finish_reason="stop"),
            # 3: real answer after nudge
            _response(
                content="The command output was 42.",
                finish_reason="stop",
            ),
        ]

        with (
            patch("run_agent.handle_function_call", return_value="42"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("run echo 42")

        assert result["final_response"] == "The command output was 42."
        assert result["api_calls"] == 3
        assert result["turn_exit_reason"].startswith("text_response")

    def test_chinese_placeholder_triggers_nudge(self):
        """Chinese placeholder '好的，我来处理' should also trigger the guard."""
        agent = _make_agent_for_integration()
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("bash", "call_1")],
            ),
            _response(content="好的，我来处理", finish_reason="stop"),
            _response(content="文件已更新完成。", finish_reason="stop"),
        ]

        with (
            patch("run_agent.handle_function_call", return_value="ok"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("更新文件")

        assert result["final_response"] == "文件已更新完成。"
        assert result["api_calls"] == 3

    def test_real_concise_answer_finalizes_normally(self):
        """A real concise answer after a tool call must NOT trigger the guard."""
        agent = _make_agent_for_integration()
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("bash", "call_1")],
            ),
            _response(content="Done.", finish_reason="stop"),
        ]

        with (
            patch("run_agent.handle_function_call", return_value="ok"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("do the task")

        assert result["final_response"] == "Done."
        assert result["api_calls"] == 2

    def test_definitive_decision_does_not_trigger_guard(self):
        """'I'll leave it as is.' is a valid final decision, not a placeholder."""
        agent = _make_agent_for_integration()
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("bash", "call_1")],
            ),
            _response(
                content="I'll leave it as is.",
                finish_reason="stop",
            ),
        ]

        with (
            patch("run_agent.handle_function_call", return_value="ok"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("check the config")

        assert result["final_response"] == "I'll leave it as is."
        assert result["api_calls"] == 2

    def test_synthetic_messages_cleaned_up_after_recovery(self):
        """After a successful nudge response, synthetic scaffold messages
        must be removed from the final transcript."""
        agent = _make_agent_for_integration()
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("bash", "call_1")],
            ),
            _response(content="working on it", finish_reason="stop"),
            _response(content="All done, result is 7.", finish_reason="stop"),
        ]

        with (
            patch("run_agent.handle_function_call", return_value="7"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("compute 3+4")

        assert result["final_response"] == "All done, result is 7."
        # No synthetic messages should remain in the final transcript
        for msg in result["messages"]:
            assert not msg.get("_empty_recovery_synthetic"), (
                f"Synthetic message not cleaned up: {msg}"
            )

    def test_second_placeholder_does_not_loop(self):
        """If the model returns a placeholder after the nudge too, the guard
        must NOT fire again (no infinite loop). The second placeholder
        becomes the final response."""
        agent = _make_agent_for_integration()
        agent.client.chat.completions.create.side_effect = [
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("bash", "call_1")],
            ),
            # First placeholder → triggers guard
            _response(content="ok", finish_reason="stop"),
            # Second placeholder after nudge → should NOT re-trigger
            _response(content="sure", finish_reason="stop"),
        ]

        with (
            patch("run_agent.handle_function_call", return_value="ok"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("do task")

        # The second placeholder becomes the final response (guard only fires once)
        assert result["final_response"] == "sure"
        assert result["api_calls"] == 3
