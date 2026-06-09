"""Tests for the post-tool placeholder guard (issue #42503).

After a tool call completes, some models return a short progress/status
string ("writing...", "working on it", "好的，我来处理") instead of a
substantive final answer.  The guard detects these and appends a
continuation nudge so the agent loop doesn't stop prematurely.

Coverage:
  - placeholder text after tool calls  → nudge and continue
  - truly empty response after tool calls  → existing empty guard, not this one
  - real concise final answers after tool calls  → finalize normally
  - placeholder without prior tool calls  → finalize normally
  - second placeholder (already nudged once)  → finalize normally (no loop)
"""

import re
import pytest


# ---------------------------------------------------------------------------
# Pull the regex out of the source so we can unit-test it independently.
# This is the same pattern used in conversation_loop.py.
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(
    r'^('
    # English: bare acknowledgements
    r'(ok(ay)?|sure|got it|understood|alright|noted)[.,!]?\s*'
    # English: forward-looking intent without content
    r'|(i\'?ll|let me|i will|i\'?m going to|i\'?m)\b.{0,60}'
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
        # Forward-looking but with a definitive decision — borderline.
        # "I'll leave it as is." matches the i'll pattern → treated as placeholder.
        # Acceptable false positive: nudge once, model confirms decision.
        assert _is_placeholder("I'll leave it as is.")

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
# Layer 2: guard behaviour in the agent loop
# ---------------------------------------------------------------------------

def _make_agent():
    """Minimal AIAgent-shaped object for testing the guard logic."""
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent._post_tool_placeholder_retried = False
    agent._post_tool_empty_retried = False
    agent._empty_content_retries = 0
    agent._thinking_prefill_retries = 0
    agent._session_messages = []
    agent.saved_session_logs = []
    agent._save_session_log = lambda msgs: agent.saved_session_logs.append(
        [m.copy() for m in msgs]
    )
    agent.emitted_statuses = []
    agent._emit_status = lambda s: agent.emitted_statuses.append(s)
    agent._strip_think_blocks = lambda t: t
    return agent


def _tool_messages():
    """A minimal messages list ending with a tool result."""
    return [
        {"role": "user", "content": "do the task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "output", "tool_call_id": "c1"},
    ]


def _run_placeholder_guard(agent, messages, response_text):
    """Execute just the placeholder guard block.

    Returns True if the guard fired (would have called continue),
    False if it let the response through.
    Mutates messages in place when it fires.
    """
    _prior_was_tool = any(
        m.get("role") == "tool" for m in messages[-6:]
    )
    if not (
        _prior_was_tool
        and not getattr(agent, "_post_tool_placeholder_retried", False)
    ):
        return False

    _clean = agent._strip_think_blocks(response_text).strip()
    if len(_clean) < 120 and _PLACEHOLDER_RE.match(_clean):
        agent._post_tool_placeholder_retried = True
        agent._emit_status(
            "⚠️ Model returned a placeholder after tool calls — nudging to continue"
        )
        messages.append({
            "role": "assistant",
            "content": response_text,
            "_empty_recovery_synthetic": True,
        })
        messages.append({
            "role": "user",
            "content": (
                "You returned a status message instead of the "
                "actual result. Please complete the task and "
                "provide your full response now."
            ),
            "_empty_recovery_synthetic": True,
        })
        agent._save_session_log(messages)
        return True   # would `continue`

    return False      # falls through to normal finalization


class TestPlaceholderGuardFires:

    def test_writing_ellipsis_triggers_nudge(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "writing...")
        assert fired
        assert agent._post_tool_placeholder_retried is True
        assert any("nudging" in s for s in agent.emitted_statuses)
        # nudge messages appended and marked synthetic
        assert messages[-1]["_empty_recovery_synthetic"]
        assert messages[-2]["_empty_recovery_synthetic"]

    def test_working_on_it_triggers_nudge(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "working on it")
        assert fired

    def test_chinese_placeholder_triggers_nudge(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "好的，我来处理")
        assert fired

    def test_let_me_process_triggers_nudge(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "Let me process that.")
        assert fired

    def test_ok_triggers_nudge(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "ok")
        assert fired


class TestPlaceholderGuardDoesNotFire:

    def test_real_answer_done(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "Done.")
        assert not fired

    def test_real_answer_file_updated(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(
            agent, messages, "The file has been updated with the new config."
        )
        assert not fired

    def test_real_answer_number(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "42")
        assert not fired

    def test_real_answer_concise_sentence(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "Yes, it works.")
        assert not fired

    def test_long_response_not_fired(self):
        agent = _make_agent()
        messages = _tool_messages()
        long = "I have analyzed the tool output. " * 6
        fired = _run_placeholder_guard(agent, messages, long)
        assert not fired

    def test_no_prior_tool_not_fired(self):
        """Guard must not fire when there are no preceding tool messages."""
        agent = _make_agent()
        messages = [{"role": "user", "content": "hello"}]
        fired = _run_placeholder_guard(agent, messages, "writing...")
        assert not fired

    def test_second_placeholder_not_fired(self):
        """After one nudge, the flag prevents a second trigger (no infinite loop)."""
        agent = _make_agent()
        agent._post_tool_placeholder_retried = True   # already nudged
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "writing...")
        assert not fired
        # No extra messages appended
        assert len(messages) == 3

    def test_chinese_done_not_fired(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(agent, messages, "处理完成。")
        assert not fired

    def test_chinese_result_sentence_not_fired(self):
        agent = _make_agent()
        messages = _tool_messages()
        fired = _run_placeholder_guard(
            agent, messages, "文件已更新，共修改了 3 处。"
        )
        assert not fired


class TestPlaceholderGuardSyntheticMessagesCleanup:

    def test_synthetic_messages_marked_for_cleanup(self):
        """Both appended messages carry _empty_recovery_synthetic so the
        pop-scaffolding block in conversation_loop removes them from the
        final transcript on a successful follow-up response."""
        agent = _make_agent()
        messages = _tool_messages()
        _run_placeholder_guard(agent, messages, "writing...")
        synthetic = [m for m in messages if m.get("_empty_recovery_synthetic")]
        assert len(synthetic) == 2
        roles = {m["role"] for m in synthetic}
        assert roles == {"assistant", "user"}
