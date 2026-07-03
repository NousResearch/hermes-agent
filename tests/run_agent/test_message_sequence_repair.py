"""Tests for pre-API-call message-sequence repair.

Covers ``_repair_message_sequence`` and the extended
``_drop_trailing_empty_response_scaffolding`` behavior that rewinds past
orphan tool-result tails. Together these prevent the self-reinforcing empty-
response loop observed in session 20260507_044111_fa7e65, where a tool-result
followed directly by a user message produced silent empty responses from
providers (violating role alternation), which retriggered the empty-retry
recovery every turn.
"""

from run_agent import AIAgent


def _bare_agent():
    return AIAgent.__new__(AIAgent)


# ── _drop_trailing_empty_response_scaffolding ──────────────────────────────

def test_drop_scaffolding_rewinds_orphan_tool_tail():
    """When scaffolding is stripped, also rewind the orphan assistant+tool pair."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
        {"role": "assistant", "content": "(empty)",
         "_empty_terminal_sentinel": True},
    ]

    AIAgent._drop_trailing_empty_response_scaffolding(agent, messages)

    assert messages == [{"role": "user", "content": "task"}]


def test_drop_scaffolding_keeps_tail_when_no_scaffolding():
    """Mid-iteration tool results must NOT be rewound — only if scaffolding fires."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
    ]
    original = [dict(m) for m in messages]

    AIAgent._drop_trailing_empty_response_scaffolding(agent, messages)

    assert messages == original


def test_drop_scaffolding_handles_multiple_parallel_tool_results():
    """Parallel tool calls (one assistant → many tool results) all rewound together."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "t1", "type": "function",
              "function": {"name": "f", "arguments": "{}"}},
             {"id": "t2", "type": "function",
              "function": {"name": "g", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "t1", "content": "out1"},
        {"role": "tool", "tool_call_id": "t2", "content": "out2"},
        {"role": "assistant", "content": "(empty)",
         "_empty_terminal_sentinel": True},
    ]

    AIAgent._drop_trailing_empty_response_scaffolding(agent, messages)

    assert messages == [{"role": "user", "content": "task"}]


# ── _repair_message_sequence ───────────────────────────────────────────────

def test_repair_merges_consecutive_user_messages():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 1
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "first\n\nsecond"


def test_repair_preserves_user_content_when_one_side_empty():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": ""},
        {"role": "user", "content": "real message"},
    ]

    AIAgent._repair_message_sequence(agent, messages)

    assert messages == [{"role": "user", "content": "real message"}]


def test_repair_does_not_rewind_ongoing_dialog_tool_pair():
    """assistant(tool_calls) + tool + user is a VALID pattern (user redirect
    before the model gets its continuation turn). Repair must not touch it —
    only the flag-gated scaffolding strip rewinds, and only when the
    empty-recovery scaffolding was actually present.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "out"},
        {"role": "user", "content": "Q2"},
    ]
    original = [dict(m) for m in messages]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert messages == original


def test_repair_drops_stray_tool_with_unknown_tool_call_id():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "tool", "tool_call_id": "orphan", "content": "stray"},
        {"role": "user", "content": "real"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs >= 1
    assert all(m.get("role") != "tool" for m in messages)


def test_repair_leaves_valid_conversation_unchanged():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "a.txt b.txt"},
        {"role": "assistant", "content": "Found 2 files"},
        {"role": "user", "content": "more"},
    ]
    original = [dict(m) for m in messages]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert messages == original


def test_repair_preserves_multimodal_user_content():
    """Multimodal (list) content must NOT be merged — risks mangling attachments."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"},
                                     {"type": "image_url", "image_url": {"url": "..."}}]},
        {"role": "user", "content": "follow-up"},
    ]

    AIAgent._repair_message_sequence(agent, messages)

    # The multimodal user message stays as a distinct message — no merge
    assert len(messages) == 2
    assert isinstance(messages[0]["content"], list)


def test_repair_empty_messages_returns_zero():
    agent = _bare_agent()
    messages = []

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert messages == []


def test_repair_preserves_system_messages():
    agent = _bare_agent()
    messages = [
        {"role": "system", "content": "You are..."},
        {"role": "user", "content": "hi"},
    ]
    original = [dict(m) for m in messages]

    AIAgent._repair_message_sequence(agent, messages)

    assert messages == original


# ── repair_message_sequence_with_cursor (#44837) ───────────────────────────

from agent.agent_runtime_helpers import repair_message_sequence_with_cursor


def test_cursor_clamped_when_compaction_shrinks_below_cursor():
    """Cursor past the new end of the list must come back in range so the
    turn-end flush doesn't skip the assistant/tool chain (#44837)."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    agent._last_flushed_db_idx = 2  # both rows already flushed

    repairs = repair_message_sequence_with_cursor(agent, messages)

    assert repairs == 1
    assert len(messages) == 1
    assert agent._last_flushed_db_idx == 1


def test_cursor_rewinds_when_compaction_happens_before_cursor():
    """Repair that drops/merges messages at indexes BELOW the cursor must
    rewind it by the number removed, or unflushed rows get skipped.
    A plain min() clamp does NOT catch this case."""
    agent = _bare_agent()
    flushed_a = {"role": "user", "content": "first"}
    flushed_b = {"role": "user", "content": "second"}  # merged into flushed_a
    unflushed_assistant = {"role": "assistant", "content": "answer"}
    messages = [flushed_a, flushed_b, unflushed_assistant]
    agent._last_flushed_db_idx = 2  # the two user rows are flushed

    repairs = repair_message_sequence_with_cursor(agent, messages)

    assert repairs == 1
    assert len(messages) == 2
    # Cursor must now point at the assistant (index 1), not stay at 2 —
    # min(2, len=2) would leave it at 2 and the flush would skip it.
    assert agent._last_flushed_db_idx == 1
    assert messages[agent._last_flushed_db_idx] is unflushed_assistant


def test_cursor_untouched_when_no_repairs():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    agent._last_flushed_db_idx = 1

    repairs = repair_message_sequence_with_cursor(agent, messages)

    assert repairs == 0
    assert agent._last_flushed_db_idx == 1


def test_cursor_helper_safe_without_cursor_attribute():
    """Bare agents (no _last_flushed_db_idx) must not crash."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]

    repairs = repair_message_sequence_with_cursor(agent, messages)

    assert repairs == 1
    assert not hasattr(agent, "_last_flushed_db_idx")


def test_flush_guard_clamps_overshooting_cursor():
    """_flush_messages_to_session_db safety net: an overshooting cursor must
    not produce a negative-start slice that skips everything (#44837)."""

    class _DB:
        def __init__(self):
            self.rows = []

        def append_message(self, **kw):
            self.rows.append(kw)

    agent = _bare_agent()
    agent._session_db = _DB()
    agent._session_db_created = True
    agent.session_id = "s1"
    agent._persist_user_message_override = None
    agent._last_flushed_db_idx = 5  # stale — past end of compacted list
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]

    AIAgent._flush_messages_to_session_db(agent, messages, conversation_history=[])

    # min(5, 2) = 2 → nothing skipped below start_idx, cursor settles at 2
    assert agent._last_flushed_db_idx == 2


# ── Pass 0: merge consecutive assistant messages (issue #29148, #49147) ─────

def test_repair_merges_parallel_tool_calls_split_across_assistants():
    """Two adjacent assistant(tool_calls) collapse into one turn (#29148).

    DeepSeek v4 rejects a replayed history where parallel calls appear as
    separate assistant turns:
        assistant(tc=[A]) → assistant(tc=[B]) → tool(A) → tool(B)
    The repair must produce:
        assistant(tc=[A, B]) → tool(A) → tool(B)
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "run both"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "session_search", "arguments": "{}"}}]},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_B", "type": "function",
                         "function": {"name": "search_files", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": "A"},
        {"role": "tool", "tool_call_id": "call_B", "content": "B"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs >= 1
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert {tc["id"] for tc in assistant_msgs[0]["tool_calls"]} == {"call_A", "call_B"}
    # Both tool results survive Pass 1 (their ids are in the merged union).
    assert sum(1 for m in messages if m.get("role") == "tool") == 2


def test_repair_merges_content_then_toolcalls_split():
    """content-only assistant followed by tool_calls-only assistant merge (#49147).

    The recovery/continuation paths can leave:
        assistant(content="Let me search") → assistant(tool_calls=[A]) → tool(A)
    which must become:
        assistant(content="Let me search", tool_calls=[A]) → tool(A)
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "search"},
        {"role": "assistant", "content": "Let me search for that."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "session_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "found"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs >= 1
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    merged = assistant_msgs[0]
    assert merged["content"] == "Let me search for that."
    assert len(merged["tool_calls"]) == 1
    assert merged["tool_calls"][0]["id"] == "call_1"
    # Tool result still follows immediately.
    assert messages[-1]["role"] == "tool"


def test_repair_merges_three_consecutive_assistant_tool_calls():
    """Three adjacent assistant(tool_calls) turns all collapse into one."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "run three"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "y", "arguments": "{}"}}]},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c3", "type": "function",
                         "function": {"name": "z", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "tool", "tool_call_id": "c3", "content": "r3"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs >= 2
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert len(assistant_msgs[0]["tool_calls"]) == 3
    assert sum(1 for m in messages if m.get("role") == "tool") == 3


def test_repair_does_NOT_merge_tool_calls_separated_by_tool_result():
    """A tool result between two assistant(tool_calls) marks distinct rounds.

    This is the critical guard: two sequential tool-call rounds must NOT be
    collapsed, or the second round's tool result would orphan.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "done"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "t2", "type": "function",
                         "function": {"name": "g", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t2", "content": "done2"},
    ]
    before = sum(1 for m in messages if m.get("role") == "assistant")

    AIAgent._repair_message_sequence(agent, messages)

    assert sum(1 for m in messages if m.get("role") == "assistant") == before
    # Both tool results survive (neither orphaned).
    assert sum(1 for m in messages if m.get("role") == "tool") == 2


def test_repair_does_NOT_merge_assistant_separated_by_user():
    """A user turn between two assistants blocks the merge (normal dialog)."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    AIAgent._repair_message_sequence(agent, messages)

    assert sum(1 for m in messages if m.get("role") == "assistant") == 2


def test_repair_merges_two_text_only_assistants():
    """Two consecutive text-only assistants (no tool_calls) still merge.

    The empty-response / thinking-prefill paths can leave two adjacent
    text assistants; strict providers reject consecutive same-role turns.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "First part."},
        {"role": "assistant", "content": "Second part."},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs >= 1
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "First part.\nSecond part."


def test_repair_preserves_reasoning_content_on_merge():
    """Merged tool-call turn keeps a reasoning_content (DeepSeek/Kimi replay)."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "thinking A",
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "b", "type": "function",
                         "function": {"name": "g", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "ra"},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
    ]

    AIAgent._repair_message_sequence(agent, messages)

    merged = [m for m in messages if m.get("role") == "assistant"][0]
    assert merged.get("reasoning_content") == "thinking A"


def test_repair_noop_on_valid_parallel_format():
    """A correctly-formatted single assistant with multiple tool_calls is unchanged."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "run both"},
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "call_A", "type": "function",
              "function": {"name": "session_search", "arguments": "{}"}},
             {"id": "call_B", "type": "function",
              "function": {"name": "search_files", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "call_A", "content": "A"},
        {"role": "tool", "tool_call_id": "call_B", "content": "B"},
    ]
    original_len = len(messages)

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert len(messages) == original_len


def test_repair_does_NOT_merge_codex_interim_assistants():
    """Codex Responses interim turns stay separate (encrypted replay state).

    The codex_responses api_mode keeps multiple consecutive incomplete
    assistant turns, each carrying distinct codex_reasoning_items /
    codex_message_items that must replay verbatim. Pass 0 must exempt them.
    Refs test_run_agent_codex_responses.py duplicate-detection tests.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "think hard"},
        {"role": "assistant", "content": "", "finish_reason": "incomplete",
         "codex_reasoning_items": [{"encrypted_content": "enc_first"}]},
        {"role": "assistant", "content": "", "finish_reason": "incomplete",
         "codex_reasoning_items": [{"encrypted_content": "enc_second"}]},
        {"role": "assistant", "content": "Final answer."},
    ]

    AIAgent._repair_message_sequence(agent, messages)

    interim = [m for m in messages if m.get("finish_reason") == "incomplete"]
    assert len(interim) == 2
    encs = [m["codex_reasoning_items"][0]["encrypted_content"] for m in interim]
    assert "enc_first" in encs and "enc_second" in encs
