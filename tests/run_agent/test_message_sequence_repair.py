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


def test_repair_preserves_persisted_mobile_receipt_proof_on_user_merge():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "interrupted", "_db_persisted": True},
        {
            "role": "user",
            "content": "mobile retry",
            "_db_persisted": True,
            "_mobile_mutation_receipt_tags": ["proof-1"],
        },
    ]

    AIAgent._repair_message_sequence(agent, messages)

    assert len(messages) == 1
    assert messages[0]["_mobile_mutation_persisted_receipt_tags"] == [
        "proof-1"
    ]


def test_repair_does_not_promote_unpersisted_mobile_receipt_proof():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "interrupted", "_db_persisted": True},
        {
            "role": "user",
            "content": "failed write",
            "_mobile_mutation_receipt_tags": ["proof-1"],
        },
    ]

    AIAgent._repair_message_sequence(agent, messages)

    assert len(messages) == 1
    assert "_mobile_mutation_persisted_receipt_tags" not in messages[0]


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


def test_repair_keeps_tool_matching_codex_call_id():
    """A valid tool result must survive when the assistant tool_call carries a
    Codex-format ``call_id`` distinct from ``id`` and the result matches on
    ``call_id`` (#58168).

    Before the fix, Pass 1 registered only ``tc.get("id")`` (``fc_...``) in the
    known-id set, so a result keyed on ``call_id`` (``call_...``) looked
    orphaned and was dropped -- leaving the assistant tool_call unanswered and
    triggering an HTTP 400 on strict providers (DeepSeek, Kimi):
    "Messages with role 'tool' must be a response to a preceding message with
    'tool_calls'".
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "fc_123", "call_id": "call_ABC",
                         "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_ABC", "content": "result"},
        {"role": "user", "content": "next"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[2]["tool_call_id"] == "call_ABC"


def test_repair_keeps_tool_matching_only_call_id():
    """Same as above but the assistant tool_call carries ONLY ``call_id`` (no
    ``id``). The result keyed on ``call_id`` must still be recognized (#58168).
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"call_id": "call_XYZ", "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_XYZ", "content": "result"},
        {"role": "user", "content": "next"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert any(m.get("role") == "tool" for m in messages)


def test_repair_keeps_tool_matching_id_when_call_id_also_present():
    """When the assistant tool_call carries both ``id`` and ``call_id`` and the
    result matches on ``id`` (OpenAI-compatible builder path), it must be kept
    (#58168 -- both keys are registered, so either matches).
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "fc_9", "call_id": "call_9",
                         "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "fc_9", "content": "result"},
        {"role": "user", "content": "next"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert any(m.get("role") == "tool" for m in messages)


def test_repair_still_drops_genuine_orphan_alongside_codex_pair():
    """Negative control for #58168: registering both id and call_id must NOT
    over-relax orphan detection. A genuinely orphaned tool result (matching
    neither the id nor the call_id of any assistant tool_call) is still
    dropped, while the valid codex-format pair in the same window survives.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "fc_1", "call_id": "call_1",
                         "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "valid"},
        {"role": "tool", "tool_call_id": "call_ORPHAN", "content": "stray"},
        {"role": "user", "content": "next"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 1
    tool_ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
    assert tool_ids == ["call_1"]


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


# ── tool_call_id de-duplication (#58327) ────────────────────────────────────
# Strict providers (DeepSeek) reject a payload where the same tool_call_id
# appears more than once with HTTP 400 "Duplicate value for 'tool_call_id'".


def test_repair_deduplicates_duplicate_tool_results():
    """A second tool result reusing an already-matched tool_call_id is dropped.

    repair_message_sequence consumes the id from known_tool_ids on first match
    so the duplicate falls into the repair/drop branch (#58327, kernel #55436).
    """
    from agent.agent_runtime_helpers import repair_message_sequence

    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "run the tool"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "test", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "res1"},
        {"role": "tool", "tool_call_id": "call_1", "content": "res1 duplicate"},
    ]
    repairs = repair_message_sequence(agent, messages)
    assert repairs == 1
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "res1"


def test_sanitize_deduplicates_duplicate_tool_results():
    """sanitize_api_messages (final pre-API chokepoint) drops duplicate tool
    results sharing a tool_call_id."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_X", "type": "function",
                         "function": {"name": "foo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_X", "content": "A"},
        {"role": "tool", "tool_call_id": "call_X", "content": "B (duplicate)"},
        {"role": "assistant", "content": "done"},
    ]
    out = sanitize_api_messages(list(messages))
    tool_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert tool_ids == ["call_X"]  # exactly one survives


def test_sanitize_deduplicates_duplicate_assistant_tool_call_ids():
    """sanitize_api_messages collapses duplicate tool_calls sharing an id
    WITHIN a single assistant message (the message[6] shape from #58327)."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_Y", "type": "function",
             "function": {"name": "foo", "arguments": "{}"}},
            {"id": "call_Y", "type": "function",
             "function": {"name": "bar", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_Y", "content": "r"},
    ]
    out = sanitize_api_messages(list(messages))
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    ids = [tc["id"] for tc in assistant["tool_calls"]]
    assert ids == ["call_Y"]  # duplicate collapsed


def test_sanitize_preserves_distinct_tool_call_ids():
    """Negative control: legitimate DISTINCT tool_call_ids must NOT be dropped
    (guards against over-dedup)."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_A", "type": "function",
             "function": {"name": "a", "arguments": "{}"}},
            {"id": "call_B", "type": "function",
             "function": {"name": "b", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_A", "content": "ra"},
        {"role": "tool", "tool_call_id": "call_B", "content": "rb"},
    ]
    out = sanitize_api_messages(list(messages))
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_A", "call_B"]
    assert sorted(m["tool_call_id"] for m in out if m.get("role") == "tool") == ["call_A", "call_B"]


def test_sanitize_drops_empty_tool_calls_array():
    """sanitize_api_messages strips ``tool_calls: []`` from assistant messages.

    DeepSeek v4 rejects an empty tool_calls array with HTTP 400 "Invalid
    'messages[N].tool_calls': empty array" (#58755). The empty array is
    semantically "no tool calls", so the key is dropped while content is
    preserved.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer", "tool_calls": []},
    ]
    out = sanitize_api_messages(list(messages))
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    assert "tool_calls" not in assistant
    assert assistant["content"] == "answer"


def test_sanitize_drops_non_list_tool_calls():
    """A malformed non-list ``tool_calls`` (e.g. None under the key) is also
    dropped so it can't reach a strict provider."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "assistant", "content": "text", "tool_calls": None},
    ]
    out = sanitize_api_messages(list(messages))
    assert "tool_calls" not in out[0]


def test_sanitize_does_not_mutate_original_on_empty_tool_calls():
    """Stripping must be non-destructive: the caller's message dicts (the
    persisted trajectory) keep their original ``tool_calls`` key."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    original_assistant = {"role": "assistant", "content": "answer", "tool_calls": []}
    messages = [{"role": "user", "content": "hi"}, original_assistant]
    sanitize_api_messages(list(messages))
    assert original_assistant["tool_calls"] == []  # untouched in-place


def test_sanitize_preserves_populated_tool_calls():
    """Negative control: a non-empty tool_calls array (with its matching tool
    result) must survive untouched."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_Z", "type": "function",
             "function": {"name": "foo", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_Z", "content": "r"},
    ]
    out = sanitize_api_messages(list(messages))
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_Z"]
