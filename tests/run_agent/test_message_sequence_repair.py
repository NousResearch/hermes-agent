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














def test_repair_keeps_tool_result_keyed_by_response_item_id():
    """The normalized call id must still pair the raw Responses item id."""
    from agent.agent_runtime_helpers import repair_message_sequence

    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_ABC",
                "call_id": "call_ABC",
                "response_item_id": "fc_123",
                "type": "function",
                "function": {"name": "x", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "fc_123", "content": "result"},
    ]

    repairs = repair_message_sequence(_bare_agent(), messages)

    assert repairs == 0
    assert messages[-1]["content"] == "result"


def test_coalesce_tool_call_id_uses_call_half_of_composite_id():
    """Raw bridge ids must use their canonical call half for tool results."""
    from agent.message_sanitization import coalesce_tool_call_id

    assert coalesce_tool_call_id({"id": "call_ABC|fc_123"}) == "call_ABC"


def test_sanitize_keeps_parallel_results_keyed_by_responses_id_variant():
    """A valid parallel Responses batch must not become all unavailable stubs.

    Responses/Codex calls can expose a response-item ``id`` (``fc_*``) and a
    distinct pairing ``call_id`` (``call_*``). The execution path may key the
    real result with either variant, but the pre-send sanitizer must recognize
    both as the same assistant call. This is the minimal reproduction of the
    reported all-results loss for a larger parallel batch.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    batch_size = 8
    messages = [
        {"role": "user", "content": "run the independent calls"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"fc_{i}",
                    "call_id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
                for i in range(batch_size)
            ],
        },
        *[
            {
                "role": "tool",
                "tool_call_id": f"fc_{i}",
                "content": f"real-result-{i}",
            }
            for i in range(batch_size)
        ],
    ]

    out = sanitize_api_messages(messages)

    results = [msg for msg in out if msg.get("role") == "tool"]
    assert [msg["tool_call_id"] for msg in results] == [
        f"fc_{i}" for i in range(batch_size)
    ]
    assert [msg["content"] for msg in results] == [
        f"real-result-{i}" for i in range(batch_size)
    ]


def test_repair_keeps_two_parallel_calls_answered_by_mixed_variants():
    """Consuming one alias group must not orphan a different parallel call."""
    from agent.agent_runtime_helpers import repair_message_sequence

    messages = [
        {"role": "user", "content": "do both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "call_id": "call_1",
                    "response_item_id": "fc_1",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "call_id": "call_2",
                    "response_item_id": "fc_2",
                    "type": "function",
                    "function": {"name": "y", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "r1"},
        {"role": "tool", "tool_call_id": "fc_2", "content": "r2"},
    ]

    repairs = repair_message_sequence(_bare_agent(), messages)

    assert repairs == 0
    assert [m["content"] for m in messages if m.get("role") == "tool"] == [
        "r1", "r2"
    ]


def test_repair_merge_drops_stale_api_content_sidecar_on_surviving_turn():
    """A pre-existing ``api_content`` sidecar on the surviving (first)
    assistant turn must be dropped when the merge rewrites ``content`` —
    otherwise the sidecar (which takes priority over ``content`` at
    API-build time, see ``conversation_loop``'s ``api_messages`` build)
    replays the STALE pre-merge bytes on the next call, silently discarding
    everything the merge just concatenated on. Same stale-field-survives-
    the-merge shape as the ``tool_calls`` gap above (#77921), for the
    ``api_content`` field instead.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "Q1"},
        {
            "role": "assistant",
            "content": "first reply",
            "api_content": "first reply (stale pre-merge bytes)",
        },
        {"role": "assistant", "content": "second reply"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 1
    assert len(messages) == 2
    assert "api_content" not in messages[1]
    assert messages[1]["content"] == "first reply\nsecond reply"


def test_repair_merge_preserves_api_content_sidecar_when_content_unchanged():
    """Negative control (#78063 review): ``api_content`` must NOT be dropped
    when the merge does not actually rewrite ``prev["content"]``.

    When the later assistant turn's content is ``None``, neither the
    both-str branch nor the ``not prev_content`` branch fires (``prev_content``
    is a truthy string, so ``not prev_content`` is False) -- ``prev["content"]``
    is left completely untouched. The sidecar is still the exact bytes
    previously sent for that UNCHANGED content, so dropping it here would
    diverge replay bytes and break the prompt-cache invariant for no reason.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "clean", "api_content": "wire bytes"},
        {"role": "assistant", "content": None},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 1
    assert len(messages) == 2
    assert messages[1]["content"] == "clean"
    assert messages[1]["api_content"] == "wire bytes"


def test_repair_merge_preserves_api_content_sidecar_when_content_unchanged_by_empty_string():
    """Negative control (wz-heng, #78063 review): ``content_rewritten`` must
    mean "the value changed", not "entered the assignment branch".

    Later turn's content is ``""`` -- the both-str branch fires and
    ``prev["content"]`` IS reassigned, but ``joined`` strips the falsy
    empty string away and collapses back to the original ``prev_content``
    unchanged. The sidecar must survive because no byte of ``content``
    actually moved.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "clean", "api_content": "wire bytes"},
        {"role": "assistant", "content": ""},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 1
    assert len(messages) == 2
    assert messages[1]["content"] == "clean"
    assert messages[1]["api_content"] == "wire bytes"


def test_repair_merge_preserves_api_content_sidecar_with_multimodal_content():
    """Same negative control, multimodal (list) content on the later turn --
    the merge intentionally leaves list content alone (see the merge's
    docstring), so ``prev["content"]`` is untouched and the sidecar must
    survive."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "clean", "api_content": "wire bytes"},
        {"role": "assistant", "content": [{"type": "text", "text": "img context"}]},
    ]

    AIAgent._repair_message_sequence(agent, messages)

    assert len(messages) == 2
    assert messages[1]["content"] == "clean"
    assert messages[1]["api_content"] == "wire bytes"



def test_sanitize_consumes_all_responses_id_variants_for_duplicate_result():
    """A sibling-id replay must not replace the first real result."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_ABC",
                "call_id": "call_ABC",
                "response_item_id": "fc_123",
                "type": "function",
                "function": {"name": "x", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "fc_123", "content": "real"},
        {"role": "tool", "tool_call_id": "call_ABC", "content": "replayed"},
    ]

    out = sanitize_api_messages(messages)

    assert [msg["content"] for msg in out if msg.get("role") == "tool"] == [
        "real"
    ]


def test_tool_executor_uses_canonical_responses_pairing_id():
    """The executor must emit the id used by the normalized assistant turn."""
    from types import SimpleNamespace

    from agent.tool_executor import _pairing_tool_call_id

    assert _pairing_tool_call_id(
        SimpleNamespace(id="fc_123", call_id="call_ABC")
    ) == "call_ABC"
    assert _pairing_tool_call_id(
        SimpleNamespace(id="call_ABC|fc_123")
    ) == "call_ABC"



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





def test_flush_guard_clamps_overshooting_cursor():
    """_flush_messages_to_session_db safety net: an overshooting cursor must
    not produce a negative-start slice that skips everything (#44837)."""

    class _DB:
        def __init__(self):
            self.rows = []

        def append_message(self, **kw):
            self.rows.append(kw)

        def append_messages_batch(self, session_id, messages, **kw):
            for m in messages:
                self.rows.append(dict(m, session_id=session_id))
            return list(range(1, len(messages) + 1))

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










# ── tool_call_id de-duplication (#58327) ────────────────────────────────────
# Strict providers (DeepSeek) reject a payload where the same tool_call_id
# appears more than once with HTTP 400 "Duplicate value for 'tool_call_id'".




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


def test_sanitize_drops_tool_result_with_missing_tool_call_id():
    """A tool message with NO ``tool_call_id`` must be dropped, not silently
    passed through.

    Before the fix: ``result_call_ids`` only ever collects TRUTHY ids, so a
    missing/empty id is never added to that set and can therefore never land
    in ``orphaned_results`` (a set-difference against ``surviving_call_ids``)
    either -- the message survives sanitize_api_messages untouched and can
    reach the provider with no ``tool_call_id`` at all, a schema violation
    on strict OpenAI-compatible providers (#78071).
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_Z", "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_Z", "content": "real result"},
        {"role": "tool", "tool_call_id": "", "content": "no id"},
        {"role": "tool", "content": "id key entirely absent"},
    ]
    out = sanitize_api_messages(list(messages))
    tool_ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
    assert tool_ids == ["call_Z"]  # only the properly-paired result survives


# ── tool_call_id reuse by local servers (#70724) ────────────────────────────
# llama.cpp emits ONE constant tool_call_id for every tool call it returns, so
# ``tool_call_id`` is not globally unique in practice. The #58327 dedup pass
# must key off outstanding calls, not "seen at any point", or every result
# after the first is deleted and the agent stops mid-task.


CONSTANT_ID = "ZsSt4SkIFMRz0HtqT7MTlimNvzlKM896"


def _call(cid, name="terminal"):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": "{}"}}]}


def _result(cid, content):
    return {"role": "tool", "tool_call_id": cid, "name": "terminal",
            "content": content}


def test_sanitize_keeps_results_when_server_reuses_one_tool_call_id():
    """Every answered call survives even when all of them share one id.

    Contract: a tool result is dropped for being unanswerable, never for
    reusing an id that an earlier call already retired.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [{"role": "user", "content": "do three steps"}]
    for i in range(3):
        messages.append(_call(CONSTANT_ID))
        messages.append(_result(CONSTANT_ID, f"step {i} output"))

    out = sanitize_api_messages(list(messages))
    results = [m for m in out if m.get("role") == "tool"]
    assert [m["content"] for m in results] == [
        "step 0 output", "step 1 output", "step 2 output",
    ]
    calls = [m for m in out if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(calls) == 3


def test_sanitize_still_drops_replayed_result_for_retired_call():
    """The #58327 protection holds: a second result for an already-answered
    call answers nothing outstanding and is still dropped."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "hi"},
        _call(CONSTANT_ID),
        _result(CONSTANT_ID, "real"),
        _result(CONSTANT_ID, "replayed by a retry/resume glitch"),
    ]
    out = sanitize_api_messages(list(messages))
    assert [m["content"] for m in out if m.get("role") == "tool"] == ["real"]


def test_sanitize_preserves_deterministic_local_ids_across_turns():
    """Hermes' own deterministic call ids (fn-name+args hashes / local
    counters) legitimately repeat across turns — both must survive.

    Scenario surfaced in #76632: two image_generate rounds emit the same
    local ids (``image_generate:0``/``:1``) in successive assistant turns.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    def _tc(cid):
        return {"id": cid, "type": "function",
                "function": {"name": "image_generate", "arguments": "{}"}}

    messages = [{"role": "user", "content": "make images"}]
    for turn in ("one", "two"):
        messages.append({"role": "assistant", "content": turn,
                         "tool_calls": [_tc("image_generate:0"), _tc("image_generate:1")]})
        messages.append({"role": "tool", "tool_call_id": "image_generate:0",
                         "content": f"imgA-{turn}"})
        messages.append({"role": "tool", "tool_call_id": "image_generate:1",
                         "content": f"imgB-{turn}"})

    out = sanitize_api_messages(list(messages))
    assistants = [m for m in out if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistants) == 2
    for a in assistants:
        assert [tc["id"] for tc in a["tool_calls"]] == [
            "image_generate:0", "image_generate:1"]
    tool_ids = sorted(m["tool_call_id"] for m in out if m.get("role") == "tool")
    assert tool_ids == ["image_generate:0", "image_generate:0",
                        "image_generate:1", "image_generate:1"]


def test_sanitize_keeps_all_results_over_fifty_turn_constant_id_session():
    """Kimi K3 / llama.cpp field repro (#70724, #70734): 50 sequential calls
    all sharing one id must all survive — stock behavior kept 1/50."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [{"role": "user", "content": "Run 50 steps."}]
    for step in range(50):
        messages.append(_call(CONSTANT_ID))
        messages.append(_result(CONSTANT_ID, f"completed-step-{step}"))

    out = sanitize_api_messages(list(messages))
    calls = [m for m in out if m.get("role") == "assistant" and m.get("tool_calls")]
    results = [m for m in out if m.get("role") == "tool"]
    assert len(calls) == 50
    assert len(results) == 50
    assert results[-1]["content"] == "completed-step-49"


def test_sanitize_drops_result_with_no_preceding_call():
    """A tool result that never had a call is an orphan regardless of id."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    out = sanitize_api_messages([
        {"role": "user", "content": "hi"},
        _result("id_never_requested", "orphan"),
    ])
    assert [m for m in out if m.get("role") == "tool"] == []


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


def test_repair_drops_stale_empty_tool_calls_on_merged_assistant():
    """repair_message_sequence must drop a stale ``tool_calls: []`` on the
    surviving message of a consecutive-assistant merge (#77921).

    The chokepoint sanitizer (sanitize_api_messages) only patches the per-call
    wire copy — a ``[]`` left on the repaired live/persisted trajectory is
    replayed on the next turn and 400s strict providers (DeepSeek v4). The
    merge's union branches only ever set non-empty lists or leave the key
    untouched, so the empty array survives into the persisted state."""
    from agent.agent_runtime_helpers import repair_message_sequence

    messages = [
        {"role": "user", "content": "hi"},
        # surviving turn carries a stale empty tool_calls from an earlier pass
        {"role": "assistant", "content": "first", "tool_calls": []},
        {"role": "assistant", "content": "second"},
    ]
    # A dummy agent object is enough — repair only reads message roles/content.
    agent = type("Agent", (), {})()
    n = repair_message_sequence(agent, messages)
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistants) == 1
    assert "tool_calls" not in assistants[0]
    assert "second" in assistants[0]["content"]


def test_repair_preserves_reasoning_and_reasoning_details_on_merged_assistant():
    """repair_message_sequence must preserve canonical reasoning and reasoning_details
    when merging consecutive assistant messages if earlier message lacked them."""
    from agent.agent_runtime_helpers import repair_message_sequence

    agent = type("Agent", (), {})()

    # Case 1: second message carries reasoning
    messages_1 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "thinking scaffold..."},
        {"role": "assistant", "content": "final answer", "reasoning": "thought trace here"},
    ]
    repair_message_sequence(agent, messages_1)
    assistants_1 = [m for m in messages_1 if m.get("role") == "assistant"]
    assert len(assistants_1) == 1
    assert assistants_1[0]["reasoning"] == "thought trace here"

    # Case 2: second message carries reasoning_details
    messages_2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "start"},
        {"role": "assistant", "content": "end", "reasoning_details": [{"type": "thought", "text": "details"}]},
    ]
    repair_message_sequence(agent, messages_2)
    assistants_2 = [m for m in messages_2 if m.get("role") == "assistant"]
    assert len(assistants_2) == 1
    assert assistants_2[0]["reasoning_details"] == [{"type": "thought", "text": "details"}]

    # Case 3: first message already has reasoning, keep first
    messages_3 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "start", "reasoning": "first reasoning"},
        {"role": "assistant", "content": "end", "reasoning": "second reasoning"},
    ]
    repair_message_sequence(agent, messages_3)
    assistants_3 = [m for m in messages_3 if m.get("role") == "assistant"]
    assert len(assistants_3) == 1
    assert assistants_3[0]["reasoning"] == "first reasoning"


def test_repair_keeps_tool_result_when_tool_calls_are_sdk_objects():
    """repair_message_sequence must not drop a valid tool result just because
    the assistant's ``tool_calls`` entries are unserialized SDK objects
    (e.g. ``ChatCompletionMessageToolCall``) instead of plain dicts.

    Host-fed / pre-serialization histories (gateway multi-queue replay,
    session resume) can carry SDK tool_call objects into this pass. The
    dict-only ``tc.get(key)`` lookup previously left ``known_tool_ids``
    empty for such messages, so the id-matching pass below misclassified
    the legitimate tool result as an orphan and silently deleted it —
    corrupting the persisted history and leaving the assistant's
    tool_calls unanswered (itself a strict-provider HTTP 400 trigger)."""
    from agent.agent_runtime_helpers import repair_message_sequence

    class SDKToolCall:
        def __init__(self, call_id):
            self.id = call_id
            self.call_id = None
            self.function = type("F", (), {"name": "read_file", "arguments": "{}"})()

    messages = [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [SDKToolCall("call_1")],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
    ]
    agent = type("Agent", (), {})()
    repair_message_sequence(agent, messages)

    roles = [m.get("role") for m in messages]
    assert "tool" in roles, "legitimate tool result was dropped as a false orphan"
    tool_msg = next(m for m in messages if m.get("role") == "tool")
    assert tool_msg["content"] == "file contents"






# ── Self-recovery: heal empty-content non-final messages ──────────────────
# Repro of the production incident: a dead stream persisted an empty-content
# assistant stub mid-transcript, and every later request 400'd with
# "all messages must have non-empty content except for the optional final
# assistant message" (INVALID_REQUEST_BODY). sanitize_api_messages now heals
# such turns on the per-call copy so the session recovers itself in memory.


def test_sanitize_interrupted_call_stubbed_and_replay_kept():
    """A crash/resume glitch replays the SAME assistant call while the first
    occurrence is still positionally unanswered.

    With the positional pairing pass (#94704), the interrupted first call is
    no longer a dedup candidate: it gets a synthetic unavailable-result stub
    at the end of its (empty) tool run, and the replayed call — now with its
    own immediate result — is a legitimate new round. This supersedes the
    pre-positional shape where the dedup pass removed the replay's tool_calls
    (the #64335 empty-key guard is still exercised by
    ``test_sanitize_drops_empty_tool_calls_array``).
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    # Simulate a crash/resume glitch that replays the SAME assistant call
    # while the first is still outstanding (no tool result has answered it
    # yet, and a second assistant turn interrupts the run).
    messages = [
        {"role": "user", "content": "step 1"},
        {"role": "assistant", "content": "running",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "foo", "arguments": "{}"}}]},
        {"role": "assistant", "content": "retrying",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "foo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": "result 1"},
    ]

    out = sanitize_api_messages(list(messages))

    # Both assistant turns survive; the interrupted one is answered by a
    # stub inserted before the replay, the replay by its real result.
    assert [m["role"] for m in out] == [
        "user", "assistant", "tool", "assistant", "tool",
    ]
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "call_A"
    assert "Result unavailable" in out[2]["content"]
    assistants = [m for m in out if m.get("role") == "assistant"]
    assert all("tool_calls" in a for a in assistants)
    assert [a["tool_calls"][0]["id"] for a in assistants] == ["call_A", "call_A"]
    assert out[4]["content"] == "result 1"


def test_repair_drops_duplicate_tool_result_keyed_on_sibling_id():
    """A duplicate result must be dropped even when it uses the OTHER id variant.

    A Codex/Responses ``tool_call`` carries both ``id`` (``fc_...``) and
    ``call_id`` (``call_...``); both are registered so a result keyed on either
    is recognised (#58168). The duplicate guard (#58327) consumed only the id
    the first result referenced, leaving its sibling live — so a second result
    keyed on that sibling sailed through and two tool messages were replayed
    for a single call, re-creating the HTTP 400 the guard exists to prevent.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "fc_1", "call_id": "call_1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "first"},
        {"role": "tool", "tool_call_id": "fc_1", "content": "duplicate"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 1
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == ["first"]


def test_repair_keeps_both_results_for_two_codex_calls_mixed_keys():
    """Consuming a call's sibling ids must not orphan a *different* call.

    Two parallel Codex tool_calls answered via different id variants
    (one by ``call_id``, one by ``id``) are both legitimate and must survive.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "fc_1", "call_id": "call_1", "type": "function",
              "function": {"name": "f", "arguments": "{}"}},
             {"id": "fc_2", "call_id": "call_2", "type": "function",
              "function": {"name": "g", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "r1"},
        {"role": "tool", "tool_call_id": "fc_2", "content": "r2"},
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert [m["content"] for m in messages if m.get("role") == "tool"] == ["r1", "r2"]


def test_sanitize_dedup_pass_keeps_batch_results_keyed_on_divergent_id():
    """The step-3 dedup pass must not delete real results whose tool_call_id
    matches the non-coalesced id variant.

    A parallel batch of Codex/Responses-style tool_calls carries divergent
    ``id`` (fc_...) and ``call_id`` (call_...). Step 2's variant-aware
    matching preserves results keyed on either variant, but the dedup pass
    tracked only the coalesced (call_id||id) value in
    ``outstanding_call_ids`` — so every result keyed on ``id`` looked like it
    answered no outstanding call and was deleted wholesale. Whole parallel
    batches vanished with no stub at all (#93251, #55626 class).
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"fc_{i}", "call_id": f"call_{i}", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}
            for i in range(4)
        ]},
    ] + [
        {"role": "tool", "tool_call_id": f"fc_{i}", "content": f"REAL {i}"}
        for i in range(4)
    ]

    out = sanitize_api_messages(messages)

    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == [
        "REAL 0", "REAL 1", "REAL 2", "REAL 3",
    ], "real batch results must survive the dedup pass"
    assert not any("Result unavailable" in m["content"] for m in tool_msgs)


def test_sanitize_dedup_pass_still_drops_result_replayed_on_sibling_id():
    """Variant-aware dedup must still drop a SECOND result for an
    already-answered call even when the replay uses the OTHER id variant
    (strict providers 400 on duplicate tool_call_id)."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fc_1", "call_id": "call_1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "fc_1", "content": "first"},
        {"role": "tool", "tool_call_id": "call_1", "content": "sibling replay"},
    ]

    out = sanitize_api_messages(messages)

    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == ["first"]


def test_sanitize_dedup_pass_rearms_constant_llamacpp_id():
    """llama.cpp emits one constant id for every call; a fresh assistant call
    re-arms the id so the second round's result still survives (#58327
    outstanding-call semantics must be preserved by the variant-set change)."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_K", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_K", "content": "round1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_K", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_K", "content": "round2"},
    ]

    out = sanitize_api_messages(messages)

    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == ["round1", "round2"]


def test_sanitize_keeps_result_keyed_on_composite_bridge_id():
    """A tool result keyed on the composite ``call|item`` bridge spelling
    (#63000) must pair with a tool_call carrying the split id/call_id
    fields — and vice versa."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    # Result keyed on composite; call carries split fields.
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fc_1", "call_id": "call_1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1|fc_1", "content": "REAL"},
    ]
    out = sanitize_api_messages(messages)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == ["REAL"]

    # Call carries only the composite id; result keyed on the bare half.
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_2|fc_2", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "REAL bare"},
    ]
    out = sanitize_api_messages(messages)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == ["REAL bare"]


def test_compressor_sanitize_keeps_composite_keyed_pair():
    """The compression sanitizer must apply the same alias expansion on the
    RESULT side: a composite-keyed result pairs with its split-field call
    instead of being dropped and its call stripped (#63000)."""
    from agent.context_compressor import ContextCompressor

    cc = ContextCompressor.__new__(ContextCompressor)
    cc.quiet_mode = True
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fc_7", "call_id": "call_7", "type": "function",
             "function": {"name": "s", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_7|fc_7", "content": "res"},
        {"role": "user", "content": "next"},
    ]
    out = cc._sanitize_tool_pairs(msgs)
    asst = next(m for m in out if m.get("role") == "assistant")
    assert asst.get("tool_calls"), "valid tool_call must not be stripped"
    assert [m["content"] for m in out if m.get("role") == "tool"] == ["res"]

    # Negative control: composite orphan (matches nothing) still dropped.
    msgs = [
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "tool_call_id": "call_z|fc_z", "content": "orphan"},
        {"role": "user", "content": "next"},
    ]
    out = cc._sanitize_tool_pairs(msgs)
    assert not any(m.get("role") == "tool" for m in out)


# ── _classify_tool_call_orphans ─────────────────────────────────────────

def test_classify_orphans_empty():
    from agent.agent_runtime_helpers import _classify_tool_call_orphans
    sv, rs, orphaned, missing = _classify_tool_call_orphans([])
    assert sv == set()
    assert rs == set()
    assert orphaned == []
    assert missing == []


def test_classify_orphans_clean_pair():
    from agent.agent_runtime_helpers import _classify_tool_call_orphans
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    sv, rs, orphaned, missing = _classify_tool_call_orphans(messages)
    assert sv == {"call_1"}
    assert rs == {"call_1"}
    assert orphaned == []
    assert missing == []


def test_classify_orphans_detects_orphaned_result():
    from agent.agent_runtime_helpers import _classify_tool_call_orphans
    messages = [
        {"role": "tool", "tool_call_id": "orphan_1", "content": "no matching call"},
    ]
    sv, rs, orphaned, missing = _classify_tool_call_orphans(messages)
    assert [m["tool_call_id"] for m in orphaned] == ["orphan_1"]
    assert missing == []


def test_classify_orphans_detects_missing_result():
    from agent.agent_runtime_helpers import _classify_tool_call_orphans
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ]
    sv, rs, orphaned, missing = _classify_tool_call_orphans(messages)
    assert orphaned == []
    assert [tc["id"] for tc in missing] == ["call_1"]


def test_classify_orphans_mixed():
    from agent.agent_runtime_helpers import _classify_tool_call_orphans
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "call_A", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "call_B", "type": "function", "function": {"name": "g", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_A", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_C", "content": "orphan"},
    ]
    sv, rs, orphaned, missing = _classify_tool_call_orphans(messages)
    assert sv == {"call_A", "call_B"}
    assert rs == {"call_A", "call_C"}
    assert [m["tool_call_id"] for m in orphaned] == ["call_C"]
    assert [tc["id"] for tc in missing] == ["call_B"]


# ── Positional tool_call <-> tool_result pairing ───────────────────────────
# Production incident (session 4d8727cbcf04): context compression displaced
# a tool result ~110 messages past its declaring assistant turn (across a
# user turn). repair_message_sequence Pass 1 dropped the displaced result as
# stray but left the declaring assistant carrying an UNANSWERED tool_call
# with empty content; sanitize_api_messages' global-set stub pass saw the
# displaced result still present, considered the call answered, and injected
# no stub — DeepSeek v4 then 400'd the payload: "An assistant message with
# 'tool_calls' must be followed by tool messages responding to each
# 'tool_call_id' (insufficient tool messages following tool_calls message)".


def _assistant_with_call(call_id, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }],
    }


def _tool_result(call_id, content="out"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_repair_prunes_tool_call_whose_result_was_displaced():
    """Pass 2: a tool_call with no result in the immediately-following run is
    pruned, even when its result exists far later (post-compression shape).
    The assistant turn keeps its plain content once the calls are pruned.
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "do it"},
        _assistant_with_call("call_A", content=""),          # declares A, never answered here
        _assistant_with_call("call_B", content="second"),    # merged into the above (Pass 0)
        _tool_result("call_B"),
        {"role": "user", "content": "meanwhile"},            # user redirect
        _tool_result("call_A", content="late result"),       # displaced: dropped by Pass 1
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs >= 1
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistants) == 1
    ids = [tc["id"] for tc in assistants[0]["tool_calls"]]
    assert ids == ["call_B"]          # unanswered call_A pruned
    assert assistants[0]["content"] == "second"
    # The legitimate call_B result survives; only the displaced late
    # call_A result was dropped.
    tools = [m for m in messages if m.get("role") == "tool"]
    assert len(tools) == 1
    assert tools[0]["tool_call_id"] == "call_B"


def test_repair_drops_turn_when_pruned_calls_were_only_payload():
    """Pass 2: when pruning empties the merged assistant turn (no content,
    no reasoning), the whole turn is dropped instead of sending an empty
    non-final assistant message (itself a 400 on most providers).
    """
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "do it"},
        _assistant_with_call("call_A"),                    # empty content
        {"role": "assistant", "content": ""},              # merged in (Pass 0)
        {"role": "user", "content": "redirected"},
        _tool_result("call_A", content="late"),            # displaced: dropped
    ]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs >= 2
    assert all(m.get("role") != "assistant" for m in messages)
    # The two user turns merge (Pass 3); nothing was lost.
    users = [m for m in messages if m.get("role") == "user"]
    assert len(users) == 1
    assert "do it" in users[0]["content"] and "redirected" in users[0]["content"]


def test_repair_keeps_calls_answered_within_following_run():
    """Negative control: a legitimate assistant(tool_calls)+tool run must
    survive Pass 2 untouched (the ongoing dialog pattern)."""
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "Q1"},
        _assistant_with_call("t1", content=""),
        _tool_result("t1"),
        {"role": "user", "content": "Q2"},
    ]
    original = [dict(m) for m in messages]

    repairs = AIAgent._repair_message_sequence(agent, messages)

    assert repairs == 0
    assert messages == original


def test_sanitize_stubs_call_unanswered_positionally_even_if_result_exists_elsewhere():
    """sanitize_api_messages must inject a stub right after the declaring
    assistant message when no result follows it, EVEN IF a (displaced)
    result exists later in the transcript — the global-set check missed
    this exact shape (production 400, session 4d8727cbcf04)."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "do it"},
        _assistant_with_call("call_A", content=""),
        {"role": "user", "content": "meanwhile"},
        _tool_result("call_A", content="late result"),
    ]

    out = sanitize_api_messages(list(messages))

    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "tool", "user"]
    stub = out[2]
    assert stub["tool_call_id"] == "call_A"
    assert "Result unavailable" in stub["content"]
    # The displaced late result is dropped (positional orphan).
    assert "late result" not in [m.get("content", "") for m in out]


def test_sanitize_drops_result_appearing_before_its_call():
    """A tool result that precedes its declaring assistant message is a
    positional orphan — strict providers reject 'role=tool' messages that
    don't follow a tool_calls message."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "do it"},
        _tool_result("call_A"),                    # before its call
        _assistant_with_call("call_A", content=""),
        _tool_result("call_A"),
    ]

    out = sanitize_api_messages(list(messages))

    tools = [m for m in out if m.get("role") == "tool"]
    assert len(tools) == 1                          # only the valid one survives
    assert tools[0]["tool_call_id"] == "call_A"


def test_sanitize_positional_pairing_untouched_valid_transcript():
    """Negative control: a fully paired transcript (each tool-calling
    assistant immediately followed by its results) gets no stubs and loses
    no results."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "do it"},
        _assistant_with_call("call_A", content=""),
        _tool_result("call_A"),
        _assistant_with_call("call_B", content=""),
        _tool_result("call_B"),
        {"role": "assistant", "content": "done"},
    ]

    out = sanitize_api_messages(list(messages))

    assert [m["role"] for m in out] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    assert all("Result unavailable" not in str(m.get("content", "")) for m in out)


def test_sanitize_stubs_replayed_call_masked_by_historical_result():
    """Issue #94704 acceptance shape: a historical result masks a later
    replayed call with no immediate result.

    ```text
    user
    assistant(tool_calls=[call_old])                # first declaration, answered
    tool(call_old)                                  # historical completed call
    user
    assistant(tool_calls=[call_old, call_new])      # call_old REPLAYED
      tool(call_new)                                # call_old unanswered here
    ```

    The global presence check sees ``call_old`` answered somewhere and
    injects no stub; DeepSeek v4 rejects the payload because the second
    declaration's contiguous tool run covers only ``call_new``. The
    positional pass must stub ``call_old`` at the end of the second run
    while keeping the historical result and the fresh result intact.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "step 1"},
        _assistant_with_call("call_old", content=""),
        _tool_result("call_old", content="historical result"),
        {"role": "user", "content": "step 2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_old", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}},
                {"id": "call_new", "type": "function",
                 "function": {"name": "g", "arguments": "{}"}},
            ],
        },
        _tool_result("call_new", content="fresh result"),
    ]

    out = sanitize_api_messages(list(messages))

    # The second run ends with a stub for the replayed call_old, inserted
    # right after the fresh result and BEFORE any user turn.
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "tool", "user", "assistant",
                     "tool", "tool"]
    second_run = out[4:]
    assert [m["tool_call_id"] for m in second_run if m["role"] == "tool"] == [
        "call_new", "call_old",
    ]
    stub = second_run[2]
    assert stub["role"] == "tool"
    assert stub["tool_call_id"] == "call_old"
    assert "Result unavailable" in stub["content"]
    # Historical + fresh results both survive untouched.
    contents = [m.get("content", "") for m in out]
    assert "historical result" in contents
    assert "fresh result" in contents


def test_sanitize_stubs_interrupted_first_occurrence_keeps_replay_pair():
    """Production shape (session 7d57a602b83d): an interrupted turn persists
    an assistant tool_call with NO result (next message is user); on resume
    the SAME call is re-issued and persisted again WITH its result. Each id
    therefore exists twice — the first occurrence must be stubbed so the
    positional invariant holds, while the replayed pair stays untouched.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "do it"},
        _assistant_with_call("call_A", content=""),   # interrupted: no result
        {"role": "user", "content": "resumed"},
        _assistant_with_call("call_A", content=""),   # replay: paired
        _tool_result("call_A", content="real result"),
    ]

    out = sanitize_api_messages(list(messages))

    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "tool", "user", "assistant", "tool"]
    # Stub inserted for the interrupted first occurrence, before the user turn.
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "call_A"
    assert "Result unavailable" in out[2]["content"]
    # Replayed pair keeps the REAL result, not a stub.
    assert out[5]["content"] == "real result"




# ── Bridged tool_call: wire-visible response name must echo the call name ──
# Google matches functionResponse.name against functionCall.name and rejects a
# mismatch with HTTP 400 INVALID_ARGUMENT. #72089 fixed this for the native
# Gemini adapter; requests reaching Gemini through an OpenAI-compatible
# gateway (OpenRouter, Vertex/LiteLLM proxies) skip that translation, so the
# chokepoint sanitizer has to hold the same invariant.


def test_sanitize_realigns_bridged_tool_result_name_with_call_name():
    """A tool_search-bridged result must go on the wire as ``tool_call``.

    The executor labels the result with the unwrapped internal tool name for
    dispatch/hooks/logging; Gemini sees that as a functionResponse whose name
    does not match its functionCall and 400s.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "file an issue"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {
                             "name": "tool_call",
                             "arguments": ('{"name": "mcp__github__create_issue",'
                                           ' "arguments": {"title": "Bug"}}'),
                         }}]},
        {"role": "tool", "name": "mcp__github__create_issue",
         "tool_name": "mcp__github__create_issue",
         "tool_call_id": "call_1", "content": '{"number": 123}'},
    ]
    out = sanitize_api_messages(list(messages))
    result = [m for m in out if m.get("role") == "tool"][0]
    assert result["name"] == "tool_call"
    # The internal name stays available for the session DB / UI, and the
    # caller's own message objects are untouched (per-call copy only).
    assert result["tool_name"] == "mcp__github__create_issue"
    assert messages[2]["name"] == "mcp__github__create_issue"


def test_sanitize_leaves_already_matching_tool_result_name_alone():
    """A directly-called tool already agrees — nothing to rewrite."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_weather",
         "tool_call_id": "call_1", "content": "sunny"},
    ]
    out = sanitize_api_messages(list(messages))
    assert [m["name"] for m in out if m.get("role") == "tool"] == ["get_weather"]


def test_sanitize_does_not_invent_a_name_on_unnamed_tool_results():
    """A result with no ``name`` is already valid — the id pairs it.

    Adding the field would make clean transcripts non-identical on the wire
    and needlessly perturb prompt caching.
    """
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    out = sanitize_api_messages(list(messages))
    assert out == messages


def test_sanitize_realigns_bridged_name_when_call_id_is_padded():
    """Ids are stripped on both sides, so a padded id still pairs."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "file an issue"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": " call_1 ", "type": "function",
                         "function": {"name": "tool_call", "arguments": "{}"}}]},
        {"role": "tool", "name": "mcp__github__create_issue",
         "tool_call_id": "call_1", "content": "{}"},
    ]
    out = sanitize_api_messages(list(messages))
    assert [m["name"] for m in out if m.get("role") == "tool"] == ["tool_call"]


def test_sanitize_drops_bridged_result_whose_call_frame_was_pruned():
    """A result cannot keep a stale name if it has no call frame to disagree
    with: the orphan pass removes it before the realignment pass runs, so the
    mismatch never reaches the provider."""
    from agent.agent_runtime_helpers import sanitize_api_messages

    messages = [
        {"role": "user", "content": "file an issue"},
        # assistant tool_calls frame dropped by compression
        {"role": "tool", "name": "mcp__github__create_issue",
         "tool_call_id": "call_1", "content": "{}"},
    ]
    out = sanitize_api_messages(list(messages))
    assert [m.get("role") for m in out] == ["user"]
