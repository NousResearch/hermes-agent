"""Regression coverage for zero-user compaction integrity (#64539)."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSION_CONTINUATION_USER_CONTENT,
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    HISTORICAL_TASK_HEADING,
    MAX_ITERATIONS_SUMMARY_REQUEST,
    SUMMARY_PREFIX,
    ContextCompressor,
    _NO_USER_TASK_SENTINEL,
)
from agent.conversation_compression import (
    _ensure_compressed_has_user_turn,
    compress_context,
)
from hermes_state import SessionDB
from tools.process_registry_notifications import format_process_notification
from tools.todo_tool import TODO_INJECTION_HEADER


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _valid_zero_user_summary(label: str = "Checked artifacts.") -> str:
    return f"""{HISTORICAL_TASK_HEADING}
{_NO_USER_TASK_SENTINEL}

## Goal
Historical cron work only.

## Completed Actions
1. {label}

## Resolved Questions
None. No user-authored questions exist.

## Historical Pending User Asks
None. No user-authored requests exist.
"""


def _assistant_tool_turns(start: int, count: int) -> list[dict]:
    turns: list[dict] = []
    for idx in range(start, start + count):
        turns.extend(
            [
                {
                    "role": "assistant",
                    "content": "Continuing scheduled work in English.",
                    "tool_calls": [
                        {
                            "id": f"call-{idx}",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{idx}",
                    "content": "/workspace/project\n" + ("x" * 300),
                },
            ]
        )
    return turns


def _assistant_turns(start: int, count: int) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"Scheduled step {idx} completed. " + ("x" * 500),
        }
        for idx in range(start, start + count)
    ]


def _lifecycle_agent(db: SessionDB, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 2
    agent.context_compressor.tail_token_budget = 80
    agent._todo_store.write(
        [{"id": "inspect", "content": "Inspect artifacts", "status": "pending"}]
    )
    return agent


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=0,
            protect_last_n=2,
            quiet_mode=True,
        )
    instance.tail_token_budget = 80
    return instance


def test_generate_summary_rejects_fabricated_user_ask(compressor):
    fabricated = f"""{HISTORICAL_TASK_HEADING}
User asked: 'Waar zijn de bestanden gedownload?'

## Goal
Vind de bestanden.
"""

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(fabricated),
    ):
        result = compressor._generate_summary(_assistant_tool_turns(0, 2))

    assert result is None
    assert compressor._previous_summary is None
    assert "invented user attribution" in compressor._last_summary_error




def test_zero_user_provenance_survives_iterative_compaction(compressor):
    messages = _assistant_tool_turns(0, 12)
    first_summary = f"{SUMMARY_PREFIX}\n{_valid_zero_user_summary('First pass').strip()}"

    with patch.object(compressor, "_generate_summary", return_value=first_summary):
        first = compressor.compress(messages, current_tokens=90_000)

    first_handoffs = [
        message
        for message in first
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(first_handoffs) == 1
    assert first_handoffs[0][COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        resumed = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=0,
            protect_last_n=2,
            quiet_mode=True,
        )
    resumed.tail_token_budget = 80
    # SessionDB persists the summary content/role but not arbitrary internal
    # message keys. Simulate that round trip: the exact sentinel must recover
    # false provenance even when both in-process metadata keys are absent.
    persisted_handoff = dict(first_handoffs[0])
    persisted_handoff.pop(COMPRESSED_SUMMARY_METADATA_KEY)
    persisted_handoff.pop(COMPRESSED_SUMMARY_HAS_USER_TURN_KEY)
    second_input = [persisted_handoff, *_assistant_tool_turns(20, 12)]

    def assert_provenance_then_summarize(*_args, **_kwargs):
        assert resumed._summary_has_user_turn is False
        return f"{SUMMARY_PREFIX}\n{_valid_zero_user_summary('Second pass').strip()}"

    with patch.object(
        resumed,
        "_generate_summary",
        side_effect=assert_provenance_then_summarize,
    ):
        second = resumed.compress(second_input, current_tokens=90_000)

    second_handoffs = [
        message
        for message in second
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(second_handoffs) == 1
    assert second_handoffs[0][COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False


def test_max_iterations_nudge_is_synthetic_not_actionable():
    """#78580: the max-iteration runtime nudge is runtime scaffolding, not a
    human turn. It is appended as ``role="user"`` and persisted verbatim in
    state.db (metadata flags do not survive projection), so recognition must be
    content-based — exactly like the continuation/todo markers."""
    # The projected form: a bare role/content row with no internal metadata.
    nudge = {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST}

    assert ContextCompressor._is_synthetic_compression_user_turn(nudge) is True
    # A real human turn with the same shape stays actionable.
    human = {"role": "user", "content": "Ship the release notes for v2."}
    assert ContextCompressor._is_synthetic_compression_user_turn(human) is False
    assert ContextCompressor._transcript_has_real_user_turn([nudge]) is False
    assert ContextCompressor._transcript_has_real_user_turn([human, nudge]) is True


def test_real_task_wins_over_trailing_max_iterations_nudge(compressor):
    """The tail anchor must resolve to the human task, not the nudge that the
    runtime appended after it when iterations were exhausted."""
    human = {"role": "user", "content": "Refactor the auth module and add tests."}
    messages = [
        human,
        {"role": "assistant", "content": "Working on it.", "tool_calls": [
            {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST},
    ]

    idx = compressor._find_last_user_message_idx(messages, head_end=0)
    assert idx == 0, "nudge was selected as the anchor instead of the human task"
    assert messages[idx]["content"] == human["content"]


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            {
                "type": "completion",
                "session_id": "proc_build",
                "command": "scripts/run_tests.sh tests/agent/",
                "exit_code": 0,
                "output": "42 passed",
            },
            id="completion",
        ),
        pytest.param(
            {
                "type": "watch_match",
                "session_id": "proc_server",
                "command": "python server.py",
                "pattern": "Application startup complete",
                "output": "Application startup complete",
            },
            id="watch_match",
        ),
    ],
)
def test_background_process_notifications_do_not_become_compaction_anchors(
    compressor, event
):
    notification = format_process_notification(event)
    assert notification is not None
    process_turn = {"role": "user", "content": notification}
    human = {"role": "user", "content": "Refactor the auth module and add tests."}
    messages = [
        human,
        {"role": "assistant", "content": "Working on it."},
        process_turn,
    ]

    assert ContextCompressor._is_synthetic_compression_user_turn(process_turn) is True
    assert ContextCompressor._transcript_has_real_user_turn([process_turn]) is False
    assert compressor._derive_auto_focus_topic(messages) == (
        "Recent user focus:\n- Refactor the auth module and add tests."
    )
    assert compressor._find_last_user_message_idx(messages, head_end=0) == 0


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "[System: The previous response was cut off by a "
            "network error mid-stream. Continue exactly where "
            "you left off. Do not restart or repeat prior text. "
            "Finish the answer directly.]",
            id="length_continuation_network_stub",
        ),
        pytest.param(
            "[System: Your previous response was truncated by the output "
            "length limit. Continue exactly where you left off. Do not "
            "restart or repeat prior text. Finish the answer directly.]",
            id="length_continuation_output_limit",
        ),
        pytest.param(
            "[System: Your previous tool call (write_file) was too large and "
            "the stream timed out before it could be delivered. Do NOT retry "
            "the same tool call with the same large content. Instead, break the "
            "content into multiple smaller tool calls (e.g. use multiple patch "
            "calls or write smaller files). Each tool call's arguments must be "
            "under ~8K tokens to avoid stream timeouts.]",
            id="length_continuation_dropped_tools",
        ),
        pytest.param(
            "[System: Your previous response contained only internal reasoning and "
            "never produced a visible answer or tool call. Do not keep thinking. "
            "Produce your final answer as plain text now (or make the tool call "
            "you were planning).]",
            id="codex_incomplete_nudge",
        ),
        pytest.param(
            "[System: Continue now. Execute the required tool calls and only "
            "send your final answer after completing the task.]",
            id="codex_ack_continuation_nudge",
        ),
        pytest.param(
            "Your previous turn indicated a tool call but none was "
            "included. Do not narrate a plan or restate intent — issue "
            "the actual tool call now to continue the task.",
            id="dropped_toolcall_nudge",
        ),
        pytest.param(
            "You just executed tool calls but returned an "
            "empty response. Please process the tool "
            "results above and continue with the task.",
            id="empty_tool_response_nudge",
        ),
    ],
)
def test_conversation_loop_retry_nudges_are_synthetic(content):
    """These are runtime recovery nudges appended by conversation_loop's retry
    loop (length-continuation, codex incomplete/ack-continuation,
    dropped-tool-call) — same "ephemeral scaffolding, not a human turn" class
    as MAX_ITERATIONS_SUMMARY_REQUEST above. A turn interrupted/crashed mid-
    retry can persist one of these as a plain role="user" row (their
    _length_continuation_nudge/_dropped_toolcall_nudge metadata tags do not
    survive SessionDB projection), so recognition must be content-based."""
    nudge = {"role": "user", "content": content}
    assert ContextCompressor._is_synthetic_compression_user_turn(nudge) is True

    human = {"role": "user", "content": "Ship the release notes for v2."}
    assert ContextCompressor._is_synthetic_compression_user_turn(human) is False


def test_real_task_wins_over_trailing_dropped_tools_continuation_nudge(compressor):
    """The dropped-tools continuation nudge interpolates the tool name list,
    so it can only be recognized by a stable prefix (unlike the other nudges,
    which are exact-matched) — this proves that prefix path actually wires
    into anchor selection, not just the classifier in isolation."""
    human = {"role": "user", "content": "Refactor the auth module and add tests."}
    messages = [
        human,
        {"role": "assistant", "content": "Working on it.", "tool_calls": [
            {"id": "c1", "function": {"name": "write_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {
            "role": "user",
            "content": (
                "[System: Your previous tool call (write_file, patch_file) was "
                "too large and the stream timed out before it could be "
                "delivered. Do NOT retry the same tool call with the same "
                "large content. Instead, break the content into multiple "
                "smaller tool calls (e.g. use multiple patch calls or write "
                "smaller files). Each tool call's arguments must be under "
                "~8K tokens to avoid stream timeouts.]"
            ),
        },
    ]

    idx = compressor._find_last_user_message_idx(messages, head_end=0)
    assert idx == 0, "nudge was selected as the anchor instead of the human task"
    assert messages[idx]["content"] == human["content"]


def test_compress_context_todo_snapshot_stays_synthetic_across_two_boundaries(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "zero-user-todo-lifecycle"
    db.create_session(session_id, source="cron", model="test/model")

    first_agent = _lifecycle_agent(db, session_id)
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(_valid_zero_user_summary("First boundary")),
    ):
        first, _ = compress_context(
            first_agent,
            _assistant_turns(0, 24),
            "system",
            approx_tokens=90_000,
            force=True,
        )

    first_handoff = next(
        message
        for message in first
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    )
    assert first_handoff[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False
    assert "First boundary" in first_handoff["content"]
    assert any(
        message.get("role") == "user"
        and str(message.get("content") or "").startswith(TODO_INJECTION_HEADER)
        for message in first
    )
    projected = db.get_messages_as_conversation(session_id)
    assert projected
    assert all(
        COMPRESSED_SUMMARY_METADATA_KEY not in message
        and COMPRESSED_SUMMARY_HAS_USER_TURN_KEY not in message
        for message in projected
    )

    second_agent = _lifecycle_agent(db, session_id)
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(_valid_zero_user_summary("Second boundary")),
    ):
        second, _ = compress_context(
            second_agent,
            [*projected, *_assistant_turns(30, 24)],
            "system",
            approx_tokens=90_000,
            force=True,
        )

    handoff = next(
        message
        for message in second
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    )
    assert handoff[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False
    assert "Second boundary" in handoff["content"]
    assert "User asked:" not in handoff["content"]
    db.close()












