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
    SUMMARY_PREFIX,
    ContextCompressor,
    _NO_USER_TASK_SENTINEL,
)
from agent.conversation_compression import (
    _ensure_compressed_has_user_turn,
    compress_context,
)
from hermes_state import SessionDB
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


def test_zero_user_prompt_anchors_source_language_and_exact_sentinel(compressor):
    captured_prompt = ""

    def fake_call_llm(**kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs["messages"][0]["content"]
        return _response(_valid_zero_user_summary())

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        result = compressor._generate_summary(_assistant_tool_turns(0, 2))

    assert result == f"{SUMMARY_PREFIX}\n{_valid_zero_user_summary().strip()}"
    assert "dominant language of the source turns" in captured_prompt
    assert _NO_USER_TASK_SENTINEL in captured_prompt
    assert "Do not write \"User asked:\"" in captured_prompt


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


def test_continuation_user_marker_is_not_reused_as_real_provenance():
    todo_snapshot = f"{TODO_INJECTION_HEADER}\n- [ ] inspect. Inspect artifacts (pending)"
    compressed = [{"role": "assistant", "content": "Scheduled work completed."}]

    _ensure_compressed_has_user_turn(
        [{"role": "user", "content": todo_snapshot}],
        compressed,
    )

    assert compressed[-1] == {
        "role": "user",
        "content": COMPRESSION_CONTINUATION_USER_CONTENT,
    }
    projected = [{"role": row["role"], "content": row["content"]} for row in compressed]
    assert ContextCompressor._transcript_has_real_user_turn(projected) is False


def test_continuation_markers_are_not_human_anchors():
    from agent.conversation_compression import _is_real_user_message

    legacy = (
        "Continue from the compressed conversation context above. "
        "This marker exists because the compacted transcript contained "
        "no preserved user turn."
    )
    assert not _is_real_user_message(
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT}
    )
    assert not _is_real_user_message({"role": "user", "content": legacy})


def test_static_fallback_does_not_attribute_synthetic_rows_to_user(compressor):
    todo_snapshot = f"{TODO_INJECTION_HEADER}\n- [ ] inspect. Inspect artifacts (pending)"
    fallback = compressor._build_static_fallback_summary(
        [
            {"role": "user", "content": todo_snapshot},
            {
                "role": "user",
                "content": COMPRESSION_CONTINUATION_USER_CONTENT,
            },
            *_assistant_tool_turns(0, 2),
        ]
    )

    assert _NO_USER_TASK_SENTINEL in fallback
    assert "User asked:" not in fallback
    assert "INTERNAL CONTEXT:" in fallback


def test_zero_user_deterministic_fallback_uses_same_provenance(compressor):
    messages = _assistant_tool_turns(0, 12)

    with patch.object(compressor, "_generate_summary", return_value=None):
        result = compressor.compress(messages, current_tokens=90_000)

    handoff = next(
        message
        for message in result
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    )
    assert _NO_USER_TASK_SENTINEL in handoff["content"]
    assert "User asked:" not in handoff["content"]
    assert handoff[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False


def test_real_user_turn_sets_provenance_true(compressor):
    messages = [
        {"role": "user", "content": "Please inspect the build artifacts."},
        *_assistant_tool_turns(0, 12),
    ]
    summary = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nUser asked: 'Please inspect the build artifacts.'"

    with patch.object(compressor, "_generate_summary", return_value=summary):
        result = compressor.compress(messages, current_tokens=90_000)

    handoff = next(
        message
        for message in result
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    )
    assert handoff[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is True


def test_session_boundaries_clear_summary_provenance(compressor):
    compressor._summary_has_user_turn = False
    compressor.on_session_reset()
    assert compressor._summary_has_user_turn is None

    compressor._summary_has_user_turn = True
    compressor.on_session_end("cron-session", [])
    assert compressor._summary_has_user_turn is None
