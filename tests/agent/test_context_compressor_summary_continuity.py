"""Regression tests for iterative context-summary continuity."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    ContextCompressor,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
)


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _messages_with_handoff(summary_body: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def test_existing_previous_summary_is_not_serialized_again_as_new_turn():
    """Same-process iterative compression should not feed the old handoff twice."""
    compressor = _compressor()
    old_summary = "OLD-SUMMARY-BODY unique continuity facts"
    compressor._previous_summary = old_summary

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_resume_rehydrates_previous_summary_from_handoff_message():
    """After restart/resume, the persisted handoff should regain summary identity."""
    compressor = _compressor()
    old_summary = "RESUMED-SUMMARY-BODY durable continuity facts"
    assert compressor._previous_summary is None

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert "TURNS TO SUMMARIZE:" not in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_handoff_in_protected_head_populates_previous_summary_before_update():
    """A resumed protected-head handoff should restore iterative-summary state."""
    compressor = _compressor()
    old_summary = "PROTECTED-HEAD-SUMMARY durable facts from before restart"
    seen_turns = []

    def fake_generate_summary(turns_to_summarize, focus_topic=None):
        seen_turns.extend(turns_to_summarize)
        return "new summary from resumed turns"

    with patch.object(compressor, "_generate_summary", side_effect=fake_generate_summary):
        compressor.compress(_messages_with_handoff(old_summary))

    assert compressor._previous_summary == old_summary
    assert seen_turns
    assert all(old_summary not in str(msg.get("content", "")) for msg in seen_turns)


def test_summary_prompt_keeps_task_snapshot_reference_only():
    """The summarizer prompt must not write live-resume directives into summaries.

    SUMMARY_PREFIX says historical task sections are reference-only unless the
    latest post-summary user message asks to continue.  The summary body must
    not contradict that by telling the next model to "pick up exactly here".
    """
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", return_value=_response("summary")) as mock_call:
        compressor._generate_summary([
            {"role": "user", "content": "older task"},
            {"role": "assistant", "content": "working"},
        ])

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert HISTORICAL_TASK_HEADING in prompt
    assert "Continuation should pick up exactly here" not in prompt
    assert "Update \"## Active Task\"" not in prompt
    assert "latest user message" in prompt
    assert "historical context only" in prompt


def test_iterative_summary_prompt_does_not_revive_active_task_heading():
    """Iterative update wording must use the historical snapshot heading too."""
    compressor = _compressor()
    compressor._previous_summary = "old summary"

    with patch("agent.context_compressor.call_llm", return_value=_response("summary")) as mock_call:
        compressor._generate_summary([
            {"role": "user", "content": "new compacted turn"},
        ])

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "Update \"## Active Task\"" not in prompt
    assert "historical task snapshot" in prompt
    assert "reference-only" in prompt


def test_incident_shape_preserves_newer_thread_work_as_new_turns():
    """Older stale summaries must not swallow newer thread-local audit work.

    Regression shape from the Discord workflow incident: a protected-head
    handoff points at an older Part 86 recovery lane, but later turns in the
    same thread contain the broad workflow/VPS/gateway/MCP audit and a user
    correction that the assistant was bypassing workflow.  Re-compression must
    treat the old handoff as PREVIOUS SUMMARY and feed the newer turns through
    NEW TURNS TO INCORPORATE instead of serializing the handoff as a fresh user
    instruction or losing the newer correction/task list.
    """
    compressor = _compressor()
    stale_part86_summary = (
        f"{HISTORICAL_TASK_HEADING}\n"
        "User asked: 'Continue Part 86 /new smoke recovery'\n\n"
        "## Historical Remaining Work\n"
        "Run live /new smoke for Part 86 after background proof."
    )
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{stale_part86_summary}"},
        {"role": "assistant", "content": "Part 86 recovery handoff acknowledged"},
        {
            "role": "user",
            "content": (
                "Review the workflow and all of its pieces. Build the active "
                "audit task list: map_gateway, map_workflow_mcp, map_vps, "
                "map_all_aspects."
            ),
        },
        {"role": "assistant", "content": "Started mapping gateway and workflow surfaces."},
        {
            "role": "user",
            "content": (
                "Yeah, the workflow definitely needs a lot of work. Im watching "
                "you bypass almost every single part of it as we speak."
            ),
        },
        {"role": "assistant", "content": "I will correct the workflow and use Kanban gates."},
        {
            "role": "user",
            "content": "Review the end of the previous session from this thread and continue properly.",
        },
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(messages, current_tokens=90000)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert prompt.count(stale_part86_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt
    assert "map_gateway, map_workflow_mcp, map_vps, map_all_aspects" in prompt
    assert "bypass almost every single part" in prompt
    assert "Review the end of the previous session" in prompt
