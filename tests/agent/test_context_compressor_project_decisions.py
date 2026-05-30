"""Regression coverage for project decisions during context compression."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


PROJECT_DECISIONS = (
    "Project A must never share memory with Project B.",
    "OneDrive integration deferred until core agent/data path is stable.",
    "No per-video manual approval gates.",
)


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def _discord_thread_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Discord thread session: Jenny project planning."},
        {"role": "user", "content": "Continue the Jenny / Hermes project thread."},
        {"role": "assistant", "content": "I will track project decisions here."},
        {"role": "user", "content": PROJECT_DECISIONS[0]},
        {"role": "assistant", "content": "Recorded the cross-project memory wall."},
        {"role": "user", "content": PROJECT_DECISIONS[1]},
        {"role": "assistant", "content": "Recorded the integration sequencing decision."},
        {"role": "user", "content": PROJECT_DECISIONS[2]},
        {"role": "assistant", "content": "Recorded the approval-flow decision."},
        {"role": "user", "content": "Implementation chatter that should be summarized."},
        {"role": "assistant", "content": "More implementation detail."},
        {"role": "user", "content": "Recent tail request stays live."},
        {"role": "assistant", "content": "Recent tail response stays live."},
        {"role": "user", "content": "Latest active request stays protected."},
    ]


def _decision_summary_from_prompt(prompt: str) -> str:
    present = [decision for decision in PROJECT_DECISIONS if decision in prompt]
    return "\n".join(
        [
            "## Active Task",
            "Latest active request stays protected.",
            "",
            "## Key Decisions",
            *[f"- {decision}" for decision in present],
            "",
            "## Critical Context",
            "Preserve project-level decisions from this Discord thread.",
        ]
    )


def _render(messages: list[dict[str, object]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def test_discord_thread_compression_preserves_project_decisions_in_summary():
    compressor = _compressor()
    messages = _discord_thread_messages()

    with patch("agent.context_compressor.call_llm") as mock_call:
        mock_call.side_effect = lambda **kwargs: _response(
            _decision_summary_from_prompt(kwargs["messages"][0]["content"])
        )
        compressed = compressor.compress(messages)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    rendered = _render(compressed)

    assert len(compressed) < len(messages)
    assert compressed[-1]["content"] == messages[-1]["content"]
    assert "## Key Decisions" in prompt
    assert "## Critical Context" in prompt
    for decision in PROJECT_DECISIONS:
        assert decision in prompt
        assert decision in rendered
    assert any(
        str(message.get("content", "")).startswith(SUMMARY_PREFIX)
        for message in compressed
    )


def test_repeated_discord_thread_compression_preserves_prior_project_decisions():
    compressor = _compressor()
    messages = _discord_thread_messages()

    with patch("agent.context_compressor.call_llm") as first_call:
        first_call.side_effect = lambda **kwargs: _response(
            _decision_summary_from_prompt(kwargs["messages"][0]["content"])
        )
        compressed_once = compressor.compress(messages)

    followup_messages = [
        *compressed_once,
        {"role": "assistant", "content": "Follow-up implementation detail."},
        {"role": "user", "content": "More thread traffic after first compaction."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Latest active request after second compaction."},
    ]

    with patch("agent.context_compressor.call_llm") as second_call:
        second_call.side_effect = lambda **kwargs: _response(
            _decision_summary_from_prompt(kwargs["messages"][0]["content"])
        )
        compressed_twice = compressor.compress(followup_messages)

    second_prompt = second_call.call_args.kwargs["messages"][0]["content"]
    rendered = _render(compressed_twice)

    assert len(compressed_twice) < len(followup_messages)
    assert compressed_twice[-1]["content"] == followup_messages[-1]["content"]
    assert "PREVIOUS SUMMARY:" in second_prompt
    assert "NEW TURNS TO INCORPORATE:" in second_prompt
    for decision in PROJECT_DECISIONS:
        assert decision in second_prompt
        assert decision in rendered
