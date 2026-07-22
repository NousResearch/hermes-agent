"""Session-scoped input contracts for context compaction.

Compression may preserve task continuity for the current session and its visible
compaction descendants, but internal retry/verifier scaffolding is not
conversation state and must never reach the summarizer.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, EPHEMERAL_SCAFFOLDING_FLAGS


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )


def _response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_summary_input_excludes_internal_verification_scaffolding():
    """Transient verifier nudges cannot become a task handoff summary."""
    compressor = _compressor()
    turns = [
        {"role": "user", "content": "Implement session-scoped compaction."},
        {"role": "assistant", "content": "I will add the regression test."},
        {
            "role": "assistant",
            "content": "VERIFIER_SCAFFOLDING_MUST_NOT_SURVIVE",
            "_verification_stop_synthetic": True,
        },
        {
            "role": "user",
            "content": "Run the focused test suite.",
            "_pre_verify_synthetic": True,
        },
    ]

    with patch(
        "agent.context_compressor.call_llm", return_value=_response("## Goal\nKeep task state.")
    ) as call_llm:
        compressor._generate_summary(turns)

    prompt = call_llm.call_args.kwargs["messages"][0]["content"]
    assert "Implement session-scoped compaction." in prompt
    assert "I will add the regression test." in prompt
    assert "VERIFIER_SCAFFOLDING_MUST_NOT_SURVIVE" not in prompt
    assert "Run the focused test suite." not in prompt


def test_fallback_excludes_explicit_global_announcement():
    """Framework-wide notices can opt out without text-pattern heuristics."""
    compressor = _compressor()
    fallback = compressor._build_static_fallback_summary(
        [
            {"role": "user", "content": "Keep the current repository test command."},
            {
                "role": "user",
                "content": "GLOBAL_ANNOUNCEMENT_MUST_NOT_SURVIVE",
                "_exclude_from_context_summary": True,
            },
        ]
    )

    assert "Keep the current repository test command." in fallback
    assert "GLOBAL_ANNOUNCEMENT_MUST_NOT_SURVIVE" not in fallback


def test_summary_prompt_explicitly_limits_continuity_to_session_lineage():
    """The LLM contract rejects stale and Hermes-wide instructions as state."""
    compressor = _compressor()
    with patch(
        "agent.context_compressor.call_llm", return_value=_response("## Goal\nKeep task state.")
    ) as call_llm:
        compressor._generate_summary([{"role": "user", "content": "Current session task."}])

    prompt = call_llm.call_args.kwargs["messages"][0]["content"]
    assert "this session and its visible compaction descendants only" in prompt
    assert "general Hermes-wide status announcements" in prompt
    assert "stale detached-session instructions" in prompt


def test_persistence_and_compression_share_scaffolding_flags():
    """A new ephemeral flag cannot be filtered by only one lifecycle path."""
    from run_agent import _EPHEMERAL_SCAFFOLDING_FLAGS

    assert _EPHEMERAL_SCAFFOLDING_FLAGS is EPHEMERAL_SCAFFOLDING_FLAGS
