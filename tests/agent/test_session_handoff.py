"""Tests for automatic context-refresh handoff generation."""

from types import SimpleNamespace

from agent.session_handoff import (
    build_context_refresh_resume_note,
    maybe_prepare_context_refresh_handoff,
    should_auto_new_after_context_refresh,
)


class FakeSessionDB:
    def get_session_title(self, session_id):
        return "Hermes Adaptive Titles"


def make_agent(tmp_path, compression_count=2, mode="prepare_only"):
    return SimpleNamespace(
        session_id="sess-123",
        _session_db=FakeSessionDB(),
        context_compressor=SimpleNamespace(compression_count=compression_count),
        context_refresh_config={
            "enabled": True,
            "handoff_after_compressions": 2,
            "mode": mode,
            "auto_new_policy": "phase_boundary",
            "write_session_handoff": True,
            "include_sha256": True,
            "max_handoff_lines": 250,
            "handoff_base_dir": str(tmp_path),
            "require_no_running_processes": False,
        },
        warnings=[],
        _emit_warning=lambda msg: None,
    )


def test_prepares_handoff_at_configured_compression_threshold(tmp_path):
    agent = make_agent(tmp_path, compression_count=2)
    messages = [
        {"role": "user", "content": "Implement adaptive session titles"},
        {"role": "assistant", "content": "Implemented Phase 1"},
    ]

    result = maybe_prepare_context_refresh_handoff(agent, messages, reason="compression_count>=2")

    assert result is not None
    assert result.session_id == "sess-123"
    assert result.path.exists()
    assert result.path.name == "AFTER_SESSION_COMPRESSION_HANDOFF.md"
    assert result.sha256
    assert result.line_count > 0
    assert "Reference session sess-123" in result.resume_prompt

    content = result.path.read_text(encoding="utf-8")
    assert "# Automatic Context Refresh Handoff" in content
    assert "Session ID: sess-123" in content
    assert "Reason: compression_count>=2" in content
    assert "Hermes Adaptive Titles" in content
    assert "Next Valid Actions" in content
    assert "unverified; verify before action" in content

    assert agent._pending_context_refresh["handoff_path"] == str(result.path)
    assert agent._context_refresh_handoff_prepared_for_count == 2


def test_skips_before_compression_threshold(tmp_path):
    agent = make_agent(tmp_path, compression_count=1)

    result = maybe_prepare_context_refresh_handoff(agent, [], reason="compression_count>=2")

    assert result is None
    assert not hasattr(agent, "_pending_context_refresh")


def test_does_not_spam_duplicate_handoff_for_same_compression_count(tmp_path):
    agent = make_agent(tmp_path, compression_count=2)

    first = maybe_prepare_context_refresh_handoff(agent, [], reason="compression_count>=2")
    second = maybe_prepare_context_refresh_handoff(agent, [], reason="compression_count>=2")

    assert first is not None
    assert second is None


def test_auto_new_decision_waits_for_completed_phase_boundary(tmp_path):
    agent = make_agent(tmp_path, compression_count=2, mode="auto_new")
    maybe_prepare_context_refresh_handoff(agent, [], reason="compression_count>=2")

    interrupted = should_auto_new_after_context_refresh(
        agent,
        {"final_response": "stopping", "completed": False, "interrupted": True},
    )
    ordinary_completion = should_auto_new_after_context_refresh(
        agent,
        {"final_response": "done", "completed": True},
    )
    phase_complete = should_auto_new_after_context_refresh(
        agent,
        {
            "final_response": "Phase 2 is complete. Next phase will begin with validation.",
            "completed": True,
        },
    )

    assert interrupted.should_auto_new is False
    assert interrupted.reason == "turn_not_completed"
    assert ordinary_completion.should_auto_new is False
    assert ordinary_completion.reason == "phase_boundary_not_detected"
    assert phase_complete.should_auto_new is True
    assert phase_complete.reason == "phase_boundary"


def test_prepare_only_mode_does_not_auto_new(tmp_path):
    agent = make_agent(tmp_path, compression_count=2, mode="prepare_only")
    maybe_prepare_context_refresh_handoff(agent, [], reason="compression_count>=2")

    decision = should_auto_new_after_context_refresh(
        agent,
        {"final_response": "done", "completed": True},
    )

    assert decision.should_auto_new is False
    assert decision.reason == "mode_is_prepare_only"


def test_build_resume_note_for_new_session(tmp_path):
    agent = make_agent(tmp_path, compression_count=2, mode="auto_new")
    handoff = maybe_prepare_context_refresh_handoff(agent, [], reason="compression_count>=2")

    note = build_context_refresh_resume_note(agent._pending_context_refresh)

    assert handoff is not None
    assert "Automatic context refresh" in note
    assert "sess-123" in note
    assert str(handoff.path) in note
