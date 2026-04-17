"""Tests for model_handoff module.

Verifies:
1. Handoff is generated on model transitions
2. Handoff is skipped when model stays the same
3. Handoff structure has correct sections
4. Tier-aware handoff generation
5. Recent messages included verbatim
6. Accomplishment and remaining extraction
7. File persistence and loading
"""

import os
import tempfile
from unittest import mock

import pytest


@pytest.fixture
def sample_conversation():
    return [
        {"role": "user", "content": "create a new file called test.py"},
        {"role": "assistant", "content": "Done. Created test.py with the requested content."},
        {"role": "user", "content": "now add a test function"},
        {"role": "assistant", "content": "Added test_function to test.py. Still need to add edge cases."},
        {"role": "user", "content": "debug the failing test"},
    ]


@pytest.fixture
def task_state_active():
    return {
        "active_task": True,
        "last_tier": 3,
        "last_model": "kimi-k2.5",
        "last_category": "code",
        "turns_in_task": 5,
        "easy_streak": 0,
    }


class TestShouldGenerateHandoff:
    """Test conditions for when a handoff should be generated."""

    def test_different_models_generate_handoff(self):
        from agent.model_handoff import should_generate_handoff
        assert should_generate_handoff("glm-5.1", "kimi-k2.5") is True

    def test_same_model_no_handoff(self):
        from agent.model_handoff import should_generate_handoff
        assert should_generate_handoff("glm-5.1", "glm-5.1") is False

    def test_empty_model_no_handoff(self):
        from agent.model_handoff import should_generate_handoff
        assert should_generate_handoff("", "glm-5.1") is False
        assert should_generate_handoff("glm-5.1", "") is False

    def test_no_active_task_descending_tier(self):
        from agent.model_handoff import should_generate_handoff
        # Descending without active task: should generate (it's still a change)
        state = {"active_task": False, "last_tier": 3}
        result = should_generate_handoff("kimi-k2.5", "glm-5.1", state)
        # Descending returns True because the models differ and it's a tier change
        assert result is True

    def test_escalation_always_handoff(self):
        from agent.model_handoff import should_generate_handoff
        state = {"active_task": True, "last_tier": 1}
        assert should_generate_handoff("glm-5.1", "kimi-k2.5", state) is True


class TestBuildHandoff:
    """Test handoff summary generation."""

    def test_basic_handoff_structure(self, sample_conversation, task_state_active):
        from agent.model_handoff import build_handoff
        handoff = build_handoff(
            "glm-5.1", "kimi-k2.5",
            sample_conversation, task_state_active,
        )
        assert handoff
        assert "glm-5.1" in handoff
        assert "kimi-k2.5" in handoff
        assert "Previous model" in handoff
        assert "Active Task" in handoff

    def test_same_model_empty_handoff(self, sample_conversation):
        from agent.model_handoff import build_handoff
        handoff = build_handoff("glm-5.1", "glm-5.1", sample_conversation)
        assert handoff == ""

    def test_handoff_includes_recent_messages(self, sample_conversation, task_state_active):
        from agent.model_handoff import build_handoff
        handoff = build_handoff(
            "glm-5.1", "kimi-k2.5",
            sample_conversation, task_state_active,
            recent_only=2,
        )
        assert "Recent Messages" in handoff

    def test_handoff_has_accomplishments(self, task_state_active):
        from agent.model_handoff import build_handoff
        conv = [
            {"role": "assistant", "content": "Successfully created the file and verified it."},
            {"role": "user", "content": "continue"},
        ]
        handoff = build_handoff("glm-5.1", "kimi-k2.5", conv, task_state_active)
        assert "What Was Done" in handoff

    def test_handoff_respects_max_chars(self, sample_conversation, task_state_active):
        from agent.model_handoff import build_handoff
        # Very long conversation should be truncated
        long_conv = [{"role": "user", "content": "x" * 300}] * 50
        handoff = build_handoff(
            "glm-5.1", "kimi-k2.5",
            long_conv, task_state_active,
            max_chars=500,
        )
        assert len(handoff) <= 503  # max_chars + "..."

    def test_handoff_no_active_task(self):
        from agent.model_handoff import build_handoff
        state = {"active_task": False, "last_tier": 0, "last_category": ""}
        handoff = build_handoff("glm-5.1", "kimi-k2.5", [], state)
        assert "No active task" in handoff

    def test_handoff_with_focus_topic(self, task_state_active):
        from agent.model_handoff import build_handoff
        handoff = build_handoff(
            "glm-5.1", "kimi-k2.5",
            [{"role": "user", "content": "fix the bug"}],
            task_state_active,
            focus_topic="Bug in authentication module",
        )
        assert "Focus" in handoff
        assert "Bug in authentication module" in handoff


class TestHandoffPersistence:
    """Test handoff file save/load."""

    def test_save_and_load_handoff(self):
        from agent.model_handoff import save_handoff, load_recent_handoffs, _HANDOFF_DIR

        with mock.patch("agent.model_handoff._HANDOFF_DIR", tempfile.mkdtemp()):
            handoff = "# Test Handoff\nThis is a test."
            path = save_handoff(handoff, "glm-5.1", "kimi-k2.5")
            assert path  # Should return a valid path
            assert os.path.exists(path)

            # Load it back
            recent = load_recent_handoffs(limit=1)
            assert len(recent) == 1
            assert "Test Handoff" in recent[0]["text"]

    def test_load_empty_directory(self):
        from agent.model_handoff import load_recent_handoffs

        with mock.patch("agent.model_handoff._HANDOFF_DIR", tempfile.mkdtemp()):
            recent = load_recent_handoffs()
            assert recent == []


class TestTierMapping:
    """Test model-to-tier mapping."""

    def test_known_model_tier(self):
        from agent.model_handoff import _tier_of
        # These should match DEFAULT_TIERS in routing_v2
        assert _tier_of("glm-5.1") == 1
        assert _tier_of("qwen3.5:397b") == 5

    def test_unknown_model_fallback(self):
        from agent.model_handoff import _tier_of
        tier = _tier_of("totally-unknown-model")
        # Should return a reasonable fallback
        assert 1 <= tier <= 5

    def test_mini_model_low_tier(self):
        from agent.model_handoff import _tier_of
        assert _tier_of("gpt-5-mini") <= 3