"""Tests for skill-graph incremental candidate injection (delta mode).

Tests the plugin-level behavior: _build_skill_candidates_context with
session_id tracking, delta filtering, and topic detection integration.

These tests mock the embedding/LLM layers to test the injection logic
in isolation — no network calls, no graph DB needed.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import importlib.util
import sys


# Load the skill-graph plugin module dynamically
_PLUGIN_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "plugins" / "skill-graph" / "__init__.py"
)


@pytest.fixture(scope="module")
def sg():
    """Load the skill-graph plugin module."""
    spec = importlib.util.spec_from_file_location(
        "skill_graph_test",
        str(_PLUGIN_PATH),
        submodule_search_locations=[str(_PLUGIN_PATH.parent)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["skill_graph_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def reset_cache(sg):
    """Clear the injection cache between tests."""
    sg._injected_names_cache.clear()
    yield
    sg._injected_names_cache.clear()


class TestSplitIntentsSignature:
    """Verify _split_intents returns the new 3-tuple."""

    def test_returns_tuple(self, sg):
        """Even on failure, should return (list, None, None), not just list."""
        # Force failure by passing empty
        result = sg._split_intents("")
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == []
        assert result[1] is None
        assert result[2] is None


class TestBuildCandidatesContextSignature:
    """Verify the new function signature and return type."""

    def test_returns_tuple(self, sg):
        """Should return (block, intents) tuple, not just block."""
        # Use a short message that triggers cost guard
        result = sg._build_skill_candidates_context(
            "hi",
            session_id="test-sig",
            is_first_turn=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Cost guard → block is None, intents is []
        assert result[0] is None
        assert result[1] == []

    def test_short_message_returns_none_block(self, sg):
        """Messages < 12 chars should be skipped."""
        block, intents = sg._build_skill_candidates_context(
            "短",
            session_id="test-short",
            is_first_turn=True,
        )
        assert block is None
        assert intents == []

    def test_trivial_prefix_skipped(self, sg):
        """Greetings and confirmations should be skipped."""
        block, intents = sg._build_skill_candidates_context(
            "好的谢谢",
            session_id="test-trivial",
            is_first_turn=True,
        )
        assert block is None


class TestDeltaInjectionCache:
    """Test the per-session injection cache behavior."""

    def test_first_turn_populates_cache(self, sg):
        """First turn should populate the cache with injected names."""
        # Mock _embedding_search to return predictable candidates
        with patch.object(sg, '_embedding_search', return_value=[
            {"name": "skill-a", "description": "A", "score": 0.9},
            {"name": "skill-b", "description": "B", "score": 0.8},
        ]), patch.object(sg, '_split_intents', return_value=(
            ["test intent"], "coding", None
        )), patch.object(sg, '_ensure_graph', return_value=MagicMock()):
            block, _ = sg._build_skill_candidates_context(
                "Help me write a Python function to parse JSON files",
                session_id="test-cache-1",
                is_first_turn=True,
            )
            assert block is not None
            cache = sg._injected_names_cache.get("_injected:test-cache-1", set())
            assert "skill-a" in cache
            assert "skill-b" in cache

    def test_same_session_delta_only_new(self, sg):
        """Second turn on same topic should only inject new candidates."""
        with patch.object(sg, '_embedding_search', return_value=[
            {"name": "skill-a", "description": "A", "score": 0.9},
            {"name": "skill-b", "description": "B", "score": 0.8},
        ]), patch.object(sg, '_split_intents', return_value=(
            ["test intent"], "coding", True  # LLM says same topic
        )), patch.object(sg, '_ensure_graph', return_value=MagicMock()):
            # Turn 1: full injection
            block1, intents1 = sg._build_skill_candidates_context(
                "Help me write a Python function",
                session_id="test-cache-2",
                is_first_turn=True,
            )
            assert block1 is not None

            # Turn 2: same candidates → delta should be empty → None
            block2, intents2 = sg._build_skill_candidates_context(
                "Now add error handling to the function",
                session_id="test-cache-2",
                is_first_turn=False,
                prev_msg="Help me write a Python function",
                prev_intents=intents1,
            )
            # Same candidates → delta empty → no injection
            assert block2 is None

    def test_topic_shift_resets_cache(self, sg):
        """Topic shift should reset cache and inject full."""
        with patch.object(sg, '_embedding_search', side_effect=[
            # Turn 1 candidates
            [{"name": "skill-a", "description": "A", "score": 0.9}],
            # Turn 2 candidates (completely different)
            [{"name": "skill-z", "description": "Z", "score": 0.9}],
        ]), patch.object(sg, '_split_intents', return_value=(
            ["docker setup"], "devops", False  # LLM says shift
        )), patch.object(sg, '_ensure_graph', return_value=MagicMock()):
            # Turn 1
            block1, intents1 = sg._build_skill_candidates_context(
                "Set up a docker container",
                session_id="test-cache-3",
                is_first_turn=True,
            )
            cache1 = sg._injected_names_cache.get("_injected:test-cache-3", set())
            assert cache1 == {"skill-a"}

            # Turn 2: different topic, LLM says shift
            block2, intents2 = sg._build_skill_candidates_context(
                "Write a novel about space travel",
                session_id="test-cache-3",
                is_first_turn=False,
                prev_msg="Set up a docker container",
                prev_intents=intents1,
            )
            assert block2 is not None
            assert "skill-z" in block2
            # Cache should be reset (not cumulative)
            cache2 = sg._injected_names_cache.get("_injected:test-cache-3", set())
            assert "skill-a" not in cache2
            assert "skill-z" in cache2


class TestInjectionBlockFormat:
    """Verify the injected block format is correct."""

    def test_block_contains_candidate_list(self, sg):
        with patch.object(sg, '_embedding_search', return_value=[
            {"name": "test-skill", "description": "A test skill", "score": 0.85},
        ]), patch.object(sg, '_split_intents', return_value=(
            ["test"], "coding", None
        )), patch.object(sg, '_ensure_graph', return_value=MagicMock()):
            block, _ = sg._build_skill_candidates_context(
                "Help me with a coding task that involves testing",
                session_id="test-format",
                is_first_turn=True,
            )
            assert block is not None
            assert "Relevant skills you may want to load" in block
            assert "test-skill" in block
            assert "skill_load" in block

    def test_block_caps_at_10_candidates(self, sg):
        """Full injection should cap at 10 candidates."""
        fake_candidates = [
            {"name": f"skill-{i}", "description": f"Skill {i}", "score": 0.9 - i * 0.01}
            for i in range(15)
        ]
        with patch.object(sg, '_embedding_search', return_value=fake_candidates), \
             patch.object(sg, '_split_intents', return_value=(
                 ["test"], "coding", None
             )), patch.object(sg, '_ensure_graph', return_value=MagicMock()):
            block, _ = sg._build_skill_candidates_context(
                "Help with many different things at once",
                session_id="test-cap",
                is_first_turn=True,
            )
            lines = [l for l in block.split('\n') if l.startswith('- ')]
            assert len(lines) <= 10
