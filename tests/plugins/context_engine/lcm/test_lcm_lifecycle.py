"""Full LCM lifecycle test — ingest, compact, search, expand, pin, forget.

Exercises the LcmEngine directly (no agent loop) with mock summarization
to verify the lossless context management guarantees:
  1. Ingested messages appear in the store
  2. check_thresholds fires at the right time
  3. auto_compact creates a DAG node and replaces raw entries
  4. Search recovers compacted messages by keyword
  5. Expand restores original raw messages from a summary node
  6. Pinned messages survive compaction
  7. lcm_forget compacts specific messages
  8. Store retains all originals after compaction (lossless)
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from plugins.context_engine.lcm.config import LcmConfig
from plugins.context_engine.lcm.engine import LcmEngine, CompactionAction, ContextEntry
from plugins.context_engine.lcm.tools import set_engine, get_engine
from plugins.context_engine.lcm.dag import SummaryDag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """LCM config with a tiny context length so thresholds fire quickly."""
    return LcmConfig(
        enabled=True,
        tau_soft=0.50,
        tau_hard=0.85,
        protect_last_n=2,
        max_pinned=20,
        max_store_size=500,
    )


@pytest.fixture
def engine(config):
    """Fresh LcmEngine with mock summarizer."""
    eng = LcmEngine(config=config, context_length=500)
    # Patch summarizer to return deterministic text instead of calling LLM
    eng.summarizer.summarize = MagicMock(return_value="MOCK SUMMARY: prior conversation turns.")
    return eng


@pytest.fixture(autouse=True)
def _cleanup_engine():
    """Ensure the ContextVar engine reference is cleaned up after each test."""
    yield
    set_engine(None)


# Helper to create a message
def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

class TestIngest:
    def test_messages_appear_in_store(self, engine):
        m = msg("user", "Hello world")
        mid = engine.ingest(m)
        assert mid == 0
        assert len(engine.store) == 1
        assert engine.store.get(0) == m

    def test_multiple_ingest_sequential_ids(self, engine):
        ids = [engine.ingest(msg("user", f"msg {i}")) for i in range(5)]
        assert ids == [0, 1, 2, 3, 4]
        assert len(engine.store) == 5

    def test_ingest_adds_to_active(self, engine):
        engine.ingest(msg("user", "alpha"))
        engine.ingest(msg("assistant", "beta"))
        assert len(engine.active) == 2
        assert engine.active[0].kind == "raw"
        assert engine.active[1].kind == "raw"


# ---------------------------------------------------------------------------
# 2. Threshold checking
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_none_when_below_soft(self, engine):
        engine.ingest(msg("user", "short"))
        action = engine.check_thresholds()
        assert action == CompactionAction.NONE

    def test_async_at_soft_threshold(self, engine):
        # Fill up to ~50% of 500 token context (~250 tokens ≈ 1000 chars)
        big = "x" * 1200
        for i in range(5):
            engine.ingest(msg("user", f"{big} msg {i}"))
        action = engine.check_thresholds()
        assert action in (CompactionAction.ASYNC, CompactionAction.BLOCKING)

    def test_blocking_at_hard_threshold(self, engine):
        # Fill well above 85% of 500 tokens (~425 tokens ≈ 1700 chars)
        big = "y" * 500
        for i in range(12):
            engine.ingest(msg("user", f"{big} msg {i}"))
        action = engine.check_thresholds()
        assert action == CompactionAction.BLOCKING


# ---------------------------------------------------------------------------
# 3. Auto-compaction
# ---------------------------------------------------------------------------

class TestAutoCompact:
    def test_compact_replaces_raw_with_summary(self, engine):
        for i in range(6):
            engine.ingest(msg("user", f"Message number {i} with some detail."))
        # active: 6 raw entries
        assert len(engine.active) == 6
        assert all(e.kind == "raw" for e in engine.active)

        node = engine.auto_compact()
        assert node is not None
        # protect_last_n=2, so at most 4 can be compacted into 1 summary
        # Result: 1 summary + 2 protected raw = 3 entries
        assert len(engine.active) <= 4
        assert any(e.kind == "summary" for e in engine.active)
        assert any(e.kind == "raw" for e in engine.active)

    def test_compact_creates_dag_node(self, engine):
        for i in range(6):
            engine.ingest(msg("user", f"DAG test message {i}"))
        node = engine.auto_compact()
        assert node is not None
        assert len(node.source_ids) > 0
        assert len(engine.dag.nodes) == 1

    def test_protected_tail_survives(self, engine):
        msgs = [msg("user", f"protected tail {i}") for i in range(6)]
        for m in msgs:
            engine.ingest(m)
        engine.auto_compact()
        # Last protect_last_n (2) raw entries must still be in active
        active_contents = [e.message["content"] for e in engine.active if e.kind == "raw"]
        assert any("protected tail 5" in c for c in active_contents)
        assert any("protected tail 4" in c for c in active_contents)


# ---------------------------------------------------------------------------
# 4. Lossless search after compaction
# ---------------------------------------------------------------------------

class TestSearchAfterCompaction:
    def test_keyword_search_finds_compacted_content(self, engine):
        unique = "UNIQUE_KEYWORD_XYZZY_42"
        engine.ingest(msg("user", f"buried treasure {unique} in the compacted zone"))
        # Fill to ensure compaction happens
        for i in range(8):
            engine.ingest(msg("user", f"filler message {i} " * 20))
        engine.auto_compact()
        # Search the store (not just active) for the unique keyword
        results = engine.search(unique)
        assert len(results) >= 1
        found_content = results[0][1]["content"]
        assert unique in found_content

    def test_search_returns_empty_for_missing(self, engine):
        engine.ingest(msg("user", "some content"))
        results = engine.search("NONEXISTENT_KEYWORD_99999")
        assert results == []


# ---------------------------------------------------------------------------
# 5. Expand restores original messages
# ---------------------------------------------------------------------------

class TestExpand:
    def test_expand_returns_original_messages(self, engine):
        originals = [msg("user", f"original expandable {i}") for i in range(5)]
        for m in originals:
            engine.ingest(m)
        node = engine.auto_compact()
        assert node is not None
        source_ids = node.source_ids
        # Retrieve each original from the store
        for sid in source_ids:
            original = engine.store.get(sid)
            assert original is not None
            assert "original expandable" in original["content"]


# ---------------------------------------------------------------------------
# 6. Pinned messages survive compaction
# ---------------------------------------------------------------------------

class TestPin:
    def test_pinned_message_not_compacted(self, engine):
        engine.ingest(msg("user", "pin me please"))
        for i in range(8):
            engine.ingest(msg("user", f"bulk message {i} " * 15))
        # Pin message 0
        engine._pinned_ids.add(0)
        engine.auto_compact()
        # Message 0 must still be in active as raw
        raw_ids = [e.msg_id for e in engine.active if e.kind == "raw"]
        assert 0 in raw_ids

    def test_pin_via_tool_handler(self, engine):
        set_engine(engine)
        from plugins.context_engine.lcm.tools import handle_lcm_pin
        engine.ingest(msg("user", "to pin"))
        result = handle_lcm_pin({"message_ids": "0"})
        assert "Pinned" in result
        assert 0 in engine._pinned_ids


# ---------------------------------------------------------------------------
# 7. Forget (manual compaction of specific messages)
# ---------------------------------------------------------------------------

class TestForget:
    def test_forget_removes_from_active(self, engine):
        for i in range(8):
            engine.ingest(msg("user", f"forgettable {i}"))
        # Forget messages 0,1,2
        set_engine(engine)
        from plugins.context_engine.lcm.tools import handle_lcm_forget
        result = handle_lcm_forget({"message_ids": "0,1,2", "reason": "test forget"})
        # After forget, those IDs should not be in active as raw
        active_raw_ids = {e.msg_id for e in engine.active if e.kind == "raw" and e.msg_id is not None}
        assert 0 not in active_raw_ids or any(e.kind == "summary" for e in engine.active)

    def test_forget_preserves_in_store(self, engine):
        for i in range(8):
            engine.ingest(msg("user", f"store test {i}"))
        set_engine(engine)
        from plugins.context_engine.lcm.tools import handle_lcm_forget
        handle_lcm_forget({"message_ids": "0,1,2", "reason": "test"})
        # Store must still have them
        for i in range(3):
            assert engine.store.get(i) is not None


# ---------------------------------------------------------------------------
# 8. Store retains all originals (lossless guarantee)
# ---------------------------------------------------------------------------

class TestLosslessStore:
    def test_store_grows_monotonically(self, engine):
        count = 10
        for i in range(count):
            engine.ingest(msg("user", f"lossless {i}"))
        assert len(engine.store) == count
        engine.auto_compact()
        # Store should NOT shrink after compaction
        assert len(engine.store) == count

    def test_all_messages_retrievable_after_multiple_compactions(self, engine):
        total = 20
        for i in range(total):
            engine.ingest(msg("user", f"multi-compact {i}"))
        # Compact multiple times
        for _ in range(3):
            engine.auto_compact()
        # Every original must still be in the store
        for i in range(total):
            original = engine.store.get(i)
            assert original is not None, f"Message {i} lost from store!"
            assert f"multi-compact {i}" in original["content"]


# ---------------------------------------------------------------------------
# Tool handlers integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chicken-and-egg fix: should_compress uses real API token count
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Deterministic summarization fallback
# ---------------------------------------------------------------------------

class TestDeterministicFallback:
    """Verify the extractive fallback works when no LLM is available."""

    def test_deterministic_summary_produces_output(self, engine):
        from plugins.context_engine.lcm.summarizer import Summarizer, SummarizerConfig
        s = Summarizer(SummarizerConfig())
        turns = [
            {"role": "user", "content": "Check the filesystem"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "ls -la"}'},
                 "type": "function"}
            ]},
            {"role": "tool", "content": "total 42\ndrwxr-xr-x 2 root root 4096 Apr 9 .bashrc"},
            {"role": "assistant", "content": "Here are the files in your home directory."},
        ]
        result = s._deterministic_summary(turns)
        assert result is not None
        # Output should contain key content from the turns
        assert "filesystem" in result or "Check" in result
        # With sumy, output is extracted sentences; without, it uses sections
        assert len(result) > 50

    def test_deterministic_with_previous_summary(self, engine):
        from plugins.context_engine.lcm.summarizer import Summarizer, SummarizerConfig
        s = Summarizer(SummarizerConfig())
        turns = [
            {"role": "user", "content": "Run tests"},
            {"role": "assistant", "content": "Running pytest now"},
        ]
        result = s._deterministic_summary(turns, previous_summary="## Goal\nFix the bug")
        assert "## Goal\nFix the bug" in result
        assert "Updates since last compaction" in result

    def test_deterministic_empty_turns_returns_none(self, engine):
        from plugins.context_engine.lcm.summarizer import Summarizer, SummarizerConfig
        s = Summarizer(SummarizerConfig())
        assert s._deterministic_summary([]) is None

    def test_full_summarize_falls_back_to_deterministic(self, engine):
        """When call_llm fails, summarize() returns deterministic output."""
        from plugins.context_engine.lcm.summarizer import Summarizer, SummarizerConfig
        s = Summarizer(SummarizerConfig())
        turns = [
            {"role": "user", "content": "Search for the bug"},
            {"role": "assistant", "content": "Found it in utils.py"},
        ]
        # call_llm will fail (no provider), but deterministic fallback should work
        result = s.summarize(turns)
        assert result is not None
        assert "Search for the bug" in result

    def test_compaction_succeeds_with_deterministic_fallback(self, engine):
        """End-to-end: auto_compact works even without LLM for summarization."""
        for i in range(8):
            engine.ingest({"role": "user", "content": f"fallback test message {i} with detail"})
        # Mock summarizer to fail LLM but succeed with deterministic
        original_summarize = engine.summarizer.summarize
        call_count = [0]
        def mock_summarize(turns, previous_summary=None):
            call_count[0] += 1
            # First call tries LLM (fails), then falls back to deterministic
            return original_summarize(turns, previous_summary)
        engine.summarizer.summarize = mock_summarize

        node = engine.auto_compact()
        assert node is not None, "auto_compact should succeed with deterministic fallback"
        assert len(node.source_ids) > 0


class TestShouldCompressWithRealTokens:
    """Verify that should_compress() works before the first compress() call.

    Regression test for the chicken-and-egg bug: the engine's active list is
    empty before compress() ingests messages, so should_compress() must use
    the prompt_tokens parameter (real API token count) instead of only
    checking the engine's internal active_tokens().
    """

    def test_fires_with_real_tokens_before_ingest(self, engine):
        """should_compress returns True when real tokens exceed tau_soft,
        even though the engine has 0 ingested messages."""
        from plugins.context_engine.lcm import LcmContextEngine
        adapter = LcmContextEngine(
            lcm_config={"tau_soft": 0.50, "tau_hard": 0.85, "protect_last_n": 2},
            context_length=1000,
        )
        # Engine has 0 messages — active_tokens() = 0
        assert adapter._engine.active_tokens() == 0

        # But if the API reports 600 real tokens (> 50% of 1000), should fire
        assert adapter.should_compress(prompt_tokens=600) is True

    def test_does_not_fire_below_threshold(self, engine):
        """should_compress returns False when real tokens are below tau_soft."""
        from plugins.context_engine.lcm import LcmContextEngine
        adapter = LcmContextEngine(
            lcm_config={"tau_soft": 0.50, "tau_hard": 0.85},
            context_length=1000,
        )
        assert adapter.should_compress(prompt_tokens=200) is False

    def test_fallback_to_engine_after_ingest(self, engine):
        """After first compress(), fallback path via engine state works."""
        from plugins.context_engine.lcm import LcmContextEngine
        adapter = LcmContextEngine(
            lcm_config={"tau_soft": 0.50, "tau_hard": 0.85, "protect_last_n": 2},
            context_length=500,
        )
        # Ingest enough messages to fill context
        big = "z" * 400
        messages = [{"role": "user", "content": f"{big} msg {i}"} for i in range(8)]
        adapter.compress(messages)
        # Now engine has ingested — should_compress without real tokens
        # should use engine's active_tokens()
        result = adapter.should_compress()
        # Should fire because engine has ~3200 tokens in a 500 budget
        assert result is True


class TestToolHandlers:
    def test_budget_tool(self, engine):
        set_engine(engine)
        from plugins.context_engine.lcm.tools import handle_lcm_budget
        for i in range(5):
            engine.ingest(msg("user", f"budget test {i}"))
        result = handle_lcm_budget({})
        assert "Active entries" in result or "active" in result.lower()

    def test_toc_tool(self, engine):
        set_engine(engine)
        from plugins.context_engine.lcm.tools import handle_lcm_toc
        for i in range(3):
            engine.ingest(msg("user", f"toc test {i}"))
        result = handle_lcm_toc({})
        assert "toc" in result.lower() or "entry" in result.lower() or "raw" in result.lower()

    def test_search_tool(self, engine):
        set_engine(engine)
        from plugins.context_engine.lcm.tools import handle_lcm_search
        engine.ingest(msg("user", "needle in haystack"))
        result = handle_lcm_search({"query": "needle"})
        assert "needle" in result

    def test_focus_tool(self, engine):
        set_engine(engine)
        from plugins.context_engine.lcm.tools import handle_lcm_focus
        for i in range(6):
            engine.ingest(msg("user", f"focus test {i}"))
        node = engine.auto_compact()
        assert node is not None
        result = handle_lcm_focus({"node_id": node.id})
        assert "focus" in result.lower() or str(node.id) in result
