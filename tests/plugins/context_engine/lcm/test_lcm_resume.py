"""Regression tests for LCM session resume / _ingested_count correctness.

Fix A: _ingested_count must track input-message count, not active-entry count.
Fix B: raw message ids must be preserved across save/rebuild when active list
       contains non-contiguous raw entries (e.g. after lcm_focus).
Fix 1: HRR store and DAM retriever must be re-attached after engine rebuild so
       resumed sessions retain L2/L3 memory layers.
Fix 3: Legacy session files (no lcm.original_messages) must seed _ingested_count
       from store size to prevent re-ingestion on next compress() call.
"""
from __future__ import annotations

import pytest

from plugins.context_engine.lcm.engine import LcmEngine, ContextEntry
from plugins.context_engine.lcm.config import LcmConfig
from plugins.context_engine.lcm.__init__ import LcmContextEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n: int) -> list[dict]:
    """Return n simple alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"msg {i}"})
    return msgs


def _build_session_data(engine: LcmEngine) -> dict:
    """Build a minimal session_data dict from a live engine (mimics on_session_end).

    Uses active_messages_for_session() so raw entries carry _lcm_raw_id, exactly
    as the real persistence layer does.
    """
    lcm_meta = engine.to_session_metadata()
    # Use annotated messages so _lcm_raw_id is preserved across rebuild
    active_msgs = engine.active_messages_for_session()
    return {"messages": active_msgs, "lcm": lcm_meta}


# ---------------------------------------------------------------------------
# Fix A — _ingested_count across session rotation
# ---------------------------------------------------------------------------

class TestIngestedCountAfterRebuild:
    """_ingested_count should equal the number of *input* messages that were
    originally fed into the engine, not the active-list length."""

    def test_rebuild_sets_ingested_count_to_original_message_count(self):
        """After rebuild from disk, _ingested_count == len(original_messages)."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)

        messages = _make_messages(100)
        outer.compress(messages)
        # Manually run a compaction to shrink the active list so active != 100
        block = outer._engine.find_compactable_block()
        if block:
            outer._engine.compact("summary text", level=1, block_start=block[0], block_end=block[1])

        active_after_compact = len(outer._engine.active)
        assert active_after_compact < 100, "Need compaction to have fired for this test to be meaningful"

        # Build session_data as on_session_end would
        session_data = _build_session_data(outer._engine)

        # Create a new engine instance and call on_session_start with the persisted data
        import json, tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "test-session-abc"
            session_file = pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            session_file.parent.mkdir(parents=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            new_outer = LcmContextEngine(lcm_config=config)
            new_outer.on_session_start(session_id, hermes_home=tmpdir)

        # _ingested_count must equal len(original_messages) saved in lcm metadata
        expected = len(session_data["lcm"]["original_messages"])
        assert new_outer._ingested_count == expected, (
            f"_ingested_count={new_outer._ingested_count} != "
            f"len(original_messages)={expected} — cursor wrongly rebased on active list"
        )

    def test_no_saved_file_ingested_count_is_zero(self):
        """Fresh session_start with no save file resets _ingested_count to 0."""
        import tempfile
        config = LcmConfig(enabled=True)
        outer = LcmContextEngine(lcm_config=config)
        # Simulate already having ingested some messages
        outer._ingested_count = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            outer.on_session_start("nonexistent-session", hermes_home=tmpdir)

        assert outer._ingested_count == 0, (
            f"_ingested_count={outer._ingested_count} should be 0 for fresh session"
        )

    def test_no_new_messages_ingested_after_rebuild(self):
        """After rebuild, calling compress() with the same 100 messages must not
        ingest any new messages (store size must remain unchanged)."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)

        messages = _make_messages(100)
        outer.compress(messages)
        block = outer._engine.find_compactable_block()
        if block:
            outer._engine.compact("summary text", level=1, block_start=block[0], block_end=block[1])

        session_data = _build_session_data(outer._engine)
        store_size_before_rebuild = len(outer._engine.store)

        import json, tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "test-no-reingest"
            session_file = pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            session_file.parent.mkdir(parents=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            new_outer = LcmContextEngine(lcm_config=config)
            new_outer.on_session_start(session_id, hermes_home=tmpdir)

        # Calling compress with the same 100 messages must not add anything to store
        new_outer.compress(messages)
        store_size_after = len(new_outer._engine.store)

        assert store_size_after == store_size_before_rebuild, (
            f"store grew from {store_size_before_rebuild} to {store_size_after} — "
            "messages were re-ingested after rebuild"
        )

    def test_new_messages_after_rebuild_are_ingested(self):
        """After rebuild with 100 messages, compress with 110 ingests exactly 10 new."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)

        messages = _make_messages(100)
        outer.compress(messages)
        block = outer._engine.find_compactable_block()
        if block:
            outer._engine.compact("summary text", level=1, block_start=block[0], block_end=block[1])

        session_data = _build_session_data(outer._engine)
        store_size_at_save = len(outer._engine.store)

        import json, tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "test-new-msgs"
            session_file = pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            session_file.parent.mkdir(parents=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            new_outer = LcmContextEngine(lcm_config=config)
            new_outer.on_session_start(session_id, hermes_home=tmpdir)

        messages_110 = _make_messages(110)
        new_outer.compress(messages_110)

        store_size_after = len(new_outer._engine.store)
        assert store_size_after == store_size_at_save + 10, (
            f"Expected {store_size_at_save + 10} store entries, got {store_size_after}"
        )


# ---------------------------------------------------------------------------
# Fix B — preserve raw message ids across save/rebuild after lcm_focus
# ---------------------------------------------------------------------------

class TestRawIdPreservationAfterFocus:
    """Raw message ids must survive serialization even when active is non-contiguous
    (i.e. after focus_summary splices original raw entries back into active)."""

    def _build_focused_engine(self) -> LcmEngine:
        """Build an engine where active[5].msg_id == 5 after focus."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        engine = LcmEngine(config)
        # Ingest 51 messages
        for i in range(51):
            role = "user" if i % 2 == 0 else "assistant"
            engine.ingest({"role": role, "content": f"msg {i}"})

        # Compact messages [0..20] into a summary
        block = engine.find_compactable_block()
        assert block is not None, "Expected a compactable block"
        node = engine.compact("summary of early msgs", level=1, block_start=block[0], block_end=block[1])

        # Expand the summary back (focus) — this makes active non-contiguous
        ok = engine.focus_summary(node.id)
        assert ok, "focus_summary should succeed"

        return engine

    def test_focused_engine_msg_ids_match_store_positions(self):
        """After focus_summary, each raw entry's msg_id must match its store index."""
        engine = self._build_focused_engine()
        for entry in engine.active:
            if entry.kind == "raw":
                assert entry.msg_id is not None
                msg = engine.store.get(entry.msg_id)
                assert msg is not None, f"msg_id {entry.msg_id} not found in store"
                # The message content must match
                assert entry.message["content"] == msg["content"]

    def test_active_position_5_msg_id_preserved_after_rebuild(self):
        """After save+rebuild, engine.active[5].msg_id equals the original id."""
        engine = self._build_focused_engine()

        # Record the msg_id at position 5 before serialization
        original_msg_id = engine.active[5].msg_id
        assert original_msg_id is not None

        session_data = _build_session_data(engine)
        config = LcmConfig(enabled=True, protect_last_n=2)
        rebuilt = LcmEngine.rebuild_from_session(session_data, config)

        assert rebuilt.active[5].msg_id == original_msg_id, (
            f"msg_id at position 5 changed from {original_msg_id} "
            f"to {rebuilt.active[5].msg_id} after rebuild"
        )

    def test_lcm_pin_after_rebuild_affects_correct_store_row(self):
        """pin() on active[5] after rebuild must mark the same store id
        that was originally at position 5, not a reassigned one."""
        engine = self._build_focused_engine()
        original_msg_id_5 = engine.active[5].msg_id
        assert original_msg_id_5 is not None

        session_data = _build_session_data(engine)
        config = LcmConfig(enabled=True, protect_last_n=2)
        rebuilt = LcmEngine.rebuild_from_session(session_data, config)

        rebuilt.pin([rebuilt.active[5].msg_id])
        assert original_msg_id_5 in rebuilt._pinned_ids, (
            f"Pinned msg_id {rebuilt.active[5].msg_id} but original was {original_msg_id_5}"
        )

    def test_rebuild_without_lcm_raw_id_falls_back_gracefully(self):
        """Sessions saved WITHOUT _lcm_raw_id (legacy) still rebuild correctly
        for the simple contiguous-suffix case."""
        config = LcmConfig(enabled=True, protect_last_n=2)
        # Build a fresh session_data without any _lcm_raw_id keys
        original_messages = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(5)
        ]
        # active messages: 1 summary + 2 raw (no _lcm_raw_id keys)
        active_messages = [
            {"role": "user", "content": "[Summary]", "_lcm_summary": True, "_lcm_node_id": 0},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
        ]
        summaries = [
            {"node_id": 0, "source_ids": [0, 1, 2], "text": "Early msgs", "level": 1, "tokens": 30}
        ]
        session_data = {
            "messages": active_messages,
            "lcm": {
                "summaries": summaries,
                "original_messages": original_messages,
            }
        }
        rebuilt = LcmEngine.rebuild_from_session(session_data, config)
        # Raw entries [1] and [2] map to store positions 3 and 4 (suffix fallback)
        assert rebuilt.active[1].kind == "raw"
        assert rebuilt.active[1].msg_id == 3
        assert rebuilt.active[2].msg_id == 4


# ---------------------------------------------------------------------------
# Fix 2 — preserve _ingested_count across compression-rollover boundaries
# ---------------------------------------------------------------------------

class TestCompressionRolloverPreservesIngestedCount:
    """on_session_start with boundary_reason='compression' must NOT reset
    _ingested_count — the engine state already covers those messages."""

    def test_compression_rollover_preserves_ingested_count(self):
        """boundary_reason='compression' + no save file → _ingested_count unchanged."""
        import tempfile
        config = LcmConfig(enabled=True)
        outer = LcmContextEngine(lcm_config=config)
        msgs = _make_messages(100)
        outer.compress(msgs)
        assert outer._ingested_count == 100

        with tempfile.TemporaryDirectory() as tmpdir:
            # new_session_id has no save file — simulate compression rollover
            outer.on_session_start(
                "new-session-after-compression",
                hermes_home=tmpdir,
                boundary_reason="compression",
            )

        assert outer._ingested_count == 100, (
            f"_ingested_count={outer._ingested_count} — "
            "should be preserved (100) across compression rollover"
        )

    def test_initial_session_start_resets_ingested_count(self):
        """boundary_reason='initial' + no save file → _ingested_count reset to 0."""
        import tempfile
        config = LcmConfig(enabled=True)
        outer = LcmContextEngine(lcm_config=config)
        outer._ingested_count = 50  # simulate prior state

        with tempfile.TemporaryDirectory() as tmpdir:
            outer.on_session_start(
                "brand-new-session",
                hermes_home=tmpdir,
                boundary_reason="initial",
            )

        assert outer._ingested_count == 0, (
            f"_ingested_count={outer._ingested_count} — "
            "should be 0 for a brand-new session"
        )

    def test_no_boundary_reason_resets_ingested_count(self):
        """Default (no boundary_reason kwarg) + no save file → _ingested_count=0."""
        import tempfile
        config = LcmConfig(enabled=True)
        outer = LcmContextEngine(lcm_config=config)
        outer._ingested_count = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            outer.on_session_start("another-new-session", hermes_home=tmpdir)

        assert outer._ingested_count == 0, (
            f"_ingested_count={outer._ingested_count} — "
            "should be 0 when no boundary_reason is provided"
        )

    def test_compression_rollover_engine_state_unchanged(self):
        """After compression rollover, engine's active list and store are intact."""
        import tempfile
        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)
        msgs = _make_messages(20)
        outer.compress(msgs)
        store_size_before = len(outer._engine.store)
        active_size_before = len(outer._engine.active)

        with tempfile.TemporaryDirectory() as tmpdir:
            outer.on_session_start(
                "rolled-over-session",
                hermes_home=tmpdir,
                boundary_reason="compression",
            )

        assert len(outer._engine.store) == store_size_before, (
            "Engine store was reset during compression rollover"
        )
        assert len(outer._engine.active) == active_size_before, (
            "Engine active list was reset during compression rollover"
        )


# ---------------------------------------------------------------------------
# Fix 1 — HRR store and DAM retriever must survive engine rebuild
# ---------------------------------------------------------------------------

class TestHrrAndDamReattachedAfterRebuild:
    """After on_session_start() rebuilds the engine from a session file, the fresh
    LcmEngine instance must have hrr_store (L3) and retriever (L2) re-attached.

    Without the fix, rebuild_from_session() returns a plain LcmEngine with no
    hrr_store / retriever, so all cross-session and per-session memory is lost.
    """

    def _save_and_reload(self, messages_n: int, tmpdir: str, session_id: str):
        """Helper: build engine, compress messages, save, reload via on_session_start."""
        import json
        import pathlib

        config = LcmConfig(enabled=True, protect_last_n=2)
        outer = LcmContextEngine(lcm_config=config)
        msgs = _make_messages(messages_n)
        outer.compress(msgs)

        session_data = _build_session_data(outer._engine)
        session_file = (
            pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
        )
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        new_outer = LcmContextEngine(lcm_config=config)
        new_outer.on_session_start(session_id, hermes_home=tmpdir)
        return new_outer

    def test_hrr_store_reattached_after_rebuild(self):
        """engine._engine.hrr_store must not be None after session resume."""
        import tempfile

        # Only meaningful when HRR sub-package is available; skip otherwise
        try:
            from plugins.context_engine.lcm.hrr.store import MemoryStore as _M
        except (ImportError, Exception):
            pytest.skip("HRR sub-package not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            new_outer = self._save_and_reload(20, tmpdir, "session-hrr-reattach")

        assert hasattr(new_outer._engine, "hrr_store"), (
            "engine._engine has no hrr_store attribute after rebuild"
        )
        assert new_outer._engine.hrr_store is not None, (
            "engine._engine.hrr_store is None after rebuild — HRR store was dropped"
        )

    def test_dam_retriever_reattached_after_rebuild(self):
        """engine._engine.retriever must not be None after session resume."""
        import tempfile

        # Only meaningful when DAM sub-package is available; skip otherwise
        try:
            from plugins.context_engine.lcm.dam import DAMRetriever as _D
        except (ImportError, Exception):
            pytest.skip("DAM sub-package not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            new_outer = self._save_and_reload(20, tmpdir, "session-dam-reattach")

        assert hasattr(new_outer._engine, "retriever"), (
            "engine._engine has no retriever attribute after rebuild"
        )
        assert new_outer._engine.retriever is not None, (
            "engine._engine.retriever is None after rebuild — DAM retriever was dropped"
        )

    def test_fresh_session_no_file_still_has_hrr_and_dam(self):
        """Baseline: a brand-new engine (no session file) has hrr_store and retriever."""
        try:
            from plugins.context_engine.lcm.hrr.store import MemoryStore as _M
            from plugins.context_engine.lcm.dam import DAMRetriever as _D
        except (ImportError, Exception):
            pytest.skip("HRR or DAM sub-package not installed")

        config = LcmConfig(enabled=True)
        outer = LcmContextEngine(lcm_config=config)
        assert hasattr(outer._engine, "hrr_store") and outer._engine.hrr_store is not None
        assert hasattr(outer._engine, "retriever") and outer._engine.retriever is not None


# ---------------------------------------------------------------------------
# Fix 3 — legacy session files must seed _ingested_count from store size
# ---------------------------------------------------------------------------

class TestLegacySessionIngestedCount:
    """When a session file has no lcm.original_messages (legacy format), the
    rebuilt engine's store is populated from active raw messages.  _ingested_count
    must be seeded from that store size so the next compress() call with the same
    messages does NOT re-ingest them.

    Without the fix: _ingested_count = 0, so compress() sees len(messages) - 0 = N
    "new" messages and re-ingests everything, duplicating the store.
    """

    def _make_legacy_session_data(self, n: int) -> dict:
        """Build a legacy-format session_data with n raw messages and no
        lcm.original_messages key (or an empty list for it)."""
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(n)
        ]
        # Legacy format: lcm block present but no original_messages
        return {
            "messages": messages,
            "lcm": {
                "summaries": [],
                # Deliberately omit "original_messages" to simulate a legacy file
            },
        }

    def test_ingested_count_seeded_from_store_for_legacy_session(self):
        """After resuming a legacy session with 50 messages, _ingested_count==50."""
        import json
        import pathlib
        import tempfile

        config = LcmConfig(enabled=True, protect_last_n=2)
        session_data = self._make_legacy_session_data(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "legacy-seed-count"
            session_file = (
                pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            )
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            outer = LcmContextEngine(lcm_config=config)
            outer.on_session_start(session_id, hermes_home=tmpdir)

        assert outer._ingested_count == 50, (
            f"_ingested_count={outer._ingested_count} but expected 50 for legacy "
            "session with 50 messages — cursor not seeded from store size"
        )

    def test_compress_does_not_reingest_after_legacy_resume(self):
        """After legacy resume, compress(same_50_msgs) must not grow the store."""
        import json
        import pathlib
        import tempfile

        config = LcmConfig(enabled=True, protect_last_n=2)
        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(50)
        ]
        session_data = self._make_legacy_session_data(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "legacy-no-reingest"
            session_file = (
                pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            )
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            outer = LcmContextEngine(lcm_config=config)
            outer.on_session_start(session_id, hermes_home=tmpdir)

        store_size_after_resume = len(outer._engine.store)
        # Calling compress with the same 50 messages must not add anything
        outer.compress(msgs)
        store_size_after_compress = len(outer._engine.store)

        assert store_size_after_compress == store_size_after_resume, (
            f"Store grew from {store_size_after_resume} to {store_size_after_compress} "
            "after compress() — legacy session messages were re-ingested"
        )

    def test_new_messages_after_legacy_resume_are_ingested(self):
        """After legacy resume with 50 msgs, compress(60 msgs) ingests exactly 10."""
        import json
        import pathlib
        import tempfile

        config = LcmConfig(enabled=True, protect_last_n=2)
        session_data = self._make_legacy_session_data(50)
        msgs_60 = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(60)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "legacy-new-msgs"
            session_file = (
                pathlib.Path(tmpdir) / "sessions" / f"session_{session_id}.json"
            )
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(json.dumps(session_data), encoding="utf-8")

            outer = LcmContextEngine(lcm_config=config)
            outer.on_session_start(session_id, hermes_home=tmpdir)

        store_after_resume = len(outer._engine.store)
        outer.compress(msgs_60)
        store_after_compress = len(outer._engine.store)

        assert store_after_compress == store_after_resume + 10, (
            f"Expected {store_after_resume + 10} store entries after adding 10 new "
            f"messages, got {store_after_compress}"
        )
