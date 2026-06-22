"""Compression deferral state persistence tests.

When the gateway restarts, ContextCompressor resets all state to zero.
This means `should_defer_preflight_to_real_usage()` can't defer because
`last_real_prompt_tokens == 0`, causing unnecessary compression on the
first turn.  The fix persists critical deferral state to `state_meta`
so it survives process restarts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB


def _make_compressor(context_length: int = 128_000):
    from agent.context_compressor import ContextCompressor

    return ContextCompressor(
        model="test/model",
        threshold_percent=0.85,
        quiet_mode=True,
        config_context_length=context_length,
    )


def _make_db(tmp_path: Path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("TEST_SESSION", source="cli")
    return db


class TestSaveDeferralState:
    """save_deferral_state writes critical fields to state_meta."""

    def test_saves_to_db(self, tmp_path):
        db = _make_db(tmp_path)
        c = _make_compressor()
        c.last_real_prompt_tokens = 50_000
        c.last_rough_tokens_when_real_prompt_fit = 95_000
        c.compression_count = 2

        c.save_deferral_state(db, "TEST_SESSION")

        raw = db.get_meta("compression_state:TEST_SESSION")
        assert raw is not None
        state = json.loads(raw)
        assert state["last_real_prompt_tokens"] == 50_000
        assert state["last_rough_tokens_when_real_prompt_fit"] == 95_000
        assert state["compression_count"] == 2

    def test_overwrites_existing(self, tmp_path):
        db = _make_db(tmp_path)
        c = _make_compressor()

        c.last_real_prompt_tokens = 10_000
        c.save_deferral_state(db, "TEST_SESSION")

        c.last_real_prompt_tokens = 60_000
        c.save_deferral_state(db, "TEST_SESSION")

        state = json.loads(db.get_meta("compression_state:TEST_SESSION"))
        assert state["last_real_prompt_tokens"] == 60_000


class TestRestoreDeferralState:
    """restore_deferral_state loads fields from state_meta."""

    def test_restores_from_db(self, tmp_path):
        db = _make_db(tmp_path)
        db.set_meta(
            "compression_state:TEST_SESSION",
            json.dumps({
                "last_real_prompt_tokens": 50_000,
                "last_rough_tokens_when_real_prompt_fit": 95_000,
                "compression_count": 2,
            }),
        )

        c = _make_compressor()
        assert c.last_real_prompt_tokens == 0

        c.restore_deferral_state(db, "TEST_SESSION")

        assert c.last_real_prompt_tokens == 50_000
        assert c.last_rough_tokens_when_real_prompt_fit == 95_000
        assert c.compression_count == 2

    def test_no_op_when_no_saved_state(self, tmp_path):
        db = _make_db(tmp_path)
        c = _make_compressor()
        c.last_real_prompt_tokens = 0

        c.restore_deferral_state(db, "TEST_SESSION")

        assert c.last_real_prompt_tokens == 0

    def test_handles_corrupt_json(self, tmp_path):
        db = _make_db(tmp_path)
        db.set_meta("compression_state:TEST_SESSION", "not-json{")

        c = _make_compressor()
        c.restore_deferral_state(db, "TEST_SESSION")

        assert c.last_real_prompt_tokens == 0

    def test_handles_missing_fields(self, tmp_path):
        db = _make_db(tmp_path)
        db.set_meta(
            "compression_state:TEST_SESSION",
            json.dumps({"last_real_prompt_tokens": 42_000}),
        )

        c = _make_compressor()
        c.restore_deferral_state(db, "TEST_SESSION")

        assert c.last_real_prompt_tokens == 42_000
        assert c.last_rough_tokens_when_real_prompt_fit == 0


class TestDeferralSurvivesRestart:
    """End-to-end: save before shutdown, restore after restart."""

    def test_deferral_works_after_restore(self, tmp_path):
        db = _make_db(tmp_path)
        # 128K context * 0.85 = 108,800 threshold
        # Set baseline high enough that rough estimate growth is within tolerance
        c1 = _make_compressor()
        c1.last_real_prompt_tokens = 50_000
        c1.last_rough_tokens_when_real_prompt_fit = 110_000
        c1.compression_count = 1
        c1.save_deferral_state(db, "TEST_SESSION")

        c2 = _make_compressor()
        assert c2.last_real_prompt_tokens == 0
        # Without restore: rough 112K >= threshold 108.8K, but no real token data → no defer
        assert not c2.should_defer_preflight_to_real_usage(112_000)

        c2.restore_deferral_state(db, "TEST_SESSION")
        # After restore: real=50K < threshold, baseline=110K, growth=2K < tolerated 5.4K → defer
        assert c2.should_defer_preflight_to_real_usage(112_000)
