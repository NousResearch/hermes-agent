"""Hygiene idle-timeout cooldowns must not block the in-agent compressor."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB

_HYGIENE_TIMEOUT_ERROR = (
    "session hygiene compression timed out with no output from the summary model"
)


def _bound_compressor(db: SessionDB, session_id: str) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    return compressor


def test_hygiene_idle_timeout_does_not_block_in_agent_compressor(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "hyg-timeout-sid"
    db.create_session(session_id, source="telegram")
    db.record_compression_failure_cooldown(
        session_id,
        time.time() + 300,
        _HYGIENE_TIMEOUT_ERROR,
    )

    compressor = _bound_compressor(db, session_id)

    # Hygiene still sees the durable row so the pre-agent pass can skip.
    assert db.get_compression_failure_cooldown(session_id) is not None
    # The in-conversation compressor must not inherit that watchdog timeout.
    assert compressor.get_active_compression_failure_cooldown() is None
    db.close()


def test_aux_model_fault_cooldown_still_blocks_in_agent_compressor(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "aux-fault-sid"
    db.create_session(session_id, source="telegram")
    db.record_compression_failure_cooldown(
        session_id,
        time.time() + 300,
        "rate limited",
    )

    compressor = _bound_compressor(db, session_id)

    state = compressor.get_active_compression_failure_cooldown()
    assert state is not None
    assert state["error"] == "rate limited"
    db.close()
