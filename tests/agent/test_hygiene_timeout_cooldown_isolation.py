"""Pre-agent hygiene retry spacing must not block in-agent compression."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

_HYGIENE_TIMEOUT_ERROR = (
    "session hygiene compression timed out with no output from the summary model"
)

_HYGIENE_TURNHOLD_ERROR = (
    "hygiene compression deferred: turn-hold budget expired while the "
    "summary was still streaming"
)


def _bound_compressor(db, session_id: str) -> ContextCompressor:
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
    compressor.bind_session_state(db, session_id, persistence=db)
    return compressor


def test_hygiene_idle_timeout_does_not_block_in_agent_compressor(tmp_path: Path):
    db = storage_handle(tmp_path)
    session_id = "hyg-timeout-sid"
    seed_session(db, session_id, "telegram")
    db.messages.record_compression_cooldown(
        session_id,
        time.time() + 300,
        _HYGIENE_TIMEOUT_ERROR,
    )

    compressor = _bound_compressor(db, session_id)

    # Hygiene still sees the durable row so the pre-agent pass can skip.
    assert db.messages.read_compression_cooldown(session_id, time.time()) is not None
    # The in-conversation compressor must not inherit that watchdog timeout.
    assert compressor.get_active_compression_failure_cooldown() is None
    db.close()


def test_hygiene_turnhold_deferral_does_not_block_in_agent_compressor(
    tmp_path: Path,
):
    db = storage_handle(tmp_path)
    session_id = "hyg-turnhold-sid"
    seed_session(db, session_id, "telegram")
    db.messages.record_compression_cooldown(
        session_id,
        time.time() + 300,
        _HYGIENE_TURNHOLD_ERROR,
    )

    compressor = _bound_compressor(db, session_id)

    # Hygiene still sees its flat retry-spacing row and avoids respawning a
    # doomed 15-second turn-hold attempt on every incoming message.
    assert db.messages.read_compression_cooldown(session_id, time.time()) is not None
    # The in-conversation compressor has a separate budget. A healthy hygiene
    # stream being deferred must not suppress that path for another minute.
    assert compressor.get_active_compression_failure_cooldown() is None
    db.close()


def test_aux_model_fault_cooldown_still_blocks_in_agent_compressor(tmp_path: Path):
    db = storage_handle(tmp_path)
    session_id = "aux-fault-sid"
    seed_session(db, session_id, "telegram")
    db.messages.record_compression_cooldown(
        session_id,
        time.time() + 300,
        "rate limited",
    )

    compressor = _bound_compressor(db, session_id)

    state = compressor.get_active_compression_failure_cooldown()
    assert state is not None
    assert state["error"] == "rate limited"
    db.close()


def test_hygiene_row_clears_stale_in_memory_aux_cooldown(tmp_path: Path):
    """A later hygiene overwrite must not leave the in-memory aux cooldown armed."""
    db = storage_handle(tmp_path)
    session_id = "hyg-overwrite-sid"
    seed_session(db, session_id, "telegram")
    db.messages.record_compression_cooldown(
        session_id,
        time.time() + 300,
        "rate limited",
    )

    compressor = _bound_compressor(db, session_id)
    loaded = compressor.get_active_compression_failure_cooldown()
    assert loaded is not None
    assert loaded["error"] == "rate limited"

    db.messages.record_compression_cooldown(
        session_id,
        time.time() + 300,
        _HYGIENE_TIMEOUT_ERROR,
    )

    assert compressor.get_active_compression_failure_cooldown(refresh=True) is None
    assert compressor.get_active_compression_failure_cooldown() is None
    db.close()
