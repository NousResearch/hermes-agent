from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


_DURABLE_COLUMNS = (
    "compression_ineffective_count, compression_fallback_streak, "
    "compression_failure_cooldown_until, compression_failure_error, model_config"
)


class _SlottedContextCompressor(ContextCompressor):
    __slots__ = ("slot_state",)


def _new_compressor() -> _SlottedContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = _SlottedContextCompressor(
            model="primary/model",
            provider="primary",
            base_url="https://primary.invalid/v1",
            api_key="primary-key",
            max_tokens=4_000,
            quiet_mode=True,
        )
        _ = compressor.context_length
    compressor.slot_state = {"origin": ["primary"]}
    compressor._private_route_state = {"nested": ["keep"]}
    return compressor


def _durable_snapshot(db: SessionDB, session_id: str) -> dict:
    row = db._conn.execute(
        f"SELECT {_DURABLE_COLUMNS} FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return dict(row)


def _live_snapshot(compressor: ContextCompressor) -> dict:
    return {
        key: value
        for key, value in vars(compressor).items()
        if key not in {"_session_db"}
    } | {"slot_state": compressor.slot_state}


def _seed_bound_route_state(db: SessionDB, compressor: ContextCompressor, session_id: str) -> None:
    db.create_session(
        session_id,
        source="telegram",
        model_config={"_proactive_prune_rearm_tokens": 4096, "unrelated": "preserve"},
    )
    db.set_compression_ineffective_count(session_id, 2)
    db.set_compression_fallback_streak(session_id, 3)
    db.record_compression_failure_cooldown(session_id, time.time() + 300, "rate limited")
    compressor.bind_session_state(db, session_id)


def test_prepare_and_abort_leave_private_slotted_and_durable_state_untouched(tmp_path):
    """Preparing or aborting must not mutate any compressor-owned or durable state.

    This catches an eager setter/write in prepare and reflective rollback that replaces
    unrelated private or slotted objects.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        compressor = _new_compressor()
        _seed_bound_route_state(db, compressor, "PREPARE_ABORT")
        original_slot_state = compressor.slot_state
        original_private_state = compressor._private_route_state
        live_before = _live_snapshot(compressor)
        durable_before = _durable_snapshot(db, "PREPARE_ABORT")

        ticket = compressor.prepare_route_update(
            "replacement/model",
            64_000,
            "https://replacement.invalid/v1",
            "replacement-key",
            "replacement",
            "chat_completions",
            8_000,
        )
        ticket.abort()
        ticket.abort()

        live_after = _live_snapshot(compressor)
        durable_after = _durable_snapshot(db, "PREPARE_ABORT")
        assert compressor.slot_state is original_slot_state
        assert compressor._private_route_state is original_private_state
        assert live_after == live_before
        assert durable_after == durable_before
    finally:
        db.close()


def test_commit_is_exactly_once_and_compensates_atomic_durable_failure(tmp_path):
    """A failed atomic reset restores live state; a successful ticket writes once."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        compressor = _new_compressor()
        session_id = "COMMIT_EXACTLY_ONCE"
        _seed_bound_route_state(db, compressor, session_id)
        live_before = _live_snapshot(compressor)
        durable_before = _durable_snapshot(db, session_id)
        db._conn.execute(
            "CREATE TRIGGER abort_compressor_ticket_reset "
            "BEFORE UPDATE ON sessions "
            f"WHEN NEW.id = '{session_id}' "
            "BEGIN SELECT RAISE(ABORT, 'forced ticket reset failure'); END"
        )

        failed = compressor.prepare_route_update(
            "replacement/model", 64_000, provider="replacement", max_tokens=8_000
        )
        try:
            failed.commit()
        except sqlite3.DatabaseError as exc:
            assert "forced ticket reset failure" in str(exc)
        else:  # pragma: no cover - the trigger is the asserted fault injection
            raise AssertionError("atomic route reset unexpectedly succeeded")

        assert _live_snapshot(compressor) == live_before
        assert _durable_snapshot(db, session_id) == durable_before

        db._conn.execute("DROP TRIGGER abort_compressor_ticket_reset")
        atomic_reset = MagicMock(wraps=db.apply_compressor_route_reset)
        db.apply_compressor_route_reset = atomic_reset
        healthy = compressor.prepare_route_update(
            "replacement/model", 64_000, provider="replacement", max_tokens=8_000
        )
        healthy.commit()
        healthy.commit()

        assert atomic_reset.call_count == 1
        assert compressor.model == "replacement/model"
        assert compressor.context_length == 64_000
        assert compressor.max_tokens == 8_000
        durable_after = _durable_snapshot(db, session_id)
        assert durable_after["compression_ineffective_count"] == 0
        assert durable_after["compression_fallback_streak"] == 0
        assert durable_after["compression_failure_cooldown_until"] is None
        assert durable_after["compression_failure_error"] is None
        assert "_proactive_prune_rearm_tokens" not in (durable_after["model_config"] or "")
    finally:
        db.close()


def test_abort_after_commit_restores_exact_live_and_durable_snapshot(tmp_path):
    """A later owner failure can compensate an already committed compressor ticket."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        compressor = _new_compressor()
        session_id = "ABORT_COMMITTED"
        _seed_bound_route_state(db, compressor, session_id)
        live_before = _live_snapshot(compressor)
        durable_before = _durable_snapshot(db, session_id)

        ticket = compressor.prepare_route_update(
            "replacement/model", 64_000, provider="replacement", max_tokens=8_000
        )
        ticket.commit()
        ticket.abort()
        ticket.abort()

        live_after = _live_snapshot(compressor)
        assert {
            key: value for key, value in live_after.items() if key != "_route_generation"
        } == {
            key: value for key, value in live_before.items() if key != "_route_generation"
        }
        assert _durable_snapshot(db, session_id) == durable_before
    finally:
        db.close()


def test_same_route_ticket_preserves_route_scoped_guards(tmp_path):
    """A window/output recalibration is not a route change and keeps route guards."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        compressor = _new_compressor()
        compressor.tail_mode = "legacy"
        session_id = "SAME_ROUTE"
        _seed_bound_route_state(db, compressor, session_id)
        compressor._aux_context_ceiling = 150_000
        cooldown_before = compressor._summary_failure_cooldown_until
        durable_before = _durable_snapshot(db, session_id)

        ticket = compressor.prepare_route_update(
            "primary/model",
            200_000,
            "https://primary.invalid/v1",
            "rotated-key",
            "primary",
            "",
            80_000,
        )
        ticket.commit()

        assert compressor._aux_context_ceiling == 150_000
        assert compressor._fallback_compression_streak == 3
        assert compressor._summary_failure_cooldown_until == cooldown_before
        # A 200K window receives the built-in 75% floor: (200K - 80K) * .75.
        assert compressor.threshold_tokens == 90_000
        assert compressor.tail_token_budget == 18_000
        durable_after = _durable_snapshot(db, session_id)
        assert durable_after["compression_fallback_streak"] == durable_before["compression_fallback_streak"]
        assert durable_after["compression_failure_cooldown_until"] == durable_before["compression_failure_cooldown_until"]
        assert durable_after["compression_failure_error"] == durable_before["compression_failure_error"]
    finally:
        db.close()


def test_stale_compressor_ticket_is_rejected_before_commit(tmp_path):
    """A ticket prepared against an old generation cannot overwrite the winner."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        compressor = _new_compressor()
        _seed_bound_route_state(db, compressor, "STALE_TICKET")
        atomic_reset = MagicMock(wraps=db.apply_compressor_route_reset)
        db.apply_compressor_route_reset = atomic_reset
        stale = compressor.prepare_route_update(
            "candidate/a", 80_000, provider="candidate-a", max_tokens=4_000
        )
        winner = compressor.prepare_route_update(
            "candidate/b", 96_000, provider="candidate-b", max_tokens=6_000
        )
        winner.commit()
        live_after_winner = _live_snapshot(compressor)
        durable_after_winner = _durable_snapshot(db, "STALE_TICKET")

        try:
            stale.commit()
        except RuntimeError as exc:
            assert "stale compressor route ticket" in str(exc)
        else:  # pragma: no cover - generation validation is the contract
            raise AssertionError("stale compressor route ticket unexpectedly committed")

        assert atomic_reset.call_count == 1
        assert _live_snapshot(compressor) == live_after_winner
        assert _durable_snapshot(db, "STALE_TICKET") == durable_after_winner
    finally:
        db.close()


def test_prepared_ticket_fails_closed_without_atomic_store():
    """Transactional callers cannot silently degrade to four independent writes."""

    class LegacyStore:
        pass

    compressor = _new_compressor()
    compressor._session_db = LegacyStore()
    compressor._session_id = "LEGACY_STRICT"
    live_before = _live_snapshot(compressor)
    ticket = compressor.prepare_route_update("replacement/model", 64_000, provider="replacement")

    try:
        ticket.commit()
    except RuntimeError as exc:
        assert "atomic compressor route reset" in str(exc)
    else:  # pragma: no cover - fail-closed is the contract
        raise AssertionError("prepared ticket degraded to legacy writes")
    assert _live_snapshot(compressor) == live_before


def test_update_model_positional_abi_preserves_legacy_store_writes():
    """The direct ABI keeps the legacy store's four best-effort write paths."""

    class LegacyStore:
        def __init__(self):
            self.calls = []

        def set_compression_ineffective_count(self, session_id, value):
            self.calls.append(("ineffective", session_id, value))

        def set_compression_fallback_streak(self, session_id, value):
            self.calls.append(("fallback", session_id, value))

        def clear_compression_failure_cooldown(self, session_id):
            self.calls.append(("cooldown", session_id))

        def patch_session_model_config(self, session_id, patch_value):
            self.calls.append(("rearm", session_id, patch_value))

    compressor = _new_compressor()
    store = LegacyStore()
    compressor._session_db = store
    compressor._session_id = "LEGACY_DIRECT"
    compressor._ineffective_compression_count = 2
    compressor._fallback_compression_streak = 3

    compressor.update_model(
        "replacement/model",
        64_000,
        "https://replacement.invalid/v1",
        "replacement-key",
        "replacement",
        "chat_completions",
        8_000,
    )

    assert store.calls == [
        ("ineffective", "LEGACY_DIRECT", 0),
        ("fallback", "LEGACY_DIRECT", 0),
        ("cooldown", "LEGACY_DIRECT"),
        ("rearm", "LEGACY_DIRECT", {"_proactive_prune_rearm_tokens": None}),
    ]
