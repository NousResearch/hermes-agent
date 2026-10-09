from __future__ import annotations

import json
import sqlite3
import threading
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


def test_abort_after_concurrent_winner_preserves_live_and_durable_winner(tmp_path):
    """A lost durable CAS must not roll the local owner back to its old route."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        older = _new_compressor()
        session_id = "CROSS_INSTANCE_CAS"
        _seed_bound_route_state(db, older, session_id)

        older_ticket = older.prepare_route_update(
            "older/replacement", 64_000, provider="older-route", max_tokens=8_000
        )
        older_ticket.commit()

        # A newer compressor instance owns a same-route recalibration that
        # intentionally preserves the guards written after the older commit.
        db.set_compression_ineffective_count(session_id, 4)
        db.set_compression_fallback_streak(session_id, 7)
        db.record_compression_failure_cooldown(
            session_id, time.time() + 600, "newer route cooldown"
        )
        db.patch_session_model_config(
            session_id,
            {"_proactive_prune_rearm_tokens": 8192, "newer": "must-survive"},
        )
        newer = _new_compressor()
        newer.bind_session_state(db, session_id)
        newer_ticket = newer.prepare_route_update(
            "primary/model",
            120_000,
            "https://primary.invalid/v1",
            "newer-key",
            "primary",
            "",
            6_000,
        )
        newer_ticket.commit()
        durable_after_newer_commit = _durable_snapshot(db, session_id)
        older_live_after_own_commit = _live_snapshot(older)
        newer_live_after_winner = _live_snapshot(newer)

        older_ticket.abort()

        assert _durable_snapshot(db, session_id) == durable_after_newer_commit
        assert _live_snapshot(older) == older_live_after_own_commit
        assert _live_snapshot(newer) == newer_live_after_winner
    finally:
        db.close()


def test_abort_does_not_overwrite_equal_valued_newer_commit_from_another_instance(tmp_path):
    """A durable revision must distinguish an ABA-equivalent newer reset."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        older = _new_compressor()
        newer = _new_compressor()
        session_id = "CROSS_INSTANCE_ABA"
        _seed_bound_route_state(db, older, session_id)
        newer.bind_session_state(db, session_id)

        older_ticket = older.prepare_route_update(
            "older/replacement", 64_000, provider="older-route", max_tokens=8_000
        )
        newer_ticket = newer.prepare_route_update(
            "newer/replacement", 96_000, provider="newer-route", max_tokens=12_000
        )

        older_ticket.commit()
        durable_after_older_commit = _durable_snapshot(db, session_id)
        newer_ticket.commit()
        durable_after_newer_commit = _durable_snapshot(db, session_id)

        # Both ordinary route changes write the same business reset values.
        for column in (
            "compression_ineffective_count",
            "compression_fallback_streak",
            "compression_failure_cooldown_until",
            "compression_failure_error",
        ):
            assert durable_after_newer_commit[column] == durable_after_older_commit[column]
        older_config = json.loads(durable_after_older_commit["model_config"])
        newer_config = json.loads(durable_after_newer_commit["model_config"])
        assert newer_config["_compressor_route_revision"] > older_config[
            "_compressor_route_revision"
        ]

        older_ticket.abort()

        assert _durable_snapshot(db, session_id) == durable_after_newer_commit
    finally:
        db.close()


def test_abort_restores_transaction_local_predecessor_after_interleaved_commit(tmp_path):
    """Abort must restore B when B commits after A's early read but before A's reset."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        older = _new_compressor()
        newer = _new_compressor()
        session_id = "CROSS_INSTANCE_PREDECESSOR"
        _seed_bound_route_state(db, older, session_id)
        newer.bind_session_state(db, session_id)

        older_ticket = older.prepare_route_update(
            "older/replacement", 64_000, provider="older-route", max_tokens=8_000
        )
        newer_ticket = newer.prepare_route_update(
            "newer/replacement", 96_000, provider="newer-route", max_tokens=12_000
        )
        original_reset = db.apply_compressor_route_reset
        interleaving = {}

        def reset_after_newer_commit(*args, **kwargs):
            db.apply_compressor_route_reset = original_reset
            newer_ticket.commit()
            interleaving["newer_commit"] = _durable_snapshot(db, session_id)
            return original_reset(*args, **kwargs)

        db.apply_compressor_route_reset = reset_after_newer_commit
        older_ticket.commit()
        older_ticket.abort()

        assert _durable_snapshot(db, session_id) == interleaving["newer_commit"]
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


def test_concurrent_generation_zero_commits_allow_exactly_one_write(tmp_path):
    """Two same-owner generation-zero tickets serialize before any publication.

    The reset barrier forces both unfixed commits past generation validation.  A
    fixed owner lock keeps the second commit outside the boundary until the
    first publishes generation one, so only the first reset reaches SQLite.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        compressor = _new_compressor()
        session_id = "CONCURRENT_GENERATION_ZERO"
        _seed_bound_route_state(db, compressor, session_id)
        tickets = [
            compressor.prepare_route_update(
                "candidate/a", 80_000, provider="candidate-a", max_tokens=4_000,
            ),
            compressor.prepare_route_update(
                "candidate/b", 96_000, provider="candidate-b", max_tokens=6_000,
            ),
        ]
        assert {ticket._owner_generation for ticket in tickets} == {0}

        original_reset = db.apply_compressor_route_reset
        reset_barrier = threading.Barrier(2)
        reset_calls = []
        reset_calls_lock = threading.Lock()

        def _barrier_reset(*args, **kwargs):
            with reset_calls_lock:
                reset_calls.append(threading.current_thread().name)
            try:
                reset_barrier.wait(timeout=0.25)
            except threading.BrokenBarrierError:
                # With the owner lock, the first call intentionally has no peer:
                # the second ticket cannot reach the reset until generation one
                # has been published.
                pass
            return original_reset(*args, **kwargs)

        db.apply_compressor_route_reset = _barrier_reset
        start = threading.Barrier(3)
        outcomes = [{"error": None}, {"error": None}]

        def _commit(index):
            start.wait()
            try:
                tickets[index].commit()
            except Exception as exc:  # one deterministic stale-ticket loser
                outcomes[index]["error"] = exc

        threads = [
            threading.Thread(target=_commit, args=(index,), name=f"route-{index}")
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(5)

        assert all(not thread.is_alive() for thread in threads)
        assert len(reset_calls) == 1
        assert sum(outcome["error"] is None for outcome in outcomes) == 1
        loser_error = next(outcome["error"] for outcome in outcomes if outcome["error"])
        assert isinstance(loser_error, RuntimeError)
        assert "stale compressor route ticket" in str(loser_error)
        winner_index = next(
            index for index, outcome in enumerate(outcomes) if outcome["error"] is None
        )
        assert compressor.model == f"candidate/{'a' if winner_index == 0 else 'b'}"
        durable = _durable_snapshot(db, session_id)
        assert json.loads(durable["model_config"])["_compressor_route_revision"] == 1
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
