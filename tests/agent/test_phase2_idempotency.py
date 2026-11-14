"""Durable idempotency lifecycle tests for Phase 2 mutating tool calls."""

from __future__ import annotations

import sqlite3
import stat
import threading

import pytest

from agent.phase2_idempotency import (
    MutationClaimError,
    MutationClaimStore,
    MutationState,
)


def _envelope(
    *,
    key: str = "d" * 64,
    attempt_id: str = "a-1",
    fence: int = 1,
    graph_id: str = "g-1",
    node_id: str = "n-1",
):
    return {
        "graph_id": graph_id,
        "node_id": node_id,
        "attempt_id": attempt_id,
        "idempotency_key": key,
        "fence": fence,
    }


def test_crash_before_effect_replay_returns_durable_in_progress_attempt(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    first = store.try_claim(
        _envelope(), tool_name="write_file", args={"path": "one"}
    )
    replay = store.try_claim(
        _envelope(attempt_id="a-2", fence=2),
        tool_name="write_file",
        args={"path": "one"},
    )

    assert first.state is MutationState.IN_PROGRESS
    assert first.created is True
    assert replay.state is MutationState.IN_PROGRESS
    assert replay.attempt_id == "a-1"
    assert replay.fence == 1
    assert replay.created is False
    assert len(store.read_attempts("d" * 64)) == 1


def test_exact_replay_returns_same_attempt_and_state(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    first = store.try_claim(_envelope(), tool_name="patch", args={"path": "one"})
    replay = store.try_claim(_envelope(), tool_name="patch", args={"path": "one"})

    assert replay == type(first)(
        idempotency_key="d" * 64,
        attempt_id="a-1",
        fence=1,
        state=MutationState.IN_PROGRESS,
        created=False,
    )


def test_mutation_claim_is_atomic_under_concurrency(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope(key="e" * 64, fence=4)
    barrier = threading.Barrier(8)
    outcomes = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        outcome = store.try_claim(envelope, tool_name="patch", args={"path": "one"})
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(outcome.created for outcome in outcomes) == 1
    assert all(outcome.state is MutationState.IN_PROGRESS for outcome in outcomes)
    assert len(store.read_attempts("e" * 64)) == 1


@pytest.mark.parametrize(
    "terminal",
    [MutationState.COMPLETED, MutationState.INDETERMINATE],
)
def test_effect_possible_terminal_state_suppresses_new_attempt(tmp_path, terminal):
    store = MutationClaimStore(tmp_path / "phase2.db")
    first_env = _envelope()
    store.try_claim(first_env, tool_name="write_file", args={"path": "one"})
    store.settle(first_env, state=terminal)

    replay = store.try_claim(
        _envelope(attempt_id="a-2", fence=2),
        tool_name="write_file",
        args={"path": "one"},
    )

    assert replay.attempt_id == "a-1"
    assert replay.state is terminal
    assert replay.created is False
    assert len(store.read_attempts("d" * 64)) == 1


@pytest.mark.parametrize(
    "terminal",
    [
        MutationState.FAILED_PROVEN_NO_EFFECT,
        MutationState.CANCELLED_BEFORE_EFFECT,
    ],
)
def test_proven_non_effect_admits_exactly_one_next_attempt(tmp_path, terminal):
    store = MutationClaimStore(tmp_path / "phase2.db")
    first_env = _envelope()
    store.try_claim(first_env, tool_name="write_file", args={})
    store.settle(first_env, state=terminal)

    second_env = _envelope(attempt_id="a-2", fence=2)
    second = store.try_claim(second_env, tool_name="write_file", args={})
    suppressed = store.try_claim(
        _envelope(attempt_id="a-3", fence=3), tool_name="write_file", args={}
    )

    assert second.created is True
    assert second.state is MutationState.IN_PROGRESS
    assert suppressed.attempt_id == "a-2"
    assert suppressed.created is False
    assert len(store.read_attempts("d" * 64)) == 2


def test_explicit_readmission_is_durable_and_single_use(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    first_env = _envelope()
    store.try_claim(first_env, tool_name="write_file", args={})
    store.settle(first_env, state=MutationState.INDETERMINATE)
    assert store.record_readmission(
        "d" * 64,
        prior_attempt_id="a-1",
        readmission_id="operator-decision-1",
        reason="operator verified external state",
    )
    assert not store.record_readmission(
        "d" * 64,
        prior_attempt_id="a-1",
        readmission_id="operator-decision-1",
        reason="operator verified external state",
    )

    second = store.try_claim(
        _envelope(attempt_id="a-2", fence=2),
        tool_name="write_file",
        args={},
        readmission_id="operator-decision-1",
    )
    assert second.created is True
    with pytest.raises(MutationClaimError, match="latest attempt|already consumed"):
        store.try_claim(
            _envelope(attempt_id="a-3", fence=3),
            tool_name="write_file",
            args={},
            readmission_id="operator-decision-1",
        )


def test_settlement_is_append_only_and_idempotent_only_when_identical(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope()
    store.try_claim(envelope, tool_name="write_file", args={})
    first = store.settle(
        envelope,
        state=MutationState.COMPLETED,
        outcome_hash="a" * 64,
    )
    replay = store.settle(
        envelope,
        state=MutationState.COMPLETED,
        outcome_hash="a" * 64,
    )

    assert first.created is True
    assert first.state is MutationState.COMPLETED
    assert replay.created is False
    with pytest.raises(MutationClaimError, match="different settlement"):
        store.settle(envelope, state=MutationState.INDETERMINATE)
    with pytest.raises(MutationClaimError, match="terminal"):
        store.settle(envelope, state=MutationState.IN_PROGRESS)


def test_invalid_readmission_fails_closed(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    with pytest.raises(MutationClaimError, match="does not exist"):
        store.record_readmission(
            "d" * 64,
            prior_attempt_id="missing",
            readmission_id="decision",
            reason="operator decision",
        )

    envelope = _envelope()
    store.try_claim(envelope, tool_name="write_file", args={})
    store.record_readmission(
        "d" * 64,
        prior_attempt_id="a-1",
        readmission_id="decision",
        reason="first reason",
    )
    with pytest.raises(MutationClaimError, match="another decision"):
        store.record_readmission(
            "d" * 64,
            prior_attempt_id="a-1",
            readmission_id="decision",
            reason="changed reason",
        )


def test_claim_persists_canonical_identity_without_raw_arguments(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope(key="f" * 64, attempt_id="a-secret", fence=2)
    store.try_claim(
        envelope,
        tool_name="write_file",
        args={"path": "/tmp/out", "content": "do-not-store-this"},
    )

    claims = store.read_all()
    assert claims[0]["graph_id"] == "g-1"
    assert claims[0]["node_id"] == "n-1"
    assert claims[0]["attempt_id"] == "a-secret"
    assert claims[0]["tool_name"] == "write_file"
    assert claims[0]["fence"] == 2
    assert claims[0]["state"] == "in_progress"
    assert len(claims[0]["args_hash"]) == 64
    assert "do-not-store-this" not in repr(claims)


def test_distinct_idempotency_keys_are_independent(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    first_a = store.try_claim(
        _envelope(key="a" * 64), tool_name="write_file", args={}
    )
    first_b = store.try_claim(
        _envelope(key="b" * 64), tool_name="write_file", args={}
    )
    replay_a = store.try_claim(
        _envelope(key="a" * 64), tool_name="write_file", args={}
    )

    assert first_a.created and first_b.created
    assert not replay_a.created
    assert len(store.read_all()) == 2


def test_logical_operation_store_is_append_only(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    store.try_claim(
        _envelope(key="c" * 64), tool_name="delete_file", args={"path": "/tmp/x"}
    )

    with sqlite3.connect(tmp_path / "phase2.db") as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "UPDATE mutation_operations SET tool_name = 'tampered' "
                "WHERE idempotency_key = ?",
                ("c" * 64,),
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("graph_id", "", "graph_id"),
        ("node_id", None, "node_id"),
        ("attempt_id", 7, "attempt_id"),
        ("idempotency_key", "short", "idempotency_key"),
        ("fence", True, "fence"),
        ("fence", "1", "fence"),
        ("fence", 2**63, "fence"),
    ],
)
def test_invalid_claim_envelope_is_rejected_typed(tmp_path, field, value, message):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope()
    envelope[field] = value

    with pytest.raises(MutationClaimError, match=message):
        store.try_claim(envelope, tool_name="write_file", args={})
    assert not (tmp_path / "phase2.db").exists()


def test_invalid_tool_and_args_are_rejected_typed(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope()
    with pytest.raises(MutationClaimError, match="tool_name"):
        store.try_claim(envelope, tool_name="", args={})
    with pytest.raises(MutationClaimError, match="args must be a mapping"):
        store.try_claim(envelope, tool_name="write_file", args=[])  # type: ignore[arg-type]
    with pytest.raises(MutationClaimError, match="canonically JSON serializable"):
        store.try_claim(envelope, tool_name="write_file", args={"bad": object()})
    with pytest.raises(MutationClaimError, match="canonically JSON serializable"):
        store.try_claim(envelope, tool_name="write_file", args={"bad": float("nan")})
    with pytest.raises(MutationClaimError, match="canonically JSON serializable"):
        store.try_claim(envelope, tool_name="write_file", args={1: "ambiguous"})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"graph_id": "g-other"}, "collision"),
        ({"node_id": "n-other"}, "collision"),
    ],
)
def test_different_logical_intent_raises_collision(tmp_path, change, message):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope(key="9" * 64, attempt_id="a-col", fence=7)
    store.try_claim(envelope, tool_name="write_file", args={"k": "v"})

    collision = {**envelope, **change}
    with pytest.raises(MutationClaimError, match=message):
        store.try_claim(collision, tool_name="write_file", args={"k": "v"})


def test_changed_tool_or_args_raises_collision(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope(key="9" * 64, attempt_id="a-col", fence=7)
    store.try_claim(envelope, tool_name="write_file", args={"k": "v"})

    with pytest.raises(MutationClaimError, match="collision"):
        store.try_claim(envelope, tool_name="delete_file", args={"k": "v"})
    with pytest.raises(MutationClaimError, match="collision"):
        store.try_claim(envelope, tool_name="write_file", args={"k": "changed"})


def test_attempt_and_fence_are_not_part_of_logical_intent(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = _envelope(key="9" * 64, attempt_id="a-1", fence=7)
    store.try_claim(envelope, tool_name="write_file", args={})

    replay = store.try_claim(
        _envelope(key="9" * 64, attempt_id="a-2", fence=8),
        tool_name="write_file",
        args={},
    )
    assert replay.attempt_id == "a-1"
    assert replay.state is MutationState.IN_PROGRESS


def test_attempt_id_cannot_move_to_another_fence(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    store.try_claim(_envelope(), tool_name="write_file", args={})
    with pytest.raises(MutationClaimError, match="cannot be reused"):
        store.try_claim(
            _envelope(fence=2), tool_name="write_file", args={}
        )


@pytest.mark.platforms("posix")
def test_claim_database_files_are_owner_only(tmp_path):
    db = tmp_path / "phase2.db"
    store = MutationClaimStore(db)
    conn = store._connect()
    try:
        sidecars = [tmp_path / f"{db.name}{suffix}" for suffix in ("", "-wal", "-shm")]
        assert all(path.exists() for path in sidecars)
        for path in sidecars:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        conn.close()


def test_claim_reader_does_not_require_write_access(tmp_path):
    db = tmp_path / "phase2.db"
    store = MutationClaimStore(db)
    store.try_claim(_envelope(key="a" * 64), tool_name="write_file", args={})
    db.chmod(0o400)

    assert store.read_all()[0]["idempotency_key"] == "a" * 64


def test_claim_database_rejects_last_component_symlink(tmp_path):
    real = tmp_path / "real.db"
    real.write_bytes(b"")
    link = tmp_path / "phase2.db"
    link.symlink_to(real)
    with pytest.raises(PermissionError, match="not a regular file"):
        MutationClaimStore(link)._connect()


def test_legacy_boolean_claim_schema_requires_explicit_migration(tmp_path):
    db = tmp_path / "phase2.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE mutation_claims (idempotency_key TEXT PRIMARY KEY)"
        )

    with pytest.raises(MutationClaimError, match="explicit migration"):
        MutationClaimStore(db)._connect()
