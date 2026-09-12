"""Atomic idempotency claims for Phase 2 mutating tool calls."""

from __future__ import annotations

import sqlite3
import stat
import threading
import pytest

from agent.phase2_idempotency import MutationClaimError, MutationClaimStore


def test_first_mutation_claim_wins_and_duplicate_is_denied(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "d" * 64,
        "fence": 1,
    }

    assert (
        store.try_claim(envelope, tool_name="write_file", args={"path": "one"}) is True
    )
    assert (
        store.try_claim(envelope, tool_name="write_file", args={"path": "one"}) is False
    )


def test_mutation_claim_is_atomic_under_concurrency(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "e" * 64,
        "fence": 4,
    }
    barrier = threading.Barrier(8)
    outcomes: list[bool] = []
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

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7


def test_claim_persists_canonical_identity_without_raw_arguments(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-secret",
        "idempotency_key": "f" * 64,
        "fence": 2,
    }
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
    assert len(claims[0]["args_hash"]) == 64
    assert "do-not-store-this" not in repr(claims)


def test_distinct_idempotency_keys_are_independent(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    base = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "fence": 1,
    }
    env_a = {**base, "idempotency_key": "a" * 64}
    env_b = {**base, "idempotency_key": "b" * 64}

    assert store.try_claim(env_a, tool_name="write_file", args={}) is True
    assert store.try_claim(env_b, tool_name="write_file", args={}) is True
    assert store.try_claim(env_a, tool_name="write_file", args={}) is False
    assert store.try_claim(env_b, tool_name="write_file", args={}) is False
    assert len(store.read_all()) == 2


def test_claim_store_is_append_only_and_rejects_updates(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "c" * 64,
        "fence": 1,
    }
    store.try_claim(envelope, tool_name="delete_file", args={"path": "/tmp/x"})

    with sqlite3.connect(tmp_path / "phase2.db") as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "UPDATE mutation_claims SET tool_name = 'tampered' WHERE idempotency_key = ?",
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
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "a" * 64,
        "fence": 1,
    }
    envelope[field] = value

    with pytest.raises(MutationClaimError, match=message):
        store.try_claim(envelope, tool_name="write_file", args={})
    assert not (tmp_path / "phase2.db").exists()


def test_invalid_tool_and_args_are_rejected_typed(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "a" * 64,
        "fence": 1,
    }
    with pytest.raises(MutationClaimError, match="tool_name"):
        store.try_claim(envelope, tool_name="", args={})
    with pytest.raises(MutationClaimError, match="args must be a mapping"):
        store.try_claim(envelope, tool_name="write_file", args=[])  # type: ignore[arg-type]
    with pytest.raises(MutationClaimError, match="canonically JSON serializable"):
        store.try_claim(envelope, tool_name="write_file", args={"bad": object()})
    with pytest.raises(MutationClaimError, match="canonically JSON serializable"):
        store.try_claim(envelope, tool_name="write_file", args={"bad": float("nan")})
    with pytest.raises(MutationClaimError, match="canonically JSON serializable"):
        store.try_claim(envelope, tool_name="write_file", args={1: "ambiguous"})


def test_non_key_integrity_error_is_not_misclassified_as_duplicate(tmp_path):
    store = MutationClaimStore(tmp_path / "phase2.db")
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "a" * 64,
        "fence": 1,
    }
    conn = store._connect()
    try:
        conn.execute(
            """
            CREATE TRIGGER mutation_claims_reject_tool
            BEFORE INSERT ON mutation_claims
            WHEN NEW.tool_name = 'reject'
            BEGIN
                SELECT RAISE(ABORT, 'tool rejected');
            END
            """
        )
    finally:
        conn.close()

    with pytest.raises(sqlite3.DatabaseError, match="tool rejected"):
        store.try_claim(envelope, tool_name="reject", args={})


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
    envelope = {
        "graph_id": "g-1",
        "node_id": "n-1",
        "attempt_id": "a-1",
        "idempotency_key": "a" * 64,
        "fence": 1,
    }
    store.try_claim(envelope, tool_name="write_file", args={})
    db.chmod(0o400)

    assert store.read_all()[0]["idempotency_key"] == "a" * 64


def test_claim_database_rejects_last_component_symlink(tmp_path):
    real = tmp_path / "real.db"
    real.write_bytes(b"")
    link = tmp_path / "phase2.db"
    link.symlink_to(real)
    with pytest.raises(PermissionError, match="not a regular file"):
        MutationClaimStore(link)._connect()


# ── Typed collision vs same-effect replay ────────────────────────────────────


_BASE_ENVELOPE = {
    "graph_id": "g-col",
    "node_id": "n-col",
    "attempt_id": "a-col",
    "idempotency_key": "9" * 64,
    "fence": 7,
}


def test_same_intent_replay_returns_false(tmp_path):
    """Exact same envelope + tool_name + args is a safe replay: must return False."""
    store = MutationClaimStore(tmp_path / "phase2.db")
    assert store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"k": "v"}) is True
    assert store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"k": "v"}) is False
    # Multiple replays are all False; the store remains consistent.
    assert store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"k": "v"}) is False


@pytest.mark.parametrize(
    ("field", "new_value"),
    [
        ("graph_id", "g-other"),
        ("node_id", "n-other"),
        ("attempt_id", "a-other"),
        ("fence", 8),
    ],
)
def test_changed_envelope_field_raises_collision_error(tmp_path, field, new_value):
    """Changing any identity field on the envelope must raise MutationClaimError."""
    store = MutationClaimStore(tmp_path / "phase2.db")
    store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={})

    collision_env = {**_BASE_ENVELOPE, field: new_value}
    with pytest.raises(MutationClaimError, match="collision"):
        store.try_claim(collision_env, tool_name="write_file", args={})


def test_changed_tool_name_raises_collision_error(tmp_path):
    """Changing tool_name for the same idempotency_key must raise MutationClaimError."""
    store = MutationClaimStore(tmp_path / "phase2.db")
    store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={})

    with pytest.raises(MutationClaimError, match="collision"):
        store.try_claim(_BASE_ENVELOPE, tool_name="delete_file", args={})


def test_changed_args_raises_collision_error(tmp_path):
    """Changing args for the same idempotency_key must raise MutationClaimError."""
    store = MutationClaimStore(tmp_path / "phase2.db")
    store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"path": "one"})

    with pytest.raises(MutationClaimError, match="collision"):
        store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"path": "two"})


def test_collision_does_not_corrupt_store(tmp_path):
    """A MutationClaimError collision must not prevent future reads or correct claims."""
    store = MutationClaimStore(tmp_path / "phase2.db")
    store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"k": "v"})

    with pytest.raises(MutationClaimError, match="collision"):
        store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"k": "changed"})

    # The store is still readable.
    claims = store.read_all()
    assert len(claims) == 1
    assert claims[0]["idempotency_key"] == "9" * 64
    # A new key can still be claimed.
    new_env = {**_BASE_ENVELOPE, "idempotency_key": "8" * 64}
    assert store.try_claim(new_env, tool_name="write_file", args={"k": "v"}) is True


def test_conflict_without_readable_prior_is_not_a_safe_replay(tmp_path, monkeypatch):
    """A conflicted insert without an inspectable row must fail closed."""
    store = MutationClaimStore(tmp_path / "phase2.db")
    store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"k": "v"})
    real_connect = store._connect

    class MissingPriorConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, parameters=()):
            result = self._conn.execute(sql, parameters)
            if "FROM mutation_claims" in sql and sql.lstrip().startswith("SELECT"):
                class MissingPriorCursor:
                    @staticmethod
                    def fetchone():
                        return None

                return MissingPriorCursor()
            return result

        def close(self):
            self._conn.close()

    monkeypatch.setattr(store, "_connect", lambda: MissingPriorConnection(real_connect()))

    with pytest.raises(MutationClaimError, match="without a readable prior claim"):
        store.try_claim(_BASE_ENVELOPE, tool_name="write_file", args={"k": "v"})
