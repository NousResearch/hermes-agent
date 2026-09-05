"""Durable authority store: seal, grant, lifecycle, budget, and chain contracts."""

from __future__ import annotations

import hashlib
import sqlite3
import stat
import sys
import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent.phase2_authority import (
    AuthorityError,
    AuthorityMigrationRequired,
    MalformedEnvelopeFence,
    Phase2AuthorityStore,
    ResultRejection,
    bind_sealed_envelope,
    current_authoritative_fence,
    current_sealed_envelope,
    validate_sealed_envelope,
)


def _node(node_id: str, surface: str = "local_tool") -> dict:
    return {
        "node_id": node_id,
        "objective": f"execute {node_id}",
        "idempotency_key": hashlib.sha256(node_id.encode()).hexdigest(),
        "execution_surface": surface,
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": "2030-01-01T00:00:00+00:00",
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["tool_exec"], "min_receipts": 1},
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "input_hash": "c" * 64,
    }


def _plan(*nodes: dict) -> dict:
    return {
        "contract_version": 2,
        "graph_id": "g-authority",
        "nodes": list(nodes or (_node("n-tool"),)),
    }


def _overdeep_json_value() -> list:
    value: list = []
    for _ in range(sys.getrecursionlimit() + 100):
        value = [value]
    return value


# ── Seal and grant ────────────────────────────────────────────────────────────


def test_seal_and_grant_produces_full_nested_v2_envelope(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    planner_hash = store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:session-1",
        ttl_s=300,
        attempt_id="at-1",
        now=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )

    assert len(planner_hash) == 64
    assert envelope["envelope_version"] == 2
    assert envelope["graph_id"] == "g-authority"
    assert envelope["node_id"] == "n-tool"
    assert envelope["execution_surface"] == "local_tool"
    assert envelope["planner_hash"] == planner_hash
    assert envelope["policy_hash"] == "d" * 64
    assert envelope["permissions"]["read"] == ["/tmp"]
    assert envelope["lease"]["holder"] == "hermes:session-1"
    assert envelope["fence"] == 1
    assert validate_sealed_envelope(envelope) == []


def test_grant_rejects_unrepresentable_finite_ttl_with_typed_error(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)

    with pytest.raises(AuthorityError, match="representable timestamp range"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:session-1",
            ttl_s=1e300,
            attempt_id="at-1",
            now=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )

    assert [event["kind"] for event in store.read_events("g-authority")] == [
        "PLAN_SEALED"
    ]


def test_model_and_tool_work_are_distinct_nodes_and_envelopes(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(
        _plan(_node("n-model", "direct_model"), _node("n-tool", "local_tool")),
        policy_hash="d" * 64,
    )
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    model = store.grant_node(
        "g-authority",
        "n-model",
        holder="hermes:model",
        ttl_s=60,
        attempt_id="at-model",
        now=now,
    )
    tool = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:tool",
        ttl_s=60,
        attempt_id="at-tool",
        now=now,
    )

    assert model["node_id"] != tool["node_id"]
    assert model["execution_surface"] == "direct_model"
    assert tool["execution_surface"] == "local_tool"


def test_invalid_or_duplicate_plan_nodes_fail_before_persistence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    duplicate = _plan(_node("n-tool"), _node("n-tool"))
    with pytest.raises(AuthorityError, match="duplicate node_id"):
        store.seal_plan(duplicate, policy_hash="d" * 64)
    bad_surface = _plan(_node("n-bad", "local_tool_alias"))
    with pytest.raises(AuthorityError, match="execution_surface"):
        store.seal_plan(bad_surface, policy_hash="d" * 64)
    assert store.read_events("g-authority") == []


def test_sealed_plan_rejects_non_finite_numbers(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    inf_node = _node("n-inf")
    inf_node["budgets"] = {
        "usd_max": float("inf"),
        "tokens_max": 1000,
        "wall_clock_s_max": 60,
    }
    with pytest.raises(AuthorityError, match="finite"):
        store.seal_plan(_plan(inf_node), policy_hash="d" * 64)


def test_sealed_plan_rejects_non_string_object_keys(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    node = _node("n-invalid-key")
    node["permissions"][1] = []

    with pytest.raises(AuthorityError, match="JSON"):
        store.seal_plan(_plan(node), policy_hash="d" * 64)


def test_sealed_plan_rejects_invalid_hashes_and_authority_owned_fields(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    invalid_hash = _node("n-invalid-hash")
    invalid_hash["input_hash"] = "not-a-digest"
    with pytest.raises(AuthorityError, match="input_hash"):
        store.seal_plan(_plan(invalid_hash), policy_hash="d" * 64)

    injected = _node("n-injected")
    injected["graph_id"] = "g-attacker"
    with pytest.raises(AuthorityError, match="authority-owned fields"):
        store.seal_plan(_plan(injected), policy_hash="d" * 64)

    incomplete = _node("n-incomplete")
    incomplete["lane"] = "untrusted-lane"
    with pytest.raises(AuthorityError, match="lane"):
        store.seal_plan(_plan(incomplete), policy_hash="d" * 64)

    oversized = _node("n-oversized")
    oversized["budgets"]["tokens_max"] = 10**1000
    with pytest.raises(AuthorityError, match="budgets.tokens_max"):
        store.seal_plan(_plan(oversized), policy_hash="d" * 64)

    with pytest.raises(AuthorityError, match="policy_hash"):
        store.seal_plan(_plan(_node("n-tool")), policy_hash="D" * 64)
    assert store.read_events("g-authority") == []


def test_validate_current_reports_non_object_lease_without_raising(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    malformed = {
        "graph_id": "g-authority",
        "node_id": "n-tool",
        "lease": "not-an-object",
    }

    errors = store.validate_current(malformed)

    assert "lease" in errors
    assert "lease_or_deadline_invalid" in errors


def test_validate_current_reports_unrepresentable_finite_ttl(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    malformed = {
        "graph_id": "g-authority",
        "node_id": "n-tool",
        "deadline_utc": "2030-01-01T00:00:00+00:00",
        "lease": {
            "granted_utc": "2026-07-30T00:00:00+00:00",
            "ttl_s": 1e300,
        },
    }

    errors = store.validate_current(
        malformed, now=datetime(2026, 7, 30, tzinfo=timezone.utc)
    )

    assert "lease_or_deadline_invalid" in errors


def test_authority_database_files_are_owner_only(tmp_path):
    db = tmp_path / "authority.db"
    store = Phase2AuthorityStore(db)
    conn = store._connect()
    try:
        sidecars = [tmp_path / f"{db.name}{suffix}" for suffix in ("", "-wal", "-shm")]
        assert all(path.exists() for path in sidecars)
        for path in sidecars:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        conn.close()


def test_authority_readers_do_not_require_write_access(tmp_path):
    db = tmp_path / "authority.db"
    store = Phase2AuthorityStore(db)
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    db.chmod(0o400)

    assert store.migration_report()["compatible"] is True
    assert store.read_events("g-authority")[0]["kind"] == "PLAN_SEALED"


def test_authority_database_rejects_last_component_symlink(tmp_path):
    real = tmp_path / "real.db"
    real.write_bytes(b"")
    link = tmp_path / "authority.db"
    link.symlink_to(real)
    with pytest.raises(PermissionError, match="not a regular file"):
        Phase2AuthorityStore(link)._connect()


def test_grant_must_fit_strictly_before_node_deadline(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    node = _node("n-tool")
    node["deadline_utc"] = "2026-07-30T00:01:00+00:00"
    store.seal_plan(_plan(node), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)

    with pytest.raises(AuthorityError, match="strictly before"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:one",
            ttl_s=60,
            attempt_id="at-1",
            now=now,
        )
    with pytest.raises(AuthorityError, match="expired deadline"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:one",
            ttl_s=1,
            attempt_id="at-1",
            now=now + timedelta(seconds=60),
        )


def test_sealed_plan_is_immutable_and_reseal_is_rejected(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    plan = _plan(_node("n-tool"))
    first = store.seal_plan(plan, policy_hash="d" * 64)
    with pytest.raises(AuthorityError, match="already sealed"):
        store.seal_plan(plan, policy_hash="d" * 64)
    reopened = Phase2AuthorityStore(tmp_path / "authority.db")
    assert reopened.get_planner_hash("g-authority") == first


# ── Lease lifecycle ───────────────────────────────────────────────────────────


def test_active_lease_blocks_regrant_and_expired_regrant_increments_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    first = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=30,
        attempt_id="at-1",
        now=now,
    )
    with pytest.raises(AuthorityError, match="active lease"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:two",
            ttl_s=30,
            attempt_id="at-2",
            now=now,
        )

    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=30,
        attempt_id="at-2",
        now=now + timedelta(seconds=31),
    )
    assert first["fence"] == 1
    assert second["fence"] == 2
    assert store.current_fence("g-authority", "n-tool") == 2
    assert "stale_fence" in store.validate_current(
        first, now=now + timedelta(seconds=31)
    )
    assert store.validate_current(second, now=now + timedelta(seconds=31)) == []


def test_attempt_id_is_globally_unique_across_nodes(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-a"), _node("n-b")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    store.grant_node(
        "g-authority", "n-a", holder="h", ttl_s=60, attempt_id="uniq-1", now=now
    )
    with pytest.raises(AuthorityError, match="already used"):
        store.grant_node(
            "g-authority", "n-b", holder="h", ttl_s=60, attempt_id="uniq-1", now=now
        )


def test_revoke_allows_immediate_regrant_at_next_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    store.revoke_node(envelope, now=now)
    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=60,
        attempt_id="at-2",
        now=now,
    )
    assert second["fence"] == 2
    # After revocation and regrant at fence 2, the fence-1 envelope is stale;
    # validate_current returns stale_fence (the authority has moved to fence 2).
    errors = store.validate_current(envelope, now=now)
    assert "stale_fence" in errors


def test_renewal_extends_effective_expiry_and_envelope_stays_immutable(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=30,
        attempt_id="at-1",
        now=now,
    )
    result = store.renew_node(envelope, ttl_s=120, now=now + timedelta(seconds=20))
    # Envelope is still the original sealed envelope, not a rewritten copy.
    assert result["envelope"] is not envelope
    assert result["envelope"]["fence"] == 1
    assert result["envelope"]["lease"]["ttl_s"] == 30.0
    # Effective expiry was extended.
    assert "lease_expired" not in store.validate_current(
        envelope, now=now + timedelta(seconds=50)
    )


def test_renewal_rejects_unrepresentable_finite_ttl_with_typed_error(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=30,
        attempt_id="at-1",
        now=now,
    )

    with pytest.raises(AuthorityError, match="representable timestamp range"):
        store.renew_node(envelope, ttl_s=1e300, now=now + timedelta(seconds=20))

    assert [event["kind"] for event in store.read_events("g-authority")] == [
        "PLAN_SEALED",
        "LEASE_GRANTED",
    ]


def test_terminal_node_cannot_be_regranted(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    result_hash = "e" * 64
    store.complete_node(envelope, result_hash=result_hash, now=now)
    with pytest.raises(AuthorityError, match="terminal"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:two",
            ttl_s=60,
            attempt_id="at-2",
            now=now,
        )
    assert store.validate_current(envelope, now=now) != []


# ── Fence binding ─────────────────────────────────────────────────────────────


def test_bind_current_node_sources_authoritative_fence_from_durable_store(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    with store.bind_current(envelope, now=now):
        assert (
            dict(current_sealed_envelope() or {})["attempt_id"]
            == envelope["attempt_id"]
        )
        assert current_sealed_envelope()["permissions"]["read"] == ("/tmp",)
        assert current_authoritative_fence() == 1

    assert current_sealed_envelope() is None
    assert current_authoritative_fence() is None


def test_bind_current_never_pairs_stale_envelope_with_newer_fence(tmp_path):
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)

    class InterleavingStore(Phase2AuthorityStore):
        interleaved = False

        def _validate_current_snapshot(self, conn, candidate, current):
            result = super()._validate_current_snapshot(conn, candidate, current)
            if not self.interleaved:
                self.interleaved = True
                writer = Phase2AuthorityStore(self.db_path)
                writer.revoke_node(candidate, now=current + timedelta(seconds=1))
                writer.grant_node(
                    "g-authority",
                    "n-tool",
                    holder="hermes:two",
                    ttl_s=60,
                    attempt_id="at-2",
                    now=current + timedelta(seconds=2),
                )
            return result

    store = InterleavingStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    with store.bind_current(envelope, now=now):
        assert current_sealed_envelope()["attempt_id"] == "at-1"
        assert current_authoritative_fence() == 1
        assert store.current_fence("g-authority", "n-tool") == 2

    assert current_sealed_envelope() is None
    assert current_authoritative_fence() is None


def test_bind_current_publishes_the_exact_validated_envelope_snapshot(tmp_path):
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)

    class MutatingStore(Phase2AuthorityStore):
        def _validate_current_snapshot(self, conn, candidate, current):
            result = super()._validate_current_snapshot(conn, candidate, current)
            envelope["attempt_id"] = "mutated-after-validation"
            envelope["permissions"]["read"].append("/stolen")
            return result

    store = MutatingStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    with store.bind_current(envelope, now=now):
        assert current_sealed_envelope()["attempt_id"] == "at-1"
        assert current_sealed_envelope()["permissions"]["read"] == ("/tmp",)
        assert current_authoritative_fence() == 1


def test_bind_current_rejects_stale_envelope_before_context_is_published(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    stale = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=1,
        attempt_id="at-1",
        now=now,
    )
    store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=60,
        attempt_id="at-2",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(AuthorityError, match="stale_fence"):
        with store.bind_current(stale, now=now + timedelta(seconds=2)):
            raise AssertionError("stale authority must not publish a context")

    assert current_sealed_envelope() is None


# ── Concurrency ───────────────────────────────────────────────────────────────


def test_concurrent_grant_has_exactly_one_winner(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    barrier = threading.Barrier(8)
    winners: list[dict] = []
    failures: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait()
        try:
            envelope = store.grant_node(
                "g-authority",
                "n-tool",
                holder=f"hermes:{index}",
                ttl_s=60,
                attempt_id=f"at-{index}",
                now=now,
            )
            with lock:
                winners.append(envelope)
        except AuthorityError as exc:
            with lock:
                failures.append(str(exc))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert len(failures) == 7
    assert winners[0]["fence"] == 1


# ── Exactly-once result acceptance ───────────────────────────────────────────


def test_complete_node_accepts_exactly_once_and_idempotent_redelivery_is_safe(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    result_hash = "f" * 64

    first = store.complete_node(envelope, result_hash=result_hash, now=now)
    assert first["result_hash"] == result_hash
    assert first.get("idempotent") is None

    # Same result_hash is idempotent — no new event, no exception.
    second = store.complete_node(envelope, result_hash=result_hash, now=now)
    assert second["idempotent"] is True

    # Different result_hash is rejected.
    with pytest.raises(ResultRejection, match="node_completed"):
        store.complete_node(envelope, result_hash="a" * 64, now=now)


def test_stale_fence_completion_appends_rejection_event(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    stale = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=1,
        attempt_id="at-1",
        now=now,
    )
    store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=60,
        attempt_id="at-2",
        now=now + timedelta(seconds=2),
    )
    with pytest.raises(ResultRejection, match="stale_fence"):
        store.complete_node(stale, result_hash="b" * 64, now=now + timedelta(seconds=2))
    events = store.read_events("g-authority")
    kinds = [e["kind"] for e in events]
    assert "RESULT_REJECTED" in kinds


def test_malformed_fence_raises_typed_exception_and_records_null_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    bad_envelope = dict(envelope)
    bad_envelope["fence"] = "not-an-int"
    with pytest.raises(MalformedEnvelopeFence) as exc_info:
        store.complete_node(bad_envelope, result_hash="c" * 64, now=now)
    assert exc_info.value.claimed_fence_reason == "not_an_integer"
    # Rejection event must have NULL fence in the column.
    events = store.read_events("g-authority")
    rejected = [e for e in events if e["kind"] == "RESULT_REJECTED"]
    assert len(rejected) == 1
    assert rejected[0]["fence"] is None


def test_out_of_range_fence_is_malformed(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    bad_envelope = dict(envelope)
    bad_envelope["fence"] = 2**63  # one beyond signed 64-bit max
    with pytest.raises(MalformedEnvelopeFence) as exc_info:
        store.complete_node(bad_envelope, result_hash="d" * 64, now=now)
    assert exc_info.value.claimed_fence_reason == "out_of_range"


def test_hostile_fence_repr_cannot_abort_rejection_audit(tmp_path):
    class HostileFence:
        def __repr__(self):
            raise RuntimeError("repr must not escape")

    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    bad_envelope = dict(envelope)
    bad_envelope["fence"] = HostileFence()

    with pytest.raises(MalformedEnvelopeFence) as exc_info:
        store.complete_node(bad_envelope, result_hash="e" * 64, now=now)

    assert exc_info.value.claimed_fence_repr == "<unrepresentable HostileFence>"
    rejected = [
        event
        for event in store.read_events("g-authority")
        if event["kind"] == "RESULT_REJECTED"
    ]
    assert rejected[-1]["fence"] is None
    assert (
        rejected[-1]["payload"]["claimed_fence_repr"]
        == "<unrepresentable HostileFence>"
    )


def test_hostile_recursive_envelope_is_typed_rejection_and_audited(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    hostile = dict(envelope)
    hostile["untrusted_extension"] = _overdeep_json_value()

    assert "envelope_not_authoritative" in store.validate_current(hostile, now=now)
    with pytest.raises(ResultRejection, match="envelope_not_authoritative"):
        store.complete_node(hostile, result_hash="f" * 64, now=now)

    rejected = [
        event
        for event in store.read_events("g-authority")
        if event["kind"] == "RESULT_REJECTED"
    ]
    assert len(rejected) == 1
    assert rejected[0]["payload"]["reason"] == "envelope_not_authoritative"


def test_bind_current_converts_serialization_failures_to_authority_error(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    for hostile_value in (object(), _overdeep_json_value()):
        hostile = dict(envelope)
        hostile["untrusted_extension"] = hostile_value
        with pytest.raises(AuthorityError, match="envelope must be JSON serializable"):
            with store.bind_current(hostile, now=now):
                pass


# ── Budget reservation and reconciliation ────────────────────────────────────


def test_budget_reservation_reconciliation_is_durable_and_fail_closed(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    reservation = store.reserve_budget(envelope, tokens=600, usd=0.6, now=now)
    with pytest.raises(AuthorityError, match="budget"):
        store.reserve_budget(envelope, tokens=500, usd=0.5, now=now)
    store.reconcile_budget(reservation, actual_tokens=400, actual_usd=0.4, now=now)

    reopened = Phase2AuthorityStore(tmp_path / "authority.db")
    usage = reopened.budget_usage("g-authority", "n-tool", fence=1)
    assert usage == {
        "reserved_tokens": 0,
        "reserved_usd": 0.0,
        "charged_tokens": 400,
        "charged_usd": 0.4,
        "cost_unknown": False,
    }
    # Budget is now free for another reservation.
    reopened.reserve_budget(envelope, tokens=500, usd=0.5, now=now)


def test_unknown_reconciled_cost_poisons_future_reservations(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    reservation = store.reserve_budget(envelope, tokens=100, usd=0.1, now=now)
    store.reconcile_budget(reservation, actual_tokens=80, actual_usd=None, now=now)

    assert store.budget_usage("g-authority", "n-tool", fence=1)["cost_unknown"] is True
    with pytest.raises(AuthorityError, match="unknown"):
        store.reserve_budget(envelope, tokens=1, usd=0.01, now=now)


def test_node_budget_is_cumulative_across_regranted_fences(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    first = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    reservation = store.reserve_budget(first, tokens=600, usd=0.6, now=now)
    store.reconcile_budget(reservation, actual_tokens=600, actual_usd=0.6, now=now)
    store.revoke_node(first, now=now + timedelta(seconds=1))
    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=60,
        attempt_id="at-2",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(AuthorityError, match="node budget"):
        store.reserve_budget(
            second,
            tokens=401,
            usd=0.4,
            now=now + timedelta(seconds=2),
        )
    store.reserve_budget(
        second,
        tokens=400,
        usd=0.4,
        now=now + timedelta(seconds=2),
    )
    assert store.budget_usage("g-authority", "n-tool", fence=1) == {
        "reserved_tokens": 0,
        "reserved_usd": 0.0,
        "charged_tokens": 600,
        "charged_usd": 0.6,
        "cost_unknown": False,
    }
    assert store.budget_usage("g-authority", "n-tool", fence=2) == {
        "reserved_tokens": 400,
        "reserved_usd": 0.4,
        "charged_tokens": 0,
        "charged_usd": 0.0,
        "cost_unknown": False,
    }


def test_unknown_cost_poison_survives_regrant(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    first = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    reservation = store.reserve_budget(first, tokens=100, usd=0.1, now=now)
    store.reconcile_budget(reservation, actual_tokens=80, actual_usd=None, now=now)
    store.revoke_node(first, now=now + timedelta(seconds=1))
    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=60,
        attempt_id="at-2",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(AuthorityError, match="unknown"):
        store.reserve_budget(
            second,
            tokens=1,
            usd=0.01,
            now=now + timedelta(seconds=2),
        )


def test_outstanding_budget_survives_expiry_regrant_and_bounds_allocations(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    first = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=1,
        attempt_id="at-1",
        now=now,
    )
    store.reserve_budget(first, tokens=600, usd=0.6, now=now)
    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=60,
        attempt_id="at-2",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(AuthorityError, match="node budget"):
        store.reserve_budget_allocations(
            second,
            [{"tokens": 401, "usd": 0.4}],
            now=now + timedelta(seconds=2),
        )
    reservation_ids = store.reserve_budget_allocations(
        second,
        [{"tokens": 400, "usd": 0.4}],
        now=now + timedelta(seconds=2),
    )
    assert len(reservation_ids) == 1


def test_budget_decimal_boundary_is_exact(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    node = _node("n-tool")
    node["budgets"]["usd_max"] = 0.3
    store.seal_plan(_plan(node), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    store.reserve_budget(envelope, tokens=0, usd=0.1, now=now)
    store.reserve_budget(envelope, tokens=0, usd=0.2, now=now)
    with pytest.raises(AuthorityError, match="USD budget"):
        store.reserve_budget(envelope, tokens=0, usd=1e-13, now=now)


# ── Hash-chain integrity and recovery ─────────────────────────────────────────


def test_budget_decimal_preserves_large_integer_boundary(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    node = _node("n-tool")
    node["budgets"]["usd_max"] = 2**53 + 1
    store.seal_plan(_plan(node), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    store.reserve_budget(envelope, tokens=0, usd=2**53, now=now)
    store.reserve_budget(envelope, tokens=0, usd=1, now=now)
    with pytest.raises(AuthorityError, match="USD budget"):
        store.reserve_budget(envelope, tokens=0, usd=1, now=now)


def test_authority_startup_rejects_tampered_event_chain(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER authority_events_no_update")
        conn.execute("UPDATE authority_events SET payload_json = '{}' WHERE id = 1")

    with pytest.raises(AuthorityError, match="hash chain"):
        Phase2AuthorityStore(path).recover()


def test_recovery_reconciles_orphaned_budget_reservation_exactly_once(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=1,
        attempt_id="at-1",
        now=now,
    )
    reservation = store.reserve_budget(envelope, tokens=100, usd=0.1, now=now)

    recovered = Phase2AuthorityStore(path)
    result = recovered.recover(now=now + timedelta(seconds=2))
    assert result["orphaned_reservations_reconciled"] == 1
    assert recovered.budget_usage("g-authority", "n-tool", fence=1) == {
        "reserved_tokens": 0,
        "reserved_usd": 0.0,
        "charged_tokens": 0,
        "charged_usd": 0.0,
        "cost_unknown": True,
    }
    assert (
        recovered.recover(now=now + timedelta(seconds=3))[
            "orphaned_reservations_reconciled"
        ]
        == 0
    )
    with pytest.raises(AuthorityError, match="already reconciled"):
        recovered.reconcile_budget(reservation, actual_tokens=1, actual_usd=0.01)


# ── Migration guard ───────────────────────────────────────────────────────────


def test_migration_report_is_read_only_and_describes_violations(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )
    store.complete_node(
        store.current_authority("g-authority", "n-tool")["envelope"],
        result_hash="a" * 64,
        now=now,
    )

    # A fresh store on the same db must see all indexes and be compatible.
    report = Phase2AuthorityStore(path).migration_report()
    assert report["compatible"] is True
    assert report["missing_indexes"] == []


def test_schema_replaces_same_name_non_unique_integrity_index(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX one_grant_per_attempt")
        conn.execute(
            "CREATE INDEX one_grant_per_attempt ON authority_events(attempt_id)"
        )

    Phase2AuthorityStore(path).current_fence("g-authority", "n-tool")
    with sqlite3.connect(path) as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("one_grant_per_attempt",),
        ).fetchone()[0]
    assert "CREATE UNIQUE INDEX" in sql.upper()
    assert "WHERE kind = 'LEASE_GRANTED'" in sql


# ── Token ceiling exactness (above 2**53 round-trip safety) ──────────────────


def _sealed_high_token_node(node_id: str, tokens_max: int) -> dict:
    node = _node(node_id)
    node["budgets"]["tokens_max"] = tokens_max
    return node


def _grant(store, tokens_max: int):
    """Seal a fresh graph with the given tokens_max and grant the single node."""
    node = _sealed_high_token_node("n-tool", tokens_max)
    plan = {
        "contract_version": 2,
        "graph_id": f"g-tok-{tokens_max}",
        "nodes": [node],
    }
    store.seal_plan(plan, policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    return store.grant_node(
        f"g-tok-{tokens_max}",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id=f"at-{tokens_max}",
        now=now,
    ), now


def test_reserve_budget_enforces_exact_integer_ceiling_round_up(tmp_path):
    """tokens_max just above a float boundary must not round down silently.

    float(9007199254740993) == float(9007199254740992) due to IEEE-754 rounding.
    The ceiling stored in the plan is 9007199254740993; if it were coerced
    through float the enforcement would use 9007199254740992 and grant one extra
    token.  The exact-integer path must refuse the over-budget request.
    """
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    # 2**53 + 1: one above the largest integer exactly representable as float64
    tokens_max = 2**53 + 1
    envelope, now = _grant(store, tokens_max)

    # Reserving exactly tokens_max should succeed.
    store.reserve_budget(envelope, tokens=tokens_max, usd=0.0, now=now)
    # Any further reservation must be refused — no phantom headroom.
    with pytest.raises(AuthorityError, match="token budget"):
        store.reserve_budget(envelope, tokens=1, usd=0.0, now=now)


def test_reserve_budget_enforces_exact_integer_ceiling_round_down(tmp_path):
    """tokens_max just below a float boundary must not round up silently.

    A float-coerced ceiling could be *larger* than the true integer value,
    granting excess headroom.  With tokens_max = 2**53 - 1 the float is exact,
    but with 2**53 + 3 float(2**53 + 3) == float(2**53 + 4) — the ceiling
    would appear one token larger than intended.
    """
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    # 2**53 + 3 rounds up to 2**53 + 4 as float64; exact int must be enforced.
    tokens_max = 2**53 + 3
    envelope, now = _grant(store, tokens_max)

    store.reserve_budget(envelope, tokens=tokens_max, usd=0.0, now=now)
    with pytest.raises(AuthorityError, match="token budget"):
        store.reserve_budget(envelope, tokens=1, usd=0.0, now=now)


def test_reserve_budget_allocations_enforces_exact_integer_ceiling(tmp_path):
    """reserve_budget_allocations must also enforce tokens_max exactly."""
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    tokens_max = 2**53 + 1
    envelope, now = _grant(store, tokens_max)

    store.reserve_budget_allocations(
        envelope,
        [{"tokens": tokens_max, "usd": 0.0}],
        now=now,
    )
    with pytest.raises(AuthorityError, match="token budget"):
        store.reserve_budget_allocations(
            envelope,
            [{"tokens": 1, "usd": 0.0}],
            now=now,
        )


@pytest.mark.parametrize("tokens_max", [1.5, True])
def test_tokens_max_non_integer_is_rejected_by_envelope_validation(tokens_max):
    envelope = {
        "envelope_version": 2,
        "graph_id": "g-tok-invalid",
        **_sealed_high_token_node("n-tool", tokens_max),
        "attempt_id": "at-invalid",
        "planner_hash": "0" * 64,
        "policy_hash": "d" * 64,
        "lease": {
            "holder": "hermes:one",
            "granted_utc": "2026-07-30T00:00:00+00:00",
            "ttl_s": 60,
            "renewable": True,
        },
        "fence": 1,
    }

    assert "budgets.tokens_max" in validate_sealed_envelope(envelope)


@pytest.mark.parametrize("tokens_max", [1.5, True])
def test_tokens_max_non_integer_is_rejected_before_plan_is_sealed(tmp_path, tokens_max):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    node = _node("n-tool")
    node["budgets"]["tokens_max"] = tokens_max
    plan = {
        "contract_version": 2,
        "graph_id": "g-tok-invalid",
        "nodes": [node],
    }

    with pytest.raises(AuthorityError, match="budgets.tokens_max"):
        store.seal_plan(plan, policy_hash="d" * 64)
    assert store.read_events("g-tok-invalid") == []


def test_reservation_metadata_recursion_is_rejected_typed(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    with pytest.raises(AuthorityError, match="not canonically serializable"):
        store.reserve_budget(
            envelope,
            tokens=1,
            usd=0.0,
            metadata={"hostile": _overdeep_json_value()},
            now=now,
        )


def test_allocation_metadata_recursion_is_rejected_typed(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=60,
        attempt_id="at-1",
        now=now,
    )

    with pytest.raises(AuthorityError, match="not canonically serializable"):
        store.reserve_budget_allocations(
            envelope,
            [
                {
                    "tokens": 1,
                    "usd": 0.0,
                    "metadata": {"hostile": _overdeep_json_value()},
                }
            ],
            now=now,
        )


# ── seal_plan duplicate idempotency_key ───────────────────────────────────────


def test_seal_plan_rejects_duplicate_idempotency_key_across_nodes(tmp_path):
    """Two nodes sharing an idempotency_key create cross-node ambiguity and must
    be refused at seal time, before any durable state is written."""
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    shared_key = "a" * 64
    node_a = _node("n-a")
    node_a["idempotency_key"] = shared_key
    node_b = _node("n-b")
    node_b["idempotency_key"] = shared_key
    plan = {
        "contract_version": 2,
        "graph_id": "g-dup-ikey",
        "nodes": [node_a, node_b],
    }

    with pytest.raises(AuthorityError, match="duplicate idempotency_key"):
        store.seal_plan(plan, policy_hash="d" * 64)
    # No events must have been written.
    assert store.read_events("g-dup-ikey") == []
