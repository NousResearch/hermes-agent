"""
tests/test_conflict_locks.py — Live end-to-end concurrency tests.

These tests require real AWS DynamoDB credentials and the hermes-skill-locks
table to exist (provisioned via infra/terraform/dynamodb-skill-locks/main.tf).

Run with:
    NEON_DATABASE_URL='postgresql://...' pytest tests/test_conflict_locks.py -v

Or with explicit AWS profile:
    AWS_PROFILE=hermes-saas pytest tests/test_conflict_locks.py -v

Tests are skipped when:
  - DynamoDB is unreachable (boto3 raises a connection/credentials error)
  - HERMES_SKILL_LOCKS_TABLE table doesn't exist (ResourceNotFoundException)

Design rationale
----------------
These tests simulate the real failure mode: two concurrent agent turns trying
to edit the same team skill.  We use threading.Thread to run both acquire
attempts in parallel and assert the invariant: exactly one succeeds, one gets
False.  This is the concrete proof that DynamoDB conditional writes provide
the distributed mutual exclusion we need.

Note on DynamoDB TTL
--------------------
DynamoDB TTL deletion has up to 48h lag, so tests that rely on TTL auto-cleanup
use explicit release_skill_lock() to clean up.  The TTL path is validated
via unit tests (test_skill_locks.py::TestTTLExpiry) where the behaviour is
simulated deterministically.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def _dynamodb_available() -> tuple[bool, str]:
    """Return (available, reason) for the live DynamoDB table."""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError, EndpointResolutionError
    except ImportError:
        return False, "boto3 not installed"

    table_name = os.environ.get("HERMES_SKILL_LOCKS_TABLE", "hermes-skill-locks")
    try:
        ddb = boto3.resource("dynamodb")
        table = ddb.Table(table_name)
        # A describe call verifies both credentials and table existence.
        table.load()
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


_DDB_AVAILABLE, _DDB_REASON = _dynamodb_available()

requires_dynamodb = pytest.mark.skipif(
    not _DDB_AVAILABLE,
    reason=f"DynamoDB not available: {_DDB_REASON}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_skill_key(suffix: str = "") -> str:
    """Generate a unique skill key for this test run to avoid cross-test collisions."""
    run_id = str(uuid.uuid4())[:8]
    return f"test/integration/skills/conflict-test-{run_id}{suffix}"


def _cleanup_lock(skill_key: str, worker_id: str) -> None:
    """Best-effort lock cleanup after a test."""
    try:
        from tools.skill_locks import release_skill_lock
        release_skill_lock(skill_key, worker_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Live DynamoDB tests
# ---------------------------------------------------------------------------

@requires_dynamodb
class TestLiveAcquireRelease:
    """Basic acquire/release against the real DynamoDB table."""

    def test_acquire_and_release_roundtrip(self):
        """
        Acquire a real DynamoDB lock, verify it returns True, then release.
        The release should succeed without raising.
        """
        from tools.skill_locks import acquire_skill_lock, release_skill_lock

        skill_key = _unique_skill_key()
        worker_id = str(uuid.uuid4())

        acquired = acquire_skill_lock(skill_key, worker_id, ttl_seconds=30)
        assert acquired is True, "Should have acquired the lock on empty table"

        # Release — must not raise
        release_skill_lock(skill_key, worker_id)

    def test_second_acquire_blocked_while_first_held(self):
        """
        Worker A holds the lock. Worker B's acquire returns False immediately.
        After A releases, B's second attempt succeeds.
        """
        from tools.skill_locks import acquire_skill_lock, release_skill_lock

        skill_key = _unique_skill_key()
        worker_a = str(uuid.uuid4())
        worker_b = str(uuid.uuid4())

        # Worker A acquires
        assert acquire_skill_lock(skill_key, worker_a, ttl_seconds=30) is True

        try:
            # Worker B blocked
            assert acquire_skill_lock(skill_key, worker_b, ttl_seconds=30) is False

            # Worker A releases
            release_skill_lock(skill_key, worker_a)

            # Worker B now succeeds
            assert acquire_skill_lock(skill_key, worker_b, ttl_seconds=30) is True
        finally:
            _cleanup_lock(skill_key, worker_b)

    def test_release_with_wrong_worker_id_is_noop(self):
        """
        A worker that doesn't own the lock cannot release it.
        The ConditionalCheckFailedException is swallowed silently.
        """
        from tools.skill_locks import acquire_skill_lock, release_skill_lock

        skill_key = _unique_skill_key()
        real_owner = str(uuid.uuid4())
        interloper = str(uuid.uuid4())

        assert acquire_skill_lock(skill_key, real_owner, ttl_seconds=30) is True
        try:
            # Interloper tries to release a lock it doesn't own — must be a no-op
            release_skill_lock(skill_key, interloper)

            # Real owner should still hold the lock (B's attempt still blocked)
            worker_c = str(uuid.uuid4())
            assert acquire_skill_lock(skill_key, worker_c, ttl_seconds=30) is False
        finally:
            _cleanup_lock(skill_key, real_owner)


@requires_dynamodb
class TestLiveConcurrentWriters:
    """Two concurrent threads simulate two Hermes workers editing the same team skill."""

    def test_exactly_one_of_two_concurrent_workers_wins(self):
        """
        The core invariant: two concurrent threads call acquire_skill_lock on the
        same key.  Exactly one must succeed and the other must get False.
        This validates that DynamoDB conditional writes provide mutual exclusion
        across processes (threads here simulate the multi-worker scenario).
        """
        from tools.skill_locks import acquire_skill_lock, release_skill_lock

        skill_key = _unique_skill_key("-concurrent")
        results: List[bool] = []
        worker_ids: List[str] = [str(uuid.uuid4()), str(uuid.uuid4())]

        errors: List[Exception] = []

        def worker(index: int) -> None:
            try:
                result = acquire_skill_lock(
                    skill_key, worker_ids[index], ttl_seconds=30
                )
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=(0,))
        t2 = threading.Thread(target=worker, args=(1,))

        # Start both threads as close together as possible.
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        if errors:
            pytest.fail(f"Workers raised exceptions: {errors}")

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if r is False)

        assert successes == 1, (
            f"Exactly one worker must win the lock. Got {successes} winners. "
            f"Results: {results}"
        )
        assert failures == 1, (
            f"Exactly one worker must be blocked. Got {failures} losers. "
            f"Results: {results}"
        )

        # Cleanup: release whichever worker won.
        for i, result in enumerate(results):
            # We can't directly map result back to worker index since results
            # list append order isn't deterministic. Try both and let
            # ConditionalCheckFailed be swallowed for the non-owner.
            for wid in worker_ids:
                _cleanup_lock(skill_key, wid)

    def test_ten_workers_exactly_one_wins(self):
        """
        Stress test: 10 concurrent threads, all racing for the same skill key.
        Exactly one must win; the other 9 must get False.
        """
        from tools.skill_locks import acquire_skill_lock

        skill_key = _unique_skill_key("-stress")
        worker_ids = [str(uuid.uuid4()) for _ in range(10)]
        results: List[bool] = []
        errors: List[Exception] = []
        results_lock = threading.Lock()

        def worker(worker_id: str) -> None:
            try:
                result = acquire_skill_lock(skill_key, worker_id, ttl_seconds=60)
                with results_lock:
                    results.append(result)
            except Exception as exc:
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(wid,)) for wid in worker_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        if errors:
            pytest.fail(f"Workers raised exceptions: {errors}")

        successes = sum(1 for r in results if r is True)
        assert successes == 1, (
            f"Exactly one of 10 workers must win. Got {successes} winners. "
            f"Results: {results}"
        )

        # Cleanup all workers (only one holds the lock; others are no-ops)
        for wid in worker_ids:
            _cleanup_lock(skill_key, wid)

    def test_context_manager_releases_on_exception(self):
        """
        team_skill_lock context manager: if the skill-write body raises,
        the lock is still released so the next worker can proceed.
        """
        from tools.skill_locks import team_skill_lock, acquire_skill_lock

        skill_key = _unique_skill_key("-ctx-exc")
        worker_a = str(uuid.uuid4())
        worker_b = str(uuid.uuid4())

        try:
            with team_skill_lock(skill_key, worker_a, ttl_seconds=30):
                raise RuntimeError("simulated write failure")
        except RuntimeError:
            pass  # Expected

        # Lock must have been released by the context manager's finally block.
        # Worker B should now be able to acquire.
        acquired = acquire_skill_lock(skill_key, worker_b, ttl_seconds=30)
        try:
            assert acquired is True, (
                "Worker B should acquire after Worker A's context manager "
                "released the lock in finally."
            )
        finally:
            _cleanup_lock(skill_key, worker_b)


@requires_dynamodb
class TestLiveTTLBehavior:
    """Validate that the TTL field is written correctly (DDB expiry itself is async)."""

    def test_lock_item_has_ttl_in_future(self):
        """The TTL attribute on the lock row must be a Unix epoch in the future."""
        import boto3
        from tools.skill_locks import acquire_skill_lock

        skill_key = _unique_skill_key("-ttl")
        worker_id = str(uuid.uuid4())
        ttl_seconds = 30

        now = int(time.time())
        acquired = acquire_skill_lock(skill_key, worker_id, ttl_seconds=ttl_seconds)
        assert acquired is True

        try:
            # Read the item back and verify the TTL field.
            ddb = boto3.resource("dynamodb")
            table_name = os.environ.get("HERMES_SKILL_LOCKS_TABLE", "hermes-skill-locks")
            table = ddb.Table(table_name)
            response = table.get_item(Key={"skill_key": skill_key})
            item = response.get("Item", {})

            assert "ttl" in item, "Lock item must have a ttl attribute"
            assert int(item["ttl"]) >= now + 20, (
                f"TTL {item['ttl']} should be at least 20s in the future from {now}"
            )
        finally:
            _cleanup_lock(skill_key, worker_id)
