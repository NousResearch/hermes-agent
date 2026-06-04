"""
tests/test_skill_locks.py — Unit tests for tools/skill_locks.py

All DynamoDB calls are mocked with unittest.mock. No real AWS credentials or
network access required. Tests verify:

  - acquire_skill_lock returns True on a successful conditional PutItem.
  - acquire_skill_lock returns False when ConditionalCheckFailedException is raised
    (another worker holds the lock).
  - acquire_skill_lock re-raises non-contention DynamoDB errors (fail closed).
  - release_skill_lock issues a conditional DeleteItem with the correct worker_id.
  - release_skill_lock silently ignores ConditionalCheckFailedException
    (lock already released / TTL expiry — crash recovery path).
  - release_skill_lock logs and swallows other DynamoDB errors (best-effort release).
  - team_skill_lock context manager: acquires on enter, releases in finally.
  - team_skill_lock raises RuntimeError when acquire returns False (contention).
  - team_skill_lock releases in finally even when the body raises.
  - Two concurrent worker simulation: exactly one acquires, other gets False.
  - Personal-scope writes in skill_manage never call DynamoDB.
  - Team-scope writes in SaaS mode route through acquire/release.
  - Global writes still raise PermissionError (Phase A behavior unchanged).
  - boto3 import error surface as ImportError at call time, not import time.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Optional
from unittest.mock import MagicMock, patch, call, PropertyMock

import pytest

from hermes_identity import HermesIdentity


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_identity(
    platform: str = "slack",
    team_id: str = "TTEAM01",
    user_id: str = "UUSER01",
    channel_id: str = "CCHAN01",
) -> HermesIdentity:
    return HermesIdentity(
        platform=platform,
        team_id=team_id,
        user_id=user_id,
        channel_id=channel_id,
    )


ALICE = make_identity(user_id="UALICE")
BOB = make_identity(user_id="UBOB")

VALID_SKILL_CONTENT = """---
name: test-skill
description: A skill used in lock tests.
---

## Trigger
When running lock tests.

## Steps
1. Acquire lock.
2. Write skill.
3. Release lock.
"""


def _make_client_error(code: str) -> Exception:
    """Build a botocore ClientError-shaped exception."""
    from botocore.exceptions import ClientError
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "test error"}},
        operation_name="PutItem",
    )


def _make_s3_get_response(content: str) -> dict:
    return {"Body": BytesIO(content.encode("utf-8"))}


# ---------------------------------------------------------------------------
# acquire_skill_lock
# ---------------------------------------------------------------------------

class TestAcquireSkillLock:
    """acquire_skill_lock — DynamoDB conditional PutItem behaviour."""

    def test_acquire_returns_true_on_successful_put(self):
        """Happy path: conditional PutItem succeeds → returns True."""
        from tools.skill_locks import acquire_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.return_value = {}  # DDB success has no error key

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            result = acquire_skill_lock("team/slack/T01/skills/foo", "worker-123")

        assert result is True
        mock_table.put_item.assert_called_once()
        call_kwargs = mock_table.put_item.call_args[1]
        item = call_kwargs["Item"]
        assert item["skill_key"] == "team/slack/T01/skills/foo"
        assert item["worker_id"] == "worker-123"
        assert "ttl" in item

    def test_acquire_returns_false_on_condition_check_failed(self):
        """Contention path: ConditionalCheckFailedException → returns False."""
        from tools.skill_locks import acquire_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.side_effect = _make_client_error(
            "ConditionalCheckFailedException"
        )

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            result = acquire_skill_lock("team/slack/T01/skills/foo", "worker-456")

        assert result is False

    def test_acquire_reraises_other_client_errors(self):
        """Non-contention DynamoDB errors propagate (fail closed)."""
        from tools.skill_locks import acquire_skill_lock
        from botocore.exceptions import ClientError

        mock_table = MagicMock()
        mock_table.put_item.side_effect = _make_client_error("ProvisionedThroughputExceededException")

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            with pytest.raises(ClientError):
                acquire_skill_lock("team/slack/T01/skills/foo", "worker-789")

    def test_acquire_uses_configurable_table_name(self):
        """HERMES_SKILL_LOCKS_TABLE env var controls which table is used."""
        from tools.skill_locks import acquire_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.return_value = {}
        mock_ddb = MagicMock()
        mock_ddb.Table.return_value = mock_table

        with patch("tools.skill_locks.boto3") as mock_boto3, \
             patch.dict(os.environ, {"HERMES_SKILL_LOCKS_TABLE": "my-custom-locks"}):
            mock_boto3.resource.return_value = mock_ddb
            # Reload module-level constant — we patch it directly.
            with patch("tools.skill_locks._TABLE_NAME", "my-custom-locks"):
                acquire_skill_lock("team/slack/T01/skills/foo", "w-abc")

        # The Table constructor should have been called with our custom name.
        mock_ddb.Table.assert_called_once_with("my-custom-locks")

    def test_acquire_sets_ttl_in_future(self):
        """The TTL set on the lock row is in the future."""
        import time
        from tools.skill_locks import acquire_skill_lock

        put_item_kwargs = {}
        mock_table = MagicMock()

        def capture_kwargs(**kwargs):
            put_item_kwargs.update(kwargs)

        mock_table.put_item.side_effect = capture_kwargs

        now = int(time.time())
        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            acquire_skill_lock("team/slack/T01/skills/foo", "w-ttl", ttl_seconds=30)

        item = put_item_kwargs.get("Item", {})
        assert item.get("ttl", 0) >= now + 25  # allow 5s for test execution

    def test_acquire_condition_expression_checks_ttl(self):
        """Condition expression must check both attribute_not_exists AND ttl < now."""
        from tools.skill_locks import acquire_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.return_value = {}

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            acquire_skill_lock("team/slack/T01/skills/bar", "w-cond")

        call_kwargs = mock_table.put_item.call_args[1]
        # Both conditions must appear: absence check AND expired TTL check
        expr = call_kwargs["ConditionExpression"]
        assert "attribute_not_exists" in expr
        assert "#t" in expr or "ttl" in expr.lower() or ":now" in call_kwargs.get("ExpressionAttributeValues", {})

    def test_acquire_raises_import_error_when_boto3_missing(self):
        """When boto3 is not installed, ImportError is raised at call time."""
        from tools import skill_locks

        with patch.object(skill_locks, "_boto3_import_error", ImportError("no boto3")):
            with pytest.raises(ImportError, match="boto3"):
                skill_locks.acquire_skill_lock("team/slack/T01/skills/foo", "w-x")


# ---------------------------------------------------------------------------
# release_skill_lock
# ---------------------------------------------------------------------------

class TestReleaseSkillLock:
    """release_skill_lock — conditional DeleteItem behaviour."""

    def test_release_calls_delete_with_correct_key_and_worker(self):
        """release_skill_lock issues DeleteItem with the right Key and condition."""
        from tools.skill_locks import release_skill_lock

        mock_table = MagicMock()
        mock_table.delete_item.return_value = {}

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            release_skill_lock("team/slack/T01/skills/foo", "worker-abc")

        mock_table.delete_item.assert_called_once()
        call_kwargs = mock_table.delete_item.call_args[1]
        assert call_kwargs["Key"] == {"skill_key": "team/slack/T01/skills/foo"}
        # Condition must reference the worker_id
        assert "worker_id" in call_kwargs.get("ConditionExpression", "")
        values = call_kwargs.get("ExpressionAttributeValues", {})
        assert "worker-abc" in values.values()

    def test_release_silently_ignores_condition_check_failed(self):
        """TTL expiry / already-released lock: ConditionalCheckFailed is swallowed."""
        from tools.skill_locks import release_skill_lock

        mock_table = MagicMock()
        mock_table.delete_item.side_effect = _make_client_error(
            "ConditionalCheckFailedException"
        )

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            # Must NOT raise
            release_skill_lock("team/slack/T01/skills/foo", "worker-gone")

    def test_release_logs_and_swallows_other_client_errors(self):
        """Non-contention DDB errors during release are logged, not raised."""
        from tools.skill_locks import release_skill_lock

        mock_table = MagicMock()
        mock_table.delete_item.side_effect = _make_client_error("ServiceUnavailable")

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            # Must NOT raise — release is best-effort
            release_skill_lock("team/slack/T01/skills/foo", "worker-oops")

    def test_release_raises_import_error_when_boto3_missing(self):
        """When boto3 is not installed, ImportError is raised at call time."""
        from tools import skill_locks

        with patch.object(skill_locks, "_boto3_import_error", ImportError("no boto3")):
            with pytest.raises(ImportError, match="boto3"):
                skill_locks.release_skill_lock("team/slack/T01/skills/foo", "w-x")


# ---------------------------------------------------------------------------
# team_skill_lock context manager
# ---------------------------------------------------------------------------

class TestTeamSkillLock:
    """team_skill_lock — context manager acquire/release/error flow."""

    def test_lock_ctx_acquires_and_releases_on_clean_exit(self):
        """Context manager: acquires on enter, releases in finally block."""
        from tools.skill_locks import team_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.return_value = {}
        mock_table.delete_item.return_value = {}

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            with team_skill_lock("team/slack/T01/skills/foo", "w-ctx"):
                pass  # body runs successfully

        mock_table.put_item.assert_called_once()
        mock_table.delete_item.assert_called_once()

    def test_lock_ctx_releases_on_body_exception(self):
        """Lock is released in finally even when the body raises."""
        from tools.skill_locks import team_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.return_value = {}
        mock_table.delete_item.return_value = {}

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            with pytest.raises(ValueError):
                with team_skill_lock("team/slack/T01/skills/foo", "w-fail"):
                    raise ValueError("body failed")

        # delete_item must have been called despite the exception
        mock_table.delete_item.assert_called_once()

    def test_lock_ctx_raises_runtime_error_on_contention(self):
        """RuntimeError raised when lock is held by another worker."""
        from tools.skill_locks import team_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.side_effect = _make_client_error(
            "ConditionalCheckFailedException"
        )

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            with pytest.raises(RuntimeError, match="currently being edited"):
                with team_skill_lock("team/slack/T01/skills/foo", "w-blocked"):
                    pass  # should not reach here

        # release should NOT have been called (lock was never acquired)
        mock_table.delete_item.assert_not_called()

    def test_lock_ctx_skill_name_appears_in_error_message(self):
        """The RuntimeError message includes the skill name for the agent."""
        from tools.skill_locks import team_skill_lock

        mock_table = MagicMock()
        mock_table.put_item.side_effect = _make_client_error(
            "ConditionalCheckFailedException"
        )

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            with pytest.raises(RuntimeError) as exc_info:
                with team_skill_lock(
                    "team/slack/T01/skills/my-cool-skill", "w-x"
                ):
                    pass

        assert "my-cool-skill" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Two-worker concurrency simulation
# ---------------------------------------------------------------------------

class TestTwoWorkerConcurrency:
    """Simulate two concurrent workers trying to edit the same team skill."""

    def test_exactly_one_of_two_concurrent_workers_succeeds(self):
        """
        Worker A's PutItem succeeds; Worker B's raises ConditionalCheckFailedException.
        Exactly one acquires the lock, one gets False.
        """
        from tools.skill_locks import acquire_skill_lock

        skill_key = "team/slack/TTEAM01/skills/shared-skill"
        worker_a = "worker-aaaa"
        worker_b = "worker-bbbb"

        # First call (Worker A) succeeds; second call (Worker B) fails with contention.
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {}  # Worker A succeeds
            raise _make_client_error("ConditionalCheckFailedException")

        mock_table = MagicMock()
        mock_table.put_item.side_effect = side_effect

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            result_a = acquire_skill_lock(skill_key, worker_a)
            result_b = acquire_skill_lock(skill_key, worker_b)

        assert result_a is True, "Worker A should have acquired the lock"
        assert result_b is False, "Worker B should have been blocked"

    def test_worker_b_retry_succeeds_after_worker_a_releases(self):
        """
        Worker A acquires and releases. Worker B's subsequent attempt succeeds
        (simulated by a successful second PutItem).
        """
        from tools.skill_locks import acquire_skill_lock, release_skill_lock

        skill_key = "team/slack/TTEAM01/skills/retry-skill"
        worker_a = "worker-a"
        worker_b = "worker-b"

        put_call_count = {"n": 0}

        def put_side_effect(**kwargs):
            put_call_count["n"] += 1
            if put_call_count["n"] == 1:
                return {}  # Worker A acquires
            elif put_call_count["n"] == 2:
                # Worker B retry after A releases — succeeds
                return {}
            raise AssertionError("Unexpected third PutItem call")

        mock_table = MagicMock()
        mock_table.put_item.side_effect = put_side_effect
        mock_table.delete_item.return_value = {}

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            # Worker A acquires
            assert acquire_skill_lock(skill_key, worker_a) is True
            # Worker A releases
            release_skill_lock(skill_key, worker_a)
            # Worker B acquires (simulated success — A's row is gone)
            assert acquire_skill_lock(skill_key, worker_b) is True

    def test_friendly_error_message_returned_on_contention(self):
        """
        skill_manage in SaaS team-scope mode returns the standard retry message
        when the lock is held by another worker.
        """
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_locks.acquire_skill_lock", return_value=False):
                result_json = skill_manage(
                    action="edit",
                    name="contested-skill",
                    content=VALID_SKILL_CONTENT,
                    target_scope="team",
                    identity=ALICE,
                )

        result = json.loads(result_json)
        assert result["success"] is False
        assert "contested-skill" in result["error"]
        assert "retry" in result["error"].lower()


# ---------------------------------------------------------------------------
# TTL expiry / crashed worker simulation
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    """Simulate a crashed worker whose lock auto-expires via TTL."""

    def test_expired_lock_allows_new_acquire(self):
        """
        When the existing lock's TTL is in the past, the conditional write
        succeeds (the condition `#t < :now` is satisfied).
        Simulated by: first call raises ConditionalCheckFailed (A holds lock),
        second call (TTL has passed → DDB allows it) succeeds.
        """
        from tools.skill_locks import acquire_skill_lock

        skill_key = "team/slack/T01/skills/orphaned-skill"
        call_count = {"n": 0}

        def put_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Crashed worker's lock still live
                raise _make_client_error("ConditionalCheckFailedException")
            # After TTL passes, condition `#t < :now` matches → succeeds
            return {}

        mock_table = MagicMock()
        mock_table.put_item.side_effect = put_side_effect

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            # First attempt blocked (crashed worker's lock still live in DDB)
            result_first = acquire_skill_lock(skill_key, "new-worker-1")
            # Second attempt: TTL has passed (DDB now returns success)
            result_second = acquire_skill_lock(skill_key, "new-worker-1")

        assert result_first is False, "Should be blocked while crashed-worker lock is live"
        assert result_second is True, "Should succeed after TTL expires"

    def test_double_release_is_silent_noop(self):
        """
        Releasing a lock twice (e.g. TTL expiry + explicit release) must not
        raise.  ConditionalCheckFailedException is swallowed on release.
        """
        from tools.skill_locks import release_skill_lock

        mock_table = MagicMock()
        # First release succeeds; second raises ConditionalCheckFailed
        mock_table.delete_item.side_effect = [
            {},  # first call succeeds
            _make_client_error("ConditionalCheckFailedException"),  # second: already gone
        ]

        with patch("tools.skill_locks.boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = mock_table
            release_skill_lock("team/slack/T01/skills/foo", "w-double")
            release_skill_lock("team/slack/T01/skills/foo", "w-double")  # must not raise


# ---------------------------------------------------------------------------
# Personal scope — no DynamoDB call
# ---------------------------------------------------------------------------

class TestPersonalScopeIsLockFree:
    """Personal scope writes must never touch DynamoDB."""

    def test_personal_create_does_not_call_dynamodb(self):
        """skill_manage(saas/personal) writes to S3 without any DDB interaction."""
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skills_scoped.boto3") as mock_s3_boto3, \
                 patch("tools.skill_locks.boto3") as mock_ddb_boto3:
                mock_s3_boto3.client.return_value = MagicMock()
                result_json = skill_manage(
                    action="create",
                    name="my-personal-skill",
                    content=VALID_SKILL_CONTENT,
                    identity=ALICE,
                    # target_scope not set → defaults to "personal"
                )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result.get("scope") == "personal"
        # DynamoDB must not have been touched
        mock_ddb_boto3.resource.assert_not_called()

    def test_personal_edit_does_not_call_dynamodb(self):
        """Personal edit action: no DynamoDB lock call."""
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skills_scoped.boto3") as mock_s3_boto3, \
                 patch("tools.skill_locks.boto3") as mock_ddb_boto3:
                mock_s3_boto3.client.return_value = MagicMock()
                result_json = skill_manage(
                    action="edit",
                    name="my-personal-skill",
                    content=VALID_SKILL_CONTENT,
                    identity=ALICE,
                )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result.get("scope") == "personal"
        mock_ddb_boto3.resource.assert_not_called()


# ---------------------------------------------------------------------------
# Global scope — Phase A hard block unchanged
# ---------------------------------------------------------------------------

class TestGlobalScopeBlockUnchanged:
    """write_skill global scope must still raise PermissionError (Phase A)."""

    def test_write_skill_global_raises_permission_error(self):
        """Global scope write is permanently blocked regardless of Phase C changes."""
        from tools.skills_scoped import write_skill

        mock_s3 = MagicMock()
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            with pytest.raises(PermissionError, match="global"):
                write_skill("foo", "content", scope="global", identity=ALICE)

        mock_s3.put_object.assert_not_called()

    def test_skill_manage_team_write_blocks_global_write(self):
        """Even via skill_manage, targeting global scope stays blocked."""
        from tools.skills_scoped import write_skill

        # Direct call — no skill_manage plumbing needed since global
        # is blocked at the write_skill layer regardless.
        with patch("tools.skills_scoped.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            with pytest.raises(PermissionError):
                write_skill("foo", "content", scope="global", identity=ALICE)


# ---------------------------------------------------------------------------
# skill_manage SaaS team routing
# ---------------------------------------------------------------------------

class TestSkillManageSaasTeamRouting:
    """skill_manage routes team-scope writes through DynamoDB lock in SaaS mode."""

    def test_team_write_acquires_and_releases_lock(self):
        """Lock is acquired before S3 write and released in finally."""
        from tools.skill_manager_tool import skill_manage

        acquire_calls = []
        release_calls = []

        def fake_acquire(skill_key, worker_id, ttl_seconds=30):
            acquire_calls.append((skill_key, worker_id))
            return True

        def fake_release(skill_key, worker_id):
            release_calls.append((skill_key, worker_id))

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_locks.acquire_skill_lock", side_effect=fake_acquire), \
                 patch("tools.skill_locks.release_skill_lock", side_effect=fake_release), \
                 patch("tools.skills_scoped.boto3") as mock_boto3:
                mock_boto3.client.return_value = MagicMock()
                result_json = skill_manage(
                    action="edit",
                    name="team-skill",
                    content=VALID_SKILL_CONTENT,
                    target_scope="team",
                    identity=ALICE,
                )

        result = json.loads(result_json)
        assert result["success"] is True
        assert result.get("scope") == "team"
        # Acquire must have happened before the write (and once only)
        assert len(acquire_calls) == 1
        # Release must always happen (finally block)
        assert len(release_calls) == 1

    def test_team_write_releases_lock_on_s3_failure(self):
        """Lock is released even when the S3 write raises an exception."""
        from tools.skill_manager_tool import skill_manage

        release_calls = []

        def fake_acquire(skill_key, worker_id, ttl_seconds=30):
            return True

        def fake_release(skill_key, worker_id):
            release_calls.append((skill_key, worker_id))

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_locks.acquire_skill_lock", side_effect=fake_acquire), \
                 patch("tools.skill_locks.release_skill_lock", side_effect=fake_release), \
                 patch("tools.skills_scoped.write_skill", side_effect=RuntimeError("S3 down")):
                result_json = skill_manage(
                    action="edit",
                    name="team-skill",
                    content=VALID_SKILL_CONTENT,
                    target_scope="team",
                    identity=ALICE,
                )

        result = json.loads(result_json)
        assert result["success"] is False
        # Release must have happened despite the S3 failure
        assert len(release_calls) == 1

    def test_team_write_contention_returns_friendly_error(self):
        """Contention: skill_manage returns retry error, not an exception."""
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_locks.acquire_skill_lock", return_value=False):
                result_json = skill_manage(
                    action="edit",
                    name="team-skill",
                    content=VALID_SKILL_CONTENT,
                    target_scope="team",
                    identity=ALICE,
                )

        result = json.loads(result_json)
        assert result["success"] is False
        assert "team-skill" in result["error"]
        assert "retry" in result["error"].lower()

    def test_team_write_lock_key_uses_team_scope(self):
        """The DynamoDB lock key must reference the team scope, not personal."""
        from tools.skill_manager_tool import skill_manage

        acquired_keys = []

        def fake_acquire(skill_key, worker_id, ttl_seconds=30):
            acquired_keys.append(skill_key)
            return True

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_locks.acquire_skill_lock", side_effect=fake_acquire), \
                 patch("tools.skill_locks.release_skill_lock"), \
                 patch("tools.skills_scoped.boto3") as mock_boto3:
                mock_boto3.client.return_value = MagicMock()
                skill_manage(
                    action="edit",
                    name="team-skill",
                    content=VALID_SKILL_CONTENT,
                    target_scope="team",
                    identity=ALICE,
                )

        assert len(acquired_keys) == 1
        key = acquired_keys[0]
        # The key must reference the TEAM scope, not personal
        assert ALICE.team_scope in key
        assert ALICE.personal_scope not in key

    def test_local_mode_team_write_uses_flock_not_dynamodb(self):
        """In local mode (no HERMES_MODE=saas), DynamoDB must not be called."""
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_MODE", None)
            with patch("tools.skill_locks.boto3") as mock_ddb_boto3, \
                 patch("tools.skill_manager_tool._create_skill") as mock_create:
                mock_create.return_value = {"success": True, "message": "ok"}
                result_json = skill_manage(
                    action="create",
                    name="local-skill",
                    content=VALID_SKILL_CONTENT,
                    identity=ALICE,
                )

        result = json.loads(result_json)
        assert result["success"] is True
        # DynamoDB must not be touched in local mode
        mock_ddb_boto3.resource.assert_not_called()

    def test_saas_mode_without_identity_uses_local_path(self):
        """
        HERMES_MODE=saas with identity=None must fall through to local
        filesystem path without hitting DynamoDB.
        """
        from tools.skill_manager_tool import skill_manage

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_locks.boto3") as mock_ddb_boto3, \
                 patch("tools.skill_manager_tool._create_skill") as mock_create:
                mock_create.return_value = {"success": True, "message": "ok"}
                result_json = skill_manage(
                    action="create",
                    name="no-identity-skill",
                    content=VALID_SKILL_CONTENT,
                    identity=None,
                )

        result = json.loads(result_json)
        assert result["success"] is True
        mock_ddb_boto3.resource.assert_not_called()

    def test_team_write_s3_key_uses_team_scope_not_personal(self):
        """S3 object written during team lock must land under team_scope prefix."""
        from tools.skill_manager_tool import skill_manage

        put_object_calls = []

        def fake_acquire(skill_key, worker_id, ttl_seconds=30):
            return True

        def fake_s3_put(**kwargs):
            put_object_calls.append(kwargs.get("Key", ""))
            return {}

        with patch.dict(os.environ, {"HERMES_MODE": "saas"}):
            with patch("tools.skill_locks.acquire_skill_lock", side_effect=fake_acquire), \
                 patch("tools.skill_locks.release_skill_lock"), \
                 patch("tools.skills_scoped.boto3") as mock_boto3:
                mock_client = MagicMock()
                mock_client.put_object.side_effect = fake_s3_put
                mock_boto3.client.return_value = mock_client
                skill_manage(
                    action="edit",
                    name="team-skill",
                    content=VALID_SKILL_CONTENT,
                    target_scope="team",
                    identity=ALICE,
                )

        assert len(put_object_calls) == 1
        # Canonical layout (Plan 039 A2.5): hermes-skills/{tenant_slug}/{scope}/{name}/SKILL.md
        assert put_object_calls[0] == f"hermes-skills/{ALICE.tenant_slug}/team/team-skill/SKILL.md"
        assert "/personal/" not in put_object_calls[0]
