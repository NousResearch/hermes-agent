"""Tests for agent/cancellation_gates.py - safe cancellation gates."""
import asyncio
import pytest
from unittest.mock import MagicMock

from agent.cancellation_gates import (
    OperationCancelled,
    guard_file_write,
    guard_commit,
    guard_push,
    guard_pr,
    guard_external_send,
    guard_deploy,
    guard_db_migration,
    guard_deletion,
    is_at_safe_cancellation_point,
    check_cancelled_or_raise,
    check_cancelled_async,
)
from agent.cancellation import CancellationToken, JobState


def _make_agent(cancelled=False, executing_tools=False, job_id="test-job-123"):
    """Create a mock agent with cancellation token."""
    agent = MagicMock()
    token = CancellationToken()
    if cancelled:
        token.request_cancel()
    agent._cancellation_token = token
    agent._job_id = job_id
    agent._interrupt_requested = False
    agent._executing_tools = executing_tools
    agent._api_call_count = 0
    return agent


class TestGuardFileWrite:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_file_write(agent, "/tmp/test.txt")  # should not raise

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="file_write"):
            guard_file_write(agent, "/tmp/test.txt")

    def test_no_token(self):
        agent = MagicMock()
        agent._cancellation_token = None
        agent._interrupt_requested = False
        guard_file_write(agent, "/tmp/test.txt")  # should not raise


class TestGuardCommit:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_commit(agent)

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="git_commit"):
            guard_commit(agent)


class TestGuardPush:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_push(agent)

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="git_push"):
            guard_push(agent)


class TestGuardPr:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_pr(agent)

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="create_pr"):
            guard_pr(agent)


class TestGuardExternalSend:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_external_send(agent, "discord:#general")

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="external_send"):
            guard_external_send(agent, "discord:#general")


class TestGuardDeploy:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_deploy(agent)

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="deploy"):
            guard_deploy(agent)


class TestGuardDbMigration:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_db_migration(agent, "add_users_table")

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="db_migration"):
            guard_db_migration(agent, "add_users_table")


class TestGuardDeletion:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        guard_deletion(agent, "users_table")

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled, match="deletion"):
            guard_deletion(agent, "users_table")


class TestInterruptAlsoTriggersGate:
    def test_legacy_interrupt_triggers_gate(self):
        agent = _make_agent(cancelled=False)
        agent._interrupt_requested = True  # legacy interrupt
        with pytest.raises(OperationCancelled):
            guard_file_write(agent, "/tmp/test.txt")


class TestSafeCancellationPoint:
    def test_idle_is_safe(self):
        agent = _make_agent(cancelled=False)
        assert is_at_safe_cancellation_point(agent)

    def test_executing_tools_not_safe(self):
        agent = _make_agent(cancelled=False)
        agent._executing_tools = True
        assert not is_at_safe_cancellation_point(agent)


class TestCheckCancelledOrRaise:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        check_cancelled_or_raise(agent)

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(OperationCancelled):
            check_cancelled_or_raise(agent)


class TestCheckCancelledAsync:
    def test_not_cancelled(self):
        agent = _make_agent(cancelled=False)
        asyncio.run(check_cancelled_async(agent))

    def test_cancelled(self):
        agent = _make_agent(cancelled=True)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(check_cancelled_async(agent))
