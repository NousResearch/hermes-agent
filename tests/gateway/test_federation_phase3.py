"""Tests for federation Phase 3 — consensus + relay."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.federation.federation_consensus import (
    ConsensusState,
    FederationConsensus,
)
from gateway.federation.federation_relay import (
    TaskCheckpoint,
    TaskExecutionState,
    TaskExecutorRelay,
)


# ========================================================================
# Consensus tests
# ========================================================================


class TestFederationConsensus:
    """Raft-lite consensus for task claiming."""

    def test_init(self):
        c = FederationConsensus(
            device_id="dev-a",
            total_peers=3,
            vote_timeout=1.0,
        )
        assert c.device_id == "dev-a"
        assert c.total_peers == 3
        assert c.active_round_count == 0

    def test_handle_claim_request_no_conflict(self):
        """When task not local, vote ACK."""
        c = FederationConsensus(device_id="dev-a", total_peers=3)
        msg = MagicMock()
        msg.payload = {"task_id": "T-001"}
        msg.sender_id = "dev-b"

        response = c.handle_claim_request(msg)
        assert response.msg_type == "task_claim_ack"
        assert response.payload["vote"] is True

    def test_handle_claim_request_conflict(self):
        """When task already claimed locally, vote NACK."""
        c = FederationConsensus(device_id="dev-a", total_peers=3)
        c._pending_votes["T-001"] = {"dev-a": True}  # We already claimed

        msg = MagicMock()
        msg.payload = {"task_id": "T-001"}
        msg.sender_id = "dev-b"

        response = c.handle_claim_request(msg)
        assert response.msg_type == "task_claim_nack"
        assert response.payload["vote"] is False

    def test_handle_vote_response(self):
        c = FederationConsensus(device_id="dev-a", total_peers=3)
        c._pending_votes["T-001"] = {"dev-a": True}

        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {"task_id": "T-001", "vote": True}

        c.handle_vote_response(msg)
        assert c._pending_votes["T-001"]["dev-b"] is True

    def test_pending_votes_tally(self):
        c = FederationConsensus(device_id="dev-a", total_peers=3)
        c._pending_votes["T-001"] = {
            "dev-a": True,
            "dev-b": True,
            "dev-c": False,
        }
        votes = c.get_pending_votes("T-001")
        assert votes == {"dev-a": True, "dev-b": True, "dev-c": False}

    @pytest.mark.asyncio
    async def test_initiate_claim_majority_accepts(self):
        """Claim should succeed when majority votes ACK."""
        c = FederationConsensus(
            device_id="dev-a",
            total_peers=3,
            vote_timeout=0.5,
        )
        # Pre-populate votes (simulating peer responses)
        c._pending_votes["T-001"] = {
            "dev-a": True,  # self-vote
            "dev-b": True,  # peer ACK
            "dev-c": False, # peer NACK
        }

        result = await c.initiate_claim("T-001")
        assert result is True  # 2/3 = majority

    @pytest.mark.asyncio
    async def test_initiate_claim_majority_rejects(self):
        """Claim should fail when majority votes NACK."""
        c = FederationConsensus(
            device_id="dev-a",
            total_peers=3,
            vote_timeout=0.5,
        )
        c._pending_votes["T-001"] = {
            "dev-a": True,   # self-vote
            "dev-b": False,  # peer NACK
            "dev-c": False,  # peer NACK
        }

        result = await c.initiate_claim("T-001")
        assert result is False  # 1/3 < majority

    def test_consensus_state(self):
        state = ConsensusState(
            task_id="T-001",
            claimer_id="dev-a",
        )
        assert state.task_id == "T-001"
        assert state.claimer_id == "dev-a"
        assert not state.resolved
        assert state.result is None


# ========================================================================
# Relay tests
# ========================================================================


class TestTaskCheckpoint:
    def test_create_checkpoint(self):
        cp = TaskCheckpoint(
            task_id="T-001",
            executor_device="dev-a",
            checkpoint_id="cp-001",
            current_step="Processing batch 3",
            step_index=3,
            total_steps=10,
            progress=0.3,
        )
        assert cp.task_id == "T-001"
        assert cp.progress == 0.3
        assert cp.step_index == 3


class TestTaskExecutionState:
    def test_initial_state(self):
        state = TaskExecutionState(
            task_id="T-001",
            title="Test task",
        )
        assert state.status == "pending"
        assert state.progress == 0.0
        assert state.relay_count == 0
        assert state.previous_executors == []


class TestTaskExecutorRelay:
    """Task execution with checkpoint/relay support."""

    def _make_relay(self, device_id="dev-a"):
        adapter = MagicMock()
        adapter.report_progress = AsyncMock(return_value=True)
        adapter.report_result = AsyncMock(return_value=True)
        adapter.submit_task = AsyncMock(return_value=True)
        adapter.send_task_heartbeat = AsyncMock(return_value=True)

        consensus = MagicMock()
        consensus.initiate_claim = AsyncMock(return_value=True)

        relay = TaskExecutorRelay(
            device_id=device_id,
            adapter=adapter,
            consensus=consensus,
            progress_interval=0.1,
            checkpoint_interval=0.1,
        )
        return relay

    @pytest.mark.asyncio
    async def test_claim_and_execute_success(self):
        relay = self._make_relay()
        await relay.start()

        async def dummy_handler(state):
            state.progress = 1.0
            return {"result": "done"}

        result = await relay.claim_and_execute(
            task_id="T-001",
            title="Test task",
            handler=dummy_handler,
        )

        assert result.status == "completed"
        assert result.result_data == {"result": "done"}

    @pytest.mark.asyncio
    async def test_claim_rejected(self):
        relay = self._make_relay()
        relay.consensus.initiate_claim = AsyncMock(return_value=False)

        result = await relay.claim_and_execute(
            task_id="T-001",
            title="Test task",
        )

        assert result.status == "failed"
        assert "rejected" in result.error_info.lower()

    @pytest.mark.asyncio
    async def test_handler_exception(self):
        relay = self._make_relay()

        async def failing_handler(state):
            raise RuntimeError("Test error")

        result = await relay.claim_and_execute(
            task_id="T-001",
            title="Test task",
            handler=failing_handler,
        )

        assert result.status == "failed"
        assert "Test error" in result.error_info

    @pytest.mark.asyncio
    async def test_handoff_task(self):
        relay = self._make_relay()
        await relay.start()

        # Start a task
        state = await relay.claim_and_execute(
            task_id="T-001",
            title="Test task",
        )
        # The default handler completes quickly, so let's manually add to active
        from gateway.federation.federation_relay import TaskExecutionState
        relay._active_tasks["T-002"] = TaskExecutionState(
            task_id="T-002",
            title="Long task",
            progress=0.5,
            current_step="Step 5",
        )

        # Handoff
        handed_off = await relay.handoff_task("T-002", reason="going offline")
        assert handed_off is not None
        assert handed_off.relay_count == 1
        assert "dev-a" in handed_off.previous_executors

        # Verify adapter.submit_task was called with relay context
        relay.adapter.submit_task.assert_called_once()
        call_kwargs = relay.adapter.submit_task.call_args
        assert call_kwargs.kwargs["task_id"] == "T-002"
        assert "RELAY" in call_kwargs.kwargs["title"]

    @pytest.mark.asyncio
    async def test_active_task_tracking(self):
        relay = self._make_relay()
        await relay.start()

        # Manually add tasks
        from gateway.federation.federation_relay import TaskExecutionState
        relay._active_tasks["T-001"] = TaskExecutionState(
            task_id="T-001", title="Task 1", status="in_progress",
        )
        relay._active_tasks["T-002"] = TaskExecutionState(
            task_id="T-002", title="Task 2", status="in_progress",
        )

        assert relay.active_task_count == 2
        assert relay.get_active_task("T-001") is not None
        assert relay.get_active_task("T-999") is None
        assert len(relay.get_all_active_tasks()) == 2

    @pytest.mark.asyncio
    async def test_stop_hands_off_all_tasks(self):
        relay = self._make_relay()
        await relay.start()

        from gateway.federation.federation_relay import TaskExecutionState
        relay._active_tasks["T-001"] = TaskExecutionState(
            task_id="T-001", title="Task 1", status="in_progress",
        )

        await relay.stop()

        assert relay.active_task_count == 0
        relay.adapter.submit_task.assert_called_once()

    def test_register_handler(self):
        relay = self._make_relay()

        async def my_handler(state):
            return {"custom": True}

        relay.register_handler("T-special", my_handler)
        assert relay._task_handlers["T-special"] == my_handler
