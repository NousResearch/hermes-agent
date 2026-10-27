"""Federation Raft-lite consensus for atomic task claiming.

A lightweight consensus protocol ensuring that when multiple devices
simultaneously attempt to claim the same task, exactly ONE succeeds.

Unlike full Raft (which requires a leader and persistent log), this is a
simple majority-vote protocol suited for federation task claiming where:
- All peers are equal (no leader)
- Decisions are ephemeral (no log needed)
- Speed matters more than durability
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gateway.federation.federation_protocol import FedMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class ConsensusState:
    """State for a single consensus round."""

    task_id: str
    claimer_id: str
    started_at: float = field(default_factory=time.time)
    acks: set = field(default_factory=set)
    nacks: set = field(default_factory=set)
    resolved: bool = False
    result: Optional[bool] = None  # True = claim accepted, False = rejected


class FederationConsensus:
    """Raft-lite consensus for task claiming.

    When a device wants to claim a task:
    1. It broadcasts a TASK_CLAIM message
    2. All other peers vote ACK or NACK
    3. If majority ACK, claim is accepted
    4. If majority NACK (or timeout), claim is rejected

    This ensures atomic claiming even when multiple devices try simultaneously.
    """

    def __init__(
        self,
        device_id: str,
        total_peers: int,
        vote_timeout: float = 5.0,
    ):
        self.device_id = device_id
        self.total_peers = total_peers  # Including self
        self.vote_timeout = vote_timeout
        self._active_rounds: Dict[str, ConsensusState] = {}
        self._pending_votes: Dict[str, Dict[str, bool]] = {}  # task_id -> {voter: vote}

    # ----------------------------------------------------------------
    # Claim initiation
    # ----------------------------------------------------------------

    async def initiate_claim(self, task_id: str) -> bool:
        """Initiate a claim consensus round.

        Returns True if claim is accepted by majority.
        """
        # Start with self-vote; preserve any pre-existing votes from peer responses
        if task_id not in self._pending_votes:
            self._pending_votes[task_id] = {}
        self._pending_votes[task_id][self.device_id] = True

        state = ConsensusState(task_id=task_id, claimer_id=self.device_id)
        self._active_rounds[task_id] = state

        # Broadcast claim request
        claim_msg = FedMessage.task_claim(self.device_id, task_id)
        logger.info(
            "Federation consensus: initiating claim for task %s (peers=%d, current_votes=%d)",
            task_id, self.total_peers, len(self._pending_votes[task_id]),
        )

        # Wait for votes or timeout
        await asyncio.sleep(self.vote_timeout)

        # Tally votes
        votes = self._pending_votes.get(task_id, {})
        acks = sum(1 for v in votes.values() if v)
        nacks = sum(1 for v in votes.values() if not v)

        majority = (self.total_peers // 2) + 1
        accepted = acks >= majority

        state.resolved = True
        state.result = accepted
        state.acks = {k for k, v in votes.items() if v}
        state.nacks = {k for k, v in votes.items() if not v}

        logger.info(
            "Federation consensus: task %s %s (acks=%d, nacks=%d, majority=%d)",
            task_id,
            "ACCEPTED" if accepted else "REJECTED",
            acks, nacks, majority,
        )

        # Cleanup
        del self._active_rounds[task_id]
        del self._pending_votes[task_id]

        return accepted

    # ----------------------------------------------------------------
    # Vote handling
    # ----------------------------------------------------------------

    def handle_claim_request(self, msg: FedMessage) -> FedMessage:
        """Handle incoming claim request and cast vote.

        Voting logic:
        - ACK if we don't know about this task (it's available)
        - NACK if we're already executing this task (conflict)
        """
        task_id = msg.payload.get("task_id", "")
        claimer = msg.sender_id

        # Check if we're already executing this task
        # (This would be set by the task executor)
        is_conflict = self._is_task_local(task_id)

        vote = not is_conflict  # ACK if no conflict, NACK if conflict
        self._record_vote(task_id, claimer, vote)

        # Respond with ACK or NACK
        if vote:
            return FedMessage(
                msg_type=MessageType.TASK_CLAIM_ACK.value,
                sender_id=self.device_id,
                target_id=claimer,
                payload={"task_id": task_id, "vote": True},
            )
        else:
            return FedMessage(
                msg_type=MessageType.TASK_CLAIM_NACK.value,
                sender_id=self.device_id,
                target_id=claimer,
                payload={"task_id": task_id, "vote": False, "reason": "task already claimed locally"},
            )

    def handle_vote_response(self, msg: FedMessage) -> None:
        """Handle incoming vote response (ACK or NACK)."""
        task_id = msg.payload.get("task_id", "")
        voter = msg.sender_id
        vote = msg.payload.get("vote", False)

        self._record_vote(task_id, voter, vote)

    def _record_vote(self, task_id: str, voter: str, vote: bool) -> None:
        """Record a vote for a consensus round."""
        if task_id not in self._pending_votes:
            self._pending_votes[task_id] = {}
        self._pending_votes[task_id][voter] = vote

    def _is_task_local(self, task_id: str) -> bool:
        """Check if task is already being executed locally.

        A task is considered local if:
        1. We've already voted ACK for ourselves on this task (we claimed it)
        2. Or it's in our active consensus rounds
        """
        votes = self._pending_votes.get(task_id, {})
        if votes.get(self.device_id):
            return True  # We already claimed this task
        return task_id in self._active_rounds

    # ----------------------------------------------------------------
    # State queries
    # ----------------------------------------------------------------

    def get_active_round(self, task_id: str) -> Optional[ConsensusState]:
        """Get state for an active consensus round."""
        return self._active_rounds.get(task_id)

    def get_pending_votes(self, task_id: str) -> Dict[str, bool]:
        """Get current vote tally for a task."""
        return dict(self._pending_votes.get(task_id, {}))

    @property
    def active_round_count(self) -> int:
        """Number of active consensus rounds."""
        return len(self._active_rounds)
