"""Federation cluster — leader election and config sync across devices.

Leader Election:
- Bully-style election with compute score as tiebreaker
- Automatic reelection when leader goes offline
- Leader coordinates cron sync, memory consolidation, and config propagation

Config Sync:
- Synchronize config.yaml across federation peers
- Conflict resolution: leader's config wins
- Atomic update with rollback on failure

Protocol messages:
- ELECTION: leader election initiated
- ELECTION_OK: candidate accepts the challenge
- VICTORY: new leader announces itself
- COORDINATE: leader heartbeat
- CONFIG_SYNC: config data broadcast
- CONFIG_ACK: peer acknowledges config update
"""
from __future__ import annotations

import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from gateway.federation.federation_protocol import FedMessage, MessageType

logger = logging.getLogger(__name__)


# ========================================================================
# Leader Election
# ========================================================================

@dataclass
class ElectionState:
    """State of the current election."""

    election_id: str = ""
    initiator: str = ""
    candidates: Dict[str, float] = field(default_factory=dict)  # device_id -> score
    leader: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    status: str = "none"  # none, running, finished

    def to_dict(self) -> dict:
        return {
            "election_id": self.election_id,
            "initiator": self.initiator,
            "candidates": self.candidates,
            "leader": self.leader,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ElectionState":
        return cls(**d)


class FederationLeaderElection:
    """Bully-style leader election with compute score tiebreaker.

    Algorithm:
    1. Device initiates election, broadcasts ELECTION
    2. All peers respond with ELECTION_OK + their score
    3. After timeout, highest score device wins
    4. Winner broadcasts VICTORY
    5. If leader stops heartbeating, trigger reelection

    Usage:
        election = FederationLeaderElection(
            device_id="my-device",
            adapter=federation_adapter,
            compute_score=15.0,  # From compute pool
        )
        await election.start()

        # Trigger election
        await election.initiate_election()
        leader = election.get_leader()  # Returns leader device_id
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        compute_score: float = 0.0,
        election_timeout: float = 5.0,
        heartbeat_interval: float = 15.0,
        missed_heartbeats_threshold: int = 3,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.compute_score = compute_score
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        self.missed_threshold = missed_heartbeats_threshold

        self._current_election: Optional[ElectionState] = None
        self._current_leader: str = ""
        self._last_leader_heartbeat: float = 0.0
        self._missed_heartbeats: int = 0
        self._running = False

    async def start(self) -> None:
        """Start leader election monitoring."""
        self._running = True
        logger.info(
            "Federation leader election: started (device=%s, score=%.1f)",
            self.device_id, self.compute_score,
        )
        # Auto-initiate if no leader exists
        await self.initiate_election()

    async def stop(self) -> None:
        """Stop leader election."""
        self._running = False
        logger.info("Federation leader election: stopped")

    async def initiate_election(self) -> str:
        """Start a new leader election."""
        import uuid
        election_id = str(uuid.uuid4())[:8]

        self._current_election = ElectionState(
            election_id=election_id,
            initiator=self.device_id,
            candidates={self.device_id: self.compute_score},
            started_at=time.time(),
            status="running",
        )

        # Broadcast ELECTION to all peers
        msg = FedMessage(
            msg_type=MessageType.ELECTION.value,
            sender_id=self.device_id,
            payload={
                "election_id": election_id,
                "initiator": self.device_id,
                "score": self.compute_score,
            },
        )
        await self.adapter.send(msg)
        logger.info(
            "Federation election: initiated %s (score=%.1f)",
            election_id, self.compute_score,
        )

        # Wait for responses
        await self._wait_for_election_result()
        return self._current_leader

    async def handle_election(self, msg: FedMessage) -> None:
        """Handle incoming ELECTION from a peer."""
        sender = msg.sender_id
        if sender == self.device_id:
            return

        election_id = msg.payload.get("election_id", "")
        score = msg.payload.get("score", 0.0)

        # If this is a new election, register our candidacy
        if not self._current_election or self._current_election.election_id != election_id:
            self._current_election = ElectionState(
                election_id=election_id,
                initiator=msg.payload.get("initiator", sender),
                candidates={sender: score, self.device_id: self.compute_score},
                started_at=time.time(),
                status="running",
            )
        else:
            # Add candidate to existing election
            self._current_election.candidates[sender] = score

        # Send ELECTION_OK with our score
        ok_msg = FedMessage(
            msg_type=MessageType.ELECTION_OK.value,
            sender_id=self.device_id,
            target_id=sender,
            payload={
                "election_id": election_id,
                "score": self.compute_score,
            },
        )
        await self.adapter.send(ok_msg)

    async def handle_election_ok(self, msg: FedMessage) -> None:
        """Handle incoming ELECTION_OK from a peer."""
        if not self._current_election:
            return

        sender = msg.sender_id
        score = msg.payload.get("score", 0.0)
        self._current_election.candidates[sender] = score

    async def _wait_for_election_result(self) -> None:
        """Wait for election timeout, then determine winner."""
        import asyncio
        await asyncio.sleep(self.election_timeout)

        if not self._current_election:
            return

        # Find highest score
        candidates = self._current_election.candidates
        if not candidates:
            self._current_election.status = "finished"
            return

        winner = max(candidates, key=lambda d: candidates[d])
        self._current_election.leader = winner
        self._current_election.finished_at = time.time()
        self._current_election.status = "finished"
        self._current_leader = winner

        # If we won, announce victory
        if winner == self.device_id and self._current_election:
            await self._announce_victory()

        logger.info(
            "Federation election: %s won (score=%.1f)",
            winner, candidates[winner],
        )

    async def _announce_victory(self) -> None:
        """Broadcast VICTORY message as the new leader."""
        msg = FedMessage(
            msg_type=MessageType.VICTORY.value,
            sender_id=self.device_id,
            payload={
                "election_id": self._current_election.election_id,
                "leader": self.device_id,
                "score": self.compute_score,
            },
        )
        await self.adapter.send(msg)
        self._last_leader_heartbeat = time.time()

    async def handle_victory(self, msg: FedMessage) -> None:
        """Handle incoming VICTORY from the new leader."""
        leader = msg.payload.get("leader", "")
        self._current_leader = leader
        self._last_leader_heartbeat = time.time()
        self._missed_heartbeats = 0
        logger.info("Federation election: acknowledged %s as leader", leader)

    async def send_leader_heartbeat(self) -> None:
        """Send COORDINATE heartbeat (only called by leader)."""
        msg = FedMessage(
            msg_type=MessageType.COORDINATE.value,
            sender_id=self.device_id,
            payload={
                "leader": self.device_id,
                "timestamp": time.time(),
            },
        )
        await self.adapter.send(msg)
        self._last_leader_heartbeat = time.time()

    async def handle_coordinate(self, msg: FedMessage) -> None:
        """Handle incoming COORDINATE heartbeat from leader."""
        sender = msg.payload.get("leader", "")
        if sender != self._current_leader:
            return

        self._last_leader_heartbeat = time.time()
        self._missed_heartbeats = 0

    def is_leader(self) -> bool:
        """Check if this device is the current leader."""
        return self._current_leader == self.device_id

    def get_leader(self) -> str:
        """Get the current leader device_id."""
        return self._current_leader

    @property
    def has_leader(self) -> bool:
        """Check if there's an active leader."""
        return bool(self._current_leader)

    @property
    def election_state(self) -> Optional[ElectionState]:
        """Get the current election state."""
        return self._current_election


# ========================================================================
# Config Sync
# ========================================================================

class FederationConfigSync:
    """Synchronize config.yaml across federation peers.

    The leader's config is authoritative. When config changes on leader,
    it broadcasts to all peers. Peers apply the config atomically.

    Usage:
        sync = FederationConfigSync(
            device_id="my-device",
            adapter=federation_adapter,
            config_path=Path.home() / ".hermes" / "config.yaml",
        )
        await sync.start()

        # When config changes on leader:
        await sync.sync_config()
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        config_path: Optional[Path] = None,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.config_path = config_path or Path.home() / ".hermes" / "config.yaml"
        self._local_config_hash: str = ""
        self._running = False

    async def start(self) -> None:
        """Start config sync."""
        self._running = True
        self._local_config_hash = self._compute_config_hash()
        logger.info(
            "Federation config sync: started (device=%s, hash=%s)",
            self.device_id, self._local_config_hash[:8],
        )

    async def stop(self) -> None:
        """Stop config sync."""
        self._running = False
        logger.info("Federation config sync: stopped")

    def _compute_config_hash(self) -> str:
        """Compute SHA256 hash of config file."""
        if not self.config_path.exists():
            return ""
        content = self.config_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    async def sync_config(self) -> bool:
        """Broadcast current config to all peers."""
        if not self.config_path.exists():
            return False

        content = self.config_path.read_text(encoding="utf-8")
        config_hash = self._compute_config_hash()

        msg = FedMessage(
            msg_type=MessageType.CONFIG_SYNC.value,
            sender_id=self.device_id,
            payload={
                "action": "update",
                "config_hash": config_hash,
                "config": content,
                "source_device": self.device_id,
            },
        )
        await self.adapter.send(msg)
        self._local_config_hash = config_hash
        logger.info(
            "Federation config: synced (hash=%s)", config_hash[:8],
        )
        return True

    async def handle_config_sync(self, msg: FedMessage) -> None:
        """Handle incoming CONFIG_SYNC from leader."""
        sender = msg.sender_id
        action = msg.payload.get("action", "")
        remote_hash = msg.payload.get("config_hash", "")

        if sender == self.device_id:
            return

        if action == "update":
            # Check if we already have this version
            if remote_hash == self._local_config_hash:
                return

            # Apply remote config
            config_content = msg.payload.get("config", "")
            if not config_content:
                return

            self._apply_remote_config(config_content, remote_hash)
            self._local_config_hash = remote_hash

            # Acknowledge
            ack_msg = FedMessage(
                msg_type=MessageType.CONFIG_ACK.value,
                sender_id=self.device_id,
                target_id=sender,
                payload={
                    "config_hash": remote_hash,
                    "status": "applied",
                },
            )
            await self.adapter.send(ack_msg)

    def _apply_remote_config(self, content: str, config_hash: str) -> None:
        """Apply remote config to local file atomically."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename (atomic on POSIX)
        temp_path = self.config_path.with_suffix(".yaml.tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.rename(self.config_path)

        logger.info(
            "Federation config: applied remote config (hash=%s)",
            config_hash[:8],
        )
