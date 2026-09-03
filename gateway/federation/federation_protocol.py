"""Federation message protocol — type definitions, serialization, and validation.

All inter-device messages flow through this schema.  Versioned so future
revisions stay backward-compatible.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2.0.0"
DEFAULT_WS_PORT = 18765
MESSAGE_TTL_SECONDS = 300  # Messages older than this are dropped (replay protection)


class MessageType(str, Enum):
    """All federation message types."""

    # ── Connectivity layer ──────────────────────────────────────────
    PEER_JOIN = "peer_join"        # New device announced on network
    PEER_LEAVE = "peer_leave"      # Device going offline gracefully
    PEER_PING = "peer_ping"        # Liveness probe
    PEER_PONG = "peer_pong"        # Liveness response
    PEER_CAPABILITIES = "peer_capabilities"  # Full capability broadcast

    # ── Task layer ──────────────────────────────────────────────────
    TASK_SUBMIT = "task_submit"    # New task posted to federation
    TASK_CLAIM = "task_claim"      # Peer claims a pending task
    TASK_CLAIM_ACK = "task_claim_ack"  # Consensus: claim accepted
    TASK_CLAIM_NACK = "task_claim_nack"  # Consensus: claim rejected (another peer claimed first)
    TASK_PROGRESS = "task_progress"  # Streaming progress update
    TASK_RESULT = "task_result"    # Task completed
    TASK_CANCEL = "task_cancel"    # Task cancelled by submitter
    TASK_HEARTBEAT = "task_heartbeat"  # Task executor still alive

    # ── Collaboration layer ─────────────────────────────────────────
    MEMORY_SYNC = "memory_sync"    # Memory entry synced
    SKILL_SYNC = "skill_sync"      # Skill update synced
    SEARCH_QUERY = "search_query"  # Distributed search request
    SEARCH_RESULT = "search_result"  # Distributed search response

    # ── Federation services ─────────────────────────────────────────
    CRON_SYNC = "cron_sync"        # Cron job synced
    CRON_HEARTBEAT = "cron_heartbeat"  # Cron leader still active
    CRON_HANDOFF = "cron_handoff"  # Cron leader transfer

    # ── Compute pool layer ──────────────────────────────────────────
    COMPUTE_REQUEST = "compute_request"  # Distributed compute chunk
    COMPUTE_RESPONSE = "compute_response"  # Chunk execution result

    # ── Cluster management ──────────────────────────────────────────
    ELECTION = "election"            # Leader election initiated
    ELECTION_OK = "election_ok"      # Candidate accepts challenge
    VICTORY = "victory"              # New leader announces itself
    COORDINATE = "coordinate"        # Leader heartbeat
    CONFIG_SYNC = "config_sync"      # Config data broadcast
    CONFIG_ACK = "config_ack"        # Peer acknowledges config update

    # ── Ops layer (Phase 22) ─────────────────────────────────────────
    OPS_HEALTH = "ops_health"        # Health snapshot broadcast (heartbeat-carried)
    OPS_ALERT = "ops_alert"          # Operational alert (lost contact / recovery / degraded)
    OPS_SOS = "ops_sos"              # Aircraft-lost call: peer requests assistance
    OPS_ASSIST_ACK = "ops_assist_ack"  # Peer acknowledges SOS / offers assist


@dataclass
class PeerInfo:
    """Information about a federation peer."""
    device_id: str
    hostname: str
    ws_url: str = ""                    # e.g. ws://192.168.1.10:18765
    status: str = "online"              # online | offline | busy
    cpu_cores: int = 0
    memory_gb: float = 0.0
    load_avg: float = 0.0
    gpu_type: str = ""
    version: str = PROTOCOL_VERSION
    last_seen: float = 0.0
    current_task_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def compute_score(self) -> float:
        """Heuristic score for scheduling — higher = more capable + less busy."""
        base = self.cpu_cores * 10 + self.memory_gb * 5
        if self.load_avg > 0:
            base /= (1 + self.load_avg)
        if self.current_task_id:
            base *= 0.3  # Busy device gets lower priority
        return base


@dataclass
class FedMessage:
    """Canonical federation message — every message between devices uses this."""

    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    msg_type: str = MessageType.PEER_PING.value
    sender_id: str = ""
    target_id: Optional[str] = None   # None = broadcast
    timestamp: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)
    signature: str = ""               # HMAC-SHA256(payload + timestamp, auth_token)

    # Internal routing fields (not serialized)
    _received_at: float = field(default=0.0, repr=False, compare=False)
    _sender_ws_url: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        if not self._received_at:
            self._received_at = time.time()

    def to_json(self) -> str:
        """Serialize to JSON for wire transmission."""
        d = {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "signature": self.signature,
        }
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "FedMessage":
        """Deserialize from JSON wire format."""
        d = json.loads(raw)
        return cls(
            msg_id=d["msg_id"],
            msg_type=d["msg_type"],
            sender_id=d["sender_id"],
            target_id=d.get("target_id"),
            timestamp=d["timestamp"],
            payload=d.get("payload", {}),
            signature=d.get("signature", ""),
        )

    def sign(self, auth_token: str) -> None:
        """Sign the message with HMAC-SHA256."""
        payload_str = json.dumps(self.payload, separators=(",", ":"))
        signing_input = f"{self.msg_id}:{self.msg_type}:{self.timestamp}:{payload_str}"
        self.signature = hashlib.sha256(
            f"{signing_input}:{auth_token}".encode()
        ).hexdigest()  # Full 64-char signature (no truncation)

    def verify(self, auth_token: str) -> bool:
        """Verify message signature."""
        if not self.signature:
            return False
        payload_str = json.dumps(self.payload, separators=(",", ":"))
        signing_input = f"{self.msg_id}:{self.msg_type}:{self.timestamp}:{payload_str}"
        expected = hashlib.sha256(
            f"{signing_input}:{auth_token}".encode()
        ).hexdigest()
        return expected == self.signature

    def is_expired(self, ttl: int = MESSAGE_TTL_SECONDS) -> bool:
        """Check if message is too old (replay protection)."""
        return (time.time() - self.timestamp) > ttl

    @classmethod
    def peer_join(cls, device_id: str, info: PeerInfo) -> "FedMessage":
        """Create a PEER_JOIN message."""
        return cls(
            msg_type=MessageType.PEER_JOIN.value,
            sender_id=device_id,
            payload={
                "peer_info": {
                    "device_id": info.device_id,
                    "hostname": info.hostname,
                    "ws_url": info.ws_url,
                    "cpu_cores": info.cpu_cores,
                    "memory_gb": info.memory_gb,
                    "load_avg": info.load_avg,
                    "gpu_type": info.gpu_type,
                    "version": info.version,
                }
            },
        )

    @classmethod
    def peer_leave(cls, device_id: str, reason: str = "offline") -> "FedMessage":
        """Create a PEER_LEAVE message."""
        return cls(
            msg_type=MessageType.PEER_LEAVE.value,
            sender_id=device_id,
            payload={"reason": reason},
        )

    @classmethod
    def task_submit(
        cls,
        device_id: str,
        task_id: str,
        title: str,
        description: str = "",
        priority: int = 3,
        context_snapshot: Optional[dict] = None,
    ) -> "FedMessage":
        """Create a TASK_SUBMIT message."""
        return cls(
            msg_type=MessageType.TASK_SUBMIT.value,
            sender_id=device_id,
            payload={
                "task_id": task_id,
                "title": title,
                "description": description,
                "priority": priority,
                "context_snapshot": context_snapshot or {},
            },
        )

    @classmethod
    def task_claim(
        cls,
        device_id: str,
        task_id: str,
    ) -> "FedMessage":
        """Create a TASK_CLAIM message."""
        return cls(
            msg_type=MessageType.TASK_CLAIM.value,
            sender_id=device_id,
            payload={"task_id": task_id},
        )

    @classmethod
    def task_progress(
        cls,
        device_id: str,
        task_id: str,
        progress: float,
        note: str = "",
    ) -> "FedMessage":
        """Create a TASK_PROGRESS message."""
        return cls(
            msg_type=MessageType.TASK_PROGRESS.value,
            sender_id=device_id,
            target_id=None,  # broadcast
            payload={
                "task_id": task_id,
                "progress": progress,  # 0.0 - 1.0
                "note": note,
            },
        )

    @classmethod
    def task_result(
        cls,
        device_id: str,
        task_id: str,
        success: bool,
        result_data: Optional[dict] = None,
        error_info: str = "",
    ) -> "FedMessage":
        """Create a TASK_RESULT message."""
        return cls(
            msg_type=MessageType.TASK_RESULT.value,
            sender_id=device_id,
            payload={
                "task_id": task_id,
                "success": success,
                "result_data": result_data or {},
                "error_info": error_info,
            },
        )

    @classmethod
    def task_heartbeat(
        cls,
        device_id: str,
        task_id: str,
    ) -> "FedMessage":
        """Create a TASK_HEARTBEAT — task executor still alive."""
        return cls(
            msg_type=MessageType.TASK_HEARTBEAT.value,
            sender_id=device_id,
            payload={"task_id": task_id},
        )
