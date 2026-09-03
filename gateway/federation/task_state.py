"""HIGH-1: Task state HMAC integrity.

Pillar 3: Data Integrity (SECURITY-BASELINE.md)
Pillar 7: Privacy (telemetry / state is encrypted-at-rest when secret is set)

Wraps task state dicts with HMAC-SHA256 signature to detect tampering
in shared state (iCloud Drive, network sync, etc.).

Use:
    from gateway.federation.task_state import SignedTaskState
    s = SignedTaskState(task_id='t-123', owner='mac-a', step=3, total=10)
    s.sign(cluster_secret='...')
    # Serialized to JSON + signature
    payload = s.to_json()
    # Verify on receive
    s2 = SignedTaskState.from_json(payload)
    assert s2.verify(cluster_secret='...')
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


SIGNATURE_ALGO = "HMAC-SHA256"


@dataclass
class SignedTaskState:
    """Task state record with HMAC signature.

    Fields:
        task_id: unique task identifier
        owner: node_id currently owning the task
        step: current step (0-indexed)
        total: total steps
        status: pending | in_progress | completed | failed | aborted
        started_at: float unix timestamp
        last_heartbeat: ts of last heartbeat from owner
        partial_result: dict (NOT signed — caller should encrypt separately)
        required_capability: set of required capabilities
        sensitivity: TaskSensitivity value
        signature: HMAC over all fields above (set by sign())
        ts_signed: float timestamp when signed
    """
    task_id: str
    owner: str
    step: int = 0
    total: int = 0
    status: str = "pending"
    started_at: float = 0.0
    last_heartbeat: float = 0.0
    partial_result: Dict[str, Any] = field(default_factory=dict)
    required_capability: list = field(default_factory=list)
    sensitivity: str = "medium"
    signature: str = ""
    ts_signed: float = 0.0

    # Fields that ARE part of the signature.
    # partial_result is NOT signed (encrypted separately by caller).
    _SIGNED_FIELDS = (
        "task_id", "owner", "step", "total", "status",
        "started_at", "last_heartbeat", "required_capability",
        "sensitivity", "ts_signed",
    )

    def _canonical_payload(self) -> str:
        """Return canonical JSON of signed fields."""
        d = {f: getattr(self, f) for f in self._SIGNED_FIELDS}
        return json.dumps(d, separators=(",", ":"), sort_keys=True)

    def sign(self, cluster_secret: str) -> None:
        """Compute HMAC signature. Mutates self.signature + ts_signed."""
        if not cluster_secret:
            raise ValueError("cluster_secret required for signing")
        self.ts_signed = time.time()
        payload = self._canonical_payload()
        self.signature = hmac.new(
            cluster_secret.encode() if isinstance(cluster_secret, str) else cluster_secret,
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, cluster_secret: str) -> bool:
        """Verify signature. Returns True if valid."""
        if not self.signature:
            return False
        if not cluster_secret:
            return False
        payload = self._canonical_payload()
        expected = hmac.new(
            cluster_secret.encode() if isinstance(cluster_secret, str) else cluster_secret,
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (including signature)."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignedTaskState":
        """Deserialize from dict. Verify separately."""
        # Filter to known fields
        known = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)

    @classmethod
    def from_json(cls, raw: str) -> "SignedTaskState":
        """Deserialize from JSON. Verify separately."""
        return cls.from_dict(json.loads(raw))


def integrity_check(payload: Dict[str, Any], cluster_secret: str) -> bool:
    """One-shot verification of a serialized task state."""
    state = SignedTaskState.from_dict(payload)
    return state.verify(cluster_secret)


__all__ = ["SignedTaskState", "SIGNATURE_ALGO", "integrity_check"]
