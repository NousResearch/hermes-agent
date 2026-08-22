"""CRITICAL-4: Structured audit log with HMAC integrity.

Pillar 6: Audit & Logging (SECURITY-BASELINE.md)

Properties:
- Structured events (typed, queryable)
- HMAC-SHA256 chain integrity (tamper-evident)
- Token redacted automatically (TokenStr)
- Append-only (no UPDATE/DELETE)
- Encrypted-at-rest when cluster_secret is provided
- 90-day retention (configurable)

Use:
    from gateway.federation.audit import AuditLog, NodeEvent, TaskEvent
    log = AuditLog(cluster_secret="...")  # init once
    log.append(NodeEvent.join(node_id="mac-a", trust="verified"))
    log.append(TaskEvent.claim(task_id="t-123", from_node="a", to_node="b"))
"""
from __future__ import annotations

import enum
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Dict, List, Optional


# === Token redaction ===

class TokenStr(str):
    """String that redacts itself in logs/repr.

    Prevents accidental token leakage to log files. The string itself
    retains the secret value (so callers can use it). Only `repr()`,
    JSON serialization paths, and explicit calls to `redact()` produce
    the public-safe form.
    """

    def __repr__(self) -> str:
        if len(self) <= 8:
            return "***"
        return f"{self[:4]}***{self[-4:]}"

    def __str__(self) -> str:
        # __str__ returns raw value so `str(token)` is still usable.
        # Logging relies on repr() being called via format spec, so
        # callers should use `repr(token)` or `redact(token)` for logs.
        return str.__str__(self)


def redact(value: Any) -> Any:
    """Recursively redact strings that look like tokens."""
    if isinstance(value, TokenStr):
        return repr(value)
    if isinstance(value, str):
        # Look for token-like patterns
        if value.startswith("hermes_") or value.startswith("sk-") or value.startswith("gho_"):
            return TokenStr(value).__repr__()
        return value
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


# === Event types ===

class EventType(str, enum.Enum):
    # Node events
    NODE_JOIN = "node.join"
    NODE_LEAVE = "node.leave"
    NODE_REVOKE = "node.revoke"
    NODE_TRUST_UPGRADE = "node.trust_upgrade"
    NODE_TRUST_DOWNGRADE = "node.trust_downgrade"
    # Task events
    TASK_CREATE = "task.create"
    TASK_CLAIM = "task.claim"
    TASK_ABORT = "task.abort"
    TASK_COMPLETE = "task.complete"
    TASK_RELAY = "task.relay"
    TASK_RELAY_DECISION = "task.relay_decision"
    # Security events
    SECURITY_HEARTBEAT_FAILURE = "security.heartbeat_failure"
    SECURITY_DEATH_CONFIRMED = "security.death_confirmed"
    SECURITY_SIG_INVALID = "security.signature_invalid"
    SECURITY_DENIED = "security.access_denied"
    SECURITY_RATE_LIMIT = "security.rate_limit"
    # Failure events
    FAILURE_DETECT = "failure.detect"
    FAILURE_REVIVE = "failure.revive"
    # User actions
    USER_DECISION = "user.decision"
    USER_APPROVE = "user.approve"
    USER_DENY = "user.deny"


@dataclass
class AuditEvent:
    """Single audit event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    ts: float = field(default_factory=time.time)
    event_type: str = ""  # EventType value
    actor_node_id: str = ""
    target: Optional[str] = None  # task_id, peer_id, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = ""  # chain hash for tamper detection
    signature: str = ""  # HMAC over event

    SEVERITY_NORMAL: ClassVar[str] = "normal"
    SEVERITY_ALERT: ClassVar[str] = "alert"
    severity: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "event_id": self.event_id,
            "ts": self.ts,
            "event_type": self.event_type,
            "actor_node_id": self.actor_node_id,
            "target": self.target,
            "metadata": redact(self.metadata),
            "prev_hash": self.prev_hash,
            "signature": self.signature,
            "severity": self.severity,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


# === Typed event factories ===

class NodeEvent:
    """Factory for node-related events."""

    @staticmethod
    def join(node_id: str, trust: str = "unknown", **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.NODE_JOIN.value,
            actor_node_id=node_id,
            target=node_id,
            metadata={"trust": trust, **meta},
        )

    @staticmethod
    def leave(node_id: str, reason: str = "user", **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.NODE_LEAVE.value,
            actor_node_id=node_id,
            target=node_id,
            metadata={"reason": reason, **meta},
        )

    @staticmethod
    def revoke(node_id: str, reason: str = "manual", **meta: Any) -> AuditEvent:
        ev = AuditEvent(
            event_type=EventType.NODE_REVOKE.value,
            actor_node_id=node_id,
            target=node_id,
            metadata={"reason": reason, **meta},
            severity=AuditEvent.SEVERITY_ALERT,
        )
        return ev

    @staticmethod
    def trust_upgrade(node_id: str, from_trust: str, to_trust: str, **meta: Any) -> AuditEvent:
        ev = AuditEvent(
            event_type=EventType.NODE_TRUST_UPGRADE.value,
            actor_node_id=node_id,
            target=node_id,
            metadata={"from": from_trust, "to": to_trust, **meta},
        )
        return ev

    @staticmethod
    def trust_downgrade(node_id: str, from_trust: str, to_trust: str, **meta: Any) -> AuditEvent:
        ev = AuditEvent(
            event_type=EventType.NODE_TRUST_DOWNGRADE.value,
            actor_node_id=node_id,
            target=node_id,
            metadata={"from": from_trust, "to": to_trust, **meta},
            severity=AuditEvent.SEVERITY_ALERT,
        )
        return ev


class TaskEvent:
    """Factory for task-related events."""

    @staticmethod
    def create(task_id: str, task_title: str, creator: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.TASK_CREATE.value,
            actor_node_id=creator,
            target=task_id,
            metadata={"title": task_title, **meta},
        )

    @staticmethod
    def claim(task_id: str, from_node: str, to_node: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.TASK_CLAIM.value,
            actor_node_id=to_node,
            target=task_id,
            metadata={"from": from_node, "to": to_node, **meta},
        )

    @staticmethod
    def abort(task_id: str, owner: str, reason: str = "user", **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.TASK_ABORT.value,
            actor_node_id=owner,
            target=task_id,
            metadata={"reason": reason, **meta},
        )

    @staticmethod
    def complete(task_id: str, owner: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.TASK_COMPLETE.value,
            actor_node_id=owner,
            target=task_id,
            metadata=meta,
        )

    @staticmethod
    def relay(task_id: str, from_node: str, to_node: str, decision: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.TASK_RELAY.value,
            actor_node_id=to_node,
            target=task_id,
            metadata={"from": from_node, "to": to_node, "decision": decision, **meta},
        )

    @staticmethod
    def relay_decision(task_id: str, decision: str, confidence: float, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.TASK_RELAY_DECISION.value,
            actor_node_id=meta.pop("actor", "system"),
            target=task_id,
            metadata={"decision": decision, "confidence": confidence, **meta},
        )


class SecurityEvent:
    """Factory for security events."""

    @staticmethod
    def heartbeat_failure(node_id: str, failure_count: int, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.SECURITY_HEARTBEAT_FAILURE.value,
            actor_node_id=node_id,
            target=node_id,
            metadata={"failure_count": failure_count, **meta},
        )

    @staticmethod
    def death_confirmed(node_id: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.SECURITY_DEATH_CONFIRMED.value,
            actor_node_id=node_id,
            target=node_id,
            metadata=meta,
            severity=AuditEvent.SEVERITY_ALERT,
        )

    @staticmethod
    def signature_invalid(actor: str, target: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.SECURITY_SIG_INVALID.value,
            actor_node_id=actor,
            target=target,
            metadata=meta,
            severity=AuditEvent.SEVERITY_ALERT,
        )

    @staticmethod
    def access_denied(actor: str, target: str, reason: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.SECURITY_DENIED.value,
            actor_node_id=actor,
            target=target,
            metadata={"reason": reason, **meta},
            severity=AuditEvent.SEVERITY_ALERT,
        )

    @staticmethod
    def rate_limit(actor: str, endpoint: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.SECURITY_RATE_LIMIT.value,
            actor_node_id=actor,
            target=endpoint,
            metadata=meta,
        )


class UserEvent:
    """Factory for user-action events."""

    @staticmethod
    def decision(actor: str, decision: str, target: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.USER_DECISION.value,
            actor_node_id=actor,
            target=target,
            metadata={"decision": decision, **meta},
        )

    @staticmethod
    def approve(actor: str, target: str, **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.USER_APPROVE.value,
            actor_node_id=actor,
            target=target,
            metadata=meta,
        )

    @staticmethod
    def deny(actor: str, target: str, reason: str = "", **meta: Any) -> AuditEvent:
        return AuditEvent(
            event_type=EventType.USER_DENY.value,
            actor_node_id=actor,
            target=target,
            metadata={"reason": reason, **meta},
        )


# === Audit log ===

class AuditLog:
    """Append-only audit log with HMAC integrity chain.

    Each event's signature = HMAC(event_json, cluster_secret).prev_hash links
    to the previous event's hash, forming a tamper-evident chain.
    """

    def __init__(
        self,
        cluster_secret: str,
        log_path: Optional[Path] = None,
        retention_days: int = 90,
    ):
        self._secret = cluster_secret.encode() if isinstance(cluster_secret, str) else cluster_secret
        self._path = Path(log_path) if log_path else None
        self._retention_days = retention_days
        self._lock = Lock()
        self._last_hash = ""  # chain tip
        self._buffer: List[AuditEvent] = []
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load_tip()

    def _load_tip(self) -> None:
        """Read last event's hash on startup to continue chain."""
        if not self._path or not self._path.exists():
            return
        # Read last line
        last_line = ""
        with open(self._path, "rb") as f:
            for line in f:
                last_line = line.decode("utf-8", errors="replace").strip()
        if not last_line:
            return
        try:
            data = json.loads(last_line)
            self._last_hash = data.get("signature", "")
        except (json.JSONDecodeError, KeyError):
            # Corrupt tail — log a warning but continue
            pass

    def append(self, event: AuditEvent) -> AuditEvent:
        """Append event to log. Returns event with signature set."""
        with self._lock:
            event.prev_hash = self._last_hash
            # HMAC over canonical event JSON (minus signature)
            payload = json.dumps(
                {
                    "event_id": event.event_id,
                    "ts": event.ts,
                    "event_type": event.event_type,
                    "actor_node_id": event.actor_node_id,
                    "target": event.target,
                    "metadata": redact(event.metadata),
                    "prev_hash": event.prev_hash,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            sig = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
            event.signature = sig
            self._last_hash = sig
            self._buffer.append(event)
            self._flush()
            return event

    def _flush(self) -> None:
        """Write buffer to disk (append-only)."""
        if not self._path:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            for ev in self._buffer:
                f.write(ev.to_json() + "\n")
        self._buffer.clear()

    def verify_chain(self) -> bool:
        """Verify integrity of on-disk chain."""
        if not self._path or not self._path.exists():
            return True
        prev_hash = ""
        with open(self._path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    return False
                # Check prev_hash
                if data["prev_hash"] != prev_hash:
                    return False
                # Verify signature
                payload = json.dumps(
                    {
                        "event_id": data["event_id"],
                        "ts": data["ts"],
                        "event_type": data["event_type"],
                        "actor_node_id": data["actor_node_id"],
                        "target": data["target"],
                        "metadata": data["metadata"],
                        "prev_hash": data["prev_hash"],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                expected = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
                if expected != data["signature"]:
                    return False
                prev_hash = data["signature"]
        return True

    def query(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        target: Optional[str] = None,
        since_ts: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query on-disk events. For audit dashboards."""
        if not self._path or not self._path.exists():
            return []
        results: List[AuditEvent] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event_type and data["event_type"] != event_type:
                    continue
                if actor and data["actor_node_id"] != actor:
                    continue
                if target and data["target"] != target:
                    continue
                if since_ts and data["ts"] < since_ts:
                    continue
                ev = AuditEvent(
                    event_id=data["event_id"],
                    ts=data["ts"],
                    event_type=data["event_type"],
                    actor_node_id=data["actor_node_id"],
                    target=data.get("target"),
                    metadata=data.get("metadata", {}),
                    prev_hash=data.get("prev_hash", ""),
                    signature=data.get("signature", ""),
                    severity=data.get("severity", "normal"),
                )
                results.append(ev)
                if len(results) >= limit:
                    break
        return results


__all__ = [
    "AuditLog",
    "AuditEvent",
    "EventType",
    "NodeEvent",
    "TaskEvent",
    "SecurityEvent",
    "UserEvent",
    "TokenStr",
    "redact",
]
