"""HIGH-3: Heartbeat payload sanitization — keep it lean & privacy-safe.

Pillar 7: Privacy (SECURITY-BASELINE.md)

Heartbeat payload is broadcast to ALL peers in the cluster. It must
NOT contain:
- Task payload (user input/output, partial results)
- Memory content
- User PII
- Conversation history
- File contents

Heartbeat payload SHOULD contain:
- Node ID (public)
- Timestamp (public)
- Status (public)
- Capability metadata (public)
- current_task_id (presence, not content)
- last_heartbeat (public)

Anything beyond this whitelist is a privacy violation.

Use:
    from gateway.federation.heartbeat_payload import sanitize_heartbeat
    payload = sanitize_heartbeat(raw_heartbeat)
    # payload is now safe to broadcast to all peers

This module is the SINGLE source of truth for what can be in a
heartbeat. Any new field added to heartbeat protocol MUST be added
here, reviewed, and a unit test added.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
import time


# === Heartbeat contract ===

# Whitelist of fields allowed in heartbeat. Any field NOT in this set
# is stripped before broadcast. Defense-in-depth: even if a developer
# accidentally adds a sensitive field, it won't leak.
HEARTBEAT_WHITELIST: Set[str] = {
    # Identity
    "node_id",
    "hostname",
    # Liveness
    "status",               # "online" | "offline" | "busy"
    "last_heartbeat",       # float unix timestamp
    "ts",                   # float unix timestamp (this heartbeat)
    # Capability (public)
    "cpu_cores",
    "memory_gb",
    "load_avg",
    "ip_address",
    "version",
    # Task presence (NOT content!)
    "has_task",             # bool — does this node have an in-flight task?
    "current_task_id",      # str — identifier only, no payload
    "current_task_step",    # int — for progress display
    "current_task_total",   # int — for progress display
    # Trust state (public)
    "trust_level",          # own trust 评级 (sanitized)
}

# Blacklist of NEVER allowed fields. Anyone setting these causes an
# audit log warning.
HEARTBEAT_BLACKLIST: Set[str] = {
    # Task content
    "task_payload",
    "task_partial_result",
    "task_input",
    "task_output",
    "task_messages",
    # Memory
    "memory_content",
    "memories",
    "memory_snapshot",
    # User data
    "user_input",
    "user_output",
    "user_pii",
    "user_email",
    "user_messages",
    "chat_history",
    # Files
    "file_content",
    "file_paths",
    "ssh_keys",
    "tokens",
    "credentials",
    # Tool results
    "tool_results",
    "tool_outputs",
    # CLI arguments
    "command",
    "args",
    # Anything else containing sensitive data
    "private_key",
    "secret",
    "password",
}


# === Heartbeat record ===

@dataclass
class HeartbeatPayload:
    """Sanitized heartbeat payload.

    All fields are PUBLIC information that any peer can know.
    """
    # Required
    node_id: str
    ts: float = field(default_factory=time.time)
    # Optional (sanitized metadata)
    hostname: Optional[str] = None
    status: str = "online"
    last_heartbeat: Optional[float] = None
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    load_avg: Optional[float] = None
    ip_address: Optional[str] = None
    version: Optional[str] = None
    # Task presence (NOT content)
    has_task: bool = False
    current_task_id: Optional[str] = None
    current_task_step: int = 0
    current_task_total: int = 0
    # Trust
    trust_level: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


# === Sanitization ===

# Field names that are LIKELY task content (heuristic, not strict)
_SUSPICIOUS_KEYWORDS = (
    "payload", "input", "output", "message", "content", "result",
    "memory", "history", "user", "file", "auth", "token",
    "secret", "private", "command", "args", "tool",
)


def sanitize_heartbeat(raw: Dict[str, Any]) -> HeartbeatPayload:
    """Sanitize a raw heartbeat dict into a safe HeartbeatPayload.

    Strips any field not in HEARTBEAT_WHITELIST or that matches a
    blacklist keyword. Returns a clean HeartbeatPayload.

    Audit log WARNING if any field is stripped (potential privacy leak).
    """
    from gateway.federation.audit import SecurityEvent
    stripped: List[str] = []

    # First pass: whitelist
    safe: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in HEARTBEAT_WHITELIST:
            safe[k] = v
        else:
            stripped.append(k)

    # Second pass: blacklist keyword check (defense-in-depth)
    # Whitelist fields take precedence — heuristic only strips fields
    # that are NOT already on the whitelist.
    for k, v in list(safe.items()):
        if k in HEARTBEAT_BLACKLIST:
            stripped.append(k)
            del safe[k]
        elif k not in HEARTBEAT_WHITELIST and any(kw in k.lower() for kw in _SUSPICIOUS_KEYWORDS):
            # Heuristic: only for fields NOT on whitelist.
            # Whitelist fields with keyword matches stay (e.g. memory_gb).
            stripped.append(k)
            del safe[k]

    if stripped:
        # Audit log warning (this would normally be hooked into AuditLog)
        try:
            from gateway.federation.audit import SecurityEvent
            # caller is expected to wire this
        except ImportError:
            pass
        import logging
        logging.warning(
            f"Heartbeat payload sanitizer stripped {len(stripped)} sensitive "
            f"fields: {stripped[:5]}{'...' if len(stripped) > 5 else ''}. "
            f"Check caller: {raw.get('node_id', 'unknown')}"
        )

    # Build HeartbeatPayload from sanitized dict
    return HeartbeatPayload(
        node_id=safe.get("node_id", "unknown"),
        ts=safe.get("ts", time.time()),
        hostname=safe.get("hostname"),
        status=safe.get("status", "online"),
        last_heartbeat=safe.get("last_heartbeat"),
        cpu_cores=safe.get("cpu_cores"),
        memory_gb=safe.get("memory_gb"),
        load_avg=safe.get("load_avg"),
        ip_address=safe.get("ip_address"),
        version=safe.get("version"),
        has_task=safe.get("has_task", False),
        current_task_id=safe.get("current_task_id"),
        current_task_step=safe.get("current_task_step", 0),
        current_task_total=safe.get("current_task_total", 0),
        trust_level=safe.get("trust_level", "unknown"),
    )


def is_safe_field(field_name: str) -> bool:
    """Check if a single field is safe to broadcast."""
    if field_name in HEARTBEAT_BLACKLIST:
        return False
    if field_name not in HEARTBEAT_WHITELIST:
        return False
    return True


def assert_safe(payload: Dict[str, Any], raise_on_violation: bool = True) -> bool:
    """Check a payload against the heartbeat contract.

    Returns True if all fields are safe. If `raise_on_violation` is True
    and any field is unsafe, raises ValueError.

    Call this BEFORE broadcasting any heartbeat to assert the contract.
    """
    bad = []
    for k in payload:
        if k not in HEARTBEAT_WHITELIST:
            bad.append(k)
        elif k in HEARTBEAT_BLACKLIST:
            bad.append(k)
    if bad:
        msg = f"Unsafe heartbeat fields: {bad}"
        if raise_on_violation:
            raise ValueError(msg)
        return False
    return True


__all__ = [
    "HEARTBEAT_WHITELIST",
    "HEARTBEAT_BLACKLIST",
    "HeartbeatPayload",
    "sanitize_heartbeat",
    "is_safe_field",
    "assert_safe",
]
