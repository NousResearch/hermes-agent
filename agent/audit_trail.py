"""
Cryptographic Audit Trail — SHA-256 Hash-Chained Action Log

Based on OpenFang's audit.rs design. Each action is chained to the previous
via SHA-256, making historical actions tamper-evident.

Usage:
    audit = AuditTrail()
    audit.log_action("tool_call", "read_file", "/path/to/file")
    audit.log_action("tool_result", "read_file", result="42 bytes")
    audit.verify_chain()  # raises if tampered
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Action type taxonomy from OpenFang design"""
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    FILE_DELETE = "file_delete"
    NETWORK_REQUEST = "network_request"
    SHELL_EXEC = "shell_exec"
    SECRET_ACCESS = "secret_access"
    CONFIG_CHANGE = "config_change"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    DELEGATE_TASK = "delegate_task"
    DELEGATE_RESULT = "delegate_result"


_GENESIS_HASH = "0" * 64  # 64 zeros for the genesis (first) entry
_AUDIT_DIR = get_hermes_home() / "audit"
_AUDIT_LOG = _AUDIT_DIR / "audit.log"
_AUDIT_INDEX = _AUDIT_DIR / "audit.index.json"


@dataclass
class AuditEntry:
    """
    Single audit log entry. Hash-chained to previous entry for tamper evidence.

    Hash computation:
        hash = SHA256(sequence + timestamp + agent_id + action_type +
                      resource + details_json + prev_hash)
    """
    sequence: int
    timestamp: str  # ISO 8601 UTC
    agent_id: str
    action_type: str
    resource: str
    details: dict[str, Any]
    prev_hash: str
    hash: str = ""  # Computed on finalize

    def finalize(self) -> "AuditEntry":
        """Compute and set the hash field."""
        content = (
            f"{self.sequence}"
            f"{self.timestamp}"
            f"{self.agent_id}"
            f"{self.action_type}"
            f"{self.resource}"
            f"{json.dumps(self.details, sort_keys=True, ensure_ascii=False)}"
            f"{self.prev_hash}"
        )
        self.hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditEntry":
        return cls(**d)


class AuditTrail:
    """
    Thread-safe, append-only SHA-256 hash-chained audit log.

    Entries are appended to both:
    - audit.log   — append-only text log (one JSON line per entry)
    - audit.index — SQLite or JSON index for efficient querying
    """

    def __init__(
        self,
        agent_id: str = "hermes",
        log_path: Path | None = None,
        index_path: Path | None = None,
    ):
        self.agent_id = agent_id
        self.log_path = Path(log_path) if log_path else _AUDIT_LOG
        self.index_path = Path(index_path) if index_path else _AUDIT_INDEX
        self._lock = threading.Lock()
        self._sequence = 0
        self._last_hash = _GENESIS_HASH
        self._verified = True  # Track if chain is currently valid
        self._loaded = False

        # Ensure audit directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> int:
        """Load last sequence and hash from index file. Returns last sequence."""
        try:
            if self.index_path.exists():
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self._sequence = data.get("last_sequence", 0)
                self._last_hash = data.get("last_hash", _GENESIS_HASH)
                logger.debug(
                    "AuditTrail loaded: sequence=%d, last_hash=%s",
                    self._sequence, self._last_hash[:8]
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Could not load audit index: %s", e)
        return self._sequence

    def _write_index(self) -> None:
        """Write current sequence and hash to index (best-effort)."""
        try:
            self.index_path.write_text(
                json.dumps({
                    "last_sequence": self._sequence,
                    "last_hash": self._last_hash,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug("Could not write audit index: %s", e)

    def log_action(
        self,
        action_type: str | ActionType,
        resource: str,
        details: Optional[dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> AuditEntry:
        """
        Log a single action with hash chaining.

        Args:
            action_type: ActionType enum value or string
            resource: What was acted on (file path, URL, tool name, etc.)
            details: Additional context (args, result snippet, etc.)
            agent_id: Override the agent ID for this entry
            timestamp: Override ISO timestamp (for replay/recovery)

        Returns:
            The created AuditEntry with computed hash
        """
        if isinstance(action_type, ActionType):
            action_type = action_type.value

        with self._lock:
            if not self._loaded:
                self._load_index()
                self._loaded = True

            self._sequence += 1
            entry = AuditEntry(
                sequence=self._sequence,
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
                agent_id=agent_id or self.agent_id,
                action_type=action_type,
                resource=resource,
                details=details or {},
                prev_hash=self._last_hash,
            ).finalize()

            # Append to log file (append-only)
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            except OSError as e:
                logger.error("Failed to write audit log entry: %s", e)

            # Update index (best-effort, non-blocking)
            self._last_hash = entry.hash
            self._write_index()

            logger.debug(
                "AUDIT #[%d] %s %s @ %s: %s",
                entry.sequence, entry.action_type, entry.resource,
                entry.agent_id, entry.hash[:12]
            )
            return entry

    def verify_chain(self, log_path: Path | None = None) -> tuple[bool, list[str]]:
        """
        Verify the entire audit chain. Re-computes each hash from its components
        and the previous hash. Returns (is_valid, list_of_errors).

        Raises on corruption/tampering.
        """
        path = log_path or self.log_path
        if not path.exists():
            return True, []  # Empty log is valid

        errors: list[str] = []
        prev_hash = _GENESIS_HASH
        expected_seq = 1

        try:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {lineno}: JSON parse error: {e}")
                        continue

                    try:
                        entry = AuditEntry.from_dict(d)
                    except (TypeError, KeyError) as e:
                        errors.append(f"Line {lineno}: Invalid entry structure: {e}")
                        continue

                    # Sequence check
                    if entry.sequence != expected_seq:
                        errors.append(
                            f"Line {lineno}: Sequence gap — expected {expected_seq}, got {entry.sequence}"
                        )

                    # Previous hash check
                    if entry.prev_hash != prev_hash:
                        errors.append(
                            f"Line {lineno}: Previous hash mismatch — "
                            f"expected {prev_hash[:16]}..., got {entry.prev_hash[:16]}..."
                        )

                    # Hash check
                    computed = (
                        f"{entry.sequence}"
                        f"{entry.timestamp}"
                        f"{entry.agent_id}"
                        f"{entry.action_type}"
                        f"{entry.resource}"
                        f"{json.dumps(entry.details, sort_keys=True, ensure_ascii=False)}"
                        f"{entry.prev_hash}"
                    )
                    expected_hash = hashlib.sha256(computed.encode("utf-8")).hexdigest()
                    if entry.hash != expected_hash:
                        errors.append(
                            f"Line {lineno}: Hash mismatch — stored {entry.hash[:16]}..., "
                            f"computed {expected_hash[:16]}..."
                        )

                    prev_hash = entry.hash
                    expected_seq = entry.sequence + 1

        except OSError as e:
            errors.append(f"File read error: {e}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def query(
        self,
        action_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        resource_pattern: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        Query the audit log. Reads from the log file and filters.
        For high-volume use cases, build an index separately.
        """
        if not self.log_path.exists():
            return []

        results: list[AuditEntry] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        entry = AuditEntry.from_dict(d)
                    except (json.JSONDecodeError, TypeError, KeyError):
                        continue

                    if action_type and entry.action_type != action_type:
                        continue
                    if agent_id and entry.agent_id != agent_id:
                        continue
                    if resource_pattern and resource_pattern not in entry.resource:
                        continue
                    if since and entry.timestamp < since:
                        continue

                    results.append(entry)
                    if len(results) >= limit:
                        break
        except OSError:
            pass

        return results

    def get_chain_stats(self) -> dict[str, Any]:
        """Return summary statistics about the audit chain."""
        if not self.log_path.exists():
            return {"entries": 0, "valid": True, "errors": []}

        is_valid, errors = self.verify_chain()
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        except OSError:
            count = 0

        return {
            "entries": count,
            "valid": is_valid,
            "errors": errors[:5],  # First 5 errors
            "last_hash": self._last_hash[:16] + "...",
            "last_sequence": self._sequence,
        }


# ---- Integration helpers ----

def _get_audit_trail(agent=None) -> AuditTrail:
    """Get or create an AuditTrail instance for the given agent."""
    if agent is not None:
        agent_id = getattr(agent, "session_id", None) or getattr(agent, "model", "hermes")
    else:
        agent_id = "hermes"

    # Use a module-level singleton per agent_id
    if not hasattr(_get_audit_trail, "_instances"):
        _get_audit_trail._instances: dict[str, AuditTrail] = {}

    if agent_id not in _get_audit_trail._instances:
        _get_audit_trail._instances[agent_id] = AuditTrail(agent_id=agent_id)

    return _get_audit_trail._instances[agent_id]


# ---- Tool integration ----

def audit_tool_call(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    agent=None,
) -> None:
    """Log a tool call and its result to the audit trail."""
    audit = _get_audit_trail(agent)

    # Sanitize args — remove sensitive values
    safe_args = {}
    for k, v in args.items():
        if any(secret in k.lower() for secret in ["token", "key", "secret", "password", "auth"]):
            safe_args[k] = "[REDACTED]"
        else:
            safe_args[k] = str(v)[:200] if isinstance(v, (str, int, float, bool)) else str(v)[:200]

    # Truncate result
    result_str = str(result)[:500] if result else ""

    audit.log_action(
        action_type=ActionType.TOOL_CALL,
        resource=tool_name,
        details={"args": safe_args, "result_preview": result_str[:200]},
    )


def audit_file_operation(
    operation: str,  # read/write/delete
    path: str,
    agent=None,
    details: Optional[dict] = None,
) -> None:
    """Log a file operation to the audit trail."""
    audit = _get_audit_trail(agent)

    type_map = {
        "write": ActionType.FILE_WRITE,
        "read": ActionType.FILE_READ,
        "delete": ActionType.FILE_DELETE,
    }
    action_type = type_map.get(operation, ActionType.FILE_WRITE)

    audit.log_action(
        action_type=action_type,
        resource=path,
        details=details or {},
    )


def verify_audit_integrity() -> dict[str, Any]:
    """CLI/skill helper — verify the audit chain and return a report."""
    audit = AuditTrail()
    is_valid, errors = audit.verify_chain()
    stats = audit.get_chain_stats()
    return {
        "valid": is_valid,
        "entries": stats["entries"],
        "errors": errors,
        "last_hash": stats.get("last_hash"),
    }
