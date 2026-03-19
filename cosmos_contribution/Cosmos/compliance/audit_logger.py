"""
cosmos Audit Logger

"Every action must be recorded! The Central Bureaucracy demands it!"

Comprehensive audit logging for compliance and security.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib
from loguru import logger


class AuditEventType(Enum):
    """Types of auditable events."""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"

    # Authorization
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"

    # Data Access
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"

    # Configuration
    CONFIG_CHANGE = "config_change"
    SETTING_UPDATE = "setting_update"

    # Security
    SECRET_ACCESS = "secret_access"
    SECRET_CREATED = "secret_created"
    SECRET_DELETED = "secret_deleted"
    CERTIFICATE_ISSUED = "certificate_issued"
    CERTIFICATE_REVOKED = "certificate_revoked"

    # System
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"

    # Custom
    CUSTOM = "custom"


class AuditSeverity(Enum):
    """Audit event severity."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """An audit log event."""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity

    # Actor information
    actor_id: str
    actor_type: str  # user, service, system
    actor_ip: str = ""

    # Action details
    action: str
    resource_type: str
    resource_id: str
    description: str = ""

    # Context
    service: str = "cosmos"
    session_id: str = ""
    request_id: str = ""
    correlation_id: str = ""

    # Result
    success: bool = True
    error_message: str = ""

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    before_state: Dict = field(default_factory=dict)
    after_state: Dict = field(default_factory=dict)

    # Integrity
    hash: str = ""
    previous_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "actor_ip": self.actor_ip,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "description": self.description,
            "service": self.service,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "hash": self.hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute integrity hash for the event."""
        data = f"{self.id}:{self.timestamp.isoformat()}:{self.event_type.value}:{self.actor_id}:{self.action}:{self.resource_id}:{previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()


class AuditLogger:
    """
    Comprehensive audit logging system.

    Features:
    - Tamper-evident logging with hash chain
    - Multiple output destinations
    - Log rotation and retention
    - Compliance-ready formats
    - Real-time alerting
    - Query and search
    """

    def __init__(
        self,
        storage_path: Path = None,
        retention_days: int = 365,
        hash_chain: bool = True,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/audit")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.retention_days = retention_days
        self.hash_chain = hash_chain

        self.events: List[AuditEvent] = []
        self.last_hash = ""
        self.output_handlers: List[Callable] = []
        self.alert_handlers: List[Callable] = []

        self._load_events()

    def _load_events(self):
        """Load recent events from storage."""
        today = datetime.utcnow().date()
        log_file = self.storage_path / f"audit-{today.isoformat()}.jsonl"

        if log_file.exists():
            try:
                with open(log_file) as f:
                    for line in f:
                        data = json.loads(line)
                        self.last_hash = data.get("hash", "")
            except Exception as e:
                logger.error(f"Failed to load audit events: {e}")

    # =========================================================================
    # LOGGING
    # =========================================================================

    async def log(
        self,
        event_type: AuditEventType,
        actor_id: str,
        actor_type: str,
        action: str,
        resource_type: str,
        resource_id: str,
        success: bool = True,
        severity: AuditSeverity = AuditSeverity.INFO,
        description: str = "",
        actor_ip: str = "",
        session_id: str = "",
        request_id: str = "",
        correlation_id: str = "",
        metadata: Dict = None,
        before_state: Dict = None,
        after_state: Dict = None,
        error_message: str = "",
    ) -> AuditEvent:
        """Log an audit event."""
        import uuid

        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            actor_id=actor_id,
            actor_type=actor_type,
            actor_ip=actor_ip,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            session_id=session_id,
            request_id=request_id,
            correlation_id=correlation_id,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
            before_state=before_state or {},
            after_state=after_state or {},
        )

        # Compute hash chain
        if self.hash_chain:
            event.previous_hash = self.last_hash
            event.hash = event.compute_hash(self.last_hash)
            self.last_hash = event.hash

        # Store event
        await self._store_event(event)
        self.events.append(event)

        # Send to output handlers
        for handler in self.output_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Output handler failed: {e}")

        # Check for alerts
        await self._check_alerts(event)

        return event

    async def _store_event(self, event: AuditEvent):
        """Store event to file."""
        date = event.timestamp.date()
        log_file = self.storage_path / f"audit-{date.isoformat()}.jsonl"

        with open(log_file, "a") as f:
            f.write(event.to_json() + "\n")

    async def _check_alerts(self, event: AuditEvent):
        """Check if event should trigger alerts."""
        # Alert on security events
        should_alert = False

        if event.event_type in [
            AuditEventType.LOGIN_FAILED,
            AuditEventType.SECRET_ACCESS,
            AuditEventType.PERMISSION_GRANTED,
            AuditEventType.ROLE_ASSIGNED,
        ]:
            should_alert = True

        if event.severity in [AuditSeverity.WARNING, AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
            should_alert = True

        if not event.success:
            should_alert = True

        if should_alert:
            for handler in self.alert_handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    async def log_login(
        self,
        user_id: str,
        success: bool,
        ip_address: str = "",
        session_id: str = "",
        error: str = "",
    ):
        """Log a login attempt."""
        await self.log(
            event_type=AuditEventType.LOGIN if success else AuditEventType.LOGIN_FAILED,
            actor_id=user_id,
            actor_type="user",
            action="login",
            resource_type="session",
            resource_id=session_id,
            success=success,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            actor_ip=ip_address,
            session_id=session_id,
            error_message=error,
        )

    async def log_data_access(
        self,
        user_id: str,
        action: str,  # read, write, delete
        resource_type: str,
        resource_id: str,
        success: bool = True,
        metadata: Dict = None,
    ):
        """Log data access."""
        event_map = {
            "read": AuditEventType.DATA_READ,
            "write": AuditEventType.DATA_WRITE,
            "delete": AuditEventType.DATA_DELETE,
            "export": AuditEventType.DATA_EXPORT,
        }

        await self.log(
            event_type=event_map.get(action, AuditEventType.DATA_READ),
            actor_id=user_id,
            actor_type="user",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            metadata=metadata,
        )

    async def log_config_change(
        self,
        user_id: str,
        setting: str,
        before: Any,
        after: Any,
    ):
        """Log a configuration change."""
        await self.log(
            event_type=AuditEventType.CONFIG_CHANGE,
            actor_id=user_id,
            actor_type="user",
            action="update",
            resource_type="config",
            resource_id=setting,
            description=f"Changed {setting}",
            before_state={"value": before},
            after_state={"value": after},
        )

    async def log_permission_change(
        self,
        admin_id: str,
        target_user: str,
        permission: str,
        granted: bool,
    ):
        """Log a permission change."""
        await self.log(
            event_type=AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_REVOKED,
            actor_id=admin_id,
            actor_type="admin",
            action="grant" if granted else "revoke",
            resource_type="permission",
            resource_id=permission,
            description=f"{'Granted' if granted else 'Revoked'} {permission} for {target_user}",
            metadata={"target_user": target_user, "permission": permission},
        )

    # =========================================================================
    # HANDLERS
    # =========================================================================

    def add_output_handler(self, handler: Callable):
        """Add an output handler for events."""
        self.output_handlers.append(handler)

    def add_alert_handler(self, handler: Callable):
        """Add an alert handler for security events."""
        self.alert_handlers.append(handler)

    # =========================================================================
    # QUERY
    # =========================================================================

    def query(
        self,
        event_type: AuditEventType = None,
        actor_id: str = None,
        resource_type: str = None,
        resource_id: str = None,
        since: datetime = None,
        until: datetime = None,
        success: bool = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events from storage."""
        results = []

        # Determine date range
        start_date = since.date() if since else (datetime.utcnow() - timedelta(days=7)).date()
        end_date = until.date() if until else datetime.utcnow().date()

        current_date = start_date
        while current_date <= end_date:
            log_file = self.storage_path / f"audit-{current_date.isoformat()}.jsonl"

            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        try:
                            data = json.loads(line)

                            # Apply filters
                            if event_type and data.get("event_type") != event_type.value:
                                continue
                            if actor_id and data.get("actor_id") != actor_id:
                                continue
                            if resource_type and data.get("resource_type") != resource_type:
                                continue
                            if resource_id and data.get("resource_id") != resource_id:
                                continue
                            if success is not None and data.get("success") != success:
                                continue

                            timestamp = datetime.fromisoformat(data["timestamp"])
                            if since and timestamp < since:
                                continue
                            if until and timestamp > until:
                                continue

                            event = AuditEvent(
                                id=data["id"],
                                timestamp=timestamp,
                                event_type=AuditEventType(data["event_type"]),
                                severity=AuditSeverity(data.get("severity", "info")),
                                actor_id=data["actor_id"],
                                actor_type=data["actor_type"],
                                actor_ip=data.get("actor_ip", ""),
                                action=data["action"],
                                resource_type=data["resource_type"],
                                resource_id=data["resource_id"],
                                description=data.get("description", ""),
                                success=data.get("success", True),
                                metadata=data.get("metadata", {}),
                                hash=data.get("hash", ""),
                            )
                            results.append(event)

                            if len(results) >= limit:
                                break
                        except Exception as e:
                            logger.error(f"Failed to parse audit event: {e}")

            current_date += timedelta(days=1)

            if len(results) >= limit:
                break

        return sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_user_activity(
        self,
        user_id: str,
        days: int = 7,
    ) -> List[AuditEvent]:
        """Get recent activity for a user."""
        return self.query(
            actor_id=user_id,
            since=datetime.utcnow() - timedelta(days=days),
        )

    def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
        days: int = 30,
    ) -> List[AuditEvent]:
        """Get history for a resource."""
        return self.query(
            resource_type=resource_type,
            resource_id=resource_id,
            since=datetime.utcnow() - timedelta(days=days),
        )

    def get_failed_actions(
        self,
        days: int = 1,
    ) -> List[AuditEvent]:
        """Get recent failed actions."""
        return self.query(
            success=False,
            since=datetime.utcnow() - timedelta(days=days),
        )

    # =========================================================================
    # INTEGRITY
    # =========================================================================

    def verify_integrity(
        self,
        date: datetime = None,
    ) -> Dict[str, Any]:
        """Verify audit log integrity for a date."""
        target_date = date.date() if date else datetime.utcnow().date()
        log_file = self.storage_path / f"audit-{target_date.isoformat()}.jsonl"

        if not log_file.exists():
            return {"valid": True, "message": "No log file for date", "checked": 0}

        valid = True
        issues = []
        checked = 0
        previous_hash = ""

        with open(log_file) as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    stored_hash = data.get("hash", "")
                    stored_previous = data.get("previous_hash", "")

                    if i > 0 and stored_previous != previous_hash:
                        valid = False
                        issues.append(f"Line {i+1}: Hash chain broken")

                    # Recompute hash
                    event_id = data["id"]
                    timestamp = data["timestamp"]
                    event_type = data["event_type"]
                    actor_id = data["actor_id"]
                    action = data["action"]
                    resource_id = data["resource_id"]

                    computed = hashlib.sha256(
                        f"{event_id}:{timestamp}:{event_type}:{actor_id}:{action}:{resource_id}:{stored_previous}".encode()
                    ).hexdigest()

                    if stored_hash and computed != stored_hash:
                        valid = False
                        issues.append(f"Line {i+1}: Hash mismatch (tampering detected)")

                    previous_hash = stored_hash
                    checked += 1

                except Exception as e:
                    valid = False
                    issues.append(f"Line {i+1}: Parse error - {e}")

        return {
            "valid": valid,
            "date": target_date.isoformat(),
            "checked": checked,
            "issues": issues,
        }

    # =========================================================================
    # RETENTION
    # =========================================================================

    async def cleanup_old_logs(self):
        """Remove logs older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        removed = 0

        for log_file in self.storage_path.glob("audit-*.jsonl"):
            try:
                date_str = log_file.stem.replace("audit-", "")
                log_date = datetime.fromisoformat(date_str)

                if log_date < cutoff:
                    log_file.unlink()
                    removed += 1
                    logger.info(f"Removed old audit log: {log_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {log_file}: {e}")

        return removed

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_to_json(
        self,
        output_path: Path,
        since: datetime = None,
        until: datetime = None,
    ):
        """Export audit events to JSON file."""
        events = self.query(since=since, until=until, limit=10000)

        with open(output_path, "w") as f:
            json.dump(
                [e.to_dict() for e in events],
                f,
                indent=2,
            )

        logger.info(f"Exported {len(events)} audit events to {output_path}")

    def generate_summary(
        self,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Generate an audit summary report."""
        events = self.query(
            since=datetime.utcnow() - timedelta(days=days),
            limit=10000,
        )

        summary = {
            "period": f"Last {days} days",
            "total_events": len(events),
            "by_type": {},
            "by_actor": {},
            "by_resource": {},
            "failed_actions": 0,
            "unique_actors": set(),
        }

        for event in events:
            # By type
            et = event.event_type.value
            summary["by_type"][et] = summary["by_type"].get(et, 0) + 1

            # By actor
            summary["by_actor"][event.actor_id] = summary["by_actor"].get(event.actor_id, 0) + 1
            summary["unique_actors"].add(event.actor_id)

            # By resource type
            rt = event.resource_type
            summary["by_resource"][rt] = summary["by_resource"].get(rt, 0) + 1

            if not event.success:
                summary["failed_actions"] += 1

        summary["unique_actors"] = len(summary["unique_actors"])

        return summary


# Singleton instance
audit_logger = AuditLogger()
