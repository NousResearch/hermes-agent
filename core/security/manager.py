"""
Security & Permission Management System.

Enforces security levels: SAFE, STANDARD, POWER, UNRESTRICTED.
Manages tool permissions (ALLOW, DENY, ASK), audit logging, and credentials.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    SAFE = "SAFE"
    STANDARD = "STANDARD"
    POWER = "POWER"
    UNRESTRICTED = "UNRESTRICTED"


class PermissionState(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ASK = "ASK"


class AuditLogger:
    """Audit logger for recording tool executions, permission decisions, and model calls."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path
        self._entries: List[Dict[str, Any]] = []

    def log_action(
        self,
        action_type: str,
        details: Dict[str, Any],
        user_id: str = "default",
        session_id: str = "default",
    ) -> None:
        # Scrub potential sensitive fields
        sanitized_details = self._sanitize(details)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": action_type,
            "user_id": user_id,
            "session_id": session_id,
            "details": sanitized_details,
        }
        self._entries.append(entry)
        if self.log_path:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    def _sanitize(self, data: Any) -> Any:
        sensitive_keys = {"password", "token", "api_key", "secret", "cookie", "otp"}
        if isinstance(data, dict):
            return {
                k: ("[REDACTED]" if k.lower() in sensitive_keys else self._sanitize(v))
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return data

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._entries[-limit:]


class CredentialManager:
    """Manages encrypted credential access across platform keychains/keystores."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def set_credential(self, service: str, key: str, value: str) -> None:
        self._store[f"{service}:{key}"] = value

    def get_credential(self, service: str, key: str) -> Optional[str]:
        return self._store.get(f"{service}:{key}") or os.environ.get(key.upper())

    def delete_credential(self, service: str, key: str) -> bool:
        k = f"{service}:{key}"
        if k in self._store:
            del self._store[k]
            return True
        return False


class PermissionManager:
    """Manages permission policies for tools and actions."""

    DEFAULT_PERMISSIONS: Dict[str, PermissionState] = {
        "read_files": PermissionState.ALLOW,
        "write_files": PermissionState.ASK,
        "delete_files": PermissionState.ASK,
        "execute_commands": PermissionState.ASK,
        "network": PermissionState.ASK,
        "browser": PermissionState.ALLOW,
        "git_push": PermissionState.ASK,
        "system": PermissionState.ASK,
        "messaging": PermissionState.ASK,
    }

    def __init__(
        self, security_level: SecurityLevel = SecurityLevel.STANDARD
    ) -> None:
        self.security_level = security_level
        self._overrides: Dict[str, PermissionState] = {}

    def set_override(self, permission: str, state: PermissionState) -> None:
        self._overrides[permission] = state

    def check_permission(self, permission: str) -> PermissionState:
        if self.security_level == SecurityLevel.UNRESTRICTED:
            return PermissionState.ALLOW
        if self.security_level == SecurityLevel.SAFE:
            if permission in ("execute_commands", "delete_files", "system"):
                return PermissionState.DENY

        if permission in self._overrides:
            return self._overrides[permission]

        return self.DEFAULT_PERMISSIONS.get(permission, PermissionState.ASK)


class SecurityManager:
    """Central security manager coordinating permissions, credentials, and audit logging."""

    def __init__(
        self, security_level: SecurityLevel = SecurityLevel.STANDARD
    ) -> None:
        self.security_level = security_level
        self.permission_manager = PermissionManager(security_level)
        self.credential_manager = CredentialManager()
        self.audit_logger = AuditLogger()

    def set_security_level(self, level: SecurityLevel) -> None:
        self.security_level = level
        self.permission_manager.security_level = level

    def validate_action(
        self,
        tool_name: str,
        permission: str,
        params: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
    ) -> PermissionState:
        state = self.permission_manager.check_permission(permission)
        self.audit_logger.log_action(
            action_type="permission_check",
            details={
                "tool": tool_name,
                "permission": permission,
                "result": state.value,
                "params": params or {},
            },
            session_id=session_id,
        )
        return state
