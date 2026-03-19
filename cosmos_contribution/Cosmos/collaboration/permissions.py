"""
cosmos Permission Manager - Access Control System

Novel Approaches:
1. Fine-Grained Permissions - Resource-level access control
2. Role-Based Access - Hierarchical role system
3. Permission Inheritance - Cascade permissions through hierarchy
4. Audit Logging - Complete access trail
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import json

from loguru import logger


class PermissionLevel(Enum):
    """Permission levels."""
    NONE = 0
    READ = 1
    WRITE = 2
    DELETE = 3
    ADMIN = 4
    OWNER = 5


class ResourceType(Enum):
    """Types of resources that can be protected."""
    MEMORY = "memory"
    MEMORY_POOL = "memory_pool"
    AGENT = "agent"
    SESSION = "session"
    USER = "user"
    CONFIG = "config"
    MODEL = "model"
    FILE = "file"


@dataclass
class Permission:
    """A specific permission grant."""
    id: str
    resource_type: ResourceType
    resource_id: str  # Specific resource or "*" for all

    # Access levels
    level: PermissionLevel

    # Scope
    granted_to: str  # User ID or role name
    granted_by: str
    granted_at: datetime = field(default_factory=datetime.now)

    # Conditions
    expires_at: Optional[datetime] = None
    conditions: dict = field(default_factory=dict)  # Additional constraints

    # Status
    is_active: bool = True

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "level": self.level.name,
            "granted_to": self.granted_to,
            "granted_by": self.granted_by,
            "is_active": self.is_active,
        }


@dataclass
class AccessControl:
    """Access control entry for a resource."""
    resource_type: ResourceType
    resource_id: str

    # Owner
    owner_id: str

    # Permissions
    permissions: dict[str, PermissionLevel] = field(default_factory=dict)  # user/role -> level

    # Inheritance
    inherit_from: Optional[str] = None  # Parent resource ID

    # Settings
    public_access: PermissionLevel = PermissionLevel.NONE
    default_permission: PermissionLevel = PermissionLevel.NONE

    def to_dict(self) -> dict:
        return {
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "owner": self.owner_id,
            "permissions": {k: v.name for k, v in self.permissions.items()},
            "public_access": self.public_access.name,
        }


@dataclass
class AccessLogEntry:
    """Log entry for access attempts."""
    user_id: str
    resource_type: ResourceType
    resource_id: str
    action: str
    allowed: bool
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


class PermissionManager:
    """
    Fine-grained permission management system.

    Features:
    - Resource-level access control
    - Role-based permissions with inheritance
    - Time-limited grants
    - Comprehensive audit logging
    """

    def __init__(
        self,
        enable_audit_log: bool = True,
        max_log_entries: int = 10000,
    ):
        self.enable_audit_log = enable_audit_log
        self.max_log_entries = max_log_entries

        # Permissions indexed by resource
        self.access_controls: dict[str, AccessControl] = {}  # resource_key -> ACL

        # Individual permission grants
        self.permissions: dict[str, Permission] = {}

        # Role definitions
        self.roles: dict[str, dict[str, PermissionLevel]] = {
            "admin": {
                "*": PermissionLevel.ADMIN,
            },
            "power_user": {
                "memory": PermissionLevel.WRITE,
                "agent": PermissionLevel.WRITE,
                "session": PermissionLevel.WRITE,
            },
            "user": {
                "memory": PermissionLevel.WRITE,
                "agent": PermissionLevel.READ,
                "session": PermissionLevel.WRITE,
            },
            "guest": {
                "memory": PermissionLevel.READ,
                "agent": PermissionLevel.NONE,
                "session": PermissionLevel.READ,
            },
        }

        # User role assignments
        self.user_roles: dict[str, list[str]] = {}

        # Audit log
        self.access_log: list[AccessLogEntry] = []

        self._lock = asyncio.Lock()
        self._permission_counter = 0

    def _resource_key(self, resource_type: ResourceType, resource_id: str) -> str:
        """Generate unique key for a resource."""
        return f"{resource_type.value}:{resource_id}"

    async def create_access_control(
        self,
        resource_type: ResourceType,
        resource_id: str,
        owner_id: str,
        inherit_from: Optional[str] = None,
    ) -> AccessControl:
        """Create access control for a resource."""
        key = self._resource_key(resource_type, resource_id)

        acl = AccessControl(
            resource_type=resource_type,
            resource_id=resource_id,
            owner_id=owner_id,
            inherit_from=inherit_from,
        )

        # Owner gets full access
        acl.permissions[owner_id] = PermissionLevel.OWNER

        async with self._lock:
            self.access_controls[key] = acl

        return acl

    async def grant_permission(
        self,
        resource_type: ResourceType,
        resource_id: str,
        user_id: str,
        level: PermissionLevel,
        granted_by: str,
        expires_at: Optional[datetime] = None,
        conditions: Optional[dict] = None,
    ) -> Permission:
        """Grant permission to a user."""
        key = self._resource_key(resource_type, resource_id)

        # Verify granter has permission
        if not await self.check_permission(
            resource_type, resource_id, granted_by, PermissionLevel.ADMIN
        ):
            raise PermissionError(f"User {granted_by} cannot grant permissions on {key}")

        async with self._lock:
            self._permission_counter += 1
            perm_id = f"perm_{self._permission_counter}"

        permission = Permission(
            id=perm_id,
            resource_type=resource_type,
            resource_id=resource_id,
            level=level,
            granted_to=user_id,
            granted_by=granted_by,
            expires_at=expires_at,
            conditions=conditions or {},
        )

        async with self._lock:
            self.permissions[perm_id] = permission

            # Update ACL
            if key in self.access_controls:
                self.access_controls[key].permissions[user_id] = level

        self._log_access(granted_by, resource_type, resource_id, "grant", True, {
            "target_user": user_id,
            "level": level.name,
        })

        return permission

    async def revoke_permission(
        self,
        permission_id: str,
        revoked_by: str,
    ) -> bool:
        """Revoke a permission grant."""
        if permission_id not in self.permissions:
            return False

        permission = self.permissions[permission_id]

        # Verify revoker has permission
        if not await self.check_permission(
            permission.resource_type,
            permission.resource_id,
            revoked_by,
            PermissionLevel.ADMIN,
        ):
            return False

        async with self._lock:
            permission.is_active = False

            # Update ACL
            key = self._resource_key(permission.resource_type, permission.resource_id)
            if key in self.access_controls:
                if permission.granted_to in self.access_controls[key].permissions:
                    del self.access_controls[key].permissions[permission.granted_to]

        self._log_access(revoked_by, permission.resource_type, permission.resource_id, "revoke", True, {
            "permission_id": permission_id,
        })

        return True

    async def check_permission(
        self,
        resource_type: ResourceType,
        resource_id: str,
        user_id: str,
        required_level: PermissionLevel,
    ) -> bool:
        """Check if user has required permission level."""
        effective_level = await self.get_effective_permission(
            resource_type, resource_id, user_id
        )

        allowed = effective_level.value >= required_level.value

        self._log_access(user_id, resource_type, resource_id, "check", allowed, {
            "required": required_level.name,
            "effective": effective_level.name,
        })

        return allowed

    async def get_effective_permission(
        self,
        resource_type: ResourceType,
        resource_id: str,
        user_id: str,
    ) -> PermissionLevel:
        """Get effective permission level for a user on a resource."""
        key = self._resource_key(resource_type, resource_id)

        # Check direct ACL
        if key in self.access_controls:
            acl = self.access_controls[key]

            # Owner has full access
            if user_id == acl.owner_id:
                return PermissionLevel.OWNER

            # Check user permission
            if user_id in acl.permissions:
                return acl.permissions[user_id]

            # Check role permissions
            user_level = PermissionLevel.NONE
            for role in self.user_roles.get(user_id, []):
                if role in acl.permissions:
                    if acl.permissions[role].value > user_level.value:
                        user_level = acl.permissions[role]

            if user_level != PermissionLevel.NONE:
                return user_level

            # Check inheritance
            if acl.inherit_from:
                return await self.get_effective_permission(
                    resource_type, acl.inherit_from, user_id
                )

            # Return default or public
            if acl.public_access != PermissionLevel.NONE:
                return acl.public_access
            return acl.default_permission

        # Check role-based permissions
        for role in self.user_roles.get(user_id, []):
            if role in self.roles:
                role_perms = self.roles[role]

                # Check wildcard
                if "*" in role_perms:
                    return role_perms["*"]

                # Check resource type
                if resource_type.value in role_perms:
                    return role_perms[resource_type.value]

        return PermissionLevel.NONE

    async def assign_role(
        self,
        user_id: str,
        role: str,
        assigned_by: str,
    ) -> bool:
        """Assign a role to a user."""
        if role not in self.roles:
            return False

        # Check assigner is admin
        is_admin = "admin" in self.user_roles.get(assigned_by, [])
        if not is_admin:
            return False

        async with self._lock:
            if user_id not in self.user_roles:
                self.user_roles[user_id] = []

            if role not in self.user_roles[user_id]:
                self.user_roles[user_id].append(role)

        logger.info(f"Assigned role {role} to user {user_id}")
        return True

    async def remove_role(
        self,
        user_id: str,
        role: str,
        removed_by: str,
    ) -> bool:
        """Remove a role from a user."""
        is_admin = "admin" in self.user_roles.get(removed_by, [])
        if not is_admin:
            return False

        async with self._lock:
            if user_id in self.user_roles:
                self.user_roles[user_id] = [
                    r for r in self.user_roles[user_id] if r != role
                ]

        return True

    async def create_role(
        self,
        role_name: str,
        permissions: dict[str, PermissionLevel],
        created_by: str,
    ) -> bool:
        """Create a custom role."""
        is_admin = "admin" in self.user_roles.get(created_by, [])
        if not is_admin:
            return False

        async with self._lock:
            self.roles[role_name] = permissions

        logger.info(f"Created role {role_name}")
        return True

    async def set_public_access(
        self,
        resource_type: ResourceType,
        resource_id: str,
        level: PermissionLevel,
        set_by: str,
    ) -> bool:
        """Set public access level for a resource."""
        key = self._resource_key(resource_type, resource_id)

        if not await self.check_permission(
            resource_type, resource_id, set_by, PermissionLevel.ADMIN
        ):
            return False

        async with self._lock:
            if key in self.access_controls:
                self.access_controls[key].public_access = level
                return True

        return False

    def _log_access(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        action: str,
        allowed: bool,
        details: dict,
    ):
        """Log an access attempt."""
        if not self.enable_audit_log:
            return

        entry = AccessLogEntry(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            allowed=allowed,
            details=details,
        )

        self.access_log.append(entry)

        # Keep log bounded
        if len(self.access_log) > self.max_log_entries:
            self.access_log = self.access_log[-self.max_log_entries // 2:]

    async def get_access_log(
        self,
        requester_id: str,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[AccessLogEntry]:
        """Get access log entries (requires admin)."""
        is_admin = "admin" in self.user_roles.get(requester_id, [])
        if not is_admin:
            return []

        entries = self.access_log

        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]
        if resource_id:
            entries = [e for e in entries if e.resource_id == resource_id]
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]

        return entries[-limit:]

    async def get_user_permissions(
        self,
        user_id: str,
    ) -> dict:
        """Get all permissions for a user."""
        result = {
            "roles": self.user_roles.get(user_id, []),
            "direct_permissions": [],
            "effective_access": {},
        }

        # Get direct permission grants
        for perm in self.permissions.values():
            if perm.granted_to == user_id and perm.is_active and not perm.is_expired():
                result["direct_permissions"].append(perm.to_dict())

        # Calculate effective access for each resource type
        for resource_type in ResourceType:
            result["effective_access"][resource_type.value] = {
                "default": PermissionLevel.NONE.name,
            }

            for role in result["roles"]:
                if role in self.roles:
                    role_perms = self.roles[role]
                    if resource_type.value in role_perms:
                        result["effective_access"][resource_type.value]["default"] = \
                            role_perms[resource_type.value].name

        return result

    def get_stats(self) -> dict:
        """Get permission manager statistics."""
        return {
            "total_acls": len(self.access_controls),
            "total_permissions": len(self.permissions),
            "active_permissions": sum(
                1 for p in self.permissions.values()
                if p.is_active and not p.is_expired()
            ),
            "total_roles": len(self.roles),
            "users_with_roles": len(self.user_roles),
            "audit_log_size": len(self.access_log),
        }
