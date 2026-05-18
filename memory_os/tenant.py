"""
Tenant resolution for multi-tenant memory systems.

Provides MemoryTenant (a dataclass representing a tenant's identity and permissions)
and TenantResolver (an abstract base that maps user context to a MemoryTenant).

Implementations should subclass TenantResolver and inject whatever backend-specific
logic is needed (database lookups, config files, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MemoryTenant:
    """Represents a tenant in a multi-tenant memory system.

    A tenant is the unit of isolation.  Every memory operation is scoped to
    exactly one tenant.  The tenant carries enough metadata for the guard
    layer to enforce permissions without reaching into backend-specific code.

    Attributes:
        namespace:        Unique namespace string, e.g. 'user:alice' or 'agent:bot1'.
        evidence_store_id: Identifier for the tenant's evidence store, e.g. 'evidence_store_alice'.
        rule_store_id:    Identifier for the tenant's rule store, e.g. 'user_alice_rules'.
        policy_store_id:  Identifier for the tenant's policy store, e.g. 'user_alice_policy'.
        boot_profile_id:  Identifier for the tenant's boot profile, e.g. 'user_alice_boot'.
        permissions:      Dict of permission flags, e.g.
                          {'can_read_core': True, 'can_write_core': False, 'is_admin': False}.
    """

    namespace: str
    evidence_store_id: str
    rule_store_id: str
    policy_store_id: str
    boot_profile_id: str
    permissions: dict = field(default_factory=lambda: {
        "can_read_core": True,
        "can_write_core": False,
        "is_admin": False,
    })


class TenantResolver(ABC):
    """Resolves user context to a MemoryTenant.

    Subclass this and override `resolve` / `resolve_admin` to plug in
    your own user-to-tenant mapping (database lookup, config file, etc.).

    Typical usage::

        resolver: TenantResolver = MyDbResolver(db_session)
        tenant = resolver.resolve({"user_id": "alice"})
    """

    @abstractmethod
    def resolve(self, user_context: dict) -> MemoryTenant:
        """Return a MemoryTenant for the given *user_context*.

        Args:
            user_context: Arbitrary dict identifying the user, e.g.
                          {"user_id": "alice"} or {"platform": "web", "user_id": "alice"}.

        Returns:
            A MemoryTenant with normal (non-admin) permissions.

        Raises:
            LookupError: If the user cannot be resolved to a tenant.
        """
        ...

    @abstractmethod
    def resolve_admin(self, user_context: dict) -> MemoryTenant:
        """Return a MemoryTenant with elevated (admin) permissions.

        Implementations should verify that the user is authorised for admin
        access before granting elevated permissions.

        Args:
            user_context: Same as :meth:`resolve`.

        Returns:
            A MemoryTenant with ``is_admin=True`` and write access to core.

        Raises:
            LookupError: If the user cannot be resolved.
            PermissionError: If the user is not authorised for admin access.
        """
        ...
