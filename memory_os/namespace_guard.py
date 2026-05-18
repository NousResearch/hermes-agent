"""
Namespace isolation guard for multi-tenant memory systems.

NamespaceGuard enforces that every memory read/write/admin operation is
scoped to the correct tenant namespace.  Admin tenants may access all
namespaces; normal tenants are restricted to their own.
"""

from __future__ import annotations

from memory_os.tenant import MemoryTenant


class NamespaceGuard:
    """Enforces namespace isolation for all memory operations.

    This is a stateless guard — it takes a MemoryTenant and a target namespace
    and returns a boolean.  No backend-specific logic is needed here; the guard
    relies solely on the tenant's ``permissions`` dict and ``namespace`` field.

    Usage::

        guard = NamespaceGuard()
        if guard.can_write(tenant, "user:alice"):
            # proceed with write
            ...
    """

    def can_read(self, tenant: MemoryTenant, target_namespace: str) -> bool:
        """Check whether *tenant* may read from *target_namespace*.

        Rules:
        - Admin tenants (``is_admin=True``) can read everything.
        - Non-admin tenants can only read their own namespace.

        Args:
            tenant: The requesting tenant.
            target_namespace: The namespace being read.

        Returns:
            True if read access is permitted.
        """
        if tenant.permissions.get("is_admin", False):
            return True
        return tenant.namespace == target_namespace

    def can_write(self, tenant: MemoryTenant, target_namespace: str) -> bool:
        """Check whether *tenant* may write to *target_namespace*.

        Rules:
        - Admin tenants can write everywhere.
        - Non-admin tenants can write to their own namespace only if
          ``can_write_core`` is True in their permissions.

        Args:
            tenant: The requesting tenant.
            target_namespace: The namespace being written to.

        Returns:
            True if write access is permitted.
        """
        if tenant.permissions.get("is_admin", False):
            return True
        if tenant.namespace != target_namespace:
            return False
        return tenant.permissions.get("can_write_core", False)

    def can_admin(self, tenant: MemoryTenant) -> bool:
        """Check whether *tenant* has admin privileges.

        Args:
            tenant: The requesting tenant.

        Returns:
            True if the tenant is an admin.
        """
        return tenant.permissions.get("is_admin", False)

    def filter_namespace(self, tenant: MemoryTenant, query_namespace: str) -> str:
        """Return the effective namespace for a query.

        Admin tenants receive an empty string (meaning "all namespaces").
        Non-admin tenants always get their own namespace, regardless of what
        was requested.

        Args:
            tenant: The requesting tenant.
            query_namespace: The namespace in the original query (may be ignored
                             for non-admin tenants).

        Returns:
            Empty string ``""`` for admins (all namespaces), otherwise the
            tenant's own namespace.
        """
        if tenant.permissions.get("is_admin", False):
            return ""
        return tenant.namespace
