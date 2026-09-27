"""Dependency-free helpers shared by the PostgreSQL Alembic runtime and revision."""

from __future__ import annotations

import re
import sys

from state_store_alembic.errors import UntrustedTenantSchemaError

TENANT_SCHEMA_PATTERN = re.compile(r"^hermes_state_store_tenant_[0-9a-f]{32}$")
V25_CORE_REVISION = "state_store_v25_core"


class TrustedTenantSchema(str):
    """A runtime-derived tenant schema, never a caller-controlled identifier."""

    def __new__(cls, name: str) -> "TrustedTenantSchema":
        # Test targets are owned-target allocators too.  Their raw SQL fixture
        # needs a string-compatible capability, but ordinary callers must use
        # the runtime resolver or importer allocator below.
        caller = sys._getframe(1).f_globals.get("__name__")
        if caller == "tests.integration.postgresql_test_target":
            return _issue_tenant_schema(name, _OWNED_TARGET_ISSUER)
        raise UntrustedTenantSchemaError(
            "State-store Alembic tenant capabilities are issued only by the runtime resolver or owned target allocator"
        )

    @property
    def name(self) -> str:
        return str(self)

    def __reduce__(self) -> tuple[object, tuple[str]]:
        """Preserve an issued capability when ``spawn`` serializes process arguments."""
        require_trusted_tenant_schema(self)
        return _restore_pickled_tenant_schema, (str(self),)


_TENANT_SCHEMA_CAPABILITY = object()
_RUNTIME_RESOLVER_ISSUER = object()
_OWNED_TARGET_ISSUER = object()
_PICKLE_RESTORE_ISSUER = object()


def _issue_tenant_schema(name: str, issuer: object) -> TrustedTenantSchema:
    """Mint a capability for one package-owned issuance path only."""

    if issuer not in (_RUNTIME_RESOLVER_ISSUER, _OWNED_TARGET_ISSUER, _PICKLE_RESTORE_ISSUER):
        raise UntrustedTenantSchemaError(
            "State-store Alembic tenant capabilities are issued only by the runtime resolver or owned target allocator"
        )
    if not isinstance(name, str) or not TENANT_SCHEMA_PATTERN.fullmatch(name):
        raise UntrustedTenantSchemaError(
            "State-store Alembic requires a trusted "
            "hermes_state_store_tenant_<32 lowercase hex> schema"
        )
    schema = str.__new__(TrustedTenantSchema, name)
    schema._state_store_tenant_capability = _TENANT_SCHEMA_CAPABILITY  # type: ignore[attr-defined]
    return schema


def _issue_runtime_tenant_schema(name: str) -> TrustedTenantSchema:
    return _issue_tenant_schema(name, _RUNTIME_RESOLVER_ISSUER)


def _issue_owned_target_tenant_schema(name: str) -> TrustedTenantSchema:
    return _issue_tenant_schema(name, _OWNED_TARGET_ISSUER)


def _restore_pickled_tenant_schema(name: str) -> TrustedTenantSchema:
    """Restore an already-issued capability from a trusted multiprocessing pickle."""
    return _issue_tenant_schema(name, _PICKLE_RESTORE_ISSUER)


def require_trusted_tenant_schema(schema: object) -> TrustedTenantSchema:
    """Reject strings and forged lookalikes before they can reach PostgreSQL."""

    if (
        type(schema) is not TrustedTenantSchema
        or getattr(schema, "_state_store_tenant_capability", None) is not _TENANT_SCHEMA_CAPABILITY
    ):
        raise UntrustedTenantSchemaError(
            "State-store Alembic requires a runtime-derived trusted tenant schema capability"
        )
    return schema


def quote_identifier(identifier: str) -> str:
    """Quote only a trusted tenant identifier; this is not generic SQL escaping."""

    if not TENANT_SCHEMA_PATTERN.fullmatch(identifier):
        raise UntrustedTenantSchemaError("Refusing to quote an untrusted tenant schema identifier")
    return f'"{identifier}"'


def qualified_name(schema: TrustedTenantSchema, relation: str) -> str:
    """Return a package-owned relation qualified by a trusted schema."""

    if not re.fullmatch(r"[a-z_][a-z0-9_]*", relation):
        raise ValueError("Migration relation names must be package-owned SQL identifiers")
    return f'{quote_identifier(schema.name)}."{relation}"'
