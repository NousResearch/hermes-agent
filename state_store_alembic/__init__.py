"""Lazy PostgreSQL-only Alembic bootstrap support for the v1-v25 core catalog."""

from state_store_alembic.errors import (
    BaselineMigrationContractError,
    LegacySchemaMigrationsReinitializationRequired,
    SchemaNotEmptyReinitializationRequired,
    StateStoreAlembicError,
    StateStoreMigrationReinitializationRequired,
    UnsupportedMigrationDialectError,
    UntrustedTenantSchemaError,
)
from state_store_alembic.migration_helpers import TrustedTenantSchema, V25_CORE_REVISION
from state_store_alembic.runner import (
    CURRENT_STATE_STORE_REVISION,
    BaselineMigrationResult,
    upgrade_new_tenant_to_current,
    upgrade_new_tenant_to_v25,
)
from state_store_alembic.semantic_catalog import V26_SQLITE_IMPORT_MANIFEST_REVISION

__all__ = (
    "BaselineMigrationContractError",
    "BaselineMigrationResult",
    "CURRENT_STATE_STORE_REVISION",
    "LegacySchemaMigrationsReinitializationRequired",
    "SchemaNotEmptyReinitializationRequired",
    "StateStoreAlembicError",
    "StateStoreMigrationReinitializationRequired",
    "TrustedTenantSchema",
    "UnsupportedMigrationDialectError",
    "UntrustedTenantSchemaError",
    "V25_CORE_REVISION",
    "V26_SQLITE_IMPORT_MANIFEST_REVISION",
    "upgrade_new_tenant_to_current",
    "upgrade_new_tenant_to_v25",
)
