"""Normalized failures for the PostgreSQL state-store Alembic cutover."""

from __future__ import annotations

from state_store import StateStoreConfigurationError


class StateStoreAlembicError(StateStoreConfigurationError):
    """Base normalized failure before or during Alembic bootstrap."""


class UnsupportedMigrationDialectError(StateStoreAlembicError):
    """The supplied connection is not PostgreSQL."""


class UntrustedTenantSchemaError(StateStoreAlembicError):
    """A caller supplied a schema outside the runtime-owned tenant namespace."""


class StateStoreMigrationReinitializationRequired(StateStoreAlembicError):
    """A legacy custom ledger prevents safe Alembic ownership."""

    def __init__(self) -> None:
        super().__init__(
            "StateStoreMigrationReinitializationRequired: legacy schema_migrations "
            "detected in tenant schema; formal reinitialization is required before "
            "Alembic migration"
        )


LegacySchemaMigrationsReinitializationRequired = StateStoreMigrationReinitializationRequired


class SchemaNotEmptyReinitializationRequired(StateStoreAlembicError):
    """The target schema has unknown objects and cannot be baselined."""


class BaselineMigrationContractError(StateStoreAlembicError):
    """Alembic metadata or the immutable v25 core baseline is invalid."""
