"""Behavior-neutral state-store contracts.

This package describes which state backend a profile intends to use. Runtime
storage remains SQLite until a later production PostgreSQL implementation
consumes the PostgreSQL spec.
"""

from state_store.resolver import (
    StateStoreBackendNotActivatedError,
    StateStoreConfigurationError,
    resolve_state_store,
)
from state_store.schema import (
    SCHEMA_V22_MANIFEST,
    SchemaManifestParity,
    SchemaV22Manifest,
    schema_v22_manifest_parity,
    sqlite_relational_table_names,
)
from state_store.spec import StateStoreSpec

__all__ = [
    "SCHEMA_V22_MANIFEST",
    "SchemaManifestParity",
    "SchemaV22Manifest",
    "StateStoreBackendNotActivatedError",
    "StateStoreConfigurationError",
    "StateStoreSpec",
    "resolve_state_store",
    "schema_v22_manifest_parity",
    "sqlite_relational_table_names",
]
