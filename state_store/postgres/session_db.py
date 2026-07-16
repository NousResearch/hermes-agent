"""Composed PostgreSQL SessionDB implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Optional

from state_store.postgres.core import (
    PostgresConfigurationError,
    PostgresSchemaValidationError,
    PostgresStateStore,
)
from state_store.postgres.ops_messages import PostgresSessionDBMessageOperations
from state_store.postgres.ops_search import PostgresSearchOperations
from state_store.postgres.ops_sessions import PostgresSessionOperations
from state_store.postgres.session_db_base import PostgresSessionDBBase
from state_store.spec import StateStoreSpec


StateStoreFactory = Callable[..., PostgresStateStore]


class PostgresSessionDB(
    PostgresSessionOperations,
    PostgresSessionDBMessageOperations,
    PostgresSearchOperations,
):
    """PostgreSQL implementation of the public durable SessionDB contract."""

    @classmethod
    def from_spec(
        cls,
        spec: StateStoreSpec,
        *,
        environ: Optional[Mapping[str, str]] = None,
        state_store_factory: StateStoreFactory = PostgresStateStore,
    ) -> "PostgresSessionDB":
        """Open the explicit PostgreSQL state spec and prepare its schema."""

        if spec.backend != "postgres":
            raise PostgresConfigurationError(
                "PostgresSessionDB requires a PostgreSQL state-store specification"
            )
        environment = os.environ if environ is None else environ
        # Deliberately resolve only the configured variable name from the spec.
        if not environment.get(spec.postgres_dsn_env):
            raise PostgresConfigurationError(
                "PostgreSQL state DSN is not available from its configured environment"
            )

        state_store = state_store_factory(spec, environ=environment)
        try:
            if spec.read_only:
                report = state_store.health_report()
                if not report.available or not report.capabilities.get(
                    "core_schema",
                    False,
                ):
                    raise PostgresSchemaValidationError(
                        "Read-only PostgreSQL state schema is unavailable or incomplete"
                    )
            else:
                state_store.migrate()
                report = state_store.health_report()
                if not report.available or not report.capabilities.get(
                    "core_schema",
                    False,
                ):
                    raise PostgresSchemaValidationError(
                        "PostgreSQL state schema did not validate after migration"
                    )
            return cls(state_store, capabilities=report.capabilities)
        except BaseException:
            state_store.close()
            raise
