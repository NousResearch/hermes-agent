"""Alembic environment for the programmatic PostgreSQL state-store bootstrap."""

from __future__ import annotations

from alembic import context


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")
    schema = context.config.attributes.get("tenant_schema")
    if connection is None or schema is None:
        raise RuntimeError("state_store_alembic requires a programmatically supplied PostgreSQL connection and schema")
    if connection.dialect.name != "postgresql":
        raise RuntimeError("state_store_alembic supports only PostgreSQL")
    context.configure(
        connection=connection,
        target_metadata=None,
        include_schemas=True,
        version_table="alembic_version",
        version_table_schema=schema.name,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
