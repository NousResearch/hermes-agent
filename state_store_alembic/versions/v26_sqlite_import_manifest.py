"""SQLite import manifest and immutable failure-receipt catalog, after core v25."""

from __future__ import annotations

from alembic import op

from state_store_alembic.migration_helpers import V25_CORE_REVISION, qualified_name

revision = "state_store_v26_sqlite_import"
down_revision = V25_CORE_REVISION
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = op.get_context().config.attributes["tenant_schema"]
    q = lambda relation: qualified_name(schema, relation)
    op.execute(
        f"CREATE TABLE {q('sqlite_import_manifests')} ("
        "import_id text PRIMARY KEY, source_fingerprint text NOT NULL, source_counts jsonb NOT NULL, "
        "source_schema jsonb NOT NULL, pre_import_target jsonb NOT NULL, destination_counts jsonb, "
        "status text NOT NULL CHECK (status IN ('running', 'failed', 'complete')), error text, "
        "created_at double precision NOT NULL, updated_at double precision NOT NULL)"
    )


def downgrade() -> None:
    raise RuntimeError("State-store SQLite import manifests cannot be downgraded")
