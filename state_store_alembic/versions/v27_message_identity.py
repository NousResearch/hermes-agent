"""Add nullable message and tool-call identity fields after SQLite import v26."""

from __future__ import annotations

from alembic import op

from state_store_alembic.migration_helpers import qualified_name

revision = "state_store_v27_message_identity"
down_revision = "state_store_v26_sqlite_import"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = op.get_context().config.attributes["tenant_schema"]
    op.execute(
        f"ALTER TABLE {qualified_name(schema, 'messages')} "
        "ADD COLUMN message_uid text, "
        "ADD COLUMN absorbed_message_uids text, "
        "ADD COLUMN tool_call_uids text, "
        "ADD COLUMN tool_call_uid text"
    )


def downgrade() -> None:
    raise RuntimeError("State-store message identity columns cannot be downgraded")
