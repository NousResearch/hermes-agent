"""PostgreSQL-native derived search schema for SessionDB search operations."""

from __future__ import annotations


POSTGRES_SEARCH_CAPABILITY = "full_text_search"
SEARCH_VECTOR_COLUMN = "search_vector"

# Keep this expression byte-identical between the generated vector, trigram
# index, and ILIKE search predicate so both index types cover the same fields.
SEARCH_DOCUMENT_EXPRESSION = (
    "COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' "
    "|| COALESCE(tool_calls, '')"
)

SEARCH_CAPABILITY_SETUP_SQL = (
    "CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public",
    f"""
    ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS {SEARCH_VECTOR_COLUMN} tsvector
    GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig, {SEARCH_DOCUMENT_EXPRESSION})
    ) STORED
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_messages_search_vector_gin
    ON messages USING GIN ({SEARCH_VECTOR_COLUMN})
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_messages_search_document_trgm
    ON messages USING GIN (({SEARCH_DOCUMENT_EXPRESSION}) public.gin_trgm_ops)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sessions_id_trgm
    ON sessions USING GIN (id public.gin_trgm_ops)
    """,
)

# REINDEX is the PostgreSQL equivalent of compacting the derived search indexes.
# These statements are static and intentionally run only for writable stores
# advertising the search capability.
SEARCH_REINDEX_STATEMENTS = (
    "REINDEX INDEX idx_messages_search_vector_gin",
    "REINDEX INDEX idx_messages_search_document_trgm",
    "REINDEX INDEX idx_sessions_id_trgm",
)


def search_capability_setup_sql() -> tuple[str, ...]:
    """Return idempotent SQL required before enabling native search."""

    return SEARCH_CAPABILITY_SETUP_SQL
