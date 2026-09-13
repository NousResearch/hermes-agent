"""Keep PostgreSQL message projections consistent with canonical row mutations."""

from typing import Any


_DISPLAY_IDENTITY_CHANGED = """
    (NEW.role, NEW.content, NEW.timestamp, NEW.tool_call_id, NEW.tool_calls,
     NEW.tool_name, NEW.display_kind, NEW.display_metadata)
    IS DISTINCT FROM
    (OLD.role, OLD.content, OLD.timestamp, OLD.tool_call_id, OLD.tool_calls,
     OLD.tool_name, OLD.display_kind, OLD.display_metadata)
"""
_SEARCH_TEXT_CHANGED = """
    (NEW.content, NEW.tool_name, NEW.tool_calls)
    IS DISTINCT FROM (OLD.content, OLD.tool_name, OLD.tool_calls)
"""

# Only canonical columns trigger UPDATE maintenance. The derived-only writes
# below and _ensure_display_order's backfill cannot recursively fire it.
_MESSAGE_TRIGGERS = (
    (
        "messages_display_order_insert",
        "INSERT",
        "NEW.display_order IS NULL",
        """
        UPDATE messages SET display_order = COALESCE((
            SELECT display_order FROM messages
            WHERE session_id = NEW.session_id AND id <> NEW.id
              AND (active = 1 OR compacted = 1)
              AND display_identity = NEW.display_identity AND display_order IS NOT NULL
            ORDER BY display_order LIMIT 1
        ), NEW.id) WHERE id = NEW.id;
        """,
    ),
    (
        "messages_projection_update",
        "UPDATE OF session_id, role, content, timestamp, tool_call_id, tool_calls, "
        "tool_name, display_kind, display_metadata, active, compacted",
        f"""({_DISPLAY_IDENTITY_CHANGED})
            OR NEW.session_id IS DISTINCT FROM OLD.session_id
            OR (NEW.active = 1 OR NEW.compacted = 1)
                IS DISTINCT FROM (OLD.active = 1 OR OLD.compacted = 1)""",
        f"""
        UPDATE messages SET
            display_identity = CASE WHEN {_DISPLAY_IDENTITY_CHANGED}
                THEN NULL ELSE display_identity END,
            display_order = NULL,
            fts_content = CASE WHEN id = NEW.id AND ({_SEARCH_TEXT_CHANGED})
                THEN NULL ELSE fts_content END
        WHERE id = NEW.id OR (
            session_id = OLD.session_id AND display_identity = OLD.display_identity
            AND (active = 1 OR compacted = 1)
        );
        """,
    ),
    (
        "messages_display_identity_delete",
        "DELETE",
        "OLD.active = 1 OR OLD.compacted = 1",
        """
        UPDATE messages SET display_order = NULL
        WHERE session_id = OLD.session_id AND display_identity = OLD.display_identity
          AND (active = 1 OR compacted = 1);
        """,
    ),
)


def _message_trigger_statements():
    for name, event, condition, body in _MESSAGE_TRIGGERS:
        # Keep procedural bodies intact: the declarative migration splitter
        # deliberately does not parse semicolons inside dollar-quoted SQL.
        yield f"""
            CREATE OR REPLACE FUNCTION hermes_{name}() RETURNS trigger
            LANGUAGE plpgsql AS $hermes$
            BEGIN
                {body}
                RETURN NULL;
            END;
            $hermes$
        """
        yield f"""
            CREATE OR REPLACE TRIGGER {name}
            AFTER {event} ON messages
            FOR EACH ROW WHEN ({condition})
            EXECUTE FUNCTION hermes_{name}()
        """


POSTGRES_MESSAGE_TRIGGER_SQL = tuple(_message_trigger_statements())


def install_postgres_message_triggers(conn: Any) -> None:
    """Install required row maintenance after columns and indexes exist.

    INSERT uses the identity index rather than scanning a conversation.
    UPDATE/DELETE invalidate affected display groups for the shared canonical
    backfill. Changed search text becomes NULL, so the existing completeness
    check chooses the all-row search path until its vector is rebuilt; a
    derived index can never keep stale text or abort the canonical write.
    """
    raw = conn.raw if hasattr(conn, "raw") else conn
    for statement in POSTGRES_MESSAGE_TRIGGER_SQL:
        raw.execute(statement)
