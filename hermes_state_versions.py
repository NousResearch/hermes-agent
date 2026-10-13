"""Durable version identity for authored rows and explicit compaction copies."""

import sqlite3

MESSAGE_VERSION_UID = "message_version_uid"
# Presentation-only metadata and counters do not create a new authored version.
VERSION_COLUMNS = (
    "role", "content", "tool_call_id", "tool_calls", "tool_name", "effect_disposition",
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "absorbed_message_uids", "tool_call_uids", "tool_call_uid",
)
_VERSION_CHANGED = " OR ".join(f"NEW.{column} IS NOT OLD.{column}" for column in VERSION_COLUMNS)
_REGISTER_VERSION = """
    INSERT INTO message_versions (session_id, message_uid, version_uid)
    SELECT NEW.session_id, NEW.message_uid, NEW.message_version_uid
    WHERE NOT EXISTS (SELECT 1 FROM message_versions
        WHERE session_id = NEW.session_id AND message_uid = NEW.message_uid AND version_uid = NEW.message_version_uid);
"""
MESSAGE_VERSION_DDL = f"""
CREATE INDEX IF NOT EXISTS idx_message_versions_session_id ON message_versions(session_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_uid_version
    ON messages(message_uid, message_version_uid, session_id);
DROP TRIGGER IF EXISTS messages_version_insert;
CREATE TRIGGER messages_version_insert AFTER INSERT ON messages
WHEN NEW.message_version_uid IS NULL OR NEW.message_version_uid = ''
BEGIN
    UPDATE messages SET message_version_uid = lower(hex(randomblob(16))) WHERE id = NEW.id;
END;
DROP TRIGGER IF EXISTS messages_version_rewrite;
CREATE TRIGGER messages_version_rewrite AFTER UPDATE OF {", ".join(VERSION_COLUMNS)} ON messages
WHEN NEW.message_version_uid IS OLD.message_version_uid AND ({_VERSION_CHANGED})
BEGIN
    UPDATE messages SET message_version_uid = lower(hex(randomblob(16))) WHERE id = NEW.id;
END;
DROP TRIGGER IF EXISTS messages_version_register_insert;
CREATE TRIGGER messages_version_register_insert AFTER INSERT ON messages
WHEN NEW.message_uid IS NOT NULL AND NEW.message_version_uid IS NOT NULL AND NEW.message_version_uid != ''
BEGIN
{_REGISTER_VERSION}
END;
DROP TRIGGER IF EXISTS messages_version_register_update;
CREATE TRIGGER messages_version_register_update AFTER UPDATE OF session_id, message_uid, message_version_uid ON messages
WHEN NEW.message_uid IS NOT NULL AND NEW.message_version_uid IS NOT NULL AND NEW.message_version_uid != ''
BEGIN
{_REGISTER_VERSION}
END;
"""


def retained_version(conn: sqlite3.Connection, values: dict, *, copy_version: bool, source_row_id=None) -> str | None:
    """Only an explicit copy or an unchanged ordinary reissue may retain a version."""
    version = values.get(MESSAGE_VERSION_UID)
    if not isinstance(version, str) or not version:
        return None
    if copy_version:
        return version
    # An addressed copy compares with that source, not an arbitrary older payload from the same version.
    addressed = isinstance(source_row_id, int) and not isinstance(source_row_id, bool)
    donor = conn.execute(
        f"SELECT {', '.join(VERSION_COLUMNS)} FROM messages WHERE message_uid = ? AND message_version_uid = ?"
        f"{' AND id = ?' if addressed else ''} ORDER BY id DESC LIMIT 1",
        (values["message_uid"], version, *((source_row_id,) if addressed else ())),
    ).fetchone()
    return version if donor is not None and all(donor[column] == values[column] for column in VERSION_COLUMNS) else None


def versioned_insert_params(
    conn: sqlite3.Connection, params: tuple, columns: tuple[str, ...], *, copy_version: bool, source_row_id=None,
) -> tuple:
    """Bind the version by the shared INSERT shape; NULL asks SQLite to mint it atomically."""
    version = retained_version(conn, dict(zip(columns, params)), copy_version=copy_version, source_row_id=source_row_id)
    index = columns.index(MESSAGE_VERSION_UID)
    return (*params[:index], version, *params[index + 1:])


def prepare_replacement_versions(conn: sqlite3.Connection, messages: list[dict], serialize) -> None:
    """Validate before destructive replacement removes the only durable version donors."""
    for message in messages:
        version = retained_version(conn, serialize(message), copy_version=False, source_row_id=message.get("_row_id"))
        if version is None:
            message.pop(MESSAGE_VERSION_UID, None)
        else:
            message[MESSAGE_VERSION_UID] = version


def message_version_column(db) -> str:
    """Read-only legacy stores have no version column and must not be migrated just to project history."""
    if db.read_only:
        with db._read_ctx() as conn:
            if MESSAGE_VERSION_UID not in db._message_column_names(conn):
                return f"NULL AS {MESSAGE_VERSION_UID}"
    return MESSAGE_VERSION_UID
