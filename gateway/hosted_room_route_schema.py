"""RoomLink retirement and scoped-grant schema additions."""

import sqlite3

from gateway.hosted_rooms_common import table_columns


def initialize_route_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_disband_fences (
            room_id TEXT PRIMARY KEY,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            started_at REAL NOT NULL,
            revocation_complete_at REAL
        )"""
    )
    disband_fence_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(hosted_room_disband_fences)")
    }
    if "revocation_complete_at" not in disband_fence_columns:
        conn.execute(
            "ALTER TABLE hosted_room_disband_fences "
            "ADD COLUMN revocation_complete_at REAL"
        )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_revoked_grant_ids (
            scope_key TEXT NOT NULL,
            grant_id TEXT NOT NULL,
            expires_at REAL NOT NULL,
            PRIMARY KEY (scope_key, grant_id)
        )"""
    )
    peer_reservation_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_peer_reservations)")
    }
    if "mutation_id" not in peer_reservation_columns:
        conn.execute(
            "ALTER TABLE hosted_room_peer_reservations "
            "ADD COLUMN mutation_id TEXT NOT NULL DEFAULT 'legacy'"
        )
    conn.execute("CREATE TRIGGER IF NOT EXISTS trg_hosted_room_links_reject_fenced_insert\n           BEFORE INSERT ON hosted_room_links\n           WHEN EXISTS (\n               SELECT 1 FROM hosted_room_disband_fences\n                WHERE room_id=NEW.room_id\n           )\n           OR EXISTS (\n               SELECT 1 FROM hosted_rooms\n                WHERE room_id=NEW.room_id AND disbanded_at IS NOT NULL\n           )\n           OR EXISTS (\n               SELECT 1 FROM hosted_room_retired_ids\n                WHERE room_id=NEW.room_id\n           )\n           BEGIN\n               SELECT RAISE(ABORT, 'Group Chat route registration is fenced');\n           END")
    conn.execute("CREATE TRIGGER IF NOT EXISTS trg_hosted_room_links_reject_fenced_update\n           BEFORE UPDATE ON hosted_room_links\n           WHEN EXISTS (\n               SELECT 1 FROM hosted_room_disband_fences\n                WHERE room_id=NEW.room_id\n           )\n           OR EXISTS (\n               SELECT 1 FROM hosted_rooms\n                WHERE room_id=NEW.room_id AND disbanded_at IS NOT NULL\n           )\n           OR EXISTS (\n               SELECT 1 FROM hosted_room_retired_ids\n                WHERE room_id=NEW.room_id\n           )\n           BEGIN\n               SELECT RAISE(ABORT, 'Group Chat route registration is fenced');\n           END")
    conn.execute("CREATE TRIGGER IF NOT EXISTS trg_hosted_room_links_reject_unrevoked_delete\n           BEFORE DELETE ON hosted_room_links\n           WHEN NOT EXISTS (\n               SELECT 1 FROM hosted_room_disband_fences\n                WHERE room_id=OLD.room_id AND revocation_complete_at IS NOT NULL\n           )\n           BEGIN\n               SELECT RAISE(ABORT, 'Group Chat routes are not revoked');\n           END")
    conn.execute("CREATE TRIGGER IF NOT EXISTS trg_hosted_peer_reservation_nonce_insert\n           AFTER INSERT ON hosted_room_peer_reservations\n           WHEN NEW.mutation_id='legacy'\n           BEGIN\n               UPDATE hosted_room_peer_reservations\n                  SET mutation_id=lower(hex(randomblob(16)))\n                WHERE room_id=NEW.room_id AND member_id=NEW.member_id\n                  AND target_profile=NEW.target_profile;\n           END")
    conn.execute('CREATE TRIGGER IF NOT EXISTS trg_hosted_peer_reservation_nonce_update\n           AFTER UPDATE ON hosted_room_peer_reservations\n           WHEN NEW.mutation_id=OLD.mutation_id\n           BEGIN\n               UPDATE hosted_room_peer_reservations\n                  SET mutation_id=lower(hex(randomblob(16)))\n                WHERE room_id=NEW.room_id AND member_id=NEW.member_id\n                  AND target_profile=NEW.target_profile;\n           END')


def route_schema_is_current(conn: sqlite3.Connection) -> bool:
    tables = {
        "hosted_room_disband_fences": {"room_id", "authority_gateway_id", "authority_epoch", "started_at", "revocation_complete_at"},
        "hosted_room_revoked_grant_ids": {"scope_key", "grant_id", "expires_at"},
        "hosted_room_peer_reservations": {"mutation_id"},
    }
    triggers = {
        "trg_hosted_room_links_reject_fenced_insert", "trg_hosted_room_links_reject_fenced_update",
        "trg_hosted_room_links_reject_unrevoked_delete", "trg_hosted_peer_reservation_nonce_insert",
        "trg_hosted_peer_reservation_nonce_update",
    }
    actual = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    pk = sorted((row for row in conn.execute("PRAGMA table_info(hosted_room_revoked_grant_ids)") if row[5]), key=lambda row: row[5])
    return (all(columns.issubset(table_columns(conn, table)) for table, columns in tables.items())
            and tuple(row[1] for row in pk) == ("scope_key", "grant_id") and triggers.issubset(actual))
