"""Compact cancelled attempts only behind a proven newer room authority epoch."""
import hashlib
import json


def room_run_scope(claims):
    fields = ("room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
              "member_id", "target_install_id", "target_profile")
    return hashlib.sha256("\0".join(str(claims[k]) for k in fields).encode()).hexdigest()


def room_authority(claims):
    # Gateway identity may change on takeover; the room's epoch orders both owners.
    fields = ("room_id", "home_install_id", "member_id", "target_install_id", "target_profile")
    key = hashlib.sha256("\0".join(str(claims[k]) for k in fields).encode()).hexdigest()
    return key, int(claims["authority_epoch"]), str(claims["authority_gateway_id"])


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_authorities (
        authority_key TEXT PRIMARY KEY, authority_epoch INTEGER NOT NULL,
        gateway_id TEXT NOT NULL, retired_through INTEGER NOT NULL DEFAULT 0)""")
    conn.execute("CREATE INDEX IF NOT EXISTS run_room_attempts ON run_idempotency(room_authority_key, room_authority_epoch)")


def superseded(conn, authority):
    if authority is None:
        return False
    key, epoch, gateway = authority
    row = conn.execute("SELECT authority_epoch,gateway_id,retired_through FROM run_room_authorities WHERE authority_key=?", (key,)).fetchone()
    return row is not None and (epoch <= row[2] or epoch < row[0] or (epoch == row[0] and gateway != row[1]))


def observe(conn, scope, authority):
    """Bind old receipts on authenticated observation, then advance one room/member watermark."""
    key, epoch, gateway = authority
    conn.execute("""UPDATE run_idempotency SET room_authority_key=?,room_authority_epoch=?
        WHERE scope=? AND room_authority_key IS NULL""", (key, epoch, scope))
    if superseded(conn, authority):
        compact(conn, key)
        return False
    conn.execute("""INSERT INTO run_room_authorities(authority_key,authority_epoch,gateway_id) VALUES(?,?,?)
        ON CONFLICT(authority_key) DO UPDATE SET
        authority_epoch=excluded.authority_epoch,gateway_id=excluded.gateway_id""", (key, epoch, gateway))
    compact(conn, key)
    return True


def retire(conn, scope, authority):
    observe(conn, scope, authority)
    conn.execute("UPDATE run_room_authorities SET retired_through=MAX(retired_through,?) WHERE authority_key=?",
                 (authority[1], authority[0]))
    compact(conn, authority[0])


def compact(conn, authority_key):
    """Drop cancelled terminal receipts; live executors retain their stop bit and status."""
    rows = conn.execute("""SELECT scope,idempotency_key,status_json,stop_requested
        FROM run_idempotency WHERE room_authority_key=? AND (room_authority_epoch<
        (SELECT authority_epoch FROM run_room_authorities WHERE authority_key=?) OR room_authority_epoch<=
        (SELECT retired_through FROM run_room_authorities WHERE authority_key=?))""",
        (authority_key, authority_key, authority_key)).fetchall()
    for scope, key, encoded, stopped in rows:
        try:
            status = json.loads(encoded)
        except (ValueError, TypeError):
            continue
        if not isinstance(status, dict):
            continue
        if (status.get("status") in {"completed", "failed", "cancelled", "interrupted"}
                and (stopped or status.get("admission_cancelled"))):
            conn.execute("DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (scope, key))
