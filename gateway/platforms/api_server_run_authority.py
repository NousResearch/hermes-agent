"""Compact cancelled attempts only behind a proven newer room authority epoch."""
import hashlib
import json


def room_run_scope(claims):
    fields = ("room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
              "member_id", "target_install_id", "target_profile")
    return hashlib.sha256("\0".join(str(claims[k]) for k in fields).encode()).hexdigest()


def room_authority(claims):
    # Home remains part of the origin namespace. Only an authenticated target-owner
    # transition may bind another home to this lineage; an epoch alone is not proof.
    fields = ("room_id", "home_install_id", "member_id", "target_install_id", "target_profile")
    key = hashlib.sha256("\0".join(str(claims[k]) for k in fields).encode()).hexdigest()
    return key, int(claims["authority_epoch"]), str(claims["authority_gateway_id"])


def room_namespace(claims):
    fields = ("room_id", "member_id", "target_install_id", "target_profile")
    return hashlib.sha256("\0".join(str(claims[key]) for key in fields).encode()).hexdigest()


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_authorities (
        authority_key TEXT PRIMARY KEY, authority_epoch INTEGER NOT NULL,
        gateway_id TEXT NOT NULL, retired_through INTEGER NOT NULL DEFAULT 0)""")
    from hermes_cli.sqlite_util import add_column_if_missing
    add_column_if_missing(conn, "run_room_authorities", "home_key", "home_key TEXT")
    conn.execute("UPDATE run_room_authorities SET home_key=authority_key WHERE home_key IS NULL")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_authority_aliases (
        home_key TEXT PRIMARY KEY, authority_key TEXT NOT NULL, origin_home TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_namespaces (
        namespace_key TEXT PRIMARY KEY, authority_key TEXT NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS run_room_attempts ON run_idempotency(room_authority_key, room_authority_epoch)")


def canonical(conn, authority):
    """Resolve only aliases previously authorized by the target, including after restart."""
    key, epoch, gateway = authority
    row = conn.execute("SELECT authority_key FROM run_room_authority_aliases WHERE home_key=?", (key,)).fetchone()
    return (row[0] if row else key), epoch, gateway


def namespace_matches(conn, namespace, authority):
    if namespace is None:
        return True
    row = conn.execute("SELECT authority_key FROM run_room_namespaces WHERE namespace_key=?", (namespace,)).fetchone()
    return row is None or row[0] == canonical(conn, authority)[0]


def origin_home(conn, authority, home):
    """Preserve the original hidden member session only across an authorized home move."""
    row = conn.execute("SELECT origin_home FROM run_room_authority_aliases WHERE home_key=?", (authority[0],)).fetchone()
    return row[0] if row else home


def successor(conn, authority, previous):
    """Validate an explicit owner-authorized transition without joining unrelated origins."""
    current = canonical(conn, authority)
    if previous is None:
        return current
    predecessor = canonical(conn, previous)
    row = conn.execute("SELECT authority_epoch,gateway_id,home_key FROM run_room_authorities WHERE authority_key=?",
                       (predecessor[0],)).fetchone()
    # A lost invitation reply may be retried: reissuing the exact current authority
    # does not advance or join anything, and still requires target-owner authorization.
    if current[0] == predecessor[0] and row == (authority[1], authority[2], authority[0]):
        return current
    if row != (previous[1], previous[2], previous[0]) or authority[1] <= previous[1]:
        raise ValueError("previous room authority is not current")
    if current[0] != predecessor[0] and (
            current[0] != authority[0]
            or conn.execute("SELECT 1 FROM run_room_authorities WHERE authority_key=?", (current[0],)).fetchone()):
        raise ValueError("successor home already belongs to another room origin")
    return predecessor[0], authority[1], authority[2]


def retirement_allowed(conn, authority):
    """An old scope may retire itself, never a different owner at the current epoch."""
    key, epoch, gateway = canonical(conn, authority)
    row = conn.execute("SELECT authority_epoch,gateway_id,home_key FROM run_room_authorities WHERE authority_key=?",
                       (key,)).fetchone()
    return row is not None and (epoch < row[0] or (epoch, gateway, authority[0]) == row)


def superseded(conn, authority):
    if authority is None:
        return False
    key, epoch, gateway = canonical(conn, authority)
    row = conn.execute("SELECT authority_epoch,gateway_id,retired_through FROM run_room_authorities WHERE authority_key=?", (key,)).fetchone()
    return row is not None and (epoch <= row[2] or epoch < row[0] or (epoch == row[0] and gateway != row[1]))


def observe(conn, scope, authority, previous=None, previous_home=None, namespace=None):
    """Bind old receipts on authenticated observation, then advance one room/member watermark."""
    key, epoch, gateway = successor(conn, authority, previous)
    if not namespace_matches(conn, namespace, (key, epoch, gateway)):
        return False
    if previous is not None and key != authority[0]:
        if not previous_home:
            raise ValueError("previous room home is required")
        conn.execute("""INSERT INTO run_room_authority_aliases(home_key,authority_key,origin_home)
            VALUES(?,?,?) ON CONFLICT(home_key) DO NOTHING""",
            (authority[0], key, origin_home(conn, previous, previous_home)))
    conn.execute("""UPDATE run_idempotency SET room_authority_key=?,room_authority_epoch=?
        WHERE scope=? AND room_authority_key IS NULL""", (key, epoch, scope))
    if superseded(conn, authority):
        compact(conn, key)
        return False
    conn.execute("""INSERT INTO run_room_authorities(authority_key,authority_epoch,gateway_id,home_key) VALUES(?,?,?,?)
        ON CONFLICT(authority_key) DO UPDATE SET
        authority_epoch=excluded.authority_epoch,gateway_id=excluded.gateway_id,home_key=excluded.home_key""",
        (key, epoch, gateway, authority[0]))
    if namespace is not None:
        conn.execute("INSERT INTO run_room_namespaces(namespace_key,authority_key) VALUES(?,?) ON CONFLICT DO NOTHING",
                     (namespace, key))
    compact(conn, key)
    return True


def retire(conn, scope, authority):
    if not retirement_allowed(conn, authority):
        raise ValueError("room retirement authority does not match retained lineage")
    observe(conn, scope, authority)
    authority = canonical(conn, authority)
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
