"""Target room authority shared by the members affected by one peer reservation."""
import hashlib


def target_key(claims):
    fields = ("room_id", "target_install_id", "target_profile")
    return hashlib.sha256("\0".join(str(claims[key]) for key in fields).encode()).hexdigest()


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_origins (
        target_key TEXT PRIMARY KEY, origin_home TEXT NOT NULL, home_install_id TEXT NOT NULL,
        authority_epoch INTEGER NOT NULL, gateway_id TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_origin_homes (
        target_key TEXT NOT NULL, home_install_id TEXT NOT NULL, PRIMARY KEY(target_key,home_install_id))""")


def retained(conn, claims):
    return conn.execute("SELECT origin_home,home_install_id,authority_epoch,gateway_id FROM run_room_origins WHERE target_key=?",
                        (target_key(claims),)).fetchone()


def accepts(conn, claims, previous_home=None, verified_origin=None):
    row = retained(conn, claims)
    if row is None:
        return True
    origin, home, epoch, gateway = row
    incoming = int(claims["authority_epoch"])
    if incoming < epoch:
        return False
    if (claims["home_install_id"], incoming, claims["authority_gateway_id"]) == (home, epoch, gateway):
        return True
    if incoming == epoch:
        return verified_origin == origin
    if claims["home_install_id"] == home or verified_origin == origin:
        return True
    return previous_home is not None and conn.execute("""SELECT 1 FROM run_room_origin_homes
        WHERE target_key=? AND home_install_id=?""", (target_key(claims), previous_home)).fetchone() is not None


def observe(conn, claims, origin):
    key = target_key(claims)
    conn.execute("""INSERT INTO run_room_origins(target_key,origin_home,home_install_id,authority_epoch,gateway_id)
        VALUES(?,?,?,?,?) ON CONFLICT(target_key) DO UPDATE SET home_install_id=excluded.home_install_id,
        authority_epoch=excluded.authority_epoch,gateway_id=excluded.gateway_id""",
        (key, origin, claims["home_install_id"], int(claims["authority_epoch"]), claims["authority_gateway_id"]))
    conn.executemany("INSERT INTO run_room_origin_homes(target_key,home_install_id) VALUES(?,?) ON CONFLICT DO NOTHING",
                     [(key, origin), (key, claims["home_install_id"])])
    return key


def effective(conn, authority_key):
    row = conn.execute("""SELECT authority_epoch,gateway_id,retired_through,target_key
        FROM run_room_authorities WHERE authority_key=?""", (authority_key,)).fetchone()
    if row is None:
        return None
    shared = conn.execute("SELECT authority_epoch,gateway_id FROM run_room_origins WHERE target_key=?", (row[3],)).fetchone()
    epoch, gateway = shared if shared is not None and shared[0] >= row[0] else row[:2]
    return epoch, gateway, row[2]
