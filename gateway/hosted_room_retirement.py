"""Exact authority-retirement obligations, independent of room/route retention.

Coordinates from a retained bearer identify cleanup; only the target's signed-grant
checks authorize it. Neither this journal nor an epoch floor proves a task never ran.
"""
import hashlib
import json

from gateway import hosted_rooms
from gateway.hosted_room_peer import _split_token
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

_TABLE = "hosted_room_authority_retirements"
_FIELDS = ("room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
           "member_id", "target_install_id", "target_profile")


def _rows(conn, room_id):
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)).fetchone() is None:
        return []
    if room_id is None:
        return conn.execute(f"SELECT retirement_id,value FROM {_TABLE}").fetchall()
    return conn.execute(f"SELECT retirement_id,value FROM {_TABLE} WHERE room_id=?", (room_id,)).fetchall()


def retain_link(db_path, link, *, proof_install_id=None):
    # The bearer has already been admitted to private route custody. Parsing it
    # does not grant any operation: settlement still requires a target response.
    claims = json.loads(_split_token(link.grant)[0].decode("ascii"))
    scope = {key: claims[key] for key in _FIELDS}
    if (type(scope["authority_epoch"]) is not int or scope["authority_epoch"] < 1
            or any(not isinstance(value, str) or not value for key, value in scope.items() if key != "authority_epoch")
            or (scope["room_id"], scope["member_id"], scope["target_profile"], scope["target_install_id"])
            != (link.room_id, link.member_id, link.target_profile, link.catalog.installation_id)):
        raise ValueError("retirement scope does not match retained route")
    identity = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    value = json.dumps({"scope": scope, "target_url": link.target_url, "grant": link.grant, "status": "pending",
                        "proof_install_id": proof_install_id})
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {_TABLE} (retirement_id TEXT PRIMARY KEY, room_id TEXT NOT NULL, value TEXT NOT NULL)")
        conn.execute(f"INSERT OR IGNORE INTO {_TABLE} VALUES (?,?,?)", (identity, link.room_id, value))
    return identity


def status(db_path, room_id):
    with hosted_rooms._transaction(db_path) as conn:
        rows = _rows(conn, room_id)
    return [{"retirement_id": row["retirement_id"], **value["scope"],
             "target_url": value["target_url"], "status": value["status"]}
            for row in rows for value in [json.loads(row["value"])]]


def _replace(db_path, identity, old, value):
    encoded = json.dumps(value)
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        changed = conn.execute(f"UPDATE {_TABLE} SET value=? WHERE retirement_id=? AND value=?",
                               (encoded, identity, old)).rowcount
    if changed != 1:
        raise RuntimeError("retirement changed; retry its current obligation")
    return encoded


def settle(db_path, room_id, identity, *, grant=None, client=None):
    with hosted_rooms._transaction(db_path) as conn:
        row = next((row for row in _rows(conn, room_id) if row["retirement_id"] == identity), None)
    if row is None:
        return  # Reply-lost replay: no retained obligation remains, not task-absence evidence.
    original = row["value"]
    value = json.loads(original)
    client = client or PeerRunsHTTPClient(base_url=value["target_url"], api_key="",
        **({"proof_install_id": value["proof_install_id"]} if value.get("proof_install_id") else {}))
    if grant is not None:
        if not isinstance(grant, str) or not grant or len(grant) > 16 * 1024:
            raise ValueError("invalid retirement grant")
        probe = client.probe(grant=grant)
        verified = {key: probe.get(key) for key in _FIELDS}
        verified["target_install_id"] = (probe.get("catalog") or {}).get("installation_id")
        if verified != value["scope"]:
            raise ValueError("retirement grant scope changed")
        value.update(grant=grant, status="pending")
        original = _replace(db_path, identity, original, value)
    try:
        result = client.revoke_grant(grant=value["grant"], retire_authority=True)
    except PeerRunsHTTPError as exc:
        from tui_gateway.hosted_room_service import _grant_revoke_is_terminal
        if not _grant_revoke_is_terminal(exc):
            raise
        value["status"] = "needs_reauthorization"
        _replace(db_path, identity, original, value)
        return
    if not isinstance(result, dict) or result.get("revoked") is not True:
        raise RuntimeError("target did not confirm grant revocation")
    if result.get("authority_retired") is not True:
        value["status"] = "needs_reauthorization"
        _replace(db_path, identity, original, value)
        return
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        conn.execute(f"DELETE FROM {_TABLE} WHERE retirement_id=? AND value=?", (identity, original))


def settle_control(service, params):
    room_id = params["room_id"]
    settle(service.db_path, room_id, params["retirement_id"], grant=params.get("grant"))
    return {"retirements": status(service.db_path, room_id)}
