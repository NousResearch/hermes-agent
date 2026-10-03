"""Per-installation Ed25519 room identity keys, pinned per Group Chat.

A Group Chat's custodians authenticate to one another with these keys: catch-up pages pulled from
another custodian, and the promises and proofs of succession. Each installation has one key, derived
from the RoomLink secret of its installation root by a domain-separated HMAC as copy retirement
derives its keys, so no new private key is stored and every profile and gateway process of the
installation (standalone or one multiplex gateway) signs as the same installation, like its
install id. Only the public key travels.

A key is pinned per (room, installation) at custody enrollment: the home pins a member
installation's key from its authenticated capabilities probe, and every custodian pins the voters'
keys from the room's configuration events, which reach it through its own grant. ``verify`` uses
the pinned key only. A different key for an installation already pinned is refused, never adopted.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import DbPath, compact_json, table_exists

PINS_TABLE = "hosted_room_identity_pins"
_SEED_DOMAIN = b"hermes.group.room-identity.seed.v1\0"
_SIGNATURE_PREFIX = "ed25519-v1."
_SIGNATURE_RE = re.compile(r"ed25519-v1\.[A-Za-z0-9_-]{86}")
_PUBLIC_KEY_RE = re.compile(r"[0-9a-f]{64}")
_DOMAIN_RE = re.compile(rb"[a-z0-9][a-z0-9._-]{0,127}")


class RoomIdentityError(rooms.HostedRoomError):
    """A room identity key is invalid, unavailable or conflicts with the pinned one."""

    reason = "room_identity_conflict"


def _private_key(secret: bytes | None = None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from gateway.hosted_room_peer import gateway_room_grant_secret
    from hermes_constants import get_default_hermes_root

    # The installation root's secret, whichever profile home this process was started in.
    secret = gateway_room_grant_secret(get_default_hermes_root()) if secret is None else secret
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise RoomIdentityError("room identity key is unavailable")
    return Ed25519PrivateKey.from_private_bytes(hmac.new(secret, _SEED_DOMAIN, hashlib.sha256).digest())


def _message(domain: bytes, payload: Mapping[str, Any]) -> bytes:
    """Distinct domains per message kind; the payload is canonical JSON."""
    if not isinstance(domain, bytes) or _DOMAIN_RE.fullmatch(domain) is None:
        raise RoomIdentityError("signature domain is invalid")
    if not isinstance(payload, Mapping):
        raise RoomIdentityError("signed payload must be an object")
    return domain + b"\0" + compact_json(dict(payload)).encode("utf-8")


def public_key_of(value: Any) -> str:
    """A validated hex Ed25519 public key."""
    if not isinstance(value, str) or _PUBLIC_KEY_RE.fullmatch(value) is None:
        raise RoomIdentityError("room identity public key is invalid")
    return value


def local_public_key(*, secret: bytes | None = None) -> str:
    """This installation's room identity public key (hex)."""
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return _private_key(secret).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def sign(domain: bytes, payload: Mapping[str, Any], *, secret: bytes | None = None) -> str:
    """Sign canonical ``payload`` under ``domain`` with this installation's key."""
    signature = _private_key(secret).sign(_message(domain, payload))
    return _SIGNATURE_PREFIX + base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")


def verify_key(public_key: str, domain: bytes, payload: Mapping[str, Any], signature: Any) -> bool:
    """Check ``signature`` against one known public key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(signature, str) or _SIGNATURE_RE.fullmatch(signature) is None:
        return False
    try:
        raw = base64.urlsafe_b64decode(signature[len(_SIGNATURE_PREFIX):] + "==")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_of(public_key))).verify(
            raw, _message(domain, payload))
        return True
    except (InvalidSignature, RoomIdentityError, ValueError, TypeError):
        return False


def initialize_locked(conn: sqlite3.Connection) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {PINS_TABLE} (
        room_id TEXT NOT NULL, install_id TEXT NOT NULL, public_key TEXT NOT NULL,
        source TEXT NOT NULL, pinned_at REAL NOT NULL, PRIMARY KEY (room_id, install_id))""")


def pin_locked(conn: sqlite3.Connection, *, room_id: str, install_id: str, public_key: str, source: str) -> bool:
    """Pin ``public_key`` for this room and installation; True when it was new.

    The same key again is a no-op. A different key for a pinned installation is refused: rotation
    would need its own enrollment, and an unannounced change looks exactly like impersonation.
    """
    public_key_of(public_key)
    initialize_locked(conn)
    current = pinned_key_locked(conn, room_id=room_id, install_id=install_id)
    if current is not None:
        if not hmac.compare_digest(current, public_key):
            raise RoomIdentityError("a different room identity key is already pinned for this installation")
        return False
    conn.execute(f"INSERT INTO {PINS_TABLE} (room_id, install_id, public_key, source, pinned_at) VALUES (?,?,?,?,?)",
                 (room_id, install_id, public_key, source, time.time()))
    return True


def pinned_key_locked(conn: sqlite3.Connection, *, room_id: str, install_id: str) -> str | None:
    if not table_exists(conn, PINS_TABLE):
        return None
    row = conn.execute(f"SELECT public_key FROM {PINS_TABLE} WHERE room_id=? AND install_id=?",
                       (room_id, install_id)).fetchone()
    return str(row[0]) if row is not None else None


def verify_locked(
    conn: sqlite3.Connection, room_id: str, install_id: str, domain: bytes, payload: Mapping[str, Any],
    signature: Any,
) -> bool:
    """Verify with the key pinned for ``(room_id, install_id)``, inside the caller's transaction."""
    key = pinned_key_locked(conn, room_id=room_id, install_id=install_id)
    return key is not None and verify_key(key, domain, payload, signature)


def verify(
    room_id: str, install_id: str, domain: bytes, payload: Mapping[str, Any], signature: Any, *, db_path: DbPath,
) -> bool:
    """Verify ``signature`` with the key ``db_path`` pinned for ``(room_id, install_id)``.

    ``db_path`` is the store that holds the room or its copy: the canonical store of the default
    profile, or the legacy ``shared-state.db``.
    """
    path = Path(db_path)
    if not path.is_file():
        return False
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
            return verify_locked(conn, room_id, install_id, domain, payload, signature)
    except sqlite3.Error:
        return False
