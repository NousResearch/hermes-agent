"""Keyed pseudonymization of the shared-metrics install identity.

``install_id`` is a persistent, profile-scoped identifier. It is deliberately
NOT transmitted: ``docs/observability/relay-shared-metrics.md`` commits that a
remote exporter "must not reuse the persistent local identifier by default".

Each transmitted package instead carries::

    HMAC-SHA256(key=rotation_salt, message=install_id)

where ``rotation_salt`` is generated locally, never leaves the machine, and
rotates on a fixed schedule. Within a rotation window the value is stable, so
distinct installs stay countable — the primary analytical question. Across
windows it changes, bounding long-term linkability.

The derivation is one-way: the service cannot recover ``install_id`` from what
it receives.

See Appendix A.2 and A.3 of the doc above for the decision record.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

#: Salt lifetime. Matches local history retention so the two ages line up.
ROTATION_INTERVAL = timedelta(days=30)

#: ``telemetry_state`` keys. The salt lives in the same store as install_id, so
#: deleting the shared-metrics directory resets both together — the documented
#: reset behaviour keeps working without a second cleanup path.
SALT_KEY = "send_rotation_salt"
SALT_ISSUED_AT_KEY = "send_rotation_salt_issued_at"

_SALT_BYTES = 32


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM telemetry_state WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    # sqlite3.Row and plain tuples both index by position.
    return str(row[0])


def _write(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO telemetry_state(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def current_salt(
    connection: sqlite3.Connection,
    *,
    now: datetime | None = None,
) -> str:
    """Return the active salt, generating or rotating it when due.

    Must be called inside a write transaction: it can write to
    ``telemetry_state``.
    """
    moment = now or datetime.now(timezone.utc)
    salt = _read(connection, SALT_KEY)
    issued_at = _parse(_read(connection, SALT_ISSUED_AT_KEY))

    fresh = (
        salt is not None
        and issued_at is not None
        # Strictly within the window. A future issued_at means the clock moved
        # backwards (or the value was tampered with), so the recorded age
        # cannot be trusted and we reissue rather than keep using a salt of
        # unknown vintage. Reissuing is the safe direction: it shortens
        # linkability, and already-prepared packages keep their frozen
        # identifier so retries stay byte-identical.
        and issued_at <= moment < issued_at + ROTATION_INTERVAL
    )
    if fresh:
        return str(salt)

    salt = secrets.token_hex(_SALT_BYTES)
    _write(connection, SALT_KEY, salt)
    _write(connection, SALT_ISSUED_AT_KEY, _isoformat(moment))
    return salt


def derive_install_id(install_id: str, salt: str) -> str:
    """Return the transmitted identifier for ``install_id`` under ``salt``."""
    return hmac.new(
        salt.encode("utf-8"),
        install_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def substitute_install_id(payload: dict, derived: str) -> dict:
    """Return ``payload`` with its ``install_id`` replaced by ``derived``.

    This is the ONLY field the exporter changes. Everything else is
    transmitted exactly as the generator wrote it, so payload schema evolution
    stays a sender-side concern. A shallow copy is enough — only a top-level
    key is replaced — and the caller's dict is left untouched.
    """
    updated = dict(payload)
    updated["install_id"] = derived
    return updated
