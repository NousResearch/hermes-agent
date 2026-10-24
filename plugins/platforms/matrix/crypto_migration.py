"""One-time migration of a mautrix crypto store from libolm pickles to vodozemac pickles.

Runs only when the vodozemac backend is active (``crypto_backend``). Order of operations in
``MatrixAdapter._connect_setup_e2ee``:

1. :func:`backup_if_libolm_store` -- before the database is opened, copy ``crypto.db`` to
   ``crypto.db.libolm-backup-<UTC timestamp>`` (0600) if it still holds libolm pickles.
   The migration is one-way, so this copy is the rollback path (downgrade Hermes and
   restore it as ``crypto.db``).
2. The existing pickle-key migration (``_migrate_legacy_crypto_pickle``) runs unchanged.
3. :func:`migrate_store_to_vodozemac` -- re-pickles the Olm account, Olm sessions and inbound
   Megolm sessions in vodozemac format (sessions first, account last, so an interrupted run
   is simply retried), and deletes libolm *outbound* Megolm sessions, which vodozemac cannot
   import. Deleting those loses nothing: recipients already hold the matching inbound keys,
   and mautrix creates and shares a fresh outbound session on the next encrypted send.

Identity keys, device ID, cross-signing state, known devices and every inbound Megolm session
(i.e. readable history) are preserved.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_SESSION_TABLES = ("crypto_olm_session", "crypto_megolm_inbound_session")
_OUTBOUND_TABLE = "crypto_megolm_outbound_session"


@dataclass
class MigrationReport:
    account: bool = False
    olm_sessions: int = 0
    inbound_group_sessions: int = 0
    outbound_group_sessions_dropped: int = 0
    unreadable: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.account or self.olm_sessions or self.inbound_group_sessions
                    or self.outbound_group_sessions_dropped)


def _is_vodozemac_pickle(vz_cls: Any, pickled: bytes, pickle_key: str) -> bool:
    from plugins.platforms.matrix.olm_vodozemac import _pickle_key
    try:
        vz_cls.from_pickle(bytes(pickled).decode("utf-8"), _pickle_key(pickle_key))
        return True
    except ValueError:  # vodozemac.PickleException / UnicodeDecodeError are ValueErrors
        return False


def backup_if_libolm_store(db_path: Path, pickle_keys: Iterable[str]) -> Path | None:
    """Copy *db_path* aside if any Olm account in it is not a vodozemac pickle under *pickle_keys*.

    Synchronous and side-effect free unless a backup is written. Never raises.
    """
    if not db_path.exists():
        return None
    try:
        import vodozemac
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='crypto_account'").fetchone()
            rows = conn.execute("SELECT account FROM crypto_account").fetchall() if has_table else []
        keys = [k for k in pickle_keys if k]
        legacy = [row[0] for row in rows if row[0] is not None and not any(
            _is_vodozemac_pickle(vodozemac.Account, row[0], key) for key in keys)]
        if not legacy:
            return None
        backup = db_path.with_name(f"{db_path.name}.libolm-backup-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        with sqlite3.connect(db_path) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)  # consistent copy, includes WAL contents
        os.chmod(backup, 0o600)
        logger.info("Matrix: backed up libolm crypto store to %s before migrating it to vodozemac", backup)
        return backup
    except (ImportError, OSError, sqlite3.Error) as exc:
        logger.warning("Matrix: could not back up crypto store before vodozemac migration: %s", exc)
        return None


async def migrate_store_to_vodozemac(crypto_db: Any, account_id: str, pickle_key: str) -> MigrationReport:
    """Re-pickle libolm rows of *account_id* as vodozemac pickles (see module docstring)."""
    import vodozemac
    from plugins.platforms.matrix import olm_vodozemac as shim

    report = MigrationReport()
    row = await crypto_db.fetchrow("SELECT account FROM crypto_account WHERE account_id=$1", account_id)
    if row is None or row["account"] is None:
        return report
    account_blob = bytes(row["account"])
    account_is_vz = _is_vodozemac_pickle(vodozemac.Account, account_blob, pickle_key)

    classes = {
        "crypto_olm_session": (vodozemac.Session, shim.Session),
        "crypto_megolm_inbound_session": (vodozemac.InboundGroupSession, shim.InboundGroupSession),
    }
    for table in _SESSION_TABLES:
        vz_cls, shim_cls = classes[table]
        rows = await crypto_db.fetch(
            f"SELECT session_id, session FROM {table} WHERE account_id=$1", account_id)
        for session_row in rows:
            if session_row["session"] is None:
                continue
            pickled = bytes(session_row["session"])
            if _is_vodozemac_pickle(vz_cls, pickled, pickle_key):
                continue
            try:
                session = shim_cls.from_pickle(pickled, pickle_key)  # falls back to the libolm format
            except (shim.OlmSessionError, shim.OlmGroupSessionError, ValueError) as exc:
                report.unreadable += 1
                logger.warning("Matrix: %s row %s is neither a vodozemac nor a libolm pickle under the "
                               "current key; leaving it untouched: %s", table, session_row["session_id"], exc)
                continue
            await crypto_db.execute(
                f"UPDATE {table} SET session=$1 WHERE account_id=$2 AND session_id=$3",
                session.pickle(pickle_key), account_id, session_row["session_id"])
            if table == "crypto_olm_session":
                report.olm_sessions += 1
            else:
                report.inbound_group_sessions += 1

    rows = await crypto_db.fetch(
        f"SELECT room_id, session FROM {_OUTBOUND_TABLE} WHERE account_id=$1", account_id)
    for outbound in rows:
        if outbound["session"] is not None and _is_vodozemac_pickle(
                vodozemac.GroupSession, bytes(outbound["session"]), pickle_key):
            continue
        await crypto_db.execute(
            f"DELETE FROM {_OUTBOUND_TABLE} WHERE account_id=$1 AND room_id=$2",
            account_id, outbound["room_id"])
        report.outbound_group_sessions_dropped += 1

    if not account_is_vz:  # account last: it is the "migration done" marker
        try:
            account = shim.Account.from_pickle(account_blob, pickle_key)
        except (shim.OlmAccountError, ValueError) as exc:
            logger.error("Matrix: Olm account is not readable as a libolm pickle under the current "
                         "key; not migrating it: %s", exc)
            report.unreadable += 1
            return report
        await crypto_db.execute("UPDATE crypto_account SET account=$1 WHERE account_id=$2",
                                account.pickle(pickle_key), account_id)
        report.account = True

    if report.changed:
        logger.info(
            "Matrix: migrated crypto store to vodozemac (account=%s, olm sessions=%d, inbound megolm "
            "sessions=%d, libolm outbound megolm sessions rotated=%d, unreadable rows left=%d)",
            report.account, report.olm_sessions, report.inbound_group_sessions,
            report.outbound_group_sessions_dropped, report.unreadable)
    return report
