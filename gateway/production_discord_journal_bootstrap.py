"""Explicit create-only bootstrap for production Discord durable journals.

Normal connector and route-back startup deliberately refuse to create journal
state.  This module is the narrow, mechanical pre-start edge: it may create a
fresh journal at one pinned production path, or re-observe an already-created
but still empty journal after an interrupted bootstrap.  It never adopts a
non-empty journal and never reads Discord credentials or signing keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.discord_connector_service import DurableDiscordConnectorJournal
from gateway.discord_edge_runtime import DurableDiscordEdgeJournal


CONNECTOR_JOURNAL_PATH = Path(
    "/var/lib/muncho-discord-connector/connector.sqlite3"
)
ROUTEBACK_JOURNAL_PATH = Path(
    "/var/lib/muncho-discord-egress/discord-edge-journal.sqlite3"
)
BOOTSTRAP_SCHEMA = "muncho-production-discord-journal-bootstrap.v1"
_KINDS = frozenset({"connector", "routeback"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EDGE_TABLES = frozenset(
    {
        "discord_edge_journal_meta_v1",
        "discord_edge_idempotency_v1",
        "discord_edge_receipt_history_v1",
    }
)
_CONNECTOR_TABLES = frozenset(
    {"connector_meta_v1", "connector_events_v1", "connector_sends_v1"}
)
_EDGE_MARKER_PREFIX = "discord-edge-journal.v1 "


class ProductionDiscordJournalBootstrapError(RuntimeError):
    """Stable, secret-free bootstrap failure."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    return {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _expected_path(kind: str) -> Path:
    if kind == "connector":
        return CONNECTOR_JOURNAL_PATH
    if kind == "routeback":
        return ROUTEBACK_JOURNAL_PATH
    raise ProductionDiscordJournalBootstrapError("journal_kind_invalid")


def _secure_parent(path: Path) -> tuple[int, int]:
    try:
        observed = os.lstat(path.parent)
    except OSError as exc:
        raise ProductionDiscordJournalBootstrapError(
            "journal_parent_invalid"
        ) from exc
    effective_uid = int(os.geteuid())
    effective_gid = int(os.getegid())
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != effective_uid
        or observed.st_gid != effective_gid
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ProductionDiscordJournalBootstrapError("journal_parent_invalid")
    return effective_uid, effective_gid


def _secure_file(path: Path, *, uid: int, gid: int) -> os.stat_result:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise ProductionDiscordJournalBootstrapError(
            "journal_file_invalid"
        ) from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != uid
        or observed.st_gid != gid
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise ProductionDiscordJournalBootstrapError("journal_file_invalid")
    return observed


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _intent_path(path: Path) -> Path:
    return Path(f"{path}.bootstrap-intent")


def _ensure_intent(
    path: Path,
    *,
    kind: str,
    intent_sha256: str,
    uid: int,
    gid: int,
) -> Path:
    if _SHA256.fullmatch(intent_sha256) is None:
        raise ProductionDiscordJournalBootstrapError("journal_intent_invalid")
    target = _intent_path(path)
    payload = _canonical_bytes(
        {
            "schema": "muncho-production-discord-journal-bootstrap-intent.v1",
            "kind": kind,
            "path": str(path),
            "intent_sha256": intent_sha256,
            "pre_state": "absent",
            "secret_material_recorded": False,
        }
    ) + b"\n"
    if not os.path.lexists(target):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short bootstrap intent write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            try:
                os.unlink(target)
            except OSError:
                pass
            raise
        finally:
            os.close(descriptor)
        _fsync_parent(target)
    observed = _secure_file(target, uid=uid, gid=gid)
    if observed.st_size != len(payload) or target.read_bytes() != payload:
        raise ProductionDiscordJournalBootstrapError("journal_intent_invalid")
    return target


def _database_shape(path: Path) -> tuple[frozenset[str], int, str, str]:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=5,
            isolation_level=None,
        )
        tables = frozenset(
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                 WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
            )
        )
        disallowed_objects = int(
            connection.execute(
                """
                SELECT count(*) FROM sqlite_master
                 WHERE type NOT IN ('table','index')
                    OR (type='index' AND sql IS NOT NULL)
                """
            ).fetchone()[0]
        )
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        quick = str(connection.execute("PRAGMA quick_check(1)").fetchone()[0]).lower()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        raise ProductionDiscordJournalBootstrapError(
            "journal_partial_bootstrap_invalid"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if disallowed_objects != 0 or quick != "ok":
        raise ProductionDiscordJournalBootstrapError(
            "journal_partial_bootstrap_invalid"
        )
    return tables, version, mode, quick


def _table_count_if_present(path: Path, table: str) -> int:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as connection:
            return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise ProductionDiscordJournalBootstrapError(
            "journal_partial_bootstrap_invalid"
        ) from exc


def _remove_clean_partial(
    path: Path,
    *,
    allowed_tables: frozenset[str],
    uid: int,
    gid: int,
) -> None:
    tables, version, _mode, _quick = _database_shape(path)
    if not tables.issubset(allowed_tables) or version not in {0, 2}:
        raise ProductionDiscordJournalBootstrapError(
            "journal_partial_bootstrap_invalid"
        )
    for table in tables:
        if table.endswith("_meta_v1"):
            continue
        if _table_count_if_present(path, table) != 0:
            raise ProductionDiscordJournalBootstrapError("journal_not_clean")
    for candidate in (Path(f"{path}-wal"), Path(f"{path}-shm"), path):
        if not os.path.lexists(candidate):
            continue
        observed = os.lstat(candidate)
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_uid != uid
            or observed.st_gid != gid
            or stat.S_IMODE(observed.st_mode) != 0o600
        ):
            raise ProductionDiscordJournalBootstrapError(
                "journal_partial_bootstrap_invalid"
            )
        candidate.unlink()
    _fsync_parent(path)


def _recover_routeback_database_only(path: Path, *, uid: int, gid: int) -> bool:
    """Recover the exact crash window after DB commit and before marker create."""

    _secure_file(path, uid=uid, gid=gid)
    tables, version, mode, _quick = _database_shape(path)
    if tables != _EDGE_TABLES or version != 2 or mode != "wal":
        return False
    if any(
        _table_count_if_present(path, table) != 0
        for table in (
            "discord_edge_idempotency_v1",
            "discord_edge_receipt_history_v1",
        )
    ):
        raise ProductionDiscordJournalBootstrapError("journal_not_clean")
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5) as connection:
            rows = connection.execute(
                """
                SELECT singleton,marker_id,schema_version
                  FROM discord_edge_journal_meta_v1
                """
            ).fetchall()
        if len(rows) != 1 or rows[0][0] != 1 or rows[0][2] != 2:
            return False
        marker_id = str(uuid.UUID(str(rows[0][1])))
        if marker_id != rows[0][1]:
            return False
        marker = Path(f"{path}.initialized")
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            payload = f"{_EDGE_MARKER_PREFIX}{marker_id}\n".encode("ascii")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short route-back marker write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_parent(marker)
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        raise ProductionDiscordJournalBootstrapError(
            "journal_partial_bootstrap_invalid"
        ) from exc
    _secure_file(marker, uid=uid, gid=gid)
    return True


def _clean_counts(path: Path, kind: str) -> dict[str, int]:
    tables = (
        ("connector_events_v1", "connector_sends_v1")
        if kind == "connector"
        else (
            "discord_edge_idempotency_v1",
            "discord_edge_receipt_history_v1",
        )
    )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=5,
            isolation_level=None,
        )
        result = {
            table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in tables
        }
        check = connection.execute("PRAGMA quick_check(1)").fetchone()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        raise ProductionDiscordJournalBootstrapError(
            "journal_clean_snapshot_failed"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if check is None or str(check[0]).lower() != "ok" or any(result.values()):
        raise ProductionDiscordJournalBootstrapError("journal_not_clean")
    return result


def ensure_clean_journal(
    kind: str,
    *,
    intent_sha256: str,
    path: Path | None = None,
    busy_timeout_ms: int = 5_000,
) -> Mapping[str, Any]:
    """Create one pinned fresh journal or re-observe its exact clean state."""

    if kind not in _KINDS:
        raise ProductionDiscordJournalBootstrapError("journal_kind_invalid")
    expected = _expected_path(kind)
    target = expected if path is None else Path(path)
    if target != expected:
        raise ProductionDiscordJournalBootstrapError("journal_path_invalid")
    if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 30_000:
        raise ProductionDiscordJournalBootstrapError("journal_timeout_invalid")
    uid, gid = _secure_parent(target)
    intent = _ensure_intent(
        target,
        kind=kind,
        intent_sha256=intent_sha256,
        uid=uid,
        gid=gid,
    )
    marker = Path(f"{target}.initialized") if kind == "routeback" else None
    target_exists = os.path.lexists(target)
    marker_exists = marker is not None and os.path.lexists(marker)
    if kind == "routeback" and marker_exists and not target_exists:
        raise ProductionDiscordJournalBootstrapError(
            "journal_partial_bootstrap_invalid"
        )
    recovered_partial = False
    if kind == "routeback" and target_exists and not marker_exists:
        recovered_partial = _recover_routeback_database_only(
            target,
            uid=uid,
            gid=gid,
        )
        if not recovered_partial:
            _remove_clean_partial(
                target,
                allowed_tables=_EDGE_TABLES,
                uid=uid,
                gid=gid,
            )
            target_exists = False
    created = False
    try:
        if not target_exists:
            if kind == "connector":
                previous_umask = os.umask(0o077)
                try:
                    running = DurableDiscordConnectorJournal.bootstrap(
                        target,
                        busy_timeout_ms=busy_timeout_ms,
                    )
                finally:
                    os.umask(previous_umask)
                os.chmod(running.path, 0o600)
            else:
                DurableDiscordEdgeJournal.bootstrap(
                    target,
                    busy_timeout_ms=busy_timeout_ms,
                )
            created = True
        elif kind == "connector":
            try:
                DurableDiscordConnectorJournal(
                    target,
                    busy_timeout_ms=busy_timeout_ms,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                _remove_clean_partial(
                    target,
                    allowed_tables=_CONNECTOR_TABLES,
                    uid=uid,
                    gid=gid,
                )
                previous_umask = os.umask(0o077)
                try:
                    running = DurableDiscordConnectorJournal.bootstrap(
                        target,
                        busy_timeout_ms=busy_timeout_ms,
                    )
                finally:
                    os.umask(previous_umask)
                os.chmod(running.path, 0o600)
                created = True
        else:
            DurableDiscordEdgeJournal(
                target,
                busy_timeout_ms=busy_timeout_ms,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProductionDiscordJournalBootstrapError(
            "journal_bootstrap_failed"
        ) from exc

    observed = _secure_file(target, uid=uid, gid=gid)
    marker_observed = None
    if marker is not None:
        marker_observed = _secure_file(marker, uid=uid, gid=gid)
    counts = _clean_counts(target, kind)
    return _receipt(
        {
            "schema": BOOTSTRAP_SCHEMA,
            "kind": kind,
            "path": str(target),
            "uid": uid,
            "gid": gid,
            "mode": "0600",
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "marker_path": None if marker is None else str(marker),
            "marker_device": (
                None if marker_observed is None else marker_observed.st_dev
            ),
            "marker_inode": (
                None if marker_observed is None else marker_observed.st_ino
            ),
            "bootstrap_intent_path": str(intent),
            "bootstrap_intent_sha256": intent_sha256,
            "created": created,
            "recovered_partial": recovered_partial,
            "clean_row_counts": counts,
            "clean": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(_KINDS), required=True)
    parser.add_argument("--intent-sha256", required=True)
    parser.add_argument("--busy-timeout-ms", type=int, default=5_000)
    arguments = parser.parse_args(argv)
    receipt = ensure_clean_journal(
        arguments.kind,
        intent_sha256=arguments.intent_sha256,
        busy_timeout_ms=arguments.busy_timeout_ms,
    )
    print(_canonical_bytes(receipt).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "CONNECTOR_JOURNAL_PATH",
    "ProductionDiscordJournalBootstrapError",
    "ROUTEBACK_JOURNAL_PATH",
    "ensure_clean_journal",
    "main",
]
