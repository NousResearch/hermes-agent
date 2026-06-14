#!/usr/bin/env python3
"""Merge Hermes desktop history from legacy install home into canonical ~/.hermes.

Windows packaged desktop defaults to %LOCALAPPDATA%\\hermes while CLI/gateway and
source-mode desktop (start-hermes-desktop.ps1) use ~/.hermes. This script:

1. Backups canonical state.db
2. Merges missing sessions/messages from legacy state.db (INSERT OR IGNORE)
3. Imports orphaned *.jsonl / request_dump_*.json transcripts into SessionDB
4. Copies missing on-disk session artifacts into canonical sessions/
5. Writes a JSON report under ~/.hermes/migration/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SESSION_ID_RE = re.compile(r"^(\d{8}_\d{6}_[0-9a-f]+)")
IMPORT_ROLES = frozenset({"user", "assistant", "tool"})


def _canonical_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser().resolve()


def _legacy_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise SystemExit("LOCALAPPDATA is unset; pass --legacy-home explicitly")
    return Path(local) / "hermes"


def _session_id_from_filename(name: str) -> str | None:
    if name.startswith("request_dump_"):
        m = SESSION_ID_RE.match(name[len("request_dump_") :])
        return m.group(1) if m else None
    m = SESSION_ID_RE.match(name)
    return m.group(1) if m else None


def _parse_iso_ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _parse_jsonl_transcript(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        role = obj.get("role")
        if role == "session_meta":
            meta = obj
            continue
        if role in IMPORT_ROLES:
            messages.append(obj)
    return meta, messages


def _table_columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    rows = conn.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
    return [row[1] for row in rows]


def _patch_session_times(
    db: Any,
    session_id: str,
    started_at: float,
    ended_at: float | None,
) -> None:
    def _do(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = COALESCE(?, ended_at) WHERE id = ?",
            (started_at, ended_at, session_id),
        )

    db._execute_write(_do)  # noqa: SLF001 — one-off migration utility


def _merge_state_db(
    target_db: Path,
    source_db: Path,
    *,
    dry_run: bool,
    report: dict[str, Any],
) -> None:
    if not source_db.exists():
        report["state_db"] = {"skipped": True, "reason": "legacy state.db missing"}
        return

    target_conn = sqlite3.connect(target_db)
    try:
        target_sessions = target_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        target_messages = target_conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        target_conn.close()

    if dry_run:
        src_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            src_sessions = src_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            src_messages = src_conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            src_conn.close()
        conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
        conn.execute("ATTACH DATABASE ? AS legacy", (str(source_db),))
        try:
            missing_sessions = conn.execute(
                "SELECT COUNT(*) FROM legacy.sessions ls "
                "WHERE ls.id NOT IN (SELECT id FROM main.sessions)"
            ).fetchone()[0]
            upgrade_sessions = conn.execute(
                """
                SELECT COUNT(*) FROM legacy.sessions ls
                JOIN main.sessions ms ON ms.id = ls.id
                WHERE ls.message_count > COALESCE(ms.message_count, 0)
                """
            ).fetchone()[0]
        finally:
            conn.close()
        report["state_db"] = {
            "dry_run": True,
            "target_sessions": target_sessions,
            "target_messages": target_messages,
            "legacy_sessions": src_sessions,
            "legacy_messages": src_messages,
            "missing_sessions": missing_sessions,
            "upgrade_sessions": upgrade_sessions,
        }
        return

    backup = target_db.with_suffix(f".db.bak-desktop-merge-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(target_db, backup)
    report["state_db"] = {"backup": str(backup)}

    conn = sqlite3.connect(target_db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("ATTACH DATABASE ? AS legacy", (str(source_db),))
    try:
        session_cols = [
            c
            for c in _table_columns(conn, "sessions", "legacy")
            if c in _table_columns(conn, "sessions", "main")
        ]
        msg_cols = [
            c
            for c in _table_columns(conn, "messages", "legacy")
            if c in _table_columns(conn, "messages", "main") and c != "id"
        ]

        sess_col_sql = ", ".join(session_cols)
        before_missing = conn.execute(
            "SELECT COUNT(*) FROM legacy.sessions ls "
            "WHERE ls.id NOT IN (SELECT id FROM main.sessions)"
        ).fetchone()[0]
        conn.execute(
            f"INSERT OR IGNORE INTO main.sessions ({sess_col_sql}) "
            f"SELECT {sess_col_sql} FROM legacy.sessions ls "
            f"WHERE ls.id NOT IN (SELECT id FROM main.sessions)"
        )
        inserted_sessions = before_missing

        msg_col_sql = ", ".join(msg_cols)
        conn.execute(
            f"INSERT INTO main.messages ({msg_col_sql}) "
            f"SELECT {msg_col_sql} FROM legacy.messages lm "
            f"WHERE lm.session_id IN ("
            f"  SELECT ls.id FROM legacy.sessions ls "
            f"  WHERE ls.id NOT IN (SELECT id FROM main.sessions)"
            f")"
        )
        inserted_messages_new_sessions = conn.execute("SELECT changes()").fetchone()[0]

        upgraded = 0
        for sid, legacy_count in conn.execute(
            """
            SELECT ls.id, ls.message_count
            FROM legacy.sessions ls
            JOIN main.sessions ms ON ms.id = ls.id
            WHERE ls.message_count > COALESCE(ms.message_count, 0)
            """
        ):
            conn.execute("DELETE FROM main.messages WHERE session_id = ?", (sid,))
            conn.execute(
                f"INSERT INTO main.messages ({msg_col_sql}) "
                f"SELECT {msg_col_sql} FROM legacy.messages WHERE session_id = ?",
                (sid,),
            )
            legacy_row = conn.execute(
                f"SELECT {sess_col_sql} FROM legacy.sessions WHERE id = ?", (sid,)
            ).fetchone()
            if legacy_row:
                assignments = ", ".join(f"{col} = ?" for col in session_cols if col != "id")
                values = [legacy_row[session_cols.index(col)] for col in session_cols if col != "id"]
                conn.execute(
                    f"UPDATE main.sessions SET {assignments} WHERE id = ?",
                    (*values, sid),
                )
            upgraded += 1

        conn.commit()
        after_sessions = conn.execute("SELECT COUNT(*) FROM main.sessions").fetchone()[0]
        after_messages = conn.execute("SELECT COUNT(*) FROM main.messages").fetchone()[0]
        report["state_db"].update(
            {
                "inserted_sessions": inserted_sessions,
                "inserted_messages_new_sessions": inserted_messages_new_sessions,
                "upgraded_sessions": upgraded,
                "sessions_after": after_sessions,
                "messages_after": after_messages,
            }
        )
    finally:
        conn.execute("DETACH DATABASE legacy")
        conn.close()


def _import_jsonl_orphans(
    canonical_home: Path,
    legacy_home: Path,
    *,
    dry_run: bool,
    report: dict[str, Any],
) -> None:
    from hermes_state import SessionDB

    canonical_sessions = canonical_home / "sessions"
    legacy_sessions = legacy_home / "sessions"
    canonical_sessions.mkdir(parents=True, exist_ok=True)

    db = None if dry_run else SessionDB(db_path=canonical_home / "state.db")
    conn = sqlite3.connect(f"file:{canonical_home / 'state.db'}?mode=ro", uri=True)
    try:
        db_ids = {row[0] for row in conn.execute("SELECT id FROM sessions")}
    finally:
        conn.close()

    candidates: dict[str, Path] = {}
    for root in (canonical_sessions, legacy_sessions):
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_file() or path.suffix != ".jsonl":
                continue
            sid = _session_id_from_filename(path.name)
            if sid and sid not in db_ids:
                candidates.setdefault(sid, path)

    imported: list[str] = []
    skipped_empty: list[str] = []
    errors: list[dict[str, str]] = []

    for sid, path in sorted(candidates.items()):
        try:
            meta, messages = _parse_jsonl_transcript(path)
            if not messages:
                skipped_empty.append(sid)
                continue
            platform = (meta.get("platform") or "unknown").strip() or "unknown"
            model = meta.get("model")
            started = _parse_iso_ts(messages[0].get("timestamp")) or time.time()
            ended = _parse_iso_ts(messages[-1].get("timestamp"))
            if dry_run:
                imported.append(sid)
                continue
            assert db is not None
            db.create_session(
                sid,
                source=platform if platform != "unknown" else "cli",
                model=model,
            )
            _patch_session_times(db, sid, started, ended)
            db.replace_messages(sid, messages)
            dest = canonical_sessions / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
            imported.append(sid)
        except Exception as exc:  # noqa: BLE001 — collect per-session failures
            errors.append({"session_id": sid, "path": str(path), "error": str(exc)})

    report["jsonl_import"] = {
        "candidates": len(candidates),
        "imported": len(imported),
        "skipped_empty": len(skipped_empty),
        "errors": errors,
        "sample_imported": imported[:10],
    }


def _copy_missing_session_files(
    canonical_home: Path,
    legacy_home: Path,
    *,
    dry_run: bool,
    report: dict[str, Any],
) -> None:
    src = legacy_home / "sessions"
    dst = canonical_home / "sessions"
    if not src.exists():
        report["session_files"] = {"copied": 0, "skipped": True}
        return
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in src.iterdir():
        if not path.is_file():
            continue
        target = dst / path.name
        if target.exists():
            continue
        if dry_run:
            copied += 1
            continue
        shutil.copy2(path, target)
        copied += 1
    report["session_files"] = {"copied": copied}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-home", default="", help="Target HERMES_HOME (~/.hermes)")
    parser.add_argument("--legacy-home", default="", help="Legacy desktop HERMES_HOME")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    canonical = _canonical_home(args.canonical_home or None)
    legacy = _legacy_home(args.legacy_home or None)
    canonical.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "canonical_home": str(canonical),
        "legacy_home": str(legacy),
        "dry_run": args.dry_run,
    }

    target_db = canonical / "state.db"
    source_db = legacy / "state.db"
    if not target_db.exists():
        raise SystemExit(f"Canonical state.db not found: {target_db}")

    _merge_state_db(target_db, source_db, dry_run=args.dry_run, report=report)
    _copy_missing_session_files(canonical, legacy, dry_run=args.dry_run, report=report)
    _import_jsonl_orphans(canonical, legacy, dry_run=args.dry_run, report=report)

    migration_dir = canonical / "migration"
    migration_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = migration_dir / f"desktop-history-merge-{stamp}.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(report_path)
        final = sqlite3.connect(target_db)
        try:
            report["final_counts"] = {
                "sessions": final.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "messages": final.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            }
        finally:
            final.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
