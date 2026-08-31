"""Inventory and quarantine legacy compaction tool-result carriers.

This module intentionally separates discovery from mutation.  Text matching only
produces ``ambiguous`` candidates; an operator-reviewed manifest must promote a
candidate to ``proven`` before ``--apply`` can set its durable hidden marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from agent.compaction_display import LEGACY_TOOL_CARRIER_QUARANTINE_METADATA_KEY
from hermes_state import SessionDB

INVENTORY_SCHEMA = "ToolCarrierInventoryV1"
# Publicly re-exported for manifest/test consumers; the typed metadata contract
# itself remains owned by agent.compaction_display.
LEGACY_CARRIER_METADATA_KEY = LEGACY_TOOL_CARRIER_QUARANTINE_METADATA_KEY
_LEGACY_CARRIER_RE = re.compile(r"^\[Tool result(?: [^\]\r\n]{1,256})?\]: ")
_TOOL_RESULT_MARKER_RE = re.compile(r"^\[Tool result", re.MULTILINE)


def _resolved_database_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"database does not exist: {resolved}")
    return resolved


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _database_snapshot_sha256(conn: sqlite3.Connection) -> str:
    """Hash immutable message identity/content, excluding display metadata.

    Display quarantine deliberately changes only presentation metadata, so the
    same reviewed manifest remains idempotently applicable. Any replacement or
    content/tool-field drift at the same path produces a different snapshot.
    """
    digest = hashlib.sha256()
    rows = conn.execute(
        "SELECT id, session_id, role, content, tool_call_id, tool_calls, tool_name, "
        "active, compacted FROM messages ORDER BY id"
    )
    for row in rows:
        digest.update(
            json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _database_descriptor(path: Path) -> dict[str, Any]:
    with _open_readonly(path) as conn:
        schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        message_count = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        snapshot_sha256 = _database_snapshot_sha256(conn)
    return {
        "path": str(path),
        "schema_version": schema_version,
        "message_count": message_count,
        "snapshot_sha256": snapshot_sha256,
    }


def build_inventory(path: str | Path) -> dict[str, Any]:
    """Return read-only ambiguous candidates; it never asserts provenance."""
    database_path = _resolved_database_path(path)
    candidates: list[dict[str, Any]] = []
    excluded = 0
    with _open_readonly(database_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, session_id, role, content, tool_calls, display_kind, display_metadata "
            "FROM messages WHERE active = 1 AND role = 'assistant' "
            "AND content LIKE '[Tool result%' ORDER BY id"
        ).fetchall()
    for row in rows:
        content = row["content"]
        if not isinstance(content, str) or not _LEGACY_CARRIER_RE.match(content):
            excluded += 1
            continue
        if row["tool_calls"]:
            excluded += 1
            continue
        metadata: dict[str, Any] = {}
        try:
            parsed = (
                json.loads(row["display_metadata"]) if row["display_metadata"] else {}
            )
            if isinstance(parsed, dict):
                metadata = parsed
        except (TypeError, json.JSONDecodeError):
            excluded += 1
            continue
        if (
            row["display_kind"] is not None
            or LEGACY_TOOL_CARRIER_QUARANTINE_METADATA_KEY in metadata
        ):
            excluded += 1
            continue
        candidates.append({
            "id": int(row["id"]),
            "session_id": str(row["session_id"]),
            "role": "assistant",
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content_chars": len(content),
            "tool_result_markers": len(_TOOL_RESULT_MARKER_RE.findall(content)),
            "disposition": "ambiguous",
            "reason": "prefix shape is insufficient proof of compactor origin",
        })
    return {
        "schema": INVENTORY_SCHEMA,
        "database": _database_descriptor(database_path),
        "candidates": candidates,
        "counts": {"ambiguous": len(candidates), "excluded": excluded},
    }


def _valid_proof(candidate: dict[str, Any]) -> bool:
    refs = candidate.get("evidence_refs")
    return (
        candidate.get("disposition") == "proven"
        and isinstance(refs, list)
        and len({ref for ref in refs if isinstance(ref, str) and ref.strip()}) >= 2
    )


def apply_manifest(
    path: str | Path, manifest: dict[str, Any], *, dry_run: bool = True
) -> dict[str, int]:
    """Apply only explicit, evidence-bearing manifest selections.

    The database descriptor prevents accidentally applying a review manifest to
    a different store. Each selected row is additionally bound by content hash
    inside SessionDB's transaction, so a concurrent transcript change aborts
    the entire batch rather than hiding a different row.
    """
    database_path = _resolved_database_path(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"manifest schema must be {INVENTORY_SCHEMA}")
    descriptor = manifest.get("database")
    if not isinstance(descriptor, dict) or descriptor != _database_descriptor(
        database_path
    ):
        raise ValueError("manifest database descriptor does not match target")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("manifest candidates must be a list")

    selected = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and _valid_proof(candidate)
    ]
    skipped = len(candidates) - len(selected)
    if not selected:
        return {"changed": 0, "unchanged": 0, "skipped": skipped}

    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    migration_id = f"legacy-tool-carrier:{manifest_digest[:16]}"
    db = SessionDB(database_path)
    try:
        outcome = db.quarantine_legacy_tool_carriers(
            selected, migration_id=migration_id, dry_run=dry_run
        )
    finally:
        db.close()
    if dry_run:
        return {
            "changed": int(outcome["changed"]),
            "unchanged": int(outcome["unchanged"]),
            "skipped": skipped,
            "would_change": len(selected) - int(outcome["unchanged"]),
        }
    return {
        "changed": int(outcome["changed"]),
        "unchanged": int(outcome["unchanged"]),
        "skipped": skipped,
    }


def _online_backup(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"backup already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_conn = _open_readonly(source)
    target_conn = sqlite3.connect(destination)
    try:
        source_conn.backup(target_conn)
        target_conn.commit()
    finally:
        target_conn.close()
        source_conn.close()
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    directory_fd = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    inventory = subcommands.add_parser(
        "inventory", help="write ambiguous candidate inventory"
    )
    inventory.add_argument("--db", required=True)
    inventory.add_argument("--out", required=True)
    migrate = subcommands.add_parser(
        "migrate", help="dry-run or apply a reviewed manifest"
    )
    migrate.add_argument("--db", required=True)
    migrate.add_argument("--manifest", required=True)
    migrate.add_argument(
        "--apply", action="store_true", help="perform the manifest-bounded mutation"
    )
    migrate.add_argument("--backup", help="required output path with --apply")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "inventory":
        inventory = build_inventory(args.db)
        output = Path(args.out).expanduser().resolve()
        if output.exists():
            raise ValueError(f"refusing to overwrite existing inventory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"out": str(output), "counts": inventory["counts"]}, sort_keys=True
            )
        )
        return 0

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.apply:
        if not args.backup:
            raise ValueError("--backup is required with --apply")
        _online_backup(
            _resolved_database_path(args.db), Path(args.backup).expanduser().resolve()
        )
    outcome = apply_manifest(args.db, manifest, dry_run=not args.apply)
    print(json.dumps({"applied": bool(args.apply), **outcome}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
