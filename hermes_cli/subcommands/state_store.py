"""``hermes state-store`` PostgreSQL-native maintenance commands."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def build_state_store_parser(subparsers, *, cmd_state_store: Callable) -> None:
    """Attach explicit PostgreSQL doctor and logical-backup operations."""
    parser = subparsers.add_parser(
        "state-store",
        help="Inspect or logically back up the selected PostgreSQL state store",
    )
    commands = parser.add_subparsers(dest="state_store_action", required=True)
    doctor = commands.add_parser("doctor", help="Validate the trusted PostgreSQL tenant catalog")
    doctor.set_defaults(func=cmd_state_store)
    backup = commands.add_parser(
        "logical-backup",
        help="Create a verified PostgreSQL logical backup in an existing owned directory",
    )
    backup.add_argument("--output-directory", required=True, type=Path, help="Existing directory owned by the active user")
    backup.add_argument(
        "--quiesced", action="store_true",
        help="Confirm writers are quiesced before taking the logical backup",
    )
    backup.set_defaults(func=cmd_state_store)
    sqlite_import = commands.add_parser(
        "sqlite-import",
        help="Import an explicit SQLite rehearsal into a newly allocated owned PostgreSQL tenant",
    )
    sqlite_import.add_argument("--source", required=True, type=Path)
    sqlite_import.add_argument("--snapshot-root", required=True, type=Path)
    sqlite_import.add_argument("--evidence", type=Path)
    sqlite_import.set_defaults(func=cmd_state_store)


def run_state_store_command(args) -> int:
    """Run selected-PG maintenance and render sanitized JSON evidence."""
    from postgresql_state_store_operations import PostgreSQLSandboxOperationsError
    from postgresql_state_store_sqlite_import import (
        SQLiteImportTargetCleanup,
        SQLitePostgreSQLImportError,
    )
    from state_store_maintenance import StateStoreMaintenanceError, StateStoreMaintenanceOperations

    try:
        operations = StateStoreMaintenanceOperations.resolve()
        if operations.selected_backend != "postgresql":
            print("state-store doctor and logical-backup require state_store.backend=postgresql; SQLite behavior is unchanged.")
            return 2
        if args.state_store_action == "doctor":
            print(json.dumps(operations.doctor(), sort_keys=True, indent=2))
            return 0
        if args.state_store_action == "logical-backup":
            backup, verification = operations.logical_backup(
                args.output_directory, quiesced=bool(args.quiesced),
            )
            print(json.dumps({
                "backup_directory": str(backup.backup_directory),
                "manifest_path": str(backup.manifest_path),
                "archive_path": str(backup.archive_path),
                "archive_sha256": backup.manifest["archive"]["sha256"],
                "restore_verification": verification,
            }, sort_keys=True, indent=2))
            return 0
        if args.state_store_action == "sqlite-import":
            # Resolve the profile-scoped PostgreSQL secret through the maintenance
            # boundary, but never use the selected tenant as a destination.
            native = operations._postgresql_operations(None)
            from postgresql_state_store_sqlite_import import import_into_allocated_target

            result, target = import_into_allocated_target(
                native._settings,
                native._dsn,
                args.source,
                snapshot_root=args.snapshot_root,
                evidence_path=args.evidence,
            )
            # A command-line rehearsal has no returned Python capability with
            # which to later prove target ownership.  Tear the disposable target
            # down before reporting success rather than leaving an inaccessible
            # completed sandbox behind.
            try:
                cleanup = target.drop()
            except SQLitePostgreSQLImportError as exc:
                raise SQLitePostgreSQLImportError(
                    "SQLite import completed but CLI target cleanup was not committed",
                    cleanup=SQLiteImportTargetCleanup(
                        "reconciliation-required", target.schema.name, str(exc)
                    ),
                ) from exc
            print(json.dumps({
                "import_id": result.import_id,
                "status": result.status,
                "source_fingerprint": result.source_fingerprint,
                "object_counts": result.object_counts,
                "target_schema": target.schema.name,
                "cleanup": cleanup.as_dict(),
            }, sort_keys=True, indent=2))
            return 0
        raise ValueError(f"unknown state-store action: {args.state_store_action}")
    except (StateStoreMaintenanceError, PostgreSQLSandboxOperationsError, SQLitePostgreSQLImportError, ValueError) as exc:
        cleanup = getattr(exc, "cleanup", None)
        if cleanup is not None:
            print(json.dumps({
                "action": args.state_store_action,
                "status": "failed",
                "error": str(exc),
                "cleanup": cleanup.as_dict(),
            }, sort_keys=True, indent=2))
        else:
            print(f"state-store {args.state_store_action} failed: {exc}")
        return 2
