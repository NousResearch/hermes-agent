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


def run_state_store_command(args) -> int:
    """Run selected-PG maintenance and render sanitized JSON evidence."""
    from postgresql_state_store_operations import PostgreSQLSandboxOperationsError
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
        raise ValueError(f"unknown state-store action: {args.state_store_action}")
    except (StateStoreMaintenanceError, PostgreSQLSandboxOperationsError, ValueError) as exc:
        print(f"state-store {args.state_store_action} failed: {exc}")
        return 2
