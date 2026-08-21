#!/usr/bin/env python3
"""Run an ad-hoc Kanban reproduction without inheriting the live board.

Dispatcher workers inherit ``HERMES_KANBAN_DB`` so every lifecycle tool stays
on the board that claimed the task.  A reproduction launched from such a
worker must replace that direct override; changing only ``HERMES_HOME`` or
``HERMES_KANBAN_HOME`` is not sufficient because the DB override wins.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Python Kanban reproduction in an isolated DB by default.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Use an explicit reproduction DB instead of a temporary one.",
    )
    parser.add_argument(
        "--allow-existing-db",
        action="store_true",
        help="DANGEROUS: allow mutation of an existing non-configured DB.",
    )
    parser.add_argument(
        "--allow-configured-db",
        action="store_true",
        help="DANGEROUS: allow mutation of the configured/live Kanban board.",
    )
    parser.add_argument("script", type=Path, help="Python reproduction script to run.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_configured_board(db: Path, env: dict[str, str]) -> bool:
    """Return whether ``db`` is the active/default/named production board."""
    direct = env.get("HERMES_KANBAN_DB", "").strip()
    if direct and db == _resolved(Path(direct)):
        return True

    # Import lazily so argument/help handling remains lightweight.  These path
    # helpers do not connect to or initialize the database.
    from hermes_cli import kanban_db as kb

    home = _resolved(kb.kanban_home())
    if db == home / "kanban.db":
        return True
    boards = home / "kanban" / "boards"
    return db.name == "kanban.db" and db.is_relative_to(boards)


def _run(script: Path, script_args: list[str], db: Path, env: dict[str, str]) -> int:
    runtime_root = db.parent
    runtime_root.mkdir(parents=True, exist_ok=True)
    scoped = env.copy()
    scoped.update({
        "HERMES_HOME": str(runtime_root),
        "HERMES_KANBAN_HOME": str(runtime_root),
        "HERMES_KANBAN_DB": str(db),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_WORKSPACES_ROOT": str(runtime_root / "kanban" / "workspaces"),
        "HERMES_KANBAN_LOGS_ROOT": str(runtime_root / "kanban" / "logs"),
    })
    completed = subprocess.run(
        [sys.executable, str(script), *script_args],
        env=scoped,
        check=False,
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    script = _resolved(args.script)
    if not script.is_file():
        print(f"reproduction script not found: {script}", file=sys.stderr)
        return 2

    inherited = os.environ.copy()
    if args.db is None:
        with tempfile.TemporaryDirectory(prefix="hermes-kanban-repro-") as temp_dir:
            return _run(
                script,
                args.script_args,
                Path(temp_dir) / "kanban.db",
                inherited,
            )

    db = _resolved(args.db)
    if _is_configured_board(db, inherited) and not args.allow_configured_db:
        print(
            "refusing configured Kanban DB without --allow-configured-db: "
            f"{db}\nUse the default temporary DB unless live-board mutation is intentional.",
            file=sys.stderr,
        )
        return 2
    if db.exists() and not (args.allow_existing_db or args.allow_configured_db):
        print(
            f"refusing existing Kanban DB without --allow-existing-db: {db}",
            file=sys.stderr,
        )
        return 2
    return _run(script, args.script_args, db, inherited)


if __name__ == "__main__":
    raise SystemExit(main())
