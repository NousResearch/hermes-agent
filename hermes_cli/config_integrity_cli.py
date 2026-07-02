"""CLI handlers for ``hermes config seal/verify/restore``.

Integrates the config-integrity-watchdog skill into the Hermes CLI.
Subcommands:
    seal    — hash config.yaml and append a signed entry to the git-committed
              integrity log, creating a tamper-evident anchor.
    verify  — check the current hash against the sealed baseline; exit 1 if
              the config has been tampered with.
    restore — revert config.yaml to the sealed baseline if it has been tampered.

Exit codes for verify:
    0 — config matches canonical baseline
    1 — config has been tampered
    2 — integrity log itself has uncommitted changes (log tampering)
    3 — no baseline found (run seal first)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Register subcommands on an existing config subparsers object
# ---------------------------------------------------------------------------


def register_subcommands(config_subparsers: argparse.Action) -> None:
    """Attach seal/verify/restore parsers to the ``hermes config`` subparser.

    Called from ``hermes_cli.main`` after the base config subparsers are set up.
    """
    config_subparsers.add_parser(
        "seal",
        help="Hash config.yaml and anchor it in the git-backed integrity log",
    )

    config_subparsers.add_parser(
        "verify",
        help=(
            "Check current config.yaml against sealed baseline; "
            "exits 1 if tampered"
        ),
    )

    config_subparsers.add_parser(
        "restore",
        help="Revert config.yaml to the sealed git baseline if tampered",
    )


# ---------------------------------------------------------------------------
# Handlers — called from hermes_cli.config.config_command dispatch
# ---------------------------------------------------------------------------


def _import_core():
    """Import the shared core module from the skill scripts directory.

    We add the skill directory to sys.path on first use rather than at
    module-import time so that the import stays lazy (fast startup) and
    doesn't conflict with any top-level package names.

    Search order:
    1. ``~/.hermes/skills/devops/config-integrity-watchdog`` (post-sync location)
    2. Repo-relative ``skills/devops/config-integrity-watchdog`` (pre-sync / dev)
    """
    candidates = [
        Path.home() / ".hermes" / "skills" / "devops" / "config-integrity-watchdog",
        Path(__file__).parent.parent / "skills" / "devops" / "config-integrity-watchdog",
    ]
    for skills_root in candidates:
        if skills_root.exists():
            if str(skills_root) not in sys.path:
                sys.path.insert(0, str(skills_root))
            try:
                import config_integrity  # noqa: PLC0415
                return config_integrity
            except ImportError:
                continue
    print(
        "ERROR: config-integrity-watchdog skill not found. "
        "Run 'hermes skills sync' first."
    )
    sys.exit(1)


def cmd_seal(args: argparse.Namespace) -> int:
    """Handle ``hermes config seal``."""
    core = _import_core()
    return core.seal()


def cmd_verify(args: argparse.Namespace) -> int:
    """Handle ``hermes config verify``."""
    core = _import_core()
    return core.verify()


def cmd_restore(args: argparse.Namespace) -> int:
    """Handle ``hermes config restore``."""
    core = _import_core()
    return core.restore()
