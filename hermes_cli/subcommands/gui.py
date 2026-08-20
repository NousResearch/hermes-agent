"""``hermes gui`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_gui_parser(subparsers, *, cmd_gui: Callable) -> None:
    """Attach the ``gui`` subcommand to ``subparsers``."""
    # =========================================================================
    gui_parser = subparsers.add_parser(
        "desktop",
        aliases=["gui"],
        help="Build and launch the native desktop app",
        description=(
            "Launch the Hermes Electron desktop app. By default this installs "
            "workspace Node dependencies, builds the current OS's unpacked "
            "Electron app, then launches that packaged artifact."
        ),
    )
    gui_parser.add_argument(
        "--source",
        action="store_true",
        help="Launch via `electron .` against apps/desktop/dist instead of the packaged app",
    )
    gui_parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the desktop app but do not launch it (used by the installer's --update flow)",
    )
    gui_parser.add_argument(
        "--fake-boot",
        action="store_true",
        help="Enable deterministic desktop boot delays for validating startup UI",
    )
    gui_parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Force Desktop to ignore any hermes CLI already on PATH during backend resolution",
    )
    gui_parser.add_argument(
        "--hermes-root",
        help="Override the Hermes source root used by Desktop (sets HERMES_DESKTOP_HERMES_ROOT)",
    )
    gui_parser.add_argument(
        "--cwd",
        help="Initial project directory for Desktop chat sessions (sets HERMES_DESKTOP_CWD)",
    )
    gui_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm install/package and launch the existing unpacked app from apps/desktop/release",
    )
    gui_parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force a full rebuild even if the content stamp matches",
    )

    # Nested subcommand: hermes desktop launcher install
    launcher_subparsers = gui_parser.add_subparsers(
        dest="gui_subcommand"
    )
    launcher_parser = launcher_subparsers.add_parser(
        "launcher",
        help="Install a macOS Spotlight-searchable launcher for Hermes Desktop",
        description=(
            "Create a macOS launcher that is searchable from Spotlight and "
            "optionally visible on the Desktop. The launcher calls the existing "
            "`hermes desktop` command with the specified --cwd."
        ),
    )
    launcher_parser.add_argument(
        "action",
        choices=["install", "uninstall"],
        help="Action to perform: install or uninstall the launcher",
    )
    launcher_parser.add_argument(
        "--cwd",
        default=None,
        help="Project directory for the launcher to target (default: $HOME)",
    )
    launcher_parser.add_argument(
        "--name",
        default="Hermes Desktop",
        help="Launcher filename without extension (default: 'Hermes Desktop')",
    )

    gui_parser.set_defaults(func=cmd_gui)
