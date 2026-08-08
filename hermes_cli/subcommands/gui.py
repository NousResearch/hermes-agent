"""``hermes gui`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_gui_parser(subparsers, *, cmd_gui: Callable, cmd_gui_install: Callable) -> None:
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
    # Default behavior: launch the desktop app
    gui_parser.set_defaults(func=cmd_gui)
    
    gui_subparsers = gui_parser.add_subparsers(dest="gui_subcommand")

    # Launch subcommand (explicit)
    launch_parser = gui_subparsers.add_parser(
        "launch",
        help="Launch the desktop app (default)",
        description="Build and launch the Hermes Electron desktop app",
        add_help=False,
    )
    launch_parser.add_argument(
        "--source",
        action="store_true",
        help="Launch via `electron .` against apps/desktop/dist instead of the packaged app",
    )
    launch_parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the desktop app but do not launch it (used by the installer's --update flow)",
    )
    launch_parser.add_argument(
        "--fake-boot",
        action="store_true",
        help="Enable deterministic desktop boot delays for validating startup UI",
    )
    launch_parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Force Desktop to ignore any hermes CLI already on PATH during backend resolution",
    )
    launch_parser.add_argument(
        "--hermes-root",
        help="Override the Hermes source root used by Desktop (sets HERMES_DESKTOP_HERMES_ROOT)",
    )
    launch_parser.add_argument(
        "--cwd",
        help="Initial project directory for Desktop chat sessions (sets HERMES_DESKTOP_CWD)",
    )
    launch_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm install/package and launch the existing unpacked app from apps/desktop/release",
    )
    launch_parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force a full rebuild even if the content stamp matches",
    )
    launch_parser.set_defaults(func=cmd_gui)

    # Install subcommand - creates desktop shortcuts/entries
    install_parser = gui_subparsers.add_parser(
        "install",
        help="Create desktop shortcut/menu entry for the Hermes desktop app",
        description=(
            "Create a desktop shortcut (Windows), .desktop file (Linux), or "
            "Application folder entry (macOS) to launch Hermes Desktop from your "
            "system's application menu."
        ),
    )
    install_parser.add_argument(
        "--user",
        action="store_true",
        help="Install for current user only (default: system-wide if possible)",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing shortcut/entry",
    )
    install_parser.set_defaults(func=cmd_gui_install)