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
        help="Build and launch Hermes Desktop",
        description=(
            "Launch Hermes Desktop. On macOS/Windows/Linux this builds and runs "
            "the Electron shell. On Termux it builds the same Desktop renderer, "
            "serves it on loopback, and opens it through Termux:X11 Chromium."
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
        help="Build Desktop but do not launch it (renderer-only on Termux)",
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
        help="Reuse an existing Desktop build (apps/desktop/dist on Termux; packaged app elsewhere)",
    )
    gui_parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force a full rebuild even if the content stamp matches",
    )
    gui_parser.add_argument(
        "--port",
        type=int,
        default=9119,
        help="Termux browser-hosted Desktop loopback port (default: 9119; ignored by Electron hosts)",
    )
    gui_parser.set_defaults(func=cmd_gui)
