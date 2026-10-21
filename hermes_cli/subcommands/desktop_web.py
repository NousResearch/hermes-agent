"""``hermes desktop-web`` command parser.

The command is a browser host for the complete Desktop renderer. Its lifecycle
is intentionally shaped like ``hermes dashboard`` while the implementation
uses a separate web bundle and never launches Electron.
"""

from __future__ import annotations

from typing import Callable


def build_desktop_web_parser(subparsers, *, cmd_desktop_web: Callable) -> None:
    parser = subparsers.add_parser(
        "desktop-web",
        help="Start the Hermes Desktop UI in a browser",
        description=(
            "Build and start the Hermes Desktop Web UI. It reuses the Desktop "
            "renderer in a browser and connects to the Hermes backend."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=13043,
        help="Port to bind (default: 13043)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the running Desktop Web process and exit",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser automatically",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the running Desktop Web process and exit",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Serve the existing web dist without building",
    )
    parser.set_defaults(func=cmd_desktop_web)
