"""``hermes novnc`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_novnc_parser(subparsers, *, cmd_novnc: Callable) -> None:
    """Attach the ``novnc`` subcommand to ``subparsers``."""
    novnc_parser = subparsers.add_parser(
        "novnc",
        help="Start/stop a noVNC web-based VNC session",
        description="Manage noVNC + websockify sessions: start a web-based VNC client, stop it, or check its status.",
    )
    novnc_sub = novnc_parser.add_subparsers(
        dest="novnc_command",
        help="Subcommands: start, stop, status",
    )

    start_parser = novnc_sub.add_parser(
        "start",
        help="Start a noVNC session",
        description="Launch websockify to proxy a VNC server over WebSocket and open the noVNC page in a browser.",
    )
    start_parser.add_argument(
        "--port", type=int, default=6080, help="WebSocket listen port (default: 6080)"
    )
    start_parser.add_argument(
        "--vnc-host", default="127.0.0.1", help="VNC server host (default: 127.0.0.1)"
    )
    start_parser.add_argument(
        "--vnc-port", type=int, default=5900, help="VNC server port (default: 5900)"
    )
    start_parser.add_argument(
        "--password",
        default=None,
        help="VNC password for auto-auth (patches noVNC html)",
    )

    novnc_sub.add_parser(
        "stop",
        help="Stop the running noVNC session",
        description="Kill the running websockify process started by ``hermes novnc start``.",
    )

    novnc_sub.add_parser(
        "status",
        help="Check if noVNC session is running",
        description="Show whether a websockify process is active.",
    )

    novnc_parser.set_defaults(func=cmd_novnc)
