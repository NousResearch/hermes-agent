"""``hermes webapp``: open the Desktop workspace in a browser, served by the gateway."""

from __future__ import annotations

from typing import Callable


def build_webapp_parser(subparsers, *, cmd_webapp: Callable) -> None:
    parser = subparsers.add_parser(
        "webapp",
        help="Open the Hermes Desktop workspace in a browser",
        description=(
            "Open the Hermes Desktop workspace in a browser. The gateway serves it when "
            "webapp.enabled is true in config.yaml; this command builds the bundle when needed, "
            "starts the gateway and opens a one-use sign-in link."
        ),
    )
    parser.add_argument("--no-open", action="store_true", help="Print the one-use link instead of opening a browser")
    parser.add_argument("--force-build", action="store_true",
                        help="Rebuild the Webapp bundle even when its build receipt is current")
    parser.add_argument("--skip-build", action="store_true", help="Use the existing Webapp bundle as is")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build-only", action="store_true", help="Build the Webapp bundle and stop")
    action.add_argument("--status", action="store_true", help="Show whether the gateway is serving the Webapp")
    action.add_argument("--sign-out", action="store_true",
                        help="Sign out every Webapp tab (each needs a new link from hermes webapp)")
    parser.set_defaults(func=cmd_webapp)
