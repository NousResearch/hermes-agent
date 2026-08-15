"""``hermes verify`` subcommand parser.

Follows the pattern of ``hermes_cli/subcommands/doctor.py``: parser built
here, handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable

# Keep in sync with agent/verify/runner.py defaults; not imported here to
# avoid paying an extra module import on every `hermes` invocation.
DEFAULT_PHASE_TIMEOUT = 600.0
DEFAULT_READY_TIMEOUT = 60.0


def build_verify_parser(subparsers, *, cmd_verify: Callable) -> None:
    """Attach the ``verify`` subcommand to ``subparsers``."""
    verify_parser = subparsers.add_parser(
        "verify",
        help="Detect a project's run recipe and smoke-test it",
        description=(
            "Detect how the current project is built, tested, and started "
            "(or load the saved manifest at .hermes/environment.json), then "
            "run a verification pass: bootstrap -> build -> test -> start in "
            "background -> poll readiness -> teardown."
        ),
    )
    verify_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Project root to verify (default: current directory)",
    )
    verify_parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect and print the recipe as JSON; run nothing",
    )
    verify_parser.add_argument(
        "--save",
        action="store_true",
        help="Save the recipe as .hermes/environment.json in the project",
    )
    verify_parser.add_argument(
        "--skip-start",
        action="store_true",
        help="Run command phases but skip starting the app / readiness poll",
    )
    verify_parser.add_argument(
        "--phase",
        action="append",
        choices=["bootstrap", "build", "test", "start"],
        default=None,
        help="Run only the given phase(s); repeatable",
    )
    verify_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the port used for the readiness poll",
    )
    verify_parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PHASE_TIMEOUT,
        help=f"Per-phase timeout in seconds (default: {DEFAULT_PHASE_TIMEOUT:.0f})",
    )
    verify_parser.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT,
        help=f"Readiness poll timeout in seconds (default: {DEFAULT_READY_TIMEOUT:.0f})",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON result",
    )
    verify_parser.set_defaults(func=cmd_verify)
