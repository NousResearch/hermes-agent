"""``hermes cleanse`` / ``hermes check-fix`` subcommand parser.

Follows the pattern of verify.py: parser built here, handler injected to
avoid importing main.
"""

from __future__ import annotations

from typing import Callable


def build_cleanse_parser(subparsers, *, cmd_cleanse: Callable) -> None:
    """Attach the ``cleanse`` and ``check-fix`` subcommands to ``subparsers``."""
    # Main command: hermes cleanse
    cleanse_parser = subparsers.add_parser(
        "cleanse",
        help="Run project-checker repair loop",
        description=(
            "Detect issues via VCS + manifests + linters, apply automated "
            "fixes with file-sticky workers, and verify. Success = empty "
            "remaining diagnostics + written verification_evidence.json."
        ),
    )
    cleanse_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project root directory (default: current directory)",
    )
    cleanse_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum parallel workers (default: 2)",
    )
    cleanse_parser.add_argument(
        "--tests",
        action="store_true",
        help="Run project tests after fixes",
    )
    cleanse_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable format",
    )
    cleanse_parser.set_defaults(func=cmd_cleanse)

    # Alias: hermes check-fix
    checkfix_parser = subparsers.add_parser(
        "check-fix",
        help="Alias for cleanse",
        description=(
            "Detect issues, apply automated fixes, and verify (alias for cleanse)"
        ),
    )
    checkfix_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project root directory (default: current directory)",
    )
    checkfix_parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum parallel workers (default: 2)",
    )
    checkfix_parser.add_argument(
        "--tests",
        action="store_true",
        help="Run project tests after fixes",
    )
    checkfix_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable format",
    )
    checkfix_parser.set_defaults(func=cmd_cleanse)
