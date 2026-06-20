"""``hermes smoke`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_smoke_parser(subparsers, *, cmd_smoke: Callable) -> None:
    """Attach the ``smoke`` subcommand to ``subparsers``."""
    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run runtime smoke checks",
        description=(
            "Run a compact runtime smoke report. By default this is side-effect-light: "
            "no artifacts are written and no persistent chat sessions are created. "
            "Use --chat and/or --write-artifacts when those side effects are desired."
        ),
    )
    smoke_parser.add_argument(
        "--profiles",
        default="default,cheap,lab",
        help="Comma-separated profiles for exact-string chat smokes. Default: default,cheap,lab",
    )
    smoke_parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory. Implies artifact writing. Default: do not write artifacts",
    )
    smoke_parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Write stdout/stderr and summary artifacts under the platform temp directory",
    )
    smoke_parser.add_argument("--chat", action="store_true", help="Opt in to real model/provider chat smokes")
    smoke_parser.add_argument("--skip-chat", action="store_true", help="Deprecated compatibility alias; chat smokes are skipped by default")
    smoke_parser.add_argument("--credits", action="store_true", help="Check OpenRouter credits if configured")
    smoke_parser.add_argument("--cli", default=None, help="Hermes CLI executable to smoke. Default: resolve hermes on PATH")
    smoke_parser.add_argument("--json", action="store_true", help="Emit JSON")
    smoke_parser.set_defaults(func=cmd_smoke)
