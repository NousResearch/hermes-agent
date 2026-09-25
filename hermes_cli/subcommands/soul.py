"""``hermes soul`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_soul_parser(subparsers, *, cmd_soul: Callable) -> None:
    """Attach the ``soul`` subcommand to ``subparsers``."""
    soul_parser = subparsers.add_parser(
        "soul", help="Testable agent constitutions (SOUL.md)",
        description="Validate a project's testable soul constitution and "
                    "deterministically score its probe suite.")
    soul_sub = soul_parser.add_subparsers(dest="soul_cmd", required=True)
    validate_parser = soul_sub.add_parser(
        "validate", help="Validate SOUL.md and its paired probe suite")
    validate_parser.set_defaults(func=cmd_soul)
    eval_parser = soul_sub.add_parser(
        "eval",
        help="Deterministically score soul probe responses and check the release gates")
    eval_parser.add_argument(
        "--suite", required=True, help="Path to the soul eval suite YAML")
    eval_parser.add_argument(
        "--responses", required=True, help="Path to probe responses JSONL")
    eval_parser.add_argument(
        "--baseline", default=None,
        help="Path to a frozen baseline JSON for drift diff")
    eval_parser.add_argument(
        "--report", default=None, help="Write the markdown report to this path")
    eval_parser.set_defaults(func=cmd_soul)
