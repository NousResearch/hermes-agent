"""Argument parser for ``hermes evals``."""

from __future__ import annotations

from pathlib import Path


def build_evals_parser(subparsers, *, cmd_evals):
    parser = subparsers.add_parser(
        "evals",
        help="Mine session traces into portable evaluation tasks",
        description=(
            "Create sanitized, review-required task candidates from real Hermes "
            "sessions and validate evaluation corpora."
        ),
    )
    actions = parser.add_subparsers(dest="evals_action")

    mine = actions.add_parser(
        "mine",
        help="Convert a session trace into a sanitized task candidate",
    )
    mine.add_argument("session_id", help="Exact session ID or unique prefix")
    mine.add_argument(
        "--output",
        type=Path,
        help="Output YAML path (default: ~/.hermes/evals/candidates/<task-id>.yaml)",
    )

    mine.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing regular output file; symlinks are always refused",
    )

    validate = actions.add_parser(
        "validate",
        help="Validate one task manifest or every YAML file in a corpus directory",
    )
    validate.add_argument("path", type=Path, help="Task YAML file or corpus directory")
    validate.add_argument(
        "--ready",
        action="store_true",
        help="Fail when a manifest is valid but still requires review or approval",
    )

    score = actions.add_parser(
        "score",
        help="Apply deterministic task checks to a recorded JSON run artifact",
    )
    score.add_argument("task", type=Path, help="Approved task YAML manifest")
    score.add_argument("run", type=Path, help="Recorded run artifact in JSON format")
    score.add_argument("--output", type=Path, help="Write the score as JSON")
    score.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing regular score file; symlinks are always refused",
    )

    parser.set_defaults(func=cmd_evals)
    return parser
