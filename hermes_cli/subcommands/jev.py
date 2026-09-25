"""``hermes jev`` subcommand parser."""

from __future__ import annotations

from hermes_cli.jev import cmd_classify_task, cmd_jev, cmd_models


def _dispatch_jev(args) -> int:
    """Preserve the original decision form while exposing task classification."""
    if args.state == "classify-task":
        if not args.task_text:
            args.parser.error("classify-task requires task text")
        return cmd_classify_task(args)
    if args.state == "models":
        if args.task_text:
            args.parser.error("models does not accept task text")
        if args.questions:
            args.parser.error("models does not accept --questions")
        return cmd_models(args)
    if args.task_text:
        args.parser.error(
            "unexpected extra argument; did you mean 'jev classify-task <text>'?"
        )
    if not args.questions:
        args.parser.error("the decision form requires --questions")
    return cmd_jev(args)


def build_jev_parser(subparsers) -> None:
    """Attach the ``jev`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "jev",
        help="Make structured decisions with OpenRouter's Jev model",
        description=(
            "Submit a state and noul, choice, or score questions to the "
            "OpenRouter Decisions API. Use `jev classify-task TASK` for advisory "
            "orchestration classification or `jev models` to inspect the current "
            "authorized candidate catalog. Output is deterministic JSON."
        ),
    )
    parser.add_argument(
        "state",
        help="Decision state, or the literal classify-task or models",
    )
    parser.add_argument(
        "task_text",
        nargs="?",
        help="Task text when the first positional argument is classify-task",
    )
    parser.add_argument(
        "--questions",
        "-q",
        metavar="JSON_OR_FILE",
        help="Question mapping for the general decision form as inline JSON or a file path",
    )
    parser.add_argument(
        "--tier",
        choices=("cheap_fast", "mid", "premium"),
        help="For `jev models`, include the safe fallback order from this tier upward",
    )
    parser.add_argument(
        "--min-context-length",
        type=int,
        default=0,
        help="Require at least this many context tokens when resolving a task or fallback order",
    )
    parser.add_argument(
        "--require-reasoning",
        action="store_true",
        help="Require catalog-declared reasoning support during task or fallback resolution",
    )
    parser.add_argument(
        "--require-structured-outputs",
        action="store_true",
        help="Require catalog-declared structured-output support during resolution",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="For `jev models`, skip Jev tiering and inspect the deterministic catalog prefilter",
    )
    parser.set_defaults(func=_dispatch_jev, parser=parser)
