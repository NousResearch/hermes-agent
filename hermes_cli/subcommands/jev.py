"""``hermes jev`` subcommand parser."""

from __future__ import annotations

from hermes_cli.jev import cmd_classify_task, cmd_jev


def _dispatch_jev(args) -> int:
    """Preserve the original decision form while exposing task classification."""
    if args.state == "classify-task":
        if not args.task_text:
            args.parser.error("classify-task requires task text")
        return cmd_classify_task(args)
    if args.task_text:
        args.parser.error("unexpected extra argument; did you mean 'jev classify-task <text>'?")
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
            "orchestration classification. Output is JSON."
        ),
    )
    parser.add_argument(
        "state",
        help="Decision state, or the literal classify-task",
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
    parser.set_defaults(func=_dispatch_jev, parser=parser)
