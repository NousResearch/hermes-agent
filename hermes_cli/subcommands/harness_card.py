"""``hermes harness-card`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag


def build_harness_card_parser(subparsers, *, cmd_harness_card: Callable) -> None:
    """Attach the ``harness-card`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "harness-card", help="Emit an opt-in, secret-safe Harness Card (reproducibility record)",
        description="Describe the effective harness a benchmark arm ran: Hermes version/commit and "
            "runtime, model route and pricing snapshot, prompt/skill/project-context hashes, toolsets, "
            "memory/compression/limits/delegation/retries, evaluator, and the verification recipe. With "
            "--session, the per-task token/cost rows usage accounting already recorded. Opt-in: enable "
            "with `hermes config set harness_card.enabled true`. Secrets are always redacted and the "
            "card never contains raw prompts, messages, or credentials.")
    parser.add_argument(
        "--platform", default="cli",
        help="Platform whose toolsets to describe (cli, telegram, discord, ...). Default: cli")
    parser.add_argument(
        "--cwd", default=None, help="Project root to describe (default: current directory)")
    parser.add_argument(
        "--session", default=None, help="Add a per-task run export for this session id")
    parser.add_argument(
        "--solved", default=None, choices=["true", "false"],
        help="Record the task outcome; omit to report it as unavailable rather than false")
    parser.add_argument(
        "--failure-class", dest="failure_class", default=None,
        help="Typed failure class for a run that did not solve the task")
    parser.add_argument(
        "--out", default=None,
        help="Write the card as JSON to this path (owner-only permissions where supported)")
    add_json_flag(parser, "Print the card as JSON instead of a summary")
    parser.set_defaults(func=cmd_harness_card)
