"""``hermes subagent`` model and reasoning parser.

Wired from ``hermes_cli/main.py``.  Provides the shell-facing entry point
for subagent model selection:

    hermes subagent                       # status
    hermes subagent model                 # interactive picker
    hermes subagent model <model>         # validated direct selection
    hermes subagent model reset           # inherit parent
    hermes subagent model --reset         # inherit parent (flag form)
    hermes subagent reasoning high        # fixed child reasoning effort
    hermes subagent reasoning reset       # inherit parent reasoning
"""

from __future__ import annotations

from typing import Callable


def build_subagent_parser(subparsers, *, cmd_subagent: Callable) -> None:
    """Attach the ``subagent`` subcommand to ``subparsers``."""
    subagent_parser = subparsers.add_parser(
        "subagent",
        help="Inspect or configure the subagent model and reasoning",
        description=(
            "Show the current subagent model selection.  When no override "
            "is configured, subagents inherit the parent model."
        ),
    )
    subparsers_sub = subagent_parser.add_subparsers(dest="subagent_command")

    # subagent (no subcommand) → status
    subagent_parser.set_defaults(func=cmd_subagent)

    # subagent model → status / select / reset
    model_parser = subparsers_sub.add_parser(
        "model",
        help="Select or reset the subagent model",
        description=(
            "Pin all subagents to a specific model, or reset to inherit "
            "the parent model. With no model argument, opens the complete "
            "`hermes model` provider setup flow, including login and custom "
            "endpoint creation. Shared provider additions are retained while "
            "the active primary model remains unchanged. The delegation "
            "provider/model is read on every child spawn — no restart needed."
        ),
    )
    model_parser.add_argument(
        "model",
        nargs="?",
        help="Model to pin, or 'reset' to inherit the parent model",
    )
    model_parser.add_argument(
        "--provider",
        default=None,
        help="Provider to route subagents through (e.g. 'openrouter', 'nous')",
    )
    model_parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the subagent model/provider override (inherit parent)",
    )
    model_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh provider model catalogs before opening the full setup picker",
    )
    model_parser.set_defaults(func=cmd_subagent)

    reasoning_parser = subparsers_sub.add_parser(
        "reasoning",
        help="Inspect, set, or reset subagent reasoning effort",
        description=(
            "Set the reasoning effort used by newly spawned subagents. "
            "Without an override, children inherit the parent agent's "
            "reasoning configuration. Changes are read on every child spawn."
        ),
    )
    reasoning_parser.add_argument(
        "effort",
        nargs="?",
        help="none|minimal|low|medium|high|xhigh|max|ultra, or 'reset'",
    )
    reasoning_parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the subagent reasoning override (inherit parent)",
    )
    reasoning_parser.set_defaults(func=cmd_subagent)
