"""Teams meeting pipeline plugin.

Registers only operator-facing CLI surfaces. The agent should invoke these via
the terminal tool; no model tools are added by this plugin.
"""

from __future__ import annotations

from plugins.teams_pipeline.cli import register_cli, teams_pipeline_command


def register(ctx) -> None:
    ctx.register_cli_command(
        name="teams-pipeline",
        help="Inspect and operate the Microsoft Teams meeting pipeline",
        setup_fn=register_cli,
        handler_fn=teams_pipeline_command,
        description=(
            "Operator CLI for the Microsoft Teams meeting pipeline. "
            "Lists jobs, inspects stored runs, replays jobs, validates Graph "
            "setup, and maintains Graph subscriptions."
        ),
    )
