"""Context Governor bundled plugin."""

from __future__ import annotations


def register(ctx):
    from hermes_cli.context_governor import (
        checkpoint_command,
        cli_main,
        pre_context_compress_handoff,
        setup_cli,
    )

    ctx.register_hook("pre_context_compress", pre_context_compress_handoff)
    ctx.register_command(
        "checkpoint",
        checkpoint_command,
        description="Write a deterministic handoff/checkpoint for the current Hermes session",
        args_hint="[current task/goal]",
    )
    ctx.register_cli_command(
        "checkpoint",
        help="Write a deterministic Hermes handoff/checkpoint",
        setup_fn=setup_cli,
        handler_fn=cli_main,
        description=(
            "Writes repo-local CURRENT.md when --cwd/current directory is inside a git repo; "
            "otherwise writes under $HERMES_HOME/handoffs/. No model call is made."
        ),
    )
