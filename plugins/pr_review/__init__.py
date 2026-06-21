"""Hermes PR reviewer plugin.

Hermes-first code review surface: no IDE integration, no SaaS-shaped wrapper.
The plugin exposes operator CLI commands that use Hermes' host-owned model/auth
through ``ctx.llm`` and GitHub access through the local ``gh`` CLI.
"""

from __future__ import annotations

from plugins.pr_review.cli import pr_review_command, register_cli


def register(ctx) -> None:
    ctx.register_cli_command(
        name="pr-review",
        help="Hermes-native pull request reviewer",
        setup_fn=register_cli,
        handler_fn=lambda args: pr_review_command(args, ctx=ctx),
        description=(
            "Review GitHub pull requests with Hermes' configured model/auth, "
            "trusted base-branch repo docs, structured diagnostics, and local artifacts."
        ),
    )
