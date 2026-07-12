"""Plugin slash command for reviewing PARA vault triage audit items."""

from __future__ import annotations

from hermes_cli.vault_para_triage import handle_feedback_command


def register(ctx) -> None:
    ctx.register_command(
        "para-feedback",
        handle_feedback_command,
        description="Review or correct PARA vault triage audit items",
        args_hint="list|status|approve <entry_id>|correct <entry_id> <target>|ignore <entry_id>",
    )
