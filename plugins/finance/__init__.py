"""Finance integration plugin — bundled, auto-loaded.

Provider-agnostic personal-finance integration (Plaid is the first backend).
Registers:

* a ``hermes finance`` CLI command (connect / sync / status / accounts /
  transactions / disconnect), and
* a gated ``finance`` toolset (``finance_*``) backed by a local SQLite cache.

The model tools are service-gated via ``_check_finance_available`` — they only
appear once the active provider's credentials are configured AND an account is
linked, so users who don't use finance pay no prompt-cache/schema cost. User
onboarding and provider API calls go through the CLI and the explicit
``finance_sync`` tool; everything else reads the local cache.

See ``website/docs/user-guide/features/finance.md`` for the full guide.
"""

from __future__ import annotations

from plugins.finance.tools import FINANCE_TOOLS, _check_finance_available


def register(ctx) -> None:
    """Register the finance CLI command and gated tools (called once at load)."""
    from plugins.finance.cli import register_cli, finance_command

    ctx.register_cli_command(
        name="finance",
        help="Personal finance: connect accounts, sync, and query balances/transactions",
        setup_fn=register_cli,
        handler_fn=finance_command,
        description="Provider-agnostic personal finance integration (Plaid).",
    )

    for name, schema, handler, emoji in FINANCE_TOOLS:
        ctx.register_tool(
            name=name,
            toolset="finance",
            schema=schema,
            handler=handler,
            check_fn=_check_finance_available,
            requires_env=["PLAID_CLIENT_ID", "PLAID_SECRET"],
            emoji=emoji,
        )
