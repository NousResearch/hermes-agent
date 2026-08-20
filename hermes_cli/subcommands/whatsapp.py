"""``hermes whatsapp`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_whatsapp_parser(subparsers, *, cmd_whatsapp: Callable) -> None:
    """Attach the ``whatsapp`` command and its subcommands to ``subparsers``."""
    # =========================================================================
    # whatsapp command
    # =========================================================================
    whatsapp_parser = subparsers.add_parser(
        "whatsapp",
        help="Set up WhatsApp integration",
        description="Configure WhatsApp and pair via QR code. "
        "Use 'hermes whatsapp setup' for guided one-command setup.",
    )
    whatsapp_sub = whatsapp_parser.add_subparsers(dest="whatsapp_command")

    # whatsapp setup (new guided wizard)
    setup_parser = whatsapp_sub.add_parser(
        "setup",
        help="Interactive WhatsApp setup wizard (pair, allowlist, policy, home channel, test)",
        description=(
            "Guided one-command wizard that configures everything: "
            "pairing via QR code, allowlist, dm_policy, cron home channel, "
            "test delivery, and gateway restart."
        ),
    )
    # No extra args needed for now — the wizard is fully interactive.
    setup_parser.set_defaults(func=cmd_whatsapp)

    # If no subcommand is given, the default (bare ``hermes whatsapp``)
    # calls cmd_whatsapp too; the handler checks args.whatsapp_command
    # to distinguish bare mode from subcommands.
    whatsapp_parser.set_defaults(func=cmd_whatsapp)
