"""``hermes whatsapp`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_whatsapp_parser(subparsers, *, cmd_whatsapp: Callable) -> None:
    """Attach the ``whatsapp`` subcommand to ``subparsers``."""
    # =========================================================================
    # whatsapp command
    # =========================================================================
    whatsapp_parser = subparsers.add_parser(
        "whatsapp",
        help="Set up WhatsApp integration",
        description="Configure WhatsApp and pair via QR code",
    )
    whatsapp_parser.set_defaults(func=cmd_whatsapp)
