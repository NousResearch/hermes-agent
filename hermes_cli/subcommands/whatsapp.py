"""``hermes whatsapp`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_whatsapp_parser(subparsers, *, cmd_whatsapp: Callable) -> None:
    """Attach the ``whatsapp`` subcommand to ``subparsers``."""
    whatsapp_parser = subparsers.add_parser(
        "whatsapp", help="Set up WhatsApp integration",
        description="Configure WhatsApp and pair via QR code")
    whatsapp_parser.set_defaults(func=cmd_whatsapp)
    whatsapp_parser.add_argument(
        "--list-groups", action="store_true",
        help="List participating groups from an already connected local bridge (read-only)",
    )
    whatsapp_parser.add_argument(
        "--bridge-port", type=int, default=None,
        help="Local bridge port for --list-groups (default: 3000)",
    )


def build_whatsapp_cloud_parser(subparsers, *, cmd_whatsapp_cloud: Callable) -> None:
    """Attach the ``whatsapp-cloud`` subcommand (official Meta Cloud API)."""
    whatsapp_cloud_parser = subparsers.add_parser(
        "whatsapp-cloud", help="Set up WhatsApp Business Cloud API integration",
        description="Configure the official Meta WhatsApp Business Cloud API "
            "adapter (Business account required, public webhook URL "
            "required). Distinct from `hermes whatsapp` which sets up "
            "the Baileys bridge for personal accounts.")
    whatsapp_cloud_parser.set_defaults(func=cmd_whatsapp_cloud)
