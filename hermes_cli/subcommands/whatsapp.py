"""``hermes whatsapp`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_whatsapp_parser(subparsers, *, cmd_whatsapp: Callable) -> None:
    """Attach the ``whatsapp`` subcommand to ``subparsers``."""
    whatsapp_parser = subparsers.add_parser(
        "whatsapp", help="Set up WhatsApp integration",
        description="Configure WhatsApp and pair via QR code")
    whatsapp_parser.set_defaults(func=cmd_whatsapp)


def build_whatsapp_cloud_parser(subparsers, *, cmd_whatsapp_cloud: Callable) -> None:
    """Attach the ``whatsapp-cloud`` subcommand (official Meta Cloud API)."""
    whatsapp_cloud_parser = subparsers.add_parser(
        "whatsapp-cloud", help="Set up WhatsApp Business Cloud API integration",
        description="Configure the official Meta WhatsApp Business Cloud API "
            "adapter (Business account required, public webhook URL "
            "required). Distinct from `hermes whatsapp` which sets up "
            "the Baileys bridge for personal accounts.")
    whatsapp_cloud_parser.set_defaults(func=cmd_whatsapp_cloud)
