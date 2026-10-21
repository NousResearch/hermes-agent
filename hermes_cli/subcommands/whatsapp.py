"""``hermes whatsapp`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_whatsapp_parser(subparsers, *, cmd_whatsapp: Callable) -> None:
    """Attach the ``whatsapp`` subcommand to ``subparsers``."""
    from hermes_cli.main_platform_setup import cmd_whatsapp_reaction

    whatsapp_parser = subparsers.add_parser(
        "whatsapp", help="Set up WhatsApp or react to a message",
        description="Configure WhatsApp, pair via QR code, or react to a recent message")
    whatsapp_parser.set_defaults(func=cmd_whatsapp)
    actions = whatsapp_parser.add_subparsers(dest="whatsapp_action")
    for action in ("react", "unreact"):
        command = actions.add_parser(action, help=f"{action.capitalize()} to a recent WhatsApp message")
        command.add_argument("--to", required=True, metavar="CHAT", help="Phone number or WhatsApp chat JID")
        command.add_argument("--message-id", required=True, metavar="ID", help="Original WhatsApp message ID")
        if action == "react":
            command.add_argument("emoji", help="Reaction emoji")
        command.set_defaults(func=cmd_whatsapp_reaction)


def build_whatsapp_cloud_parser(subparsers, *, cmd_whatsapp_cloud: Callable) -> None:
    """Attach the ``whatsapp-cloud`` subcommand (official Meta Cloud API)."""
    whatsapp_cloud_parser = subparsers.add_parser(
        "whatsapp-cloud", help="Set up WhatsApp Business Cloud API integration",
        description="Configure the official Meta WhatsApp Business Cloud API "
            "adapter (Business account required, public webhook URL "
            "required). Distinct from `hermes whatsapp` which sets up "
            "the Baileys bridge for personal accounts.")
    whatsapp_cloud_parser.set_defaults(func=cmd_whatsapp_cloud)
