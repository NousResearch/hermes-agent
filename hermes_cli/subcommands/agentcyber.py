"""``hermes agentcyber`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_agentcyber_parser(subparsers, *, cmd_agentcyber: Callable) -> None:
    parser = subparsers.add_parser(
        "agentcyber",
        help="Inspect or configure Hermes AgentCyber runtime health",
        description=(
            "AgentCyber status/setup helpers for local/open-weight routing, "
            "authorized asset registry checks, and cyber toolset visibility."
        ),
    )
    sub = parser.add_subparsers(dest="agentcyber_action")

    status = sub.add_parser("status", help="Show AgentCyber health and readiness")
    status.add_argument("--json", action="store_true", help="Emit JSON")
    status.add_argument("--platform", default="cli", help="Platform toolset scope to inspect")

    setup = sub.add_parser("setup", help="Prepare AgentCyber config (dry-run by default)")
    setup.add_argument("--apply", action="store_true", help="Write config.yaml")
    setup.add_argument("--platform", default="cli", help="Platform toolset scope to update")
    setup.add_argument("--provider", default="ollama", help="Local/open-weight provider")
    setup.add_argument("--model", default="qwen3-coder:30b", help="Local/open-weight model")
    setup.add_argument("--base-url", default="http://192.168.1.120:11434/v1", help="Local OpenAI-compatible base URL")
    setup.add_argument("--api-mode", default="chat_completions", help="Runtime API mode")
    setup.add_argument("--enable-live-usb", action="store_true", help="Also enable live_usb toolset")

    breakglass = sub.add_parser("breakglass", help="Manage AgentCyber S4/S5 break-glass approvals")
    bg_sub = breakglass.add_subparsers(dest="breakglass_action")

    create = bg_sub.add_parser("create", aliases=["request"], help="Create a scoped break-glass approval")
    create.add_argument("--tool", required=True, help="Tool name, e.g. terminal")
    create.add_argument("--args-json", required=True, help="Exact tool arguments as JSON")
    create.add_argument("--operator", required=True, help="Approving operator label")
    create.add_argument("--reason", required=True, help="Human approval reason")
    create.add_argument("--ttl-minutes", type=int, default=15, help="Approval lifetime")
    create.add_argument("--store", default="", help="Override break-glass store path")
    create.add_argument("--apply", action="store_true", help="Write approval; default is dry-run")
    create.add_argument("--json", action="store_true", help="Emit JSON")

    list_p = bg_sub.add_parser("list", help="List break-glass approvals without raw args")
    list_p.add_argument("--store", default="", help="Override break-glass store path")
    list_p.add_argument("--json", action="store_true", help="Emit JSON")

    revoke = bg_sub.add_parser("revoke", help="Revoke a break-glass approval")
    revoke.add_argument("approval_id", help="Approval id to revoke")
    revoke.add_argument("--store", default="", help="Override break-glass store path")
    revoke.add_argument("--json", action="store_true", help="Emit JSON")

    parser.set_defaults(func=cmd_agentcyber)
