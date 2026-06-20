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

    parser.set_defaults(func=cmd_agentcyber)
