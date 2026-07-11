"""``hermes slack`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_slack_parser(subparsers, *, cmd_slack: Callable) -> None:
    """Attach the ``slack`` subcommand to ``subparsers``."""
    # =========================================================================
    # slack command
    # =========================================================================
    slack_parser = subparsers.add_parser(
        "slack",
        help="Slack integration helpers (manifest generation, etc.)",
        description="Slack integration helpers for HT AI Agent.",
    )
    slack_sub = slack_parser.add_subparsers(dest="slack_command")
    slack_manifest = slack_sub.add_parser(
        "manifest",
        help="Print or write a Slack app manifest with every gateway command "
        "registered as a native slash (/btw, /stop, /model, ...)",
        description=(
            "Generate a Slack app manifest that registers every gateway "
            "command in COMMAND_REGISTRY as a first-class Slack slash "
            "command (matching Discord and Telegram parity). Paste the "
            "output into Slack app config → Features → App Manifest → "
            "Edit, then Save. Reinstall the app if Slack prompts for it."
        ),
    )
    slack_manifest.add_argument(
        "--write",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help="Write manifest to a file instead of stdout. With no PATH "
        "writes to $HERMES_HOME/slack-manifest.json.",
    )
    slack_manifest.add_argument(
        "--name",
        default=None,
        help='Bot display name (default: "HT AI Agent")',
    )
    slack_manifest.add_argument(
        "--description",
        default=None,
        help="Bot description shown in Slack's app directory.",
    )
    slack_manifest.add_argument(
        "--slashes-only",
        action="store_true",
        help="Emit only the features.slash_commands array (for merging "
        "into an existing manifest manually).",
    )
    slack_manifest.add_argument(
        "--no-assistant",
        action="store_true",
        help="Omit Slack AI Assistant mode (assistant_view, assistant:write "
        "scope, assistant_thread_* events). DMs then render as a flat chat "
        "where bare slash commands (/help, /new) work inline instead of "
        "Slack's Assistant thread pane.",
    )
    slack_parser.set_defaults(func=cmd_slack)
