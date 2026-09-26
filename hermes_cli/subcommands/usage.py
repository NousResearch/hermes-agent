"""``hermes usage`` — the account-limits block of the REPL ``/usage`` without starting a session.

Script-friendly Codex / Anthropic / OpenRouter quota view (issue #33094): same fetch and renderer as
``/usage`` (``agent.account_usage``), same credential resolution as a session with no live agent, plus
``--json`` for cron jobs and shell loops. Slim redo of #81819 (@himanusia).
"""

from __future__ import annotations

import argparse
import json
import sys


def usage_snapshot_document(snapshot) -> dict:
    """``hermes usage --json`` document. Schema is documented in website/docs/reference/cli-commands.md —
    keep the keys stable; extend only by adding keys."""
    return {
        "provider": snapshot.provider,
        "source": snapshot.source,
        "title": snapshot.title,
        "plan": snapshot.plan,
        "fetched_at": snapshot.fetched_at.isoformat(),
        "windows": [
            {
                "label": window.label,
                "used_percent": window.used_percent,
                "resets_at": window.reset_at.isoformat() if window.reset_at else None,
                "detail": window.detail,
            }
            for window in snapshot.windows
        ],
        "details": list(snapshot.details),
        "unavailable_reason": snapshot.unavailable_reason,
    }


def cmd_usage(args: argparse.Namespace) -> int:
    """Print the configured (or ``--provider``) account's usage windows; exit 1 when nothing could be fetched."""
    from agent.account_usage import fetch_account_usage, render_account_usage_lines
    from hermes_cli.runtime_provider import resolve_requested_provider

    provider = resolve_requested_provider(getattr(args, "provider", None))
    # No explicit key: the fetcher resolves the credential exactly as a session without a live agent
    # would (singleton store, then credential pool) — it never adopts or refreshes anything else.
    snapshot = fetch_account_usage(provider)
    if snapshot is None or snapshot.unavailable_reason:
        reason = (
            snapshot.unavailable_reason
            if snapshot is not None
            else (
                "no credential is configured for it, the provider has no usage endpoint, or the fetch failed."
            )
        )
        print(
            f"No account usage available for provider '{provider}': {reason}",
            file=sys.stderr,
        )
        return 1
    if getattr(args, "json", False):
        print(json.dumps(usage_snapshot_document(snapshot), indent=2))
    else:
        print("\n".join(render_account_usage_lines(snapshot)))
    return 0


def build_usage_parser(subparsers) -> None:
    """Attach the ``usage`` subcommand to ``subparsers``."""
    usage_parser = subparsers.add_parser(
        "usage", help="Show account rate-limit windows (the /usage block) without starting a session",
        description="Fetch the configured provider's account limits (Codex 5h/weekly windows, plan, banked "
                    "resets; Anthropic OAuth windows; OpenRouter credits) — the same block the /usage slash "
                    "command prints — and exit. Exit code 1 when no credential is configured or the fetch fails.",
    )
    usage_parser.add_argument(
        "--provider", default=None, help="Provider to query (default: the configured model provider)")
    usage_parser.add_argument(
        "--json", action="store_true", help="Print one JSON document instead of the human-readable block")
    usage_parser.set_defaults(func=cmd_usage)
