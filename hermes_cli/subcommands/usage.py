"""``hermes usage`` — every signed-in account's limits without starting a session.

The default view is a table, one row per account of every signed-in provider
(``hermes_cli.usage_table``). ``--details`` prints the account-limits block of the REPL ``/usage``
instead, a credential pool account by account (``hermes_cli.usage_pool``).

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
        "email": snapshot.email,
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


def pool_usage_document(provider: str, accounts: list[dict]) -> dict:
    """``hermes usage --json`` document for a credential pool: the single-account keys plus
    ``accounts`` (the picker's per-account rows) and ``pool`` (counts per state). A pool has no
    provider-wide gauge (one login's quota says nothing about its siblings), so ``windows`` stays
    empty and the per-account rows carry the numbers."""
    from datetime import UTC, datetime

    from hermes_cli.usage_pool import pool_summary

    return {
        "provider": provider, "source": "credential_pool", "title": "Account pool", "plan": None, "email": None,
        "fetched_at": datetime.now(UTC).isoformat(), "windows": [], "details": [],
        "unavailable_reason": None, "accounts": accounts, "pool": pool_summary(accounts),
    }


def _usage_providers(args: argparse.Namespace) -> list[str]:
    """``--provider`` alone; otherwise the configured provider, then every pooled provider (the
    table shows them all; ``--json`` and ``--details`` only with ``--all``)."""
    from hermes_cli.auth import read_credential_pool
    from hermes_cli.runtime_provider import resolve_requested_provider

    requested = resolve_requested_provider(getattr(args, "provider", None))
    if getattr(args, "provider", None) or not _shows_every_provider(args):
        return [requested]
    return list(dict.fromkeys([requested, *(slug for slug, rows in read_credential_pool().items() if rows)]))


def _shows_every_provider(args: argparse.Namespace) -> bool:
    return getattr(args, "all", False) or not (getattr(args, "json", False) or getattr(args, "details", False))


def cmd_usage(args: argparse.Namespace) -> int:
    """Print one row per account of every signed-in provider (``--details``: the ``/usage`` block
    per provider, a pool account by account; ``--json``: documents). Exit 1 when nothing could be
    fetched."""
    from agent.account_usage import fetch_account_usage, render_account_usage_lines
    from agent.account_usage_cache import has_account_usage
    from hermes_cli.inventory import pool_usage_accounts
    from hermes_cli.usage_pool import fetch_pool_accounts, render_pool_lines
    from hermes_cli.usage_table import pool_rows, render_table, snapshot_row

    documents: list[dict] = []
    blocks: list[list[str]] = []
    rows = []
    providers = _usage_providers(args)
    for provider in providers:
        accounts = fetch_pool_accounts(provider)
        if accounts:
            documents.append(pool_usage_document(provider, accounts))
            blocks.append(render_pool_lines(provider, accounts))
            rows += pool_rows(provider, accounts, has_usage=True)
            continue
        # No explicit key: the fetcher resolves the credential exactly as a session without a live agent
        # would (singleton store, then credential pool) — it never adopts or refreshes anything else.
        snapshot = fetch_account_usage(provider) if has_account_usage(provider) else None
        if snapshot is not None:
            documents.append(usage_snapshot_document(snapshot))
            blocks.append(render_account_usage_lines(snapshot))
            rows.append(snapshot_row(provider, snapshot))
        elif not has_account_usage(provider):
            # Signed in but no usage endpoint (Copilot): still a row, so every usable login is listed.
            rows += pool_rows(provider, pool_usage_accounts(provider, []) or [], has_usage=False)
    if not documents:
        print(
            f"No account usage available for provider '{providers[0]}': no credential is configured for it, "
            "the provider has no usage endpoint, or the fetch failed.",
            file=sys.stderr,
        )
        return 1
    if getattr(args, "json", False):
        print(json.dumps({"providers": documents} if getattr(args, "all", False) else documents[0], indent=2))
    elif getattr(args, "details", False):
        print("\n\n".join("\n".join(block) for block in blocks))
    else:
        print("\n".join(render_table(rows)))
    return 0


def build_usage_parser(subparsers) -> None:
    """Attach the ``usage`` subcommand to ``subparsers``."""
    usage_parser = subparsers.add_parser(
        "usage", help="Show every signed-in account's rate-limit windows without starting a session",
        description="Fetch account limits (Codex 5h/weekly windows, plan, banked resets; Anthropic OAuth windows; "
                    "OpenRouter credits) and print one row per account of every signed-in provider; --details "
                    "prints the block the /usage slash command shows. Exit code 1 when no credential is "
                    "configured or the fetch fails.",
    )
    usage_parser.add_argument(
        "--provider", default=None, help="Provider to query (default: the configured model provider)")
    usage_parser.add_argument(
        "--json", action="store_true", help="Print one JSON document (the configured provider's) instead of the table")
    usage_parser.add_argument(
        "--all", action="store_true",
        help="With --json or --details, also show every other pooled provider (the table always does); "
             "with --json, print {\"providers\": [one document per provider]}")
    usage_parser.add_argument(
        "--details", action="store_true",
        help="Print the /usage block per provider (every window, a pool account by account) instead of the table")
    usage_parser.set_defaults(func=cmd_usage)
