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


def usage_failure_document(provider: str, kind: str, detail: str) -> dict:
    """``hermes usage --json`` when nothing could be fetched. Keys are stable; ``error`` is one of
    ``no_credential``, ``no_usage_endpoint`` or ``fetch_failed``."""
    return {"provider": provider, "error": kind, "detail": detail}


def _usage_failure(provider: str, failures: list[BaseException]) -> tuple[str, str]:
    """``(kind, detail)`` for an empty fetch: keep the fetch exception when there is one, otherwise
    name the missing prerequisite."""
    if failures:
        from agent.account_usage import describe_fetch_error
        return "fetch_failed", f"the fetch failed: {describe_fetch_error(failures[0])}"
    from agent.account_usage_cache import has_account_usage
    if not has_account_usage(provider):
        return "no_usage_endpoint", "the provider has no usage endpoint"
    return "no_credential", "no credential is configured for it"


def cmd_usage(args: argparse.Namespace) -> int:
    """Print the configured (or ``--provider``) account's usage windows; exit 1 when nothing could be fetched."""
    from agent.account_usage import fetch_account_usage, render_account_usage_lines
    from hermes_cli.runtime_provider import resolve_requested_provider

    provider = resolve_requested_provider(getattr(args, "provider", None))
    # No explicit key: the fetcher resolves the credential exactly as a session without a live agent
    # would (singleton store, then credential pool) — it never adopts or refreshes anything else.
    failures: list[BaseException] = []
    snapshot = fetch_account_usage(provider, on_error=failures.append)
    if snapshot is None:
        kind, detail = _usage_failure(provider, failures)
        if getattr(args, "json", False):
            print(json.dumps(usage_failure_document(provider, kind, detail), indent=2))
        print(f"No account usage available for provider '{provider}': {detail}.", file=sys.stderr)
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
