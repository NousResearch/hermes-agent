"""``hermes usage`` for credential pools: every account, every window, and why a blank one is blank.

The picker reads usage from a cache and refreshes it in the background; a command can afford to wait,
so this fetches each pooled account now (read-only, in parallel) and then renders the same per-account
rows the picker gets (``hermes_cli.inventory.pool_usage_accounts``), so both surfaces agree on state.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime
from typing import Optional

# Upper bound on concurrent usage requests, so a large pool does not open one connection per account.
_FETCH_WORKERS = 8


def fetch_pool_accounts(provider: str, *, refresh_expired: bool = False) -> Optional[list[dict]]:
    """Per-account usage rows for *provider*'s pool, fetched now; None unless the provider has a
    multi-entry pool and a usage endpoint (the picker's rule), so every other case keeps the
    single-account path.

    Each account is read with its own token and never refreshed on failure. *refresh_expired* first
    runs the pool's pre-request refresh (``CredentialPool.reclaim``, what ``select()`` runs) for each
    live entry whose access token has expired, so it reports usage instead of a 401."""
    from agent.account_usage import fetch_account_usage
    from agent.account_usage_cache import has_account_usage
    from agent.credential_pool import STATUS_DEAD, load_pool
    from agent.credential_pool_timestamps import token_expires_at
    from hermes_cli.auth import read_credential_pool
    from hermes_cli.inventory import pool_usage_accounts

    if len(read_credential_pool(provider)) < 2 or not has_account_usage(provider):
        return None
    if refresh_expired:
        pool = load_pool(provider)
        now = time.time()
        for entry in pool.entries():
            expires = token_expires_at(entry)
            if entry.last_status != STATUS_DEAD and expires is not None and expires <= now:
                pool.reclaim(entry.id)
    requests: list[dict] = []
    pool_usage_accounts(provider, requests)
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as workers:
        # Each fetch runs in the caller's context, so its cache slot is keyed to the caller's profile.
        fetches = [
            workers.submit(copy_context().run, fetch_account_usage, provider, base_url=request["base_url"],
                           api_key=request["api_key"], read_only=True, identity_id=request["identity_id"])
            for request in requests
        ]
    for fetch in fetches:
        fetch.result()
    # Rebuilt from the cache the fetches just filled, so each state reflects fresh windows.
    return pool_usage_accounts(provider, [])


def pool_summary(accounts: list[dict]) -> dict[str, int]:
    counts = {"ready": 0, "limited": 0, "unknown": 0, "unavailable": 0}
    for account in accounts:
        counts[account["state"]] = counts.get(account["state"], 0) + 1
    return {**counts, "total": len(accounts)}


def _token_line(provider: str, account: dict) -> str:
    if account["state"] == "unavailable":
        return f"login dead: sign in again with `hermes auth add {provider}`"
    expires = account.get("token_expires_at")
    if not expires:
        return ""
    left = datetime.fromisoformat(expires).timestamp() - time.time()
    if left <= 0:
        return "token expired (idle): refreshes on its next request, or run `hermes usage --refresh`"
    return f"token expires in {max(1, round(left / 3600))}h" if left >= 3600 else f"token expires in {round(left / 60)}m"


def render_pool_lines(provider: str, accounts: list[dict]) -> list[str]:
    """One summary line, then one block per account: state, every window, token health."""
    from agent.account_usage import _format_reset, render_account_usage_body
    from agent.account_usage_cache import cached_account_usage

    counts = pool_summary(accounts)
    parts = [f"{counts['ready']} of {counts['total']} accounts ready"]
    parts += [f"{counts[state]} {state}" for state in ("limited", "unknown", "unavailable") if counts[state]]
    lines = [f"📈 Account pool: {provider} — " + " · ".join(parts)]
    for index, account in enumerate(accounts, start=1):
        state = account["state"]
        if state == "limited":
            resets_at = account.get("resets_at")
            state += f" until {_format_reset(datetime.fromisoformat(resets_at) if resets_at else None)}"
        lines.append("")
        lines.append(f"• {account.get('label') or f'Account {index}'}: {state}")
        snapshot = cached_account_usage(provider, identity_id=account["id"]) if account["windows"] else None
        body = render_account_usage_body(snapshot) if snapshot else []
        if snapshot and snapshot.plan:
            body.insert(0, f"Plan: {snapshot.plan}")
        token = _token_line(provider, account)
        lines.extend(f"    {line}" for line in [*body, *([token] if token else [])])
    return lines
