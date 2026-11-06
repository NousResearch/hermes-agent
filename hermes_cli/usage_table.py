"""``hermes usage`` default view: one table row per account across every signed-in provider.

Pooled accounts come from the same per-account rows the model picker gets
(``hermes_cli.usage_pool``); a provider with one login comes from its ``/usage`` snapshot. The
block view (``--details``) keeps every window and detail line; the table keeps what decides where
work can go: the 5-hour and weekly windows, when the week resets and whether the account is usable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# The account-wide windows each built-in fetcher reports, by label (agent/account_usage.py).
_SESSION_LABELS = frozenset({"Current session", "Session"})
_WEEK_LABELS = frozenset({"Current week", "Weekly"})
_FULL_PERCENT = 100.0
_HEADER = ("Provider", "Account", "ID", "Email", "Plan", "5h", "Week", "Week resets", "Usable", "Note")


@dataclass(frozen=True)
class UsageRow:
    provider: str
    account: str
    entry_id: str = ""
    email: str = ""
    plan: str = ""
    session_percent: Optional[float] = None
    week_percent: Optional[float] = None
    week_resets_at: Optional[datetime] = None
    usable: str = "?"
    note: str = ""


def _windows(windows: list[dict]) -> tuple[Optional[float], Optional[float], Optional[datetime]]:
    """(5h %, week %, week reset) from wire windows (label, used_percent, resets_at, scope)."""
    session = week = None
    week_reset = None
    for window in windows:
        if window.get("scope", "account") != "account":
            continue
        if window["label"] in _SESSION_LABELS:
            session = window.get("used_percent")
        elif window["label"] in _WEEK_LABELS:
            week = window.get("used_percent")
            week_reset = datetime.fromisoformat(window["resets_at"]) if window.get("resets_at") else None
    return session, week, week_reset


def pool_rows(provider: str, accounts: list[dict], *, has_usage: bool) -> list[UsageRow]:
    """One row per pooled account (the picker's ``accounts`` rows), with the entry id, plan and
    email of the login behind it."""
    from agent.account_usage_cache import _identity_id_for, cached_account_usage
    from agent.credential_pool import load_pool

    entry_ids: dict[str, list[str]] = {}
    for entry in load_pool(provider).entries():
        entry_ids.setdefault(_identity_id_for(provider, entry), []).append(entry.id)
    rows = []
    for account in accounts:
        snapshot = cached_account_usage(provider, identity_id=account["id"]) if has_usage else None
        session, week, week_reset = _windows(account["windows"])
        usable, note = _pool_state(provider, account, has_usage)
        rows.append(UsageRow(
            provider=provider, account=account["label"], entry_id=",".join(entry_ids.get(account["id"], [])),
            email=(snapshot.email if snapshot else None) or "", plan=(snapshot.plan if snapshot else None) or "",
            session_percent=session, week_percent=week, week_resets_at=week_reset, usable=usable, note=note,
        ))
    return rows


def _pool_state(provider: str, account: dict, has_usage: bool) -> tuple[str, str]:
    state = account["state"]
    if state == "ready":
        return "yes", ""
    if state == "unavailable":
        return "no", f"sign in again: hermes auth add {provider}"
    # A blank capped row still knows from its cooldown when it is back: give both.
    blank = _blank_reason(account, has_usage) if not account["windows"] else ""
    if state == "limited":
        back = account.get("resets_at")
        notes = [f"back {_when(datetime.fromisoformat(back))}" if back else "", blank]
        return "no", ", ".join(note for note in notes if note)
    return "?", blank


def _blank_reason(account: dict, has_usage: bool) -> str:
    if not has_usage:
        return "provider reports no usage"
    expires = account.get("token_expires_at")
    if expires and datetime.fromisoformat(expires).timestamp() <= time.time():
        return "token renewal failed: try again"
    return "usage read failed"


def snapshot_row(provider: str, snapshot) -> UsageRow:
    """The row of a provider with one login, from its ``/usage`` snapshot."""
    from hermes_cli.auth import read_credential_pool

    entries: list = list(read_credential_pool(provider) or [])
    entry = entries[0] if len(entries) == 1 else {}
    windows = [{"label": w.label, "used_percent": w.used_percent, "scope": w.scope,
                "resets_at": w.reset_at.isoformat() if w.reset_at else None} for w in snapshot.windows]
    session, week, week_reset = _windows(windows)
    full = [w.reset_at for w in snapshot.windows
            if w.scope == "account" and w.used_percent is not None and w.used_percent >= _FULL_PERCENT]
    if snapshot.unavailable_reason:
        usable, note = "?", snapshot.unavailable_reason
    elif full:
        usable, note = "no", f"back {_when(max(full))}" if all(full) else ""
    else:
        usable, note = "yes", "" if windows else next(iter(snapshot.details), "")
    return UsageRow(
        provider=provider, account=str(entry.get("label") or "default"), entry_id=str(entry.get("id") or ""),
        email=snapshot.email or "", plan=snapshot.plan or "", session_percent=session, week_percent=week,
        week_resets_at=week_reset, usable=usable, note=note,
    )


def _when(moment: Optional[datetime]) -> str:
    from hermes_time import safe_strftime

    if moment is None:
        return ""
    local = moment.astimezone()
    return f"{safe_strftime(local, '%b')} {local.day}, {safe_strftime(local, '%H:%M')}"


def _percent(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.0f}%"


def render_table(rows: list[UsageRow]) -> list[str]:
    """Aligned columns, usable accounts first, then by the soonest weekly reset; one summary line
    per provider."""
    order = {"yes": 0, "?": 1, "no": 2}
    providers = list(dict.fromkeys(r.provider for r in rows))
    rows = sorted(rows, key=lambda r: (providers.index(r.provider), order.get(r.usable, 1),
                                       r.week_resets_at.timestamp() if r.week_resets_at else float("inf")))
    cells = [_HEADER] + [
        (r.provider, r.account, r.entry_id, r.email, r.plan, _percent(r.session_percent),
         _percent(r.week_percent), _when(r.week_resets_at), r.usable, r.note)
        for r in rows
    ]
    widths = [max(len(row[i]) for row in cells) for i in range(len(_HEADER))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in cells]
    lines.append("")
    for provider in providers:
        mine = [r for r in rows if r.provider == provider]
        usable, unknown = sum(r.usable == "yes" for r in mine), sum(r.usable == "?" for r in mine)
        line = f"{provider}: {usable} of {len(mine)} account{'s' if len(mine) != 1 else ''} usable"
        lines.append(line + (f", {unknown} unknown" if unknown else ""))
    return lines
