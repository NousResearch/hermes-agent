"""Reported model usage, read-only.

**This is observation, not control.** Nothing here enforces anything, and nothing here
can: the runtime's LLM-call hooks discard their return values, so no plugin can veto a
model call. Token and cost figures are reportable and never a ceiling.

Two caveats travel with every figure and are carried in :class:`UsageSummary` rather than
left to documentation:

*Lagging* — the runtime writes usage through a coalescing background thread, so a reading
taken now can trail the true figure.

*Estimated* — ``estimated_cost_usd`` is the runtime's own arithmetic, not an invoice.
Reconcile against the provider's billing before anyone is charged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from nova.runtime.base import ModelUsage, UsageSummary

#: The runtime's session store inside a profile.
STATE_DB_NAME = "state.db"

#: Named explicitly rather than SELECT * so a schema change upstream surfaces as a clear
#: error here instead of silently reshaping the numbers.
_COLUMNS = (
    "model, billing_provider, api_call_count, input_tokens, output_tokens, "
    "cache_read_tokens, cache_write_tokens, reasoning_tokens, estimated_cost_usd, "
    "actual_cost_usd, cost_status"
)


def state_db_path(profile_dir: Path) -> Path:
    return profile_dir / STATE_DB_NAME


def read_usage(profile_dir: Path, agent_id: str) -> UsageSummary:
    """Aggregate reported usage for one agent across its sessions.

    An absent store is normal on an agent that has not run yet and is reported as such
    rather than raised.
    """
    path = state_db_path(profile_dir)
    if not path.is_file():
        return UsageSummary(
            agent_id=agent_id,
            available=False,
            detail="this agent has not run yet, so the runtime has recorded no usage",
        )

    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT {_COLUMNS} FROM session_model_usage"  # noqa: S608 — fixed column list
        ).fetchall()
    except sqlite3.Error as exc:
        return UsageSummary(
            agent_id=agent_id, available=False, detail=f"usage store unreadable: {exc}"
        )
    finally:
        if connection is not None:
            connection.close()

    # Fold sessions together per (model, provider): an operator cares what a model cost
    # this agent, not how many sessions it took.
    totals: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"] or ""), str(row["billing_provider"] or ""))
        bucket = totals.setdefault(
            key,
            {
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost_usd": 0.0,
                "actual_cost_usd": 0.0,
                "cost_status": "",
            },
        )
        bucket["api_calls"] += int(row["api_call_count"] or 0)
        bucket["input_tokens"] += int(row["input_tokens"] or 0)
        bucket["output_tokens"] += int(row["output_tokens"] or 0)
        bucket["cache_read_tokens"] += int(row["cache_read_tokens"] or 0)
        bucket["cache_write_tokens"] += int(row["cache_write_tokens"] or 0)
        bucket["reasoning_tokens"] += int(row["reasoning_tokens"] or 0)
        bucket["estimated_cost_usd"] += float(row["estimated_cost_usd"] or 0.0)
        bucket["actual_cost_usd"] += float(row["actual_cost_usd"] or 0.0)
        bucket["cost_status"] = str(row["cost_status"] or "") or bucket["cost_status"]

    models = tuple(
        ModelUsage(model=model, provider=provider, **bucket)
        for (model, provider), bucket in sorted(totals.items())
    )
    return UsageSummary(agent_id=agent_id, available=True, models=models)


def month_to_date_spend(profile_dir: Path, since: float) -> Optional[float]:
    """USD spent since ``since`` (Unix time), as the budget check counts it. None if unreadable.

    The same arithmetic as the policy plugin's budget check (actual cost where the runtime
    has one, its estimate otherwise; a session counts in the month it was last active), so
    the Usage screen and the stop agree about how close an agent is.
    """
    path = state_db_path(profile_dir)
    if not path.is_file():
        return 0.0
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        try:
            row = connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN actual_cost_usd > 0 THEN actual_cost_usd "
                "ELSE estimated_cost_usd END), 0) FROM session_model_usage "
                "WHERE COALESCE(last_seen, first_seen, 0) >= ?",
                (since,),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    return float(row[0] or 0.0)
