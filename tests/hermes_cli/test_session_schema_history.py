"""``hermes_cli.session_schema_history`` must track SCHEMA_SQL.

The page-level salvage lane infers a salvaged store's physical column order
from this history. If a column is added to SCHEMA_SQL without an event here,
the newest rows of every upgraded store map to nothing (their width matches
no chain state) and silently take the positional fallback.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import session_schema_history as history
from hermes_state_common import SCHEMA_SQL


def _declared_now(table: str) -> tuple[str, ...]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA_SQL)
        return tuple(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")'))
    finally:
        conn.close()


@pytest.mark.parametrize("table", sorted(history.SCHEMA_HISTORY))
def test_replayed_history_ends_at_current_schema(table: str) -> None:
    """Replaying every recorded edit must reproduce today's declared order.

    Failing here means SCHEMA_SQL changed for ``table``: append an event to
    ``SCHEMA_HISTORY[table].events`` describing the edit (never rewrite
    older events — real stores were shaped by them).
    """

    labels = [int(label.split()[0]) for label, _ in history.SCHEMA_HISTORY[table].events]
    assert labels == list(range(1, len(labels) + 1)), (
        f"SCHEMA_HISTORY[{table!r}].events is out of order: append new events at the END with the next "
        f"sequence number, never insert mid-list (got {labels})"
    )
    replayed = history.current_declared_columns(table)
    declared = _declared_now(table)
    missing = [c for c in declared if c not in replayed]
    extra = [c for c in replayed if c not in declared]
    hint = f"SCHEMA_HISTORY[{table!r}].events in hermes_cli/session_schema_history.py"
    assert not missing, (
        f"SCHEMA_SQL declares {missing} for {table} but the replayed history does not: "
        f"append ('+', <column>, <declared predecessor>) events to {hint}"
    )
    assert not extra, (
        f"the replayed history declares {extra} for {table} but SCHEMA_SQL no longer does: "
        f"append ('-', <column>) events to {hint}"
    )
    assert replayed == declared, (
        f"{table} column order drifted at index "
        f"{next(i for i, (a, b) in enumerate(zip(replayed, declared)) if a != b)}: "
        f"replayed {replayed} vs SCHEMA_SQL {declared} — record the move as ('-', c) + ('+', c, after) in {hint}"
    )
