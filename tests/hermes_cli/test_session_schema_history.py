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


@pytest.mark.parametrize("table", sorted(history.SCHEMA_HISTORY))
def test_snapshots_are_distinct_and_replay_is_well_formed(table: str) -> None:
    snapshots = history.declared_snapshots(table)
    assert len(snapshots) == len(history.SCHEMA_HISTORY[table].events) + 1
    for snapshot in snapshots:
        assert len(set(snapshot)) == len(snapshot), "duplicate column in a snapshot"
    # Snapshots may repeat (a column was declared, reverted, then declared
    # again) but consecutive ones must differ or the event was a no-op.
    for before, after in zip(snapshots, snapshots[1:]):
        assert before != after, "an event changed nothing"
    sequence = [int(label.split()[0]) for label, _ in history.SCHEMA_HISTORY[table].events]
    assert sequence == list(range(1, len(sequence) + 1)), (
        "event labels must be numbered 01, 02, ... in replay order: append new "
        "events at the END with the next number, never insert mid-list"
    )


def test_reachable_layouts_include_the_upgraded_store_from_101409() -> None:
    """The reporter's physical order (created ~May 2026, upgraded through
    v27) is a chain over the recorded snapshots."""

    reported = (
        "id", "source", "user_id", "model", "model_config", "system_prompt",
        "parent_session_id", "started_at", "ended_at", "end_reason",
        "message_count", "tool_call_count", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        "billing_provider", "billing_base_url", "billing_mode",
        "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source",
        "pricing_version", "title", "api_call_count", "handoff_state",
        "handoff_platform", "handoff_error", "cwd", "rewind_count", "archived",
        "session_key", "chat_id", "chat_type", "thread_id", "git_branch",
        "git_repo_root", "compression_failure_cooldown_until",
        "compression_failure_error", "display_name", "origin_json",
        "expiry_finalized", "compression_fallback_streak", "profile_name",
        "compression_ineffective_count", "pinned", "system_prompt_hash",
        "last_activity_at", "last_activity_description",
        "last_activity_provenance", "git_metadata_generation", "title_source",
        "hidden", "last_read_at",
    )

    def accept(layout: tuple[str, ...], first_new: int) -> bool:
        return layout[first_new:len(reported)] == reported[first_new:len(layout)]

    assert any(
        layout[: len(reported)] == reported
        for layout in history.reachable_physical_layouts("sessions", accept)
    )


def test_pruning_callback_stops_branches() -> None:
    """``accept`` returning False must prune, not just filter the output."""

    offered: list[tuple[tuple[str, ...], int]] = []

    def accept(layout: tuple[str, ...], first_new: int) -> bool:
        offered.append((layout, first_new))
        return len(layout) <= 10

    layouts = list(history.reachable_physical_layouts("messages", accept))
    assert layouts and all(len(layout) <= 10 for layout in layouts)
    # Every extension offered must grow from an ACCEPTED parent: its prefix
    # up to first_new is a layout that passed. Seeds (first_new == 0) are
    # always offered.
    accepted = set(layouts)
    for layout, first_new in offered:
        if first_new:
            assert layout[:first_new] in accepted, "a rejected layout was extended"
