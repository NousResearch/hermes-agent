"""Physical column history of the state.db tables the page-level salvage lane maps.

``_reconcile_columns()`` (hermes_state_schema.py) upgrades a live store with
``ALTER TABLE ADD COLUMN``, which APPENDS each newly declared column. A store
therefore keeps its columns in the order they were *added*, which diverges
from the order ``SCHEMA_SQL`` declares them whenever a later release inserts a
column mid-definition (#101409). SQLite never rewrites existing rows on ADD
COLUMN either, so a row written when the table had ``k`` columns carries
exactly the first ``k`` physical names.

A ``lost_and_found`` record carries no schema, so the salvage lane has to
recognise the physical layout that produced it. This module is the single
source of truth for that: the declared order of every table as first shipped
(``base``) and, per release that changed it, the edits made to the
declaration (``events``). Replaying the events reproduces every historical
declared order, and every physical layout a real store can have is some
prefix chain over those snapshots (created at snapshot ``i``, then each later
upgrade appending the columns it had not seen yet).

``tests/hermes_cli/test_session_schema_history.py`` asserts the replay ends
at the current ``SCHEMA_SQL``. When that test fails you added or moved a
column: append an event at the END, labelled with the next sequence number,
describing the edit (``("+", column, after)`` / ``("-", column)``) — do NOT
rewrite or reorder older events, real stores were shaped by them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence


Edit = tuple  # ("+", column, previous_column_or_None) | ("-", column)


@dataclass(frozen=True)
class _TableHistory:
    base: tuple[str, ...]
    # (label, edits) in replay order. Labels are "<seq> <commit time UTC>
    # <sha>" of the main commit that shipped the edit; the sequence number
    # is what the ordering test checks, so an event appended out of order
    # (or inserted mid-list "to keep dates sorted") fails loudly.
    events: tuple[tuple[str, tuple[Edit, ...]], ...]


def _apply(columns: list[str], edits: Sequence[Edit]) -> None:
    for edit in edits:
        if edit[0] == "-":
            columns.remove(edit[1])
            continue
        _, column, after = edit
        columns.insert(0 if after is None else columns.index(after) + 1, column)


def declared_snapshots(table: str) -> list[tuple[str, ...]]:
    """Every declared column order the table has shipped with, oldest first."""

    history = SCHEMA_HISTORY[table]
    columns = list(history.base)
    snapshots = [tuple(columns)]
    for _label, edits in history.events:
        _apply(columns, edits)
        snapshots.append(tuple(columns))
    return snapshots


def current_declared_columns(table: str) -> tuple[str, ...]:
    return declared_snapshots(table)[-1]


def reachable_physical_layouts(
    table: str,
    accept: Optional[Callable[[tuple[str, ...], int], bool]] = None,
) -> Iterator[tuple[str, ...]]:
    """Yield every physical layout an upgraded store of ``table`` can have.

    A store created at snapshot ``i`` and later opened by releases
    ``j1 < j2 < ...`` appends, at each step, the snapshot's columns it does
    not have yet. Columns declared and later removed stay physically present
    (``_reconcile_columns`` never drops), so they appear in layouts too.

    ``accept(layout, first_new_index)`` lets the caller prune: it is called
    for every new layout with the index where it starts to differ from its
    parent, and a False return stops extending that branch. The graph is
    large without pruning (~230k sessions layouts) but salvage always
    prunes against real cells, which keeps it in the low thousands.
    """

    snapshots = declared_snapshots(table)
    seen: dict[tuple[str, ...], int] = {}
    frontier: list[tuple[tuple[str, ...], int]] = []
    for index, snapshot in enumerate(snapshots):
        if snapshot in seen:
            continue
        if accept is not None and not accept(snapshot, 0):
            continue
        seen[snapshot] = index
        frontier.append((snapshot, index))
        yield snapshot
    while frontier:
        next_frontier: list[tuple[tuple[str, ...], int]] = []
        for layout, index in frontier:
            have = set(layout)
            for later in range(index + 1, len(snapshots)):
                added = tuple(c for c in snapshots[later] if c not in have)
                if not added:
                    continue
                candidate = layout + added
                if candidate in seen and seen[candidate] <= later:
                    continue
                if accept is not None and not accept(candidate, len(layout)):
                    continue
                seen[candidate] = later
                next_frontier.append((candidate, later))
                yield candidate
        frontier = next_frontier


SCHEMA_HISTORY: dict[str, _TableHistory] = {
    "sessions": _TableHistory(
        base=(
            'id', 'source', 'user_id', 'model', 'model_config',
            'system_prompt', 'parent_session_id', 'started_at', 'ended_at',
            'end_reason', 'message_count', 'tool_call_count', 'input_tokens',
            'output_tokens',
        ),
        events=(
        ('01 2026-03-08T23:32Z c5e8166c8b', (('+', 'title', 'output_tokens'),)),
        ('02 2026-03-17T10:44Z d417ba2a48', (
            ('+', 'cache_read_tokens', 'output_tokens'),
            ('+', 'cache_write_tokens', 'cache_read_tokens'),
            ('+', 'reasoning_tokens', 'cache_write_tokens'),
            ('+', 'billing_provider', 'reasoning_tokens'),
            ('+', 'billing_base_url', 'billing_provider'),
            ('+', 'billing_mode', 'billing_base_url'),
            ('+', 'estimated_cost_usd', 'billing_mode'),
            ('+', 'actual_cost_usd', 'estimated_cost_usd'),
            ('+', 'cost_status', 'actual_cost_usd'),
            ('+', 'cost_source', 'cost_status'),
            ('+', 'pricing_version', 'cost_source'),
        )),
        ('03 2026-04-22T12:51Z 5fb143169b', (('+', 'api_call_count', 'title'),)),
        ('04 2026-05-10T20:06Z 878611a79d', (
            ('+', 'handoff_pending', 'api_call_count'),
            ('+', 'handoff_platform', 'handoff_pending'),
        )),
        ('05 2026-05-10T20:06Z 00ce5f04d9', (
            ('-', 'handoff_pending'),
            ('+', 'handoff_state', 'api_call_count'),
            ('+', 'handoff_error', 'handoff_platform'),
        )),
        ('06 2026-05-31T22:46Z 51c68d4ab1', (('+', 'cwd', 'reasoning_tokens'),)),
        ('07 2026-06-01T08:22Z 3e59be0c41', (('+', 'rewind_count', 'handoff_error'),)),
        ('08 2026-06-02T01:41Z 85b65e29f0', (('+', 'archived', 'rewind_count'),)),
        ('09 2026-06-25T22:49Z ffa3d3c811', (
            ('+', 'git_branch', 'cwd'),
            ('+', 'git_repo_root', 'git_branch'),
        )),
        ('10 2026-06-28T22:10Z 86e64900b9', (
            ('+', 'session_key', 'user_id'),
            ('+', 'chat_id', 'session_key'),
            ('+', 'chat_type', 'chat_id'),
            ('+', 'thread_id', 'chat_type'),
        )),
        ('11 2026-06-30T08:06Z f2ccb2859f', (
            ('+', 'compression_failure_cooldown_until', 'handoff_error'),
            ('+', 'compression_failure_error', 'compression_failure_cooldown_until'),
        )),
        ('12 2026-07-05T21:01Z 747386ecfa', (
            ('+', 'display_name', 'thread_id'),
            ('+', 'origin_json', 'display_name'),
            ('+', 'expiry_finalized', 'origin_json'),
        )),
        ('13 2026-07-13T20:49Z af7dceaf77', (
            ('+', 'compression_fallback_streak', 'compression_failure_error'),
        )),
        ('14 2026-07-15T16:50Z e8b7ce8c19', (('+', 'profile_name', 'compression_fallback_streak'),)),
        ('15 2026-07-23T15:08Z ec5835ab8b', (
            ('+', 'compression_ineffective_count', 'compression_fallback_streak'),
        )),
        ('16 2026-07-24T16:15Z 951d606730', (('+', 'pinned', 'archived'),)),
        ('17 2026-07-27T19:14Z cfb206fe2e', (
            ('+', 'last_activity_at', 'title'),
            ('+', 'last_activity_description', 'last_activity_at'),
            ('+', 'last_activity_provenance', 'last_activity_description'),
        )),
        ('18 2026-07-27T19:22Z 5646fed97e', (
            ('-', 'last_activity_at'),
            ('-', 'last_activity_description'),
            ('-', 'last_activity_provenance'),
        )),
        ('19 2026-08-02T23:16Z c2088efe9e', (
            ('+', 'last_activity_at', 'title'),
            ('+', 'last_activity_description', 'last_activity_at'),
            ('+', 'last_activity_provenance', 'last_activity_description'),
        )),
        ('20 2026-08-03T15:07Z 7d066c3c56', (('+', 'system_prompt_hash', 'system_prompt'),)),
        ('21 2026-08-04T19:20Z d98287fe3c', (('+', 'last_read_at', 'pinned'),)),
        ('22 2026-08-08T22:17Z fe42097865', (('+', 'title_source', 'title'),)),
        ('23 2026-08-15T07:31Z fbaea9bddc', (('+', 'hidden', 'pinned'),)),
        ('24 2026-08-15T07:33Z e89532d97e', (('+', 'git_metadata_generation', 'git_repo_root'),)),
        ('25 2026-09-02T11:14Z 238b6c1ab9', (
            ('+', 'compression_recovery_deadline', 'compression_ineffective_count'),
        )),
        ('26 2026-09-02T14:22Z 8e4366d358', (('+', 'tool_names', 'last_read_at'),)),
        ),
    ),
    "messages": _TableHistory(
        base=(
            'id', 'session_id', 'role', 'content', 'tool_call_id',
            'tool_calls', 'tool_name', 'timestamp', 'token_count',
        ),
        events=(
        ('01 2026-02-21T08:05Z b33ed9176f', (('+', 'finish_reason', 'token_count'),)),
        ('02 2026-03-25T16:47Z 42fec19151', (
            ('+', 'reasoning', 'finish_reason'),
            ('+', 'reasoning_details', 'reasoning'),
            ('+', 'codex_reasoning_items', 'reasoning_details'),
        )),
        ('03 2026-04-22T11:31Z a7d78d3bfd', (('+', 'reasoning_content', 'reasoning'),)),
        ('04 2026-04-26T01:22Z 81e01f6ee9', (('+', 'codex_message_items', 'codex_reasoning_items'),)),
        ('05 2026-05-20T20:00Z 31a0100104', (('+', 'platform_message_id', 'codex_message_items'),)),
        ('06 2026-05-23T08:33Z 4a91e36495', (('+', 'observed', 'platform_message_id'),)),
        ('07 2026-06-01T08:22Z 3e59be0c41', (('+', 'active', 'observed'),)),
        ('08 2026-06-11T03:45Z aaccaada28', (('+', 'anthropic_content_blocks', 'reasoning_details'),)),
        ('09 2026-06-11T03:45Z efcbbde48c', (('-', 'anthropic_content_blocks'),)),
        ('10 2026-06-20T17:57Z 854d75723f', (('+', 'compacted', 'active'),)),
        ('11 2026-07-11T12:41Z a0a6cd80f5', (('+', 'effect_disposition', 'tool_name'),)),
        ('12 2026-07-19T02:55Z 7b3dcee928', (('+', 'api_content', 'compacted'),)),
        ('13 2026-07-23T18:46Z a4bc1ca502', (
            ('+', 'display_kind', 'api_content'),
            ('+', 'display_metadata', 'display_kind'),
        )),
        ('14 2026-08-25T10:55Z 1104ffe0b9', (('+', '_compressed_summary', 'observed'),)),
        ('15 2026-09-09T17:05Z 1c6683e8e0', (
            ('+', 'display_identity', 'display_metadata'),
            ('+', 'display_order', 'display_identity'),
        )),
        ),
    ),
    "session_model_usage": _TableHistory(
        base=(
            'session_id', 'model', 'billing_provider', 'billing_base_url',
            'api_call_count', 'input_tokens', 'output_tokens',
            'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens',
            'estimated_cost_usd', 'first_seen', 'last_seen',
        ),
        events=(
        ('01 2026-07-11T12:59Z 0d63c23f36', (
            ('+', 'billing_mode', 'billing_base_url'),
            ('+', 'actual_cost_usd', 'estimated_cost_usd'),
            ('+', 'cost_status', 'actual_cost_usd'),
            ('+', 'cost_source', 'cost_status'),
        )),
        ('02 2026-07-16T11:23Z eb6aa03609', (('+', 'task', 'billing_mode'),)),
        ),
    ),
}
