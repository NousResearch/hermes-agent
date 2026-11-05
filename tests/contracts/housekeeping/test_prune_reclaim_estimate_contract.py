"""Contract cases for ``SessionMaintenance.prune_reclaim_estimate`` (record ``PruneReclaimEstimate``).

The reclaim estimate is one whole-lineage prune selection counted against every row, then the stored bytes of the selected rows' messages and of every message row, both read only when something is selected. Byte values are backend measures, so the cases assert their relationships, never a number. Sessions enter through ``import_sessions`` (export dicts) and the sessions and messages repositories.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.maintenance import PruneCriteria, PruneReclaimEstimate
from tests.contracts._staged import operation

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=UTC).timestamp()
_NOTHING = PruneCriteria(last_active_before=_BASE - 1)
_EVERYTHING = PruneCriteria(last_active_before=_FAR_FUTURE)


def _msg(role: str, content: str, offset: float, **fields) -> dict:
    return {"role": role, "content": content, "timestamp": _BASE + offset, **fields}


def _payload(sid: str, messages: list, *, source: str = "cli", parent: str | None = None, ended_at: float | None = _BASE + 100, end_reason: str | None = "user_exit") -> dict:
    return {"id": sid, "source": source, "started_at": _BASE, "ended_at": ended_at, "end_reason": end_reason, "parent_session_id": parent, "messages": messages}


def _seed(store) -> None:
    """Two old ended sessions (the cron one also holds an inactive row), an old compression ancestor whose continuation is still open, and one recent ended session: five rows."""
    two = [_msg("user", "hello there", 0), _msg("assistant", "general kenobi", 1)]
    result = store.maintenance.import_sessions([
        _payload("old-cli", two),
        _payload("old-cron", [*two, _msg("user", "a rewound turn", 2, active=0)], source="cron"),
        _payload("ancestor", [_msg("user", "first segment", 0)], end_reason="compression"),
        _payload("continuation", [_msg("user", "second segment", 5)], parent="ancestor", ended_at=None, end_reason=None),
    ])
    assert result.ok and result.imported == 4 and result.detached == 0
    store.sessions.create(NewSession(session_id=SessionId("recent"), source="cli"))
    store.sessions.end(SessionId("recent"), "user_exit")
    store.messages.append(SessionId("recent"), NewMessage(role="user", content="fresh"))


def test_nothing_selected_reads_no_bytes(backend, scope_a):
    store = backend.open(scope_a)
    estimate = operation(store, "maintenance", "prune_reclaim_estimate")
    _seed(store)
    # CONTRACT: no selected row gives 0 bytes and an unmeasured total.
    assert estimate(90, criteria=_NOTHING) == PruneReclaimEstimate(prunable=0, total=5, prunable_bytes=0, total_bytes=None)


def test_retention_keeps_a_compression_ancestor_while_its_continuation_stays(backend, scope_a):
    store = backend.open(scope_a)
    estimate = operation(store, "maintenance", "prune_reclaim_estimate")
    _seed(store)
    got = estimate(90)
    # CONTRACT: whole lineages: the ancestor of an open continuation is not selected.
    assert (got.prunable, got.total) == (2, 5)
    assert got.prunable == len(store.maintenance.list_prune_candidates(90, whole_lineages=True))
    assert 0 < got.prunable_bytes <= got.total_bytes
    # CONTRACT: without whole lineages the ancestor is selected on its own age.
    assert estimate(90, whole_lineages=False).prunable == 3
    # CONTRACT: total_bytes is None when not measured; the rest is unchanged.
    assert estimate(90, measure_total_bytes=False) == PruneReclaimEstimate(prunable=2, total=5, prunable_bytes=got.prunable_bytes, total_bytes=None)


def test_inactive_rows_count_and_every_row_selected_measures_the_whole_store(backend, scope_a):
    store = backend.open(scope_a)
    estimate = operation(store, "maintenance", "prune_reclaim_estimate")
    _seed(store)
    cli, cron = estimate(90, "cli"), estimate(90, "cron")
    assert (cli.prunable, cron.prunable) == (1, 1)
    # CONTRACT: the size covers inactive rows too (a prune deletes the soft-archived rows with the session).
    assert cron.prunable_bytes > cli.prunable_bytes > 0
    store.sessions.end(SessionId("continuation"), "user_exit")
    every = estimate(None, criteria=_EVERYTHING)
    # CONTRACT: when every row is selected, the selected rows' bytes are every message row's bytes.
    assert (every.prunable, every.total) == (5, 5)
    assert every.prunable_bytes == every.total_bytes > cron.prunable_bytes


def test_a_negative_retention_is_refused(backend, scope_a):
    store = backend.open(scope_a)
    estimate = operation(store, "maintenance", "prune_reclaim_estimate")
    _seed(store)
    # CONTRACT: the prune selection refuses a negative age.
    with pytest.raises(ValueError):
        estimate(-1)
