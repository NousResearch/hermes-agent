"""Contract cases for the unbounded listing window: ``SessionWindow(limit=None)`` returns every row from ``offset`` on, on every verb that takes a window.

Backend cases seed through public protocol operations only; the expected rows are the same verb's answer under a window wider than the seed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow

T0 = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC)
#: More rows than the default window of 20, so a bounded default would cut the answer.
ROWS = 25


def _seed(store) -> list[str]:
    ids = [f"cron_unb_{(T0 + timedelta(seconds=n)).strftime('%Y%m%d_%H%M%S')}" for n in range(ROWS)]
    for sid in ids:
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cron"))
    return ids


def test_none_limit_is_valid_and_negative_bounds_still_refused():
    assert SessionWindow(limit=None).limit is None
    assert SessionWindow(limit=None, offset=3).offset == 3
    with pytest.raises(ValueError):
        SessionWindow(limit=-1)
    with pytest.raises(ValueError):
        SessionWindow(limit=None, offset=-1)


def test_list_sessions_without_limit_returns_every_row(backend, scope_a):
    store = backend.open(scope_a)
    ids = _seed(store)
    query = SessionQuery(include_children=True, project_compression_tips=False)
    every = [e.session.session_id for e in store.sessions.list_sessions(query, SessionWindow(limit=1000))]
    # CONTRACT: no limit is every row the unbounded read returns, in the listing's order.
    assert sorted(every) == sorted(ids)
    assert [e.session.session_id for e in store.sessions.list_sessions(query, SessionWindow(limit=None))] == every
    # CONTRACT: the offset still applies with no limit (SQLite ``LIMIT -1 OFFSET n``).
    assert [e.session.session_id for e in store.sessions.list_sessions(query, SessionWindow(limit=None, offset=22))] == every[22:]
    assert store.sessions.list_sessions(query, SessionWindow(limit=None, offset=ROWS)) == ()


def test_search_sessions_without_limit_returns_every_row(backend, scope_a):
    store = backend.open(scope_a)
    ids = _seed(store)
    every = [e.session.session_id for e in store.sessions.search_sessions(window=SessionWindow(limit=1000))]
    assert sorted(every) == sorted(ids)
    assert [e.session.session_id for e in store.sessions.search_sessions(window=SessionWindow(limit=None))] == every
    assert [e.session.session_id for e in store.sessions.search_sessions(window=SessionWindow(limit=None, offset=20))] == every[20:]


def test_cron_job_runs_without_limit_return_every_run(backend, scope_a):
    store = backend.open(scope_a)
    ids = _seed(store)
    every = [e.entry.session.session_id for e in store.sessions.list_cron_job_runs("unb", SessionWindow(limit=1000))]
    assert every == list(reversed(ids))
    assert [e.entry.session.session_id for e in store.sessions.list_cron_job_runs("unb", SessionWindow(limit=None))] == every
    assert [e.entry.session.session_id for e in store.sessions.list_cron_job_runs("unb", SessionWindow(limit=None, offset=24))] == every[24:]
