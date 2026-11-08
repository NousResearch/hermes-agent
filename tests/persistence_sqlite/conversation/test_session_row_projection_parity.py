"""The SQLite session row projections equal the legacy ``SessionDB`` row dicts read from the same store file: ``get_session``, ``list_sessions_rich`` (plain, compact, by last activity, with children) and ``list_cron_job_runs``, key order included."""

from __future__ import annotations

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionLineageKind, SessionOrder, SessionQuery, SessionWindow
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]


def _items(rows):
    return [list(row.items()) for row in rows]


def _seed(store):
    s = store.sessions
    s.create(NewSession(session_id=SessionId("pp-a"), source="telegram", model="m", model_config={"a": 1, "b": [2]},
                        system_prompt="sp", origin={"platform": "telegram", "chat": 7}, cwd="/w"))
    s.update_tool_pin(SessionId("pp-a"), ["terminal", "web"])
    s.set_title(SessionId("pp-a"), "Alpha")
    s.set_pinned(SessionId("pp-a"), True)
    store.messages.append(SessionId("pp-a"), NewMessage(role="user", content="hello there"))
    s.set_read(SessionId("pp-a"))
    s.create(NewSession(session_id=SessionId("pp-b"), source="cli", parent_session_id=SessionId("pp-a"),
                        lineage_kind=SessionLineageKind.BRANCH))
    s.create(NewSession(session_id=SessionId("pp-root"), source="cli", system_prompt="root"))
    store.messages.append(SessionId("pp-root"), NewMessage(role="user", content="start"))
    s.end(SessionId("pp-root"), "compression")
    s.create(NewSession(session_id=SessionId("pp-tip"), source="cli", parent_session_id=SessionId("pp-root"), system_prompt="tip"))
    store.messages.append(SessionId("pp-tip"), NewMessage(role="user", content="go on"))
    s.create(NewSession(session_id=SessionId("cron_pj_20260923_120000"), source="cron"))
    s.create(NewSession(session_id=SessionId("cron_pj_20260923_120001"), source="cron", model_config={"_reset_from": "x"}))
    s.set_archived(SessionId("cron_pj_20260923_120001"), True)


def test_projections_equal_the_legacy_dicts(backend, scope_a, legacy_store):
    from hermes_state import SessionDB

    store = backend.open(scope_a)
    _seed(store)
    legacy = SessionDB(db_path=store_for(scope_a))
    try:
        for sid in ("pp-a", "pp-b", "pp-root", "pp-tip", "cron_pj_20260923_120001"):
            assert list(store.sessions.get(SessionId(sid)).as_session_row().items()) == list(legacy.get_session(sid).items())
        for query, kwargs in (
            (SessionQuery(), {}),
            (SessionQuery(order=SessionOrder.LAST_ACTIVE), {"order_by_last_active": True}),
            (SessionQuery(include_children=True), {"include_children": True}),
            (SessionQuery(include_archived=True, include_pinned=True), {"include_archived": True, "include_pinned": True}),
        ):
            for compact in (False, True):
                served = store.sessions.list_sessions(query, SessionWindow(limit=50), compact_rows=compact)
                expected = legacy.list_sessions_rich(limit=50, compact_rows=compact, **kwargs)
                assert _items(e.as_list_row() for e in served) == _items(expected), (query, compact)
        runs = store.sessions.list_cron_job_runs("pj", SessionWindow(limit=10))
        assert _items(r.as_cron_row() for r in runs) == _items(legacy.list_cron_job_runs("pj", limit=10))
    finally:
        legacy.close()
