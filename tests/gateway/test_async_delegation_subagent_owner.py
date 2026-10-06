"""Async-delegation completions pinned to a delegate_task child session (#92611, #92859, #131578, #132360).

A running subagent's own background-process notice carries the CHILD session id. The gateway must
route it to the chat that owns the child instead of re-pointing the chat into the child and ending
the real chat session as a user boundary (which made every later delegation result for that chat
"target a permanently-gone session" and be dropped).

Real SessionDB + SessionStore + gateway resolver/classifier; only the runner's unrelated state is
stubbed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_state import AsyncSessionDB, SessionDB


@pytest.fixture()
def env(tmp_path):
    from gateway.run import GatewayRunner

    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm",
                           user_id="12345", user_name="tester")
    entry = store.get_or_create_session(source)

    runner = object.__new__(GatewayRunner)
    runner._session_db = AsyncSessionDB(db)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._current_session_run_generation = lambda session_key: 0
    runner._is_session_run_current = lambda session_key, generation: True

    def child(child_id, parent_id):
        db.create_session(child_id, source="subagent", parent_session_id=parent_id,
                          model_config={"_delegate_from": parent_id})
        return child_id

    yield SimpleNamespace(
        store=store, db=db, runner=runner, key=entry.session_key, chat=entry.session_id, child=child,
        route=lambda: store._entries[entry.session_key].session_id,
    )
    db.close()


def _serve_rows(env, fake_rows):
    """Serve ``fake_rows`` from get_session (rows the real schema's foreign key refuses)."""
    real = env.runner._session_db

    class _Overlay:
        def __getattr__(self, name):
            return getattr(real, name)

        async def get_session(self, sid):
            if sid in fake_rows:
                return fake_rows[sid]
            return await real.get_session(sid)

    env.runner._session_db = _Overlay()


async def _notice(env, pinned):
    """What gateway/run_turn.py does with an internal event pinned to ``pinned``."""
    return await env.runner._resolve_async_delegation_session(env.store._entries[env.key], pinned)


@pytest.mark.asyncio
async def test_child_notice_keeps_route_and_chat_session_open(env):
    child = env.child("sess_child", env.chat)
    resolved = await _notice(env, child)
    assert resolved is not None and resolved.session_id == env.chat
    assert env.route() == env.chat
    assert env.db.get_session(env.chat)["ended_at"] is None


@pytest.mark.asyncio
async def test_later_results_for_the_chat_are_delivered_after_child_notices(env):
    first = env.child("sess_child_a", env.chat)
    second = env.child("sess_child_b", env.chat)
    await _notice(env, first)
    await _notice(env, second)
    for _ in range(2):
        assert await env.runner._classify_completion_target(env.chat) == "deliver"
        resolved = await _notice(env, env.chat)
        assert resolved is not None and resolved.session_id == env.chat
    assert env.route() == env.chat


@pytest.mark.asyncio
async def test_nested_grandchild_notice_maps_to_the_chat(env):
    child = env.child("sess_child", env.chat)
    grandchild = env.child("sess_grandchild", child)
    assert await env.runner._classify_completion_target(grandchild) == "deliver"
    resolved = await _notice(env, grandchild)
    assert resolved is not None and resolved.session_id == env.chat
    assert env.db.get_session(env.chat)["ended_at"] is None


@pytest.mark.asyncio
async def test_child_of_a_user_closed_chat_is_terminal_and_leaves_route(env):
    child = env.child("sess_child", env.chat)
    env.db.end_session(env.chat, end_reason="session_reset")
    assert await env.runner._classify_completion_target(child) == "terminal"
    route_before = env.route()
    assert await _notice(env, child) is None
    assert env.route() == route_before


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [
    {"sess_loop_a": {"id": "sess_loop_a", "source": "subagent", "parent_session_id": "sess_loop_b"},
     "sess_loop_b": {"id": "sess_loop_b", "source": "subagent", "parent_session_id": "sess_loop_a"}},
    {"sess_loop_a": {"id": "sess_loop_a", "source": "subagent", "parent_session_id": "sess_missing"}},
    {"sess_loop_a": {"id": "sess_loop_a", "source": "subagent", "parent_session_id": ""}},
], ids=["cycle", "missing-parent", "no-parent"])
async def test_unverifiable_subagent_lineage_fails_closed(env, rows):
    _serve_rows(env, rows)
    assert await env.runner._classify_completion_target("sess_loop_a") == "terminal"
    assert await _notice(env, "sess_loop_a") is None
    assert env.route() == env.chat


@pytest.mark.asyncio
async def test_repin_to_another_live_session_does_not_strand_the_old_one(env):
    """A legitimate repin still moves the route, but results pinned to the session it left are
    retargeted to the chat's current session rather than dropped."""
    env.db.create_session("sess_other", source="telegram")
    resolved = await _notice(env, "sess_other")
    assert resolved is not None and resolved.session_id == "sess_other"
    assert env.db.get_session(env.chat)["ended_at"] is not None
    assert await env.runner._classify_completion_target(env.chat) == "deliver"
    resolved = await _notice(env, env.chat)
    assert resolved is not None and resolved.session_id == "sess_other"


@pytest.mark.asyncio
async def test_resume_and_new_still_drop_late_results(env):
    """/resume and /new remain user boundaries for results pinned to the session they closed."""
    env.db.create_session("sess_resume_target", source="telegram")
    env.db.end_session("sess_resume_target", end_reason="user_exit")
    assert env.store.switch_session(env.key, "sess_resume_target") is not None
    assert await env.runner._classify_completion_target(env.chat) == "terminal"
    env.db.create_session("sess_reset", source="telegram")
    env.db.end_session("sess_reset", end_reason="session_reset")
    assert await env.runner._classify_completion_target("sess_reset") == "terminal"


@pytest.mark.asyncio
async def test_chat_row_carrying_a_delegate_marker_is_still_the_owner(env):
    """A chat-sourced row may carry ``_delegate_from`` (a former child that became the route). It
    is a live chat session: results pinned to it deliver there, never to the marker's session."""
    env.db.create_session("sess_old_chat", source="telegram")
    env.db.end_session("sess_old_chat", end_reason="session_switch")
    env.db.create_session("sess_marked_chat", source="telegram",
                          model_config={"_delegate_from": "sess_old_chat"})
    env.store.switch_session(env.key, "sess_marked_chat")
    assert await env.runner._classify_completion_target("sess_marked_chat") == "deliver"
    resolved = await _notice(env, "sess_marked_chat")
    assert resolved is not None and resolved.session_id == "sess_marked_chat"
