"""Regression: every session-switch surface must rebind the Telegram DM topic row.

Topic mode persists a ``(chat_id, thread_id) -> session_id`` row in
``telegram_dm_topic_bindings`` so reopening a topic in a fresh process resumes
the right Hermes session, and so the retire/archive helpers resolve the correct
topic to delete. ``/new`` already rewrites this row after resetting the session
(``_record_telegram_topic_binding``; regression covered by
``test_new_inside_telegram_topic_rewrites_binding_to_new_session``).

But four OTHER surfaces move routing via ``SessionStore.switch_session`` WITHOUT
the paired rebind, so the binding goes stale (keeps pointing at the old session)
or is never written:

1. ``/resume``  — ``_handle_resume_command`` (gateway/slash_commands_session.py)
2. ``/branch``  — ``_handle_branch_command`` (gateway/slash_commands_session.py)
3. handoff / CLI-injection worker — ``_process_handoff`` (gateway/run_startup.py)
4. async-delegation completion — ``_resolve_async_delegation_session``
   (gateway/run_notifications.py)

A stale binding is the retire-safety hazard that drove this fix: the archiver
resolves its delete target from the binding row, so a wrong row = deleting the
wrong Telegram topic.

Each test drives the surface against a REAL SessionStore + SessionDB (SQLite in
tmp_path, topic mode enabled) and asserts the binding row points at the session
the surface switched to.
"""

from __future__ import annotations

from unittest import mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import AsyncSessionDB

CHAT_ID = "208214988"
THREAD_ID = "17585"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Real SessionStore backed by a real SessionDB with topic mode enabled."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db.enable_telegram_topic_mode(chat_id=CHAT_ID, user_id=CHAT_ID)
    return store


def _make_source(thread_id: str | None = THREAD_ID) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=CHAT_ID,
        chat_id=CHAT_ID,
        chat_type="dm",
        thread_id=thread_id,
    )


def _make_event(text: str, thread_id: str | None = THREAD_ID) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(thread_id), message_id="m1")


def _binding_session_id(store: SessionStore) -> str | None:
    row = store._db.get_telegram_topic_binding(chat_id=CHAT_ID, thread_id=THREAD_ID)
    return row["session_id"] if row else None


def _make_runner(store: SessionStore):
    """Minimal GatewayRunner wired to a REAL session_store/session_db."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = GatewayConfig()
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    # Collaborators the switch handlers touch but which are irrelevant here.
    runner._release_running_agent_state = mock.MagicMock()
    runner._evict_cached_agent = mock.MagicMock()
    runner._clear_conversation_scope = mock.MagicMock()
    runner._clear_session_boundary_security_state = mock.MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner._resume_caller_is_admin = lambda _source: True
    return runner


def _seed_topic_session(store: SessionStore, source: SessionSource) -> str:
    """Create the initial topic session + its binding (the /new-established state)."""
    entry = store.get_or_create_session(source)
    session_key = build_session_key(source)
    store._db.bind_telegram_topic(
        chat_id=CHAT_ID, thread_id=THREAD_ID, user_id=CHAT_ID,
        session_key=session_key, session_id=entry.session_id,
    )
    return entry.session_id


# ---------------------------------------------------------------- /resume

@pytest.mark.asyncio
async def test_resume_rebinds_topic_to_resumed_session(store):
    """/resume inside a topic must rebind the row to the resumed session."""
    source = _make_source()
    original_id = _seed_topic_session(store, source)

    # A second, older session in the same topic that we will /resume back into.
    store._db.create_session(session_id="older-session", source="telegram",
                             user_id=CHAT_ID, chat_id=CHAT_ID, chat_type="dm",
                             thread_id=THREAD_ID, session_key=build_session_key(source))
    store._db.set_session_title("older-session", "Older")

    runner = _make_runner(store)
    reply = await runner._handle_resume_command(_make_event("/resume Older"))

    assert _binding_session_id(store) == "older-session", (
        f"/resume left the topic binding stale (reply={reply!r}); "
        f"still points at {_binding_session_id(store)}, expected older-session"
    )
    assert original_id != "older-session"


# ---------------------------------------------------------------- /branch

@pytest.mark.asyncio
async def test_branch_rebinds_topic_to_branched_session(store):
    """/branch inside a topic must rebind the row to the new branched session."""
    source = _make_source()
    parent_id = _seed_topic_session(store, source)
    store._db.append_message(parent_id, role="user", content="hello")
    store._db.append_message(parent_id, role="assistant", content="world")

    runner = _make_runner(store)
    reply = await runner._handle_branch_command(_make_event("/branch"))

    bound = _binding_session_id(store)
    assert bound is not None and bound != parent_id, (
        f"/branch left the topic binding on the parent (reply={reply!r}); "
        f"binding={bound}, parent={parent_id}"
    )
    # The bound session is the live route after the branch.
    live = store.get_or_create_session(source).session_id
    assert bound == live, f"binding {bound} != live route {live}"


# ---------------------------------------------------------------- handoff worker

@pytest.mark.asyncio
async def test_handoff_rebinds_topic_to_cli_session(store):
    """The CLI-handoff worker switches routing to the CLI session; it must rebind."""
    source = _make_source()
    _seed_topic_session(store, source)

    # The incoming CLI session that the handoff injects into this topic.
    session_key = build_session_key(source)
    store._db.create_session(session_id="cli-session", source="cli", user_id=CHAT_ID,
                             chat_id=CHAT_ID, chat_type="dm", thread_id=THREAD_ID,
                             session_key=session_key)

    runner = _make_runner(store)

    # _process_handoff resolves a destination, switches, then dispatches a synthetic
    # turn. We stub the destination + the synthetic turn/send so the test isolates the
    # switch+rebind. dest.source carries the topic SessionSource.
    dest = mock.MagicMock()
    dest.source = source
    dest.platform_name = "telegram"
    dest.home = mock.MagicMock(chat_id=CHAT_ID)
    dest.effective_thread_id = THREAD_ID
    runner._handoff_resolve_destination = mock.AsyncMock(return_value=dest)
    runner._handoff_session_key = lambda _dest, _profile: session_key
    runner._handle_message = mock.AsyncMock(return_value="")  # synthetic turn no-op

    await runner._process_handoff({"id": "cli-session", "title": "CLI"})

    assert _binding_session_id(store) == "cli-session", (
        f"handoff worker left the topic binding stale; "
        f"binding={_binding_session_id(store)}, expected cli-session"
    )


# ---------------------------------------------------------------- async delegation

@pytest.mark.asyncio
async def test_async_delegation_rebinds_topic_to_owning_session(store):
    """Async-delegation completion retargets routing to the owning session; rebind."""
    source = _make_source()
    original_id = _seed_topic_session(store, source)
    session_key = build_session_key(source)

    # The delegation's owning ("pinned") session, distinct from the current route.
    store._db.create_session(session_id="owning-session", source="telegram", user_id=CHAT_ID,
                             chat_id=CHAT_ID, chat_type="dm", thread_id=THREAD_ID,
                             session_key=session_key)

    runner = _make_runner(store)
    current_entry = store.get_or_create_session(source)
    assert current_entry.session_id == original_id

    resolved = await runner._resolve_async_delegation_session(current_entry, "owning-session")
    assert resolved is not None and resolved.session_id == "owning-session"

    assert _binding_session_id(store) == "owning-session", (
        f"async-delegation left the topic binding stale; "
        f"binding={_binding_session_id(store)}, expected owning-session"
    )
