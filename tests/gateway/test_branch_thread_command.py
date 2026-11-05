"""/branch on thread-capable platforms opens a sibling thread and keeps the origin (#66023).

Drives the REAL ``_handle_branch_command`` against a REAL SessionStore and its real store (SQLite in
the test HERMES_HOME); only the platform adapter is a fake whose ``create_handoff_thread`` returns a fixed id.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from tests.fixtures.storage_read import session_row, transcript_rows
from tests.fixtures.storage_seed import seed_messages


class _ThreadAdapter:
    """Discord-shaped fake: the only thing /branch needs from an adapter."""

    def __init__(self, thread_id="777000", fail=False):
        self.thread_id, self.fail, self.calls = thread_id, fail, []

    async def create_handoff_thread(self, parent_chat_id, name):
        self.calls.append((parent_chat_id, name))
        return None if self.fail else self.thread_id


@pytest.fixture()
def store(tmp_path):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    yield store
    store.close_all_db_handles()


def _runner(store, adapter):
    from gateway.run import GatewayRunner

    class _Runner(GatewayRunner):
        persistence = None  # the session store's storage interface handle, set below

    runner = object.__new__(_Runner)
    runner.adapters = {Platform.DISCORD: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner.persistence = store.persistence
    runner._pending_skills_reload_notes = {}
    return runner


def _discord_channel_source():
    return SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="group", user_id="u1",
                         user_name="ann", scope_id="g9")


def _seed(store, source):
    entry = store.get_or_create_session(source)
    seed_messages(store.persistence, entry.session_id, {"role": "user", "content": "hello"},
                  {"role": "assistant", "content": "world"})
    return entry


@pytest.mark.asyncio
async def test_plain_branch_binds_new_thread_and_keeps_origin(store):
    source = _discord_channel_source()
    parent = _seed(store, source)
    adapter = _ThreadAdapter()
    runner = _runner(store, adapter)

    reply = await runner._handle_branch_command(MessageEvent(text="/branch side quest", source=source))

    assert adapter.calls == [("123", "side quest")]
    # The chat the command came from is still on the original session.
    assert store.get_or_create_session(source).session_id == parent.session_id
    # The new thread (Discord keys it on its own id) is on the clone, with the routing columns of
    # the THREAD, so a restart routes the next in-thread message to the branch.
    thread_source = SessionSource(platform=Platform.DISCORD, chat_id="777000", chat_type="thread",
                                  thread_id="777000", parent_chat_id="123", user_id="u1", scope_id="g9")
    branch = store.get_or_create_session(thread_source)
    assert branch.session_id != parent.session_id
    row = session_row(store.persistence, branch.session_id)
    assert (row["parent_session_id"], row["chat_id"], row["thread_id"], row["chat_type"]) == (
        parent.session_id, "777000", "777000", "thread")
    assert [m["content"] for m in transcript_rows(store.persistence, branch.session_id)] == ["hello", "world"]
    assert "<#777000>" in reply and parent.session_id in reply


@pytest.mark.asyncio
@pytest.mark.parametrize("text, adapter", [
    ("/branch --here side quest", _ThreadAdapter()),   # explicit opt-out
    ("/branch side quest", _ThreadAdapter(fail=True)),  # platform could not open a thread
    ("/branch side quest", None),                       # no adapter for the platform
])
async def test_here_or_no_thread_branches_in_place(store, text, adapter):
    source = _discord_channel_source()
    parent = _seed(store, source)
    runner = _runner(store, adapter)

    reply = await runner._handle_branch_command(MessageEvent(text=text, source=source))

    current = store.get_or_create_session(source)
    assert current.session_id != parent.session_id
    row = session_row(store.persistence, current.session_id)
    assert row["parent_session_id"] == parent.session_id
    assert store.persistence.sessions.get_title(current.session_id) == "side quest"
    if adapter is not None:
        # ``--here`` never even asks the platform for a thread.
        assert adapter.calls == ([] if "--here" in text else [("123", "side quest")])
    assert "side quest" in reply
