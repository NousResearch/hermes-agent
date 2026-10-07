"""/branch on thread-capable platforms opens a sibling thread and keeps the origin (#66023).

Drives the REAL ``_handle_branch_command`` against a REAL SessionStore + SessionDB (SQLite in
tmp_path); only the platform adapter is a fake whose ``create_handoff_thread`` returns a fixed id.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB


class _ThreadAdapter:
    """Discord-shaped fake: the only thing /branch needs from an adapter."""

    def __init__(self, thread_id="777000", fail=False):
        self.thread_id, self.fail, self.calls = thread_id, fail, []

    async def create_handoff_thread(self, parent_chat_id, name):
        self.calls.append((parent_chat_id, name))
        return None if self.fail else self.thread_id


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _runner(store, adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
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
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    return runner


def _discord_channel_source():
    return SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="group", user_id="u1",
                         user_name="ann", scope_id="g9")


def _seed(store, source):
    entry = store.get_or_create_session(source)
    store._db.append_message(entry.session_id, role="user", content="hello")
    store._db.append_message(entry.session_id, role="assistant", content="world")
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
    row = store._db.get_session(branch.session_id)
    assert (row["parent_session_id"], row["chat_id"], row["thread_id"], row["chat_type"]) == (
        parent.session_id, "777000", "777000", "thread")
    assert [m["content"] for m in store._db.get_messages(branch.session_id)] == ["hello", "world"]
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
    row = store._db.get_session(current.session_id)
    assert row["parent_session_id"] == parent.session_id
    assert store._db.get_session_title(current.session_id) == "side quest"
    if adapter is not None:
        # ``--here`` never even asks the platform for a thread.
        assert adapter.calls == ([] if "--here" in text else [("123", "side quest")])
    assert "side quest" in reply
