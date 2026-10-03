"""/branch on thread-capable platforms opens a sibling thread and keeps the origin (#66023).

Drives the REAL ``_handle_branch_command`` against a REAL SessionStore + SessionDB (SQLite in
tmp_path); only the platform adapter is a fake whose ``create_handoff_thread`` returns a fixed id.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionEntry, SessionSource, SessionStore
from hermes_constants import get_hermes_home
from hermes_state import AsyncSessionDB


class _ThreadAdapter(BasePlatformAdapter):
    """Discord-shaped fake: the only thing /branch needs from an adapter."""

    def __init__(self, thread_id="777000", fail=False, platform=Platform.DISCORD):
        self.thread_id, self.fail, self.calls = thread_id, fail, []
        self.platform = platform

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("send is outside this test fake's contract")

    async def create_handoff_thread(self, parent_chat_id, name):
        self.calls.append((parent_chat_id, name))
        return None if self.fail else self.thread_id

    async def get_chat_info(self, chat_id):
        return {}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _runner(store, adapter, platform=Platform.DISCORD):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {platform: adapter} if adapter else {}
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
    if adapter is not None:
        adapter.platform, adapter.gateway_runner = platform, runner
    return runner


def _discord_channel_source():
    return SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="group", user_id="u1",
                         user_name="ann", scope_id="g9")


def _seed(store, source):
    entry = store.get_or_create_session(source)
    store._db.append_message(entry.session_id, role="user", content="hello")
    store._db.append_message(entry.session_id, role="assistant", content="world")
    return entry


def _session_ids(db_path):
    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        return {row[0] for row in conn.execute("SELECT id FROM sessions")}


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
@pytest.mark.parametrize(
    "platform, branch_chat_id, branch_chat_type, branch_parent_chat_id",
    [
        (Platform.DISCORD, "777000", "thread", "123"),
        (Platform.TELEGRAM, "123", "group", None),
    ],
)
async def test_thread_branch_keeps_receiving_bot_identity_after_restore(
    tmp_path, monkeypatch, platform, branch_chat_id, branch_chat_type, branch_parent_chat_id,
):
    """A routed branch stays in its runtime DB but keeps the receiving bot's auth/delivery identity."""
    from agent import secret_scope
    import hermes_cli.profiles as profiles
    import hermes_state
    from gateway.run import _SESSION_DB_UNPINNED, _profile_runtime_scope

    home = tmp_path / "hh"
    runtime_home = home / "profiles" / "team_b"
    runtime_home.mkdir(parents=True)
    (runtime_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    allowlist = f"{platform.value.upper()}_ALLOWED_USERS"
    (home / ".env").write_text(f"{allowlist}=u1\n", encoding="utf-8")
    (runtime_home / ".env").write_text(f"{allowlist}=somebody-else\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(allowlist, raising=False)
    # Undo tests/conftest.py's fixed DB override so production's scope-aware resolver is exercised.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        profiles,
        "get_profile_dir",
        lambda name: home if name == "default" else home / "profiles" / name,
    )
    monkeypatch.setattr(
        profiles,
        "profiles_to_serve",
        lambda **_kwargs: [("default", home), ("team_b", runtime_home)],
    )

    config = GatewayConfig(
        multiplex_profiles=True,
        profile_routes=parse_profile_routes([
            {"name": "shared-destination", "platform": platform.value,
             "profile": "team_b", "chat_id": "123"},
        ]),
    )
    config.platforms = {platform: PlatformConfig(enabled=True, extra={})}
    store = SessionStore(sessions_dir=home / "sessions", config=config)
    primary = _ThreadAdapter(platform=platform)
    team_b = _ThreadAdapter(thread_id="unused", platform=platform)
    runner = _runner(store, primary, platform)
    runner.config = config
    runner._primary_profile_name = "default"
    runner._profile_adapters = {"team_b": {platform: team_b}}
    team_b.gateway_runner = runner
    # Unpin the helper's single-profile DB wrapper: the handler must resolve both store facades
    # from the runtime scope exactly as production does.
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles, runner._session_db_handles_lock = {}, threading.Lock()

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        # Real A→B→A storage traversal: root session, routed parent + /branch, root session.
        before = primary.build_source(chat_id="root-before", user_id="u1")
        runner._canonicalize(before, primary_home=home)
        with _profile_runtime_scope(home):
            before_entry = _seed(store, before)

        source = primary.build_source(
            chat_id="123", chat_type="group", user_id="u1", scope_id="g9",
        )
        identity = runner._canonicalize(source, primary_home=home)
        assert source.profile == "team_b"  # selected by the real profile_routes matcher
        assert identity is not None
        assert (identity.transport_profile, identity.runtime_profile) == ("default", "team_b")
        with _profile_runtime_scope(runtime_home):
            parent = _seed(store, source)

        async def dispatch(event):
            key = runner._session_key_for_source(event.source)
            return await runner._hm_dispatch_canonical_command(event, event.source, key, "branch")

        setattr(runner, "_handle_message", dispatch)
        handled, reply = await runner._primary_message_handler()(
            MessageEvent(text="/branch side quest", source=source)
        )

        after = primary.build_source(chat_id="root-after", user_id="u1")
        runner._canonicalize(after, primary_home=home)
        with _profile_runtime_scope(home):
            after_entry = _seed(store, after)
        assert get_hermes_home() == home

        assert handled and "side quest" in reply
        assert primary.calls == [("123", "side quest")]
        branch = next(
            entry for entry in store.list_sessions()
            if entry.origin is not None
            and entry.origin.thread_id == "777000"
            and entry.session_id != parent.session_id
        )
        assert branch.origin is not None
        assert (
            branch.origin.chat_id,
            branch.origin.chat_type,
            branch.origin.parent_chat_id,
        ) == (branch_chat_id, branch_chat_type, branch_parent_chat_id)
        assert branch.transport_profile == "default"

        root_ids = _session_ids(home / "state.db")
        runtime_ids = _session_ids(runtime_home / "state.db")
        root_session_ids = {before_entry.session_id, after_entry.session_id}
        runtime_session_ids = {parent.session_id, branch.session_id}
        assert root_session_ids <= root_ids and root_session_ids.isdisjoint(runtime_ids)
        assert runtime_session_ids <= runtime_ids and runtime_session_ids.isdisjoint(root_ids)

        # The runtime profile deliberately denies u1. Live and restored branch sources must both
        # authorize against the primary transport home and deliver through that receiving bot.
        with _profile_runtime_scope(runtime_home):
            assert runner._is_user_authorized(branch.origin) is False
            assert runner._is_user_authorized_for_source(branch.origin) is True
        assert runner._delivery_adapter_for(branch.origin) is primary

        restored = SessionEntry.from_dict(branch.to_dict())
        restored_source = runner._restored_source(restored)
        assert restored_source is not None
        with _profile_runtime_scope(runtime_home):
            assert runner._is_user_authorized(restored_source) is False
            assert runner._is_user_authorized_for_source(restored_source) is True
        assert runner._delivery_adapter_for(restored_source) is primary
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)


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
