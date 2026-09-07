"""Gateway conversation-worktree ownership boundaries.

These tests exercise the SessionStore boundary rather than a mock-only wrapper:
the observable contract is the persisted session entry's effective cwd and the
manager's durable root identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.conversation_worktree import ConversationWorktreeBinding, ConversationWorktreeError
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import (
    AsyncSessionStore,
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_context,
    build_session_key,
)
from gateway.slash_commands import GatewaySlashCommandsMixin


@dataclass
class _Manager:
    root: Path
    fail_new: bool = False

    def __post_init__(self) -> None:
        self.bound_roots: list[str] = []
        self.resolved_roots: list[str] = []

    def bind_new_root_session(
        self, root_session_id: str, *, conversation_kind: str
    ) -> ConversationWorktreeBinding:
        assert conversation_kind == "interactive"
        if self.fail_new:
            raise ConversationWorktreeError("bootstrap did not complete", phase="bootstrap")
        path = self.root / root_session_id
        path.mkdir(parents=True, exist_ok=True)
        self.bound_roots.append(root_session_id)
        return ConversationWorktreeBinding(
            root_session_id=root_session_id,
            path=path,
            branch=f"hermes/session/{root_session_id}",
            base_commit="a" * 40,
            repo_common_dir=self.root,
        )

    def resolve_existing_session(self, root_session_id: str):
        self.resolved_roots.append(root_session_id)
        path = self.root / root_session_id
        if not path.is_dir():
            return None
        return ConversationWorktreeBinding(
            root_session_id=root_session_id,
            path=path,
            branch=f"hermes/session/{root_session_id}",
            base_commit="a" * 40,
            repo_common_dir=self.root,
        )


@pytest.fixture()
def manager(tmp_path) -> _Manager:
    return _Manager(tmp_path / "worktrees")


@pytest.fixture()
def store(tmp_path, manager) -> SessionStore:
    result = SessionStore(
        sessions_dir=tmp_path / "sessions",
        config=GatewayConfig(),
        conversation_worktree_manager_factory=lambda _db: manager,
    )
    # The test exercises the JSON routing mirror. A production SessionDB is
    # neither required for this boundary nor appropriate in a unit test.
    result._db = None
    return result


@pytest.fixture(params=[Platform.DISCORD, Platform("photon")])
def source(request) -> SessionSource:
    return SessionSource(
        platform=request.param,
        chat_id="chat-1",
        chat_type="dm",
        user_id="operator-1",
    )


def test_first_interactive_gateway_session_gets_certified_worktree(store, manager, source):
    """Removing the pre-publish bind must leave this entry without its cwd."""
    entry = store.get_or_create_session(source)

    assert entry.cwd == str(manager.root / entry.session_id)
    assert entry.conversation_worktree["root_session_id"] == entry.session_id
    assert manager.bound_roots == [entry.session_id]


def test_agent_context_uses_certified_session_cwd(store, manager, source):
    """Dropping the task-local cwd bind must route tools back to process cwd."""
    from agent.runtime_cwd import resolve_agent_cwd
    from gateway.run import GatewayRunner

    entry = store.get_or_create_session(source)
    context = build_session_context(source, store.config, entry)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}

    tokens = runner._set_session_env(context)
    try:
        assert resolve_agent_cwd() == Path(entry.cwd)
    finally:
        runner._clear_session_env(tokens)


def test_ordinary_gateway_reconnect_reuses_durable_worktree(store, manager, source):
    """Replacing existing-entry resolution with a new bind must fail this."""
    first = store.get_or_create_session(source)
    second = store.get_or_create_session(source)

    assert second is first
    assert second.cwd == first.cwd
    assert manager.bound_roots == [first.session_id]
    assert manager.resolved_roots == [first.session_id]
    assert list(store._conversation_root_leases) == [first.session_id]


def test_gateway_releases_mandatory_root_lease_only_at_store_teardown(
    store, manager, source
):
    entry = store.get_or_create_session(source)
    lease = store._conversation_root_leases[entry.session_id]

    assert lease.released is False

    store.close_all_db_handles()

    assert lease.released is True
    assert store._conversation_root_leases == {}


def test_gateway_releases_root_lease_when_route_ownership_tears_down(
    store, manager, source
):
    old_entry = store.get_or_create_session(source)
    lease = store._conversation_root_leases[old_entry.session_id]
    new_entry = store.reset_session(old_entry.session_key)

    assert new_entry is not None
    assert store.release_conversation_root_lease(old_entry.session_id) is True

    assert lease.released is True
    assert old_entry.session_id not in store._conversation_root_leases


def test_gateway_keeps_root_lease_while_any_route_still_references_it(
    store, manager, source
):
    entry = store.get_or_create_session(source)
    lease = store._conversation_root_leases[entry.session_id]

    assert store.release_conversation_root_lease(entry.session_id) is False
    assert lease.released is False


def test_gateway_resume_reacquires_target_root_lease_and_workspace(
    store, manager, source
):
    first = store.get_or_create_session(source)
    second = store.reset_session(first.session_key)
    assert second is not None
    assert store.release_conversation_root_lease(first.session_id) is True

    resumed = store.switch_session(first.session_key, first.session_id)

    assert resumed is not None
    assert resumed.cwd == first.cwd
    assert resumed.conversation_worktree == first.conversation_worktree
    assert first.session_id in store._conversation_root_leases


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.DISCORD, Platform("photon")])
async def test_gateway_branch_allocates_and_switches_to_a_distinct_worktree(
    tmp_path, monkeypatch, manager, platform
):
    """Resolving the branch through its parent lineage would reuse the old root."""
    import hermes_state
    from gateway.run import GatewayRunner
    from hermes_state import AsyncSessionDB

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    branch_store = SessionStore(
        sessions_dir=tmp_path / "sessions",
        config=GatewayConfig(),
        conversation_worktree_manager_factory=lambda _db: manager,
    )
    branch_source = SessionSource(
        platform=platform,
        chat_id="branch-chat",
        chat_type="dm",
        user_id="operator-1",
    )
    parent = branch_store.get_or_create_session(branch_source)
    branch_store._db.append_message(parent.session_id, role="user", content="investigate")

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = branch_store
    runner._session_db = AsyncSessionDB(branch_store._db)
    runner._pending_skills_reload_notes = {}

    await runner._handle_branch_command(
        MessageEvent(text="/branch isolated approach", source=branch_source)
    )

    child = branch_store.lookup_by_session_key(parent.session_key)
    assert child is not None
    assert child.session_id != parent.session_id
    assert manager.bound_roots == [parent.session_id, child.session_id]
    assert child.cwd == str(manager.root / child.session_id)
    assert child.cwd != parent.cwd
    assert child.conversation_worktree["root_session_id"] == child.session_id

    assert branch_store.switch_session(parent.session_key, parent.session_id) is not None
    resumed_child = branch_store.switch_session(parent.session_key, child.session_id)
    assert resumed_child is not None
    assert resumed_child.cwd == str(manager.root / child.session_id)
    assert manager.resolved_roots[-1] == child.session_id


def test_failed_gateway_switch_releases_only_new_target_candidate(
    store, manager, source, monkeypatch
):
    first = store.get_or_create_session(source)
    old_lease = store._conversation_root_leases[first.session_id]
    target = "resume-target"
    (manager.root / target).mkdir(parents=True)
    monkeypatch.setattr(store, "_save", lambda: (_ for _ in ()).throw(OSError("denied")))

    with pytest.raises(OSError, match="denied"):
        store.switch_session(first.session_key, target)

    assert store.lookup_by_session_key(first.session_key) is first
    assert old_lease.released is False
    assert target not in store._conversation_root_leases


def test_concurrent_gateway_switch_releases_discarded_target_candidate(
    store, manager, source, monkeypatch
):
    first = store.get_or_create_session(source)
    target = "resume-target"
    (manager.root / target).mkdir(parents=True)
    winner = SimpleNamespace(session_id="winner", conversation_worktree={})
    original_resolve = manager.resolve_existing_session

    def resolve_then_publish_winner(root_session_id):
        binding = original_resolve(root_session_id)
        with store._lock:
            store._entries[first.session_key] = winner
        return binding

    monkeypatch.setattr(manager, "resolve_existing_session", resolve_then_publish_winner)

    assert store.switch_session(first.session_key, target) is winner
    assert target not in store._conversation_root_leases


def test_new_binds_distinct_root_before_replacing_old_session(store, manager, source):
    """Moving binding after the routing replacement must fail this contract."""
    first = store.get_or_create_session(source)
    second = store.reset_session(first.session_key)

    assert second is not None
    assert second.session_id != first.session_id
    assert second.cwd != first.cwd
    assert manager.bound_roots == [first.session_id, second.session_id]


def test_concurrent_reset_loser_releases_unpublished_candidate(
    store, manager, source, monkeypatch
):
    first = store.get_or_create_session(source)
    winner = SimpleNamespace(
        session_id="winner",
        conversation_worktree=dict(first.conversation_worktree),
    )
    original_bind = manager.bind_new_root_session

    def bind_then_publish_winner(root_session_id, *, conversation_kind):
        binding = original_bind(root_session_id, conversation_kind=conversation_kind)
        with store._lock:
            store._entries[first.session_key] = winner
        return binding

    monkeypatch.setattr(manager, "bind_new_root_session", bind_then_publish_winner)

    assert store.reset_session(first.session_key) is winner
    assert list(store._conversation_root_leases) == [first.session_id]


def test_concurrent_create_loser_releases_only_unpublished_candidate(
    store, manager, source, monkeypatch
):
    """A losing first-contact bind must not retain an unrouteable root lease."""
    session_key = build_session_key(source)
    winner_root = "winner-root"
    winner_binding = manager.bind_new_root_session(
        winner_root, conversation_kind="interactive"
    )
    winner = SessionEntry(
        session_key=session_key,
        session_id=winner_root,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )
    store._apply_conversation_worktree_binding(winner, winner_binding)
    winner_lease = store._conversation_root_leases[winner_root]
    original_bind = manager.bind_new_root_session

    def bind_then_publish_winner(root_session_id, *, conversation_kind):
        binding = original_bind(root_session_id, conversation_kind=conversation_kind)
        if root_session_id != winner_root:
            with store._lock:
                store._entries[session_key] = winner
        return binding

    monkeypatch.setattr(manager, "bind_new_root_session", bind_then_publish_winner)

    assert store.get_or_create_session(source) is winner
    assert winner_lease.released is False
    assert list(store._conversation_root_leases) == [winner_root]


def test_failed_new_preserves_old_gateway_boundary(store, manager, source):
    """Publishing reset state before binding would destroy the assertions below."""
    first = store.get_or_create_session(source)
    manager.fail_new = True

    with pytest.raises(ConversationWorktreeError, match="bootstrap did not complete"):
        store.reset_session(first.session_key)

    assert store.lookup_by_session_key(first.session_key) is first
    assert first.cwd == str(manager.root / first.session_id)
    assert manager.bound_roots == [first.session_id]


def test_task_gateway_source_never_allocates_conversation_worktree(store, manager, source):
    """Dropping the explicit task classification must make this allocate a root."""
    entry = store.get_or_create_session(source, conversation_kind="task")

    assert entry.cwd is None
    assert entry.conversation_worktree == {}
    assert manager.bound_roots == []


def _enable_production_policy(monkeypatch, tmp_path) -> None:
    source_root = tmp_path / "stable"
    worktree_root = tmp_path / "worktrees"
    source_root.mkdir(exist_ok=True)
    worktree_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "conversation_worktree": {
                "enabled": True,
                "source_worktree": str(source_root),
                "worktree_root": str(worktree_root),
                "retain_until_explicit_cleanup": True,
            }
        },
    )


@pytest.mark.asyncio
async def test_production_factory_first_contact_fails_before_gateway_state_mutation(
    tmp_path, monkeypatch
):
    """Returning None for an enabled policy would publish a source-cwd route."""
    import agent.runtime_cwd as runtime_cwd
    from agent.runtime_cwd import resolve_agent_cwd, set_session_cwd
    from gateway.run import GatewayRunner

    _enable_production_policy(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="first-contact",
        chat_type="dm",
        user_id="operator-1",
    )
    session_key = build_session_key(source)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = None
    sentinel_cwd = tmp_path / "existing-task-worktree"
    sentinel_cwd.mkdir()

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    unrelated_agent = object()
    runner._agent_cache = {"unrelated": unrelated_agent}
    runner._session_context_prompts = {"unrelated": "context"}
    runner._session_run_generation = {"unrelated": 9}

    token = set_session_cwd(str(sentinel_cwd))
    try:
        with pytest.raises(ConversationWorktreeError, match="SessionDB is unavailable"):
            await runner.async_session_store.get_or_create_session(
                source, conversation_kind="interactive"
            )
        assert resolve_agent_cwd() == sentinel_cwd
    finally:
        runtime_cwd._SESSION_CWD.reset(token)

    assert store.lookup_by_session_key(session_key) is None
    assert runner._agent_cache == {"unrelated": unrelated_agent}
    assert runner._session_context_prompts == {"unrelated": "context"}
    assert runner._session_run_generation == {"unrelated": 9}


@pytest.mark.asyncio
async def test_production_factory_new_failure_preserves_all_old_gateway_state(
    tmp_path, monkeypatch, manager
):
    """A disabled manager fallback during /new must not rotate any state."""
    import agent.runtime_cwd as runtime_cwd
    from agent.runtime_cwd import resolve_agent_cwd, set_session_cwd
    from gateway.session import _default_conversation_worktree_manager_factory

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="existing-chat",
        chat_type="dm",
        user_id="operator-1",
    )
    store = SessionStore(
        tmp_path / "sessions",
        GatewayConfig(),
        conversation_worktree_manager_factory=lambda _db: manager,
    )
    store._db = None
    old_entry = store.get_or_create_session(source)
    store._conversation_worktree_manager_factory = (
        _default_conversation_worktree_manager_factory
    )
    _enable_production_policy(monkeypatch, tmp_path)
    session_key = old_entry.session_key
    cached_agent = object()
    sentinel_cwd = Path(old_entry.cwd)

    class _Runner(GatewaySlashCommandsMixin):
        def __init__(self):
            self.session_store = store
            self._async_session_store = AsyncSessionStore(store)
            self._agent_cache = {session_key: cached_agent}
            self._session_context_prompts = {session_key: "old-context"}
            self._session_run_generation = {session_key: 4}

        @property
        def async_session_store(self):
            return self._async_session_store

        def _session_key_for_source(self, _source):
            return session_key

        def _invalidate_session_run_generation(self, *_args, **_kwargs):
            raise AssertionError("generation changed before replacement root was ready")

        def _release_running_agent_state(self, _key):
            raise AssertionError("running state changed before replacement root was ready")

    runner = _Runner()
    token = set_session_cwd(str(sentinel_cwd))
    try:
        reply = await runner._handle_reset_command(
            MessageEvent(text="/new", source=source, timestamp=datetime.now())
        )
        assert resolve_agent_cwd() == sentinel_cwd
    finally:
        runtime_cwd._SESSION_CWD.reset(token)

    assert "conversation worktree setup failed" in str(reply)
    assert store.lookup_by_session_key(session_key) is old_entry
    assert runner._agent_cache[session_key] is cached_agent
    assert runner._session_context_prompts == {session_key: "old-context"}
    assert runner._session_run_generation == {session_key: 4}


@pytest.mark.asyncio
async def test_first_cli_handoff_reuses_verified_workspace_without_binding_claim(
    tmp_path, manager
):
    """Default interactive creation would claim a second root and lose CLI cwd."""
    from gateway.run import GatewayRunner

    cli_workspace = manager.root / "cli-session"
    cli_workspace.mkdir(parents=True)
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="test")}
    )
    config.platforms[Platform.DISCORD].home_channel = HomeChannel(
        platform=Platform.DISCORD,
        chat_id="home-channel",
        name="Example Project",
    )
    store = SessionStore(
        tmp_path / "sessions",
        config,
        conversation_worktree_manager_factory=lambda _db: manager,
    )
    store._db = SimpleNamespace(get_conversation_root=lambda session_id: session_id)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value=None)
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))

    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {Platform.DISCORD: adapter}
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._evict_cached_agent = lambda _key: None
    runner._release_running_agent_state = lambda _key: None
    observed: dict[str, str] = {}

    async def capture_handoff(event):
        entry = store.lookup_by_session_key(build_session_key(event.source))
        observed["cwd"] = build_session_context(event.source, config, entry).cwd
        return "handoff ready"

    runner._handle_message = capture_handoff

    await runner._process_handoff(
        {
            "id": "cli-session",
            "title": "CLI work",
            "handoff_platform": "discord",
            "cwd": str(cli_workspace),
        }
    )

    destination = SessionSource(
        platform=Platform.DISCORD,
        chat_id="home-channel",
        chat_name="Example Project",
        chat_type="dm",
        user_id="system:handoff",
        user_name="Handoff",
    )
    entry = store.lookup_by_session_key(build_session_key(destination))
    assert manager.bound_roots == []
    assert entry.session_id == "cli-session"
    assert entry.cwd == str(cli_workspace)
    assert observed["cwd"] == str(cli_workspace)
    assert list(store._conversation_root_leases) == ["cli-session"]


@pytest.mark.asyncio
async def test_new_bind_failure_preserves_cached_agent_and_generation(tmp_path):
    """Moving cache/generation cleanup before reset preparation must fail this."""
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-1",
        chat_type="dm",
        user_id="operator-1",
    )
    session_key = "agent:main:discord:dm:chat-1"
    old_entry = SimpleNamespace(session_id="old-root")
    cached_agent = object()

    class _FailingAsyncStore:
        async def reset_session(self, key, **_kwargs):
            assert key == session_key
            raise ConversationWorktreeError("bootstrap did not complete", phase="bootstrap")

    class _Runner(GatewaySlashCommandsMixin):
        def __init__(self):
            self.session_store = SimpleNamespace(_entries={session_key: old_entry})
            self._async_session_store = _FailingAsyncStore()
            self._agent_cache = {session_key: cached_agent}
            self.invalidations: list[tuple[str, str]] = []

        @property
        def async_session_store(self):
            return self._async_session_store

        def _session_key_for_source(self, _source):
            return session_key

        def _invalidate_session_run_generation(self, key, *, reason):
            self.invalidations.append((key, reason))

        def _release_running_agent_state(self, _key):
            raise AssertionError("old running state was released before new root was ready")

    runner = _Runner()
    event = MessageEvent(text="/new", source=source, timestamp=datetime.now())

    reply = await runner._handle_reset_command(event)

    assert "conversation worktree setup failed" in str(reply)
    assert runner.session_store._entries[session_key] is old_entry
    assert runner._agent_cache[session_key] is cached_agent
    assert runner.invalidations == []
