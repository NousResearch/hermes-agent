"""Provider ownership through the shared inline memory executor and approval replay."""

import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import InlineToolContext, _memory
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, MemoryWriteResult
from tools.memory_tool import MemoryStore


class Provider(MemoryProvider):
    def __init__(self, *, fail=False):
        self.fail = fail
        self.writes = []
        self.mirrors = []
        self.initialized = None

    @property
    def name(self):
        return "external"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.initialized = (session_id, kwargs)

    def get_tool_schemas(self):
        return []

    def wants_memory_write(self, intent):
        return intent.target == "user"

    def handle_memory_write(self, intent):
        self.writes.append(intent)
        if self.fail or intent.action != "add":
            return MemoryWriteResult(handled=True, provider=self.name, error="provider refused")
        return MemoryWriteResult(handled=True, success=True, provider=self.name)

    def memory_write_replay_context(self):
        return {"session_id": "session-1", "kwargs": {"platform": "cli", "user_id": "user-1"}}

    def on_memory_write(self, action, target, content, metadata=None):
        self.mirrors.append((action, target, content))

    def shutdown(self):
        pass


@pytest.fixture
def setup_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path / "memories")
    store = MemoryStore()
    store.load_from_disk()
    provider = Provider()
    manager = MemoryManager()
    manager.add_provider(provider)
    agent = SimpleNamespace(
        _memory_store=store, _memory_manager=manager,
        _build_memory_write_metadata=lambda **kw: {"task_id": kw["task_id"]},
    )
    return agent, provider


def call(agent, **args):
    return json.loads(_memory(agent, args, InlineToolContext(effective_task_id="task-1")))


def test_provider_add_is_primary_and_local_add_keeps_mirror(setup_memory):
    agent, provider = setup_memory
    result = call(agent, action="add", target="user", content="Likes jazz")
    assert result["success"] is True and result["provider"] == "external"
    assert agent._memory_store.user_entries == []
    assert [write.content for write in provider.writes] == ["Likes jazz"]
    assert provider.mirrors == []

    local = call(agent, action="add", target="memory", content="Use pytest")
    assert local["success"] is True
    assert agent._memory_store.memory_entries == ["Use pytest"]
    agent._memory_manager._drain_sync_executor()
    assert ("add", "memory", "Use pytest") in provider.mirrors


@pytest.mark.parametrize("action,extra", [
    ("replace", {"old_text": "jazz", "content": "Likes blues"}),
    ("remove", {"old_text": "jazz"}),
    (None, {"operations": [{"action": "add", "content": "Likes jazz"},
                            {"action": "add", "content": "Likes blues"}]}),
])
def test_claimed_edits_and_atomic_batches_fail_without_local_write(setup_memory, action, extra):
    agent, provider = setup_memory
    result = call(agent, action=action, target="user", **extra)
    assert result["success"] is False and result["provider"] == "external"
    assert agent._memory_store.user_entries == []
    assert len(provider.writes) == 1


def test_claimed_provider_failure_never_falls_back(setup_memory):
    agent, provider = setup_memory
    provider.fail = True
    result = call(agent, action="add", target="user", content="Likes jazz")
    assert result["success"] is False and result["error"] == "provider refused"
    assert agent._memory_store.user_entries == []


def test_approval_stages_then_replays_with_same_provider_identity(setup_memory, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    agent, provider = setup_memory
    config = load_config()
    config.setdefault("memory", {}).update({"provider": "external", "write_approval": True})
    save_config(config)
    staged = call(agent, action="add", target="user", content="Likes jazz")
    assert staged["staged"] is True and not provider.writes
    assert agent._memory_store.user_entries == []
    payload = wa.get_pending("memory", staged["pending_id"])["payload"]
    assert payload["provider"] == "external"

    replay = Provider()
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: replay)
    output = handle_pending_subcommand("memory", ["approve", staged["pending_id"]],
                                       memory_store=agent._memory_store)
    assert "Approved 1" in output
    assert replay.initialized == ("session-1", {"platform": "cli", "user_id": "user-1"})
    assert replay.writes[0].content == "Likes jazz"
    assert agent._memory_store.user_entries == []
    assert wa.get_pending("memory", staged["pending_id"]) is None


def test_provider_change_keeps_pending_and_never_writes_local(setup_memory):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    agent, _ = setup_memory
    config = load_config()
    config.setdefault("memory", {})["write_approval"] = True
    save_config(config)
    staged = call(agent, action="add", target="user", content="Likes jazz")
    output = handle_pending_subcommand("memory", ["approve", staged["pending_id"]],
                                       memory_store=agent._memory_store)
    assert "active memory provider changed" in output
    assert wa.get_pending("memory", staged["pending_id"]) is not None
    assert agent._memory_store.user_entries == []


def test_both_runtime_dispatch_paths_use_provider_owned_executor(setup_memory):
    from agent.inline_tool_executors import resolve_invoke_tool_executor
    from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch

    agent, provider = setup_memory
    args = {"action": "add", "target": "user", "content": "Likes jazz"}
    concurrent = resolve_invoke_tool_executor(agent, "memory")
    sequential = _resolve_sequential_dispatch(
        agent, _ToolCallRef("memory", args, "task-1", "call-1", []), [],
    ).execute
    assert json.loads(concurrent(agent, args, InlineToolContext("task-1")))["provider"] == "external"
    assert json.loads(sequential(args))["provider"] == "external"
    assert len(provider.writes) == 2
    assert agent._memory_store.user_entries == []


def test_unclaimed_local_batch_keeps_atomic_store_semantics(setup_memory):
    agent, provider = setup_memory
    result = call(agent, target="memory", operations=[
        {"action": "add", "content": "old entry"},
        {"action": "replace", "old_text": "old entry", "content": "new entry"},
    ])
    assert result["success"] is True
    assert agent._memory_store.memory_entries == ["new entry"]
    assert provider.writes == []


def test_provider_exception_is_a_claimed_failure(setup_memory):
    agent, provider = setup_memory
    def crash(intent):
        raise RuntimeError("remote unavailable")
    provider.handle_memory_write = crash
    result = call(agent, action="add", target="user", content="Likes jazz")
    assert result["success"] is False and "remote unavailable" in result["error"]
    assert agent._memory_store.user_entries == []


def test_claimed_success_without_provider_label_is_not_mirrored(setup_memory):
    agent, provider = setup_memory
    provider.handle_memory_write = lambda intent: MemoryWriteResult(handled=True, success=True)
    result = call(agent, action="add", target="user", content="Likes jazz")
    assert result["provider"] == "external"
    assert provider.mirrors == []
    assert agent._memory_store.user_entries == []


def test_approval_without_replay_identity_fails_before_staging(setup_memory):
    from hermes_cli.config import load_config, save_config
    from tools import write_approval as wa

    agent, provider = setup_memory
    config = load_config()
    config.setdefault("memory", {})["write_approval"] = True
    save_config(config)
    provider.memory_write_replay_context = lambda: {}
    result = call(agent, action="add", target="user", content="Likes jazz")
    assert result["success"] is False and "replayable session identity" in result["error"]
    assert wa.pending_count("memory") == 0
    assert agent._memory_store.user_entries == []


def test_pending_provider_write_keeps_profile_and_user_identity_across_homes(tmp_path, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
    pending_ids = []
    for home, user_id in ((homes[0], "user-a"), (homes[1], "user-b"), (homes[0], "user-a")):
        monkeypatch.setenv("HERMES_HOME", str(home))
        config = load_config()
        config.setdefault("memory", {}).update({"provider": "external", "write_approval": True})
        save_config(config)
        store = MemoryStore(); store.load_from_disk()
        provider = Provider()
        provider.memory_write_replay_context = lambda user_id=user_id: {
            "session_id": "session-1", "kwargs": {"platform": "telegram", "user_id": user_id}}
        manager = MemoryManager(); manager.add_provider(provider)
        agent = SimpleNamespace(_memory_store=store, _memory_manager=manager,
                                _build_memory_write_metadata=lambda **kw: {})
        pending_ids.append(call(agent, action="add", target="user", content=f"Fact for {user_id}")["pending_id"])
        assert store.user_entries == [] and provider.writes == []

    replay = Provider()
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: replay)
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    assert wa.pending_count("memory") == 2
    assert "Approved 1" in handle_pending_subcommand(
        "memory", ["approve", pending_ids[0]], memory_store=MemoryStore())
    assert replay.initialized[1]["user_id"] == "user-a"
    monkeypatch.setenv("HERMES_HOME", str(homes[1]))
    assert wa.pending_count("memory") == 1
    assert "Approved 1" in handle_pending_subcommand(
        "memory", ["approve", pending_ids[1]], memory_store=MemoryStore())
    assert replay.initialized[1]["user_id"] == "user-b"
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    assert wa.pending_count("memory") == 1


def test_claimed_add_batch_is_preserved_through_approval(setup_memory, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.write_approval_commands import handle_pending_subcommand

    agent, provider = setup_memory
    provider.handle_memory_write = lambda intent: (
        provider.writes.append(intent) or MemoryWriteResult(handled=True, success=True, provider="external"))
    config = load_config()
    config.setdefault("memory", {}).update({"provider": "external", "write_approval": True})
    save_config(config)
    operations = [{"action": "add", "content": "Likes jazz"},
                  {"action": "add", "content": "Prefers concise replies"}]
    staged = call(agent, target="user", operations=operations)
    assert staged["staged"] and provider.writes == []

    replay = Provider()
    replay.handle_memory_write = lambda intent: (
        replay.writes.append(intent) or MemoryWriteResult(handled=True, success=True, provider="external"))
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: replay)
    assert "Approved 1" in handle_pending_subcommand(
        "memory", ["approve", staged["pending_id"]], memory_store=agent._memory_store)
    assert len(replay.writes) == 1 and replay.writes[0].operations == operations
    assert agent._memory_store.user_entries == []


def test_claimed_destructive_write_with_approval_cannot_bypass_review(setup_memory):
    from hermes_cli.config import load_config, save_config
    from tools import write_approval as wa

    agent, provider = setup_memory
    config = load_config()
    config.setdefault("memory", {})["write_approval"] = True
    save_config(config)
    result = call(agent, action="replace", target="user", old_text="jazz", content="Likes blues")
    assert result["success"] is False and "provider-specific review" in result["error"]
    assert provider.writes == [] and wa.pending_count("memory") == 0
    assert agent._memory_store.user_entries == []
