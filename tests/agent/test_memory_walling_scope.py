"""Regression coverage for Discord thread memory-provider walling.

These tests use fake providers and temporary profile directories only. They
document that providers can isolate project threads when they honor
``gateway_session_key``, while profile-only memory remains shared across
Discord project threads in the same Hermes profile.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.background_review import build_memory_write_metadata
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from tools.memory_tool import ENTRY_DELIMITER, MemoryStore


THREAD_A = {
    "session_id": "sess-thread-a",
    "platform": "discord",
    "chat_type": "thread",
    "chat_id": "200001",
    "thread_id": "200001",
    "gateway_session_key": "agent:main:discord:thread:200001:200001",
    "agent_identity": "main",
}

THREAD_B = {
    "session_id": "sess-thread-b",
    "platform": "discord",
    "chat_type": "thread",
    "chat_id": "200002",
    "thread_id": "200002",
    "gateway_session_key": "agent:main:discord:thread:200002:200002",
    "agent_identity": "main",
}


class _RecordingScopeProvider(MemoryProvider):
    """Minimal provider that records scope-sensitive lifecycle calls."""

    def __init__(self, *, namespace_field: str = "gateway_session_key") -> None:
        self._name = "scope-recorder"
        self.namespace_field = namespace_field
        self.init_calls: list[dict[str, str]] = []
        self.memory_writes: list[dict[str, object]] = []
        self._store: dict[str, list[str]] = {}
        self._current_scope: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._current_scope = {"session_id": session_id, **kwargs}
        self.init_calls.append(dict(self._current_scope))

    def get_tool_schemas(self):
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        namespace = self._namespace()
        return "\n".join(self._store.get(namespace, []))

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        namespace = self._namespace()
        self._store.setdefault(namespace, []).append(user_content)

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        self.memory_writes.append(
            {
                "action": action,
                "target": target,
                "content": content,
                "metadata": dict(metadata or {}),
            }
        )

    def _namespace(self) -> str:
        value = self._current_scope.get(self.namespace_field)
        return value or "global"


def _initialize_provider(scope: dict[str, str], *, namespace_field: str) -> _RecordingScopeProvider:
    provider = _RecordingScopeProvider(namespace_field=namespace_field)
    manager = MemoryManager()
    manager.add_provider(provider)
    kwargs = dict(scope)
    session_id = kwargs.pop("session_id")
    manager.initialize_all(session_id=session_id, hermes_home="/tmp/fake-hermes-home", **kwargs)
    return provider


def test_discord_thread_provider_init_receives_full_scope_metadata():
    provider = _initialize_provider(THREAD_A, namespace_field="gateway_session_key")

    assert provider.init_calls == [
        {
            "session_id": THREAD_A["session_id"],
            "platform": "discord",
            "chat_type": "thread",
            "chat_id": THREAD_A["chat_id"],
            "thread_id": THREAD_A["thread_id"],
            "gateway_session_key": THREAD_A["gateway_session_key"],
            "agent_identity": "main",
            "hermes_home": "/tmp/fake-hermes-home",
        }
    ]


def test_gateway_session_key_scoped_provider_isolates_discord_project_threads():
    provider_a = _initialize_provider(THREAD_A, namespace_field="gateway_session_key")
    provider_a.sync_turn("Project A uses pinned storyboard rules.", "noted")

    provider_b = _initialize_provider(THREAD_B, namespace_field="gateway_session_key")
    provider_b._store = provider_a._store
    provider_b.sync_turn("Project B uses a separate research queue.", "noted")

    assert "Project B" not in provider_a.prefetch("Project A")
    assert "Project A uses pinned storyboard rules" not in provider_b.prefetch("Project B")
    assert "Project A uses pinned storyboard rules" in provider_a.prefetch("Project A")
    assert "Project B uses a separate research queue" in provider_b.prefetch("Project B")


def test_profile_only_provider_leaks_across_discord_project_threads():
    provider_a = _initialize_provider(THREAD_A, namespace_field="agent_identity")
    provider_a.sync_turn("Project A must never share memory with Project B.", "noted")

    provider_b = _initialize_provider(THREAD_B, namespace_field="agent_identity")
    provider_b._store = provider_a._store

    assert provider_b.prefetch("Project B") == "Project A must never share memory with Project B."


def test_memory_write_metadata_includes_discord_thread_scope():
    agent = SimpleNamespace(
        session_id=THREAD_A["session_id"],
        _parent_session_id="",
        platform="discord",
        _chat_type="thread",
        _chat_id=THREAD_A["chat_id"],
        _thread_id=THREAD_A["thread_id"],
        _gateway_session_key=THREAD_A["gateway_session_key"],
        _user_id="user-123",
        _user_name="fake-user",
        _chat_name="fake-forum-thread",
    )

    metadata = build_memory_write_metadata(agent)

    assert metadata["platform"] == "discord"
    assert metadata["chat_type"] == "thread"
    assert metadata["chat_id"] == THREAD_A["chat_id"]
    assert metadata["thread_id"] == THREAD_A["thread_id"]
    assert metadata["gateway_session_key"] == THREAD_A["gateway_session_key"]
    assert metadata["user_id"] == "user-123"
    assert metadata["user_name"] == "fake-user"
    assert metadata["chat_name"] == "fake-forum-thread"


def test_builtin_memory_files_are_profile_scoped_not_thread_scoped(tmp_path, monkeypatch):
    profile_home = tmp_path / "main-profile"
    memory_dir = profile_home / "memories"
    memory_dir.mkdir(parents=True)
    (memory_dir / "MEMORY.md").write_text(
        ENTRY_DELIMITER.join(["Shared profile note visible to every project thread."]),
        encoding="utf-8",
    )
    (memory_dir / "USER.md").write_text(
        ENTRY_DELIMITER.join(["Shared user preference visible to every project thread."]),
        encoding="utf-8",
    )

    monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: profile_home)

    thread_a_store = MemoryStore()
    thread_b_store = MemoryStore()
    thread_a_store.load_from_disk()
    thread_b_store.load_from_disk()

    assert thread_a_store.format_for_system_prompt("memory") == thread_b_store.format_for_system_prompt("memory")
    assert thread_a_store.format_for_system_prompt("user") == thread_b_store.format_for_system_prompt("user")
    assert "Shared profile note visible to every project thread." in (
        thread_a_store.format_for_system_prompt("memory") or ""
    )


@pytest.mark.xfail(
    reason=(
        "Built-in memory has no default-off walled_project mode yet; "
        "profile-global MEMORY.md is still injected for every thread."
    ),
    strict=True,
)
def test_future_walled_project_mode_excludes_profile_global_memory_by_default(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "main-profile"
    memory_dir = profile_home / "memories"
    memory_dir.mkdir(parents=True)
    (memory_dir / "MEMORY.md").write_text(
        ENTRY_DELIMITER.join(["Profile-global memory should not enter a walled project."]),
        encoding="utf-8",
    )

    monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: profile_home)

    store = MemoryStore()
    store.load_from_disk()

    assert store.format_for_system_prompt("memory") is None


@pytest.mark.xfail(
    reason=(
        "Built-in memory has no thread-scoped path resolver yet; both stores "
        "still read and write the same profile-global MEMORY.md."
    ),
    strict=True,
)
def test_future_discord_thread_scoped_mode_uses_distinct_builtin_memory_paths(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "main-profile"
    memory_dir = profile_home / "memories"
    memory_dir.mkdir(parents=True)
    monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: profile_home)

    thread_a_store = MemoryStore()
    thread_b_store = MemoryStore()
    assert thread_a_store.add("memory", "Only Project A should see this.")["success"] is True
    thread_b_store.load_from_disk()

    assert "Only Project A should see this." not in (
        thread_b_store.format_for_system_prompt("memory") or ""
    )
