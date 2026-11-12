"""A terminal A2A result remains queryable after constructing a new discovered adapter."""

import asyncio
import sqlite3

import pytest
from concurrent.futures import ThreadPoolExecutor

from tests.plugins.test_a2a_plugin import _free_port, _post_json, _send_body


def test_discovered_adapter_restores_original_terminal_result_in_owning_profile(tmp_path, monkeypatch):
    import hermes_constants
    from gateway.config import PlatformConfig
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginManager

    home = tmp_path / "owner"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: [a2a]\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("A2A_PORT", raising=False)
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    monkeypatch.setenv("A2A_HOST", "127.0.0.1")
    manager = PluginManager()
    manager.discover_and_load()
    first = platform_registry.create_adapter("a2a", PlatformConfig(enabled=True, extra={"port": _free_port()}))
    assert first is not None
    calls = []

    async def reply(event):
        calls.append(event.text)
        await first.send(event.source.chat_id, "Original result after restart", metadata={"notify": True})

    first.handle_message = reply
    first._message_handler = object()

    async def roundtrip():
        assert await first.connect()
        try:
            return await asyncio.to_thread(_post_json, f"http://127.0.0.1:{first.port}/", _send_body("Remember this result"))
        finally:
            await first.disconnect()

    response = asyncio.run(roundtrip())
    original = response["result"]
    second = platform_registry.create_adapter("a2a", PlatformConfig(enabled=True, extra={"port": _free_port()}))
    task_id = original["id"]
    other = tmp_path / "other"
    other.mkdir()
    scope = hermes_constants.set_hermes_home_override(str(other))
    try:
        async def lookup():
            assert await second.connect()
            try:
                return await asyncio.to_thread(_post_json, f"http://127.0.0.1:{second.port}/", {
                    "jsonrpc": "2.0", "id": "read", "method": "tasks/get", "params": {"id": task_id},
                })
            finally:
                await second.disconnect()
        restored = asyncio.run(lookup())
        task = restored["result"]
        assert (task["id"], task["contextId"], task["status"]["state"]) == (
            original["id"], original["contextId"], original["status"]["state"],
        )
        assert task["artifacts"][0]["parts"] == original["artifacts"][0]["parts"]
        assert len(calls) == 1
    finally:
        hermes_constants.reset_hermes_home_override(scope)
        manager.unload()


def test_terminal_commit_failure_leaves_observers_pending_and_profile_history_bounded(tmp_path):
    import hermes_constants
    from plugins.platforms.a2a import protocol

    class SmallStore(protocol.TaskStore):
        _MAX_TERMINAL = 2

    owner, other = tmp_path / "owner", tmp_path / "other"
    scope = hermes_constants.set_hermes_home_override(str(owner))
    try:
        store = SmallStore.for_profile()
    finally:
        hermes_constants.reset_hermes_home_override(scope)
    store.create("task-original", "context-original", "peer", agent_slug="alpha", tenant="tenant-a")
    store.set_push_config("task-original", "https://callback.invalid/private-token")
    watcher = store.watch("task-original")
    database = owner / "a2a" / "terminal_tasks.sqlite3"
    saved = database.with_suffix(".backup")
    database.rename(saved)
    database.mkdir()
    try:
        with pytest.raises(sqlite3.OperationalError):
            store.complete("task-original", protocol.STATE_COMPLETED, "original result")
        assert not watcher.done()
        assert store.get("task-original")["state"] == protocol.STATE_SUBMITTED
    finally:
        database.rmdir()
        saved.rename(database)
    scope = hermes_constants.set_hermes_home_override(str(other))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(store.complete, "task-original", protocol.STATE_COMPLETED, "original result").result()
        assert watcher.result() == (protocol.STATE_COMPLETED, "original result")
        assert SmallStore.for_profile().get("task-original") is None
    finally:
        hermes_constants.reset_hermes_home_override(scope)
    restored = SmallStore(profile_home=owner)
    assert restored.get("task-original", agent_slug="alpha", tenant="tenant-a")["reply"] == "original result"
    assert restored.get("task-original", agent_slug="beta") is None
    assert restored.get("task-original", tenant="tenant-b") is None
    assert restored.get_push_config("task-original") is None
    for number in range(2):
        restored.create(str(number), "another-context", "peer")
        restored.complete(str(number), protocol.STATE_COMPLETED, "retained result")
    retained = SmallStore(profile_home=owner)
    assert retained.get("task-original") is None
    assert retained.get("0")["reply"] == retained.get("1")["reply"] == "retained result"
