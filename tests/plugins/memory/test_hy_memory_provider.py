import json
import sys
import types

from plugins.memory.hy_memory import HyMemoryProvider


class FakeHyMemoryClient:
    instances = []

    def __init__(self, mode="lite"):
        self.mode = mode
        self.add_calls = []
        self.search_calls = []
        self.closed = False
        self.search_payload = {
            "memories": {
                "normal": [
                    {"content": "User prefers local Android validation.", "score": 0.91},
                ],
            },
        }
        FakeHyMemoryClient.instances.append(self)

    def add(self, data, *, user_id="", agent_id="default_agent", session_id="default_session", metadata=None):
        self.add_calls.append(
            {
                "data": data,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "metadata": metadata,
            }
        )
        return {"id": "hy-1"}

    def search(self, query, *, user_ids=None, agent_ids=None, session_ids=None, limit=10, min_score=0.4):
        self.search_calls.append(
            {
                "query": query,
                "user_ids": user_ids,
                "agent_ids": agent_ids,
                "session_ids": session_ids,
                "limit": limit,
                "min_score": min_score,
            }
        )
        return self.search_payload

    def close(self):
        self.closed = True


def install_fake_hy_memory(monkeypatch):
    FakeHyMemoryClient.instances.clear()
    module = types.ModuleType("hy_memory")
    module.HyMemoryClient = FakeHyMemoryClient
    monkeypatch.setitem(sys.modules, "hy_memory", module)
    monkeypatch.setattr("plugins.memory.hy_memory.importlib.util.find_spec", lambda name: object() if name == "hy_memory" else None)
    monkeypatch.setattr(
        "plugins.memory.hy_memory.load_config",
        lambda: {"memory": {"hy_memory": {"mode": "lite", "limit": 3}}},
    )


def test_hy_memory_provider_uses_package_client_for_recall_and_turn_sync(monkeypatch, tmp_path):
    install_fake_hy_memory(monkeypatch)
    provider = HyMemoryProvider()

    assert provider.is_available() is True
    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        user_id="user-1",
        agent_identity="coder",
    )

    prefetch = provider.prefetch("android validation")
    assert "User prefers local Android validation." in prefetch
    client = FakeHyMemoryClient.instances[-1]
    assert client.search_calls[-1]["user_ids"] == ["user-1"]
    assert client.search_calls[-1]["agent_ids"] == ["coder"]

    provider.sync_turn("remember the emulator proof", "I will retain it", session_id="session-1")
    provider._write_queue.join()
    assert client.add_calls[-1]["session_id"] == "session-1"
    assert client.add_calls[-1]["metadata"]["source"] == "completed_turn"

    provider.shutdown()
    assert client.closed is True


def test_hy_memory_tools_do_not_collide_with_builtin_memory_tool(monkeypatch, tmp_path):
    install_fake_hy_memory(monkeypatch)
    provider = HyMemoryProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")

    tool_names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert tool_names == {"hy_memory_retain", "hy_memory_recall", "hy_memory_status"}
    assert "memory" not in tool_names

    retained = json.loads(provider.handle_tool_call("hy_memory_retain", {"content": "Keep Shizuku status."}))
    recalled = json.loads(provider.handle_tool_call("hy_memory_recall", {"query": "Shizuku", "limit": 2}))
    status = json.loads(provider.handle_tool_call("hy_memory_status", {}))

    assert retained["success"] is True
    assert recalled["success"] is True
    assert status["provider"] == "hy_memory"
    assert status["success"] is True
    provider.shutdown()
