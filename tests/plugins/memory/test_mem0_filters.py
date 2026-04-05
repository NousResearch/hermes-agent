from plugins.memory.mem0 import Mem0MemoryProvider


def test_mem0_search_uses_filters_not_top_level_user_id(monkeypatch):
    provider = Mem0MemoryProvider()
    provider.initialize("test")
    provider._user_id = "u123"

    captured = {}

    class FakeClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": []}

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    result = provider.handle_tool_call(
        "mem0_search",
        {"query": "hello", "top_k": 3, "rerank": False},
    )

    assert '"result": "No relevant memories found."' in result
    assert captured["query"] == "hello"
    assert captured["top_k"] == 3
    assert captured["rerank"] is False
    assert captured["filters"] == {"user_id": "u123"}
    assert "user_id" not in captured


def test_mem0_profile_uses_filters_not_top_level_user_id(monkeypatch):
    provider = Mem0MemoryProvider()
    provider.initialize("test")
    provider._user_id = "u123"

    captured = {}

    class FakeClient:
        def get_all(self, **kwargs):
            captured.update(kwargs)
            return {"results": []}

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    result = provider.handle_tool_call("mem0_profile", {})

    assert '"result": "No memories stored yet."' in result
    assert captured["filters"] == {"user_id": "u123"}
    assert "user_id" not in captured


def test_mem0_prefetch_uses_filters_not_top_level_user_id(monkeypatch):
    provider = Mem0MemoryProvider()
    provider.initialize("test")
    provider._user_id = "u123"

    captured = {}

    class FakeClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": []}

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    provider.queue_prefetch("hello")
    provider._prefetch_thread.join(timeout=2)

    assert captured["query"] == "hello"
    assert captured["top_k"] == 5
    assert captured["filters"] == {"user_id": "u123"}
    assert "user_id" not in captured


def test_mem0_profile_accepts_dict_results(monkeypatch):
    provider = Mem0MemoryProvider()
    provider.initialize("test")

    class FakeClient:
        def get_all(self, **kwargs):
            return {"results": [{"memory": "alpha"}, {"memory": "beta"}]}

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    result = provider.handle_tool_call("mem0_profile", {})

    assert '"count": 2' in result
    assert "alpha" in result
    assert "beta" in result


def test_mem0_search_accepts_dict_results(monkeypatch):
    provider = Mem0MemoryProvider()
    provider.initialize("test")

    class FakeClient:
        def search(self, **kwargs):
            return {"results": [{"memory": "alpha", "score": 0.9}]}

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    result = provider.handle_tool_call("mem0_search", {"query": "alpha"})

    assert '"count": 1' in result
    assert "alpha" in result
    assert '"score": 0.9' in result
