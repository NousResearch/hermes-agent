"""Tests for Mem0Backend abstraction — PlatformBackend and OSSBackend."""

import pytest

from plugins.memory.mem0._backend import Mem0Backend, PlatformBackend, OSSBackend


class FakePlatformClient:
    """Fake MemoryClient for PlatformBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.9}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"count": 1, "next": None, "results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"status": "PENDING", "event_id": "evt-1"}

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {"id": kwargs["memory_id"], "text": kwargs["text"]}

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))


class TestPlatformBackend:

    def _make(self):
        client = FakePlatformClient()
        backend = PlatformBackend.__new__(PlatformBackend)
        backend._client = client
        return backend, client

    def test_search_forwards_params(self):
        backend, client = self._make()
        result = backend.search("test query", filters={"user_id": "u1"}, top_k=5)
        assert client.calls[0][0] == "search"
        assert client.calls[0][1] == "test query"
        assert client.calls[0][2]["filters"] == {"user_id": "u1"}
        assert client.calls[0][2]["top_k"] == 5

    def test_search_forwards_rerank(self):
        backend, client = self._make()
        backend.search("q", filters={}, rerank=False)
        assert client.calls[0][2]["rerank"] is False

    def test_search_rerank_default_true(self):
        backend, client = self._make()
        backend.search("q", filters={})
        assert client.calls[0][2]["rerank"] is True

    def test_search_returns_list(self):
        backend, _ = self._make()
        result = backend.search("q", filters={})
        assert isinstance(result, list)
        assert result[0]["id"] == "m1"

    def test_get_all_forwards_pagination(self):
        backend, client = self._make()
        result = backend.get_all(filters={"user_id": "u1"}, page=2, page_size=50)
        assert client.calls[0][1]["page"] == 2
        assert client.calls[0][1]["page_size"] == 50
        assert "count" in result

    def test_add_forwards_kwargs(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        result = backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)
        call = client.calls[0]
        assert call[2]["user_id"] == "u1"
        assert call[2]["infer"] is False
        # metadata kwarg should be omitted entirely when not provided so we
        # don't surprise older mem0 client versions with an unknown kwarg.
        assert "metadata" not in call[2]

    def test_add_forwards_metadata_when_present(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(
            msgs,
            user_id="u1",
            agent_id="hermes",
            infer=False,
            metadata={"channel": "telegram"},
        )
        assert client.calls[0][2]["metadata"] == {"channel": "telegram"}

    def test_add_omits_empty_metadata(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(msgs, user_id="u1", agent_id="hermes", infer=False, metadata={})
        assert "metadata" not in client.calls[0][2]

    def test_update_forwards(self):
        backend, client = self._make()
        backend.update("m1", "new text")
        assert client.calls[0][1] == {"memory_id": "m1", "text": "new text"}

    def test_delete_forwards(self):
        backend, client = self._make()
        backend.delete("m1")
        assert client.calls[0][1] == {"memory_id": "m1"}


class FakeOSSMemory:
    """Fake mem0.Memory for OSSBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.8}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "event": "ADD"}]}

    def update(self, memory_id, **kwargs):
        self.calls.append(("update", memory_id, kwargs))
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        return {"message": "Memory deleted successfully!"}


class TestOSSBackend:

    def _make(self):
        memory = FakeOSSMemory()
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = memory
        return backend, memory

    def test_search_returns_list(self):
        backend, _ = self._make()
        result = backend.search("test", filters={"user_id": "u1"})
        assert isinstance(result, list)
        assert result[0]["id"] == "m1"

    def test_search_passes_filters(self):
        backend, memory = self._make()
        backend.search("q", filters={"user_id": "u1"}, top_k=3)
        assert memory.calls[0][2]["filters"] == {"user_id": "u1"}
        assert memory.calls[0][2]["top_k"] == 3

    def test_search_forwards_rerank(self):
        """OSS backend forwards rerank param to Memory.search()."""
        backend, memory = self._make()
        backend.search("q", filters={}, rerank=True)
        assert memory.calls[0][2]["rerank"] is True

    def test_search_forwards_rerank_false(self):
        backend, memory = self._make()
        backend.search("q", filters={}, rerank=False)
        assert memory.calls[0][2]["rerank"] is False

    def test_init_forwards_reranker_config(self):
        """OSSBackend.__init__ forwards reranker config to Memory.from_config()."""
        import sys
        import unittest.mock as mock

        captured_configs = []

        def fake_from_config(config):
            captured_configs.append(config)
            return FakeOSSMemory()

        mock_mem0 = mock.MagicMock()
        mock_mem0.Memory.from_config = fake_from_config

        oss_config = {
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/test"}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small", "embedding_dims": 1536}},
            "reranker": {"provider": "cohere", "config": {"model": "rerank-v3.5"}},
        }

        with mock.patch.dict(sys.modules, {"mem0": mock_mem0}):
            backend = OSSBackend(oss_config)

        assert len(captured_configs) == 1
        assert "reranker" in captured_configs[0]
        assert captured_configs[0]["reranker"] == {"provider": "cohere", "config": {"model": "rerank-v3.5"}}

    def test_init_omits_reranker_when_absent(self):
        """OSSBackend.__init__ omits reranker key when not in config."""
        import sys
        import unittest.mock as mock

        captured_configs = []

        def fake_from_config(config):
            captured_configs.append(config)
            return FakeOSSMemory()

        mock_mem0 = mock.MagicMock()
        mock_mem0.Memory.from_config = fake_from_config

        oss_config = {
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/test"}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small", "embedding_dims": 1536}},
        }

        with mock.patch.dict(sys.modules, {"mem0": mock_mem0}):
            backend = OSSBackend(oss_config)

        assert "reranker" not in captured_configs[0]

    def test_get_all_ignores_pagination(self):
        """OSSBackend accepts page/page_size but does NOT forward to Memory.get_all()."""
        backend, memory = self._make()
        result = backend.get_all(filters={"user_id": "u1"}, page=2, page_size=50)
        call_kwargs = memory.calls[0][1]
        assert "page" not in call_kwargs
        assert "page_size" not in call_kwargs
        assert result["count"] == 1

    def test_get_all_returns_envelope(self):
        backend, _ = self._make()
        result = backend.get_all(filters={"user_id": "u1"})
        assert "results" in result
        assert "count" in result

    def test_add_forwards_kwargs(self):
        backend, memory = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)
        assert memory.calls[0][2]["user_id"] == "u1"
        assert memory.calls[0][2]["infer"] is False

    def test_update_maps_text_to_data(self):
        """OSS Memory.update uses `data=` param, not `text=`."""
        backend, memory = self._make()
        backend.update("m1", "new text")
        assert memory.calls[0][0] == "update"
        assert memory.calls[0][1] == "m1"
        assert memory.calls[0][2] == {"data": "new text"}

    def test_delete_positional_arg(self):
        backend, memory = self._make()
        backend.delete("m1")
        assert memory.calls[0] == ("delete", "m1")

    def test_update_normalizes_response(self):
        backend, _ = self._make()
        result = backend.update("m1", "text")
        assert result == {"result": "Memory updated.", "memory_id": "m1"}

    def test_delete_normalizes_response(self):
        backend, _ = self._make()
        result = backend.delete("m1")
        assert result == {"result": "Memory deleted.", "memory_id": "m1"}