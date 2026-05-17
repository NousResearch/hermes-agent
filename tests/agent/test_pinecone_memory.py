from types import SimpleNamespace

import pytest

from agent.pinecone_memory import PineconeMemoryClient


class FakeIndex:
    def __init__(self):
        self.upsert_calls = []
        self.query_calls = []
        self.delete_calls = []

    def upsert(self, *, vectors, namespace):
        self.upsert_calls.append({"vectors": vectors, "namespace": namespace})
        return {"upserted_count": len(vectors)}

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "matches": [
                {"id": "chunk-1", "score": 0.99, "metadata": {"source_id": "doc-1"}},
            ]
        }

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)
        return {"ok": True}


class RaisingIndex:
    def upsert(self, **kwargs):
        raise RuntimeError("boom")

    def query(self, **kwargs):
        raise RuntimeError("boom")

    def delete(self, **kwargs):
        raise RuntimeError("boom")


def test_missing_config_returns_empty_results_and_logs_warning(caplog):
    client = PineconeMemoryClient(api_key="", index_name="", fail_open=True)

    with caplog.at_level("WARNING"):
        assert client.upsert_records([{"id": "1", "values": [1, 2], "metadata": {}}]) == 0
        assert client.query([0.1, 0.2]) == []
        assert client.delete_by_source(source_kind="file", source_id="abc") == 0

    assert "missing configuration" in caplog.text


def test_successful_upsert_query_and_delete():
    index = FakeIndex()
    client = PineconeMemoryClient(
        api_key="key",
        index_name="memory-index",
        namespace="ns1",
        fail_open=True,
        index=index,
    )

    count = client.upsert_records([
        {"id": "chunk-1", "values": [1, 2, 3], "metadata": {"source_id": "doc-1"}},
    ])
    matches = client.query([0.1, 0.2, 0.3], top_k=3, filter={"scope": {"$eq": "repo"}})
    deleted = client.delete_by_source(source_kind="file", source_id="doc-1")

    assert count == 1
    assert matches == [{"id": "chunk-1", "score": 0.99, "metadata": {"source_id": "doc-1"}}]
    assert deleted == 1
    assert index.upsert_calls[0]["namespace"] == "ns1"
    assert index.query_calls[0]["top_k"] == 3
    assert index.query_calls[0]["filter"] == {"scope": {"$eq": "repo"}}
    assert index.delete_calls[0]["filter"] == {
        "source_kind": {"$eq": "file"},
        "source_id": {"$eq": "doc-1"},
    }


def test_sdk_exception_is_fail_open_when_enabled(caplog):
    client = PineconeMemoryClient(
        api_key="key",
        index_name="memory-index",
        fail_open=True,
        index=RaisingIndex(),
    )

    with caplog.at_level("WARNING"):
        assert client.upsert_records([{"id": "1", "values": [1], "metadata": {}}]) == 0
        assert client.query([0.1]) == []
        assert client.delete_by_source(source_kind="file", source_id="abc") == 0

    assert "continuing fail-open" in caplog.text


def test_sdk_exception_raises_when_fail_open_disabled():
    client = PineconeMemoryClient(
        api_key="key",
        index_name="memory-index",
        fail_open=False,
        index=RaisingIndex(),
    )

    with pytest.raises(RuntimeError):
        client.query([0.1])
