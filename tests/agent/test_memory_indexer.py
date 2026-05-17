from __future__ import annotations

from pathlib import Path

from agent.memory_embeddings import MemoryEmbedder
from agent.memory_indexer import MemoryIndexer, build_session_summary_text, extract_links_and_ids
from agent.pinecone_memory import PineconeMemoryClient
from hermes_state import SessionDB


class _FakeEmbeddingsAPI:
    def create(self, *, model, input):
        if isinstance(input, list):
            data = [type("Emb", (), {"embedding": [float(i), float(i) + 0.25]})() for i, _ in enumerate(input)]
        else:
            data = [type("Emb", (), {"embedding": [1.0, 2.0]})()]
        return type("Resp", (), {"data": data})()


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingsAPI()


class _FakeIndex:
    def __init__(self):
        self.upserts = []

    def upsert(self, *, vectors, namespace):
        self.upserts.append({"vectors": vectors, "namespace": namespace})
        return {"upserted_count": len(vectors)}


def _make_embedder(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "agent.memory_embeddings.resolve_provider_client",
        lambda provider, model: (fake_client, model),
    )
    return MemoryEmbedder(provider="openrouter", model="text-embedding-test")


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def test_topic_file_reindex_skips_unchanged_content(monkeypatch, tmp_path):
    db = _make_db(tmp_path)
    embedder = _make_embedder(monkeypatch)
    index = _FakeIndex()
    pinecone = PineconeMemoryClient(api_key="key", index_name="idx", index=index)
    topic_file = tmp_path / "platform-state.md"
    topic_file.write_text("# Platform\n\nCurrent rollout state.")

    indexer = MemoryIndexer(
        pinecone=pinecone,
        embedder=embedder,
        session_db=db,
        topic_files=[topic_file],
        config={"memory": {"pinecone_ingest_topic_files": True}},
    )

    first = indexer.index_topic_files()
    second = indexer.index_topic_files()

    assert first.indexed == 1
    assert first.upserted >= 1
    assert second.indexed == 0
    assert second.skipped == 1
    assert len(index.upserts) == 1


def test_session_summary_index_replaces_on_update(monkeypatch, tmp_path):
    db = _make_db(tmp_path)
    db.create_session(session_id="sess-1", source="cron")
    db.set_session_title("sess-1", "WAT-1374 ingest summaries")
    db.append_message(session_id="sess-1", role="user", content="Investigate WAT-1374 and https://linear.app/watsiai/issue/WAT-1374")
    db.append_message(session_id="sess-1", role="tool", content="done", tool_name="terminal")
    db.end_session("sess-1", end_reason="completed")

    embedder = _make_embedder(monkeypatch)
    index = _FakeIndex()
    pinecone = PineconeMemoryClient(api_key="key", index_name="idx", index=index)
    indexer = MemoryIndexer(
        pinecone=pinecone,
        embedder=embedder,
        session_db=db,
        config={"memory": {"pinecone_ingest_session_summaries": True}},
    )

    first = indexer.index_session_summaries()
    second = indexer.index_session_summaries()
    db.set_session_title("sess-1", "WAT-1374 ingest summaries updated")
    third = indexer.index_session_summaries()

    assert first.indexed == 1
    assert second.skipped == 1
    assert third.indexed == 1
    assert len(index.upserts) == 2
    assert all("Transcript body intentionally omitted" in vec["metadata"]["text"] for call in index.upserts for vec in call["vectors"])


def test_session_summary_ingest_is_noop_when_unconfigured(monkeypatch, tmp_path):
    db = _make_db(tmp_path)
    db.create_session(session_id="sess-2", source="cli")
    db.set_session_title("sess-2", "Ignored summary")
    embedder = _make_embedder(monkeypatch)
    index = _FakeIndex()
    pinecone = PineconeMemoryClient(api_key="key", index_name="idx", index=index)
    indexer = MemoryIndexer(
        pinecone=pinecone,
        embedder=embedder,
        session_db=db,
        config={"memory": {"pinecone_ingest_session_summaries": False}},
    )

    result = indexer.index_session_summaries()

    assert result == result.__class__()
    assert index.upserts == []


def test_extract_links_and_ids_dedupes_and_strips_trailing_punctuation():
    text = "See WAT-1374, WAT-1374, and https://example.com/demo)."
    assert extract_links_and_ids(text) == ["WAT-1374", "https://example.com/demo"]


def test_build_session_summary_text_omits_raw_transcript_body():
    summary = build_session_summary_text(
        title="WAT-1374 ingest summaries",
        tools=["terminal", "terminal", "pytest"],
        links_and_ids=["WAT-1374", "https://linear.app/watsiai/issue/WAT-1374"],
        end_reason="completed",
    )
    assert "WAT-1374 ingest summaries" in summary
    assert "Tools used: pytest, terminal" in summary
    assert "Transcript body intentionally omitted" in summary
