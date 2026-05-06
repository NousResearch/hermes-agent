import json

import pytest

from agent.research_search import classify_topic_type, generate_query_plan
from agent.research_search.store import (
    ResearchSearchStore,
    ResearchSearchUnavailableError,
    chunk_text,
    duckdb_available,
)


class _UnavailableStore:
    def connect(self):
        raise ResearchSearchUnavailableError("DuckDB is not installed")


class _RecordingStore:
    def __init__(self):
        self.documents = []
        self.runs = []

    def connect(self):
        return self

    def upsert_document(self, document):
        self.documents.append(document)
        return {"url": document["url"], "id": "doc-1", "status": document["status"]}

    def record_run(self, run):
        self.runs.append(run)


def test_classify_topic_type_current_events():
    assert (
        classify_topic_type("latest Detroit Pistons current starting five")
        == "current_events"
    )


def test_generate_query_plan_adds_technical_source_packs():
    plan = generate_query_plan(
        "Hermes Agent API bug", topic_type="technical", depth="thorough"
    )
    queries = [item["query"] for item in plan["queries"]]

    assert plan["topic_type"] == "technical"
    assert any("official" in query for query in queries)
    assert any("documentation changelog" in query for query in queries)
    assert any("GitHub issues" in query for query in queries)
    assert "official" in plan["source_requirements"]
    assert "adversarial" in plan["source_requirements"]


def test_research_gather_returns_evidence_bundle_without_duckdb(monkeypatch):
    from agent.research_search import orchestrator

    calls = []

    def fake_dispatch(name, args):
        calls.append((name, args))
        if name == "web_search":
            return json.dumps(
                {
                    "data": {
                        "web": [
                            {
                                "url": "https://www.nba.com/pistons/roster",
                                "title": "Detroit Pistons roster",
                                "description": "Official roster and lineup notes.",
                            }
                        ]
                    }
                }
            )
        if name == "web_extract":
            return json.dumps(
                {
                    "results": [
                        {
                            "url": args["urls"][0],
                            "title": "Detroit Pistons roster",
                            "content": "Cade Cunningham and Jalen Duren are listed starters.",
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected tool dispatch: {name}")

    monkeypatch.setattr(
        orchestrator,
        "research_local_search",
        lambda *a, **k: {
            "success": False,
            "error": "DuckDB is not installed",
            "results": [],
        },
    )
    monkeypatch.setattr(orchestrator, "_store", lambda *_a, **_k: _UnavailableStore())
    monkeypatch.setattr(orchestrator, "dispatch_tool", fake_dispatch)

    bundle = orchestrator.research_gather(
        "current Detroit Pistons starting five",
        max_queries=1,
        max_pages=1,
        config={"research_search": {"browser_fallback": False}},
    )

    assert bundle["success"] is True
    assert bundle["sources"][0]["status"] == "extracted"
    assert bundle["sources"][0]["url"] == "https://www.nba.com/pistons/roster"
    assert "Cade Cunningham" in bundle["sources"][0]["content"]
    assert bundle["usage"]["search_calls"] == 1
    assert bundle["usage"]["extract_calls"] == 1
    assert bundle["usage"]["browser_fallbacks"] == 0
    assert bundle["usage"]["indexed_documents"] == 0
    assert bundle["usage"]["indexed_chunks"] == 0
    assert bundle["usage"]["indexed_embeddings"] == 0
    assert bundle["usage"]["indexed_evidence"] == 0
    assert bundle["usage"]["gap_passes"] == 0
    assert [name for name, _ in calls] == ["web_search", "web_extract"]


def test_research_gather_records_malformed_tool_results_as_errors(monkeypatch):
    from agent.research_search import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "research_local_search",
        lambda *a, **k: {"success": False, "error": "no local index", "results": []},
    )
    monkeypatch.setattr(orchestrator, "_store", lambda *_a, **_k: _UnavailableStore())
    monkeypatch.setattr(orchestrator, "dispatch_tool", lambda *_a, **_k: "not json")

    bundle = orchestrator.research_gather(
        "obscure exact phrase lookup",
        max_queries=1,
        max_pages=2,
        config={"research_search": {"browser_fallback": False}},
    )

    assert bundle["success"] is True
    assert bundle["sources"] == []
    assert any("Malformed JSON result" in error for error in bundle["errors"])
    assert "No sources were gathered." in bundle["gaps"]


def test_research_gather_auto_indexes_when_config_enabled(monkeypatch):
    from agent.research_search import orchestrator

    store = _RecordingStore()

    monkeypatch.setattr(
        orchestrator,
        "research_local_search",
        lambda *a, **k: {"success": False, "error": "empty local index", "results": []},
    )
    monkeypatch.setattr(orchestrator, "_store", lambda *_a, **_k: store)
    monkeypatch.setattr(
        orchestrator,
        "dispatch_tool",
        lambda name, args: json.dumps(
            {
                "data": {
                    "web": [
                        {
                            "url": "https://example.com/source",
                            "title": "Source",
                            "description": "Search snippet.",
                        }
                    ]
                }
            }
        )
        if name == "web_search"
        else json.dumps(
            {
                "results": [
                    {
                        "url": args["urls"][0],
                        "title": "Source",
                        "content": "Extracted evidence.",
                    }
                ]
            }
        ),
    )

    bundle = orchestrator.research_gather(
        "current fact",
        max_queries=1,
        max_pages=1,
        config={
            "research_search": {
                "browser_fallback": False,
                "auto_index_research_results": True,
            }
        },
    )

    assert bundle["usage"]["indexed_documents"] == 1
    assert store.documents[0]["url"] == "https://example.com/source"
    assert store.runs[0]["question"] == "current fact"


def test_research_gather_does_not_auto_index_when_config_disabled(monkeypatch):
    from agent.research_search import orchestrator

    store = _RecordingStore()

    monkeypatch.setattr(
        orchestrator,
        "research_local_search",
        lambda *a, **k: {"success": False, "error": "empty local index", "results": []},
    )
    monkeypatch.setattr(orchestrator, "_store", lambda *_a, **_k: store)
    monkeypatch.setattr(
        orchestrator,
        "dispatch_tool",
        lambda name, args: json.dumps(
            {
                "data": {
                    "web": [
                        {
                            "url": "https://example.com/source",
                            "title": "Source",
                            "description": "Search snippet.",
                        }
                    ]
                }
            }
        )
        if name == "web_search"
        else json.dumps(
            {
                "results": [
                    {
                        "url": args["urls"][0],
                        "title": "Source",
                        "content": "Extracted evidence.",
                    }
                ]
            }
        ),
    )

    bundle = orchestrator.research_gather(
        "current fact",
        max_queries=1,
        max_pages=1,
        config={
            "research_search": {
                "browser_fallback": False,
                "auto_index_research_results": False,
            }
        },
    )

    assert bundle["usage"]["indexed_documents"] == 0
    assert store.documents == []
    assert store.runs == []


def test_chunk_text_produces_overlapping_chunks():
    chunks = chunk_text(" ".join(str(i) for i in range(300)), chunk_chars=120, overlap_chars=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 320 for chunk in chunks)
    assert chunks[0] != chunks[1]


@pytest.mark.skipif(not duckdb_available(), reason="duckdb extra is optional")
def test_store_tracks_chunks_evidence_and_embeddings(tmp_path):
    store = ResearchSearchStore(tmp_path / "research.duckdb")
    doc = store.upsert_document(
        {
            "url": "https://example.com/research",
            "title": "Research",
            "content": "alpha beta gamma " * 200,
            "vertical": "web",
            "source_type": "docs",
            "status": "extracted",
        }
    )
    chunks = store.upsert_chunks(doc["id"], "alpha beta gamma " * 200, chunk_chars=200, overlap_chars=40)
    store.upsert_evidence(
        {
            "document_id": doc["id"],
            "chunk_id": chunks[0]["id"],
            "query": "alpha",
            "claim": "Research",
            "excerpt": chunks[0]["text"],
            "relevance_score": 0.9,
            "source_quality_score": 0.8,
            "confidence": 0.7,
        }
    )
    store.upsert_embedding(chunks[0]["id"], "test", "model", b"\x00" * 8, 2)

    status = store.status()

    assert status["documents"] == 1
    assert status["chunks"] == len(chunks)
    assert status["evidence"] == 1
    assert status["embeddings"] == 1


def test_research_gap_analyze_reports_missing_official_source():
    from agent.research_search import research_gap_analyze

    result = research_gap_analyze(
        "product safety",
        sources=[
            {
                "url": "https://forum.example.com/thread",
                "status": "extracted",
                "source_type": "community",
            }
        ],
        plan={"source_requirements": ["official", "adversarial"], "topic_type": "product"},
    )

    assert result["success"] is True
    assert "No official or primary source was gathered." in result["gaps"]
    assert result["next_queries"]


def test_research_status_includes_web_and_vector_status(monkeypatch):
    from agent.research_search import orchestrator

    class _StatusStore:
        def status(self):
            return {
                "success": True,
                "duckdb_available": True,
                "fts_available": True,
                "documents": 0,
                "chunks": 0,
                "evidence": 0,
                "embeddings": 0,
            }

    monkeypatch.setattr(orchestrator, "_store", lambda *_a, **_k: _StatusStore())
    monkeypatch.setattr(
        orchestrator,
        "_web_backend_status",
        lambda: {"active_backend": "searxng", "searxng": {"reachable": True}},
    )

    status = orchestrator.research_status(
        config={"research_search": {"auto_index_research_results": True}}
    )

    assert status["web"]["active_backend"] == "searxng"
    assert "vector" in status
    assert status["auto_index_research_results"] is True
