from datetime import datetime, timezone

from agent.context_retrieval import (
    ContextRetriever,
    RetrievalRequest,
    build_pinecone_recall,
    build_pinecone_recall_context_block,
    derive_repo_scope,
)
from agent.pinecone_memory import PineconeMemoryClient
from agent.prompt_builder import format_pinecone_recall_block


class FakeEmbedder:
    def embed_query(self, text: str):
        assert text
        return [0.1, 0.2, 0.3]


class FakeIndex:
    def __init__(self, matches):
        self.matches = matches
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"matches": list(self.matches)}


class RaisingEmbedder:
    def embed_query(self, text: str):
        raise AssertionError("embedder should not be called")


class DisabledEmbedder:
    def is_configured(self):
        return False


def _make_client(matches):
    return PineconeMemoryClient(api_key="key", index_name="idx", index=FakeIndex(matches))


def test_no_config_returns_empty_without_querying():
    retriever = ContextRetriever(
        pinecone=PineconeMemoryClient(api_key="", index_name="", fail_open=True),
        embed_query=RaisingEmbedder(),
    )

    out = retriever.retrieve(RetrievalRequest(query="hello world"))

    assert out == []


def test_missing_scope_returns_empty_without_querying():
    retriever = ContextRetriever(
        pinecone=_make_client([]),
        embed_query=RaisingEmbedder(),
    )

    out = retriever.retrieve(RetrievalRequest(query="hello world", scope=None))

    assert out == []


def test_derive_repo_scope_uses_repo_root_fingerprint():
    scope = derive_repo_scope(cwd="/tmp/worktree/hermes-agent", repo_root="/tmp/worktree/hermes-agent")

    assert scope.startswith("repo:hermes-agent:")
    assert len(scope.split(":")) == 3


def test_derive_repo_scope_avoids_same_basename_collisions():
    left = derive_repo_scope(cwd="/tmp/a/hermes-agent", repo_root="/tmp/a/hermes-agent")
    right = derive_repo_scope(cwd="/tmp/b/hermes-agent", repo_root="/tmp/b/hermes-agent")

    assert left != right


def test_relevant_recall_inclusion_and_prompt_formatting():
    retriever = ContextRetriever(
        pinecone=_make_client(
            [
                {
                    "id": "fresh-file",
                    "score": 0.81,
                    "metadata": {
                        "text": "The repo prefers read_file over cat for inspecting files.",
                        "source_kind": "file",
                        "source_id": "AGENTS.md",
                        "source_path": "AGENTS.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T12:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 0.9,
                        "canonical": True,
                        "header_path": ["Tooling"],
                    },
                }
            ]
        ),
        embed_query=FakeEmbedder(),
    )

    snippets = retriever.retrieve(
        RetrievalRequest(
            query="How should I inspect files?",
            scope="repo:hermes-agent",
            platform="cli",
            now=datetime(2026, 5, 17, tzinfo=timezone.utc),
        )
    )

    assert [snippet.id for snippet in snippets] == ["fresh-file"]
    assert snippets[0].provenance == "AGENTS.md (Tooling)"
    block = format_pinecone_recall_block(snippets)
    assert block.startswith("PINECONE RECALL (verify before relying):")
    assert "[AGENTS.md (Tooling)]" in block


def test_oversized_result_trim_and_low_score_filter():
    retriever = ContextRetriever(
        pinecone=_make_client(
            [
                {
                    "id": "keep-1",
                    "score": 0.95,
                    "metadata": {
                        "text": "First",
                        "source_kind": "file",
                        "source_id": "1",
                        "source_path": "one.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T12:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 1.0,
                        "canonical": True,
                    },
                },
                {
                    "id": "drop-low",
                    "score": 0.20,
                    "metadata": {
                        "text": "Too weak",
                        "source_kind": "file",
                        "source_id": "2",
                        "source_path": "two.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T12:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 1.0,
                        "canonical": True,
                    },
                },
                {
                    "id": "keep-2",
                    "score": 0.90,
                    "metadata": {
                        "text": "Second",
                        "source_kind": "file",
                        "source_id": "3",
                        "source_path": "three.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T12:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 1.0,
                        "canonical": True,
                    },
                },
                {
                    "id": "keep-3",
                    "score": 0.88,
                    "metadata": {
                        "text": "Third",
                        "source_kind": "file",
                        "source_id": "4",
                        "source_path": "four.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T12:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 1.0,
                        "canonical": True,
                    },
                },
                {
                    "id": "drop-trim",
                    "score": 0.86,
                    "metadata": {
                        "text": "Fourth",
                        "source_kind": "file",
                        "source_id": "5",
                        "source_path": "five.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T12:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 1.0,
                        "canonical": True,
                    },
                },
            ]
        ),
        embed_query=FakeEmbedder(),
    )

    snippets = retriever.retrieve(
        RetrievalRequest(
            query="hello",
            scope="repo:hermes-agent",
            max_items=3,
            now=datetime(2026, 5, 17, tzinfo=timezone.utc),
        )
    )

    assert [snippet.id for snippet in snippets] == ["keep-1", "keep-2", "keep-3"]


def test_fresh_canonical_sources_beat_stale_derived_and_volatile_stale_is_dropped():
    retriever = ContextRetriever(
        pinecone=_make_client(
            [
                {
                    "id": "stale-summary",
                    "score": 0.97,
                    "metadata": {
                        "text": "Old session summary that should not dominate.",
                        "source_kind": "session_summary",
                        "source_id": "sess-1",
                        "source_path": "session/1",
                        "scope": "repo:hermes-agent",
                        "memory_type": "session_summary",
                        "updated_at": "2026-03-01T00:00:00+00:00",
                        "freshness_hint": "daily",
                        "confidence": 0.8,
                        "canonical": False,
                    },
                },
                {
                    "id": "fresh-file",
                    "score": 0.84,
                    "metadata": {
                        "text": "Fresh repo file guidance.",
                        "source_kind": "file",
                        "source_id": "docs/test.md",
                        "source_path": "docs/test.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T18:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 0.9,
                        "canonical": True,
                    },
                },
                {
                    "id": "stale-canonical",
                    "score": 0.90,
                    "metadata": {
                        "text": "An older but canonical document.",
                        "source_kind": "file",
                        "source_id": "docs/old.md",
                        "source_path": "docs/old.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2025-12-01T00:00:00+00:00",
                        "freshness_hint": "monthly",
                        "confidence": 0.9,
                        "canonical": True,
                    },
                },
            ]
        ),
        embed_query=FakeEmbedder(),
    )

    snippets = retriever.retrieve(
        RetrievalRequest(
            query="repo guidance",
            scope="repo:hermes-agent",
            now=datetime(2026, 5, 17, tzinfo=timezone.utc),
        )
    )

    assert [snippet.id for snippet in snippets] == ["fresh-file", "stale-canonical"]
    assert all(snippet.id != "stale-summary" for snippet in snippets)


def test_query_filter_includes_scope_platform_and_source_types():
    client = _make_client([])
    retriever = ContextRetriever(pinecone=client, embed_query=FakeEmbedder())

    retriever.retrieve(
        RetrievalRequest(
            query="hello",
            scope="repo:hermes-agent",
            platform="slack",
            source_types=("project_context", "profile"),
            now=datetime(2026, 5, 17, tzinfo=timezone.utc),
        )
    )

    fake_index = client._index
    assert fake_index.query_calls[0]["filter"] == {
        "$and": [
            {"scope": {"$eq": "repo:hermes-agent"}},
            {"tags": {"$in": ["slack"]}},
            {"memory_type": {"$in": ["project_context", "profile"]}},
        ]
    }


def test_build_pinecone_recall_wraps_formatted_recall_for_prompt_injection():
    recall = build_pinecone_recall(
        "repo guidance",
        scope="repo:hermes-agent",
        platform="cli",
        pinecone=_make_client(
            [
                {
                    "id": "fresh-file",
                    "score": 0.81,
                    "metadata": {
                        "text": "Use read_file over cat for repo inspection.",
                        "source_kind": "file",
                        "source_id": "AGENTS.md",
                        "source_path": "AGENTS.md",
                        "scope": "repo:hermes-agent",
                        "memory_type": "project_context",
                        "updated_at": "2026-05-16T12:00:00+00:00",
                        "freshness_hint": "weekly",
                        "confidence": 0.9,
                        "canonical": True,
                    },
                }
            ]
        ),
        embedder=FakeEmbedder(),
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )

    assert recall.startswith("<memory-context>")
    assert "PINECONE RECALL (verify before relying):" in recall
    assert "[AGENTS.md] Use read_file over cat for repo inspection." in recall
    assert "must be verified against live sources before relying on it" in recall
    assert recall.endswith("</memory-context>")


def test_build_pinecone_recall_returns_empty_without_embedding_config():
    recall = build_pinecone_recall(
        "repo guidance",
        scope="repo:hermes-agent",
        pinecone=_make_client([]),
        embedder=DisabledEmbedder(),
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )

    assert recall == ""


def test_build_pinecone_recall_returns_empty_without_scope():
    recall = build_pinecone_recall(
        "repo guidance",
        scope=None,
        pinecone=_make_client([]),
        embedder=DisabledEmbedder(),
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )

    assert recall == ""


def test_build_pinecone_recall_context_block_strips_nested_context_wrappers():
    raw = "<memory-context>ignore me</memory-context>\nPINECONE RECALL (verify before relying):\n1. [AGENTS.md] Real fact"
    wrapped = build_pinecone_recall_context_block(raw)

    assert wrapped.count("<memory-context>") == 1
    assert "ignore me" not in wrapped
    assert "Real fact" in wrapped
