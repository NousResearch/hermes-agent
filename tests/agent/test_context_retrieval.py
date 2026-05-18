from datetime import datetime, timezone

import pytest

from agent.context_retrieval import (
    ContextRetriever,
    RetrievalRequest,
    build_pinecone_recall,
    build_pinecone_recall_context_block,
    resolve_pinecone_recall_settings,
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


@pytest.fixture(autouse=True)
def pinecone_rollout_env(monkeypatch):
    monkeypatch.setenv("PINECONE_RECALL_ENABLED", "1")
    monkeypatch.setenv("PINECONE_RECALL_PHASE", "3")


def _make_client(matches):
    return PineconeMemoryClient(api_key="key", index_name="idx", index=FakeIndex(matches))


def test_no_config_returns_empty_without_querying():
    retriever = ContextRetriever(
        pinecone=PineconeMemoryClient(api_key="", index_name="", fail_open=True),
        embed_query=RaisingEmbedder(),
    )

    out = retriever.retrieve(RetrievalRequest(query="hello world"))

    assert out == []


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
        RetrievalRequest(query="hello", max_items=3, now=datetime(2026, 5, 17, tzinfo=timezone.utc))
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
        RetrievalRequest(query="repo guidance", now=datetime(2026, 5, 17, tzinfo=timezone.utc))
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


def test_build_pinecone_recall_returns_empty_without_hits():
    recall = build_pinecone_recall(
        "repo guidance",
        scope="repo:hermes-agent",
        platform="cli",
        pinecone=_make_client([]),
        embedder=FakeEmbedder(),
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )

    assert recall == ""


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


def test_long_user_message_keeps_best_candidates_and_respects_top_k():
    client = _make_client(
        [
            {
                "id": "best-profile",
                "score": 0.93,
                "metadata": {
                    "text": "Blas prefers verifying cross-session recall before acting.",
                    "source_kind": "profile",
                    "source_id": "blas-profile",
                    "scope": "repo:hermes-agent",
                    "memory_type": "profile",
                    "updated_at": "2026-05-16T18:00:00+00:00",
                    "freshness_hint": "durable",
                    "confidence": 1.0,
                    "canonical": True,
                },
            },
            {
                "id": "good-file",
                "score": 0.86,
                "metadata": {
                    "text": "Use read_file instead of cat when inspecting repository files.",
                    "source_kind": "file",
                    "source_id": "AGENTS.md",
                    "source_path": "AGENTS.md",
                    "scope": "repo:hermes-agent",
                    "memory_type": "project_context",
                    "updated_at": "2026-05-16T18:00:00+00:00",
                    "freshness_hint": "weekly",
                    "confidence": 0.9,
                    "canonical": True,
                },
            },
            {
                "id": "weak-summary",
                "score": 0.48,
                "metadata": {
                    "text": "A weaker session summary candidate.",
                    "source_kind": "session_summary",
                    "source_id": "sess-weak",
                    "scope": "repo:hermes-agent",
                    "memory_type": "session_summary",
                    "updated_at": "2026-05-16T18:00:00+00:00",
                    "freshness_hint": "daily",
                    "confidence": 0.4,
                    "canonical": False,
                },
            },
        ]
    )

    recall = build_pinecone_recall(
        "Please compare the repo guidance, session history, and profile preferences for this long debugging request. " * 8,
        scope="repo:hermes-agent",
        platform="cli",
        pinecone=client,
        embedder=FakeEmbedder(),
        max_items=2,
        top_k=9,
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )

    fake_index = client._index
    assert fake_index.query_calls[0]["top_k"] == 9
    assert "[blas-profile]" in recall
    assert "[AGENTS.md]" in recall
    assert "weak-summary" not in recall
    assert "A weaker session summary candidate" not in recall


def test_phase_zero_and_one_do_not_enable_prompt_recall(monkeypatch):
    monkeypatch.setenv("PINECONE_RECALL_ENABLED", "1")
    monkeypatch.setenv("PINECONE_RECALL_PHASE", "0")
    assert resolve_pinecone_recall_settings(platform="cli", scope="repo:hermes-agent", max_items=4, min_score=0.35, top_k=None) is None

    phase_zero_client = _make_client([
        {
            "id": "should-not-query",
            "score": 0.9,
            "metadata": {
                "text": "Should stay hidden while rollout is off.",
                "source_kind": "file",
                "source_id": "AGENTS.md",
                "source_path": "AGENTS.md",
                "scope": "repo:hermes-agent",
                "memory_type": "project_context",
                "updated_at": "2026-05-16T18:00:00+00:00",
                "freshness_hint": "weekly",
                "confidence": 0.9,
                "canonical": True,
            },
        }
    ])
    recall = build_pinecone_recall(
        "repo guidance",
        scope="repo:hermes-agent",
        platform="cli",
        pinecone=phase_zero_client,
        embedder=FakeEmbedder(),
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )
    assert recall == ""
    assert phase_zero_client._index.query_calls == []

    monkeypatch.setenv("PINECONE_RECALL_PHASE", "1")
    assert resolve_pinecone_recall_settings(platform="cli", scope="repo:hermes-agent", max_items=4, min_score=0.35, top_k=None) is None

    phase_one_client = _make_client([])
    recall = build_pinecone_recall(
        "repo guidance",
        scope="repo:hermes-agent",
        platform="cli",
        pinecone=phase_one_client,
        embedder=FakeEmbedder(),
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )
    assert recall == ""
    assert phase_one_client._index.query_calls == []


def test_phase_two_limits_snippets_and_honors_allowlists(monkeypatch):
    monkeypatch.setenv("PINECONE_RECALL_ENABLED", "1")
    monkeypatch.setenv("PINECONE_RECALL_PHASE", "2")
    monkeypatch.setenv("PINECONE_RECALL_PLATFORMS", "cli,slack")
    monkeypatch.setenv("PINECONE_RECALL_SCOPES", "repo:hermes-agent")

    settings = resolve_pinecone_recall_settings(
        platform="cli",
        scope="repo:hermes-agent",
        max_items=4,
        min_score=0.35,
        top_k=None,
    )

    assert settings is not None
    assert settings["max_items"] == 2
    assert settings["top_k"] == 6

    blocked_platform = resolve_pinecone_recall_settings(
        platform="discord",
        scope="repo:hermes-agent",
        max_items=4,
        min_score=0.35,
        top_k=None,
    )
    blocked_scope = resolve_pinecone_recall_settings(
        platform="cli",
        scope="repo:other",
        max_items=4,
        min_score=0.35,
        top_k=None,
    )

    assert blocked_platform is None
    assert blocked_scope is None

    blocked_client = _make_client([])
    recall = build_pinecone_recall(
        "repo guidance",
        scope="repo:hermes-agent",
        platform="discord",
        pinecone=blocked_client,
        embedder=FakeEmbedder(),
        now=datetime(2026, 5, 17, tzinfo=timezone.utc),
    )
    assert recall == ""
    assert blocked_client._index.query_calls == []


def test_phase_three_uses_configured_source_types_and_limits(monkeypatch):
    monkeypatch.setenv("PINECONE_RECALL_ENABLED", "1")
    monkeypatch.setenv("PINECONE_RECALL_PHASE", "3")
    monkeypatch.setenv("PINECONE_RECALL_SOURCE_TYPES", "project_context,profile")
    monkeypatch.setenv("PINECONE_RECALL_MAX_ITEMS", "3")
    monkeypatch.setenv("PINECONE_RECALL_TOP_K", "11")

    settings = resolve_pinecone_recall_settings(
        platform="cli",
        scope="repo:hermes-agent",
        max_items=4,
        min_score=0.35,
        top_k=None,
    )

    assert settings is not None
    assert settings["max_items"] == 3
    assert settings["top_k"] == 11
    assert settings["source_types"] == ("project_context", "profile")


def test_invalid_rollout_env_values_fall_back_to_safe_defaults(monkeypatch):
    monkeypatch.setenv("PINECONE_RECALL_ENABLED", "1")
    monkeypatch.setenv("PINECONE_RECALL_PHASE", "3")
    monkeypatch.setenv("PINECONE_RECALL_MAX_ITEMS", "0")
    monkeypatch.setenv("PINECONE_RECALL_TOP_K", "-5")

    settings = resolve_pinecone_recall_settings(
        platform="cli",
        scope="repo:hermes-agent",
        max_items=4,
        min_score=0.35,
        top_k=9,
    )

    assert settings is not None
    assert settings["max_items"] == 4
    assert settings["top_k"] == 9
