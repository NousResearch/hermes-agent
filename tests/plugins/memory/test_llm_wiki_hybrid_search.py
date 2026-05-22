from types import SimpleNamespace

from hermes_wiki.config import WikiConfig
from hermes_wiki.search import WikiSearch, WikiSearchResult


def _result(page_path, *, score, chunk_index=0, title=None, page_type="concept", text="", tags=None):
    return WikiSearchResult(
        page_path=page_path,
        title=title or page_path.rsplit("/", 1)[-1].removesuffix(".md"),
        page_type=page_type,
        chunk_index=chunk_index,
        text=text,
        score=score,
        tags=tags or [],
    )


class FakeScrollClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.scroll_calls = []

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return [SimpleNamespace(payload=payload) for payload in self.payloads], None


class FakeDenseSearch(WikiSearch):
    def __init__(self, dense_results, lexical_payloads):
        self.config = WikiConfig(wiki_name="test")
        self._dense_results = dense_results
        self._client = FakeScrollClient(lexical_payloads)

    def _dense_search(self, query, *, limit, page_type=None, tags=None, exclude_sources=False):
        return self._dense_results[:limit]


def test_tokenize_for_sparse_search_keeps_paths_ids_and_acronyms():
    from hermes_wiki.search import tokenize_for_sparse_search

    assert tokenize_for_sparse_search("Qwen3-Embedding-8B concepts/foo_bar.md GPT-5.5") == [
        "qwen3",
        "embedding",
        "8b",
        "concepts",
        "foo_bar",
        "md",
        "gpt",
        "5",
        "5",
    ]


def test_reciprocal_rank_fusion_promotes_results_seen_by_both_rankers():
    from hermes_wiki.search import reciprocal_rank_fusion

    dense = [
        _result("concepts/dense-only.md", score=0.99),
        _result("concepts/shared.md", score=0.90),
    ]
    sparse = [
        _result("concepts/shared.md", score=4.0),
        _result("concepts/sparse-only.md", score=3.0),
    ]

    fused = reciprocal_rank_fusion([dense, sparse], limit=3, k=10)

    assert [r.page_path for r in fused] == [
        "concepts/shared.md",
        "concepts/dense-only.md",
        "concepts/sparse-only.md",
    ]
    assert fused[0].score > fused[1].score


def test_sparse_search_scores_literal_query_terms_from_payloads():
    searcher = WikiSearch.__new__(WikiSearch)
    searcher.config = WikiConfig(wiki_name="test")
    searcher._client = FakeScrollClient([
        {
            "page_path": "concepts/random.md",
            "title": "Random",
            "page_type": "concept",
            "chunk_index": 0,
            "text": "semantic discussion without the literal token",
            "tags": [],
        },
        {
            "page_path": "entities/qwen3-embedding-8b.md",
            "title": "Qwen3 Embedding 8B",
            "page_type": "entity",
            "chunk_index": 0,
            "text": "Local embedding endpoint for Qwen3-Embedding-8B",
            "tags": ["embedding"],
        },
    ])

    results = searcher.sparse_search("Qwen3-Embedding-8B", limit=2)

    assert [r.page_path for r in results] == ["entities/qwen3-embedding-8b.md"]
    assert results[0].score > 0
    assert searcher._client.scroll_calls[0]["collection_name"] == "hermes_wiki_test"


def test_sparse_search_honors_filters_and_exclude_sources():
    searcher = WikiSearch.__new__(WikiSearch)
    searcher.config = WikiConfig(wiki_name="test")
    searcher._client = FakeScrollClient([
        {
            "page_path": "raw/articles/qwen.md",
            "title": "Qwen Source",
            "page_type": "source",
            "chunk_index": 0,
            "text": "Qwen3-Embedding-8B",
            "tags": [],
        },
        {
            "page_path": "entities/qwen.md",
            "title": "Qwen Entity",
            "page_type": "entity",
            "chunk_index": 0,
            "text": "Qwen3-Embedding-8B",
            "tags": ["embedding"],
        },
        {
            "page_path": "concepts/qwen.md",
            "title": "Qwen Concept",
            "page_type": "concept",
            "chunk_index": 0,
            "text": "Qwen3-Embedding-8B",
            "tags": ["embedding"],
        },
    ])

    results = searcher.sparse_search(
        "Qwen3-Embedding-8B",
        limit=5,
        page_type="entity",
        tags=["embedding"],
        exclude_sources=True,
    )

    assert [r.page_path for r in results] == ["entities/qwen.md"]


def test_hybrid_search_fuses_dense_and_sparse_results():
    searcher = FakeDenseSearch(
        dense_results=[
            _result("concepts/semantic.md", score=0.99),
            _result("entities/qwen3-embedding-8b.md", score=0.60),
        ],
        lexical_payloads=[
            {
                "page_path": "entities/qwen3-embedding-8b.md",
                "title": "Qwen3 Embedding 8B",
                "page_type": "entity",
                "chunk_index": 0,
                "text": "Qwen3-Embedding-8B literal config key",
                "tags": ["embedding"],
            },
            {
                "page_path": "concepts/path-only.md",
                "title": "Path Only",
                "page_type": "concept",
                "chunk_index": 0,
                "text": "Qwen3-Embedding-8B appears here too",
                "tags": [],
            },
        ],
    )

    results = searcher.search("Qwen3-Embedding-8B", limit=3, search_mode="hybrid")

    assert [r.page_path for r in results] == [
        "entities/qwen3-embedding-8b.md",
        "concepts/semantic.md",
        "concepts/path-only.md",
    ]


def test_search_defaults_to_dense_mode_for_backward_compatibility():
    searcher = FakeDenseSearch(
        dense_results=[_result("concepts/semantic.md", score=0.99)],
        lexical_payloads=[
            {
                "page_path": "entities/qwen3-embedding-8b.md",
                "title": "Qwen3 Embedding 8B",
                "page_type": "entity",
                "chunk_index": 0,
                "text": "Qwen3-Embedding-8B",
                "tags": [],
            }
        ],
    )

    results = searcher.search("Qwen3-Embedding-8B", limit=2)

    assert [r.page_path for r in results] == ["concepts/semantic.md"]
    assert searcher._client.scroll_calls == []


def test_search_supports_sparse_only_mode():
    searcher = FakeDenseSearch(
        dense_results=[_result("concepts/semantic.md", score=0.99)],
        lexical_payloads=[
            {
                "page_path": "entities/qwen3-embedding-8b.md",
                "title": "Qwen3 Embedding 8B",
                "page_type": "entity",
                "chunk_index": 0,
                "text": "Qwen3-Embedding-8B",
                "tags": [],
            }
        ],
    )

    results = searcher.search("Qwen3-Embedding-8B", limit=2, search_mode="sparse")

    assert [r.page_path for r in results] == ["entities/qwen3-embedding-8b.md"]
