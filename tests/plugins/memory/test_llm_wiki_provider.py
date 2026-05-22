"""Tests for the LLM Wiki memory provider."""

from __future__ import annotations

import json
import tomllib

import pytest
import yaml

from hermes_wiki.config import WikiConfig
from hermes_wiki.engine import WikiEngine
from hermes_wiki.eval import (
    RetrievalEvalCase,
    _build_searcher,
    evaluate_retrieval,
    load_retrieval_cases,
    main as eval_main,
    result_to_dict,
)
from hermes_wiki.frontmatter import parse_frontmatter, write_page
from hermes_wiki.indexer import WikiIndexer
from hermes_wiki.search import WikiSearch
from agent.memory_manager import MemoryManager
from plugins.memory import load_memory_provider
from plugins.memory.llm_wiki import LLMWikiMemoryProvider, register


class DummyPluginContext:
    def __init__(self) -> None:
        self.registered = []

    def register_memory_provider(self, provider) -> None:
        self.registered.append(provider)


def test_provider_name():
    provider = LLMWikiMemoryProvider()

    assert provider.name == "llm_wiki"


class FakeRetrievalSearcher:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        return self.results_by_query.get(query, [])


def test_retrieval_eval_passes_when_expected_page_is_in_top_k():
    searcher = FakeRetrievalSearcher(
        {
            "How autonomous should Hermes be?": [
                {"page_path": "entities/hermes.md", "title": "Hermes", "score": 0.8},
                {"page_path": "concepts/user-autonomy-operating-policy.md", "title": "Policy", "score": 0.7},
            ],
        }
    )

    results = evaluate_retrieval(
        searcher,
        [
            RetrievalEvalCase(
                query="How autonomous should Hermes be?",
                expected_pages={"concepts/user-autonomy-operating-policy.md"},
                top_k=2,
            )
        ],
    )

    assert results.passed is True
    assert results.total == 1
    assert results.failures == []
    assert searcher.calls == [("How autonomous should Hermes be?", 2)]


def test_retrieval_eval_reports_missing_expected_pages():
    searcher = FakeRetrievalSearcher(
        {
            "What are the memory boundaries?": [
                {"page_path": "entities/hermes.md", "title": "Hermes", "score": 0.8},
            ],
        }
    )

    results = evaluate_retrieval(
        searcher,
        [
            RetrievalEvalCase(
                query="What are the memory boundaries?",
                expected_pages={"concepts/context-injection-policy.md", "concepts/manual-curated-ingestion.md"},
                top_k=3,
            )
        ],
    )

    assert results.passed is False
    assert results.total == 1
    assert results.failures[0].query == "What are the memory boundaries?"
    assert results.failures[0].missing_pages == {
        "concepts/context-injection-policy.md",
        "concepts/manual-curated-ingestion.md",
    }
    assert results.failures[0].retrieved_pages == ["entities/hermes.md"]


def test_retrieval_eval_accepts_object_search_results():
    class Result:
        page_path = "concepts/context-injection-policy.md"
        title = "Context Injection Policy"
        score = 0.9

    searcher = FakeRetrievalSearcher({"How should memory context be injected?": [Result()]})

    results = evaluate_retrieval(
        searcher,
        [
            RetrievalEvalCase(
                query="How should memory context be injected?",
                expected_pages={"concepts/context-injection-policy.md"},
            )
        ],
    )

    assert results.passed is True
    assert results.cases[0].retrieved_pages == ["concepts/context-injection-policy.md"]


def test_retrieval_eval_deduplicates_retrieved_pages_preserving_order():
    searcher = FakeRetrievalSearcher(
        {
            "What did we decide about memory?": [
                {"page_path": "concepts/memory-log-wiki-boundary.md"},
                {"page_path": "concepts/memory-log-wiki-boundary.md"},
                {"page_path": "concepts/manual-curated-ingestion.md"},
            ],
        }
    )

    results = evaluate_retrieval(
        searcher,
        [
            RetrievalEvalCase(
                query="What did we decide about memory?",
                expected_pages={"concepts/manual-curated-ingestion.md"},
            )
        ],
    )

    assert results.cases[0].retrieved_pages == [
        "concepts/memory-log-wiki-boundary.md",
        "concepts/manual-curated-ingestion.md",
    ]


def test_load_retrieval_cases_from_yaml(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        """
cases:
  - query: How autonomous should Hermes be?
    expected_pages:
      - concepts/user-autonomy-operating-policy.md
      - entities/hermes.md
    top_k: 7
""".strip(),
        encoding="utf-8",
    )

    cases = load_retrieval_cases(cases_path)

    assert cases == [
        RetrievalEvalCase(
            query="How autonomous should Hermes be?",
            expected_pages={"concepts/user-autonomy-operating-policy.md", "entities/hermes.md"},
            top_k=7,
        )
    ]


def test_load_retrieval_cases_from_json_list(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "query": "What should Hermes call the example user?",
                    "expected_pages": "entities/example-user.md",
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_retrieval_cases(cases_path)

    assert cases == [
        RetrievalEvalCase(
            query="What should Hermes call the example user?",
            expected_pages={"entities/example-user.md"},
            top_k=5,
        )
    ]


def test_load_retrieval_cases_rejects_unsafe_entries(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text("cases:\n  - expected_pages: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="query"):
        load_retrieval_cases(cases_path)


def test_load_retrieval_cases_rejects_traversal_expected_pages(tmp_path):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        "cases:\n  - query: unsafe\n    expected_pages:\n      - ../outside.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative wiki page paths"):
        load_retrieval_cases(cases_path)


def test_result_to_dict_is_json_serializable():
    result = evaluate_retrieval(
        FakeRetrievalSearcher({"query": [{"page_path": "concepts/a.md"}]}),
        [RetrievalEvalCase(query="query", expected_pages={"concepts/a.md"})],
    )

    payload = result_to_dict(result)

    assert payload == {
        "passed": True,
        "total": 1,
        "failures": 0,
        "cases": [
            {
                "query": "query",
                "passed": True,
                "expected_pages": ["concepts/a.md"],
                "retrieved_pages": ["concepts/a.md"],
                "missing_pages": [],
            }
        ],
    }
    json.dumps(payload)


def test_retrieval_eval_main_returns_nonzero_on_failure(tmp_path, monkeypatch, capsys):
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        "cases:\n  - query: missing\n    expected_pages:\n      - concepts/missing.md\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_wiki.eval._build_searcher", lambda config_path=None: FakeRetrievalSearcher({"missing": []}))

    code = eval_main([str(cases_path), "--pretty"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["passed"] is False
    assert payload["cases"][0]["missing_pages"] == ["concepts/missing.md"]


def test_retrieval_eval_main_returns_zero_on_success(tmp_path, monkeypatch, capsys):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps([{"query": "ok", "expected_pages": ["concepts/ok.md"]}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_wiki.eval._build_searcher",
        lambda config_path=None: FakeRetrievalSearcher({"ok": [{"page_path": "concepts/ok.md"}]}),
    )

    code = eval_main([str(cases_path)])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["passed"] is True
    assert payload["total"] == 1


def test_retrieval_eval_build_searcher_rejects_missing_explicit_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="Hermes config not found"):
        _build_searcher(str(tmp_path / "missing.yaml"))


def test_retrieval_eval_build_searcher_rejects_explicit_config_without_wiki(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("memory:\n  provider: llm_wiki\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no wiki section"):
        _build_searcher(str(config_path))


def test_retrieval_eval_build_searcher_rejects_malformed_wiki_config_sections(tmp_path):
    for config_text, message in [
        ("wiki: true\n", "wiki config section"),
        ("wiki:\n  embedding: true\n", "wiki.embedding"),
        ("wiki:\n  vector_store: true\n", "wiki.vector_store"),
        ("wiki:\n  llm: true\n", "wiki.llm"),
    ]:
        config_path = tmp_path / f"config-{abs(hash(config_text))}.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            _build_searcher(str(config_path))


def test_frontmatter_requires_delimiter_on_own_line():
    text = "---\ntitle: Demo\n--- appears in the body, not as a delimiter\n"

    fm, body = parse_frontmatter(text)

    assert fm == {}
    assert body == text.strip()


def test_write_page_replaces_existing_file_atomically(tmp_path):
    page = tmp_path / "concepts" / "memory.md"
    write_page(page, {"title": "Old"}, "old body")

    write_page(page, {"title": "New"}, "new body")

    text = page.read_text(encoding="utf-8")
    assert "title: New" in text
    assert "new body" in text
    assert not list(page.parent.glob(".memory.md.*.tmp"))


def test_log_rotation_preserves_existing_archive(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", log_rotation_threshold=1)
    config.ensure_dirs()
    config.log_path.write_text("# Wiki Log\n\n## [2026-01-01] old | entry\n", encoding="utf-8")
    existing_archive = config.wiki_path / "log-2026.md"
    existing_archive.write_text("older archived logs\n", encoding="utf-8")
    indexer = WikiIndexer(config)

    indexer.append_log("test", "new entry")

    archives = sorted(config.wiki_path.glob("log-2026*.md"))
    assert existing_archive in archives
    assert existing_archive.read_text(encoding="utf-8") == "older archived logs\n"
    assert len(archives) == 2


def test_search_index_page_embeds_before_deleting_existing_chunks(tmp_path):
    class FailingEmbedder:
        def embed_batch(self, texts):
            raise RuntimeError("embedding down")

    search = WikiSearch.__new__(WikiSearch)
    search.config = WikiConfig(wiki_path=tmp_path / "wiki")
    search.config.ensure_dirs()
    page = search.config.concepts_dir / "memory.md"
    write_page(page, {"title": "Memory", "type": "concept"}, "Body text")
    search._embedder = FailingEmbedder()
    search._delete_page_chunks = lambda rel_path: (_ for _ in ()).throw(AssertionError("deleted before embed"))

    try:
        search.index_page(page)
    except RuntimeError as exc:
        assert "embedding down" in str(exc)
    else:
        raise AssertionError("expected embedding failure")


def test_search_index_page_upserts_before_deleting_stale_chunks(tmp_path):
    class Embedder:
        def embed_batch(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    class Client:
        def upsert(self, *args, **kwargs):
            raise RuntimeError("qdrant down")

    search = WikiSearch.__new__(WikiSearch)
    search.config = WikiConfig(wiki_path=tmp_path / "wiki", embedding_dim=4)
    search.config.ensure_dirs()
    page = search.config.concepts_dir / "memory.md"
    write_page(page, {"title": "Memory", "type": "concept"}, "Body text")
    search._embedder = Embedder()
    search._client = Client()
    search._point = lambda point_id, vector, payload: {"id": point_id, "payload": payload}
    search._delete_stale_page_chunks = lambda rel_path, keep_chunks: (_ for _ in ()).throw(
        AssertionError("deleted stale chunks before successful upsert")
    )

    try:
        search.index_page(page)
    except RuntimeError as exc:
        assert "qdrant down" in str(exc)
    else:
        raise AssertionError("expected upsert failure")


def test_reindex_all_does_not_delete_collection_first(tmp_path):
    class FakeClient:
        def delete_collection(self, name):
            raise AssertionError("collection should not be deleted before safe reindex")

        def scroll(self, **kwargs):
            return [], None

    search = WikiSearch.__new__(WikiSearch)
    search.config = WikiConfig(wiki_path=tmp_path / "wiki")
    search.config.ensure_dirs()
    search._client = FakeClient()
    search._ensure_collection = lambda: None
    search.index_page = lambda path: 1
    search.index_source = lambda path: 1
    search._existing_indexed_paths = lambda: set()
    search._delete_page_chunks = lambda rel_path: None
    write_page(search.config.concepts_dir / "memory.md", {"title": "Memory", "type": "concept"}, "Body text")

    counts = search.reindex_all()

    assert counts == {"pages": 1, "sources": 0, "chunks": 1}


def test_engine_rejects_unsafe_llm_slugs(tmp_path):
    engine = WikiEngine.__new__(WikiEngine)

    assert engine._safe_slug("Memory Policy") == "memory-policy"
    assert engine._safe_slug("../escape") is None
    assert engine._safe_slug("/tmp/escape") is None
    assert engine._safe_slug("safe/escape") is None


def test_engine_safe_child_path_blocks_traversal(tmp_path):
    engine = WikiEngine.__new__(WikiEngine)
    root = tmp_path / "wiki" / "entities"
    root.mkdir(parents=True)

    assert engine._safe_child_path(root, "memory").parent == root.resolve()
    assert engine._safe_child_path(root, "../escape") is None


def test_engine_ingest_text_rejects_unknown_source_type(tmp_path):
    engine = WikiEngine.__new__(WikiEngine)
    engine.config = WikiConfig(wiki_path=tmp_path / "wiki")

    try:
        engine.ingest_text("content", "Example", source_type="../../escape", dry_run=True)
    except ValueError as exc:
        assert "Unsupported source_type" in str(exc)
    else:
        raise AssertionError("unsafe source_type should be rejected before path construction")


def test_engine_read_only_constructor_does_not_create_storage(monkeypatch, tmp_path):
    calls = []

    class FakeSearch:
        def __init__(self, config, *, ensure_collection=True, read_only=False):
            calls.append(("search", ensure_collection, read_only))

    class FakeLLM:
        def __init__(self, config):
            calls.append(("llm", None))

    class FakeIndexer:
        def __init__(self, config):
            calls.append(("indexer", None))

    monkeypatch.setattr("hermes_wiki.engine.WikiSearch", FakeSearch)
    monkeypatch.setattr("hermes_wiki.engine.WikiLLM", FakeLLM)
    monkeypatch.setattr("hermes_wiki.engine.WikiIndexer", FakeIndexer)
    config = WikiConfig(wiki_path=tmp_path / "wiki")

    engine = WikiEngine(config, read_only=True)

    assert engine.read_only is True
    assert not config.wiki_path.exists()
    assert ("search", False, True) in calls


def test_plugin_metadata_exists():
    metadata_path = "plugins/memory/llm_wiki/plugin.yaml"

    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    assert metadata["name"] == "llm_wiki"
    assert "memory" in metadata["description"].lower()


def test_plugin_metadata_is_included_as_package_data():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert "plugins" in package_data
    assert "**/plugin.yaml" in package_data["plugins"]


def test_retrieval_eval_console_script_is_registered():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    assert pyproject["project"]["scripts"]["hermes-wiki-eval"] == "hermes_wiki.eval:main"


def test_config_schema_exposes_local_wiki_settings():
    provider = LLMWikiMemoryProvider()

    fields = {field["key"]: field for field in provider.get_config_schema()}

    assert {"path", "name", "embedding_url", "embedding_model", "embedding_dim", "qdrant_url"} <= set(fields)
    assert fields["path"]["default"] == "~/.hermes/wiki/personal"
    assert fields["embedding_url"]["default"] == "http://localhost:22222"


def test_save_config_writes_wiki_section_without_clobbering_existing_config(tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  default: gpt-test\n", encoding="utf-8")
    provider = LLMWikiMemoryProvider()

    provider.save_config(
        {
            "path": str(tmp_path / "wiki" / "memory"),
            "name": "memory",
            "embedding_url": "http://embeddings.local",
            "embedding_model": "embed-test",
            "embedding_dim": "1024",
            "qdrant_url": "http://qdrant.local",
            "collection_prefix": "wiki_test",
            "llm_url": "http://llm.local/v1",
            "llm_model": "gpt-test",
        },
        str(tmp_path),
    )

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"]["default"] == "gpt-test"
    assert saved["wiki"]["path"] == str(tmp_path / "wiki" / "memory")
    assert saved["wiki"]["name"] == "memory"
    assert saved["wiki"]["embedding"]["url"] == "http://embeddings.local"
    assert saved["wiki"]["embedding"]["model"] == "embed-test"
    assert saved["wiki"]["embedding"]["dim"] == 1024
    assert saved["wiki"]["vector_store"]["url"] == "http://qdrant.local"
    assert saved["wiki"]["vector_store"]["collection_prefix"] == "wiki_test"
    assert saved["wiki"]["llm"]["url"] == "http://llm.local/v1"
    assert saved["wiki"]["llm"]["model"] == "gpt-test"


def test_is_available_requires_vendored_wiki_package_not_optional_qdrant(monkeypatch):
    provider = LLMWikiMemoryProvider()

    def fake_find_spec(name):
        if name == "hermes_wiki.engine":
            return object()
        if name == "qdrant_client":
            return None
        raise AssertionError(name)

    monkeypatch.setattr("plugins.memory.llm_wiki.importlib.util.find_spec", fake_find_spec)

    assert provider.is_available() is True


def test_engine_lazy_ensures_optional_deps(monkeypatch, tmp_path):
    calls = []

    class FakeWikiEngine:
        def __init__(self, config, *, read_only=False):
            calls.append((config, read_only))

    monkeypatch.setattr("plugins.memory.llm_wiki.WikiEngine", FakeWikiEngine, raising=False)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda feature, prompt=True: calls.append((feature, prompt)))
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    provider._engine()

    assert ("memory.llm_wiki", False) in calls


def test_non_primary_engine_does_not_lazy_install_optional_deps(monkeypatch, tmp_path):
    calls = []

    class FakeWikiEngine:
        def __init__(self, config, *, read_only=False):
            calls.append((config, read_only))

    monkeypatch.setattr("plugins.memory.llm_wiki.WikiEngine", FakeWikiEngine, raising=False)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda feature, prompt=True: calls.append((feature, prompt)))
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="cron")

    provider._engine()

    assert ("memory.llm_wiki", False) not in calls
    assert calls[0][1] is True


def test_register_registers_provider():
    ctx = DummyPluginContext()

    register(ctx)

    assert len(ctx.registered) == 1
    assert isinstance(ctx.registered[0], LLMWikiMemoryProvider)


def test_initialize_uses_hermes_home_for_default_wiki(tmp_path):
    provider = LLMWikiMemoryProvider()

    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    assert provider.wiki_path == tmp_path / "wiki" / "personal"
    assert provider.wiki_name == "personal"


def test_initialize_reads_config_wiki_section(tmp_path, monkeypatch):
    custom_wiki = tmp_path / "configured-wiki"
    (tmp_path / "config.yaml").write_text(
        "wiki:\n"
        f"  path: {custom_wiki}\n"
        "  name: hermes_memory\n"
        "  embedding:\n"
        "    url: http://embeddings.local\n"
        "    model: text-embedding-test\n"
        "    dim: 1024\n"
        "  vector_store:\n"
        "    url: http://qdrant.local\n"
        "    collection_prefix: test_wiki\n"
        "  llm:\n"
        "    url: http://llm.local/v1\n"
        "    model: gpt-test\n",
        encoding="utf-8",
    )
    provider = LLMWikiMemoryProvider()

    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    wiki_config = provider._wiki_config()

    assert provider.wiki_path == custom_wiki
    assert provider.wiki_name == "hermes_memory"
    assert wiki_config.embedding_url == "http://embeddings.local"
    assert wiki_config.embedding_model == "text-embedding-test"
    assert wiki_config.embedding_dim == 1024
    assert wiki_config.qdrant_url == "http://qdrant.local"
    assert wiki_config.collection_prefix == "test_wiki"
    assert wiki_config.llm_base_url == "http://llm.local/v1"
    assert wiki_config.llm_model == "gpt-test"


def test_initialize_expands_home_in_config(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    (tmp_path / "config.yaml").write_text(
        "wiki:\n"
        "  path: ~/custom-wiki\n"
        "  name: memory\n",
        encoding="utf-8",
    )
    provider = LLMWikiMemoryProvider()

    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    assert provider.wiki_path == fake_home / "custom-wiki"
    assert provider.wiki_name == "memory"


def test_initialize_normalizes_agent_context_case(tmp_path):
    provider = LLMWikiMemoryProvider()

    provider.initialize("s1", hermes_home=str(tmp_path), agent_context=" Primary ")

    assert provider._agent_context == "primary"
    assert provider._writes_allowed() is True


def test_initialize_does_not_construct_engine(monkeypatch, tmp_path):
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("engine should be lazy")

    monkeypatch.setattr("plugins.memory.llm_wiki.WikiEngine", boom, raising=False)
    provider = LLMWikiMemoryProvider()

    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    assert called is False


def test_engine_constructs_once_when_requested(monkeypatch, tmp_path):
    calls = []

    class FakeWikiEngine:
        def __init__(self, config, *, read_only=False):
            calls.append((config, read_only))

    monkeypatch.setattr("plugins.memory.llm_wiki.WikiEngine", FakeWikiEngine, raising=False)
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_ensure_optional_deps", lambda: None)

    first = provider._engine()
    second = provider._engine()

    assert first is second
    assert len(calls) == 1
    assert calls[0][0].wiki_path == tmp_path / "wiki" / "personal"
    assert calls[0][0].wiki_name == "personal"
    assert calls[0][1] is False


def test_non_primary_engine_is_constructed_read_only(monkeypatch, tmp_path):
    calls = []

    class FakeWikiEngine:
        def __init__(self, config, *, read_only=False):
            calls.append((config, read_only))

    monkeypatch.setattr("plugins.memory.llm_wiki.WikiEngine", FakeWikiEngine, raising=False)
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="cron")
    monkeypatch.setattr(provider, "_ensure_optional_deps", lambda: None)

    provider._engine()

    assert calls[0][0].wiki_path == tmp_path / "wiki" / "personal"
    assert calls[0][1] is True


def test_read_tool_schemas_present():
    provider = LLMWikiMemoryProvider()

    names = {schema["name"] for schema in provider.get_tool_schemas()}

    assert {
        "wiki_status",
        "wiki_orient",
        "wiki_search",
        "wiki_read",
        "wiki_query",
    } <= names


def test_query_tool_schema_defaults_to_read_only():
    provider = LLMWikiMemoryProvider()

    query_schema = next(schema for schema in provider.get_tool_schemas() if schema["name"] == "wiki_query")
    properties = query_schema["parameters"]["properties"]

    assert properties["file_result"].get("default") is False
    assert properties["log_query"].get("default") is False


class FakeEngine:
    def __init__(self) -> None:
        self.query_calls = []

    def status(self):
        return {"total_pages": 2, "sources": 1, "indexed_chunks": 7, "wiki": "/tmp/wiki"}

    def orient(self):
        return "# Wiki Orientation\nUseful index"

    def query(self, question, file_result=False, log_query=False):
        self.query_calls.append((question, file_result, log_query))
        return "wiki answer"

    def search(self, query, limit=5, exclude_sources=False):
        return []

    def lint(self, write_log=False):
        return {"issues": [], "write_log": write_log}

    def ingest_file(self, file_path, dry_run=True):
        return {"file_path": str(file_path), "dry_run": dry_run}

    def reindex(self):
        return {"pages": 2, "sources": 1, "chunks": 7}


def test_handle_status_tool_call_uses_engine(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_status", {})

    assert "total_pages" in result
    assert "2" in result


def test_handle_orient_tool_call_uses_engine(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_orient", {})

    assert result == "# Wiki Orientation\nUseful index"


def test_handle_query_defaults_to_read_only(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path))
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_query", {"question": "What is memory?"})

    assert "answer" in result
    assert fake_engine.query_calls == [("What is memory?", False, False)]


def test_boolean_tool_args_parse_string_false_as_false(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path))
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    provider.handle_tool_call(
        "wiki_query",
        {"question": "What is memory?", "file_result": "false", "log_query": "false"},
    )

    assert fake_engine.query_calls == [("What is memory?", False, False)]


def test_write_booleans_parse_string_false_safely(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="cron")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_ingest", {"file_path": "source.md", "dry_run": "false"})

    assert "blocked" in result.lower()


def test_handle_read_page_by_relative_path(tmp_path):
    page = tmp_path / "wiki" / "personal" / "concepts" / "memory.md"
    page.parent.mkdir(parents=True)
    page.write_text("# Memory\n\nUseful page", encoding="utf-8")

    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    result = provider.handle_tool_call("wiki_read", {"page": "concepts/memory.md"})

    assert "# Memory" in result
    assert "Useful page" in result


def test_handle_read_page_by_bare_slug(tmp_path):
    page = tmp_path / "wiki" / "personal" / "concepts" / "memory-policy.md"
    page.parent.mkdir(parents=True)
    page.write_text("# Memory Policy\n\nSlug lookup works", encoding="utf-8")

    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    result = provider.handle_tool_call("wiki_read", {"page": "memory-policy"})

    assert "# Memory Policy" in result
    assert "Slug lookup works" in result


def test_handle_read_page_blocks_traversal(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    result = provider.handle_tool_call("wiki_read", {"page": "../outside.md"})

    assert "secret" not in result
    assert "not found" in result.lower()


def test_system_prompt_block_is_tiny_and_retrieval_oriented(tmp_path):
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    block = provider.system_prompt_block()

    assert "LLM Wiki" in block
    assert "wiki_search" in block
    assert len(block) < 600


def test_sync_turn_is_noop(monkeypatch, tmp_path):
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: (_ for _ in ()).throw(AssertionError("sync_turn must not load engine")))

    provider.sync_turn("user", "assistant")


def test_prefetch_returns_bounded_cited_results(monkeypatch, tmp_path):
    class Result:
        def __init__(self, page_path, title, text, score):
            self.page_path = page_path
            self.title = title
            self.text = text
            self.score = score

    class Searcher:
        def search(self, query, limit=3, exclude_sources=True):
            assert exclude_sources is True
            return [
                Result("concepts/context.md", "Context Policy", "A" * 900, 0.91),
                Result("entities/hermes.md", "Hermes", "B" * 900, 0.82),
            ]

    class Engine:
        search = Searcher()

    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: Engine())

    result = provider.prefetch("memory policy")

    assert "LLM Wiki relevant context" in result
    assert "concepts/context.md" in result
    assert "entities/hermes.md" in result
    assert len(result) <= provider.prefetch_max_chars


def test_prefetch_formats_dict_search_results(monkeypatch, tmp_path):
    class Searcher:
        def search(self, query, limit=3, exclude_sources=True):
            return [{"page_path": "concepts/memory.md", "title": "Memory", "text": "Cited context", "score": 0.7}]

    class Engine:
        search = Searcher()

    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: Engine())

    result = provider.prefetch("memory")

    assert "concepts/memory.md" in result
    assert "Cited context" in result


def test_write_tool_schemas_have_safe_defaults():
    provider = LLMWikiMemoryProvider()
    schemas = {schema["name"]: schema for schema in provider.get_tool_schemas()}

    assert {"wiki_lint", "wiki_ingest", "wiki_reindex"} <= set(schemas)
    assert schemas["wiki_lint"]["parameters"]["properties"]["write_log"]["default"] is False
    assert schemas["wiki_ingest"]["parameters"]["properties"]["dry_run"]["default"] is True


def test_lint_defaults_to_non_mutating(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_lint", {})

    assert '"write_log": false' in result


def test_ingest_actual_write_blocked_in_non_primary(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="subagent")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_ingest", {"file_path": "/tmp/source.md", "dry_run": False})

    assert "blocked" in result.lower()
    assert "subagent" in result


def test_ingest_dry_run_blocked_in_non_primary(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="subagent")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_ingest", {"file_path": "/tmp/source.md"})

    assert "blocked" in result.lower()
    assert "subagent" in result


def test_reindex_blocked_in_non_primary(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="cron")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_reindex", {})

    assert "blocked" in result.lower()
    assert "cron" in result


def test_mutating_query_options_blocked_in_non_primary(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="subagent")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    filed = provider.handle_tool_call("wiki_query", {"question": "save this", "file_result": True})
    logged = provider.handle_tool_call("wiki_query", {"question": "log this", "log_query": True})

    assert "blocked" in filed.lower()
    assert "blocked" in logged.lower()
    assert fake_engine.query_calls == []


def test_lint_write_log_blocked_in_non_primary(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="cron")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    result = provider.handle_tool_call("wiki_lint", {"write_log": True})

    assert "blocked" in result.lower()
    assert "cron" in result


def test_search_limit_is_clamped(monkeypatch, tmp_path):
    calls = []

    class Searcher:
        def search(self, query, limit=5):
            calls.append(limit)
            return []

    class Engine:
        search = Searcher()

    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: Engine())

    provider.handle_tool_call("wiki_search", {"query": "memory", "limit": 9999})
    provider.handle_tool_call("wiki_search", {"query": "memory", "limit": -5})

    assert calls == [20, 1]


def test_read_page_is_output_bounded(tmp_path):
    page = tmp_path / "wiki" / "personal" / "concepts" / "large.md"
    page.parent.mkdir(parents=True)
    page.write_text("x" * 150_000, encoding="utf-8")
    provider = LLMWikiMemoryProvider()
    provider.initialize("s1", hermes_home=str(tmp_path), agent_context="primary")

    result = provider.handle_tool_call("wiki_read", {"page": "concepts/large.md"})

    assert len(result) < 70_000
    assert "truncated" in result.lower()


def test_plugin_loader_returns_llm_wiki_provider():
    provider = load_memory_provider("llm_wiki")

    assert isinstance(provider, LLMWikiMemoryProvider)


def test_memory_manager_routes_llm_wiki_tools(monkeypatch, tmp_path):
    fake_engine = FakeEngine()
    provider = LLMWikiMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: fake_engine)

    assert manager.has_tool("wiki_status")
    result = manager.handle_tool_call("wiki_status", {})

    assert "total_pages" in result


def test_memory_manager_prefetches_llm_wiki_context(monkeypatch, tmp_path):
    class Result:
        page_path = "concepts/memory.md"
        title = "Memory"
        text = "Durable source-backed memory"
        score = 0.99

    class Searcher:
        def search(self, query, limit=3, exclude_sources=True):
            return [Result()]

    class Engine:
        search = Searcher()

    provider = LLMWikiMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all("s1", hermes_home=str(tmp_path), agent_context="primary")
    monkeypatch.setattr(provider, "_engine", lambda: Engine())

    result = manager.prefetch_all("memory")

    assert "LLM Wiki relevant context" in result
    assert "concepts/memory.md" in result
