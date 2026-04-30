from benchmarks.backends.mnemoria_adapter import (
    BACKEND_CAPABILITIES,
    BACKEND_CLASS,
    BACKEND_NAME,
)


def test_mnemoria_adapter_exports_backend_contract():
    assert BACKEND_NAME == "mnemoria"
    assert BACKEND_CLASS is not None
    assert BACKEND_CAPABILITIES.time_simulation is True
    assert BACKEND_CAPABILITIES.typed_facts is True
    assert BACKEND_CAPABILITIES.reward_learning is True
    assert BACKEND_CAPABILITIES.forgetting is True


def test_mnemoria_adapter_smoke_store_recall_reset():
    backend = BACKEND_CLASS(embedding_model="tfidf")
    backend.store("The project database is PostgreSQL 16.", category="factual")

    results = backend.recall("What database does the project use?", top_k=3)

    assert any("PostgreSQL 16" in result for result in results)
    assert backend.get_stats()["fact_count"] == 1

    backend.reset()
    assert backend.get_stats()["fact_count"] == 0


def test_mnemoria_adapter_converts_recall_with_ids_tuple_order():
    backend = BACKEND_CLASS(embedding_model="tfidf")
    backend.store("V[auth.mfa]: Multi-factor authentication is mandatory.", category="factual")

    results = backend.recall_with_ids("Is MFA required?", top_k=3)

    assert results
    content, memory_id = results[0]
    assert "Multi-factor" in content
    assert isinstance(memory_id, str)


def test_mnemoria_adapter_includes_target_context_for_typed_recall():
    backend = BACKEND_CLASS(embedding_model="tfidf")
    backend.store("V[api.url]: https://api.example.test", category="factual")

    results = backend.recall("What is the API URL?", top_k=1)

    assert results
    assert results[0].startswith("api url:")


def test_mnemoria_adapter_forget_removes_matching_content_and_retains_other_facts():
    backend = BACKEND_CLASS(embedding_model="tfidf")
    backend.store("private secret phrase: silver moth", category="private")
    backend.store("public project note: keep this memory", category="factual")

    backend.forget("silver moth")

    forgotten_results = backend.recall("silver moth", top_k=5)
    retained_results = backend.recall("project note", top_k=5)
    assert all("silver moth" not in result for result in forgotten_results)
    assert any("keep this memory" in result for result in retained_results)
