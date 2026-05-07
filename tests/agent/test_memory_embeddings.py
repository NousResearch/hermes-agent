from types import SimpleNamespace

from agent.memory_embeddings import MemoryEmbedder


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        if isinstance(input, list):
            data = [SimpleNamespace(embedding=[float(i), float(i) + 0.5]) for i, _ in enumerate(input)]
        else:
            data = [SimpleNamespace(embedding=[1.0, 2.0, 3.0])]
        return SimpleNamespace(data=data)


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingsAPI()


def test_embed_texts_batches_inputs(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "agent.memory_embeddings.resolve_provider_client",
        lambda provider, model: (fake_client, model),
    )

    embedder = MemoryEmbedder(provider="openrouter", model="text-embedding-test")
    vectors = embedder.embed_texts(["alpha", "beta"])

    assert vectors == [[0.0, 0.5], [1.0, 1.5]]
    assert fake_client.embeddings.calls == [
        {"model": "text-embedding-test", "input": ["alpha", "beta"]}
    ]


def test_embed_query_uses_single_string_input(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "agent.memory_embeddings.resolve_provider_client",
        lambda provider, model: (fake_client, model),
    )

    embedder = MemoryEmbedder(provider="openrouter", model="text-embedding-test")
    vector = embedder.embed_query("find this")

    assert vector == [1.0, 2.0, 3.0]
    assert fake_client.embeddings.calls == [
        {"model": "text-embedding-test", "input": "find this"}
    ]


def test_chunk_id_generation_is_stable_and_content_based(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "agent.memory_embeddings.resolve_provider_client",
        lambda provider, model: (fake_client, model),
    )

    embedder = MemoryEmbedder(provider="openrouter", model="text-embedding-test")

    text = "same input text"
    same_id = embedder.chunk_id_for_text(text)

    assert same_id == embedder.chunk_id_for_text(text)
    assert same_id == "memory-chunk:df5a302847fc7c0cdba82e0c6c263841bff5dc3d80eb693889b66d4d6f650f8d"
    assert same_id != embedder.chunk_id_for_text("different text")
    assert embedder.chunk_ids_for_texts([text, "different text"]) == [
        same_id,
        embedder.chunk_id_for_text("different text"),
    ]
