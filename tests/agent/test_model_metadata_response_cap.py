import httpx

from agent import model_metadata


class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: int, chunk_size: int = 1024 * 1024):
        self.chunks = chunks
        self.chunk = b"x" * chunk_size
        self.pulled = 0
        self.closed = False

    def __iter__(self):
        for _ in range(self.chunks):
            self.pulled += 1
            yield self.chunk

    def close(self) -> None:
        self.closed = True


def _install_transport(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)


def test_detect_local_server_type_stops_oversized_tags_stream(monkeypatch):
    oversized = _ChunkStream(chunks=32)

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, stream=oversized)
        return httpx.Response(404, json={})

    _install_transport(monkeypatch, handler)
    model_metadata._endpoint_probe_path_cache.clear()

    assert model_metadata.detect_local_server_type("http://127.0.0.1:11434") is None
    assert oversized.closed is True
    assert oversized.pulled < oversized.chunks


def test_ollama_show_stops_oversized_stream(monkeypatch):
    oversized = _ChunkStream(chunks=32)
    _install_transport(
        monkeypatch,
        lambda _request: httpx.Response(200, stream=oversized),
    )

    result = model_metadata._query_ollama_api_show_uncached(
        "model", "http://127.0.0.1:11434"
    )

    assert result is None
    assert oversized.closed is True
    assert oversized.pulled < oversized.chunks
