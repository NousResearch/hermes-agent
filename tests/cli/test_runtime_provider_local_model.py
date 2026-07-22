import json

import requests

from hermes_cli.runtime_provider import (
    _LOCAL_MODEL_DISCOVERY_BODY_LIMIT,
    _auto_detect_local_model,
)


class _FakeResponse:
    def __init__(self, chunks, *, ok=True, encoding="utf-8"):
        self._chunks = list(chunks)
        self.ok = ok
        self.encoding = encoding
        self.closed = False

    def iter_content(self, chunk_size=1):
        yield from self._chunks

    def close(self):
        self.closed = True

    def json(self):
        raise AssertionError("local model discovery must not call response.json()")


def test_auto_detect_local_model_reads_bounded_stream(monkeypatch):
    response = _FakeResponse(
        [json.dumps({"data": [{"id": "local-model"}]}).encode("utf-8")]
    )
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return response

    monkeypatch.setattr(requests, "get", fake_get)

    assert _auto_detect_local_model("http://127.0.0.1:8000") == "local-model"
    assert captured["url"] == "http://127.0.0.1:8000/v1/models"
    assert captured["kwargs"]["stream"] is True
    assert response.closed is True


def test_auto_detect_local_model_drops_oversized_response(monkeypatch):
    response = _FakeResponse([b"x" * (_LOCAL_MODEL_DISCOVERY_BODY_LIMIT + 1)])
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return response

    monkeypatch.setattr(requests, "get", fake_get)

    assert _auto_detect_local_model("http://localhost:1234/v1") == ""
    assert captured["url"] == "http://localhost:1234/v1/models"
    assert captured["kwargs"]["stream"] is True
    assert response.closed is True
