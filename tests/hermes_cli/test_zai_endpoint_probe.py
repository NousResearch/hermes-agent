from hermes_cli import auth


class _ProbeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_bytes(self, *args, **kwargs):
        raise AssertionError("Z.AI endpoint probe should not read response bodies")


def test_detect_zai_endpoint_streams_probe_without_reading_body(monkeypatch):
    calls = []
    monkeypatch.setattr(
        auth,
        "ZAI_ENDPOINTS",
        [("global", "https://api.z.ai/api/paas/v4", ["glm-5"], "Global")],
    )

    def _fake_stream(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _ProbeResponse(200)

    monkeypatch.setattr(auth.httpx, "stream", _fake_stream)

    result = auth.detect_zai_endpoint("zai-key", timeout=3.5)

    assert result == {
        "id": "global",
        "base_url": "https://api.z.ai/api/paas/v4",
        "model": "glm-5",
        "label": "Global",
    }
    assert calls
    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == "https://api.z.ai/api/paas/v4/chat/completions"
    assert kwargs["timeout"] == 3.5
    assert kwargs["json"]["model"] == "glm-5"
    assert kwargs["headers"]["Authorization"] == "Bearer zai-key"


def test_detect_zai_endpoint_ignores_non_200_without_reading_body(monkeypatch):
    monkeypatch.setattr(
        auth,
        "ZAI_ENDPOINTS",
        [("global", "https://api.z.ai/api/paas/v4", ["glm-5"], "Global")],
    )
    monkeypatch.setattr(auth.httpx, "stream", lambda *_args, **_kwargs: _ProbeResponse(401))

    assert auth.detect_zai_endpoint("zai-key") is None
