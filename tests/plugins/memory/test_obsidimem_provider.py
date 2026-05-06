import json
import threading

import pytest

from plugins.memory.obsidimem import ObsidimemProvider

_DEFAULT_CONFIG = {
    "api_base_url": "http://127.0.0.1:8000",
    "observer_name": "hermes",
    "observed_name": "doug",
    "recall_mode": "hybrid",
    "budget": 1200,
    "timeout": 60.0,
    "trigger_dreamer_on_session_end": False,
}


class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self._data = data or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class FakeClient:
    """Records all HTTP calls; returns sensible defaults."""

    def __init__(self, **kwargs):
        self.calls: list[tuple[str, str, object]] = []

    def post(self, url, *, json=None, **kwargs):
        self.calls.append(("POST", url, json))
        if "/memory/sessions" in url:
            return FakeResponse({"id": "obs-session-abc"})
        return FakeResponse({})

    def get(self, url, *, params=None, **kwargs):
        self.calls.append(("GET", url, params))
        return FakeResponse({})

    def patch(self, url, **kwargs):
        self.calls.append(("PATCH", url, None))
        return FakeResponse({})

    def close(self):
        self.calls.append(("CLOSE", None, None))


@pytest.fixture
def fake_client(monkeypatch):
    """Patch httpx.Client so the plugin gets FakeClient on initialize()."""
    client = FakeClient()
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    return client


@pytest.fixture
def provider(fake_client, tmp_path):
    """Initialized ObsidimemProvider backed by FakeClient."""
    (tmp_path / "obsidimem.json").write_text(json.dumps(_DEFAULT_CONFIG))
    p = ObsidimemProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    return p
