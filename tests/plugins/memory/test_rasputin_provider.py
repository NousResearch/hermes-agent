import json

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.memory.rasputin import RasputinMemoryProvider
from plugins.memory.rasputin.client import (
    RasputinClient,
    RasputinClientConfig,
    _MAX_COMMIT_TEXT_CHARS,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_provider_loading_and_discovery_manifest():
    provider = load_memory_provider("rasputin")

    assert provider is not None
    assert provider.name == "rasputin"
    assert provider.is_available() is True

    providers = {name: (description, available) for name, description, available in discover_memory_providers()}
    assert "rasputin" in providers
    description, available = providers["rasputin"]
    assert "derived-memory sidecar" in description
    assert available is True


def test_search_uses_post_body(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"results": [{"id": "r1", "text": "match"}]})

    monkeypatch.setattr("plugins.memory.rasputin.client.request.urlopen", fake_urlopen)

    client = RasputinClient(RasputinClientConfig(base_url="http://127.0.0.1:7777"))
    results = client.search("x" * 12000, limit=3)

    assert results == [{"id": "r1", "text": "match"}]
    assert captured["url"] == "http://127.0.0.1:7777/search"
    assert captured["method"] == "POST"
    assert captured["body"] == {"query": "x" * 12000, "limit": 3}
    assert captured["headers"]["Content-type"] == "application/json"


def test_prepare_commit_payload_truncates_oversized_text():
    original = "x" * (_MAX_COMMIT_TEXT_CHARS + 250)

    prepared = RasputinClient._prepare_commit_payload(
        {
            "text": original,
            "source": "hermes-turn",
            "metadata": {"kind": "conversation_window"},
        }
    )

    assert len(prepared["text"]) == _MAX_COMMIT_TEXT_CHARS
    assert prepared["text"] == original[:_MAX_COMMIT_TEXT_CHARS]
    assert prepared["metadata"]["kind"] == "conversation_window"
    assert prepared["metadata"]["rasputin_truncated"] is True
    assert prepared["metadata"]["rasputin_original_text_length"] == len(original)


def test_commit_sends_truncated_payload(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("plugins.memory.rasputin.client.request.urlopen", fake_urlopen)

    client = RasputinClient(RasputinClientConfig(base_url="http://127.0.0.1:7777"))
    ok = client.commit({"text": "y" * (_MAX_COMMIT_TEXT_CHARS + 10), "source": "hermes-turn"})

    assert ok is True
    assert captured["url"] == "http://127.0.0.1:7777/commit"
    assert captured["method"] == "POST"
    assert len(captured["body"]["text"]) == _MAX_COMMIT_TEXT_CHARS
    assert captured["body"]["metadata"]["rasputin_truncated"] is True


def test_provider_on_pre_compress_returns_notes_and_mirrors_checkpoint():
    provider = RasputinMemoryProvider()
    provider._client = object()
    provider._session_id = "session-abc"
    provider._platform = "cli"
    provider._agent_context = "primary"
    provider._user_id = "user-1"

    captured = {}

    def _capture(payload, *, label):
        captured["payload"] = payload
        captured["label"] = label

    provider._commit_best_effort = _capture

    notes = provider.on_pre_compress(
        [
            {"role": "user", "content": "Please preserve the exact backlink gap and port 8642 issue."},
            {"role": "assistant", "content": "I am preparing the final architecture recommendation now."},
        ]
    )

    assert "backlink gap" in notes
    assert "8642" in notes
    assert captured["label"] == "compaction-checkpoint"
    assert captured["payload"]["source"] == "hermes-compaction-checkpoint"
    assert captured["payload"]["metadata"]["kind"] == "compaction_checkpoint"
    assert captured["payload"]["metadata"]["session_id"] == "session-abc"
