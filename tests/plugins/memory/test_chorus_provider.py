import json
import os
from unittest.mock import patch

from plugins.chorus_common import ChorusClient, compact_resume, get_chorus_config
from plugins.memory import load_memory_provider


def test_chorus_memory_provider_loads_from_bundled_plugins(monkeypatch):
    monkeypatch.setenv("CHORUS_URL", "http://chorus.local")
    monkeypatch.setenv("CHORUS_API_KEY", "test-key")
    provider = load_memory_provider("chorus")
    assert provider is not None
    assert provider.name == "chorus"
    assert provider.is_available()
    tool_names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert {
        "chorus_resume_context",
        "chorus_memory_query",
        "chorus_memory_store",
        "chorus_emit_signal",
    }.issubset(tool_names)


def test_chorus_config_falls_back_to_mcp_server_env(monkeypatch):
    monkeypatch.delenv("CHORUS_URL", raising=False)
    monkeypatch.delenv("CHORUS_API_KEY", raising=False)
    with patch("hermes_cli.config.load_config", return_value={
        "mcp_servers": {
            "chorus": {
                "env": {
                    "CHORUS_URL": "http://from-config",
                    "CHORUS_API_KEY": "from-config-key",
                }
            }
        }
    }):
        assert get_chorus_config() == {"url": "http://from-config", "api_key": "from-config-key"}


def test_chorus_client_rpc_wraps_jsonrpc(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"jsonrpc": "2.0", "id": "x", "result": {"ok": True}}).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ChorusClient(url="http://chorus", api_key="secret", timeout=7)
    assert client.rpc("identity/whoami", {"x": 1}) == {"ok": True}
    assert captured["url"] == "http://chorus/rpc"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["method"] == "identity/whoami"
    assert captured["body"]["params"] == {"x": 1}
    assert captured["timeout"] == 7


def test_compact_resume_keeps_operational_sections():
    text = compact_resume({
        "identity": {"name": "vesta"},
        "project": {"tag": "agents-of-proto"},
        "workstream": {"title": "Deep Chorus", "status": "open"},
        "inbox_now": [{"preview": "gate waiting"}],
        "active_tasks": [{"content": "worker audit"}],
        "suggested_next_action": "check the bridge",
    })
    assert "identity: vesta" in text
    assert "project: agents-of-proto" in text
    assert "gate waiting" in text
    assert "check the bridge" in text
