from shared_tool_broker.config import BrokerSettings
from shared_tool_broker.core import IdempotencyStore, provider_error_category


def test_provider_error_category():
    assert provider_error_category(RuntimeError("401 unauthorized")) == "auth"
    assert provider_error_category(RuntimeError("429 rate limit")) == "rate_limit"
    assert provider_error_category(RuntimeError("timed out")) == "timeout"


def test_idempotency_store_round_trip(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.json")
    assert store.get("scope", "key") is None
    store.put("scope", "key", {"ok": True})
    assert store.get("scope", "key") == {"ok": True}


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    settings = BrokerSettings.load()
    assert settings.port == 8767
    assert settings.grain_mcp_url == "https://api.grain.com/_/mcp"
