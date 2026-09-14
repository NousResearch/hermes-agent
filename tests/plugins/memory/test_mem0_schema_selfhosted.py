"""The mem0 config schema must agree with is_available(): a self-hosted server needs a host, not a key."""

import plugins.memory.mem0 as mem0


def _required(monkeypatch, config):
    monkeypatch.setattr(mem0, "_load_config", lambda: config)
    field = next(f for f in mem0.Mem0MemoryProvider().get_config_schema() if f["key"] == "api_key")
    return field["required"]


def test_platform_mode_requires_the_api_key(monkeypatch):
    assert _required(monkeypatch, {"mode": "platform"}) is True


def test_self_hosted_server_does_not_require_the_api_key(monkeypatch):
    # AUTH_DISABLED deployments have no key; the dashboard must not hold activation hostage to one.
    assert _required(monkeypatch, {"mode": "platform", "host": "http://mem0.local:8888"}) is False
    assert mem0.Mem0MemoryProvider().is_available() is True, "is_available already treats the host as enough"


def test_oss_mode_does_not_require_the_api_key(monkeypatch):
    assert _required(monkeypatch, {"mode": "oss", "oss": {"vector_store": {"provider": "qdrant"}}}) is False
