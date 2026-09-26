"""The mem0 config schema must agree with is_available(): a self-hosted server needs a host, not a key."""

import pytest

import plugins.memory.mem0 as mem0


@pytest.mark.parametrize(
    ("config", "required"),
    [
        ({"mode": "platform", "host": "http://mem0.local:8888"}, False),  # an AUTH_DISABLED server has no key
        ({"mode": "platform"}, True),  # the Platform always needs one
        ({"mode": "oss", "oss": {"vector_store": {"provider": "qdrant"}}}, False),
    ],
    ids=["self-hosted", "platform", "oss"],
)
def test_the_api_key_is_required_only_where_is_available_needs_one(monkeypatch, config, required):
    """A key marked required with a host configured left a self-hosted server "needs config" in the
    dashboard, and activation refuses anything that is not "ready"."""
    monkeypatch.setattr(mem0, "_load_config", lambda: config)
    provider = mem0.Mem0MemoryProvider()
    field = next(f for f in provider.get_config_schema() if f["key"] == "api_key")

    assert field["required"] is required
    if "host" in config:
        assert provider.is_available() is True
