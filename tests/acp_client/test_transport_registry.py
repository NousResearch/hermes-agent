"""Tests for acp_client.transport_registry — opt-in entries + env allowlist."""

import pytest

from acp_client.transport_registry import (
    DEFAULT_REGISTRY,
    TransportRegistry,
    TransportSpec,
)


class TestResolve:
    def test_known_transports_present(self):
        names = DEFAULT_REGISTRY.names()
        assert {"claude", "codex", "gemini-cli", "hermes-acp-sibling"} <= set(names)

    def test_resolve_returns_spec(self):
        spec = DEFAULT_REGISTRY.resolve("claude")
        assert spec.command == "claude"
        assert "--acp" in spec.args

    def test_unknown_transport_raises(self):
        with pytest.raises(KeyError):
            DEFAULT_REGISTRY.resolve("definitely-not-a-transport")

    def test_get_unknown_returns_none(self):
        assert DEFAULT_REGISTRY.get("nope") is None


class TestEnvAllowlist:
    def test_only_allowlisted_keys_forwarded(self):
        spec = DEFAULT_REGISTRY.resolve("claude")
        base = {
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "ANTHROPIC_API_KEY": "sk-secret",
            "OPENAI_API_KEY": "sk-other",
            "HERMES_HOME": "/home/u/.hermes",
            "RANDOM_THING": "x",
        }
        env = spec.resolve_env(base)
        assert env == {"PATH": "/usr/bin", "HOME": "/home/u"}

    def test_no_credential_keys_in_any_default_allowlist(self):
        for name in DEFAULT_REGISTRY.names():
            spec = DEFAULT_REGISTRY.resolve(name)
            joined = " ".join(spec.env_allowlist).upper()
            for marker in ("API_KEY", "TOKEN", "SECRET", "ANTHROPIC", "OPENAI", "HERMES"):
                assert marker not in joined, (name, marker)

    def test_extra_allowlist_key_is_forwarded(self):
        spec = TransportSpec(
            name="custom", command="x", env_allowlist=("MY_FLAG",)
        )
        env = spec.resolve_env({"MY_FLAG": "1", "OTHER": "2", "PATH": "/bin"})
        assert env == {"MY_FLAG": "1", "PATH": "/bin"}


class TestRegister:
    def test_register_adds_entry(self):
        reg = TransportRegistry(specs=[])
        assert reg.names() == []
        reg.register(TransportSpec(name="x", command="x"))
        assert reg.resolve("x").command == "x"
