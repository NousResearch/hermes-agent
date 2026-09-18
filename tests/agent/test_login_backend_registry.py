"""Plugin registration for browser credential-vault login backends."""

from __future__ import annotations

from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta


class _PluginBackend(LoginBackend):
    name = "testpass"
    display_name = "TestPass"
    prefix = "tp:"

    def __init__(self, config):
        self.config = config

    @classmethod
    def is_available(cls, config):
        return bool(config.get("available"))

    def list_items(self):
        return []

    def get_meta(self, handle):
        return None

    def resolve_password(self, handle):
        return ""


def test_registered_backend_is_enabled_and_routes_by_handle(monkeypatch):
    from agent.vault_backends import registry
    from agent.vault_backends.base import backend_for_handle, enabled_backends

    registry._reset_for_tests()
    monkeypatch.setattr(
        "agent.vault_backends.base._cfg",
        lambda: {"testpass": {"enabled": True, "available": True}},
    )
    registry.register_backend(_PluginBackend)

    assert [backend.name for backend in enabled_backends()] == ["local", "testpass"]
    assert isinstance(backend_for_handle("tp:item"), _PluginBackend)


def test_registration_rejects_builtin_names_and_prefixes(caplog):
    from agent.vault_backends import registry

    class ShadowName(_PluginBackend):
        name = "bitwarden"

    class ShadowPrefix(_PluginBackend):
        name = "otherpass"
        prefix = "bw:"

    class DuplicateName(_PluginBackend):
        prefix = "other:"

    registry._reset_for_tests()
    assert registry.register_backend(_PluginBackend) is True
    assert registry.register_backend(ShadowName) is False
    assert registry.register_backend(ShadowPrefix) is False
    assert registry.register_backend(DuplicateName) is False
    assert registry.list_backends() == [_PluginBackend]
    assert "built-in" in caplog.text


def test_plugin_context_registration_is_profile_scoped_and_reversible():
    from agent.vault_backends import registry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    registry._reset_for_tests()
    manager = PluginManager(scope_key="login-backend-test-profile")
    context = PluginContext(PluginManifest(name="testpass-plugin", key="testpass-plugin"), manager)

    registration = context.register_login_backend(_PluginBackend)
    assert registration is not None
    assert registry.list_backends(scope="login-backend-test-profile") == [_PluginBackend]
    assert registry.list_backends(scope="other-profile") == []

    registration.dispose()
    assert registry.list_backends(scope="login-backend-test-profile") == []
