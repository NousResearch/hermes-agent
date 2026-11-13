"""Plugin vault registration uses existing scoped leases; no browser/secrets."""
import json
from unittest.mock import patch
from textwrap import dedent
import pytest
from agent.vault_backends.base import LoginBackend, backend_for_handle
from agent import vault_backend_registry as registry


class Example(LoginBackend):
    name, prefix, display_name = "example", "example:", "Example"
    def list_items(self):
        return []
    def get_meta(self, handle):
        return None
    def resolve_password(self, handle):
        raise AssertionError("registry tests must never resolve passwords")


def _record_verified_install(home, dirname):
    """The installer-owned record a catalog install at tier ``verified`` leaves (login backends need one)."""
    path = home / "plugins" / ".install-metadata.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    sha = "a" * 40
    records[dirname] = {"source": f"https://github.com/example/{dirname}", "revision": sha, "pinned": True,
                        "catalog": {"name": dirname, "repo": f"https://github.com/example/{dirname}",
                                    "tier": "verified", "pin": sha, "sha": sha}}
    path.write_text(json.dumps(records), encoding="utf-8")


def test_registration_requires_explicit_profile_enablement():
    with patch.dict(registry._providers, clear=True), \
         patch("agent.vault_backends.base.is_enabled", return_value=False), \
         patch("agent.vault_backends.base._cfg", return_value={}) as cfg:
        registry.register_provider(Example())
        assert backend_for_handle("example:one") is None
        cfg.return_value = {"example": {"enabled": True}}
        assert backend_for_handle("example:one").name == "example"


def test_builtin_names_and_handle_prefixes_are_reserved():
    class Reserved(Example):
        name = "local"
    with pytest.raises(ValueError):
        registry.register_provider(Reserved())
    class ReservedPrefix(Example):
        prefix = "op:"
    with pytest.raises(ValueError):
        registry.register_provider(ReservedPrefix())


def test_reload_cannot_change_prefix_and_leave_ambiguous_restore(tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest
    scope = str(tmp_path / "home")
    context = PluginContext(PluginManifest(name="login-example", source="bundled"), PluginManager(scope_key=scope))
    original = Example()
    first = context.register_login_backend(original)
    class ChangedPrefix(Example):
        prefix = "changed:"
    assert context.register_login_backend(ChangedPrefix()) is None
    class Other(Example):
        name = "other"
    assert context.register_login_backend(Other()) is None
    assert registry.get_provider("example", scope=scope) is original
    first.dispose()
    assert registry.get_provider("example", scope=scope) is None


def test_lease_reload_and_unload_do_not_cross_profiles(tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    manifest = PluginManifest(name="test-vault", source="bundled")
    first_ctx = PluginContext(manifest, PluginManager(scope_key=first))
    second_ctx = PluginContext(manifest, PluginManager(scope_key=second))
    original, replacement, sibling = Example(), Example(), Example()
    a = first_ctx.register_login_backend(original)
    assert first_ctx.register_login_backend(replacement) is None  # a held name is never taken over
    c = second_ctx.register_login_backend(sibling)
    assert registry.get_provider("example", scope=first) is original
    assert registry.get_provider("example", scope=second) is sibling
    a.dispose()
    assert registry.get_provider("example", scope=first) is None
    assert registry.get_provider("example", scope=second) is sibling
    c.dispose()


def test_real_plugin_discovery_reload_and_disable(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginManager
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    home = tmp_path / "home"
    plugin = home / "plugins" / "login_example"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: login_example\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(dedent('''
        from agent.vault_backends.base import LoginBackend
        class Backend(LoginBackend):
            name, prefix, display_name = "example", "example:", "Example"
            def list_items(self):
                return []
            def get_meta(self, handle):
                return None
            def resolve_password(self, handle):
                raise AssertionError("no secret reads in discovery")
        def register(ctx):
            ctx.register_login_backend(Backend())
    '''), encoding="utf-8")
    _record_verified_install(home, "login_example")
    config = home / "config.yaml"
    config.write_text("plugins:\n  enabled: [login_example]\nvault:\n  example:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    manager = PluginManager(scope_key=str(home))
    try:
        manager.discover_and_load()
        first = backend_for_handle("example:one")
        assert first is not None
        manager.discover_and_load(force=True)
        assert backend_for_handle("example:one") is not first
        config.write_text("plugins:\n  enabled: []\nvault:\n  example:\n    enabled: true\n", encoding="utf-8")
        manager.discover_and_load(force=True)
        assert backend_for_handle("example:one") is None
    finally:
        manager.unload()
        reset_hermes_home_override(token)


def test_host_plugin_refuses_server_local_fill_context():
    from hermes_cli.plugin_host_child import RemotePluginContext
    from hermes_cli.plugin_host_wire import PluginHostUnsupported
    remote = RemotePluginContext(None, "example", {"ctx_methods": ["register_login_backend"]})
    with pytest.raises(PluginHostUnsupported, match="server-local"):
        remote.register_login_backend(Example())
