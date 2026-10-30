"""Registration must supply the scope copied by native plugin workers (#139123)."""
from pathlib import Path

import pytest

from agent import secret_scope
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_constants import get_hermes_home, hermes_home_key


_PLUGIN = '''
import contextvars
import threading
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home

release = threading.Event()
done = threading.Event()
seen = []

def register(ctx):
    def read():
        try:
            if not release.wait(5):
                raise RuntimeError("fixture was not released")
            seen.append((get_secret("PLUGIN_SCOPE_PROBE"), str(get_hermes_home())))
        except Exception as exc:
            seen.append(type(exc).__name__)
        finally:
            done.set()
    worker = threading.Thread(target=contextvars.copy_context().run, args=(read,), daemon=True)
    worker.start()
    globals()["worker"] = worker
'''


def _load(home, *, deferred=False):
    directory = home / "plugins" / "registration-probe"
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(_PLUGIN if not deferred else "", encoding="utf-8")
    manager = PluginManager(scope_key=hermes_home_key(home))
    manifest = PluginManifest(name="registration-probe", key="registration-probe",
                              source="user", path=str(directory))
    if deferred:
        import sys
        from hermes_cli.plugins import LoadedPlugin
        from hermes_cli.plugins_loader import _plugin_home_scope

        (directory / "tools.py").write_text(_PLUGIN.replace("def register(ctx):", "def register_tools(ctx):"), encoding="utf-8")
        manifest.provides_tools = ["scope-probe"]
        loaded = LoadedPlugin(manifest=manifest, enabled=True, deferred=True)
        with _plugin_home_scope(manager.home_path):
            manager._register_deferred_platform_tools(manifest, loaded)
        return manager, sys.modules[f"{loaded.module.__name__}.tools"]
    manager._load_plugin(manifest)
    loaded = manager._plugins[manifest.key]
    assert loaded.enabled, loaded.error
    return manager, loaded.module


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("activate_before", [False, True])
def test_registration_worker_resolves_own_profile_across_activation(tmp_path, monkeypatch, activate_before, deferred):
    launch, secondary = tmp_path / "launch", tmp_path / "secondary"
    launch.mkdir()
    secondary.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PLUGIN_SCOPE_PROBE", "ambient")
    for home, value in ((launch, "launch-value"), (secondary, "secondary-value")):
        (home / ".env").write_text(f"PLUGIN_SCOPE_PROBE={value}\n", encoding="utf-8")
    prior_mode = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    loaded = []
    try:
        secret_scope.set_multiplex_active(activate_before)
        loaded = [_load(launch, deferred=deferred), _load(secondary, deferred=deferred)]
        assert secret_scope.current_secret_scope() is None
        assert get_hermes_home() == launch
        secret_scope.set_multiplex_active(True)
        for manager, module in loaded:
            module.release.set()
            assert module.done.wait(5)
            expected = "launch-value" if manager.home_path == launch else "secondary-value"
            assert module.seen == [(expected, str(manager.home_path))]
        with pytest.raises(secret_scope.UnscopedSecretError):
            secret_scope.get_secret("PLUGIN_SCOPE_PROBE")
    finally:
        for manager, module in loaded:
            module.release.set()
            module.worker.join(5)
            manager.unload()
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(prior_mode)


def test_registration_scope_keeps_owner_mapping_and_restores_foreign_caller(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from hermes_cli.plugins_loader import _plugin_home_scope, _plugin_registration_scope

    @contextmanager
    def registration(home):
        with _plugin_home_scope(home), _plugin_registration_scope(home):
            yield

    launch, secondary = tmp_path / "launch", tmp_path / "secondary"
    launch.mkdir()
    secondary.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PLUGIN_ENV_ONLY", "launch-only")
    owner_mapping = {"PLUGIN_SCOPE_PROBE": "hydrated-owner"}
    (secondary / ".env").write_text("PLUGIN_SCOPE_PROBE=secondary-value\n", encoding="utf-8")
    token = secret_scope.set_secret_scope(owner_mapping, profile_home=str(launch))
    try:
        with registration(launch):
            assert secret_scope.current_secret_scope() is owner_mapping
        with pytest.raises(RuntimeError, match="fixture"):
            with registration(secondary):
                assert secret_scope.get_secret("PLUGIN_SCOPE_PROBE") == "secondary-value"
                assert secret_scope.get_secret("PLUGIN_ENV_ONLY") is None
                raise RuntimeError("fixture")
        assert secret_scope.current_secret_scope() is owner_mapping
        assert get_hermes_home() == launch
    finally:
        secret_scope.reset_secret_scope(token)
    with registration(launch):
        assert secret_scope.get_secret("PLUGIN_ENV_ONLY") == "launch-only"
    assert secret_scope.current_secret_scope() is None


def test_discovered_worker_receives_new_plugin_source_after_registration(tmp_path, monkeypatch):
    """A registration snapshot must not hide credentials hydrated after the sweep."""
    import os
    from hermes_cli import env_loader, plugins as plugins_mod
    from agent.secret_sources import registry
    from tests.hermes_cli.test_secret_source_bootstrap import _FIXTURE_PLUGIN

    home = tmp_path / "launch"
    directory = home / "plugins" / "registration-probe"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("name: registration-probe\nversion: 0.1.0\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [registration-probe]\nsecrets:\n  fixturevault:\n    enabled: true\n",
        encoding="utf-8",
    )
    source = _FIXTURE_PLUGIN.replace("def register(ctx):", "def register_source(ctx):")
    worker = _PLUGIN.replace('"PLUGIN_SCOPE_PROBE"', '"REVIEW_PLUGIN_SECRET"')
    worker = worker.replace("def register(ctx):", "def register(ctx):\n    register_source(ctx)")
    (directory / "__init__.py").write_text(source + worker, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    monkeypatch.delenv("REVIEW_PLUGIN_SECRET", raising=False)
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    prior_mode = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    manager, module = PluginManager(scope_key=hermes_home_key(home)), None
    try:
        secret_scope.set_multiplex_active(False)
        manager.discover_and_load()
        loaded = manager._plugins["registration-probe"]
        assert loaded.enabled, loaded.error
        module = loaded.module
        assert env_loader.get_secret_source_values(home)["REVIEW_PLUGIN_SECRET"] == "from-plugin"
        secret_scope.set_multiplex_active(True)
        module.release.set()
        assert module.done.wait(5)
        assert module.seen == [("from-plugin", str(manager.home_path))]
    finally:
        if module is not None:
            module.release.set()
            module.worker.join(5)
        manager.unload()
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(prior_mode)
        os.environ.pop("REVIEW_PLUGIN_SECRET", None)
        registry._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()


@pytest.mark.parametrize("capture_at_activation", [False, True])
def test_launch_worker_does_not_capture_post_activation_ambient_values(tmp_path, monkeypatch, capture_at_activation):
    from tui_gateway import launch_profile_policy

    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setenv("PLUGIN_SCOPE_PROBE", "before-activation")
    prior_mode = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    manager = module = None
    try:
        secret_scope.set_multiplex_active(False)
        manager, module = _load(home)
        expected = "before-activation"
        if capture_at_activation:
            expected = "last-boot-injection"
            monkeypatch.setenv("PLUGIN_SCOPE_PROBE", expected)
            launch_profile_policy.activate_multi_profile_hosting()
        else:
            secret_scope.set_multiplex_active(True)
        monkeypatch.setenv("PLUGIN_SCOPE_PROBE", "foreign-after-activation")
        module.release.set()
        assert module.done.wait(5)
        assert module.seen == [(expected, str(manager.home_path))]
    finally:
        if module is not None:
            module.release.set()
            module.worker.join(5)
            manager.unload()
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(prior_mode)


@pytest.mark.parametrize("phase", ["construct", "read"])
def test_registration_mapping_discards_capture_when_activation_intervenes(tmp_path, monkeypatch, phase):
    from hermes_cli.plugins_loader import _PluginSecretScope
    from tui_gateway import launch_profile_policy

    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PLUGIN_SCOPE_PROBE", "safe-launch")
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    prior_mode = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    scope = _PluginSecretScope(home) if phase == "read" else None
    original = secret_scope.is_multiplex_active
    intercepted = False

    def activate_after_check():
        nonlocal intercepted
        if not intercepted:
            intercepted = True
            launch_profile_policy.activate_multi_profile_hosting()
            monkeypatch.setenv("PLUGIN_SCOPE_PROBE", "foreign-after-check")
            return False  # the caller observed inactive immediately before activation
        return original()

    monkeypatch.setattr(secret_scope, "is_multiplex_active", activate_after_check)
    try:
        if scope is None:
            scope = _PluginSecretScope(home)
            assert scope.launch_env == {}
        assert scope.get("PLUGIN_SCOPE_PROBE") == "safe-launch"
    finally:
        secret_scope.set_multiplex_active(prior_mode)
