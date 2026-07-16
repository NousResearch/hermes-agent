"""Tests for the model-providers plugin discovery system.

Verifies that:
 1. All bundled providers at plugins/model-providers/<name>/ are discovered
 2. User plugins at $HERMES_HOME/plugins/model-providers/<name>/ override bundled
 3. plugin.yaml manifests with kind=model-provider are correctly categorized
"""

from __future__ import annotations

import importlib.metadata
import pkgutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg
    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._discovered = False
    # Evict any cached plugin modules so the next import re-executes.
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
            or mod.startswith("_test_provider_ep_")
        ):
            del sys.modules[mod]


@pytest.fixture(autouse=True)
def _reset_provider_state_after_test(monkeypatch):
    yield
    _clear_provider_caches()


def _write_provider_module(path: Path, name: str, base_url: str) -> None:
    path.write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(\n"
        f"    name={name!r},\n"
        f"    base_url={base_url!r},\n"
        "    auth_type='none',\n"
        "))\n"
    )


def _entry_point(
    name: str, module: str, attribute: str | None = None
) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(
        name=name,
        value=f"{module}:{attribute}" if attribute else module,
        group="hermes_agent.model_providers",
    )


def _isolate_discovery(tmp_path, monkeypatch, entry_points=()) -> Path:
    import providers
    from hermes_cli import config

    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(providers, "_BUNDLED_PLUGINS_DIR", empty_bundled)
    monkeypatch.setattr(pkgutil, "iter_modules", lambda _paths: [])
    selected = importlib.metadata.EntryPoints(entry_points)

    def fake_entry_points(**params):
        return selected.select(**params) if params else selected

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    _clear_provider_caches()
    return home


def _write_activation_config(home: Path, *, enabled=(), disabled=()) -> None:
    lines = ["plugins:", "  enabled:"]
    lines.extend(f"    - {name}" for name in enabled)
    if not enabled:
        lines.append("    []")
    lines.append("  disabled:")
    lines.extend(f"    - {name}" for name in disabled)
    if not disabled:
        lines.append("    []")
    (home / "config.yaml").write_text("\n".join(lines) + "\n")


def _write_user_provider(home: Path, directory: str, name: str, base_url: str) -> Path:
    plugin_dir = home / "plugins" / "model-providers" / directory
    plugin_dir.mkdir(parents=True)
    _write_provider_module(plugin_dir / "__init__.py", name, base_url)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {directory}\nkind: model-provider\nversion: 0.0.1\n"
    )
    return plugin_dir


def test_bundled_plugins_discovered():
    """Every plugins/model-providers/<name>/ should contain a plugin.yaml + __init__.py."""
    plugins_dir = REPO_ROOT / "plugins" / "model-providers"
    assert plugins_dir.is_dir(), f"Missing {plugins_dir}"

    child_dirs = [c for c in plugins_dir.iterdir() if c.is_dir()]
    assert len(child_dirs) >= 28, f"Expected at least 28 provider plugins, found {len(child_dirs)}"

    for child in child_dirs:
        assert (child / "__init__.py").exists(), f"{child.name} missing __init__.py"
        assert (child / "plugin.yaml").exists(), f"{child.name} missing plugin.yaml"


def test_all_profiles_register():
    """After discovery, the registry must contain every bundled provider directory.

    This is an invariant — the number of profiles matches the number of plugin
    directories, not a hardcoded count. Counts shift when providers are
    added/removed; that's expected and shouldn't break CI.
    """
    _clear_provider_caches()
    from providers import list_providers

    plugins_dir = REPO_ROOT / "plugins" / "model-providers"
    plugin_dir_count = sum(1 for c in plugins_dir.iterdir() if c.is_dir())

    profiles = list_providers()
    names = sorted(p.name for p in profiles)
    # Some plugin __init__.py files register multiple profiles, so the registry
    # count is >= the directory count (never less).
    assert len(names) >= plugin_dir_count, (
        f"Expected at least {plugin_dir_count} profiles (one per plugin dir), got {len(names)}: {names}"
    )

    # Spot-check representative providers from different categories
    for required in (
        "openrouter", "anthropic", "custom", "bedrock", "openai-codex",
        "minimax-oauth", "gmi", "xiaomi", "alibaba-coding-plan", "fireworks",
    ):
        assert required in names, f"Missing profile: {required}"


def test_user_plugin_overrides_bundled(tmp_path, monkeypatch):
    """A user plugin with the same name must override the bundled profile."""
    # Point HERMES_HOME at a fresh temp dir
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # get_hermes_home() may be module-cached depending on codebase; ensure the
    # env var is the source of truth. Most code paths re-read it each call.

    # Drop a user plugin that replaces 'gmi'
    user_gmi = hermes_home / "plugins" / "model-providers" / "gmi"
    user_gmi.mkdir(parents=True)
    (user_gmi / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "\n"
        "custom_gmi = ProviderProfile(\n"
        '    name="gmi",\n'
        '    aliases=("gmi-user-override-test",),\n'
        '    env_vars=("GMI_API_KEY",),\n'
        '    base_url="https://user-override.example.com/v1",\n'
        '    auth_type="api_key",\n'
        ")\n"
        "register_provider(custom_gmi)\n"
    )
    (user_gmi / "plugin.yaml").write_text(
        "name: gmi-user-override\n"
        "kind: model-provider\n"
        "version: 0.0.1\n"
        "description: Test user override\n"
    )

    _clear_provider_caches()
    from providers import get_provider_profile

    gmi = get_provider_profile("gmi")
    assert gmi is not None
    assert gmi.base_url == "https://user-override.example.com/v1", (
        f"User override not applied; got base_url={gmi.base_url!r}"
    )
    assert "gmi-user-override-test" in gmi.aliases

    # Clean up: reset discovery state so other tests see the bundled version
    _clear_provider_caches()


def test_general_plugin_manager_skips_model_provider_kind(tmp_path, monkeypatch):
    """The general PluginManager must NOT import model-provider plugins
    (providers/__init__.py handles them). It records the manifest only."""
    from hermes_cli import plugins as plugin_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Create a user-installed plugin with an explicit kind: model-provider.
    user_plugin = hermes_home / "plugins" / "test-model-provider"
    user_plugin.mkdir(parents=True)
    (user_plugin / "plugin.yaml").write_text(
        "name: test-model-provider\n"
        "kind: model-provider\n"
        "version: 0.0.1\n"
    )
    (user_plugin / "__init__.py").write_text(
        # Intentionally broken import — if the general loader tries to
        # import this module, the test will fail with ImportError.
        "raise AssertionError('model-provider plugins must not be imported by PluginManager')\n"
    )

    # Fresh manager
    manager = plugin_mod.PluginManager()
    manager.discover_and_load(force=True)

    # The manifest should be recorded but not loaded
    loaded = manager._plugins.get("test-model-provider")
    assert loaded is not None
    assert loaded.manifest.kind == "model-provider"
    # No import means the module must NOT be in the plugins list as a loaded one.
    # We check that the general loader didn't crash and didn't raise from the
    # broken __init__.py.


def _write_marker_module(path: Path, marker: Path) -> None:
    path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")


def test_enabled_model_provider_entry_point_invokes_registrar(tmp_path, monkeypatch):
    module = "_test_provider_ep_enabled"
    (tmp_path / f"{module}.py").write_text(
        "def register():\n"
        "    from providers import register_provider\n"
        "    from providers.base import ProviderProfile\n"
        "    register_provider(ProviderProfile(name='package-enabled', "
        "base_url='https://package.example/v1', auth_type='none'))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    home = _isolate_discovery(
        tmp_path, monkeypatch, [_entry_point("package-enabled", module, "register")]
    )
    _write_activation_config(home, enabled=["model-providers/package-enabled"])

    from providers import get_provider_profile

    assert get_provider_profile("package-enabled").base_url == "https://package.example/v1"


@pytest.mark.parametrize(
    ("enabled", "disabled"),
    [([], []), (["model-providers/package-gated"], ["model-providers/package-gated"])],
)
def test_package_provider_requires_unopposed_canonical_enable(
    tmp_path, monkeypatch, enabled, disabled
):
    marker = tmp_path / "package-executed"
    module = "_test_provider_ep_gated"
    _write_marker_module(tmp_path / f"{module}.py", marker)
    monkeypatch.syspath_prepend(str(tmp_path))
    home = _isolate_discovery(
        tmp_path, monkeypatch, [_entry_point("package-gated", module)]
    )
    _write_activation_config(home, enabled=enabled, disabled=disabled)

    from providers import list_providers

    list_providers()
    assert not marker.exists()


def test_failing_package_entry_point_does_not_block_another(tmp_path, monkeypatch):
    failing = "_test_provider_ep_failing"
    working = "_test_provider_ep_working"
    (tmp_path / f"{failing}.py").write_text("raise RuntimeError('broken')\n")
    _write_provider_module(
        tmp_path / f"{working}.py", "package-working", "https://working.example/v1"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    home = _isolate_discovery(
        tmp_path,
        monkeypatch,
        [_entry_point("failing", failing), _entry_point("working", working)],
    )
    _write_activation_config(
        home, enabled=["model-providers/failing", "model-providers/working"]
    )

    from providers import get_provider_profile

    assert get_provider_profile("package-working").base_url == "https://working.example/v1"


def test_user_directory_claim_suppresses_package_registration(tmp_path, monkeypatch):
    module = "_test_provider_ep_overridden"
    marker = tmp_path / "package-executed"
    _write_marker_module(tmp_path / f"{module}.py", marker)
    monkeypatch.syspath_prepend(str(tmp_path))
    home = _isolate_discovery(
        tmp_path, monkeypatch, [_entry_point("shared-provider", module)]
    )
    _write_activation_config(home, enabled=["model-providers/shared-provider"])
    _write_user_provider(
        home, "shared-provider", "shared-provider", "https://user.example/v1"
    )

    from providers import get_provider_profile

    assert get_provider_profile("shared-provider").base_url == "https://user.example/v1"
    assert not marker.exists()


@pytest.mark.parametrize(
    ("config_text", "executes"),
    [
        (None, True),
        ("plugins:\n  enabled: []\n  disabled: []\n", True),
        ("plugins:\n  enabled: []\n  disabled: [model-providers/directory]\n", False),
    ],
)
def test_user_directory_activation_matrix(tmp_path, monkeypatch, config_text, executes):
    home = _isolate_discovery(tmp_path, monkeypatch)
    if config_text is not None:
        (home / "config.yaml").write_text(config_text)
    marker = tmp_path / "directory-executed"
    plugin_dir = home / "plugins" / "model-providers" / "directory"
    plugin_dir.mkdir(parents=True)
    _write_marker_module(plugin_dir / "__init__.py", marker)

    from providers import list_providers

    list_providers()
    assert marker.exists() is executes


@pytest.mark.parametrize(
    "config_text",
    [
        "plugins: [not valid\n",
        "plugins: []\n",
        "plugins:\n  enabled:\n    model-providers/shape-evil: true\n",
        "plugins:\n  enabled: model-providers/shape-evil\n",
        "plugins:\n  enabled: []\n  disabled:\n    shape-evil: true\n",
    ],
)
def test_external_providers_fail_closed_for_invalid_activation_config(
    tmp_path, monkeypatch, config_text
):
    package_marker = tmp_path / "package-executed"
    module = "_test_provider_ep_shape_evil"
    _write_marker_module(tmp_path / f"{module}.py", package_marker)
    monkeypatch.syspath_prepend(str(tmp_path))
    home = _isolate_discovery(
        tmp_path, monkeypatch, [_entry_point("shape-evil", module)]
    )
    (home / "config.yaml").write_text(config_text)
    user_marker = tmp_path / "user-executed"
    plugin_dir = home / "plugins" / "model-providers" / "shape-evil"
    plugin_dir.mkdir(parents=True)
    _write_marker_module(plugin_dir / "__init__.py", user_marker)

    from providers import list_providers

    list_providers()
    assert not package_marker.exists()
    assert not user_marker.exists()


def test_duplicate_package_names_execute_neither_and_are_absent_from_inventory(
    tmp_path, monkeypatch
):
    markers = [tmp_path / "duplicate-a", tmp_path / "duplicate-b"]
    modules = ["_test_provider_ep_duplicate_a", "_test_provider_ep_duplicate_b"]
    for module, marker in zip(modules, markers):
        _write_marker_module(tmp_path / f"{module}.py", marker)
    monkeypatch.syspath_prepend(str(tmp_path))
    home = _isolate_discovery(
        tmp_path, monkeypatch, [_entry_point("duplicate", module) for module in modules]
    )
    _write_activation_config(home, enabled=["model-providers/duplicate"])

    from hermes_cli.plugins_cmd import _discover_model_provider_entrypoints
    from providers import list_providers

    assert _discover_model_provider_entrypoints() == []
    list_providers()
    assert not any(marker.exists() for marker in markers)


def test_package_metadata_enumeration_failure_is_isolated(tmp_path, monkeypatch):
    _isolate_discovery(tmp_path, monkeypatch)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("metadata unavailable")),
    )

    from providers import list_providers

    assert list_providers() == []
