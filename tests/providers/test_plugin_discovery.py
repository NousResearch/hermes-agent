"""Tests for the model-providers plugin discovery system.

Verifies that:
 1. All bundled providers at plugins/model-providers/<name>/ are discovered
 2. User plugins at $HERMES_HOME/plugins/model-providers/<name>/ override bundled
 3. plugin.yaml manifests with kind=model-provider are correctly categorized
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hermes_agent.compat.paths import current_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clear_provider_caches():
    """Force hermes_agent/providers/__init__.py to re-discover on next list_providers()."""
    import hermes_agent.providers as _pkg
    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    # Evict any cached plugin modules so the next import re-executes.
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
        ):
            del sys.modules[mod]


def test_bundled_plugins_discovered():
    """Every plugins/model-providers/<name>/ should contain a plugin.yaml + __init__.py."""
    plugins_dir = REPO_ROOT / current_path("plugins") / "model-providers"
    assert plugins_dir.is_dir(), f"Missing {plugins_dir}"

    child_dirs = [c for c in plugins_dir.iterdir() if c.is_dir()]
    assert child_dirs, f"No provider plugins under {plugins_dir}"

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
    from hermes_agent.providers import list_providers

    plugins_dir = REPO_ROOT / current_path("plugins") / "model-providers"
    plugin_dir_count = sum(1 for c in plugins_dir.iterdir() if c.is_dir())

    profiles = list_providers()
    names = sorted(p.name for p in profiles)
    # Some plugin __init__.py files register multiple profiles, so the registry
    # count is >= the directory count (never less).
    assert len(names) >= plugin_dir_count, (
        f"Expected at least {plugin_dir_count} profiles (one per plugin dir), got {len(names)}: {names}"
    )


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
        "from hermes_agent.providers import register_provider\n"
        "from hermes_agent.providers.base import ProviderProfile\n"
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
    from hermes_agent.providers import get_provider_profile

    gmi = get_provider_profile("gmi")
    assert gmi is not None
    assert gmi.base_url == "https://user-override.example.com/v1", (
        f"User override not applied; got base_url={gmi.base_url!r}"
    )
    assert "gmi-user-override-test" in gmi.aliases

    # Clean up: reset discovery state so other tests see the bundled version
    _clear_provider_caches()


def test_provider_file_in_the_old_providers_folder_still_registers(tmp_path, monkeypatch):
    """``<repo>/providers/<name>.py`` is the documented legacy drop-in; the move must keep it loading."""
    import hermes_agent.providers as _pkg
    import hermes_agent.hermes_cli._startup_fast as startup_fast

    (tmp_path / "providers").mkdir()
    (tmp_path / "providers" / "legacy_dropin.py").write_text(
        "from hermes_agent.providers import register_provider\n"
        "from hermes_agent.providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='legacy-dropin'))\n"
    )
    monkeypatch.setattr(startup_fast, "project_root_str", lambda: str(tmp_path))
    monkeypatch.setattr(_pkg, "__path__", list(_pkg.__path__))
    monkeypatch.setattr(_pkg, "_SOURCES", {})
    _pkg._add_old_folder()
    _clear_provider_caches()
    try:
        assert _pkg.get_provider_profile("legacy-dropin") is not None
    finally:
        sys.modules.pop("hermes_agent.providers.legacy_dropin", None)
        vars(_pkg).pop("legacy_dropin", None)
        _clear_provider_caches()


def test_old_folder_module_imports_by_its_old_name_in_a_fresh_process(tmp_path):
    """Before the move, ``import providers.<name>`` loaded a drop-in with no discovery pass first."""
    (tmp_path / "providers").mkdir()
    (tmp_path / "providers" / "legacy_direct.py").write_text("VALUE = 'loaded'\n")
    script = (
        "import hermes_agent.hermes_cli._startup_fast as startup_fast\n"
        f"startup_fast.project_root_str = lambda: {str(tmp_path)!r}\n"
        "from providers.legacy_direct import VALUE\n"
        "print(VALUE)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.stdout.strip() == "loaded", result.stderr
