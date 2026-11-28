"""An unattended plugin install never lifts a ``plugins.disabled`` entry, also one written during its download.

The default installer, the left-core migrator and the gateway-start catalog platform recovery all install
through one seam (``left_core_migration._install_into``), whose enable step used to write the plugin into
``plugins.enabled`` and purge it from ``plugins.disabled`` whatever the operator had put there while the
clone ran. Real catalog install path (real clone of a local repo, real enable and config writes); only the
https transport is redirected, and the "download" is held while the operator's disable lands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import hermes_yaml as yaml
from tests.hermes_cli.plugin_worker_support import (
    isolated_python as isolated_python,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
    plugin_world as plugin_world,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
)

URL = "https://example.invalid/acme/demo.git"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _write_disabled(home: Path, names: list[str]) -> None:
    """What ``hermes config set plugins.disabled`` leaves in config.yaml."""
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    config = read_user_config_raw(home / "config.yaml")
    config.setdefault("plugins", {})["disabled"] = names
    atomic_config_write(home / "config.yaml", config)


def _plugins(home: Path) -> dict:
    return (yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}).get("plugins") or {}


@pytest.fixture
def world(plugin_world, tmp_path, monkeypatch):
    """``(home, during_download)``: catalog entry ``demo`` served from a local repo; callables appended to
    *during_download* run while its clone is in flight."""
    from hermes_cli import plugins_cmd, plugins_cmd_catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "plugin.yaml").write_text(yaml.safe_dump({"name": "demo", "version": "1.0.0"}), encoding="utf-8")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v1")
    entry = PluginCatalogEntry(name="demo", repo=URL, sha=_git(repo, "rev-parse", "HEAD"), description="d",
                               maintainer="m", tier="official", default=True)
    monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda name: entry if name == "demo" else None)
    during_download: list = []
    real = plugins_cmd._clone_plugin_repo

    def clone(dest, _url, revision, subdir=None):
        for action in during_download:
            action()
        return real(dest, repo.as_uri(), revision, subdir)
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo", clone)
    return plugin_world.home, entry, during_download


def test_a_disable_written_during_an_automatic_install_keeps_the_plugin_off(world):
    from hermes_cli import default_plugins
    home, entry, during_download = world
    during_download.append(lambda: _write_disabled(home, ["demo"]))
    said: list[str] = []

    outcome = default_plugins.install_defaults(home, say=said.append, entries=[entry])

    assert (home / "plugins" / "demo" / "plugin.yaml").is_file(), "the install itself still lands"
    assert outcome["installed"] == ["demo"] and not outcome["owed"], outcome
    plugins = _plugins(home)
    assert plugins.get("disabled") == ["demo"], f"the operator's disable was lifted: {plugins}"
    assert "demo" not in (plugins.get("enabled") or []), plugins
    assert any("is installed but off" in line and "`hermes plugins enable demo`" in line for line in said), said
    assert default_plugins.install_defaults(home, say=said.append, entries=[entry]) == {"installed": [], "owed": []}


def test_a_users_own_install_with_enable_still_lifts_the_disable(world):
    """``hermes plugins install --enable`` (and the dashboard's Install) is the user's explicit request."""
    from hermes_cli.plugins_cmd_install import dashboard_install_plugin
    home, _entry, during_download = world
    during_download.append(lambda: _write_disabled(home, ["demo"]))

    result = dashboard_install_plugin("", force=False, enable=True, catalog_name="demo")

    assert result["ok"] and result["enabled"], result
    plugins = _plugins(home)
    assert "demo" in plugins.get("enabled", []) and "demo" not in (plugins.get("disabled") or []), plugins
