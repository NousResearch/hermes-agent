"""Who may register a password-manager login backend (``ctx.register_login_backend``).

Only plugins bundled with Hermes, or installed from the plugin catalog at tier ``verified`` or ``official``
and at the catalog's reviewed pin, are accepted. Provenance comes from the installer-owned
``.install-metadata.json`` record, never from the plugin tree. Real git, file:// repos, the real installer,
and the real plugin manager loading into a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess as sp
from pathlib import Path

import pytest

from hermes_cli import plugin_catalog as pc_cat
from hermes_cli import plugins_cmd as pc
from hermes_cli import plugins_cmd_catalog as cat
from tests.pm._fixtures import client, isolated_python  # pytest fixtures: PM-isolated HOME/HERMES_HOME

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

_PLUGIN = '''
from agent.vault_backends.base import LoginBackend


class Backend(LoginBackend):
    name, prefix, display_name = "acme", "acme:", "Acme"

    def list_items(self):
        return []

    def get_meta(self, handle):
        return None

    def resolve_password(self, handle):
        raise AssertionError("provenance tests never resolve passwords")


def register(ctx):
    ctx.register_login_backend(Backend())
'''

RULE = "verified or official"


def _commit(repo: Path, msg: str) -> str:
    sp.run(["git", "add", "-A"], cwd=repo, check=True, env=_GIT_ENV)
    sp.run(["git", "commit", "-q", "-m", msg], cwd=repo, check=True, env=_GIT_ENV)
    return sp.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def world(request, tmp_path, monkeypatch):
    """A file:// login-backend plugin repo (two commits), a catalog entry pinned to the first, a temp home."""
    request.getfixturevalue("client")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "plugin.yaml").write_text("name: acme-vault\nversion: 1.0.0\ndescription: d\n")
    (repo / "__init__.py").write_text(_PLUGIN)
    sp.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    sha1 = _commit(repo, "v1")
    (repo / "__init__.py").write_text(_PLUGIN + "# v2\n")
    sha2 = _commit(repo, "v2")

    home = tmp_path / "home"
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: False)
    monkeypatch.setattr(pc, "_console", lambda: type("C", (), {"print": lambda *a, **k: None})())

    def publish_without_environment(*_args, plugins=None, **_kwargs):
        from hermes_cli.runtime_state import finish_publication
        from pm import paths
        from pm.plugin_inputs import StagedUpdate
        from pm.publication import PluginSelection, StagedPlugin

        change = StagedPlugin(dict(plugins.data)) if isinstance(plugins, StagedUpdate) else PluginSelection(
            dict(plugins.data))
        change.publish(paths.repo_root())
        finish_publication(paths.repo_root())

    monkeypatch.setattr("pm.client.sync_venv", publish_without_environment)
    monkeypatch.setattr(pc_cat, "fetch_live_catalog", lambda **_: None)  # in-tree only, no network
    monkeypatch.setattr(pc_cat, "load_removed_list", lambda catalog_dir=None: [])
    tier = {"value": "verified"}

    def entry() -> pc_cat.PluginCatalogEntry:
        return pc_cat.PluginCatalogEntry(name="acme-vault", repo=repo.as_uri(), sha=sha1, description="d",
                                         maintainer="Acme", tier=tier["value"])

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: [entry()])
    managers = []

    def load():
        from agent import vault_backend_registry as registry
        from hermes_cli.plugins import PluginManager

        (home / "config.yaml").write_text("plugins:\n  enabled: [acme-vault]\nvault:\n  acme: {enabled: true}\n")
        manager = PluginManager(scope_key=str(home))
        manager.discover_and_load()
        managers.append(manager)
        return registry.get_provider("acme", scope=str(home))

    try:
        yield {"repo": repo, "sha1": sha1, "sha2": sha2, "home": home, "plugins_dir": plugins_dir,
               "tier": tier, "entry": entry, "load": load}
    finally:
        for manager in managers:
            manager.unload()
        reset_hermes_home_override(token)


@pytest.mark.parametrize("tier", ["verified", "official"])
def test_catalog_install_at_verified_or_official_tier_registers(world, tier):
    world["tier"]["value"] = tier
    cat.install_catalog_entry(world["entry"](), force=False)
    backend = world["load"]()
    assert backend is not None and backend.name == "acme"


def test_community_catalog_install_is_refused(world, caplog):
    world["tier"]["value"] = "community"
    cat.install_catalog_entry(world["entry"](), force=False)
    with caplog.at_level(logging.WARNING):
        assert world["load"]() is None
    assert RULE in caplog.text and "tier community" in caplog.text


def test_local_directory_plugin_is_refused(world, caplog):
    plugin = world["plugins_dir"] / "acme-vault"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: acme-vault\n")
    (plugin / "__init__.py").write_text(_PLUGIN)
    with caplog.at_level(logging.WARNING):
        assert world["load"]() is None
    assert RULE in caplog.text and "not installed from the Hermes plugin catalog" in caplog.text


def test_url_install_with_forged_in_tree_catalog_file_is_refused(world, caplog):
    """A repo can commit a ``.hermes-catalog.json`` claiming a verified entry; Python never reads it."""
    (world["repo"] / ".hermes-catalog.json").write_text(json.dumps(
        {"catalog_name": "acme-vault", "repo": world["repo"].as_uri(), "sha": world["sha1"], "tier": "verified"}))
    forged = _commit(world["repo"], "forge provenance")
    pc._install_plugin_core(world["repo"].as_uri(), force=False, ref=forged)
    assert json.loads((world["plugins_dir"] / "acme-vault" / ".hermes-catalog.json").read_text())["tier"] == "verified"
    with caplog.at_level(logging.WARNING):
        assert world["load"]() is None
    assert RULE in caplog.text


def test_verified_entry_installed_off_the_reviewed_pin_is_refused(world, caplog):
    cat.install_catalog_entry(world["entry"](), force=False, ref=world["sha2"])
    with caplog.at_level(logging.WARNING):
        assert world["load"]() is None
    assert RULE in caplog.text and "reviewed pin" in caplog.text


def test_verified_entry_installed_with_allow_removed_is_refused(world, caplog):
    cat.install_catalog_entry(world["entry"](), force=False, allow_removed=True)
    with caplog.at_level(logging.WARNING):
        assert world["load"]() is None
    assert RULE in caplog.text and "--allow-removed" in caplog.text


@pytest.mark.parametrize("source, accepted", [("bundled", True), ("project", False), ("entrypoint", False)])
def test_bundled_plugins_register_and_project_or_pip_plugins_do_not(tmp_path, source, accepted):
    from agent import vault_backend_registry as registry
    from agent.vault_backends.base import LoginBackend
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    class Backend(LoginBackend):
        name, prefix, display_name = "acme", "acme:", "Acme"

        def list_items(self):
            return []

        def get_meta(self, handle):
            return None

        def resolve_password(self, handle):
            raise AssertionError("never resolved")

    scope = str(tmp_path / "home")
    manifest = PluginManifest(name="acme-vault", source=source, path=str(tmp_path / "acme-vault"))
    handle = PluginContext(manifest, PluginManager(scope_key=scope)).register_login_backend(Backend())
    try:
        assert (registry.get_provider("acme", scope=scope) is not None) is accepted
    finally:
        if handle is not None:
            handle.dispose()
