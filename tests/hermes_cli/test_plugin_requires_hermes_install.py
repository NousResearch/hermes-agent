"""``hermes plugins install`` refuses a plugin whose ``requires_hermes`` rejects the running version.

A portable ``plugin.json`` declares it at ``extensions."com.nousresearch.hermes".requires_hermes``; a
catalog entry declares it on the entry. The refusal names the spec, the running version and the update
command, and runs before server declarations are parsed, so a plugin written for a newer Hermes is
refused with "update Hermes" instead of a parse error. Real git, file:// repos, temp HERMES_HOME.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess as sp
from pathlib import Path

import pytest

from hermes_cli import plugin_catalog as pc_cat
from hermes_cli import plugins_cmd as pc
from hermes_cli import plugins_cmd_catalog as cat
from tests.pm._fixtures import client, isolated_python  # noqa: F401

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
_RUNNING = "1.2.3"


def _portable_repo(repo: Path, requires_hermes: str, *, future_server_shape: bool = False) -> str:
    """A committed portable plugin repo; returns its HEAD sha. *future_server_shape* adds a server
    declaration key this client rejects, standing in for a shape a newer Hermes introduced."""
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1

    repo.mkdir()
    namespace: dict = {"requires_hermes": requires_hermes}
    if future_server_shape:
        namespace["servers"] = {"worker": {"requires": {"app": True}, "shape-from-the-future": {}}}
    (repo / "plugin.json").write_text(json.dumps({
        "$schema": PLUGIN_SCHEMA_V1, "name": "gated-plugin",
        "extensions": {"com.nousresearch.hermes": namespace},
    }), encoding="utf-8")
    (repo / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1, "mcpServers": {"worker": {"type": "stdio", "command": "python"}},
    }), encoding="utf-8")
    sp.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    sp.run(["git", "add", "-A"], cwd=repo, check=True, env=_GIT_ENV)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=_GIT_ENV)
    return sp.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def plugins_dir(request, tmp_path, monkeypatch):
    request.getfixturevalue("client")  # isolated PM home for publication
    home = tmp_path / "home"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: False)
    monkeypatch.setattr("hermes_cli.plugins_manifest.running_hermes_version", lambda: _RUNNING)
    # Pin the update command to the git-checkout answer (see test_managed_installs.py).
    monkeypatch.setattr("hermes_cli.config.get_managed_update_command", lambda: None)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda *_a, **_k: "git")
    return plugins


def _assert_refused_and_nothing_installed(exc: pytest.ExceptionInfo, plugins: Path, spec: str) -> None:
    message = str(exc.value)
    assert spec in message and _RUNNING in message and "hermes update" in message, message
    assert not (plugins / "gated-plugin").exists()
    assert [p.name for p in plugins.iterdir() if not p.name.startswith(".")] == []
    assert "gated-plugin" not in pc._read_install_metadata()


def test_portable_requires_hermes_refuses_before_server_declarations_are_parsed(plugins_dir, tmp_path):
    repo = tmp_path / "repo"
    _portable_repo(repo, ">=99.0.0", future_server_shape=True)

    with pytest.raises(pc.PluginOperationError) as exc:
        pc._install_plugin_core(repo.as_uri(), force=False)

    _assert_refused_and_nothing_installed(exc, plugins_dir, ">=99.0.0")
    assert "unavailable" not in str(exc.value)


def test_portable_requires_hermes_satisfied_installs(plugins_dir, tmp_path):
    repo = tmp_path / "repo"
    # PM publication re-checks the spec in a worker process at the real running version.
    _portable_repo(repo, ">=0.1.0, <99.0.0")

    target, manifest, name = pc._install_plugin_core(repo.as_uri(), force=False)

    assert name == "gated-plugin"
    assert target == (plugins_dir / "gated-plugin").resolve()
    assert (target / "plugin.json").is_file()


def test_catalog_entry_requires_hermes_refuses(plugins_dir, tmp_path):
    repo = tmp_path / "repo"
    sha = _portable_repo(repo, "")
    entry = pc_cat.PluginCatalogEntry(name="gated-plugin", repo=repo.as_uri(), sha=sha, description="d",
                                      maintainer="t", requires_hermes=">=99.0.0")

    with pytest.raises(pc.PluginOperationError) as exc:
        cat.install_catalog_entry(entry, force=False)

    _assert_refused_and_nothing_installed(exc, plugins_dir, ">=99.0.0")


def test_catalog_repin_to_entry_requiring_newer_hermes_refuses(plugins_dir, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    old_sha = _portable_repo(repo, "")
    entry = pc_cat.PluginCatalogEntry(name="gated-plugin", repo=repo.as_uri(), sha=old_sha, description="d",
                                      maintainer="t")
    target, _manifest, _name = cat.install_catalog_entry(entry, force=False)
    (repo / "NEW").write_text("0.2.0", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True, env=_GIT_ENV)
    sp.run(["git", "commit", "-q", "-m", "0.2.0"], cwd=repo, check=True, env=_GIT_ENV)
    new_sha = sp.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    moved = pc_cat.PluginCatalogEntry(name="gated-plugin", repo=repo.as_uri(), sha=new_sha, description="d",
                                      maintainer="t", requires_hermes=">=99.0.0")
    monkeypatch.setattr("hermes_cli.plugins_cmd_catalog.get_live_catalog_entry", lambda _n: moved)

    with pytest.raises(pc.PluginOperationError) as exc:
        cat.repin_catalog_plugin(target, cat.catalog_install_record(target))

    message = str(exc.value)
    assert ">=99.0.0" in message and _RUNNING in message and "hermes update" in message, message
    assert not (target / "NEW").exists()
    assert cat.catalog_install_record(target)["sha"] == old_sha
