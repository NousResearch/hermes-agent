"""``hermes plugins install`` refuses a plugin whose ``requires_hermes`` rejects the running version.

A portable ``plugin.json`` declares it at ``extensions."com.nousresearch.hermes".requires_hermes``. The
refusal names the spec, the running version and the update command, and runs before server declarations
are parsed, so a plugin written for a newer Hermes is refused with "update Hermes" instead of a parse
error. The loader and ``hermes plugins validate`` read the same key. Real git, file:// repos, temp
HERMES_HOME. The catalog entry's own ``requires_hermes`` is covered by
``test_plugins_catalog_requires_hermes.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess as sp
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_cli import plugins_cmd as pc
from hermes_cli.plugin_validate import validate_plugin_dir
from tests.pm._fixtures import client, isolated_python  # pytest fixtures: PM-isolated HOME/HERMES_HOME

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
_RUNNING = "1.2.3"


def _write_portable(root: Path, requires_hermes: str, *, name: str = "gated-plugin", server: str = "worker",
                    future_server_shape: bool = False) -> Path:
    """A portable package declaring *requires_hermes* in its Hermes extension namespace.
    *future_server_shape* adds a server declaration key this client does not know, standing in for a
    shape a newer Hermes introduced."""
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1

    root.mkdir(parents=True)
    namespace: dict = {"requires_hermes": requires_hermes}
    if future_server_shape:
        namespace["servers"] = {server: {"requires": {"app": True}, "shape-from-the-future": {}}}
    (root / "plugin.json").write_text(json.dumps({
        "$schema": PLUGIN_SCHEMA_V1, "name": name,
        "extensions": {"com.nousresearch.hermes": namespace},
    }), encoding="utf-8")
    (root / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1, "mcpServers": {server: {"type": "stdio", "command": "python"}},
    }), encoding="utf-8")
    return root


def _portable_repo(repo: Path, requires_hermes: str, *, future_server_shape: bool = False) -> str:
    """A committed portable plugin repo; returns its HEAD sha."""
    _write_portable(repo, requires_hermes, future_server_shape=future_server_shape)
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

    target, _manifest, name = pc._install_plugin_core(repo.as_uri(), force=False)

    assert name == "gated-plugin"
    assert target == (plugins_dir / "gated-plugin").resolve()
    assert (target / "plugin.json").is_file()


def test_installed_portable_plugin_outside_requires_hermes_is_skipped_at_load(tmp_path, monkeypatch):
    # An install that passed the gate, then a Hermes downgrade: the loader must not register its servers.
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / "home"
    _write_portable(home / "plugins" / "too-new", ">=99.0.0", name="too-new", server="future-worker")
    _write_portable(home / "plugins" / "fits", ">=1.0, <2", name="fits", server="worker")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["too-new", "fits"]}}))
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr("hermes_cli.plugins_manifest.running_hermes_version", lambda: _RUNNING)

    manager = plugins_mod.PluginManager()
    manager.discover_and_load()

    skipped = manager._plugins["too-new"]
    assert not skipped.enabled
    assert skipped.error == f"requires hermes >=99.0.0, running {_RUNNING}"
    assert list(manager.get_portable_mcp_servers()) == ["worker"]
    assert manager._plugins["fits"].enabled is True


def test_plugins_validate_reads_the_portable_requires_hermes(tmp_path):
    good = validate_plugin_dir(_write_portable(tmp_path / "good", ">=0.22, <1.0"))
    assert good.ok, good.failures
    assert ("requires_hermes", True, "spec '>=0.22, <1.0' parses") in good.checks

    typo = validate_plugin_dir(_write_portable(tmp_path / "typo", ">=0.21.1,<0.x"))
    assert not typo.ok
    assert any("requires_hermes" in f and "does not parse" in f for f in typo.failures), typo.failures
