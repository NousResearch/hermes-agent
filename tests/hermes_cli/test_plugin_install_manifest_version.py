"""Installer accepts every manifest_version the runtime loader supports (#85879).

The installer used to carry its own private manifest-version cap, which
drifted behind the loader's ``SUPPORTED_MANIFEST_VERSION`` and refused v2
plugins the runtime happily loads. These tests install through the real
clone path so the two can never split again on the next bump.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from hermes_cli.plugins import SUPPORTED_MANIFEST_VERSION


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _plugin_repo(root: Path, manifest: dict) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture")
    (repo / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    return repo


@pytest.mark.parametrize(
    "manifest_version", list(range(1, SUPPORTED_MANIFEST_VERSION + 1))
)
def test_install_accepts_every_loader_supported_manifest_version(
    monkeypatch, tmp_path, manifest_version
):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo = _plugin_repo(
        tmp_path,
        {"name": "demo", "version": "1.0.0", "manifest_version": manifest_version},
    )
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    target, manifest, name = _install_plugin_core(repo.as_uri(), force=False)

    assert name == "demo"
    assert target.exists()
    assert int(manifest["manifest_version"]) == manifest_version


def test_manifest_version_above_shared_support_is_refused_cleanly(
    monkeypatch, tmp_path
):
    from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

    repo = _plugin_repo(
        tmp_path,
        {
            "name": "demo",
            "version": "1.0.0",
            "manifest_version": SUPPORTED_MANIFEST_VERSION + 1,
        },
    )
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(
        PluginOperationError,
        match=rf"supports up to {SUPPORTED_MANIFEST_VERSION}\b",
    ):
        _install_plugin_core(repo.as_uri(), force=False)

    assert not (home / "plugins" / "demo").exists()
    assert not (home / "plugins" / ".install-metadata.json").exists()
