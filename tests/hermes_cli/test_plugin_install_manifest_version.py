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
import hermes_yaml as yaml

from hermes_cli.plugins import SUPPORTED_MANIFEST_VERSION
from tests.pm._fixtures import client, isolated_python  # noqa: F401


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
    client, monkeypatch, tmp_path, manifest_version
):
    from hermes_cli import plugins_cmd

    monkeypatch.setattr(plugins_cmd, "_scan_on_install_enabled", lambda: False)
    repo = _plugin_repo(
        tmp_path,
        {"name": "demo", "version": "1.0.0", "manifest_version": manifest_version},
    )
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    target, manifest, name = plugins_cmd._install_plugin_core(repo.as_uri(), force=False)

    assert name == "demo"
    assert target.exists()
    assert int(manifest["manifest_version"]) == manifest_version


def test_manifest_version_above_shared_support_is_refused_cleanly(
    client, monkeypatch, tmp_path
):
    from hermes_cli import plugins_cmd

    monkeypatch.setattr(plugins_cmd, "_scan_on_install_enabled", lambda: False)
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
        plugins_cmd.PluginOperationError,
        match=rf"supports up to {SUPPORTED_MANIFEST_VERSION}\b",
    ):
        plugins_cmd._install_plugin_core(repo.as_uri(), force=False)

    assert not (home / "plugins" / "demo").exists()
    assert not (home / "plugins" / ".install-metadata.json").exists()


def _refuse(monkeypatch, manifest: dict) -> str:
    """Drive ``_check_manifest_version`` directly; the message is the deliverable (#130656)."""
    from hermes_cli import plugins_cmd, plugins_cmd_install, plugins_manifest

    monkeypatch.setattr(plugins_cmd, "PluginOperationError", RuntimeError)
    monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: "0.21.5")
    with pytest.raises(RuntimeError) as excinfo:
        plugins_cmd_install._check_manifest_version(manifest, "demo")
    return str(excinfo.value)


def test_calver_requires_hermes_floor_refusal_does_not_recommend_update(monkeypatch):
    """A floor written in the release-tag (CalVer) space can never be satisfied by a base_version
    release, so the refusal must say that instead of advising an update that cannot succeed (#130656)."""
    for spec in (">=2026.9.24", ">=v2026.9.24"):
        message = _refuse(monkeypatch, {"requires_hermes": spec})
        assert "cannot satisfy" in message, message
        assert "Run " not in message  # an update cannot fix a wrong-space floor


def test_reachable_requires_hermes_floor_refusal_still_recommends_update(monkeypatch):
    """A floor the running base_version merely doesn't meet yet is fixable by updating."""
    message = _refuse(monkeypatch, {"requires_hermes": ">=99.0"})
    assert "Run " in message
    assert "cannot satisfy" not in message
