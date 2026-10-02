"""External builder metadata is stamped without changing inert source versions."""
from __future__ import annotations

import json
import tomllib

import pytest

from scripts.releases import stamping


def _tree(tmp_path, *, installer=True):
    files = {
        "hermes_cli/__init__.py": '__version__ = "0.0.1"\n__release_date__ = "2026.1.1"\n',
        "pyproject.toml": '[project]\nversion = "0.0.1"\n',
        "apps/desktop/package.json": '{"version":"0.0.1"}\n',
        "package-lock.json": '{"packages":{"apps/desktop":{"version":"0.0.1"}}}\n',
        "nix/hermes-agent.nix": '{\n  version ? "0.0.1",\n}: {}\n',
    }
    if installer:
        files.update({
            "apps/bootstrap-installer/package.json": '{"version":"0.0.1"}\n',
            "apps/bootstrap-installer/src-tauri/tauri.conf.json": '{"version":"0.0.1"}\n',
            "apps/bootstrap-installer/src-tauri/Cargo.toml": '[package]\nversion = "0.0.1"\n',
        })
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return {name: (tmp_path / name).read_bytes() for name in files}


def _assert_source_unchanged(root, before):
    for name, contents in before.items():
        if name.startswith("nix/") or name.startswith("apps/bootstrap-installer/src-tauri/"):
            continue
        assert (root / name).read_bytes() == contents, name


def test_update_version_files_stamps_bootstrap_installer(tmp_path):
    before = _tree(tmp_path)
    written = stamping.stamp(tmp_path, "0.21.1")
    installer = tmp_path / "apps/bootstrap-installer/src-tauri"
    assert json.loads((installer / "tauri.conf.json").read_text())["version"] == "0.21.1"
    assert tomllib.loads((installer / "Cargo.toml").read_text())["package"]["version"] == "0.21.1"
    assert (installer / "tauri.conf.json") in written and (installer / "Cargo.toml") in written
    _assert_source_unchanged(tmp_path, before)


def test_update_version_files_skips_missing_installer_dir(tmp_path):
    before = _tree(tmp_path, installer=False)
    assert stamping.stamp(tmp_path, "0.21.1") == [tmp_path / "nix/hermes-agent.nix"]
    assert not (tmp_path / "apps/bootstrap-installer").exists()
    _assert_source_unchanged(tmp_path, before)


def test_update_version_files_does_not_invent_version_keys(tmp_path):
    before = _tree(tmp_path)
    path = tmp_path / "apps/bootstrap-installer/src-tauri/tauri.conf.json"
    original = '{"name":"x","productName":"Hermes"}\n'
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="Bootstrap installer version differs"):
        stamping.stamp(tmp_path, "0.21.1")
    assert path.read_text() == original
    _assert_source_unchanged(tmp_path, before)


@pytest.mark.parametrize("indent", [None, 4])
def test_workspace_lock_stamps_preserve_dependency_resolutions(tmp_path, indent):
    _tree(tmp_path)
    dependency = {"version": "7.8.9", "resolved": "https://registry.example.test/example.tgz", "integrity": "fixture"}
    original = {"lockfileVersion": 3, "packages": {
        "apps/desktop": {"version": "0.1.0"}, "apps/bootstrap-installer": {"version": "0.2.0"},
        "node_modules/example": dependency, "apps/desktop/node_modules/example": dependency}}
    path = tmp_path / "package-lock.json"
    path.write_text(json.dumps(original, indent=indent) + "\n", encoding="utf-8")
    before = path.read_bytes()
    stamping.stamp(tmp_path, "1.2.3")
    assert path.read_bytes() == before and json.loads(path.read_text()) == original


def test_version_files_to_stage_includes_installer_when_present(tmp_path):
    _tree(tmp_path)
    assert {path.relative_to(tmp_path).as_posix() for path in stamping.stamp(tmp_path, "1.2.3")} == {
        "nix/hermes-agent.nix", "apps/bootstrap-installer/src-tauri/tauri.conf.json",
        "apps/bootstrap-installer/src-tauri/Cargo.toml"}


def test_version_files_to_stage_omits_missing_installer(tmp_path):
    _tree(tmp_path, installer=False)
    assert stamping.stamp(tmp_path, "1.2.3") == [tmp_path / "nix/hermes-agent.nix"]
