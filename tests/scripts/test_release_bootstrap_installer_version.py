"""release.py must stamp bootstrap-installer versions with the release semver.

Tauri CFBundleShortVersionString is read from
apps/bootstrap-installer/src-tauri/tauri.conf.json (and the sibling
package.json). Those files were hardcoded 0.0.1 and omitted from
update_version_files / the --publish --bump git add list, so Hermes-Setup.dmg
always shipped 0.0.1. The root package-lock workspace entry had the same
drift and was also omitted from the release stage list. Same class as the
desktop stamp (#68783 / PR #68796).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "release_bootstrap_installer_version", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


release = _load()


def _json_version(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["version"]


def _patch_repo(tmp_path, monkeypatch, *, with_installer: bool = True):
    repo = tmp_path
    init_py = repo / "hermes_cli" / "__init__.py"
    init_py.parent.mkdir(parents=True)
    init_py.write_text(
        '__version__ = "0.0.1"\n__release_date__ = "2026.1.1"\n',
        encoding="utf-8",
    )
    pyproject = repo / "pyproject.toml"
    pyproject.write_text('version = "0.0.1"\n', encoding="utf-8")

    desktop_pkg = repo / "apps" / "desktop" / "package.json"
    desktop_pkg.parent.mkdir(parents=True)
    desktop_pkg.write_text('{"version":"0.0.1"}\n', encoding="utf-8")

    installer_pkg = repo / "apps" / "bootstrap-installer" / "package.json"
    tauri_conf = (
        repo / "apps" / "bootstrap-installer" / "src-tauri" / "tauri.conf.json"
    )
    cargo_toml = repo / "apps" / "bootstrap-installer" / "src-tauri" / "Cargo.toml"
    if with_installer:
        tauri_conf.parent.mkdir(parents=True)
        installer_pkg.write_text(
            '{"name":"x","version":"0.0.1"}\n', encoding="utf-8"
        )
        tauri_conf.write_text(
            '{"productName":"Hermes","version":"0.0.1"}\n', encoding="utf-8"
        )
        cargo_toml.write_text('[package]\nversion = "0.0.1"\n', encoding="utf-8")

        (repo / "package-lock.json").write_text(
            json.dumps({
                "packages": {
                    "apps/bootstrap-installer": {
                        "name": "@hermes/bootstrap-installer", "version": "0.0.1",
                    },
                    "apps/desktop": {"name": "hermes", "version": "0.0.1"},
                },
            }) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(release, "REPO_ROOT", repo)
    monkeypatch.setattr(release, "VERSION_FILE", init_py)
    monkeypatch.setattr(release, "PYPROJECT_FILE", pyproject)
    return {
        "repo": repo,
        "init_py": init_py,
        "pyproject": pyproject,
        "desktop_pkg": desktop_pkg,
        "installer_pkg": installer_pkg,
        "package_lock": repo / "package-lock.json",
        "tauri_conf": tauri_conf,
        "cargo_toml": cargo_toml,
    }


def test_update_version_files_stamps_bootstrap_installer(tmp_path, monkeypatch):
    paths = _patch_repo(tmp_path, monkeypatch)

    release.update_version_files("0.21.1", "2026.9.10")

    assert _json_version(paths["installer_pkg"]) == "0.21.1"
    lock = json.loads(paths["package_lock"].read_text(encoding="utf-8"))
    assert lock["packages"]["apps/bootstrap-installer"]["version"] == "0.21.1"
    assert lock["packages"]["apps/desktop"]["version"] == "0.21.1"
    assert _json_version(paths["tauri_conf"]) == "0.21.1"
    assert 'version = "0.21.1"' in paths["cargo_toml"].read_text(encoding="utf-8")

    # CONTROL: existing desktop / Python stamps still happen.
    assert _json_version(paths["desktop_pkg"]) == "0.21.1"
    assert 'version = "0.21.1"' in paths["pyproject"].read_text(encoding="utf-8")
    init_text = paths["init_py"].read_text(encoding="utf-8")
    assert '__version__ = "0.21.1"' in init_text
    assert '__release_date__ = "2026.9.10"' in init_text


def test_update_version_files_skips_missing_installer_dir(tmp_path, monkeypatch):
    paths = _patch_repo(tmp_path, monkeypatch, with_installer=False)

    release.update_version_files("0.21.1", "2026.9.10")

    assert not paths["installer_pkg"].exists()
    assert not paths["tauri_conf"].exists()
    assert _json_version(paths["desktop_pkg"]) == "0.21.1"
    assert 'version = "0.21.1"' in paths["pyproject"].read_text(encoding="utf-8")
    assert '__version__ = "0.21.1"' in paths["init_py"].read_text(encoding="utf-8")


def test_update_version_files_does_not_invent_version_keys(tmp_path, monkeypatch):
    paths = _patch_repo(tmp_path, monkeypatch)
    original = '{"name":"x","productName":"Hermes"}\n'
    paths["installer_pkg"].write_text(original, encoding="utf-8")
    paths["tauri_conf"].write_text(original, encoding="utf-8")
    original_lock = json.dumps({
        "packages": {
            "apps/desktop": {"name": "hermes", "dependencies": {"version": "1.2.3"}},
            "node_modules/example": {"version": "4.5.6"},
        },
    }) + "\n"
    paths["package_lock"].write_text(original_lock, encoding="utf-8")

    release.update_version_files("0.21.1", "2026.9.10")

    assert paths["installer_pkg"].read_text(encoding="utf-8") == original
    assert paths["tauri_conf"].read_text(encoding="utf-8") == original
    assert _json_version(paths["desktop_pkg"]) == "0.21.1"
    assert paths["package_lock"].read_text(encoding="utf-8") == original_lock


@pytest.mark.parametrize("indent", [None, 4])
def test_workspace_lock_stamps_preserve_dependency_resolutions(tmp_path, monkeypatch, indent):
    paths = _patch_repo(tmp_path, monkeypatch)
    dependency = {
        "version": "7.8.9",
        "resolved": "https://registry.example.test/example/-/example-7.8.9.tgz",
        "integrity": "sha512-test-integrity",
    }
    original = {
        "name": "workspace-root",
        "version": "1.0.0",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "workspace-root", "version": "1.0.0"},
            "apps/desktop": {
                "dependencies": {"example": "7.8.9"}, "version": "0.1.0", "name": "hermes",
            },
            "apps/bootstrap-installer": {
                "dependencies": {"example": "7.8.9"}, "version": "0.2.0",
            },
            "node_modules/example": dependency,
            "apps/desktop/node_modules/example": dependency,
        },
    }
    paths["package_lock"].write_text(json.dumps(original, indent=indent) + "\n", encoding="utf-8")

    release.update_version_files("1.2.3", "2026.9.11")

    updated = json.loads(paths["package_lock"].read_text(encoding="utf-8"))
    for workspace in ("apps/desktop", "apps/bootstrap-installer"):
        assert updated["packages"][workspace]["version"] == "1.2.3"
        updated["packages"][workspace]["version"] = original["packages"][workspace]["version"]
    assert updated == original, "Only the two workspace version fields may change"


def test_version_files_to_stage_includes_installer_when_present(tmp_path, monkeypatch):
    paths = _patch_repo(tmp_path, monkeypatch)

    staged = release.version_files_to_stage()

    assert str(paths["init_py"]) in staged
    assert str(paths["pyproject"]) in staged
    assert str(paths["desktop_pkg"]) in staged
    assert str(paths["installer_pkg"]) in staged
    assert str(paths["package_lock"]) in staged
    assert str(paths["tauri_conf"]) in staged
    assert str(paths["cargo_toml"]) in staged


def test_version_files_to_stage_omits_missing_installer(tmp_path, monkeypatch):
    paths = _patch_repo(tmp_path, monkeypatch, with_installer=False)

    staged = release.version_files_to_stage()

    assert str(paths["installer_pkg"]) not in staged
    assert str(paths["tauri_conf"]) not in staged
    assert str(paths["cargo_toml"]) not in staged
    assert str(paths["desktop_pkg"]) in staged
    assert str(paths["init_py"]) in staged
    assert str(paths["pyproject"]) in staged
