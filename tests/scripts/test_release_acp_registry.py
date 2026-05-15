"""Tests for the ACP Registry version-lockstep bump in scripts/release.py.

The official ACP Registry manifest, the @nousresearch/hermes-agent-acp npm
package, and the npm launcher's HERMES_AGENT_VERSION constant must all match
``pyproject.toml`` exactly — ``tests/acp/test_registry_manifest.py`` enforces
this at lint time. The release script is the single place that bumps them in
lockstep with pyproject; if that bump ever silently breaks, weekly releases
fail the manifest test until someone hand-edits four files.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_release_module(monkeypatch, tmp_root: Path):
    """Import scripts/release.py with REPO_ROOT pinned to a temp tree."""
    spec = importlib.util.spec_from_file_location(
        "_release_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "release.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Repoint every REPO_ROOT-derived path at our temp tree.
    monkeypatch.setattr(module, "REPO_ROOT", tmp_root)
    monkeypatch.setattr(
        module, "ACP_REGISTRY_MANIFEST", tmp_root / "acp_registry" / "agent.json"
    )
    monkeypatch.setattr(
        module,
        "ACP_NPM_PACKAGE_JSON",
        tmp_root / "packages" / "hermes-agent-acp" / "package.json",
    )
    monkeypatch.setattr(
        module,
        "ACP_NPM_LAUNCHER",
        tmp_root / "packages" / "hermes-agent-acp" / "bin" / "hermes-agent-acp.js",
    )
    return module


def _write_fixture(root: Path, version: str) -> None:
    """Write the three ACP-registry files we expect release.py to bump."""
    manifest_dir = root / "acp_registry"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "agent.json").write_text(
        json.dumps(
            {
                "id": "hermes-agent",
                "name": "Hermes Agent",
                "version": version,
                "description": "test",
                "distribution": {
                    "npx": {"package": f"@nousresearch/hermes-agent-acp@{version}"}
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    package_dir = root / "packages" / "hermes-agent-acp"
    (package_dir / "bin").mkdir(parents=True)
    (package_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "@nousresearch/hermes-agent-acp",
                "version": version,
                "bin": {"hermes-agent-acp": "bin/hermes-agent-acp.js"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (package_dir / "bin" / "hermes-agent-acp.js").write_text(
        f"const HERMES_AGENT_VERSION = '{version}';\n"
        f"const HERMES_SPEC = `hermes-agent[acp]==${{HERMES_AGENT_VERSION}}`;\n",
        encoding="utf-8",
    )


def test_update_acp_registry_versions_bumps_all_three_files(monkeypatch, tmp_path):
    _write_fixture(tmp_path, "0.13.0")
    module = _load_release_module(monkeypatch, tmp_path)

    module._update_acp_registry_versions("0.14.0")

    manifest = json.loads(
        (tmp_path / "acp_registry" / "agent.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == "0.14.0"
    assert (
        manifest["distribution"]["npx"]["package"]
        == "@nousresearch/hermes-agent-acp@0.14.0"
    )

    package = json.loads(
        (
            tmp_path / "packages" / "hermes-agent-acp" / "package.json"
        ).read_text(encoding="utf-8")
    )
    assert package["version"] == "0.14.0"

    launcher = (
        tmp_path / "packages" / "hermes-agent-acp" / "bin" / "hermes-agent-acp.js"
    ).read_text(encoding="utf-8")
    assert "const HERMES_AGENT_VERSION = '0.14.0';" in launcher
    assert "0.13.0" not in launcher


def test_update_acp_registry_versions_is_silent_when_files_missing(
    monkeypatch, tmp_path
):
    """Older release branches predate the ACP Registry assets — must no-op."""
    module = _load_release_module(monkeypatch, tmp_path)

    # No fixture written; function should not raise.
    module._update_acp_registry_versions("0.14.0")


def test_update_version_files_bumps_acp_assets_alongside_pyproject(
    monkeypatch, tmp_path
):
    """End-to-end: update_version_files() is the function release.py actually
    calls, so it must drive the ACP bump too."""
    _write_fixture(tmp_path, "0.13.0")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "0.13.0"\n', encoding="utf-8"
    )
    version_dir = tmp_path / "hermes_cli"
    version_dir.mkdir()
    (version_dir / "__init__.py").write_text(
        '__version__ = "0.13.0"\n__release_date__ = "2026-05-14"\n',
        encoding="utf-8",
    )

    module = _load_release_module(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "VERSION_FILE", version_dir / "__init__.py")
    monkeypatch.setattr(module, "PYPROJECT_FILE", tmp_path / "pyproject.toml")

    module.update_version_files("0.14.0", "2026-05-21")

    pyproject_text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.14.0"' in pyproject_text

    manifest = json.loads(
        (tmp_path / "acp_registry" / "agent.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == "0.14.0"
    assert (
        manifest["distribution"]["npx"]["package"]
        == "@nousresearch/hermes-agent-acp@0.14.0"
    )
