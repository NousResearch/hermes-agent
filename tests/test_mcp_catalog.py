from __future__ import annotations

import tomllib
from pathlib import Path

from hermes_cli import mcp_catalog


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_MCPS = ROOT / "optional-mcps"
PYPROJECT = ROOT / "pyproject.toml"


def _manifest_dirs() -> list[Path]:
    return sorted(path.parent for path in OPTIONAL_MCPS.glob("*/manifest.yaml"))


def test_shipped_mcp_manifests_are_valid() -> None:
    entries = [mcp_catalog._parse_manifest(path / "manifest.yaml") for path in _manifest_dirs()]

    names = [entry.name for entry in entries]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert all(entry.manifest_path.name == "manifest.yaml" for entry in entries)


def test_shipped_mcp_manifests_are_declared_as_packaged_data_files() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    data_files = data["tool"]["setuptools"]["data-files"]

    for manifest_dir in _manifest_dirs():
        rel_dir = manifest_dir.relative_to(ROOT).as_posix()
        rel_manifest = f"{rel_dir}/manifest.yaml"
        assert data_files.get(rel_dir) == [rel_manifest]
