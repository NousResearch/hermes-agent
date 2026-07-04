"""Regression test for #48659 — photon sidecar files in wheel/sdist.

The photon/iMessage plugin ships a Node sidecar under
``plugins/platforms/photon/sidecar/`` with package.json, index.mjs, lockfile,
and a postinstall patch.  These non-Python assets must be declared in
``[tool.setuptools.package-data]`` and ``MANIFEST.in`` or the built
distribution omits them and ``hermes photon install-sidecar`` fails with
ENOENT.
"""

from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plugin_sidecar_files_covered_by_package_data():
    sidecar_dir = REPO_ROOT / "plugins" / "platforms" / "photon" / "sidecar"
    if not sidecar_dir.is_dir():
        pytest.skip("photon sidecar directory not present in this checkout")

    # Verify critical sidecar files exist on disk.
    critical = ["package.json", "index.mjs"]
    for name in critical:
        assert (sidecar_dir / name).is_file(), (
            f"expected {name} in photon sidecar directory"
        )

    # Wheel channel: package-data must declare a glob that matches sidecar
    # assets anywhere under the plugins package.
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    plugins_pkg_data = data["tool"]["setuptools"]["package-data"].get("plugins", [])
    assert any(
        "sidecar" in g for g in plugins_pkg_data
    ), "pyproject package-data 'plugins' must ship sidecar/**/* (wheel)"

    # Sdist channel: MANIFEST.in must include sidecar files.
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "sidecar" in manifest, (
        "MANIFEST.in must include sidecar files (sdist)"
    )
