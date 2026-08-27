from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from ares_runtime.errors import AresRuntimeError
from ares_runtime.image import (
    stage_runtime_image,
    verify_release_manifest,
    write_release_manifest,
)


def _source_runtime(tmp_path: Path) -> tuple[Path, Path]:
    python = tmp_path / "python"
    site = tmp_path / "site-packages"
    (python / "bin").mkdir(parents=True)
    (python / "bin" / "python").write_bytes(b"python-runtime")
    (site / "dependency").mkdir(parents=True)
    (site / "dependency" / "__init__.py").write_text("VALUE = 1\n")
    (site / "editable.pth").write_text("/development\n")
    (site / "__editable___hermes.py").write_text("finder\n")
    (site / "dependency.dist-info").mkdir()
    (site / "dependency.dist-info" / "direct_url.json").write_text("{}\n")
    return python, site


def test_runtime_image_excludes_editable_import_metadata_and_writes_manifest(
    tmp_path: Path,
):
    python, site = _source_runtime(tmp_path)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "ares").mkdir()
    (payload / "ares" / "hermes_cli.py").write_text("pass\n")

    image = stage_runtime_image(payload, python_root=python, site_packages_root=site)
    manifest = write_release_manifest(payload)

    assert image.bootstrap.is_file()
    assert '"AresRuntimeIdentityV1"' in image.bootstrap.read_text(encoding="utf-8")
    assert (image.python / "bin" / "python").read_bytes() == b"python-runtime"
    assert (image.site_packages / "dependency" / "__init__.py").is_file()
    assert not (image.site_packages / "editable.pth").exists()
    assert not (image.site_packages / "__editable___hermes.py").exists()
    assert not (
        image.site_packages / "dependency.dist-info" / "direct_url.json"
    ).exists()
    persisted = json.loads((payload / "release-manifest.json").read_text())
    assert persisted["runtime_tree_sha256"] == manifest["runtime_tree_sha256"]
    assert "release-manifest.json" not in {entry["path"] for entry in manifest["files"]}
    assert (
        verify_release_manifest(
            payload,
            expected_manifest_sha256=(
                hashlib.sha256(
                    (payload / "release-manifest.json").read_bytes()
                ).hexdigest()
            ),
            expected_runtime_tree_sha256=str(manifest["runtime_tree_sha256"]),
        )
        == manifest
    )


def test_release_manifest_rejects_post_seal_runtime_change(tmp_path: Path):
    python, site = _source_runtime(tmp_path)
    payload = tmp_path / "payload"
    payload.mkdir()
    stage_runtime_image(payload, python_root=python, site_packages_root=site)
    manifest = write_release_manifest(payload)
    digest = hashlib.sha256(
        (payload / "release-manifest.json").read_bytes()
    ).hexdigest()
    (payload / "runtime" / "site-packages" / "dependency" / "__init__.py").write_text(
        "CHANGED = 1\n"
    )

    with pytest.raises(AresRuntimeError, match="RUNTIME_TREE_MISMATCH"):
        verify_release_manifest(
            payload,
            expected_manifest_sha256=digest,
            expected_runtime_tree_sha256=str(manifest["runtime_tree_sha256"]),
        )


def test_runtime_image_materializes_safe_relative_symlinks(tmp_path: Path):
    python, site = _source_runtime(tmp_path)
    (site / "linked.py").symlink_to("dependency/__init__.py")
    payload = tmp_path / "payload"
    payload.mkdir()

    stage_runtime_image(payload, python_root=python, site_packages_root=site)
    manifest = write_release_manifest(payload)

    assert not (payload / "runtime" / "site-packages" / "linked.py").is_symlink()
    entry = next(
        item
        for item in manifest["files"]
        if item["path"] == "runtime/site-packages/linked.py"
    )
    assert entry["kind"] == "file"
    assert entry["size"] == len(b"VALUE = 1\n")


def test_runtime_image_rejects_source_symlinks_outside_runtime(tmp_path: Path):
    python, site = _source_runtime(tmp_path)
    (site / "linked.py").symlink_to("/development/source.py")
    payload = tmp_path / "payload"
    payload.mkdir()

    with pytest.raises(AresRuntimeError, match="RUNTIME_SOURCE_SYMLINK"):
        stage_runtime_image(payload, python_root=python, site_packages_root=site)
