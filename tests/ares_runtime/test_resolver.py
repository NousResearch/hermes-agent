from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ares_runtime import (
    AresRuntimeError,
    AresRuntimeLayout,
    AresRuntimeResolver,
    InstalledRuntimePointer,
    ReleaseReference,
)
from ares_runtime.image import write_release_manifest


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolved_layout(tmp_path: Path) -> tuple[AresRuntimeLayout, ReleaseReference]:
    layout = AresRuntimeLayout(tmp_path / "hermes" / "ares")
    layout.initialize()
    release_id = "a" * 64
    root = layout.release_dir(release_id)
    payload = root / "payload"
    python = payload / "runtime" / "python" / "bin" / "python"
    bootstrap = payload / "bootstrap" / "ares_bootstrap.py"
    python.parent.mkdir(parents=True)
    bootstrap.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    python.chmod(0o555)
    bootstrap.write_bytes(b"bootstrap")
    bootstrap.chmod(0o444)
    manifest = write_release_manifest(payload)
    reference = ReleaseReference(
        "sealed_candidate",
        release_id,
        _digest((payload / "release-manifest.json").read_bytes()),
        str(manifest["runtime_tree_sha256"]),
    )
    layout.write_pointer_atomic(
        InstalledRuntimePointer(
            generation=1,
            current=reference,
            previous=None,
            committed_transaction_id="b" * 64,
            state_root=str(layout.root.parent),
        )
    )
    return layout, reference


def test_resolver_accepts_only_the_exact_current_release(tmp_path: Path):
    layout, reference = _resolved_layout(tmp_path)
    resolver = AresRuntimeResolver(layout)

    resolved = resolver.resolve({})

    assert resolved.release == reference
    assert resolver.launch_argv("gateway", resolved)[1:4] == ["-I", "-S", "-B"]
    assert resolver.launch_environment(resolved, {})["ARES_RUNTIME_MODE"] == "stable"


@pytest.mark.parametrize(
    "name",
    [
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "HERMES_PYTHON_SRC_ROOT",
        "HERMES_DESKTOP_HERMES_ROOT",
        "VIRTUAL_ENV",
    ],
)
def test_resolver_rejects_every_development_import_override(tmp_path: Path, name: str):
    layout, _reference = _resolved_layout(tmp_path)

    with pytest.raises(AresRuntimeError, match="SOURCE_SHADOWING_DETECTED"):
        AresRuntimeResolver(layout).resolve({name: "/development"})


def test_resolver_rejects_a_changed_release_manifest(tmp_path: Path):
    layout, reference = _resolved_layout(tmp_path)
    manifest = (
        layout.release_dir(reference.release_id) / "payload" / "release-manifest.json"
    )
    manifest.chmod(0o644)
    manifest.write_bytes(b'{"changed":true}\n')

    with pytest.raises(AresRuntimeError, match="RELEASE_MANIFEST_MISMATCH"):
        AresRuntimeResolver(layout).resolve({})
