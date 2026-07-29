"""Executable compatibility checks for the pinned Spectrum SDK."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


_SIDECAR = Path("plugins/platforms/photon/sidecar")


def test_spectrum_dependencies_are_exactly_pinned_and_lockstep() -> None:
    """Spectrum breaking majors require an explicit, lockstep upgrade."""
    package = json.loads((_SIDECAR / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((_SIDECAR / "package-lock.json").read_text(encoding="utf-8"))
    spectrum_packages = [
        "@spectrum-ts/core",
        "@spectrum-ts/imessage-local",
        "spectrum-ts",
    ]

    versions = {package["dependencies"][name] for name in spectrum_packages}
    assert len(versions) == 1
    version = versions.pop()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    for name in spectrum_packages:
        assert lock["packages"][f"node_modules/{name}"]["version"] == version


def test_spectrum_local_smoke_script_runs() -> None:
    """Exercise v12 providers, builders, and native mixed-message ordering."""
    result = subprocess.run(
        ["node", "smoke-spectrum-local.mjs"],
        cwd=_SIDECAR,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    package = json.loads((_SIDECAR / "package.json").read_text(encoding="utf-8"))
    assert payload["spectrumVersion"] == package["dependencies"]["spectrum-ts"]
    assert "native-mixed-attachments" in payload["checks"]
