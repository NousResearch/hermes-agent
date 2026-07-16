"""Regression tests for Photon sidecar runtime placement.

The bundled plugin tree can be immutable in managed installs, so Photon must
install and run its Node sidecar from a writable HERMES_HOME mirror instead of
writing into `/opt/hermes/plugins/.../sidecar`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import cli


def test_install_sidecar_uses_writable_runtime_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/fake/{Path(str(name)).name}")

    calls: list[tuple[list[str], str]] = []

    class _Proc:
        returncode = 0

    def _fake_run(cmd, cwd, check):  # type: ignore[no-untyped-def]
        calls.append((list(cmd), cwd))
        return _Proc()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    rc = cli._install_sidecar()

    runtime_dir = hermes_home / "platforms" / "photon" / "sidecar"
    assert rc == 0
    assert calls == [(["/fake/npm", "ci"], str(runtime_dir))]
    assert (runtime_dir / "index.mjs").is_file()
    assert (runtime_dir / "package.json").is_file()
    assert (runtime_dir / "package-lock.json").is_file()
    assert (runtime_dir / "patch-spectrum-mixed-attachments.mjs").is_file()


def test_check_requirements_uses_runtime_node_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hermes_home = tmp_path / ".hermes"
    runtime_dir = hermes_home / "platforms" / "photon" / "sidecar" / "node_modules"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(photon_adapter, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(
        photon_adapter.shutil,
        "which",
        lambda name: "/fake/node" if Path(str(name)).name == "node" else None,
    )

    assert photon_adapter.check_requirements() is True
