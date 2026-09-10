"""The canonical desktop-app resolver must give one answer to every caller.

``hermes desktop``, ``hermes doctor``, ``scripts/install.sh`` and the updater's
``posix.sh`` each used to decide for themselves which bundle under
``apps/desktop/release`` to use, and only one of them checked whether the bundle could
actually run. These tests pin the shared contract so the copies cannot come back.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import desktop_app_path
from tests.hermes_cli.test_desktop_macos_bundle_integrity import make_app_bundle


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    (root / "apps" / "desktop" / "release").mkdir(parents=True)
    return root


@pytest.mark.macos_only
def test_resolves_the_loadable_bundle_over_a_newer_broken_one(tmp_path):
    """The whole point: every caller now inherits the arm64 preference."""
    root = _tree(tmp_path)
    release = root / "apps" / "desktop" / "release"
    good = make_app_bundle(release, "mac-arm64", prebuild_arch="arm64")
    broken = make_app_bundle(release, "mac", prebuild_arch="x64")
    import os
    os.utime(broken, (10**9, 10**9))
    os.utime(good, (10**8, 10**8))

    bundle, error = desktop_app_path.validated_app_bundle(root)

    assert error is None
    assert bundle == good.parents[2]
    assert bundle.name == "Hermes.app"


@pytest.mark.macos_only
def test_validated_refuses_an_unloadable_bundle(tmp_path):
    root = _tree(tmp_path)
    make_app_bundle(root / "apps" / "desktop" / "release", "mac", prebuild_arch="x64")

    bundle, error = desktop_app_path.validated_app_bundle(root)

    assert bundle is None
    assert error is not None and "darwin-arm64" in error


@pytest.mark.macos_only
def test_app_bundle_still_reports_a_broken_install(tmp_path):
    """``hermes doctor`` must describe the app that IS installed, broken or not."""
    root = _tree(tmp_path)
    broken = make_app_bundle(root / "apps" / "desktop" / "release", "mac", prebuild_arch="x64")

    assert desktop_app_path.app_bundle(root) == broken.parents[2]


def test_reports_nothing_when_no_app_is_built(tmp_path):
    root = _tree(tmp_path)

    bundle, error = desktop_app_path.validated_app_bundle(root)

    assert bundle is None
    assert "no packaged desktop app" in error
    assert desktop_app_path.app_bundle(root) is None


# ─── the CLI surface the shell callers depend on ────────────────────────────


def _run_cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.desktop_app_path", str(root), *args],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


@pytest.mark.macos_only
def test_cli_prints_the_path_and_exits_zero(tmp_path):
    root = _tree(tmp_path)
    good = make_app_bundle(root / "apps" / "desktop" / "release", "mac-arm64")

    result = _run_cli(root)

    assert result.returncode == 0
    assert result.stdout.strip() == str(good.parents[2])


@pytest.mark.macos_only
def test_cli_exits_2_on_an_unusable_bundle(tmp_path):
    """install.sh and posix.sh key on this: 2 means REFUSE, never "guess instead"."""
    root = _tree(tmp_path)
    make_app_bundle(root / "apps" / "desktop" / "release", "mac", prebuild_arch="x64")

    result = _run_cli(root)

    assert result.returncode == 2
    assert result.stdout.strip() == ""
    assert "darwin-arm64" in result.stderr


def test_cli_exits_1_when_nothing_is_built(tmp_path):
    result = _run_cli(_tree(tmp_path))

    assert result.returncode == 1
    assert result.stdout.strip() == ""
    assert "no packaged desktop app" in result.stderr


@pytest.mark.macos_only
def test_cli_allow_unusable_reports_the_broken_bundle(tmp_path):
    root = _tree(tmp_path)
    broken = make_app_bundle(root / "apps" / "desktop" / "release", "mac", prebuild_arch="x64")

    result = _run_cli(root, "--allow-unusable")

    assert result.returncode == 0
    assert result.stdout.strip() == str(broken.parents[2])
