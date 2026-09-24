"""Subprocess regression test for ESBUILD_BINARY_PATH env-var poisoning.

build.mjs must delete process.env.ESBUILD_BINARY_PATH before calling
require("esbuild"), otherwise a stale binary path inherited from the
environment (e.g. ~/.hermes/esbuild-built at a different version) will
cause esbuild's generateBinPath() to return the wrong binary, triggering:

    Cannot start service: Host version "X" does not match binary version "Y"

See PR #27263.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def ui_tui_dir() -> Path:
    """Resolve the ui-tui/ directory relative to the repo root."""
    # tests/hermes_cli/ → repo root
    return Path(__file__).resolve().parent.parent.parent / "ui-tui"


def test_build_succeeds_with_incompatible_esbuild_binary_path(
    ui_tui_dir: Path,
) -> None:
    """build.mjs must tolerate an inherited ESBUILD_BINARY_PATH pointing to a
    non-existent binary — regression for the Android/Termux version-mismatch
    bug where a stale ~/.hermes/esbuild-built pollutes the build.

    The script deletes ``process.env.ESBUILD_BINARY_PATH`` before invoking
    ``require('esbuild')``, so esbuild falls back to its normal binary
    resolution and finds the correct platform-specific binary from
    node_modules.
    """
    env = os.environ.copy()
    env["ESBUILD_BINARY_PATH"] = "/dev/null/esbuild-binary-stale"

    result = subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [  # fmt: skip
            "node",
            str(ui_tui_dir / "scripts" / "build.mjs"),
        ],
        cwd=str(ui_tui_dir),
        capture_output=True,
        text=True,
        env=env,
    )

    # The build should succeed despite the bogus ESBUILD_BINARY_PATH
    assert result.returncode == 0, (
        f"build.mjs failed with a stale ESBUILD_BINARY_PATH:\n"
        f"  stderr: {result.stderr.strip()}\n"
        f"  stdout: {result.stdout.strip()}"
    )

    # Sanity: confirm dist/entry.js was actually written
    entry_js = ui_tui_dir / "dist" / "entry.js"
    assert entry_js.exists(), (
        "build.mjs claimed success but did not produce dist/entry.js"
    )
    assert entry_js.stat().st_size > 100_000, (
        f"dist/entry.js is suspiciously small ({entry_js.stat().st_size} bytes)"
    )
