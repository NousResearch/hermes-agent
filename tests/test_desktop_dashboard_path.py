"""Regression guard: desktop dashboard spawns inherit ~/.hermes/node/bin on PATH.

Issue #41385 — macOS GUI apps launched via Launch Services inherit a minimal
PATH (/usr/bin:/bin:/usr/sbin:/sbin) that does NOT include ~/.hermes/node/bin.
The dashboard backend creation functions must extend PATH so that tools
installed by Hermes's own Node (e.g. codex) are discoverable.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_CJS = REPO_ROOT / "apps" / "desktop" / "electron" / "main.cjs"


def _read_main_cjs() -> str:
    return MAIN_CJS.read_text(encoding="utf-8")


def test_create_python_backend_extends_path():
    """createPythonBackend() must include node/bin in the spawned PATH."""
    src = _read_main_cjs()

    # Find the createPythonBackend function body
    match = re.search(
        r"function createPythonBackend\b.*?\n\}",
        src,
        re.DOTALL,
    )
    assert match, "createPythonBackend function not found in main.cjs"
    body = match.group()

    assert "node" in body and "bin" in body, (
        "createPythonBackend does not reference node/bin for PATH extension"
    )
    assert "PATH:" in body, (
        "createPythonBackend does not set PATH in the env object"
    )


def test_create_active_backend_extends_path():
    """createActiveBackend() must include node/bin in the spawned PATH."""
    src = _read_main_cjs()

    match = re.search(
        r"function createActiveBackend\b.*?\n\}",
        src,
        re.DOTALL,
    )
    assert match, "createActiveBackend function not found in main.cjs"
    body = match.group()

    assert "node" in body and "bin" in body, (
        "createActiveBackend does not reference node/bin for PATH extension"
    )
    assert "PATH:" in body, (
        "createActiveBackend does not set PATH in the env object"
    )
