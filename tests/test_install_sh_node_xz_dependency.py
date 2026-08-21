"""Regression coverage for Node.js tar.xz extraction on minimal Debian.

Node's Linux binaries are distributed as ``.tar.xz`` archives. Minimal
Debian/Ubuntu images can have ``tar`` and ``curl`` without the external ``xz``
helper that GNU tar uses to decompress those archives.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _function_body(name: str) -> str:
    text = INSTALL_SH.read_text()
    match = re.search(
        rf"^{re.escape(name)}\(\)\s*\{{\s*\n(?P<body>.*?)^\}}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{name}() not found in scripts/install.sh"
    return match["body"]


def test_debian_installs_xz_utils_for_node_tar_xz_extraction() -> None:
    body = _function_body("install_system_packages")

    assert "command -v xz" in body
    assert "xz-utils" in body
    assert "Node.js archive extraction" in body


def test_node_tar_xz_extraction_checks_for_xz_before_tar() -> None:
    body = _function_body("install_node")
    tar_xz_branch = re.search(
        r'if \[\[ "\$tarball_name" == \*\.tar\.xz \]\]; then(?P<body>.*?)else',
        body,
        re.DOTALL,
    )

    assert tar_xz_branch is not None
    assert "command -v xz" in tar_xz_branch["body"]
    assert 'tar xf "$tmp_dir/$tarball_name"' in tar_xz_branch["body"]
