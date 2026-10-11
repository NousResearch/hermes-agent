"""Pinned git artifacts must stay stock-Windows compatible (see #125350).

Stock Windows ``tar.exe`` has no bzip2 backend (``unable to run program
"bzip2 -d"``), so a ``.tar.bz2`` git pin bricks fresh installs. The pins moved
to self-extracting ``PortableGit-*.7z.exe`` (own extractor); this guards the
fixed state.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_git_artifacts_need_no_external_decompressor():
    lock = json.loads((REPO_ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))
    packages = lock.get("packages", lock)
    git = packages.get("git", {})
    artifacts = git.get("artifacts", {})
    assert artifacts, "git pins missing from pm/lock.json"
    for target, spec in artifacts.items():
        url = spec.get("url", "")
        assert url.lower().endswith((".7z.exe", ".zip")), (
            f"git artifact for {target} needs an external decompressor: {url}"
        )
        assert ".tar." not in url.lower(), (
            f"git artifact for {target} is a tarball (stock tar.exe lacks bzip2): {url}"
        )
