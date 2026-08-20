"""Regression: curator snapshots must exclude .venv / caches, not just top-level.

Before the fix, `tarfile.add(recursive=True)` only filtered _EXCLUDE_TOP_LEVEL
(.curator_backups, .hub); skill-internal .venv dirs (torch, opencv, model
weights) were rolled into every weekly snapshot — GBs of non-restorable content
that also made the snapshot corrupt mid-flight on large trees.
"""

import tarfile
from pathlib import Path

from agent.curator_backup import _tar_filter


def test_venv_paths_excluded():
    keep = tarfile.TarInfo("skills/realesrgan/SKILL.md")
    venv_py = tarfile.TarInfo("skills/realesrgan/.venv/Scripts/python.exe")
    venv_lib = tarfile.TarInfo("skills/realesrgan/.venv/Lib/site-packages/torch/__init__.py")
    pycache = tarfile.TarInfo("skills/foo/__pycache__/mod.cpython-311.pyc")
    node_mod = tarfile.TarInfo("skills/foo/node_modules/pkg/index.js")

    assert _tar_filter(keep) is not None
    assert _tar_filter(venv_py) is None
    assert _tar_filter(venv_lib) is None
    assert _tar_filter(pycache) is None
    assert _tar_filter(node_mod) is None


def test_hub_top_level_still_excluded_by_caller_only():
    # .hub is in EXCLUDED_SKILL_DIRS too, so nested .hub content is also dropped
    # by _tar_filter — that is fine; the *top-level* .hub is excluded separately
    # by _EXCLUDE_TOP_LEVEL in the snapshot loop, not here.
    nested = tarfile.TarInfo("skills/foo/.hub/lock.json")
    assert _tar_filter(nested) is None
