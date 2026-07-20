"""Regression test for #68055 — _secure_dir() must not chmod through symlinks.

``os.chmod`` follows symlinks. ``ensure_hermes_home()`` runs ``_secure_dir()``
over a fixed subdir list (including ``skills``) on every ``load_config()``. When
a HERMES_HOME subdir is a symlink to a group-shared library, securing the link
clamps the *target* to 0700 and strips the shared group's r+x — silently, on
every CLI invocation. ``follow_symlinks=False`` is not a fix: Linux has no
``lchmod`` and raises ``NotImplementedError``. The guard is to skip symlinked
entries entirely, leaving the target's permissions to the operator.
"""
from __future__ import annotations

import os
import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX mode bits / symlink chmod semantics do not apply on Windows",
)


def test_secure_dir_leaves_symlink_target_untouched(tmp_path, monkeypatch):
    """A symlinked subdir keeps its group-shared target mode (0750), not 0700."""
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    from hermes_cli.config import _secure_dir

    target = tmp_path / "shared_skills"
    target.mkdir()
    os.chmod(target, 0o750)  # group-shared: owner rwx, group r-x

    link = tmp_path / "skills"
    link.symlink_to(target, target_is_directory=True)

    _secure_dir(link)

    # Target mode must survive — the operator owns the shared dir's perms.
    assert (target.stat().st_mode & 0o777) == 0o750


def test_secure_dir_still_secures_real_directories(tmp_path, monkeypatch):
    """A real (non-symlinked) directory is still clamped to 0700."""
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    from hermes_cli.config import _secure_dir

    real = tmp_path / "sessions"
    real.mkdir()
    os.chmod(real, 0o755)

    _secure_dir(real)

    assert (real.stat().st_mode & 0o777) == 0o700
