"""``atomic_write_text(..., fsync_dir=True)`` must sync the directory that actually received the
replacement file.

``atomic_replace`` resolves a symlinked destination so the write lands in the real file's
directory while the symlink survives — but the directory fsync still used the unresolved input
path's parent. A config stored as ``<state>/links/config.yaml`` → ``<state>/real/config.yaml``
then fsynced the wrong directory, leaving the rename that published the new file outside the
requested durability window (#115030).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the repo root is importable when running via `pytest tests/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import atomic_write_text, fsync_directory


@pytest.mark.require_symlinks
def test_fsync_dir_targets_resolved_parent_for_symlinked_destination(tmp_path: Path) -> None:
    real = tmp_path / "real"
    links = tmp_path / "links"
    real.mkdir()
    links.mkdir()
    target = real / "value"
    target.write_text("old", encoding="utf-8")
    link = links / "value"
    link.symlink_to(target)

    with patch.object(sys.modules["utils"], "fsync_directory") as sync:
        atomic_write_text(link, "new", mode=0o600, fsync_dir=True)

    assert target.read_text(encoding="utf-8") == "new"
    assert link.is_symlink(), "symlink must survive the rewrite"
    assert sync.call_args.args[0] == real


def test_fsync_dir_uses_destination_parent_without_symlink(tmp_path: Path) -> None:
    target = tmp_path / "value"
    target.write_text("old", encoding="utf-8")

    with patch.object(sys.modules["utils"], "fsync_directory") as sync:
        atomic_write_text(target, "new", fsync_dir=True)

    assert target.read_text(encoding="utf-8") == "new"
    assert sync.call_args.args[0] == tmp_path


def test_fsync_directory_is_skipped_without_the_flag(tmp_path: Path) -> None:
    target = tmp_path / "value"
    target.write_text("old", encoding="utf-8")

    with patch.object(sys.modules["utils"], "fsync_directory") as sync:
        atomic_write_text(target, "new")

    sync.assert_not_called()


def test_real_fsync_directory_no_ops_on_windows(tmp_path: Path) -> None:
    """Guards the mocked tests against drift: the real helper stays silent and safe off-POSIX."""
    if sys.platform != "win32":
        pytest.skip("Windows-only no-op check")
    fsync_directory(tmp_path)  # must not raise
