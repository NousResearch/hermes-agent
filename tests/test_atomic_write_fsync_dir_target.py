"""``atomic_write_text(..., fsync_dir=True)`` must sync the directory that actually received the
replacement file.

``atomic_replace`` resolves a symlinked destination so the write lands in the real file's
directory while the symlink survives — but the directory fsync used the unresolved input path's
parent, so the rename that published the new file sat outside the requested durability window
(#115030). Both sides are compared through ``Path.resolve()``: on macOS the temp root is reached via
``/var`` -> ``/private/var`` and a raw comparison fails even on a correct fix.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import utils
from utils import atomic_write_text


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

    with patch.object(utils, "fsync_directory") as sync:
        atomic_write_text(link, "new", mode=0o600, fsync_dir=True)

    assert target.read_text(encoding="utf-8") == "new"
    assert link.is_symlink(), "symlink must survive the rewrite"
    assert Path(sync.call_args.args[0]).resolve() == real.resolve()


def test_fsync_dir_uses_destination_parent_without_symlink(tmp_path: Path) -> None:
    """Control: a plain destination keeps today's behaviour (``atomic_replace`` returns it unchanged)."""
    target = tmp_path / "value"
    target.write_text("old", encoding="utf-8")

    with patch.object(utils, "fsync_directory") as sync:
        atomic_write_text(target, "new", fsync_dir=True)

    assert target.read_text(encoding="utf-8") == "new"
    assert Path(sync.call_args.args[0]).resolve() == tmp_path.resolve()
