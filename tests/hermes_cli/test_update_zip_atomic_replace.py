"""Regression: the ZIP-update directory replace must never leave a half-deleted tree.

Issue #49145: on Windows the ZIP-update path did ``rmtree(dst); copytree(...)``.
A copy that failed partway (file locks / flaky I/O — the very conditions the ZIP
path exists to work around) left the directory deleted with nothing copied back,
which broke ``hermes --tui`` because ``ui-tui/`` had vanished.

``_atomic_replace_dir`` stages the new copy first and only swaps it in on full
success, so a mid-copy failure leaves the original directory intact.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hermes_cli.main import _atomic_replace_dir


def test_atomic_replace_swaps_content_on_success(tmp_path: Path) -> None:
    src = tmp_path / "src" / "ui-tui"
    src.mkdir(parents=True)
    (src / "new.txt").write_text("NEW")

    dst = tmp_path / "install" / "ui-tui"
    dst.mkdir(parents=True)
    (dst / "old.txt").write_text("OLD")

    _atomic_replace_dir(str(src), str(dst))

    assert (dst / "new.txt").read_text() == "NEW"
    assert not (dst / "old.txt").exists()
    # No staging/backup siblings left behind.
    assert not (dst.parent / "ui-tui.hermes-update-staging").exists()
    assert not (dst.parent / "ui-tui.hermes-update-old").exists()


def test_atomic_replace_leaves_original_intact_when_copy_fails(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "src" / "ui-tui"
    src.mkdir(parents=True)
    (src / "a.txt").write_text("A")

    dst = tmp_path / "install" / "ui-tui"
    dst.mkdir(parents=True)
    (dst / "keep.txt").write_text("PRECIOUS")

    def boom(*_a, **_k):
        raise OSError("[WinError 5] Access is denied")

    monkeypatch.setattr(shutil, "copytree", boom)

    with pytest.raises(OSError):
        _atomic_replace_dir(str(src), str(dst))

    # The whole point: the live directory survives a failed update untouched.
    assert dst.is_dir()
    assert (dst / "keep.txt").read_text() == "PRECIOUS"
    assert not (dst.parent / "ui-tui.hermes-update-staging").exists()


def test_atomic_replace_clears_stale_staging_leftovers(tmp_path: Path) -> None:
    """A previously-interrupted update can leave staging/backup dirs behind."""
    src = tmp_path / "src" / "ui-tui"
    src.mkdir(parents=True)
    (src / "new.txt").write_text("NEW")

    dst = tmp_path / "install" / "ui-tui"
    dst.mkdir(parents=True)

    stale_staging = dst.parent / "ui-tui.hermes-update-staging"
    stale_backup = dst.parent / "ui-tui.hermes-update-old"
    stale_staging.mkdir()
    stale_backup.mkdir()
    (stale_staging / "junk").write_text("junk")

    _atomic_replace_dir(str(src), str(dst))

    assert (dst / "new.txt").read_text() == "NEW"
    assert not stale_staging.exists()
    assert not stale_backup.exists()
