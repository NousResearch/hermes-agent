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

from hermes_cli.update_cmd import _atomic_replace_dir


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


