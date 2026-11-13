"""A leftover release tree holding a DOS reserved name must not block the desktop promotion.

Windows resolves a file named ``nul`` to the DOS device, not to the file in the directory, so
``shutil.rmtree`` cannot delete it through the normal path-based APIs. With
``ignore_errors=True`` that failure was swallowed, the leftover directory survived, and the
promotion rename then failed with WinError 183 (ERROR_ALREADY_EXISTS) -- surfacing as
"Could not publish the desktop build" while the real cause was an undeletable leftover.

Regression for the desktop build that could not be published on Windows hosts where a build
step wrote ``> nul`` (a bash redirect to a literal file) into the release tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import main_desktop
from hermes_cli.desktop_release_cleanup import remove_reserved_name_entries, remove_tree
from tests.hermes_cli.test_desktop_swap_file_lock_retry import _staged_over_live


def _write_reserved_name_file(directory: Path, name: str = "nul") -> Path:
    """Create a file whose name is a DOS reserved device name.

    Only the device path form can create one: the plain path resolves to the device itself.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with open("\\\\?\\" + os.path.normpath(str(path)), "w", encoding="utf-8") as handle:
        handle.write("x")
    return path


@pytest.mark.platforms("windows")
def test_a_leftover_release_tree_holding_a_reserved_name_does_not_block_the_promotion(
    tmp_path, monkeypatch
):
    """The regression: the promotion must succeed even when the aside holds a "nul" file."""
    desktop_dir, staging, live_exe, _slept = _staged_over_live(tmp_path, monkeypatch)
    live_root = live_exe.parent
    previous = live_root.with_name(
        live_root.name + main_desktop._DESKTOP_PREVIOUS_SUFFIX
    )
    _write_reserved_name_file(previous)
    assert previous.is_dir()

    promoted = main_desktop._swap_staged_desktop_app(desktop_dir, staging)

    assert promoted == live_exe
    assert live_exe.read_text(encoding="utf-8-sig") == "new"
    assert not previous.exists()


@pytest.mark.platforms("windows")
def test_remove_tree_removes_a_tree_holding_a_reserved_name(tmp_path):
    tree = tmp_path / "win-unpacked.previous"
    tree.mkdir()
    (tree / "keep.txt").write_text("keep", encoding="utf-8")
    _write_reserved_name_file(tree)

    assert remove_tree(tree) is None
    assert not tree.exists()


@pytest.mark.platforms("windows")
def test_remove_reserved_name_entries_leaves_ordinary_files_alone(tmp_path):
    tree = tmp_path / "release"
    tree.mkdir()
    (tree / "keep.txt").write_text("keep", encoding="utf-8")
    reserved = _write_reserved_name_file(tree)

    removed = remove_reserved_name_entries(tree)

    assert removed == [str(reserved)]
    assert (tree / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not os.path.exists("\\\\?\\" + os.path.normpath(str(reserved)))


@pytest.mark.platforms("windows")
def test_remove_tree_reports_the_reason_instead_of_swallowing_it(tmp_path, monkeypatch):
    """A tree that cannot be removed must return the reason, not fail silently."""
    tree = tmp_path / "locked"
    tree.mkdir()
    (tree / "file.txt").write_text("x", encoding="utf-8")

    def failing_rmtree(path, *args, **kwargs):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(
        "hermes_cli.desktop_release_cleanup.shutil.rmtree", failing_rmtree
    )

    error = remove_tree(tree)

    assert error is not None
    assert "could not remove" in error
    # The reason is carried, so the caller can report WHY the swap failed.
    assert "Access is denied" in error or "errno 5" in error
