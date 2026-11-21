"""The ZIP update's path removal must honor ``ignore_errors`` for directories too.

The rollback loop in ``_commit_staged_replacements`` calls ``_remove_path(dst)`` with the
default and relies on a raise to report the entry that could not be cleared: "a silent
failure here turns a recoverable rollback into a mixed tree". The directory branch used to
hardcode ``ignore_errors=True``, hiding exactly that failure.
"""

import pytest

from hermes_cli import update_cmd_zip


def test_remove_path_removes_plain_directories(tmp_path):
    target = tmp_path / "entry"
    target.mkdir()
    (target / "file.txt").write_text("x", encoding="utf-8")
    update_cmd_zip._remove_path(str(target))
    assert not target.exists()


@pytest.mark.platforms("posix")
def test_remove_path_honors_ignore_errors_for_directories(tmp_path):
    """A read-only parent makes ``rmtree`` fail: with ``ignore_errors=False`` the failure must
    surface (the rollback loop depends on the raise), with ``True`` it must be swallowed."""
    parent = tmp_path / "locked"
    victim = parent / "entry"
    victim.mkdir(parents=True)
    (victim / "file.txt").write_text("x", encoding="utf-8")
    parent.chmod(0o555)
    try:
        with pytest.raises(OSError):
            update_cmd_zip._remove_path(str(victim), ignore_errors=False)
        update_cmd_zip._remove_path(str(victim), ignore_errors=True)
    finally:
        parent.chmod(0o755)
