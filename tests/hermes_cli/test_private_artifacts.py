import os
import stat

import pytest

from hermes_cli.private_artifacts import (
    ensure_private_dir,
    private_text_writer,
    write_private_text,
)


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_private_text_writer_creates_private_file_and_missing_dirs(tmp_path):
    old_umask = os.umask(0)
    try:
        target = tmp_path / "new" / "nested" / "artifact.jsonl"
        with private_text_writer(target) as handle:
            handle.write("private\n")
    finally:
        os.umask(old_umask)

    assert target.read_text(encoding="utf-8") == "private\n"
    assert _mode(tmp_path / "new") == 0o700
    assert _mode(tmp_path / "new" / "nested") == 0o700
    assert _mode(target) == 0o600


def test_private_text_writer_tightens_existing_file_mode(tmp_path):
    target = tmp_path / "artifact.jsonl"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o666)

    with private_text_writer(target) as handle:
        handle.write("new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert _mode(target) == 0o600


def test_ensure_private_dir_does_not_chmod_existing_parent(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    existing.chmod(0o755)

    ensure_private_dir(existing)

    assert _mode(existing) == 0o755


def test_write_private_text_refuses_non_directory_parent(tmp_path):
    parent = tmp_path / "not-a-dir"
    parent.write_text("nope", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        write_private_text(parent / "artifact.txt", "payload")
