"""Failed exclusive archive writes clean up only the file they created."""

import errno
from contextlib import contextmanager
from unittest.mock import Mock, patch

from tools.tool_result_storage import PRUNED_SUBDIR, _write_to_spillover


def test_failed_exclusive_write_removes_partial_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    @contextmanager
    def partial_write(path, mode):
        with open(path, mode) as handle:
            def fail_after_prefix(data):
                handle.write(data[:3])
                handle.flush()
                raise OSError(errno.ENOSPC, "synthetic disk full")

            yield Mock(write=fail_after_prefix)

    with patch("tools.tool_result_storage.open", partial_write, create=True):
        assert _write_to_spillover("original", "partial.txt", PRUNED_SUBDIR, exclusive=True) is None
    assert list((tmp_path / PRUNED_SUBDIR).iterdir()) == []


def test_exclusive_open_collision_preserves_existing_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    existing = tmp_path / PRUNED_SUBDIR / "existing.txt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"earlier original")

    assert _write_to_spillover("new original", existing.name, PRUNED_SUBDIR, exclusive=True) is None
    assert existing.read_bytes() == b"earlier original"
