import os
import pytest

import utils
from utils import _BOUNDED_MKSTEMP_ATTEMPTS, bounded_mkstemp


class TestBoundedMkstemp:
    def test_creates_file_and_returns_open_fd(self, tmp_path):
        fd, path = bounded_mkstemp(dir=str(tmp_path), prefix="test_", suffix=".tmp")
        try:
            assert os.path.exists(path)
            assert os.path.basename(path).startswith("test_")
            assert path.endswith(".tmp")
            os.write(fd, b"hello")
        finally:
            os.close(fd)
            if os.path.exists(path):
                os.unlink(path)

    def test_distinct_names_across_calls(self, tmp_path):
        paths = []
        fds = []
        try:
            for _ in range(5):
                fd, path = bounded_mkstemp(dir=str(tmp_path))
                fds.append(fd)
                paths.append(path)
            assert len(set(paths)) == 5
        finally:
            for fd in fds:
                os.close(fd)
            for path in paths:
                if os.path.exists(path):
                    os.unlink(path)

    def test_windows_semantics_fail_fast_not_tmp_max(self, tmp_path, monkeypatch):
        """Regression: a denied directory raises within _BOUNDED_MKSTEMP_ATTEMPTS attempts."""
        monkeypatch.setattr(utils, "_MKSTEMP_RETRY_PERMISSION_ERROR", True)
        calls = {"n": 0}

        def denying_open(path, flags, mode=0o600):
            calls["n"] += 1
            raise PermissionError(13, "Access is denied", path)

        monkeypatch.setattr(os, "open", denying_open)
        with pytest.raises(PermissionError):
            bounded_mkstemp(dir=str(tmp_path))
        assert calls["n"] == _BOUNDED_MKSTEMP_ATTEMPTS
