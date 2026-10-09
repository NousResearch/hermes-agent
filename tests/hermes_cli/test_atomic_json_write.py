"""Tests for utils.atomic_json_write — crash-safe JSON file writes."""

import json
import os
from unittest.mock import patch

import pytest

from utils import atomic_json_write


@pytest.fixture
def windows_text_mode(monkeypatch):
    """Make a text file opened with the default ``newline=None`` write ``\\n`` as ``\\r\\n``.

    That is what the default does on Windows. The fixture applies it on any host, so the tests can
    tell a writer that opts out of the translation from one that leaves it on.
    """
    real_fdopen = os.fdopen

    def fdopen(fd, mode="r", *args, **kwargs):
        if "b" not in mode and kwargs.get("newline") is None:
            kwargs["newline"] = "\r\n"
        return real_fdopen(fd, mode, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", fdopen)


class TestAtomicJsonWrite:
    """Core atomic write behavior."""

    def test_file_ends_with_a_single_newline(self, tmp_path):
        """A JSON file that a user keeps in git must not show "No newline at end of file"."""
        target = tmp_path / "data.json"

        atomic_json_write(target, {"a": 1})

        # Raw bytes: ``read_text()`` turns a ``\r\n`` back into ``\n`` and would hide the difference.
        data = target.read_bytes()
        assert data.endswith(b"}\n")
        assert not data.endswith(b"\n\n")
        assert json.loads(data) == {"a": 1}

    def test_surrogate_fallback_also_ends_with_a_newline(self, tmp_path):
        """The ``ensure_ascii`` retry for non-UTF-8 strings writes the file too."""
        target = tmp_path / "data.json"

        atomic_json_write(target, {"path": "bad-\udcff-name"})

        assert target.read_bytes().endswith(b"}\n")

    def test_line_endings_are_lf_even_where_text_mode_translates(self, tmp_path, windows_text_mode):
        """On Windows the default text mode would write ``\\r\\n``; the file must stay LF."""
        target = tmp_path / "data.json"

        atomic_json_write(target, {"a": 1})

        assert target.read_bytes() == b'{\n  "a": 1\n}\n'

    def test_surrogate_fallback_is_lf_where_text_mode_translates(self, tmp_path, windows_text_mode):
        target = tmp_path / "data.json"

        atomic_json_write(target, {"path": "bad-\udcff-name"})

        data = target.read_bytes()
        assert data.endswith(b"}\n")
        assert b"\r" not in data


    def test_cleans_up_temp_file_on_baseexception(self, tmp_path):
        class SimulatedAbort(BaseException):
            pass

        target = tmp_path / "data.json"
        original = {"preserved": True}
        target.write_text(json.dumps(original), encoding="utf-8")

        with patch("utils.json.dumps", side_effect=SimulatedAbort):
            with pytest.raises(SimulatedAbort):
                atomic_json_write(target, {"new": True})

        tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
        assert len(tmp_files) == 0
        assert json.loads(target.read_text(encoding="utf-8")) == original


    def test_concurrent_writes_dont_corrupt(self, tmp_path):
        """Multiple rapid writes should each produce valid JSON."""
        import threading

        target = tmp_path / "concurrent.json"
        errors = []

        def writer(n):
            try:
                atomic_json_write(target, {"writer": n, "data": list(range(100))})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # File should contain valid JSON from one of the writers
        result = json.loads(target.read_text())
        assert "writer" in result
        assert len(result["data"]) == 100
