"""Tests for utils.atomic_json_write — crash-safe JSON file writes."""

import json
from unittest.mock import patch

import pytest

from utils import atomic_json_write


class TestAtomicJsonWrite:
    """Core atomic write behavior."""

    def test_file_ends_with_a_single_newline(self, tmp_path):
        """A JSON file that a user keeps in git must not show "No newline at end of file"."""
        target = tmp_path / "data.json"

        atomic_json_write(target, {"a": 1})

        text = target.read_text(encoding="utf-8")
        assert text.endswith("}\n")
        assert not text.endswith("\n\n")
        assert json.loads(text) == {"a": 1}

    def test_surrogate_fallback_also_ends_with_a_newline(self, tmp_path):
        """The ``ensure_ascii`` retry for non-UTF-8 strings writes the file too."""
        target = tmp_path / "data.json"

        atomic_json_write(target, {"path": "bad-\udcff-name"})

        assert target.read_text(encoding="utf-8").endswith("}\n")


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
