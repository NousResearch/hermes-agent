"""Tests for utils.atomic_json_write — crash-safe JSON file writes."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from utils import atomic_json_write


class TestAtomicJsonWrite:
    """Core atomic write behavior."""

    def test_writes_valid_json(self, tmp_path):
        target = tmp_path / "data.json"
        data = {"key": "value", "nested": {"a": 1}}
        atomic_json_write(target, data)

        result = json.loads(target.read_text(encoding="utf-8"))
        assert result == data

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "dir" / "data.json"
        atomic_json_write(target, {"ok": True})

        assert target.exists()
        assert json.loads(target.read_text())["ok"] is True

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "data.json"
        target.write_text('{"old": true}')

        atomic_json_write(target, {"new": True})
        result = json.loads(target.read_text())
        assert result == {"new": True}

    def test_preserves_original_on_serialization_error(self, tmp_path):
        target = tmp_path / "data.json"
        original = {"preserved": True}
        target.write_text(json.dumps(original))

        # Try to write non-serializable data — should fail
        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": object()})

        # Original file should be untouched
        result = json.loads(target.read_text())
        assert result == original

    def test_no_leftover_temp_files_on_success(self, tmp_path):
        target = tmp_path / "data.json"
        atomic_json_write(target, [1, 2, 3])

        # No .tmp files should be left behind
        tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
        assert len(tmp_files) == 0
        assert target.exists()

    def test_no_leftover_temp_files_on_failure(self, tmp_path):
        target = tmp_path / "data.json"

        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": object()})

        # No temp files should be left behind
        tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
        assert len(tmp_files) == 0

    def test_cleans_up_temp_file_on_baseexception(self, tmp_path):
        class SimulatedAbort(BaseException):
            pass

        target = tmp_path / "data.json"
        original = {"preserved": True}
        target.write_text(json.dumps(original), encoding="utf-8")

        with patch("utils.json.dump", side_effect=SimulatedAbort):
            with pytest.raises(SimulatedAbort):
                atomic_json_write(target, {"new": True})

        tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
        assert len(tmp_files) == 0
        assert json.loads(target.read_text(encoding="utf-8")) == original

    def test_accepts_string_path(self, tmp_path):
        target = str(tmp_path / "string_path.json")
        atomic_json_write(target, {"string": True})

        result = json.loads(Path(target).read_text())
        assert result == {"string": True}

    def test_writes_list_data(self, tmp_path):
        target = tmp_path / "list.json"
        data = [1, "two", {"three": 3}]
        atomic_json_write(target, data)

        result = json.loads(target.read_text())
        assert result == data

    def test_empty_list(self, tmp_path):
        target = tmp_path / "empty.json"
        atomic_json_write(target, [])

        result = json.loads(target.read_text())
        assert result == []

    def test_custom_indent(self, tmp_path):
        target = tmp_path / "custom.json"
        atomic_json_write(target, {"a": 1}, indent=4)

        text = target.read_text()
        assert '    "a"' in text  # 4-space indent

    def test_accepts_json_dump_default_hook(self, tmp_path):
        class CustomValue:
            def __str__(self):
                return "custom-value"

        target = tmp_path / "custom_default.json"
        atomic_json_write(target, {"value": CustomValue()}, default=str)

        result = json.loads(target.read_text(encoding="utf-8"))
        assert result == {"value": "custom-value"}

    def test_unicode_content(self, tmp_path):
        target = tmp_path / "unicode.json"
        data = {"emoji": "🎉", "japanese": "日本語"}
        atomic_json_write(target, data)

        result = json.loads(target.read_text(encoding="utf-8"))
        assert result["emoji"] == "🎉"
        assert result["japanese"] == "日本語"

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
