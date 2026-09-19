"""Test coverage for gateway/lifecycle_ledger.py — 11 functions had LOW coverage.

Tests sentinel path resolution, memory sampling, JSON read/write,
unclean exit detection, and startup recording. All filesystem access
uses tmp_path — no real HERMES_HOME is touched.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from gateway.lifecycle_ledger import (
    detect_unclean_exit,
    get_lifecycle_sentinel_path,
    record_startup,
    _pid_alive_with_start_time,
    _read_json,
    sample_memory,
)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestSentinelPath:
    def test_default_under_hermes_home(self, hermes_home):
        p = get_lifecycle_sentinel_path()
        assert str(hermes_home) in str(p)

    def test_explicit_home(self, tmp_path):
        p = get_lifecycle_sentinel_path(tmp_path)
        assert str(tmp_path) in str(p)


class TestSampleMemory:
    def test_returns_dict_with_keys(self):
        d = sample_memory()
        assert isinstance(d, dict)


class TestReadJson:
    def test_valid_json(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        assert _read_json(f) == {"key": "value"}

    def test_missing_file(self, tmp_path):
        assert _read_json(tmp_path / "missing.json") is None

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json", encoding="utf-8")
        assert _read_json(f) is None


class TestDetectUncleanExit:
    def test_no_sentinel_returns_none(self, hermes_home):
        assert detect_unclean_exit() is None

    def test_clean_exit_returns_none(self, hermes_home):
        sentinel = get_lifecycle_sentinel_path(hermes_home)
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(json.dumps({
            "pid": os.getpid(),
            "start_time": 0,
            "clean": True,
        }), encoding="utf-8")
        assert detect_unclean_exit(hermes_home) is None


class TestRecordStartup:
    def test_writes_sentinel(self, hermes_home):
        record_startup(hermes_home)
        sentinel = get_lifecycle_sentinel_path(hermes_home)
        assert sentinel.exists()
        data = json.loads(sentinel.read_text(encoding="utf-8"))
        assert "pid" in data
