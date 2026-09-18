"""``load_yaml_file_readonly`` re-parses only when the file signature changes."""
import os

import pytest

from utils import load_yaml_file_readonly


def test_cache_hit_returns_same_object_and_invalidates_on_rewrite(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("secrets: {a: 1}\n")
    first = load_yaml_file_readonly(path)
    assert first == {"secrets": {"a": 1}}
    assert load_yaml_file_readonly(path) is first

    path.write_text("secrets: {a: 2}\n")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 1_000_000,) * 2)
    second = load_yaml_file_readonly(path)
    assert second == {"secrets": {"a": 2}}
    assert second is not first


def test_parse_error_propagates_and_is_not_cached(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("secrets: [unclosed\n")
    with pytest.raises(Exception):
        load_yaml_file_readonly(path)
    path.write_text("secrets: {}\n")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 1_000_000,) * 2)
    assert load_yaml_file_readonly(path) == {"secrets": {}}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml_file_readonly(tmp_path / "nope.yaml")
