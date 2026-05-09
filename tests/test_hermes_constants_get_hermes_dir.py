"""Tests for hermes_constants.get_hermes_dir() — new vs old layout fallback."""
from hermes_constants import get_hermes_dir


class TestGetHermesDir:
    def test_returns_old_path_when_legacy_dir_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        legacy = tmp_path / "image_cache"
        legacy.mkdir()
        assert get_hermes_dir("cache/images", "image_cache") == legacy

    def test_returns_new_path_when_legacy_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert get_hermes_dir("cache/images", "image_cache") == tmp_path / "cache" / "images"

    def test_legacy_can_be_a_file_not_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        legacy = tmp_path / "old_file"
        legacy.write_text("x")
        assert get_hermes_dir("new/path", "old_file") == legacy

    def test_returns_new_when_legacy_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        result = get_hermes_dir("new/sub", "missing_legacy")
        assert result == tmp_path / "new" / "sub"
        assert not result.exists()

    def test_uses_current_hermes_home_each_call(self, tmp_path, monkeypatch):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(a))
        first = get_hermes_dir("cache/x", "old")
        monkeypatch.setenv("HERMES_HOME", str(b))
        second = get_hermes_dir("cache/x", "old")
        assert first == a / "cache" / "x"
        assert second == b / "cache" / "x"
