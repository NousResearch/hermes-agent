"""Tests for session_storage_health(): the non-mutating health probe that
reports unavailable session storage as a DISTINCT failing state (naming the
configured path and its resolved/intended target) instead of letting the
condition collapse into the opaque mkdir EEXIST error.

Covers: healthy real directory, healthy valid symlink, absent path, dangling
symlink, dangling symlink in the parent chain, and non-directory entries.
No real external volume is mounted, unmounted, or otherwise altered — all
states are simulated under tmp_path.
"""

import pytest

from gateway.session import session_storage_health


class TestSessionStorageHealth:
    def test_healthy_real_directory(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        health = session_storage_health(sessions_dir)

        assert health["status"] == "ok"
        assert health["reason"] == "directory"
        assert health["path"] == str(sessions_dir)
        assert health["target"] is None

    def test_healthy_valid_symlink(self, tmp_path):
        target = tmp_path / "mounted_volume" / "sessions"
        target.mkdir(parents=True)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.symlink_to(target)

        health = session_storage_health(sessions_dir)

        assert health["status"] == "ok"
        assert health["reason"] == "valid_symlink"
        assert health["target"] == str(target)
        assert str(target) in health["detail"]

    def test_missing_directory_reports_ok_missing(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        assert not sessions_dir.exists() and not sessions_dir.is_symlink()

        health = session_storage_health(sessions_dir)

        assert health["status"] == "ok"
        assert health["reason"] == "missing"
        # Nothing was created by the probe:
        assert not sessions_dir.exists()

    def test_dangling_symlink_reports_distinct_unavailable(self, tmp_path):
        """The core case: ~/.hermes/sessions -> unmounted volume target."""
        missing_target = tmp_path / "unmounted_volume" / "sessions"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.symlink_to(missing_target)
        assert sessions_dir.is_symlink() and not sessions_dir.exists()

        health = session_storage_health(sessions_dir)

        assert health["status"] == "unavailable"
        assert health["reason"] == "dangling_symlink"
        # The configured path and the intended target are both identified:
        assert health["path"] == str(sessions_dir)
        assert health["target"] == str(missing_target)
        assert str(missing_target) in health["detail"]
        assert "unavailable" in health["detail"]
        # The probe is read-only:
        assert sessions_dir.is_symlink() and not sessions_dir.exists()

    def test_dangling_symlink_in_parent_chain(self, tmp_path):
        """Storage unavailable higher in the tree (e.g. ~/.hermes itself is a
        dangling link) must also surface as unavailable, not as 'missing'."""
        (tmp_path / "dangling_home").symlink_to(tmp_path / "no_such_volume")
        sessions_dir = tmp_path / "dangling_home" / "sessions"

        health = session_storage_health(sessions_dir)

        assert health["status"] == "unavailable"
        assert health["reason"] == "dangling_symlink_in_parent_chain"
        assert health["target"] == str(tmp_path / "no_such_volume")

    def test_non_directory_entry_reports_error(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.write_text("not a directory", encoding="utf-8")

        health = session_storage_health(sessions_dir)

        assert health["status"] == "error"
        assert health["reason"] == "not_a_directory"
        assert str(sessions_dir) in health["detail"]

    def test_symlink_to_non_directory_reports_error(self, tmp_path):
        file_target = tmp_path / "a_file"
        file_target.write_text("data", encoding="utf-8")
        sessions_dir = tmp_path / "sessions"
        sessions_dir.symlink_to(file_target)

        health = session_storage_health(sessions_dir)

        assert health["status"] == "error"
        assert health["reason"] == "not_a_directory"

    def test_no_hardcoded_paths(self, tmp_path, monkeypatch):
        """Result content is derived from the given path and symlink metadata
        only — no username or mount path is baked in. Home is neutralized so
        any implicit Path.home() / $HOME leakage would show up."""
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "gone"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.symlink_to(target)

        health = session_storage_health(sessions_dir)

        # Every string the probe emits is composed from the inputs above.
        assert health["path"] == str(sessions_dir)
        assert health["target"] == str(target)
        for value in health.values():
            if isinstance(value, str):
                assert "/Users/" not in value
                assert "/Volumes/" not in value
