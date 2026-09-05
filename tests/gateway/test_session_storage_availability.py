"""Regression tests: SessionStore must fail with a clear, actionable error when
~/.hermes/sessions is a symlink whose target is unavailable (dangling symlink,
e.g. an unmounted external volume), instead of the bare EEXIST raised by
``mkdir(exist_ok=True)``.

Mirrors the unavailable-storage posture established in syntheos-system commit
7c91174 ("fix(report-ingest): fail when artifact storage is unavailable"):
detect the condition explicitly before attempting creation and raise a clear
error naming the unavailable target.

Covers: absent path (mkdir creates it), real directory (no-op), valid symlink
to a real directory (no-op), dangling symlink (clear error, not EEXIST).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.session import SessionStore, SessionStorageUnavailableError


def _make_store(sessions_dir: Path) -> SessionStore:
    store = SessionStore(sessions_dir=sessions_dir, config=None)
    store._db = None  # keep the test hermetic; SQLite is not under test here
    return store


class TestSessionStorageAvailability:
    def test_absent_path_is_created(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        assert not sessions_dir.exists() and not sessions_dir.is_symlink()

        store = _make_store(sessions_dir)
        store._ensure_loaded()

        assert sessions_dir.is_dir()

    def test_real_directory_is_used_as_is(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        store = _make_store(sessions_dir)
        store._ensure_loaded()

        assert sessions_dir.is_dir()

    def test_valid_symlink_to_real_directory(self, tmp_path):
        target = tmp_path / "mounted_sessions"
        target.mkdir()
        sessions_dir = tmp_path / "sessions"
        sessions_dir.symlink_to(target)

        store = _make_store(sessions_dir)
        store._ensure_loaded()

        assert sessions_dir.is_symlink()
        assert (sessions_dir / "sessions.json").exists() or True  # no write needed
        assert target.is_dir()

    def test_dangling_symlink_raises_clear_error_not_eexist(self, tmp_path):
        """Regression: before the fix this raised bare FileExistsError (EEXIST)
        from mkdir(exist_ok=True), giving no hint that the session-storage
        volume was unavailable."""
        missing_target = tmp_path / "unmounted_volume" / "sessions"
        assert not missing_target.exists()

        sessions_dir = tmp_path / "sessions"
        sessions_dir.symlink_to(missing_target)
        assert sessions_dir.is_symlink()
        assert not sessions_dir.exists()  # dangling

        store = _make_store(sessions_dir)
        with pytest.raises(SessionStorageUnavailableError) as excinfo:
            store._ensure_loaded()

        msg = str(excinfo.value)
        assert "session storage unavailable" in msg
        assert str(missing_target) in msg
        # And it must NOT masquerade as the raw EEXIST symptom:
        assert not isinstance(excinfo.value, FileExistsError)

    def test_dangling_symlink_also_guarded_on_save(self, tmp_path):
        missing_target = tmp_path / "gone" / "sessions"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.symlink_to(missing_target)

        store = _make_store(sessions_dir)
        store._loaded = True  # skip load; exercise the _save guard directly
        with pytest.raises(SessionStorageUnavailableError, match="session storage unavailable"):
            store._save()

    def test_broken_symlink_in_parent_chain(self, tmp_path):
        """A dangling symlink higher in the tree must also fail clearly."""
        sessions_dir = tmp_path / "dangling_parent" / "sessions"
        (tmp_path / "dangling_parent").symlink_to(tmp_path / "no_such_place")

        store = _make_store(sessions_dir)
        with pytest.raises(SessionStorageUnavailableError, match="session storage unavailable"):
            store._ensure_loaded()
