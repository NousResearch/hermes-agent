"""Tests for hermes_cli.desktop_composer_images — composer-images path resolution,
@image: ref extraction, and ref-counted cleanup on session deletion + orphan sweep.

Behavior contracts tested (invariant tests, never change-detector snapshots):

- Path resolution: ``get_composer_images_dir`` is platform-correct, honors
  ``HERMES_DESKTOP_USER_DATA_DIR`` and ``HERMES_DATA_DIR_SUFFIX`` overrides mirroring
  the Electron side.
- Ref extraction: ``extract_composer_image_paths`` returns only composer-image paths
  under the configured dir, tolerates str / parts-list / dict content shapes, and
  strips the three quote styles the persistence layer uses around paths with spaces.
- Ref collection: ``collect_composer_refs_for_sessions`` reads refs from a live DB
  and ``collect_active_composer_refs`` skips a caller-provided exclude set.
- Deletion cleanup: ``cleanup_composer_for_deletion`` unlinks only refs that appear
  in the deleted set AND have no surviving reference in the remaining DB rows.
- Orphan sweep: ``cleanup_orphaned_composer_images`` respects the *max_age_hours*
  grace window (newer files never touched), the DB reference set, and handles a
  missing state.db gracefully.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import desktop_composer_images as dci


_MESSAGES_SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    timestamp REAL NOT NULL
);
"""


def _env_dir(var, fallback: Path) -> Path:
    import os
    return Path(os.environ[var]) if var in os.environ else fallback


class TestComposerDir:
    def test_linux_default_uses_xdg_config_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("HERMES_DESKTOP_USER_DATA_DIR", raising=False)
        monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
        # Reimport to pick up the sys.platform monkeypatch inside the module.
        import importlib
        importlib.reload(dci)
        assert dci.get_composer_images_dir() == tmp_path / "cfg" / "Hermes" / "composer-images"

    def test_macos_default_uses_application_support(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("HERMES_DESKTOP_USER_DATA_DIR", raising=False)
        monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib
        importlib.reload(dci)
        assert dci.get_composer_images_dir() == (
            tmp_path / "Library" / "Application Support" / "Hermes" / "composer-images"
        )

    def test_windows_default_uses_appdata(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delenv("HERMES_DESKTOP_USER_DATA_DIR", raising=False)
        monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
        monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib
        importlib.reload(dci)
        assert dci.get_composer_images_dir() == (
            tmp_path / "Roaming" / "Hermes" / "composer-images"
        )

    def test_full_override_env_wins_over_platform_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("HERMES_DESKTOP_USER_DATA_DIR", str(tmp_path / "custom"))
        monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
        import importlib
        importlib.reload(dci)
        # The override replaces the whole userData dir; suffix is ignored when override is set.
        assert dci.get_composer_images_dir() == tmp_path / "custom" / "composer-images"

    def test_suffix_appended_to_platform_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("HERMES_DESKTOP_USER_DATA_DIR", raising=False)
        monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "-dev")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
        import importlib
        importlib.reload(dci)
        # userData becomes "Hermes-dev", composer-images still a subdir of it.
        assert dci.get_composer_images_dir() == (
            tmp_path / "cfg" / "Hermes-dev" / "composer-images"
        )


class TestExtractComposerImagePaths:
    @pytest.fixture()
    def composer_dir(self, tmp_path):
        return tmp_path / "user" / "Hermes" / "composer-images"

    def test_plain_string_content_finds_ref_under_dir(self, composer_dir):
        img = composer_dir / "snap.png"
        text = f"hello\n@image:{img}"
        assert dci.extract_composer_image_paths(text, composer_dir) == [str(img)]

    def test_skips_refs_outside_composer_dir(self, composer_dir, tmp_path):
        outside = tmp_path / "other.png"
        text = f"caption\n@image:{outside}\n@image:{composer_dir / 'inside.png'}"
        result = dci.extract_composer_image_paths(text, composer_dir)
        assert str(outside) not in result
        assert str(composer_dir / "inside.png") in result

    def test_backtick_quoted_path_with_spaces(self, composer_dir):
        # macOS "Application Support" / Windows paths with spaces land here.
        img = composer_dir / "my file name.png"
        text = f"look\n@image:`{img}`"
        assert dci.extract_composer_image_paths(text, composer_dir) == [str(img)]

    def test_double_and_single_quoted_forms(self, composer_dir):
        a = composer_dir / "a.png"
        b = composer_dir / "b.png"
        text = f'row1\n@image:"{a}"\nrow2\n@image:\'{b}\''
        assert set(dci.extract_composer_image_paths(text, composer_dir)) == {str(a), str(b)}

    def test_parts_list_content_reads_text_parts_only(self, composer_dir):
        img = composer_dir / "shot.png"
        content = [
            {"type": "text", "text": f"see\n@image:{img}"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        assert dci.extract_composer_image_paths(content, composer_dir) == [str(img)]

    def test_dict_content_single_shape(self, composer_dir):
        img = composer_dir / "dict.png"
        assert dci.extract_composer_image_paths(
            {"content": f"@image:{img}"}, composer_dir) == [str(img)]

    def test_none_and_empty_returns_empty(self, composer_dir):
        assert dci.extract_composer_image_paths(None, composer_dir) == []
        assert dci.extract_composer_image_paths("", composer_dir) == []
        assert dci.extract_composer_image_paths([], composer_dir) == []


class TestRefCollectionAndDeletionCleanup:
    @pytest.fixture()
    def conn(self):
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.executescript(_MESSAGES_SCHEMA)
        return c

    @pytest.fixture()
    def composer_dir(self, tmp_path, monkeypatch):
        userdata = tmp_path / "userdata"
        monkeypatch.setenv("HERMES_DESKTOP_USER_DATA_DIR", str(userdata))
        import importlib
        importlib.reload(dci)
        d = userdata / "composer-images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _seed(self, conn, sid, *images):
        now = time.time()
        conn.execute("INSERT INTO sessions (id) VALUES (?)", (sid,))
        for idx, img in enumerate(images):
            text = f"msg-{idx}\n" + "\n".join(f"@image:{p}" for p in img) if img else f"msg-{idx}"
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', ?, ?)",
                (sid, text, now),
            )
        conn.commit()

    def test_collect_for_sessions_reads_only_requested_sessions(self, conn, composer_dir):
        a, b, shared = [composer_dir / f"{n}.png" for n in ("a", "b", "shared")]
        for p in (a, b, shared):
            p.write_bytes(b"\x89PNG")
        self._seed(conn, "S1", [a, shared])
        self._seed(conn, "S2", [b, shared])
        refs = dci.collect_composer_refs_for_sessions(conn, ["S1"])
        assert str(a) in refs and str(shared) in refs
        assert str(b) not in refs

    def test_collect_active_excludes_caller_ids(self, conn, composer_dir):
        a, b = [composer_dir / f"{n}.png" for n in ("a", "b")]
        a.write_bytes(b"1")
        b.write_bytes(b"2")
        self._seed(conn, "S1", [a])
        self._seed(conn, "S2", [b])
        refs = dci.collect_active_composer_refs(conn, composer_dir, exclude_session_ids={"S2"})
        assert str(a) in refs and str(b) not in refs

    def test_cleanup_deletion_removes_refs_unique_to_deleted_set(self, conn, composer_dir):
        only_s1, only_s2, shared = [composer_dir / f"{n}.png" for n in ("only1", "only2", "both")]
        for p in (only_s1, only_s2, shared):
            p.write_bytes(b"\x89PNG")
        self._seed(conn, "S1", [only_s1, shared])
        self._seed(conn, "S2", [only_s2, shared])
        deleted_refs = dci.collect_composer_refs_for_sessions(conn, ["S1"])
        # Simulate S1's rows being gone from the DB before cleanup runs.
        conn.execute("DELETE FROM messages WHERE session_id = 'S1'")
        conn.execute("DELETE FROM sessions WHERE id = 'S1'")
        conn.commit()
        removed = dci.cleanup_composer_for_deletion(conn, deleted_refs)
        # only_s1 is gone, shared stays because S2 still references it.
        assert removed == 1
        assert not only_s1.exists()
        assert shared.exists() and only_s2.exists()

    def test_cleanup_deletion_tolerates_missing_files(self, conn, composer_dir):
        ghost = composer_dir / "ghost.png"
        self._seed(conn, "S1", [ghost])  # never actually written to disk
        deleted_refs = {str(ghost)}
        conn.execute("DELETE FROM messages WHERE session_id = 'S1'")
        conn.commit()
        # _safe_unlink_many must not raise; count reflects actually-unlinked files (0).
        assert dci.cleanup_composer_for_deletion(conn, deleted_refs) == 0


class TestOrphanSweep:
    @pytest.fixture()
    def composer_dir(self, tmp_path, monkeypatch):
        userdata = tmp_path / "userdata"
        monkeypatch.setenv("HERMES_DESKTOP_USER_DATA_DIR", str(userdata))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        import importlib
        importlib.reload(dci)
        d = userdata / "composer-images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write(self, path: Path, *, days_old: int) -> Path:
        path.write_bytes(b"\x89PNG")
        old = time.time() - days_old * 86400
        Path.touch(path) if False else None  # touch placeholder, see timestamp set below
        import os
        os.utime(path, (old, old))
        return path

    def test_grace_window_preserves_new_files_even_if_unreferenced(self, composer_dir, tmp_path):
        fresh = self._write(composer_dir / "fresh.png", days_old=1)
        stale_unreferenced = self._write(composer_dir / "stale.png", days_old=30)
        count = dci.cleanup_orphaned_composer_images(max_age_hours=168)
        assert count == 1
        assert fresh.exists()
        assert not stale_unreferenced.exists()

    def test_stale_file_referenced_in_state_db_is_kept(self, composer_dir, tmp_path, monkeypatch):
        from hermes_constants import get_hermes_home
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        db_path = home / "state.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(_MESSAGES_SCHEMA)
        kept = self._write(composer_dir / "kept.png", days_old=30)
        conn.execute("INSERT INTO sessions (id) VALUES ('S')")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('S','user',?,?)",
            (f"@image:{kept}", time.time()),
        )
        conn.commit()
        conn.close()
        count = dci.cleanup_orphaned_composer_images(max_age_hours=1)
        assert count == 0
        assert kept.exists()

    def test_missing_state_db_sweeps_all_old_files(self, composer_dir, tmp_path, monkeypatch):
        # No state.db under HERMES_HOME → every stale file qualifies.
        home = tmp_path / "empty_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        self._write(composer_dir / "a.png", days_old=30)
        self._write(composer_dir / "b.png", days_old=30)
        self._write(composer_dir / "new.png", days_old=1)
        assert dci.cleanup_orphaned_composer_images(max_age_hours=24) == 2
        assert (composer_dir / "new.png").exists()
