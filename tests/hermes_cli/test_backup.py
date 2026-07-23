"""Tests for hermes backup and import commands."""

import errno
import json
import os
import shutil
import sqlite3
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeScandirIterator:
    """A faithful stand-in for os.scandir()'s return value: an ITERATOR
    (supports direct next(), not just `for x in it`) AND a context manager
    (create_quick_snapshot() does ``with scandir_it:`` then advances via
    next(), matching CPython's own os.walk() implementation — see #68907
    review pass 6). A plain list supports neither: no __enter__/__exit__,
    and calling next() on a list itself (rather than iter(list)) raises
    TypeError. Tests that fake os.scandir() must wrap their entries in
    this so they exercise the real code path instead of an unrelated
    protocol mismatch.

    Tracks `closed` so a test can assert the `with scandir_it:` block
    actually released the (simulated) OS handle — including on an early
    break, e.g. when the traversal budget trips mid-listing (#68907
    review pass 7 nit)."""

    def __init__(self, entries):
        self._it = iter(entries)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def close(self):
        self.closed = True


def _make_hermes_tree(root: Path) -> None:
    """Create a realistic ~/.hermes directory structure for testing."""
    (root / "config.yaml").write_text("model:\n  provider: openrouter\n")
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-test-123\n")
    for db_name in ("memory_store.db", "hermes_state.db"):
        with sqlite3.connect(root / db_name) as conn:
            conn.execute("CREATE TABLE sample (value TEXT)")
            conn.execute("INSERT INTO sample VALUES ('test')")

    # Sessions
    (root / "sessions").mkdir(exist_ok=True)
    (root / "sessions" / "abc123.json").write_text("{}")

    # Skills
    (root / "skills").mkdir(exist_ok=True)
    (root / "skills" / "my-skill").mkdir()
    (root / "skills" / "my-skill" / "SKILL.md").write_text("# My Skill\n")

    # Skins
    (root / "skins").mkdir(exist_ok=True)
    (root / "skins" / "cyber.yaml").write_text("name: cyber\n")

    # Cron
    (root / "cron").mkdir(exist_ok=True)
    (root / "cron" / "jobs.json").write_text("[]")

    # Memories
    (root / "memories").mkdir(exist_ok=True)
    (root / "memories" / "notes.json").write_text("{}")

    # Profiles
    (root / "profiles").mkdir(exist_ok=True)
    (root / "profiles" / "coder").mkdir()
    (root / "profiles" / "coder" / "config.yaml").write_text("model:\n  provider: anthropic\n")
    (root / "profiles" / "coder" / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-123\n")

    # hermes-agent repo (should be EXCLUDED)
    (root / "hermes-agent").mkdir(exist_ok=True)
    (root / "hermes-agent" / "run_agent.py").write_text("# big file\n")
    (root / "hermes-agent" / ".git").mkdir()
    (root / "hermes-agent" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    # __pycache__ (should be EXCLUDED)
    (root / "plugins").mkdir(exist_ok=True)
    (root / "plugins" / "__pycache__").mkdir()
    (root / "plugins" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")

    # PID files (should be EXCLUDED)
    (root / "gateway.pid").write_text("12345")

    # Logs (should be included)
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "agent.log").write_text("log line\n")


def _symlink_file_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")


# ---------------------------------------------------------------------------
# _should_exclude tests
# ---------------------------------------------------------------------------

class TestShouldExclude:
    def test_excludes_hermes_agent(self):
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("hermes-agent/run_agent.py"))
        assert _should_exclude(Path("hermes-agent/.git/HEAD"))

    def test_excludes_pycache(self):
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("plugins/__pycache__/mod.cpython-312.pyc"))

    def test_excludes_pyc_files(self):
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("some/module.pyc"))

    def test_excludes_pid_files(self):
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("gateway.pid"))
        assert _should_exclude(Path("cron.pid"))

    def test_excludes_checkpoints(self):
        """checkpoints/ is session-local trajectory cache — hash-keyed,
        regenerated per-session, won't port to another machine anyway."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("checkpoints/abc123/trajectory.json"))
        assert _should_exclude(Path("checkpoints/deadbeef/step_0001.json"))

    def test_excludes_backups_dir(self):
        """backups/ is excluded so pre-update backups don't nest exponentially."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("backups/pre-update-2026-04-27-063400.zip"))

    def test_excludes_sqlite_sidecars(self):
        """SQLite WAL/SHM/journal sidecars must not ship alongside the
        safe-copied .db — pairing a fresh snapshot with stale sidecar state
        produces a torn restore."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("state.db-wal"))
        assert _should_exclude(Path("state.db-shm"))
        assert _should_exclude(Path("state.db-journal"))
        assert _should_exclude(Path("memory_store.db-wal"))
        # The .db itself is still included (and safe-copied separately)
        assert not _should_exclude(Path("state.db"))

    def test_includes_config(self):
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("config.yaml"))

    def test_includes_env(self):
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path(".env"))

    def test_includes_skills(self):
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/my-skill/SKILL.md"))

    def test_includes_profiles(self):
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("profiles/coder/config.yaml"))

    def test_includes_sessions(self):
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("sessions/abc.json"))

    def test_includes_logs(self):
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("logs/agent.log"))

    def test_includes_nested_hermes_agent_in_skills(self):
        """skills/autonomous-ai-agents/hermes-agent/ must NOT be excluded —
        only the root-level hermes-agent/ repo is skipped."""
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/autonomous-ai-agents/hermes-agent/SKILL.md"))
        assert not _should_exclude(Path("skills/autonomous-ai-agents/hermes-agent/sub/item.txt"))

    @pytest.mark.parametrize(
        "rel",
        [
            "plugins/my-plugin/.venv/lib/python3.12/site-packages/x/__init__.py",
            "plugins/my-plugin/venv/bin/python",
            "mcp/server/site-packages/pkg/mod.py",
            ".cache/uv/wheels/abc.whl",
            "plugins/p/.cache/pip/http/deadbeef",
            ".tox/py312/log.txt",
            ".nox/tests/bin/pytest",
            "plugins/p/.pytest_cache/v/cache/lastfailed",
            ".mypy_cache/3.12/agent.meta.json",
            ".ruff_cache/0.4.0/abc",
        ],
    )
    def test_excludes_regeneratable_dependency_and_cache_dirs(self, rel):
        """Python dep trees and tool caches under HERMES_HOME must be skipped —
        these are what balloon a backup to hundreds of thousands of files."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path(rel))

    def test_does_not_exclude_curator_archive(self):
        """skills/.archive/ holds restorable archived skills and MUST survive
        a backup — it is intentionally NOT in the exclusion set."""
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/.archive/old-skill/SKILL.md"))

    def test_does_not_exclude_legit_files_resembling_cache_names(self):
        """Only directory-component matches are excluded; a normal file is kept."""
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/my-skill/venv-notes.md"))
        assert not _should_exclude(Path("memories/cache.json"))

# ---------------------------------------------------------------------------
# Backup tests
# ---------------------------------------------------------------------------

class TestBackup:
    def test_creates_zip(self, tmp_path, monkeypatch):
        """Backup creates a valid zip containing expected files."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # get_default_hermes_root needs this
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        assert out_zip.exists()
        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            # Config should be present
            assert "config.yaml" in names
            assert ".env" in names
            # Skills
            assert "skills/my-skill/SKILL.md" in names
            # Profiles
            assert "profiles/coder/config.yaml" in names
            assert "profiles/coder/.env" in names
            # Sessions
            assert "sessions/abc123.json" in names
            # Logs
            assert "logs/agent.log" in names
            # Skins
            assert "skins/cyber.yaml" in names

    def test_failed_sqlite_backup_never_raw_copies_live_wal_db(self, tmp_path, monkeypatch, capsys):
        """A failed backup() must not silently archive the stale main DB file.

        Keep a real, uncheckpointed WAL transaction live so a raw copy of only
        ``state.db`` would be a valid-looking but torn snapshot.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")
        db_path = hermes_home / "state.db"

        writer = sqlite3.connect(db_path)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE events (value TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO events VALUES ('only-in-wal')")
        writer.commit()
        assert Path(f"{db_path}-wal").stat().st_size > 0

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        import hermes_cli.backup as backup_mod
        real_connect = backup_mod.sqlite3.connect

        class FailingBackupConnection:
            def __init__(self, connection):
                self._connection = connection

            def backup(self, _destination):
                raise sqlite3.OperationalError("forced backup failure")

            def close(self):
                self._connection.close()

        def connect_with_failed_backup(database, *args, **kwargs):
            connection = real_connect(database, *args, **kwargs)
            if str(database).startswith(f"file:{db_path}"):
                return FailingBackupConnection(connection)
            return connection

        monkeypatch.setattr(backup_mod.sqlite3, "connect", connect_with_failed_backup)
        out_zip = tmp_path / "backup.zip"
        try:
            backup_mod.run_backup(Namespace(output=str(out_zip)))
        finally:
            writer.close()

        with zipfile.ZipFile(out_zip) as zf:
            assert "config.yaml" in zf.namelist()
            assert "state.db" not in zf.namelist()

        output = capsys.readouterr().out
        assert "Backup incomplete" in output
        assert "state.db: SQLite safe copy failed" in output
        assert "Restore with:" not in output

    def test_db_snapshots_staged_beside_output_zip(self, tmp_path, monkeypatch):
        """SQLite staging temp files must be created on the output zip's
        filesystem (dir=out_path.parent), NOT the system /tmp default — a
        small tmpfs there silently drops large DBs from the backup (#35376)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_dir = tmp_path / "external-drive"
        out_dir.mkdir()
        out_zip = out_dir / "backup.zip"
        args = Namespace(output=str(out_zip))

        import hermes_cli.backup as backup_mod
        staged_dirs = []
        real_ntf = backup_mod.tempfile.NamedTemporaryFile

        def _spy(*a, **kw):
            staged_dirs.append(kw.get("dir"))
            return real_ntf(*a, **kw)

        monkeypatch.setattr(backup_mod.tempfile, "NamedTemporaryFile", _spy)
        backup_mod.run_backup(args)

        # At least one .db was staged, and every staging call targeted the
        # output zip's directory rather than the system temp default.
        assert staged_dirs, "no SQLite snapshot was staged"
        assert all(d == str(out_dir) for d in staged_dirs), staged_dirs

    def test_pre_update_db_snapshots_staged_beside_output_zip(self, tmp_path, monkeypatch):
        """The pre-update/pre-migration zip path (_write_full_zip_backup) must
        also stage SQLite snapshots beside its output zip, not in /tmp."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = hermes_home / "backups" / "pre-update-test.zip"
        out_zip.parent.mkdir(parents=True, exist_ok=True)

        import hermes_cli.backup as backup_mod
        staged_dirs = []
        real_ntf = backup_mod.tempfile.NamedTemporaryFile

        def _spy(*a, **kw):
            staged_dirs.append(kw.get("dir"))
            return real_ntf(*a, **kw)

        monkeypatch.setattr(backup_mod.tempfile, "NamedTemporaryFile", _spy)
        result = backup_mod._write_full_zip_backup(out_zip, hermes_home)

        assert result is not None
        assert staged_dirs, "no SQLite snapshot was staged"
        assert all(d == str(out_zip.parent) for d in staged_dirs), staged_dirs

    def test_excludes_hermes_agent(self, tmp_path, monkeypatch):
        """Backup does NOT include hermes-agent/ directory."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            agent_files = [n for n in names if "hermes-agent" in n]
            assert agent_files == [], f"hermes-agent files leaked into backup: {agent_files}"

    def test_excludes_dependency_and_cache_trees(self, tmp_path, monkeypatch):
        """A plugin venv / site-packages / pip cache under HERMES_HOME must be
        pruned by the walk, while real data (skills, config) is preserved.
        This is the regression guard for the ballooning-backup bug."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        # Simulate the heavy regeneratable trees that ballooned the backup.
        venv_pkg = hermes_home / "plugins" / "heavy" / ".venv" / "lib" / "site-packages" / "dep"
        venv_pkg.mkdir(parents=True)
        (venv_pkg / "__init__.py").write_text("# dep\n")
        pip_cache = hermes_home / ".cache" / "uv" / "wheels"
        pip_cache.mkdir(parents=True)
        (pip_cache / "abc.whl").write_bytes(b"\x00")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        from hermes_cli.backup import run_backup
        run_backup(Namespace(output=str(out_zip)))

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
        leaked = [n for n in names if ".venv" in n or "site-packages" in n or ".cache" in n]
        assert leaked == [], f"regeneratable trees leaked into backup: {leaked}"
        # Real data still present.
        assert "skills/my-skill/SKILL.md" in names
        assert "config.yaml" in names

    def test_includes_nested_hermes_agent_in_skills(self, tmp_path, monkeypatch):
        """Backup includes skills/.../hermes-agent/ but NOT root hermes-agent/."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        # Add a nested hermes-agent directory inside skills (like the real layout)
        nested = hermes_home / "skills" / "autonomous-ai-agents" / "hermes-agent"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("# Hermes Agent Skill\n")
        (nested / "sub").mkdir()
        (nested / "sub" / "item.txt").write_text("nested content\n")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            # Root hermes-agent must be excluded
            root_agent = [n for n in names if n.startswith("hermes-agent/")]
            assert root_agent == [], f"root hermes-agent leaked: {root_agent}"
            # Nested skill hermes-agent must be included
            assert "skills/autonomous-ai-agents/hermes-agent/SKILL.md" in names
            assert "skills/autonomous-ai-agents/hermes-agent/sub/item.txt" in names

    def test_excludes_pycache(self, tmp_path, monkeypatch):
        """Backup does NOT include __pycache__ dirs."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            pycache_files = [n for n in names if "__pycache__" in n]
            assert pycache_files == []

    def test_excludes_pid_files(self, tmp_path, monkeypatch):
        """Backup does NOT include PID files."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            pid_files = [n for n in names if n.endswith(".pid")]
            assert pid_files == []

    def test_default_output_path(self, tmp_path, monkeypatch):
        """When no output path given, zip goes to ~/hermes-backup-*.zip."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        args = Namespace(output=None)

        from hermes_cli.backup import run_backup
        run_backup(args)

        # Should exist in home dir
        zips = list(tmp_path.glob("hermes-backup-*.zip"))
        assert len(zips) == 1

    def test_skips_symlinked_files(self, tmp_path, monkeypatch):
        """Backup must not dereference symlinks and leak files outside HERMES_HOME."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)
        outside = tmp_path / "outside-secret.txt"
        outside.write_text("outside secret\n")
        _symlink_file_or_skip(hermes_home / "skills" / "outside-link.txt", outside)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "backup.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            assert "skills/outside-link.txt" not in names
            assert all(zf.read(name) != b"outside secret\n" for name in names)


# ---------------------------------------------------------------------------
# _validate_backup_zip tests
# ---------------------------------------------------------------------------

class TestValidateBackupZip:
    def _make_zip(self, zip_path: Path, filenames: list[str]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name in filenames:
                zf.writestr(name, "dummy")

    def test_state_db_passes(self, tmp_path):
        """A zip containing state.db is accepted as a valid Hermes backup."""
        from hermes_cli.backup import _validate_backup_zip
        zip_path = tmp_path / "backup.zip"
        self._make_zip(zip_path, ["state.db", "sessions/abc.json"])
        with zipfile.ZipFile(zip_path, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok, reason

    def test_old_wrong_db_name_fails(self, tmp_path):
        """A zip with only hermes_state.db (old wrong name) is rejected."""
        from hermes_cli.backup import _validate_backup_zip
        zip_path = tmp_path / "old.zip"
        self._make_zip(zip_path, ["hermes_state.db", "memory_store.db"])
        with zipfile.ZipFile(zip_path, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert not ok

    def test_config_yaml_passes(self, tmp_path):
        """A zip containing config.yaml is accepted (existing behaviour preserved)."""
        from hermes_cli.backup import _validate_backup_zip
        zip_path = tmp_path / "backup.zip"
        self._make_zip(zip_path, ["config.yaml", "skills/x/SKILL.md"])
        with zipfile.ZipFile(zip_path, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok, reason


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        """Create a test zip with given files."""
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                if isinstance(content, bytes):
                    zf.writestr(name, content)
                else:
                    zf.writestr(name, content)

    def test_restores_files(self, tmp_path, monkeypatch):
        """Import extracts files into hermes home."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model:\n  provider: openrouter\n",
            ".env": "OPENROUTER_API_KEY=sk-test\n",
            "skills/my-skill/SKILL.md": "# My Skill\n",
            "profiles/coder/config.yaml": "model:\n  provider: anthropic\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        assert (hermes_home / "config.yaml").read_text() == "model:\n  provider: openrouter\n"
        assert (hermes_home / ".env").read_text() == "OPENROUTER_API_KEY=sk-test\n"
        assert (hermes_home / "skills" / "my-skill" / "SKILL.md").read_text() == "# My Skill\n"
        assert (hermes_home / "profiles" / "coder" / "config.yaml").exists()

    def test_strips_hermes_prefix(self, tmp_path, monkeypatch):
        """Import strips .hermes/ prefix if all entries share it."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            ".hermes/config.yaml": "model: test\n",
            ".hermes/skills/a/SKILL.md": "# A\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        assert (hermes_home / "config.yaml").read_text() == "model: test\n"
        assert (hermes_home / "skills" / "a" / "SKILL.md").read_text() == "# A\n"

    def test_rejects_empty_zip(self, tmp_path, monkeypatch):
        """Import rejects an empty zip."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass  # empty

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    def test_rejects_non_hermes_zip(self, tmp_path, monkeypatch):
        """Import rejects a zip that doesn't look like a hermes backup."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "random.zip"
        self._make_backup_zip(zip_path, {
            "some/random/file.txt": "hello",
            "another/thing.json": "{}",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    def test_blocks_path_traversal(self, tmp_path, monkeypatch):
        """Import blocks zip entries with path traversal."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "evil.zip"
        # Include a marker file so validation passes
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "../../etc/passwd": "root:x:0:0\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        # config.yaml should be restored
        assert (hermes_home / "config.yaml").exists()
        # traversal file should NOT exist outside hermes home
        assert not (tmp_path / "etc" / "passwd").exists()

    def test_preserves_live_gateway_state(self, tmp_path, monkeypatch):
        """Import must not overwrite the target's gateway_state.json.

        The backup carries the *source* machine's gateway run/desired state.
        Restoring it onto a hosted container drives the boot reconciler off
        stale/foreign state and leaves the gateway stuck "starting",
        disconnecting it from the Nous portal (NS-508). The live file wins.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # The target (e.g. hosted container) already has its own live state.
        live_state = '{"gateway_state": "running", "desired_state": "running"}'
        (hermes_home / "gateway_state.json").write_text(live_state)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            # A backup from a laptop where the gateway was stopped.
            "gateway_state.json": '{"gateway_state": "stopped", "desired_state": "stopped"}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        # config.yaml is restored normally...
        assert (hermes_home / "config.yaml").read_text() == "model: test\n"
        # ...but the live gateway_state.json is untouched.
        assert (hermes_home / "gateway_state.json").read_text() == live_state

    def test_does_not_seed_gateway_state_when_absent(self, tmp_path, monkeypatch):
        """A backup's gateway_state.json is dropped, not written, when the
        target has none — a foreign state must never seed the reconciler."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "gateway_state.json": '{"gateway_state": "stopped"}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        assert (hermes_home / "config.yaml").exists()
        assert not (hermes_home / "gateway_state.json").exists()

    def test_preserves_per_profile_gateway_state(self, tmp_path, monkeypatch):
        """The skip is matched by basename, so a named profile's
        gateway_state.json (profiles/<name>/gateway_state.json) is preserved
        the same way the root profile's is."""
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "profiles" / "coder").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        live_state = '{"gateway_state": "running"}'
        (hermes_home / "profiles" / "coder" / "gateway_state.json").write_text(live_state)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "profiles/coder/config.yaml": "model: anthropic\n",
            "profiles/coder/gateway_state.json": '{"gateway_state": "stopped"}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        # Profile config is restored, but its live gateway state is preserved.
        assert (hermes_home / "profiles" / "coder" / "config.yaml").read_text() == "model: anthropic\n"
        assert (
            hermes_home / "profiles" / "coder" / "gateway_state.json"
        ).read_text() == live_state

    def test_preserves_runtime_pid_and_process_files(self, tmp_path, monkeypatch):
        """gateway.pid / cron.pid / gateway.lock / processes.json from a backup
        reference the source machine's process namespace and must never be
        written over the target's."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Live runtime files belonging to the target's own processes.
        (hermes_home / "gateway.pid").write_text("4242")
        (hermes_home / "processes.json").write_text('{"live": true}')

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "gateway.pid": "9999",
            "cron.pid": "8888",
            "gateway.lock": "7777",
            "processes.json": '{"stale": true}',
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        # Live runtime files are untouched; the backup's foreign ones never land.
        assert (hermes_home / "gateway.pid").read_text() == "4242"
        assert (hermes_home / "processes.json").read_text() == '{"live": true}'
        # cron.pid / gateway.lock had no live copy and were not seeded.
        assert not (hermes_home / "cron.pid").exists()
        assert not (hermes_home / "gateway.lock").exists()

    def test_confirmation_prompt_abort(self, tmp_path, monkeypatch):
        """Import aborts when user says no to confirmation."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        # Pre-existing config triggers the confirmation
        (hermes_home / "config.yaml").write_text("existing: true\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: restored\n",
        })

        args = Namespace(zipfile=str(zip_path), force=False)

        from hermes_cli.backup import run_import
        with patch("builtins.input", return_value="n"):
            run_import(args)

        # Original config should be unchanged
        assert (hermes_home / "config.yaml").read_text() == "existing: true\n"

    def test_force_skips_confirmation(self, tmp_path, monkeypatch):
        """Import with --force skips confirmation and overwrites."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("existing: true\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: restored\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        assert (hermes_home / "config.yaml").read_text() == "model: restored\n"

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        """Import exits with error for nonexistent file."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        args = Namespace(zipfile=str(tmp_path / "nonexistent.zip"), force=True)

        from hermes_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
    def test_restores_secret_files_with_0600_perms(self, tmp_path, monkeypatch):
        """Secret files must end up at 0600 after restore (zipfile drops mode bits)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: openrouter\n",
            ".env": "OPENROUTER_API_KEY=sk-secret\n",
            "auth.json": '{"providers": {"nous": "token"}}',
            "state.db": b"SQLite format 3\x00",
            "profiles/coder/.env": "ANTHROPIC_API_KEY=sk-ant-secret\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        for rel in (".env", "auth.json", "state.db", "profiles/coder/.env"):
            mode = (hermes_home / rel).stat().st_mode & 0o777
            assert mode == 0o600, f"{rel} restored with mode {oct(mode)}, expected 0o600"


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_backup_then_import(self, tmp_path, monkeypatch):
        """Full round-trip: backup -> import to a new location -> verify."""
        # Source
        src_home = tmp_path / "source" / ".hermes"
        src_home.mkdir(parents=True)
        _make_hermes_tree(src_home)

        monkeypatch.setenv("HERMES_HOME", str(src_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "source")

        # Backup
        out_zip = tmp_path / "roundtrip.zip"
        from hermes_cli.backup import run_backup, run_import

        run_backup(Namespace(output=str(out_zip)))
        assert out_zip.exists()

        # Import into a different location
        dst_home = tmp_path / "dest" / ".hermes"
        dst_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(dst_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "dest")

        run_import(Namespace(zipfile=str(out_zip), force=True))

        # Verify key files
        assert (dst_home / "config.yaml").read_text() == "model:\n  provider: openrouter\n"
        assert (dst_home / ".env").read_text() == "OPENROUTER_API_KEY=sk-test-123\n"
        assert (dst_home / "skills" / "my-skill" / "SKILL.md").exists()
        assert (dst_home / "profiles" / "coder" / "config.yaml").exists()
        assert (dst_home / "sessions" / "abc123.json").exists()
        assert (dst_home / "logs" / "agent.log").exists()

        # hermes-agent should NOT be present
        assert not (dst_home / "hermes-agent").exists()
        # __pycache__ should NOT be present
        assert not (dst_home / "plugins" / "__pycache__").exists()
        # PID files should NOT be present
        assert not (dst_home / "gateway.pid").exists()


# ---------------------------------------------------------------------------
# Validate / detect-prefix unit tests
# ---------------------------------------------------------------------------

class TestFormatSize:
    def test_bytes(self):
        from hermes_cli.backup import _format_size
        assert _format_size(512) == "512 B"

    def test_kilobytes(self):
        from hermes_cli.backup import _format_size
        assert "KB" in _format_size(2048)

    def test_megabytes(self):
        from hermes_cli.backup import _format_size
        assert "MB" in _format_size(5 * 1024 * 1024)

    def test_gigabytes(self):
        from hermes_cli.backup import _format_size
        assert "GB" in _format_size(3 * 1024 ** 3)

    def test_terabytes(self):
        from hermes_cli.backup import _format_size
        assert "TB" in _format_size(2 * 1024 ** 4)


class TestValidation:
    def test_validate_with_config(self):
        """Zip with config.yaml passes validation."""
        import io
        from hermes_cli.backup import _validate_backup_zip

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("config.yaml", "test")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok

    def test_validate_with_env(self):
        """Zip with .env passes validation."""
        import io
        from hermes_cli.backup import _validate_backup_zip

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(".env", "KEY=val")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert ok

    def test_validate_rejects_random(self):
        """Zip without hermes markers fails validation."""
        import io
        from hermes_cli.backup import _validate_backup_zip

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("random/file.txt", "hello")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            ok, reason = _validate_backup_zip(zf)
        assert not ok

    def test_detect_prefix_hermes(self):
        """Detects .hermes/ prefix wrapping all entries."""
        import io
        from hermes_cli.backup import _detect_prefix

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(".hermes/config.yaml", "test")
            zf.writestr(".hermes/skills/a/SKILL.md", "skill")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert _detect_prefix(zf) == ".hermes/"

    def test_detect_prefix_none(self):
        """No prefix when entries are at root."""
        import io
        from hermes_cli.backup import _detect_prefix

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("config.yaml", "test")
            zf.writestr("skills/a/SKILL.md", "skill")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert _detect_prefix(zf) == ""

    def test_detect_prefix_only_dirs(self):
        """Prefix detection returns empty for zip with only directory entries."""
        import io
        from hermes_cli.backup import _detect_prefix

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # Only directory entries (trailing slash)
            zf.writestr(".hermes/", "")
            zf.writestr(".hermes/skills/", "")
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf:
            assert _detect_prefix(zf) == ""


# ---------------------------------------------------------------------------
# Edge case tests for uncovered paths
# ---------------------------------------------------------------------------

class TestBackupEdgeCases:
    def test_nonexistent_hermes_home(self, tmp_path, monkeypatch):
        """Backup exits when hermes home doesn't exist."""
        fake_home = tmp_path / "nonexistent" / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "nonexistent")

        args = Namespace(output=str(tmp_path / "out.zip"))

        from hermes_cli.backup import run_backup
        with pytest.raises(SystemExit):
            run_backup(args)

    def test_output_is_directory(self, tmp_path, monkeypatch):
        """When output path is a directory, zip is created inside it."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_dir = tmp_path / "backups"
        out_dir.mkdir()

        args = Namespace(output=str(out_dir))

        from hermes_cli.backup import run_backup
        run_backup(args)

        zips = list(out_dir.glob("hermes-backup-*.zip"))
        assert len(zips) == 1

    def test_output_without_zip_suffix(self, tmp_path, monkeypatch):
        """Output path without .zip gets suffix appended."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_path = tmp_path / "mybackup.tar"
        args = Namespace(output=str(out_path))

        from hermes_cli.backup import run_backup
        run_backup(args)

        # Should have .tar.zip suffix
        assert (tmp_path / "mybackup.tar.zip").exists()

    def test_empty_hermes_home(self, tmp_path, monkeypatch):
        """Backup handles empty hermes home (no files to back up)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        # Only excluded dirs, no actual files
        (hermes_home / "__pycache__").mkdir()
        (hermes_home / "__pycache__" / "foo.pyc").write_bytes(b"\x00")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        args = Namespace(output=str(tmp_path / "out.zip"))

        from hermes_cli.backup import run_backup
        run_backup(args)

        # No zip should be created
        assert not (tmp_path / "out.zip").exists()

    def test_permission_error_during_backup(self, tmp_path, monkeypatch):
        """Backup handles permission errors gracefully."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")

        # Create an unreadable file
        bad_file = hermes_home / "secret.db"
        bad_file.write_text("data")
        bad_file.chmod(0o000)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "out.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        try:
            run_backup(args)
        finally:
            # Restore permissions for cleanup
            bad_file.chmod(0o644)

        # Zip should still be created with the readable files
        assert out_zip.exists()

    def test_pre1980_timestamp_skipped(self, tmp_path, monkeypatch):
        """Backup skips files with pre-1980 timestamps (ZIP limitation)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")

        # Create a file with epoch timestamp (1970-01-01)
        old_file = hermes_home / "ancient.txt"
        old_file.write_text("old data")
        os.utime(old_file, (0, 0))

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        out_zip = tmp_path / "out.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        # Zip should still be created with the valid files
        assert out_zip.exists()
        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            assert "config.yaml" in names
            # The pre-1980 file should be skipped, not crash the backup
            assert "ancient.txt" not in names

    def test_skips_output_zip_inside_hermes(self, tmp_path, monkeypatch):
        """Backup skips its own output zip if it's inside hermes root."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Output inside hermes home
        out_zip = hermes_home / "backup.zip"
        args = Namespace(output=str(out_zip))

        from hermes_cli.backup import run_backup
        run_backup(args)

        # The zip should exist but not contain itself
        assert out_zip.exists()
        with zipfile.ZipFile(out_zip, "r") as zf:
            assert "backup.zip" not in zf.namelist()


class TestImportEdgeCases:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

    def test_not_a_zip(self, tmp_path, monkeypatch):
        """Import rejects a non-zip file."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        not_zip = tmp_path / "fake.zip"
        not_zip.write_text("this is not a zip")

        args = Namespace(zipfile=str(not_zip), force=True)

        from hermes_cli.backup import run_import
        with pytest.raises(SystemExit):
            run_import(args)

    def test_eof_during_confirmation(self, tmp_path, monkeypatch):
        """Import handles EOFError during confirmation prompt."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("existing\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {"config.yaml": "new\n"})

        args = Namespace(zipfile=str(zip_path), force=False)

        from hermes_cli.backup import run_import
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit):
                run_import(args)

    def test_keyboard_interrupt_during_confirmation(self, tmp_path, monkeypatch):
        """Import handles KeyboardInterrupt during confirmation prompt."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / ".env").write_text("KEY=val\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {"config.yaml": "new\n"})

        args = Namespace(zipfile=str(zip_path), force=False)

        from hermes_cli.backup import run_import
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit):
                run_import(args)

    def test_permission_error_during_import(self, tmp_path, monkeypatch):
        """Import handles permission errors during extraction."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create a read-only directory so extraction fails
        locked_dir = hermes_home / "locked"
        locked_dir.mkdir()
        locked_dir.chmod(0o555)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "locked/secret.txt": "data",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        try:
            run_import(args)
        finally:
            locked_dir.chmod(0o755)

        # config.yaml should still be restored despite the error
        assert (hermes_home / "config.yaml").exists()

    def test_progress_with_many_files(self, tmp_path, monkeypatch):
        """Import shows progress with 500+ files."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "big.zip"
        files = {"config.yaml": "model: test\n"}
        for i in range(600):
            files[f"sessions/s{i:04d}.json"] = "{}"

        self._make_backup_zip(zip_path, files)

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        assert (hermes_home / "config.yaml").exists()
        assert (hermes_home / "sessions" / "s0599.json").exists()


# ---------------------------------------------------------------------------
# Profile restoration tests
# ---------------------------------------------------------------------------

class TestProfileRestoration:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

    def test_import_creates_profile_wrappers(self, tmp_path, monkeypatch):
        """Import auto-creates wrapper scripts for restored profiles."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock the wrapper dir to be inside tmp_path
        wrapper_dir = tmp_path / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model:\n  provider: openrouter\n",
            "profiles/coder/config.yaml": "model:\n  provider: anthropic\n",
            "profiles/coder/.env": "ANTHROPIC_API_KEY=sk-test\n",
            "profiles/researcher/config.yaml": "model:\n  provider: deepseek\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        # Profile directories should exist
        assert (hermes_home / "profiles" / "coder" / "config.yaml").exists()
        assert (hermes_home / "profiles" / "researcher" / "config.yaml").exists()

        # Wrapper scripts should be created
        assert (wrapper_dir / "coder").exists()
        assert (wrapper_dir / "researcher").exists()

        # Wrappers should contain the right content
        coder_wrapper = (wrapper_dir / "coder").read_text()
        assert "hermes -p coder" in coder_wrapper

    def test_import_skips_profile_dirs_without_config(self, tmp_path, monkeypatch):
        """Import doesn't create wrappers for profile dirs without config."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        wrapper_dir = tmp_path / ".local" / "bin"
        wrapper_dir.mkdir(parents=True)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "profiles/valid/config.yaml": "model: test\n",
            "profiles/empty/readme.txt": "nothing here\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        from hermes_cli.backup import run_import
        run_import(args)

        # Only valid profile should get a wrapper
        assert (wrapper_dir / "valid").exists()
        assert not (wrapper_dir / "empty").exists()

    def test_import_without_profiles_module(self, tmp_path, monkeypatch):
        """Import gracefully handles missing profiles module (fresh install)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "profiles/coder/config.yaml": "model: test\n",
        })

        args = Namespace(zipfile=str(zip_path), force=True)

        # Simulate profiles module not being available
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *a, **kw):
            if name == "hermes_cli.profiles":
                raise ImportError("no profiles module")
            return original_import(name, *a, **kw)

        from hermes_cli.backup import run_import
        with patch("builtins.__import__", side_effect=fake_import):
            run_import(args)

        # Files should still be restored even if wrappers can't be created
        assert (hermes_home / "profiles" / "coder" / "config.yaml").exists()


# ---------------------------------------------------------------------------
# SQLite safe copy tests
# ---------------------------------------------------------------------------

class TestSafeCopyDb:
    def test_copies_valid_database(self, tmp_path):
        from hermes_cli.backup import _safe_copy_db
        src = tmp_path / "test.db"
        dst = tmp_path / "copy.db"

        conn = sqlite3.connect(str(src))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
        conn.commit()
        conn.close()

        ok, err = _safe_copy_db(src, dst)
        assert ok is True
        assert err is None

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT x FROM t").fetchall()
        conn.close()
        assert rows == [(42,)]

    def test_copies_wal_mode_database(self, tmp_path):
        from hermes_cli.backup import _safe_copy_db
        src = tmp_path / "wal.db"
        dst = tmp_path / "copy.db"

        conn = sqlite3.connect(str(src))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.execute("INSERT INTO t VALUES ('wal-test')")
        conn.commit()
        conn.close()

        ok, err = _safe_copy_db(src, dst)
        assert ok is True
        assert err is None

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT x FROM t").fetchall()
        conn.close()
        assert rows == [("wal-test",)]

    def test_unreadable_database_fails_closed_with_reason(self, tmp_path):
        """A present-but-unreadable DB (e.g. zeroed by storage failure,
        issue #68474) must fail closed AND surface the sqlite error so
        callers can record why the file was not captured."""
        from hermes_cli.backup import _safe_copy_db
        src = tmp_path / "zeroed.db"
        src.write_bytes(b"\x00" * 4096)  # valid size, no SQLite header
        dst = tmp_path / "copy.db"

        ok, err = _safe_copy_db(src, dst)
        assert ok is False
        assert err is not None and "not a database" in err
        assert not dst.exists()


# ---------------------------------------------------------------------------
# Quick state snapshot tests
# ---------------------------------------------------------------------------

class TestQuickSnapshot:
    @pytest.fixture
    def hermes_home(self, tmp_path):
        """Create a fake HERMES_HOME with critical state files."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text("model:\n  provider: openrouter\n")
        (home / ".env").write_text("OPENROUTER_API_KEY=test-key-123\n")
        (home / "auth.json").write_text('{"providers": {}}\n')
        (home / "channel_aliases.json").write_text(
            '{"whatsapp": {"120363408391911677@g.us": "general"}}\n'
        )
        (home / "cron").mkdir()
        (home / "cron" / "jobs.json").write_text('{"jobs": []}\n')

        # Real SQLite database
        db_path = home / "state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('s1', 'hello world')")
        conn.commit()
        conn.close()
        return home

    def test_creates_snapshot(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id
        assert snap_dir.is_dir()
        assert (snap_dir / "manifest.json").exists()

    def test_label_in_id(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(label="before-upgrade", hermes_home=hermes_home)
        assert "before-upgrade" in snap_id

    def test_state_db_safely_copied(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        db_copy = hermes_home / "state-snapshots" / snap_id / "state.db"
        assert db_copy.exists()

        conn = sqlite3.connect(str(db_copy))
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("s1", "hello world")

    def test_copies_nested_files(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        assert (hermes_home / "state-snapshots" / snap_id / "cron" / "jobs.json").exists()

    def test_copies_discord_recovery_ledger(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot

        gateway_dir = hermes_home / "gateway"
        gateway_dir.mkdir()
        ledger = gateway_dir / "discord_message_recovery.db"
        conn = sqlite3.connect(ledger)
        conn.execute("CREATE TABLE handled (message_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO handled VALUES ('123')")
        conn.commit()
        conn.close()

        snap_id = create_quick_snapshot(hermes_home=hermes_home)

        copied = hermes_home / "state-snapshots" / snap_id / "gateway" / ledger.name
        assert copied.exists()
        conn = sqlite3.connect(copied)
        assert conn.execute("SELECT message_id FROM handled").fetchall() == [("123",)]
        conn.close()

    def test_copies_channel_aliases(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        copied = hermes_home / "state-snapshots" / snap_id / "channel_aliases.json"
        assert copied.exists()
        assert "120363408391911677@g.us" in copied.read_text()

    def test_missing_files_skipped(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        with open(hermes_home / "state-snapshots" / snap_id / "manifest.json") as f:
            meta = json.load(f)
        # gateway_state.json etc. don't exist in fixture
        assert "gateway_state.json" not in meta["files"]

    def test_empty_home_returns_none(self, tmp_path):
        from hermes_cli.backup import create_quick_snapshot
        empty = tmp_path / "empty"
        empty.mkdir()
        assert create_quick_snapshot(hermes_home=empty) is None

    def test_max_file_size_skips_oversized_file(self, hermes_home, capsys):
        """Files above the cap are skipped with a warning; small files
        (the pairing/cron data the snapshot exists for) still land."""
        from hermes_cli.backup import create_quick_snapshot
        # state.db in the fixture is a few KB — cap below it
        cap = 1024
        snap_id = create_quick_snapshot(
            hermes_home=hermes_home, max_file_size=cap
        )
        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id
        assert not (snap_dir / "state.db").exists()
        # Small files still captured
        assert (snap_dir / "cron" / "jobs.json").exists()
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)
        assert "state.db" not in meta["files"]
        out = capsys.readouterr().out
        assert "skipping state.db" in out
        assert "exceeds" in out

    def test_oversized_protected_file_does_not_evict_prior_snapshot(
        self, hermes_home, capsys
    ):
        """A protected file skipped for exceeding the size cap makes the
        snapshot incomplete, so the keep=1 prune must NOT delete the previous
        complete snapshot. Otherwise a state.db that crosses the cap on a
        pre-update run would destroy the last good recovery source — the exact
        loss the #68474 fix exists to prevent (#68907 review)."""
        from hermes_cli.backup import create_quick_snapshot

        # A prior COMPLETE snapshot that must survive the incomplete run.
        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text('{"id": "20200101-000000", "files": {}}')

        # state.db in the fixture is a few KB — cap below it so it is skipped
        # for SIZE (not corruption), under the pre-update keep=1 policy.
        snap_id = create_quick_snapshot(
            hermes_home=hermes_home, max_file_size=1024, keep=1
        )
        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id

        # The prior complete snapshot is retained, not evicted.
        assert prior.is_dir(), "prior complete snapshot was pruned by an incomplete run"
        # state.db was skipped for size; small protected files still captured.
        assert not (snap_dir / "state.db").exists()
        assert (snap_dir / "cron" / "jobs.json").exists()
        # The manifest records the size skip as the reason it is incomplete.
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)
        assert "state.db" in meta["size_skipped"]
        assert "state.db" not in meta["files"]
        out = capsys.readouterr().out
        assert "keeping older snapshots" in out

    def test_oversized_file_inside_protected_dir_blocks_eviction(
        self, hermes_home
    ):
        """The residual-bypass guard must cover the rglob path too: an oversized
        file inside a protected DIRECTORY (not just a top-level protected file)
        is recorded in size_skipped and must not let the prune evict the prior
        complete snapshot (#68907 review)."""
        from hermes_cli.backup import create_quick_snapshot

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {"state.db": 100}}'
        )
        # Oversized file inside the protected `kanban/boards` directory, which the
        # snapshot walks via rglob (exercises the directory-contained call site).
        board = hermes_home / "kanban" / "boards" / "board1"
        board.mkdir(parents=True)
        (board / "kanban.db").write_bytes(b"x" * 4096)

        snap_id = create_quick_snapshot(
            hermes_home=hermes_home, max_file_size=1024, keep=1
        )
        snap_dir = hermes_home / "state-snapshots" / snap_id
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)
        # The directory-contained oversized file is recorded via the rglob path.
        assert "kanban/boards/board1/kanban.db" in meta["size_skipped"]
        # The prior complete snapshot survives the incomplete run.
        assert prior.is_dir()

    def test_failed_capture_never_prunes_any_snapshot(self, hermes_home):
        """A HARD capture failure blocks pruning entirely: an older snapshot may
        be the only copy of the file this run failed on, so nothing is evicted
        (the #68474 no-evict guarantee, locked in for the failed path)."""
        from hermes_cli.backup import create_quick_snapshot

        snaps = hermes_home / "state-snapshots"
        for name in ("20200101-000000", "20200102-000000"):
            d = snaps / name
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(f'{{"id": "{name}", "files": {{}}}}')

        # Zero out state.db so the copy fails (not a size skip).
        (hermes_home / "state.db").write_bytes(b"\x00" * 8192)

        snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)
        assert snap_id is not None
        # Nothing is pruned in the hard-failure case.
        assert (snaps / "20200101-000000").is_dir()
        assert (snaps / "20200102-000000").is_dir()
        assert (snaps / snap_id).is_dir()

    def test_max_file_size_none_copies_everything(self, hermes_home):
        """Default (no cap) preserves manual /snapshot behavior."""
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home, max_file_size=None)
        assert (hermes_home / "state-snapshots" / snap_id / "state.db").exists()

    def test_max_file_size_under_cap_copies(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(
            hermes_home=hermes_home, max_file_size=1 << 30
        )
        assert (hermes_home / "state-snapshots" / snap_id / "state.db").exists()

    def test_failed_db_capture_is_loud_and_recorded(self, hermes_home, capsys):
        """An existing state.db that cannot be captured (e.g. zeroed to null
        bytes, issue #68474) must not ride through silently: the failure is
        printed prominently and persisted in the manifest for forensics,
        while the small files the snapshot exists to protect still land."""
        from hermes_cli.backup import create_quick_snapshot
        (hermes_home / "state.db").write_bytes(b"\x00" * 8192)

        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id
        assert not (snap_dir / "state.db").exists()
        assert (snap_dir / "cron" / "jobs.json").exists()

        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)
        assert "state.db" not in meta["files"]
        assert "not a database" in meta["failed"]["state.db"]

        out = capsys.readouterr().out
        assert "could not capture state.db" in out
        assert "Snapshot INCOMPLETE" in out
        assert "NOT protected" in out

    def test_failed_plain_copy_is_recorded(self, hermes_home, capsys):
        """Non-DB copy failures (OSError from shutil.copy2) are recorded in
        the manifest too, not just logged."""
        from hermes_cli.backup import create_quick_snapshot
        real_copy2 = shutil.copy2

        def failing_copy2(src, dst, **kw):
            if str(src).endswith(".env"):
                raise OSError("disk full")
            return real_copy2(src, dst, **kw)

        with patch("hermes_cli.backup.shutil.copy2", side_effect=failing_copy2):
            snap_id = create_quick_snapshot(hermes_home=hermes_home)

        assert snap_id is not None
        with open(hermes_home / "state-snapshots" / snap_id / "manifest.json") as f:
            meta = json.load(f)
        assert ".env" not in meta["files"]
        assert "disk full" in meta["failed"][".env"]
        assert "Snapshot INCOMPLETE" in capsys.readouterr().out

    def test_incomplete_snapshot_never_prunes_older_snapshots(self, hermes_home):
        """An incomplete snapshot must not evict older (possibly complete)
        snapshots: with the pre-update keep=1 policy, pruning would delete
        the last snapshot still holding a good copy of the very file this
        run failed to capture (issue #68474)."""
        from hermes_cli.backup import create_quick_snapshot
        good_id = create_quick_snapshot(label="good", hermes_home=hermes_home, keep=1)
        assert good_id is not None

        (hermes_home / "state.db").write_bytes(b"\x00" * 8192)
        bad_id = create_quick_snapshot(label="bad", hermes_home=hermes_home, keep=1)
        assert bad_id is not None

        root = hermes_home / "state-snapshots"
        assert (root / good_id).is_dir(), "complete snapshot was evicted"
        assert (root / bad_id).is_dir()

    def test_complete_snapshot_still_prunes(self, hermes_home):
        """Prune behavior is unchanged when every capture succeeds."""
        from hermes_cli.backup import create_quick_snapshot
        first = create_quick_snapshot(label="a", hermes_home=hermes_home, keep=1)
        second = create_quick_snapshot(label="b", hermes_home=hermes_home, keep=1)
        root = hermes_home / "state-snapshots"
        assert not (root / first).exists()
        assert (root / second).is_dir()

    def test_clean_snapshot_has_no_failed_key(self, hermes_home, capsys):
        """The failed key and the INCOMPLETE warning appear only on failure."""
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        with open(hermes_home / "state-snapshots" / snap_id / "manifest.json") as f:
            meta = json.load(f)
        assert "failed" not in meta
        assert "Snapshot INCOMPLETE" not in capsys.readouterr().out

    def test_list_snapshots(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot, list_quick_snapshots
        id1 = create_quick_snapshot(label="first", hermes_home=hermes_home)
        id2 = create_quick_snapshot(label="second", hermes_home=hermes_home)

        snaps = list_quick_snapshots(hermes_home=hermes_home)
        assert len(snaps) == 2
        assert snaps[0]["id"] == id2  # most recent first
        assert snaps[1]["id"] == id1

    def test_list_limit(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot, list_quick_snapshots
        for i in range(5):
            create_quick_snapshot(label=f"s{i}", hermes_home=hermes_home)
        snaps = list_quick_snapshots(limit=3, hermes_home=hermes_home)
        assert len(snaps) == 3

    def test_restore_config(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)

        (hermes_home / "config.yaml").write_text("model:\n  provider: anthropic\n")
        assert "anthropic" in (hermes_home / "config.yaml").read_text()

        result = restore_quick_snapshot(snap_id, hermes_home=hermes_home)
        assert result is True
        assert "openrouter" in (hermes_home / "config.yaml").read_text()

    def test_restore_state_db(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)

        conn = sqlite3.connect(str(hermes_home / "state.db"))
        conn.execute("INSERT INTO sessions VALUES ('s2', 'new')")
        conn.commit()
        conn.close()

        restore_quick_snapshot(snap_id, hermes_home=hermes_home)

        conn = sqlite3.connect(str(hermes_home / "state.db"))
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_restore_nonexistent(self, hermes_home):
        from hermes_cli.backup import restore_quick_snapshot
        assert restore_quick_snapshot("nonexistent", hermes_home=hermes_home) is False

    def test_auto_prune(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot, list_quick_snapshots, _QUICK_DEFAULT_KEEP
        for i in range(_QUICK_DEFAULT_KEEP + 5):
            create_quick_snapshot(label=f"snap-{i:03d}", hermes_home=hermes_home)
        snaps = list_quick_snapshots(limit=100, hermes_home=hermes_home)
        assert len(snaps) <= _QUICK_DEFAULT_KEEP

    def test_manual_prune(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot, prune_quick_snapshots, list_quick_snapshots
        for i in range(10):
            create_quick_snapshot(label=f"s{i}", hermes_home=hermes_home)
        deleted = prune_quick_snapshots(keep=3, hermes_home=hermes_home)
        assert deleted == 7
        assert len(list_quick_snapshots(hermes_home=hermes_home)) == 3

    def test_snapshot_includes_pairing_directories(self, hermes_home):
        """Pairing JSONs live outside state.db — snapshot must capture them
        recursively (generic + per-platform) so approved-user lists survive
        disasters like #15733."""
        from hermes_cli.backup import create_quick_snapshot

        # Generic pairing store (new location)
        (hermes_home / "platforms" / "pairing").mkdir(parents=True)
        (hermes_home / "platforms" / "pairing" / "telegram-approved.json").write_text(
            '{"12345": {"user_name": "alice"}}'
        )
        (hermes_home / "platforms" / "pairing" / "discord-approved.json").write_text(
            '{"67890": {"user_name": "bob"}}'
        )
        # Legacy pairing store (old location)
        (hermes_home / "pairing").mkdir()
        (hermes_home / "pairing" / "matrix-approved.json").write_text(
            '{"@charlie:server": {"user_name": "charlie"}}'
        )
        # Feishu's separate JSON
        (hermes_home / "feishu_comment_pairing.json").write_text(
            '{"doc_abc": {"allow_from": ["user_xyz"]}}'
        )

        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        assert snap_id is not None

        snap_dir = hermes_home / "state-snapshots" / snap_id
        assert (snap_dir / "platforms" / "pairing" / "telegram-approved.json").exists()
        assert (snap_dir / "platforms" / "pairing" / "discord-approved.json").exists()
        assert (snap_dir / "pairing" / "matrix-approved.json").exists()
        assert (snap_dir / "feishu_comment_pairing.json").exists()

        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)
        files = meta["files"]
        assert "platforms/pairing/telegram-approved.json" in files
        assert "platforms/pairing/discord-approved.json" in files
        assert "pairing/matrix-approved.json" in files
        assert "feishu_comment_pairing.json" in files

    def test_restore_recovers_pairing_data(self, hermes_home):
        """After restore, deleted pairing files reappear with original content."""
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

        pairing_dir = hermes_home / "platforms" / "pairing"
        pairing_dir.mkdir(parents=True)
        approved = pairing_dir / "telegram-approved.json"
        approved.write_text('{"12345": {"user_name": "alice"}}')
        feishu = hermes_home / "feishu_comment_pairing.json"
        feishu.write_text('{"doc_abc": {"allow_from": ["user_xyz"]}}')

        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        assert snap_id is not None

        # Simulate the disaster — user loses both pairing files.
        approved.unlink()
        feishu.unlink()
        assert not approved.exists()
        assert not feishu.exists()

        assert restore_quick_snapshot(snap_id, hermes_home=hermes_home) is True
        assert approved.exists()
        assert '"alice"' in approved.read_text()
        assert feishu.exists()
        assert '"user_xyz"' in feishu.read_text()

    def test_empty_pairing_dir_does_not_fail(self, hermes_home):
        """An empty pairing directory should be silently skipped."""
        from hermes_cli.backup import create_quick_snapshot

        (hermes_home / "platforms" / "pairing").mkdir(parents=True)
        # Directory exists but contains no files.
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        # Other state still present → snapshot succeeds.
        assert snap_id is not None

    def test_directory_scandir_open_failure_blocks_prune_and_is_recorded(
        self, hermes_home, capsys
    ):
        """A protected directory that cannot be opened at all for listing
        (e.g. a mode-000 ``pairing/`` on POSIX — os.scandir() itself raises)
        must be treated like a hard capture failure: recorded in the
        manifest so it is visible why the snapshot is incomplete, and
        blocking the keep=1 prune so the prior complete snapshot survives.

        Python 3.13's ``Path.rglob()`` SILENTLY suppresses ``OSError`` raised
        while scanning a subdirectory it cannot list, so a snapshot walk built
        on rglob would just yield fewer manifest entries with neither
        ``failed`` nor ``size_skipped`` set — reintroducing #68474's
        recovery-loss via directory-backed state (#68907 review).

        The failure is driven through a monkeypatched ``os.scandir`` (rather
        than a real mode-000 directory) so the reproduction is deterministic
        on both POSIX and Windows — real permission bits don't work the same
        way on Windows CI.
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        # A prior COMPLETE snapshot that must survive the incomplete run.
        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        pairing_dir = hermes_home / "pairing"
        pairing_dir.mkdir()
        (pairing_dir / "users.json").write_text(
            '{"12345": {"user_name": "alice"}}'
        )

        real_scandir = backup_mod.os.scandir

        def fake_scandir(path):
            if Path(path) == pairing_dir:
                raise OSError(13, "Permission denied", str(pairing_dir))
            return real_scandir(path)

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id

        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        # The enumeration failure is recorded so the manifest shows WHY the
        # snapshot is incomplete (same forensic contract as `failed` for a
        # per-file capture error).
        assert "failed" in meta, "directory enumeration failure was not recorded"
        assert "pairing" in meta["failed"]
        assert "Permission denied" in meta["failed"]["pairing"]

        # The file inside the unreadable directory was never captured.
        assert "pairing/users.json" not in meta["files"]
        assert not (snap_dir / "pairing" / "users.json").exists()

        # The prior complete snapshot must survive: an enumeration failure
        # blocks the keep=1 prune exactly like a hard capture failure does.
        assert prior.is_dir(), (
            "prior complete snapshot was pruned despite a directory "
            "enumeration failure"
        )

        out = capsys.readouterr().out
        assert "Snapshot INCOMPLETE" in out

    def test_dir_entry_classification_failure_blocks_prune_and_is_recorded(
        self, hermes_home, capsys
    ):
        """A subdirectory whose type cannot be classified — DirEntry.is_dir()
        itself raises OSError, distinct from scandir() failing to open the
        parent — must also be recorded and block the keep=1 prune.

        Verified against CPython 3.13's ``Lib/os.py`` ``walk()``: when
        ``entry.is_dir()`` raises, ``os.walk`` catches the OSError
        internally and puts the entry in ``filenames`` WITHOUT calling
        ``onerror`` (this is why an os.walk(onerror=...)-based fix is not
        enough — the traversal must control classification itself via
        os.scandir, per Finding 1 of the #68907 review). A caller then sees
        the misclassified name in ``filenames`` and either silently drops
        it or crashes — either way nothing is recorded and #68474's
        recovery-loss reproduces via a subtly different enumeration op.

        Driven through a monkeypatched ``os.scandir`` that returns a
        wrapper whose ``is_dir()`` raises for one specific entry, so the
        reproduction is deterministic on both POSIX and Windows.
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        pairing_dir = hermes_home / "pairing"
        pairing_dir.mkdir()
        # A sibling FILE directly in pairing/ — the success path for a
        # normal entry must be unaffected by the misclassified sibling.
        (pairing_dir / "telegram-approved.json").write_text(
            '{"12345": {"user_name": "alice"}}'
        )
        # A subdirectory whose classification will be made to raise.
        private_dir = pairing_dir / "private"
        private_dir.mkdir()
        (private_dir / "users.json").write_text('{"67890": {"user_name": "bob"}}')

        class _RaisingIsDirEntry:
            """Wraps a real os.DirEntry, forcing is_dir() to raise —
            simulating the ESTALE/EIO/transient-error class of failure
            CPython's os.walk() swallows silently."""

            def __init__(self, real_entry):
                self._real = real_entry
                self.name = real_entry.name
                self.path = real_entry.path

            def is_dir(self, *args, **kwargs):
                raise OSError(5, "Simulated classification failure", self.path)

            def is_file(self, *args, **kwargs):
                return self._real.is_file(*args, **kwargs)

        real_scandir = backup_mod.os.scandir

        def fake_scandir(path):
            if Path(path) == pairing_dir:
                wrapped = []
                for entry in real_scandir(path):
                    if entry.name == "private":
                        wrapped.append(_RaisingIsDirEntry(entry))
                    else:
                        wrapped.append(entry)
                return _FakeScandirIterator(wrapped)
            return real_scandir(path)

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id

        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        # The classification failure is recorded ...
        assert "failed" in meta, "DirEntry classification failure was not recorded"
        assert "pairing/private" in meta["failed"]
        assert "Simulated classification failure" in meta["failed"]["pairing/private"]

        # ... the file inside the misclassified subdirectory was never
        # captured ...
        assert "pairing/private/users.json" not in meta["files"]
        assert not (snap_dir / "pairing" / "private" / "users.json").exists()

        # ... but the readable sibling file is captured normally (success
        # path unchanged).
        assert "pairing/telegram-approved.json" in meta["files"]
        assert (snap_dir / "pairing" / "telegram-approved.json").exists()

        # The prior complete snapshot must survive.
        assert prior.is_dir(), (
            "prior complete snapshot was pruned despite a DirEntry "
            "classification failure"
        )

        out = capsys.readouterr().out
        assert "Snapshot INCOMPLETE" in out

    def test_unreadable_excluded_subtree_does_not_block_prune(self, hermes_home):
        """An enumeration failure INSIDE an excluded subtree (workspaces/
        attachments under a kanban board) must NOT mark the snapshot
        incomplete: nothing under those subtrees is ever captured, so a
        failure there carries no recovery-loss risk and must not block
        pruning forever (#68907 review, Finding 2).

        The exclusion must be applied BEFORE descending into the
        subdirectory — asserted here by poisoning os.scandir for the
        excluded path: if the implementation ever descended into it, this
        test would surface a failure entry instead of a clean prune.
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        board = hermes_home / "kanban" / "boards" / "board1"
        board.mkdir(parents=True)
        conn = sqlite3.connect(str(board / "kanban.db"))
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        workspaces_dir = board / "workspaces"
        workspaces_dir.mkdir()
        (workspaces_dir / "scratch.txt").write_text("regenerable scratch data")

        real_scandir = backup_mod.os.scandir

        def fake_scandir(path):
            p = Path(path)
            if p == workspaces_dir or workspaces_dir in p.parents:
                raise OSError(13, "Permission denied", str(path))
            return real_scandir(path)

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id

        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        # The board db is still captured normally.
        assert "kanban/boards/board1/kanban.db" in meta["files"]
        # Nothing under the excluded, unreadable workspaces/ subtree was
        # ever touched, so the snapshot is complete.
        assert not meta.get("failed"), (
            f"unreadable excluded subtree wrongly marked incomplete: {meta.get('failed')}"
        )

        # A complete capture prunes normally — the excluded subtree's
        # unreadability must not block pruning forever.
        assert not prior.is_dir(), (
            "prior snapshot was retained even though the only failure was "
            "inside an excluded (never-descended) subtree"
        )

    def test_top_level_classification_failure_blocks_prune_and_is_recorded(
        self, hermes_home, capsys
    ):
        """A transient classification failure on a PRESENT top-level
        protected file (e.g. state.db) must be recorded and block the
        keep=1 prune — not silently treated as "doesn't exist".

        Path.exists()/is_dir()/is_file() swallow OSError for a specific
        errno set (pathlib._IGNORED_ERRNOS = ENOENT, ENOTDIR, EBADF,
        ELOOP) and return False. EBADF in particular is a real,
        documented transient failure mode (pathlib's own source notes it
        guards against a macOS stat() quirk) — using those methods for
        the top-level src classification means a present file can look
        identical to an absent one, silently dropping it from the
        manifest with nothing recorded (#68907 review, Finding 1).

        Driven through a monkeypatched os.stat (deterministic on any OS —
        the errno is synthesized, not produced by a real syscall).
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        target = hermes_home / "state.db"
        real_stat = backup_mod.os.stat

        def fake_stat(path, *args, **kwargs):
            if Path(path) == target:
                raise OSError(errno.EBADF, "Bad file descriptor", str(target))
            return real_stat(path, *args, **kwargs)

        with patch.object(backup_mod.os, "stat", side_effect=fake_stat):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id

        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        # The classification failure is recorded ...
        assert "failed" in meta, "top-level classification failure was not recorded"
        assert "state.db" in meta["failed"]
        assert "Bad file descriptor" in meta["failed"]["state.db"]

        # ... state.db was never captured ...
        assert "state.db" not in meta["files"]
        assert not (snap_dir / "state.db").exists()

        # ... but another protected file is still captured normally
        # (success path unaffected by the unrelated failure).
        assert "cron/jobs.json" in meta["files"]
        assert (snap_dir / "cron" / "jobs.json").exists()

        # The prior complete snapshot must survive.
        assert prior.is_dir(), (
            "prior complete snapshot was pruned despite a top-level "
            "classification failure"
        )

        out = capsys.readouterr().out
        assert "Snapshot INCOMPLETE" in out

    def test_multi_subdirectory_windows_stat_collision_does_not_drop_files(
        self, hermes_home
    ):
        """Two distinct sibling subdirectories under a protected root must
        BOTH be captured, even when DirEntry.stat() reports colliding
        identity for them.

        Verified by direct measurement on a real Windows machine: CPython
        3.13's DirEntry.stat() — the cached fast-path stat, as opposed to
        os.stat() — returns st_dev=0, st_ino=0 for every entry. A prior
        fix (commit dd860c07c) used a visited-set keyed on (st_dev,
        st_ino) from entry.stat() to guard against directory cycles. On
        Windows this silently collided ANY two sibling directories on
        (0, 0): the second one visited was treated as "already seen" and
        never scanned, dropping its files from the manifest with neither
        `failed` nor `size_skipped` set — a regression worse than the
        junction bug it targeted, breaking every multi-subdirectory
        protected root (pairing/, kanban/boards/) on Windows. The fix
        removes identity checking entirely in favor of a depth bound
        (#68907 review, pass 4) — this test's fake scandir has zero
        effect on that new code path.

        The (0, 0) collision is reproduced deterministically on any host
        OS (not just Windows) by wrapping every scanned entry so its
        stat() zeroes out st_dev/st_ino, exactly matching the real
        Windows behavior regardless of which OS runs this test.
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        pairing_dir = hermes_home / "pairing"
        pairing_dir.mkdir()
        (pairing_dir / "a").mkdir()
        (pairing_dir / "a" / "one.json").write_text('{"one": true}')
        (pairing_dir / "b").mkdir()
        (pairing_dir / "b" / "two.json").write_text('{"two": true}')

        class _ZeroIdentityStat:
            """Delegates to a real stat_result but zeroes st_dev/st_ino —
            replicating DirEntry.stat()'s measured Windows behavior."""

            def __init__(self, real_result):
                self._real = real_result
                self.st_dev = 0
                self.st_ino = 0

            def __getattr__(self, name):
                return getattr(self._real, name)

        class _ZeroIdentityEntry:
            def __init__(self, real_entry):
                self._real = real_entry
                self.name = real_entry.name
                self.path = real_entry.path

            def is_dir(self, *args, **kwargs):
                return self._real.is_dir(*args, **kwargs)

            def is_file(self, *args, **kwargs):
                return self._real.is_file(*args, **kwargs)

            def stat(self, *args, **kwargs):
                return _ZeroIdentityStat(self._real.stat(*args, **kwargs))

        real_scandir = backup_mod.os.scandir

        def fake_scandir(path):
            return _FakeScandirIterator(
                [_ZeroIdentityEntry(e) for e in real_scandir(path)]
            )

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home)

        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        assert "pairing/a/one.json" in meta["files"], (
            "first sibling directory's file missing from manifest"
        )
        assert "pairing/b/two.json" in meta["files"], (
            "second sibling directory dropped — (st_dev, st_ino) identity "
            "collision silently skipped it"
        )
        assert not meta.get("failed"), (
            f"a working (non-colliding) traversal should not need to "
            f"record any failure here: {meta.get('failed')}"
        )
        assert (snap_dir / "pairing" / "a" / "one.json").exists()
        assert (snap_dir / "pairing" / "b" / "two.json").exists()

    def test_linear_cycle_terminates_via_traversal_budget_and_blocks_prune(
        self, hermes_home, monkeypatch
    ):
        """A directory structure that recurses unboundedly in a straight
        line (e.g. a Windows junction looping back to an ancestor) must
        be caught by the traversal work budget: the traversal
        terminates, the overrun is recorded as a failure (visible,
        forensic), and the keep=1 prune is blocked — never silent data
        loss, never an unbounded hang (#68907 review, pass 5).

        A prior version of this guard used a max recursion DEPTH (64)
        instead of a total-work budget. Depth alone only bounds path
        LENGTH: it happens to catch a purely linear chain like this one
        fine, but a BRANCHING cycle (see
        test_binary_cycle_terminates_via_traversal_budget_and_blocks_prune
        below) blows up exponentially long before any fixed depth is
        reached. The work budget subsumes the linear case too, so this
        test now exercises the budget instead of a separate depth check.

        The budget (_QUICK_SNAPSHOT_MAX_TRAVERSAL_ENTRIES) is patched
        down to a small value so the test runs fast without needing
        200k synthetic iterations; production still uses 200_000.
        Simulated by an os.scandir fake that always yields exactly one
        (synthetic) child directory, however deep the traversal goes —
        behaviorally identical to a self-referencing junction. The test
        carries its own independent safety ceiling, comfortably above
        the patched budget: if the production guard somehow failed to
        stop it, the test's sentinel (non-OSError) exception fires
        instead of letting the test hang.
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        # raising=False: on a pre-pass-5 commit this attribute doesn't
        # exist yet, and the patch should be a harmless no-op there (that
        # commit's own — different — bound mechanism is exercised as-is).
        monkeypatch.setattr(
            backup_mod, "_QUICK_SNAPSHOT_MAX_TRAVERSAL_ENTRIES", 50, raising=False
        )

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        pairing_dir = hermes_home / "pairing"
        pairing_dir.mkdir()

        class _LoopEntry:
            """A synthetic subdirectory whose own listing always yields
            another copy of itself."""

            name = "loop"

            def __init__(self, path):
                self.path = str(path)

            def is_dir(self, *args, **kwargs):
                return True

            def is_file(self, *args, **kwargs):
                return False

        class _RunawayRecursion(Exception):
            """Raised only if the traversal exceeds the test's own
            ceiling — proves the traversal budget failed to terminate
            the recursion, without ever letting the test actually
            hang."""

        call_count = {"n": 0}
        CALL_CEILING = 500  # far above the patched budget (50)

        real_scandir = backup_mod.os.scandir
        created_iterators = []

        def fake_scandir(path):
            call_count["n"] += 1
            if call_count["n"] > CALL_CEILING:
                raise _RunawayRecursion(
                    f"os.scandir called {call_count['n']} times — the "
                    "traversal work budget did not stop recursion"
                )
            p = Path(path)
            if p == pairing_dir or p.name == "loop":
                it = _FakeScandirIterator([_LoopEntry(p / "loop")])
                created_iterators.append(it)
                return it
            return real_scandir(path)

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        assert call_count["n"] <= CALL_CEILING, (
            "traversal did not terminate within the safety ceiling"
        )

        # `with scandir_it:` must close the handle for every directory
        # scanned, including the last one — the one whose iteration is
        # what actually trips the budget and breaks out early (#68907
        # review pass 7 nit: this was previously true by inspection only).
        assert created_iterators, "fake scandir was never exercised"
        assert all(it.closed for it in created_iterators), (
            "not every scandir iterator was closed — `with scandir_it:` "
            "did not release the handle on early budget break"
        )

        snap_dir = hermes_home / "state-snapshots" / snap_id
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        assert "failed" in meta, "traversal budget was exceeded but not recorded"
        assert any("cycle" in reason for reason in meta["failed"].values()), (
            meta["failed"]
        )

        assert prior.is_dir(), (
            "prior complete snapshot was pruned despite an unterminated "
            "traversal budget"
        )

    def test_binary_cycle_terminates_via_traversal_budget_and_blocks_prune(
        self, hermes_home, monkeypatch
    ):
        """A BRANCHING directory cycle — e.g. two Windows junctions
        pairing/a -> pairing and pairing/b -> pairing (junctions bypass
        islink()) — must also be caught, and caught FAST.

        This is the case a depth-only bound misses: reaching a fixed
        depth of 64 down EVERY branch of a binary cycle requires
        2**65-1 ~= 3.7e19 scandir calls, so the snapshot would hang or
        exhaust resources long before any failure is recorded. A
        total-work budget (counting entries visited, not path length)
        catches this almost immediately, because branching makes the
        entry count explode exponentially per level.

        The budget is patched down to a small value for a fast,
        deterministic test; production still uses 200_000. The test
        carries its own independent safety ceiling — generously above
        what the patched budget needs, but tiny compared to what a
        depth-only guard would need for this shape — so a version with
        no total-work budget (only depth) fails this test LOUDLY AND
        FAST (RED against 959130e40) instead of hanging, and the
        work-budgeted version passes almost immediately (GREEN).
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        # raising=False: on a pre-pass-5 commit (959130e40, depth-only)
        # this attribute doesn't exist — the patch is then a harmless
        # no-op and that commit's own depth-64 bound runs unmodified,
        # which is exactly what should fail this test's ceiling.
        monkeypatch.setattr(
            backup_mod, "_QUICK_SNAPSHOT_MAX_TRAVERSAL_ENTRIES", 50, raising=False
        )

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        pairing_dir = hermes_home / "pairing"
        pairing_dir.mkdir()

        class _BranchEntry:
            """A synthetic subdirectory whose own listing always yields
            TWO more copies of the same shape — an infinite binary tree,
            modeling two junctions that each loop back into the cycle."""

            def __init__(self, path, name):
                self.path = str(path)
                self.name = name

            def is_dir(self, *args, **kwargs):
                return True

            def is_file(self, *args, **kwargs):
                return False

        class _RunawayRecursion(Exception):
            """Raised only if the traversal exceeds the test's own
            ceiling — proves the traversal budget failed to bound the
            branching cycle, without ever letting the test actually
            hang."""

        call_count = {"n": 0}
        # Generous vs. what the patched budget (50) needs (~25-30 calls,
        # since each call yields 2 entries) but minuscule compared to
        # what an unbounded-branching depth-64 traversal would need.
        CALL_CEILING = 2000

        real_scandir = backup_mod.os.scandir

        def fake_scandir(path):
            call_count["n"] += 1
            if call_count["n"] > CALL_CEILING:
                raise _RunawayRecursion(
                    f"os.scandir called {call_count['n']} times — the "
                    "traversal budget did not bound the branching cycle"
                )
            p = Path(path)
            if p == pairing_dir or p.name in ("a", "b"):
                return _FakeScandirIterator([
                    _BranchEntry(p / "a", "a"),
                    _BranchEntry(p / "b", "b"),
                ])
            return real_scandir(path)

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        assert call_count["n"] <= CALL_CEILING, (
            "branching traversal did not terminate within the safety ceiling"
        )

        snap_dir = hermes_home / "state-snapshots" / snap_id
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        assert "failed" in meta, "traversal budget was exceeded but not recorded"
        # Exactly one failure: the whole protected-root traversal must
        # abort immediately on budget overrun, not just skip the
        # offending branch and keep burning budget on the others.
        assert len(meta["failed"]) == 1, (
            f"expected exactly one recorded failure (traversal aborts "
            f"immediately), got: {meta['failed']}"
        )
        assert any("cycle" in reason for reason in meta["failed"].values()), (
            meta["failed"]
        )

        assert prior.is_dir(), (
            "prior complete snapshot was pruned despite an unbounded "
            "branching cycle"
        )

    def test_lazy_high_fan_out_scandir_is_not_eagerly_drained(
        self, hermes_home, monkeypatch
    ):
        """A SINGLE os.scandir() call whose listing itself yields many
        entries — a junction whose target directory has high fan-out —
        must have the traversal budget checked PER YIELD, not after the
        whole listing has been pulled.

        A prior version of this guard did
        ``entries = list(os.scandir(current))``: that fully DRAINS the
        directory listing before entries_visited is ever incremented.
        Neither the binary-cycle test above nor the linear-cycle test can
        catch this — both simulate the cycle across MANY separate
        os.scandir() calls (one entry per call), so even an eager list()
        of a 1-element listing is trivially cheap either way. This test
        isolates the eager-vs-lazy question specifically: ONE scandir()
        call, many entries (#68907 review, pass 6).

        Reproduced with a Python generator (the same lazy, pull-based
        shape a real ScandirIterator has under the hood) that yields
        entries one at a time and raises a distinct sentinel exception if
        ever pulled past 100 yields. The production budget is patched
        down to 50. A correct (lazy) implementation stops pulling at
        ~entries_visited + 1 (to detect the overrun) — the generator must
        NEVER be asked for its 60th+ item, let alone its 100th. An eager
        implementation (list(os.scandir(...))) drains the whole
        generator up front to build the list, tripping the sentinel with
        entries_visited still at 0 — the unambiguous RED signal.
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        monkeypatch.setattr(
            backup_mod, "_QUICK_SNAPSHOT_MAX_TRAVERSAL_ENTRIES", 50, raising=False
        )

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        pairing_dir = hermes_home / "pairing"
        pairing_dir.mkdir()

        class _LazyOverdrawn(Exception):
            """Raised only if the generator below is pulled past
            SENTINEL_AFTER — proves entries were drained without the
            budget ever stopping the pull (eager materialization), not a
            real resource limit or an infinite hang."""

        class _ChildEntry:
            """A synthetic subdirectory entry — never actually descended
            into either way (a correct implementation aborts the whole
            traversal before popping any of these off the stack; an
            eager implementation crashes on _LazyOverdrawn before ever
            reaching the stack at all)."""

            def __init__(self, path, name):
                self.path = str(path)
                self.name = name

            def is_dir(self, *args, **kwargs):
                return True

            def is_file(self, *args, **kwargs):
                return False

        yielded = {"n": 0}
        SENTINEL_AFTER = 100  # far above the patched budget (50)

        def lazy_children():
            """Yields _ChildEntry objects ONE AT A TIME from a single
            (simulated) directory listing — exactly how a real
            ScandirIterator behaves: it does not pre-build a list
            internally either."""
            i = 0
            while True:
                yielded["n"] += 1
                if yielded["n"] > SENTINEL_AFTER:
                    raise _LazyOverdrawn(
                        f"generator was pulled {yielded['n']} times — the "
                        "traversal budget did not stop the pull; entries "
                        "were drained eagerly instead of lazily"
                    )
                yield _ChildEntry(pairing_dir / f"child-{i}", f"child-{i}")
                i += 1

        real_scandir = backup_mod.os.scandir
        created_iterator = {}

        def fake_scandir(path):
            if Path(path) == pairing_dir:
                it = _FakeScandirIterator(lazy_children())
                created_iterator["it"] = it
                return it
            return real_scandir(path)

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        assert yielded["n"] <= SENTINEL_AFTER, (
            "the generator's own sentinel had to intervene — the "
            "production code drained more entries than the budget "
            "should ever have allowed"
        )
        # The real assertion: pulls stop close to the patched budget (50),
        # nowhere near the generator's full 100-yield sentinel. A margin
        # (not exact equality) avoids coupling the test to the "+1 to
        # detect overrun" implementation detail.
        assert yielded["n"] <= 60, (
            f"generator was pulled {yielded['n']} times for a budget of "
            f"50 — entries were not stopped promptly (looks eager, not "
            f"lazy)"
        )

        # `with scandir_it:` must release the handle even though the
        # budget cut iteration short partway through the listing (#68907
        # review pass 7 nit: previously true by inspection only).
        assert "it" in created_iterator, "fake scandir was never exercised"
        assert created_iterator["it"].closed, (
            "the scandir iterator was not closed — `with scandir_it:` did "
            "not release the handle on early budget break"
        )

        snap_dir = hermes_home / "state-snapshots" / snap_id
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        assert "failed" in meta, "traversal budget was exceeded but not recorded"
        assert any("cycle" in reason for reason in meta["failed"].values()), (
            meta["failed"]
        )

        assert prior.is_dir(), (
            "prior complete snapshot was pruned despite an unbounded "
            "high-fan-out listing"
        )

    def test_mid_iteration_scandir_error_blocks_prune_and_is_recorded(
        self, hermes_home, capsys
    ):
        """An OSError raised WHILE ITERATING a directory listing — not
        just at os.scandir()'s open call — must be recorded and block
        the keep=1 prune, exactly like an open-time failure does.
        scandir can fail mid-iteration on some filesystems (transient
        ESTALE/EIO partway through a listing), and the production code
        already routes this through the same next()/except OSError path
        CPython's own os.walk() uses (#68907 review pass 7) — this test
        is the previously-missing regression guard for that path, not a
        behavior change.

        A generator yields one REAL DirEntry (proving a file enumerated
        before the failure is still captured — partial success, not
        total loss) and then raises OSError instead of returning
        normally — a mid-listing failure the try/except around
        os.scandir(current) itself (open time) cannot catch, since
        scandir() already succeeded there.
        """
        from hermes_cli import backup as backup_mod
        from hermes_cli.backup import create_quick_snapshot

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text(
            '{"id": "20200101-000000", "files": {}}'
        )

        pairing_dir = hermes_home / "pairing"
        pairing_dir.mkdir()
        (pairing_dir / "telegram-approved.json").write_text(
            '{"12345": {"user_name": "alice"}}'
        )

        real_scandir = backup_mod.os.scandir

        def mid_iteration_entries():
            # os.scandir(pairing_dir) itself succeeds — the failure
            # happens partway through consuming the results, which is
            # exactly what a try/except wrapped only around the open
            # call would miss.
            yield from real_scandir(pairing_dir)
            raise OSError(5, "Input/output error", str(pairing_dir))

        def fake_scandir(path):
            if Path(path) == pairing_dir:
                return _FakeScandirIterator(mid_iteration_entries())
            return real_scandir(path)

        with patch.object(backup_mod.os, "scandir", side_effect=fake_scandir):
            snap_id = create_quick_snapshot(hermes_home=hermes_home, keep=1)

        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id
        with open(snap_dir / "manifest.json") as f:
            meta = json.load(f)

        # The file enumerated BEFORE the mid-iteration error is still
        # captured — a partial listing failure doesn't undo already
        # completed work.
        assert "pairing/telegram-approved.json" in meta["files"]
        assert (snap_dir / "pairing" / "telegram-approved.json").exists()

        # The mid-iteration failure itself is recorded ...
        assert "failed" in meta, "mid-iteration scandir error was not recorded"
        assert "pairing" in meta["failed"]
        assert "Input/output error" in meta["failed"]["pairing"]

        # ... and blocks the keep=1 prune exactly like an open-time
        # failure or a per-entry classification failure would.
        assert prior.is_dir(), (
            "prior complete snapshot was pruned despite a mid-iteration "
            "scandir error"
        )

        out = capsys.readouterr().out
        assert "Snapshot INCOMPLETE" in out

# ---------------------------------------------------------------------------
# Pre-update backup (hermes update safety net)
# ---------------------------------------------------------------------------

    # -- security: path traversal regression coverage -----------------------
    # Per @egilewski audit on PR #9217: restore_quick_snapshot must reject
    # malicious snapshot_id values (the directory selector) AND malicious
    # rel paths inside the manifest (the per-file selector). Both surfaces
    # need explicit regression tests because they validate independent
    # traversal vectors.

    def test_restore_rejects_snapshot_id_traversal(self, hermes_home):
        """restore_quick_snapshot must reject snapshot_id values that
        contain path separators, POSIX traversal entries, or are empty.
        These are rejected on the input string before any filesystem
        lookup, so the guard cannot be bypassed by arranging a directory
        layout that would otherwise satisfy ``snap_dir.is_dir()``.

        Regression for the path-traversal surface where ``root /
        snapshot_id`` could resolve above the snapshots root."""
        from hermes_cli.backup import restore_quick_snapshot

        hostile_ids = [
            "../../etc",                # parent traversal
            "../outside",               # single parent
            "..",                       # bare parent dir
            ".",                        # bare current dir
            "subdir/snap",              # forward slash
            "subdir\\snap",           # backslash (Windows-style)
            "",                         # empty string
        ]
        for hostile in hostile_ids:
            assert restore_quick_snapshot(
                hostile, hermes_home=hermes_home
            ) is False, f"hostile snapshot_id was not rejected: {hostile!r}"

    def test_restore_rejects_manifest_rel_traversal(self, hermes_home):
        """A snapshot whose manifest.json contains a rel path that escapes
        the snapshot directory (e.g. ``../../outside.txt``) must skip that
        entry rather than restoring outside HERMES_HOME."""
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        assert snap_id is not None
        snap_dir = hermes_home / "state-snapshots" / snap_id

        # Inject a traversal entry into manifest.json AND seed the source
        # file outside the snapshot directory so a vulnerable implementation
        # would actually write something at the escaped destination.
        manifest_path = snap_dir / "manifest.json"
        with open(manifest_path) as f:
            meta = json.load(f)
        meta["files"]["../../outside.txt"] = 9
        with open(manifest_path, "w") as f:
            json.dump(meta, f)

        # Source: ../../outside.txt resolves above the snapshot root.
        # Place a payload there so we can detect a successful escape.
        escape_src = snap_dir.parent.parent / "outside.txt"
        escape_src.write_text("pwned-source")

        # Pre-condition: the destination must not exist before restore.
        escape_dst = hermes_home.parent.parent / "outside.txt"
        assert not escape_dst.exists()

        # Restore should succeed for legitimate files but skip the hostile
        # entry. We don't assert on the return value (other legitimate
        # entries may still restore); we assert on the file-system effect.
        restore_quick_snapshot(snap_id, hermes_home=hermes_home)

        assert not escape_dst.exists(), (
            f"manifest rel traversal escaped HERMES_HOME: {escape_dst} exists"
        )

        # Cleanup the seeded escape source so the test is hermetic.
        escape_src.unlink()


class TestQuickSnapshotProjectsKanban:
    """Regression for #52889: projects.db / kanban.db must survive an upgrade.

    Both are per-profile user-created stores outside the git checkout. If they
    are not in the pre-update snapshot, the post-update ``CREATE TABLE IF NOT
    EXISTS`` runs against a missing file and every project / board row is lost.
    """

    @pytest.fixture
    def hermes_home(self, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        # Minimal critical file so the snapshot is non-empty.
        (home / "config.yaml").write_text("model:\n  provider: openrouter\n")

        for name, table, row in (
            ("projects.db", "projects", ("p1", "demo")),
            ("kanban.db", "tasks", ("t1", "todo")),
        ):
            conn = sqlite3.connect(str(home / name))
            conn.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, data TEXT)")
            conn.execute(f"INSERT INTO {table} VALUES (?, ?)", row)
            conn.commit()
            conn.close()
        return home

    def test_in_quick_state_files(self):
        from hermes_cli.backup import _QUICK_STATE_FILES
        # All per-profile user-created stores that the upgrade can wipe.
        for name in (
            "projects.db", "kanban.db", "kanban/boards",
            "response_store.db", "memory_store.db", "verification_evidence.db",
        ):
            assert name in _QUICK_STATE_FILES, name

    def test_projects_db_snapshotted(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        copy = hermes_home / "state-snapshots" / snap_id / "projects.db"
        assert copy.exists()
        conn = sqlite3.connect(str(copy))
        rows = conn.execute("SELECT * FROM projects").fetchall()
        conn.close()
        assert rows == [("p1", "demo")]

    def test_kanban_db_snapshotted(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        copy = hermes_home / "state-snapshots" / snap_id / "kanban.db"
        assert copy.exists()
        conn = sqlite3.connect(str(copy))
        rows = conn.execute("SELECT * FROM tasks").fetchall()
        conn.close()
        assert rows == [("t1", "todo")]

    def test_restore_recreates_emptied_projects_db(self, hermes_home):
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)

        # Simulate the upgrade wiping the store back to an empty schema.
        conn = sqlite3.connect(str(hermes_home / "projects.db"))
        conn.execute("DELETE FROM projects")
        conn.commit()
        conn.close()

        assert restore_quick_snapshot(snap_id, hermes_home=hermes_home) is True
        conn = sqlite3.connect(str(hermes_home / "projects.db"))
        rows = conn.execute("SELECT * FROM projects").fetchall()
        conn.close()
        assert rows == [("p1", "demo")]

    def test_non_default_kanban_board_snapshotted(self, hermes_home):
        """#52889 completeness: non-default boards live at
        <root>/kanban/boards/<slug>/kanban.db, not <root>/kanban.db. The
        ``kanban/boards`` dir entry must capture them too, or multi-board
        users still lose every board except ``default`` on upgrade."""
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

        board_dir = hermes_home / "kanban" / "boards" / "work"
        board_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(board_dir / "kanban.db"))
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO tasks VALUES (?, ?)", ("w1", "ship"))
        conn.commit()
        conn.close()

        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        copy = (
            hermes_home / "state-snapshots" / snap_id
            / "kanban" / "boards" / "work" / "kanban.db"
        )
        assert copy.exists(), "non-default board kanban.db was not snapshotted"

        # Simulate the upgrade wiping the board, then restore it.
        conn = sqlite3.connect(str(board_dir / "kanban.db"))
        conn.execute("DELETE FROM tasks")
        conn.commit()
        conn.close()

        assert restore_quick_snapshot(snap_id, hermes_home=hermes_home) is True
        conn = sqlite3.connect(str(board_dir / "kanban.db"))
        rows = conn.execute("SELECT * FROM tasks").fetchall()
        conn.close()
        assert rows == [("w1", "ship")]

    def test_additional_per_profile_dbs_round_trip(self, hermes_home):
        """#52889 completeness: response_store.db (conversation history),
        memory_store.db (holographic memory) and verification_evidence.db are
        the same upgrade-wiped data-loss class as projects.db and must also be
        snapshotted + restored."""
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

        seeded = {
            "response_store.db": ("responses", ("r1", "hello")),
            "memory_store.db": ("facts", ("f1", "the sky is blue")),
            "verification_evidence.db": ("verification_events", ("v1", "passed")),
        }
        for name, (table, row) in seeded.items():
            conn = sqlite3.connect(str(hermes_home / name))
            conn.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, data TEXT)")
            conn.execute(f"INSERT INTO {table} VALUES (?, ?)", row)
            conn.commit()
            conn.close()

        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        # Wipe every store (the upgrade failure), then restore.
        for name, (table, _row) in seeded.items():
            conn = sqlite3.connect(str(hermes_home / name))
            conn.execute(f"DELETE FROM {table}")
            conn.commit()
            conn.close()

        assert restore_quick_snapshot(snap_id, hermes_home=hermes_home) is True
        for name, (table, row) in seeded.items():
            conn = sqlite3.connect(str(hermes_home / name))
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            conn.close()
            assert rows == [row], name

    def test_board_workspaces_and_attachments_are_skipped(self, hermes_home):
        """#52889 W3: the kanban/boards walk must capture board DBs + metadata
        but SKIP the heavy regenerable workspaces/ and attachments/ subtrees so
        snapshots don't bloat (×20 retained)."""
        from hermes_cli.backup import create_quick_snapshot

        board = hermes_home / "kanban" / "boards" / "work"
        (board / "workspaces" / "scratch").mkdir(parents=True)
        (board / "attachments" / "t1").mkdir(parents=True)
        conn = sqlite3.connect(str(board / "kanban.db"))
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, data TEXT)")
        conn.commit()
        conn.close()
        (board / "board.json").write_text('{"name": "work"}')
        (board / "workspaces" / "scratch" / "big.bin").write_bytes(b"x" * 4096)
        (board / "attachments" / "t1" / "file.bin").write_bytes(b"y" * 4096)

        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        snap = hermes_home / "state-snapshots" / snap_id / "kanban" / "boards" / "work"
        # Board db + metadata captured...
        assert (snap / "kanban.db").exists()
        assert (snap / "board.json").exists()
        # ...but the heavy subtrees skipped.
        assert not (snap / "workspaces" / "scratch" / "big.bin").exists()
        assert not (snap / "attachments" / "t1" / "file.bin").exists()

    def test_board_db_copied_wal_safely(self, hermes_home, monkeypatch):
        """#52889 W2: a non-default board's .db (dir-branch) must go through the
        WAL-safe _safe_copy_db, not a raw shutil.copy2, so an open WAL doesn't
        produce an inconsistent copy."""
        import hermes_cli.backup as bk
        from hermes_cli.backup import create_quick_snapshot

        board = hermes_home / "kanban" / "boards" / "work"
        board.mkdir(parents=True)
        conn = sqlite3.connect(str(board / "kanban.db"))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('w1', 'ship')")
        conn.commit()
        conn.close()

        called = {"db": []}
        real = bk._safe_copy_db

        def _spy(src, dst):
            called["db"].append(str(src))
            return real(src, dst)

        monkeypatch.setattr(bk, "_safe_copy_db", _spy)
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        # The board db was copied via _safe_copy_db (not raw copy).
        assert any(s.endswith("boards/work/kanban.db") for s in called["db"]), called["db"]
        copy = hermes_home / "state-snapshots" / snap_id / "kanban" / "boards" / "work" / "kanban.db"
        rows = sqlite3.connect(str(copy)).execute("SELECT * FROM tasks").fetchall()
        assert rows == [("w1", "ship")]


class TestPreUpdateBackup:
    """Tests for create_pre_update_backup — the auto-backup ``hermes update``
    runs before touching anything."""

    def test_failed_sqlite_snapshot_removes_incomplete_archive(self, tmp_path, monkeypatch):
        """The non-interactive full-zip helper must fail the entire archive
        rather than return success after omitting a live WAL database."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model: test\n")
        db_path = hermes_home / "state.db"

        writer = sqlite3.connect(db_path)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE events (value TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO events VALUES ('only-in-wal')")
        writer.commit()
        assert Path(f"{db_path}-wal").stat().st_size > 0

        import hermes_cli.backup as backup_mod
        real_connect = backup_mod.sqlite3.connect

        class FailingBackupConnection:
            def __init__(self, connection):
                self._connection = connection

            def backup(self, _destination):
                raise sqlite3.OperationalError("forced backup failure")

            def close(self):
                self._connection.close()

        def connect_with_failed_backup(database, *args, **kwargs):
            connection = real_connect(database, *args, **kwargs)
            if str(database).startswith(f"file:{db_path}"):
                return FailingBackupConnection(connection)
            return connection

        monkeypatch.setattr(backup_mod.sqlite3, "connect", connect_with_failed_backup)
        out_zip = tmp_path / "pre-update.zip"
        try:
            result = backup_mod._write_full_zip_backup(out_zip, hermes_home)
        finally:
            writer.close()

        assert result is None
        assert not out_zip.exists()

    @pytest.fixture
    def hermes_home(self, tmp_path):
        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        return root

    def test_creates_backup_under_backups_dir(self, hermes_home):
        from hermes_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(hermes_home=hermes_home)
        assert out is not None
        assert out.exists()
        assert out.parent == hermes_home / "backups"
        assert out.name.startswith("pre-update-")
        assert out.suffix == ".zip"

    def test_backup_contents_match_full_backup(self, hermes_home):
        """Pre-update backup should include the same user data that
        ``hermes backup`` would, and should exclude the same directories."""
        from hermes_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(hermes_home=hermes_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        # User data present
        assert "config.yaml" in names
        assert ".env" in names
        assert "sessions/abc123.json" in names
        assert "skills/my-skill/SKILL.md" in names
        assert "profiles/coder/config.yaml" in names
        # hermes-agent repo excluded
        assert not any(n.startswith("hermes-agent/") for n in names)
        # __pycache__ excluded
        assert not any("__pycache__" in n for n in names)
        # pid files excluded
        assert "gateway.pid" not in names

    def test_does_not_recurse_into_prior_backups(self, hermes_home):
        """The ``backups/`` directory must be excluded so that each backup
        doesn't grow exponentially by including all prior backups."""
        from hermes_cli.backup import create_pre_update_backup
        # First backup
        out1 = create_pre_update_backup(hermes_home=hermes_home)
        assert out1 is not None
        # Second backup — must not include the first
        out2 = create_pre_update_backup(hermes_home=hermes_home)
        assert out2 is not None
        with zipfile.ZipFile(out2) as zf:
            names = zf.namelist()
        assert not any(n.startswith("backups/") for n in names), (
            f"Pre-update backup recursed into backups/ — leaked: "
            f"{[n for n in names if n.startswith('backups/')]}"
        )

    def test_rotation_keeps_only_n(self, hermes_home):
        """After more than ``keep`` backups are created, older ones are
        pruned automatically."""
        import time as _t
        from hermes_cli.backup import create_pre_update_backup

        created = []
        for _ in range(5):
            out = create_pre_update_backup(hermes_home=hermes_home, keep=3)
            created.append(out)
            _t.sleep(1.05)  # ensure distinct seconds in timestamp

        remaining = sorted(
            p.name for p in (hermes_home / "backups").iterdir()
            if p.name.startswith("pre-update-")
        )
        assert len(remaining) == 3
        # Oldest two should have been pruned
        assert created[0].name not in remaining
        assert created[1].name not in remaining
        # Newest three should remain
        assert created[4].name in remaining

    def test_rotation_preserves_manual_files(self, hermes_home):
        """Hand-dropped zips in ``backups/`` must not be touched by
        rotation — it only prunes files matching ``pre-update-*.zip``."""
        import time as _t
        from hermes_cli.backup import create_pre_update_backup

        (hermes_home / "backups").mkdir(exist_ok=True)
        manual = hermes_home / "backups" / "my-manual.zip"
        manual.write_bytes(b"manual backup")

        for _ in range(5):
            create_pre_update_backup(hermes_home=hermes_home, keep=2)
            _t.sleep(1.05)

        assert manual.exists(), "Manual backup zip was incorrectly pruned"

    def test_returns_none_if_root_missing(self, tmp_path):
        from hermes_cli.backup import create_pre_update_backup
        assert create_pre_update_backup(hermes_home=tmp_path / "does-not-exist") is None

    def test_keep_zero_does_not_delete_freshly_created_backup(self, hermes_home):
        """Regression: ``backup_keep: 0`` previously triggered ``backups[0:]``
        in the pruner — wiping the just-created zip and leaving the user
        with no recovery point.  The floor (keep>=1) preserves the new file
        regardless of misconfiguration; users who don't want backups should
        set ``pre_update_backup: false`` instead.
        """
        from hermes_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(hermes_home=hermes_home, keep=0)
        assert out is not None
        assert out.exists(), (
            "keep=0 silently deleted the freshly-created backup; floor "
            "should preserve the just-written file."
        )

    def test_keep_negative_does_not_delete_freshly_created_backup(self, hermes_home):
        """Mirror coverage: any value <1 should be floored, not literally
        applied as a slice index."""
        from hermes_cli.backup import create_pre_update_backup
        out = create_pre_update_backup(hermes_home=hermes_home, keep=-3)
        assert out is not None
        assert out.exists()

    def test_keep_zero_still_prunes_older_backups(self, hermes_home):
        """The floor preserves the new backup but should NOT regress the
        rotation behaviour for older zips: a third call with keep=0 must
        still remove pre-existing backups beyond the (floored) limit of 1.
        """
        import time as _t
        from hermes_cli.backup import create_pre_update_backup

        first = create_pre_update_backup(hermes_home=hermes_home, keep=5)
        _t.sleep(1.05)
        second = create_pre_update_backup(hermes_home=hermes_home, keep=5)
        _t.sleep(1.05)
        third = create_pre_update_backup(hermes_home=hermes_home, keep=0)

        remaining = {
            p.name for p in (hermes_home / "backups").iterdir()
            if p.name.startswith("pre-update-")
        }
        assert third.name in remaining, "Floor must preserve the new backup"
        assert first.name not in remaining and second.name not in remaining, (
            f"keep=0 floor of 1 should still prune older backups; "
            f"remaining={remaining}"
        )

    def test_skips_symlinked_files(self, hermes_home, tmp_path):
        """Pre-update backups must not dereference symlinks outside HERMES_HOME."""
        from hermes_cli.backup import create_pre_update_backup

        outside = tmp_path / "outside-secret.txt"
        outside.write_text("outside secret\n")
        _symlink_file_or_skip(hermes_home / "skills" / "outside-link.txt", outside)

        out = create_pre_update_backup(hermes_home=hermes_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "skills/outside-link.txt" not in names
            assert all(zf.read(name) != b"outside secret\n" for name in names)


class TestRunPreUpdateBackup:
    """Tests for the ``_run_pre_update_backup`` wrapper in main.py —
    covers the consolidated off/quick/full mode gate, CLI flags, and
    user-facing output."""

    @pytest.fixture
    def hermes_home(self, tmp_path, monkeypatch):
        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        # Point HERMES_HOME at the temp dir so config + backup paths resolve here
        monkeypatch.setenv("HERMES_HOME", str(root))
        # Make Path.home() point at tmp_path for anything that uses it
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Bust caches for hermes_cli.config + hermes_constants so they pick up HERMES_HOME
        for mod in list(__import__("sys").modules.keys()):
            if mod.startswith("hermes_cli.config") or mod == "hermes_constants":
                del __import__("sys").modules[mod]
        return root

    @staticmethod
    def _set_mode(hermes_home, value):
        import yaml
        (hermes_home / "config.yaml").write_text(yaml.safe_dump({
            "_config_version": 22,
            "updates": {"pre_update_backup": value},
        }))
        import sys as _sys
        for mod in list(_sys.modules.keys()):
            if mod.startswith("hermes_cli.config"):
                del _sys.modules[mod]

    @staticmethod
    def _zips(hermes_home):
        d = hermes_home / "backups"
        return list(d.glob("pre-update-*.zip")) if d.exists() else []

    @staticmethod
    def _snaps(hermes_home):
        d = hermes_home / "state-snapshots"
        return [p for p in d.iterdir() if p.is_dir()] if d.exists() else []

    def test_default_creates_quick_snapshot_only(self, hermes_home, capsys):
        """With no config, the default mode is ``quick``: a state snapshot is
        created but NOT the full zip."""
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
        assert "Creating pre-update backup" not in out
        assert self._snaps(hermes_home)
        assert not self._zips(hermes_home)

    def test_snapshot_creation_failure_is_surfaced_loudly(self, hermes_home, capsys):
        """A pre-update snapshot that never gets created -- e.g. snap_dir.mkdir
        failing on a full or read-only filesystem, which raises before any
        per-file reporting can run -- must be surfaced loudly through the caller,
        not swallowed at debug level, so the user knows the update is proceeding
        without a recovery point (#68907 review)."""
        from hermes_cli.main import _run_pre_update_backup
        with patch(
            "hermes_cli.backup.create_quick_snapshot",
            side_effect=OSError("[Errno 30] Read-only file system"),
        ):
            snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        # The failure does not block the update...
        assert snap_id is None
        # ...but it is loud, not silent.
        out = capsys.readouterr().out
        assert "Pre-update snapshot FAILED" in out
        assert "WITHOUT a recovery snapshot" in out
        assert "Read-only file system" in out

    def test_snapshot_returning_none_is_surfaced_loudly(self, hermes_home, capsys):
        """create_quick_snapshot() can also return None with nothing captured --
        another silent no-recovery-point path the caller must surface, not only
        the raise path. The trust-boundary wrapper reports a missing snapshot
        regardless of HOW the helper failed (#68907 review, Sol)."""
        from hermes_cli.main import _run_pre_update_backup
        with patch("hermes_cli.backup.create_quick_snapshot", return_value=None):
            snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        assert snap_id is None
        out = capsys.readouterr().out
        assert "Pre-update snapshot FAILED" in out
        assert "WITHOUT a recovery snapshot" in out

    def test_backup_flag_forces_full(self, hermes_home, capsys):
        """--backup forces the full zip (plus quick snapshot) for one run."""
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=True))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
        assert "Creating pre-update backup" in out
        assert "Saved:" in out
        assert "hermes import" in out
        assert len(self._zips(hermes_home)) == 1

    def test_no_backup_flag_skips_everything(self, hermes_home, capsys):
        """--no-backup skips BOTH the quick snapshot and the zip."""
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=True, backup=False))
        out = capsys.readouterr().out
        assert snap_id is None
        assert "skipped (--no-backup)" in out
        assert "Pre-update snapshot" not in out
        assert not self._snaps(hermes_home)
        assert not self._zips(hermes_home)

    def test_config_off_disables_everything_silently(self, hermes_home, capsys):
        """pre_update_backup: off — an explicit opt-out disables the quick
        snapshot too (it previously ran unconditionally), with no output."""
        self._set_mode(hermes_home, "off")
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is None
        assert out == ""
        assert not self._snaps(hermes_home)
        assert not self._zips(hermes_home)

    def test_legacy_false_maps_to_off(self, hermes_home, capsys):
        """Legacy boolean ``false`` (the old zip opt-out) now means off."""
        self._set_mode(hermes_home, False)
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        assert snap_id is None
        assert capsys.readouterr().out == ""
        assert not self._snaps(hermes_home)
        assert not self._zips(hermes_home)

    def test_legacy_true_maps_to_full(self, hermes_home, capsys):
        """Legacy boolean ``true`` (the old always-zip opt-in) means full."""
        self._set_mode(hermes_home, True)
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Creating pre-update backup" in out
        assert "Saved:" in out
        assert len(self._zips(hermes_home)) == 1

    def test_config_full_mode(self, hermes_home, capsys):
        self._set_mode(hermes_home, "full")
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
        assert "Creating pre-update backup" in out
        assert len(self._zips(hermes_home)) == 1

    def test_config_quick_mode(self, hermes_home, capsys):
        self._set_mode(hermes_home, "quick")
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
        assert "Creating pre-update backup" not in out
        assert not self._zips(hermes_home)

    def test_unknown_mode_falls_back_to_quick(self, hermes_home, capsys):
        self._set_mode(hermes_home, "bogus-mode")
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
        assert not self._zips(hermes_home)

    def test_no_backup_flag_overrides_full_config(self, hermes_home, capsys):
        """--no-backup wins even when config says full."""
        self._set_mode(hermes_home, "full")
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=True, backup=False))
        out = capsys.readouterr().out
        assert snap_id is None
        assert "skipped (--no-backup)" in out
        assert not self._snaps(hermes_home)
        assert not self._zips(hermes_home)

    def test_backup_flag_overrides_off_config(self, hermes_home, capsys):
        """--backup wins over config off for a single run."""
        self._set_mode(hermes_home, "off")
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=True))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Creating pre-update backup" in out
        assert len(self._zips(hermes_home)) == 1


# ---------------------------------------------------------------------------
# Pre-migration backup (hermes claw migrate safety net)
# ---------------------------------------------------------------------------

class TestPreMigrationBackup:
    """Tests for create_pre_migration_backup — the auto-backup
    ``hermes claw migrate`` runs before mutating ~/.hermes/."""

    @pytest.fixture
    def hermes_home(self, tmp_path):
        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        return root

    def test_creates_backup_under_backups_dir(self, hermes_home):
        from hermes_cli.backup import create_pre_migration_backup
        out = create_pre_migration_backup(hermes_home=hermes_home)
        assert out is not None
        assert out.exists()
        # Shares the backups/ directory with pre-update backups so `hermes
        # import` and the update-backup listing both pick them up.
        assert out.parent == hermes_home / "backups"
        assert out.name.startswith("pre-migration-")
        assert out.suffix == ".zip"

    def test_backup_uses_shared_exclusion_rules(self, hermes_home):
        """Pre-migration backup reuses the same exclusion rules as
        ``hermes backup`` / ``create_pre_update_backup`` — no drift."""
        from hermes_cli.backup import create_pre_migration_backup
        out = create_pre_migration_backup(hermes_home=hermes_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        # User data present
        assert "config.yaml" in names
        assert ".env" in names
        assert "skills/my-skill/SKILL.md" in names
        # Same exclusions as the shared helper
        assert not any(n.startswith("hermes-agent/") for n in names)
        assert not any("__pycache__" in n for n in names)
        assert "gateway.pid" not in names

    def test_restorable_with_hermes_import(self, hermes_home, tmp_path):
        """The zip produced by pre-migration backup must be a valid Hermes
        backup — `hermes import` should accept it."""
        from hermes_cli.backup import create_pre_migration_backup, _validate_backup_zip
        out = create_pre_migration_backup(hermes_home=hermes_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            valid, _reason = _validate_backup_zip(zf)
        assert valid, "pre-migration zip failed _validate_backup_zip"

    def test_does_not_recurse_into_prior_backups(self, hermes_home):
        from hermes_cli.backup import create_pre_migration_backup
        out1 = create_pre_migration_backup(hermes_home=hermes_home)
        assert out1 is not None
        out2 = create_pre_migration_backup(hermes_home=hermes_home)
        assert out2 is not None
        with zipfile.ZipFile(out2) as zf:
            names = zf.namelist()
        assert not any(n.startswith("backups/") for n in names)

    def test_rotation_keeps_only_n(self, hermes_home):
        import time as _t
        from hermes_cli.backup import create_pre_migration_backup

        created = []
        for _ in range(7):
            out = create_pre_migration_backup(hermes_home=hermes_home, keep=3)
            if out is not None:
                created.append(out)
            _t.sleep(1.05)  # timestamp resolution

        remaining = sorted((hermes_home / "backups").glob("pre-migration-*.zip"))
        assert len(remaining) <= 3, f"expected <=3 backups retained, got {len(remaining)}"

    def test_missing_hermes_home_returns_none(self, tmp_path):
        """Fresh install with no ~/.hermes yet — nothing to back up."""
        from hermes_cli.backup import create_pre_migration_backup
        missing = tmp_path / "does-not-exist"
        out = create_pre_migration_backup(hermes_home=missing)
        assert out is None

    def test_does_not_touch_pre_update_backups(self, hermes_home):
        """Pre-migration rotation must only prune pre-migration-*.zip files,
        leaving pre-update-*.zip backups untouched."""
        from hermes_cli.backup import create_pre_update_backup, create_pre_migration_backup
        update_backup = create_pre_update_backup(hermes_home=hermes_home, keep=5)
        assert update_backup is not None and update_backup.exists()
        # Spin up a lot of migration backups with keep=1
        import time as _t
        for _ in range(3):
            out = create_pre_migration_backup(hermes_home=hermes_home, keep=1)
            assert out is not None
            _t.sleep(1.05)
        # Update backup must still be there
        assert update_backup.exists(), "pre-migration rotation wrongly pruned the pre-update backup"


# ---------------------------------------------------------------------------
# Cron jobs auto-restore after silent migration loss (issue #34600)
# ---------------------------------------------------------------------------

class TestRestoreCronJobsIfEmptied:
    """`hermes update` config migration can leave cron/jobs.json valid-but-empty,
    silently dropping every scheduled job. `restore_cron_jobs_if_emptied` is the
    post-migration safety net that restores from the pre-update snapshot."""

    @staticmethod
    def _seed_jobs(path: Path, jobs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"jobs": jobs}))

    def _make_snapshot(self, hermes_home: Path, label="pre-update"):
        from hermes_cli.backup import create_quick_snapshot
        return create_quick_snapshot(label=label, hermes_home=hermes_home, keep=5)

    def test_restores_when_emptied_after_migration(self, tmp_path):
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        # Pre-update: 3 real jobs.
        self._seed_jobs(jobs_path, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
        snap_id = self._make_snapshot(hermes_home)
        assert snap_id

        # Migration silently empties the file (valid JSON, zero jobs).
        jobs_path.write_text(json.dumps({"jobs": []}))

        result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
        assert result is not None
        assert result["restored"] is True
        assert result["job_count"] == 3
        assert result["snapshot_id"] == snap_id

        # The live file now has the jobs back.
        restored = json.loads(jobs_path.read_text())
        assert len(restored["jobs"]) == 3

    def test_noop_when_live_file_still_has_jobs(self, tmp_path):
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        self._seed_jobs(jobs_path, [{"id": "a"}, {"id": "b"}])
        snap_id = self._make_snapshot(hermes_home)

        # Healthy path: file unchanged after update.
        result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
        assert result is None

    def test_restores_when_partial_job_loss(self, tmp_path):
        """Desktop scheduler overwrites jobs.json with its own small set,
        losing tool-created crons while keeping desktop-tracked ones."""
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        # Pre-update: 19 jobs (18 tool-created + 1 desktop watchdog).
        self._seed_jobs(
            jobs_path,
            [{"id": f"job-{i}"} for i in range(19)],
        )
        snap_id = self._make_snapshot(hermes_home)
        assert snap_id

        # Desktop scheduler overwrites with only its own 1 job.
        jobs_path.write_text(json.dumps({"jobs": [{"id": "desktop-watchdog"}]}))

        result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
        assert result is not None
        assert result["restored"] is True
        assert result["job_count"] == 19

        # The live file now has all 19 jobs back.
        restored = json.loads(jobs_path.read_text())
        assert len(restored["jobs"]) == 19

    def test_noop_when_snapshot_had_no_jobs(self, tmp_path):
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        # Pre-update genuinely had zero jobs; current is also empty.
        self._seed_jobs(jobs_path, [])
        snap_id = self._make_snapshot(hermes_home)
        jobs_path.write_text(json.dumps({"jobs": []}))

        result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
        assert result is None

    def test_bom_live_file_still_counted(self, tmp_path):
        """A UTF-8 BOM on the live jobs.json (Windows editors) must not make
        _count_cron_jobs report None — that would silently disable the
        auto-restore safety net. utf-8-sig matches cron/jobs.load_jobs."""
        from hermes_cli.backup import _count_cron_jobs, restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        self._seed_jobs(jobs_path, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
        snap_id = self._make_snapshot(hermes_home)
        assert snap_id

        # Migration empties the file AND a Windows editor leaves a BOM.
        jobs_path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"jobs": []}).encode())
        assert _count_cron_jobs(jobs_path) == 0  # not None

        result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
        assert result is not None
        assert result["restored"] is True
        assert result["job_count"] == 3

    def test_noop_when_live_file_unreadable(self, tmp_path):
        """An unparseable live file is left alone — that's a different failure
        mode the user should see, not silently overwrite."""
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        self._seed_jobs(jobs_path, [{"id": "a"}])
        snap_id = self._make_snapshot(hermes_home)
        jobs_path.write_text("{ this is not valid json")

        result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
        assert result is None
        # File left untouched.
        assert jobs_path.read_text() == "{ this is not valid json"

    def test_noop_when_snapshot_id_missing(self, tmp_path):
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        self._seed_jobs(jobs_path, [])
        assert restore_cron_jobs_if_emptied(None, hermes_home=hermes_home) is None
        assert restore_cron_jobs_if_emptied("", hermes_home=hermes_home) is None

    def test_restores_legacy_bare_list_snapshot_shape(self, tmp_path):
        """A legacy snapshot storing a bare JSON list (not {"jobs": [...]}) is
        still counted and restored."""
        from hermes_cli.backup import restore_cron_jobs_if_emptied
        hermes_home = tmp_path / ".hermes"
        jobs_path = hermes_home / "cron" / "jobs.json"
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
        snap_id = self._make_snapshot(hermes_home)

        jobs_path.write_text(json.dumps({"jobs": []}))
        result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
        assert result is not None
        assert result["job_count"] == 2


# ---------------------------------------------------------------------------
# Memory-provider external paths (~/.honcho, ~/.hindsight, ...) — captured via
# MemoryProvider.backup_paths() and restored to their original home-relative
# location, NOT under HERMES_HOME. (backup/import cycle data-loss fix)
# ---------------------------------------------------------------------------

class TestMemoryProviderExternalPaths:
    def _make_min_tree(self, hermes_home: Path) -> None:
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "config.yaml").write_text("model:\n  provider: openrouter\n")
        (hermes_home / ".env").write_text("OPENROUTER_API_KEY=sk-test\n")
        (hermes_home / "state.db").write_bytes(b"x")

    def test_backup_captures_external_paths_under_external_prefix(self, tmp_path, monkeypatch):
        """Provider state under ~/.honcho is archived beneath _external/,
        encoded relative to the home directory."""
        hermes_home = tmp_path / ".hermes"
        self._make_min_tree(hermes_home)
        # External provider state living OUTSIDE HERMES_HOME.
        honcho = tmp_path / ".honcho"
        honcho.mkdir()
        (honcho / "config.json").write_text('{"peer":"alice"}')
        (honcho / "sub").mkdir()
        (honcho / "sub" / "x.json").write_text('{"a":1}')

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        import hermes_cli.backup as backup_mod
        monkeypatch.setattr(
            backup_mod, "_collect_memory_provider_external_paths", lambda: [honcho]
        )

        out_zip = tmp_path / "backup.zip"
        backup_mod.run_backup(Namespace(output=str(out_zip)))

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())
        assert "_external/.honcho/config.json" in names
        assert "_external/.honcho/sub/x.json" in names
        # In-home files still present.
        assert "config.yaml" in names

    def test_backup_skips_external_paths_outside_home(self, tmp_path, monkeypatch):
        """A declared path outside the home dir is not portable and must be
        skipped, never archived."""
        hermes_home = tmp_path / ".hermes"
        self._make_min_tree(hermes_home)
        outside = tmp_path.parent / "outside-home-secret"
        outside.mkdir(exist_ok=True)
        (outside / "leak.json").write_text('{"secret":1}')

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        import hermes_cli.backup as backup_mod
        monkeypatch.setattr(
            backup_mod, "_collect_memory_provider_external_paths", lambda: [outside]
        )

        out_zip = tmp_path / "backup.zip"
        backup_mod.run_backup(Namespace(output=str(out_zip)))

        with zipfile.ZipFile(out_zip) as zf:
            names = set(zf.namelist())
        assert not any(n.startswith("_external/") for n in names)
        assert not any("leak.json" in n for n in names)
        (outside / "leak.json").unlink()
        outside.rmdir()

    def test_import_restores_external_to_home_relative_location(self, tmp_path, monkeypatch):
        """_external/ members restore to ~/<relpath>, not under HERMES_HOME,
        and credential-shaped files get 0600."""
        dst_home = tmp_path / "dst"
        dst_home.mkdir()
        hermes_home = dst_home / ".hermes"
        hermes_home.mkdir()

        zip_path = tmp_path / "backup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("config.yaml", "model: {}\n")
            zf.writestr(".env", "X=1\n")
            zf.writestr("state.db", "")
            zf.writestr("_external/.honcho/config.json", '{"peer":"bob"}')

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: dst_home)

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        restored = dst_home / ".honcho" / "config.json"
        assert restored.exists()
        assert restored.read_text() == '{"peer":"bob"}'
        # Credential-shaped file tightened.
        assert (restored.stat().st_mode & 0o777) == 0o600
        # External state did NOT leak into HERMES_HOME.
        assert not (hermes_home / "_external").exists()

    def test_import_blocks_external_path_traversal(self, tmp_path, monkeypatch):
        """A malicious _external/ member that escapes the home dir is blocked."""
        dst_home = tmp_path / "dst"
        dst_home.mkdir()
        hermes_home = dst_home / ".hermes"
        hermes_home.mkdir()
        sentinel = tmp_path / "PWNED"

        zip_path = tmp_path / "backup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("config.yaml", "model: {}\n")
            zf.writestr(".env", "X=1\n")
            zf.writestr("state.db", "")
            zf.writestr("_external/../../PWNED", "pwned")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: dst_home)

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert not sentinel.exists()

    def test_abc_backup_paths_defaults_empty(self):
        """The ABC default returns [] so providers opt in explicitly."""
        from agent.memory_provider import MemoryProvider

        class _Dummy(MemoryProvider):
            @property
            def name(self):
                return "dummy"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return []

        assert _Dummy().backup_paths() == []

    def test_honcho_provider_declares_global_config_dir(self, tmp_path, monkeypatch):
        """The honcho provider's backup_paths() resolves to ~/.honcho."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        from plugins.memory.honcho import HonchoMemoryProvider

        paths = HonchoMemoryProvider().backup_paths()
        assert str(tmp_path / ".honcho") in paths

    def test_hindsight_provider_declares_legacy_dir(self, tmp_path, monkeypatch):
        """The hindsight provider's backup_paths() resolves to ~/.hindsight."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        from plugins.memory.hindsight import HindsightMemoryProvider

        paths = HindsightMemoryProvider().backup_paths()
        assert str(tmp_path / ".hindsight") in paths
