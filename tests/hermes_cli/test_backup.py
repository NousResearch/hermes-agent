"""Tests for hermes backup and import commands."""

import json
import os
import sqlite3
import stat
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    """run_import() auto-installs the gateway service post-restore; tests must
    never touch the host's systemd/launchd. Individual tests re-patch these to
    assert the wiring."""
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def _advance_backup_clock(seconds: float = 1.1) -> None:
    """Skew hermes_cli.backup's datetime forward instead of sleeping.

    Snapshot ids have 1-second resolution; tests that need two distinct
    timestamps previously slept >1s. This installs (once) a datetime shim in
    the backup module whose now() adds a cumulative offset, then bumps it.
    """
    import datetime as _dt

    import hermes_cli.backup as _backup

    shim = getattr(_backup.datetime, "_hermes_test_shim", None)
    if shim is None:
        class _ShimDatetime(_dt.datetime):
            _hermes_test_shim = True
            _offset = _dt.timedelta(0)

            @classmethod
            def now(cls, tz=None):  # noqa: D102
                return _dt.datetime.now(tz) + cls._offset

        _backup.datetime = _ShimDatetime
        shim = _ShimDatetime
    else:
        shim = _backup.datetime
    shim._offset += _dt.timedelta(seconds=seconds)


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


# ---------------------------------------------------------------------------
# Backup tests
# ---------------------------------------------------------------------------

class TestBackup:


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

    def test_import_auto_installs_gateway_service(self, tmp_path, monkeypatch):
        """After a restore, run_import brings the gateway service up without
        prompting — restored cron jobs and bot tokens must not sit dormant
        (the install-then-import dead-gateway bug)."""
        import hermes_cli.gateway as gateway_mod

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        calls = []
        monkeypatch.setattr(
            gateway_mod, "ensure_gateway_service",
            lambda **kw: calls.append(kw) or True,
        )
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {"config.yaml": "model: test\n"})

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert calls and calls[0].get("context") == "import"

    def test_import_skips_service_when_already_running(self, tmp_path, monkeypatch):
        """A live gateway is left alone — no reinstall churn during import."""
        import hermes_cli.gateway as gateway_mod

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        calls = []
        monkeypatch.setattr(
            gateway_mod, "ensure_gateway_service",
            lambda **kw: calls.append(kw) or True,
        )
        monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: True)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {"config.yaml": "model: test\n"})

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert not calls

    def test_import_survives_service_layer_import_failure(self, tmp_path, monkeypatch, capsys):
        """If the service helpers can't even be reached, import still completes
        and prints the manual fallback."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        import hermes_cli.gateway as gateway_mod

        def boom():
            raise RuntimeError("service layer unavailable")

        monkeypatch.setattr(gateway_mod, "_is_service_running", boom)

        zip_path = tmp_path / "backup.zip"
        self._make_backup_zip(zip_path, {"config.yaml": "model: test\n"})

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        out = capsys.readouterr().out
        assert "Done. Your Hermes configuration has been restored." in out
        assert "hermes gateway install" in out







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



class TestImportEdgeCases:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)


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


class _ExplodingMember:
    """Zip member whose stream dies mid-restore (ENOSPC / corrupt member).

    Both the pre-fix ``dst.write(src.read())`` and the atomic
    ``shutil.copyfileobj`` path pull bytes through ``read()``, so injecting
    here exercises whichever implementation is in the tree.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, *args):
        raise OSError(28, "No space left on device")

    def close(self):
        pass


def _break_member(monkeypatch, failing_member: str) -> None:
    """Make ``ZipFile.open`` hand back a dying stream for one member only."""
    real_open = zipfile.ZipFile.open

    def _patched(self, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if filename == failing_member:
            return _ExplodingMember()
        return real_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", _patched)


class TestImportAtomicWrites:
    """`hermes import` must never leave a user's file truncated.

    The pre-fix code did ``open(target, "wb")`` then ``dst.write(src.read())``,
    which zeroes the existing file *before* any replacement bytes exist. These
    tests pin the invariant for both restore branches: the HERMES_HOME branch
    and the ``_external/`` branch that writes into third-party configs under
    the user's home.
    """

    def _zip(self, zip_path: Path, files: dict) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

    def test_failed_member_leaves_existing_file_intact(self, tmp_path, monkeypatch):
        """A dying member must not destroy the file it was replacing."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        original = "model: original\napi_key: keep-me\n"
        (hermes_home / "config.yaml").write_text(original)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._zip(zip_path, {"config.yaml": "model: replacement\n", "state.db": ""})
        _break_member(monkeypatch, "config.yaml")

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        # Pre-fix this file is 0 bytes: the truncate landed, the write did not.
        assert (hermes_home / "config.yaml").read_text() == original
        # And the aborted write must not litter the directory it staged in.
        assert list(hermes_home.glob(".config.yaml.*")) == []

    def test_failed_external_member_leaves_existing_file_intact(self, tmp_path, monkeypatch):
        """Same invariant on the `_external/` branch, which writes outside HERMES_HOME."""
        dst_home = tmp_path / "dst"
        dst_home.mkdir()
        hermes_home = dst_home / ".hermes"
        hermes_home.mkdir()
        honcho = dst_home / ".honcho"
        honcho.mkdir()
        original = '{"peer":"original"}'
        (honcho / "config.json").write_text(original)

        zip_path = tmp_path / "backup.zip"
        self._zip(zip_path, {
            "config.yaml": "model: {}\n",
            "_external/.honcho/config.json": '{"peer":"replacement"}',
        })

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: dst_home)
        _break_member(monkeypatch, "_external/.honcho/config.json")

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert (honcho / "config.json").read_text() == original
        assert list(honcho.glob(".config.json.*")) == []

    @pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks")
    def test_symlinked_target_keeps_its_symlink(self, tmp_path, monkeypatch):
        """A symlinked target is written through, not replaced by a regular file.

        Guards the atomic rewrite against a naive ``os.replace``, which would
        detach dotfiles-managed deployments (GitHub #16743). ``atomic_replace``
        resolves the link first.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        store = hermes_home / "store"
        store.mkdir()
        real = store / "config.yaml"
        real.write_text("model: original\n")
        link = hermes_home / "config.yaml"
        link.symlink_to(real)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._zip(zip_path, {"config.yaml": "model: restored\n", "state.db": ""})

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert link.is_symlink(), "import replaced the symlink with a regular file"
        assert real.read_text() == "model: restored\n"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks")
    def test_symlinked_external_target_keeps_its_symlink(self, tmp_path, monkeypatch):
        """Same guard on the `_external/` branch — the realistic dotfiles case."""
        dst_home = tmp_path / "dst"
        dst_home.mkdir()
        hermes_home = dst_home / ".hermes"
        hermes_home.mkdir()
        dotfiles = dst_home / "dotfiles"
        dotfiles.mkdir()
        real = dotfiles / "honcho.json"
        real.write_text('{"peer":"original"}')
        honcho = dst_home / ".honcho"
        honcho.mkdir()
        link = honcho / "config.json"
        link.symlink_to(real)

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: dst_home)

        zip_path = tmp_path / "backup.zip"
        self._zip(zip_path, {
            "config.yaml": "model: {}\n",
            "_external/.honcho/config.json": '{"peer":"restored"}',
        })

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert link.is_symlink(), "import replaced the symlink with a regular file"
        assert real.read_text() == '{"peer":"restored"}'

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
    def test_restore_preserves_existing_file_mode(self, tmp_path, monkeypatch):
        """Staging through mkstemp must not silently tighten restored files to 0600.

        ``tempfile.mkstemp`` creates at 0600; the mode of the file being
        replaced has to survive the publish, or Docker/NAS installs that rely
        on broader permissions break on restore.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        target = hermes_home / "config.yaml"
        target.write_text("model: original\n")
        os.chmod(target, 0o644)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._zip(zip_path, {"config.yaml": "model: restored\n", "state.db": ""})

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert target.read_text() == "model: restored\n"
        assert (target.stat().st_mode & 0o777) == 0o644

    @pytest.mark.skipif(os.name != "posix", reason="POSIX ownership")
    def test_restore_preserves_existing_file_owner(self, tmp_path, monkeypatch):
        """A root-run import must not re-own the user's files to root.

        ``os.replace`` swaps in a temp file owned by the *writing* user, so a
        ``sudo hermes import`` onto a user-owned (or Docker/NAS volume-owned)
        HERMES_HOME would hand every restored file to root. The uid/gid is
        forced so the assertion does not require running as root.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        target = hermes_home / "config.yaml"
        target.write_text("model: original\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._zip(zip_path, {"config.yaml": "model: restored\n", "state.db": ""})

        chown_calls: list[tuple[Path, int, int]] = []
        monkeypatch.setattr(
            "hermes_cli.backup._preserve_file_owner",
            lambda p: (123, 456) if Path(p).exists() else None,
        )
        monkeypatch.setattr(
            "utils.os.chown",
            lambda path, uid, gid: chown_calls.append((Path(path), uid, gid)),
        )

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert target.read_text() == "model: restored\n"
        # config.yaml pre-existed, so its owner is captured and re-applied;
        # state.db is newly created, so there is no prior owner to restore.
        assert chown_calls == [(target, 123, 456)]

    @pytest.mark.skipif(not hasattr(os, "fchmod"), reason="needs fchmod present to remove it")
    def test_mode_is_applied_before_the_replace_without_fchmod(self, tmp_path, monkeypatch):
        """Covers the Windows branch: no ``fchmod``, so ``chmod`` the temp path.

        Applying the mode only *after* ``atomic_replace`` leaves the published
        file at mkstemp's 0600 until that chmod lands (and permanently if the
        process dies in between), and ``atomic_replace``'s EXDEV/EBUSY
        ``shutil.copystat`` fallback would copy 0600 onto the target. Mirrors
        the transit-window fix ``atomic_yaml_write`` already carries.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        target = hermes_home / "config.yaml"
        target.write_text("model: original\n")
        os.chmod(target, 0o644)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._zip(zip_path, {"config.yaml": "model: restored\n", "state.db": ""})

        import hermes_cli.backup as backup_mod

        real_replace = backup_mod.atomic_replace
        staged_modes: list[int] = []

        def spying_replace(tmp, dst):
            if Path(dst).name == "config.yaml":
                staged_modes.append(os.stat(tmp).st_mode & 0o777)
            return real_replace(tmp, dst)

        monkeypatch.delattr(os, "fchmod")
        monkeypatch.setattr(backup_mod, "atomic_replace", spying_replace)

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        # Without the pre-replace chmod this reads 0o600 (mkstemp's mode).
        assert staged_modes == [0o644]
        assert (target.stat().st_mode & 0o777) == 0o644

    @pytest.mark.skipif(os.name != "posix", reason="POSIX setuid/setgid bits")
    def test_restore_does_not_carry_setuid_onto_archive_content(
        self, tmp_path, monkeypatch
    ):
        """An imported member must not inherit a privileged target's identity.

        ``_preserve_file_mode`` returns ``stat.S_IMODE``, i.e. all twelve bits,
        so a target sitting at 0o6755 hands setuid/setgid straight back to a
        file whose contents now come from the zip.  Whoever produced the
        archive would then get whatever that file executes as.  The other
        ``utils`` writers can preserve the full mode safely because they
        re-serialize content this process produced; ``hermes import`` is the
        one write path where the bytes are untrusted, and it is also the path
        that documents ``sudo`` use for owner preservation.

        The sibling assertions in this class mask with ``& 0o777``, which
        discards exactly the bits at issue, so this failure is invisible to
        them.
        """
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        target = hermes_home / "helper.sh"
        target.write_text("#!/bin/sh\necho original\n")
        os.chmod(target, 0o6755)
        if stat.S_IMODE(target.stat().st_mode) != 0o6755:
            pytest.skip("filesystem refuses setuid/setgid on a user-owned file")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        self._zip(
            zip_path,
            {"helper.sh": "#!/bin/sh\necho attacker\n", "state.db": ""},
        )

        import hermes_cli.backup as backup_mod

        real_replace = backup_mod.atomic_replace
        staged_modes: list[int] = []

        def spying_replace(tmp, dst):
            if Path(dst).name == "helper.sh":
                staged_modes.append(stat.S_IMODE(os.stat(tmp).st_mode))
            return real_replace(tmp, dst)

        monkeypatch.setattr(backup_mod, "atomic_replace", spying_replace)

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        published = stat.S_IMODE(target.stat().st_mode)
        assert target.read_text() == "#!/bin/sh\necho attacker\n"
        assert not published & stat.S_ISUID, (
            f"archive content kept the target's setuid bit (mode 0o{published:o})"
        )
        assert not published & stat.S_ISGID, (
            f"archive content kept the target's setgid bit (mode 0o{published:o})"
        )
        # The ordinary permission bits are still preserved — this drops the
        # elevated bits, it does not fall back to mkstemp's 0600.
        assert published == 0o755
        # And there must be no transient elevation either: the temp file is
        # chmod'd before the replace, so it must never carry the bits.
        assert staged_modes == [0o755], (
            f"the staged temp file was elevated before publish: {staged_modes}"
        )


# ---------------------------------------------------------------------------
# Profile restoration tests
# ---------------------------------------------------------------------------

class TestProfileRestoration:
    def _make_backup_zip(self, zip_path: Path, files: dict[str, str | bytes]) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)


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

        result = _safe_copy_db(src, dst)
        assert result is True

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT x FROM t").fetchall()
        conn.close()
        assert rows == [(42,)]


    def test_is_zeroed_sqlite_file_detects_nul_header(self, tmp_path):
        from hermes_cli.backup import is_zeroed_sqlite_file
        p = tmp_path / "state.db"
        p.write_bytes(bytes(4096))  # all NULs
        assert is_zeroed_sqlite_file(p) is True


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

    def test_failed_state_db_copy_is_loud(self, hermes_home, monkeypatch, capsys):
        """#68474: unreadable state.db must not look like a silent success."""
        from hermes_cli import backup as backup_mod

        def boom(src, dst):
            return False

        monkeypatch.setattr(backup_mod, "_safe_copy_db", boom)
        snap_id = backup_mod.create_quick_snapshot(hermes_home=hermes_home)
        err = capsys.readouterr().out
        assert "SQLite safe copy FAILED" in err or "CRITICAL" in err
        assert "state.db" in err
        # Other small files may still snapshot
        if snap_id:
            manifest = (hermes_home / "state-snapshots" / snap_id / "manifest.json")
            assert manifest.exists()
            data = json.loads(manifest.read_text(encoding="utf-8"))
            assert "state.db" not in data.get("files", {})
            assert "state.db" in data.get("failed_dbs", [])














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



# ---------------------------------------------------------------------------
# Pre-update backup (hermes update safety net)
# ---------------------------------------------------------------------------

    # -- security: path traversal regression coverage -----------------------
    # Per @egilewski audit on PR #9217: restore_quick_snapshot must reject
    # malicious snapshot_id values (the directory selector) AND malicious
    # rel paths inside the manifest (the per-file selector). Both surfaces
    # need explicit regression tests because they validate independent
    # traversal vectors.



    def test_oversized_db_suppresses_pruning(self, hermes_home, capsys):
        """#68805: an oversized state.db skipped for size must suppress
        pruning so the older complete snapshot (containing the only
        recoverable database) is preserved.

        Reproduces the reviewer's scenario: keep=1 + a state.db exceeding
        the size cap → the new snapshot omits state.db, failed_dbs stays
        empty (the file wasn't unreadable, just too large), and without
        tracking oversized_skipped the older complete snapshot would be
        pruned — losing the only recovery copy.
        """
        import json
        from hermes_cli.backup import create_quick_snapshot, list_quick_snapshots

        # First snapshot: complete (state.db is small, under any cap)
        first_id = create_quick_snapshot(label="complete", hermes_home=hermes_home)
        assert first_id is not None
        first_dir = hermes_home / "state-snapshots" / first_id
        assert (first_dir / "state.db").exists()

        _advance_backup_clock()

        # Second snapshot: state.db exceeds the 1024-byte cap → skipped for
        # size, but small config files (32-54 bytes) still land in the manifest.
        second_id = create_quick_snapshot(
            label="oversized", hermes_home=hermes_home, max_file_size=1024, keep=1
        )
        assert second_id is not None
        second_dir = hermes_home / "state-snapshots" / second_id
        assert not (second_dir / "state.db").exists()

        # Manifest must record the oversized skip
        with open(second_dir / "manifest.json") as f:
            meta = json.load(f)
        assert "state.db" in meta.get("oversized_skipped", [])

        # CRITICAL: the first (complete) snapshot must survive pruning
        # because the second snapshot is incomplete (oversized state.db).
        all_snaps = list_quick_snapshots(limit=100, hermes_home=hermes_home)
        snap_ids = {s["id"] for s in all_snaps}
        assert first_id in snap_ids, (
            f"Complete snapshot {first_id} was pruned by an incomplete "
            f"(oversized) snapshot — the recovery copy was lost!"
        )
        assert second_id in snap_ids

        out = capsys.readouterr().out
        assert "skipping state.db" in out.lower() or "skipping snapshot prune" in out.lower()


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


    @pytest.fixture
    def hermes_home(self, tmp_path):
        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        return root


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


    def test_rotation_keeps_only_n(self, hermes_home):
        """After more than ``keep`` backups are created, older ones are
        pruned automatically."""
        from hermes_cli.backup import create_pre_update_backup

        created = []
        for _ in range(5):
            out = create_pre_update_backup(hermes_home=hermes_home, keep=3)
            created.append(out)
            _advance_backup_clock()

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
        # Config reads resolve HERMES_HOME dynamically and their caches are
        # keyed by config path. Do not remove shared modules from sys.modules:
        # other test modules may retain imports from the existing module object.
        return root

    @staticmethod
    def _set_mode(hermes_home, value):
        import yaml
        (hermes_home / "config.yaml").write_text(yaml.safe_dump({
            "_config_version": 22,
            "updates": {"pre_update_backup": value},
        }))

    @staticmethod
    def _zips(hermes_home):
        d = hermes_home / "backups"
        return list(d.glob("pre-update-*.zip")) if d.exists() else []

    @staticmethod
    def _snaps(hermes_home):
        d = hermes_home / "state-snapshots"
        return [p for p in d.iterdir() if p.is_dir()] if d.exists() else []




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



    def test_config_full_mode(self, hermes_home, capsys):
        self._set_mode(hermes_home, "full")
        from hermes_cli.main import _run_pre_update_backup
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
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


    def test_restorable_with_hermes_import(self, hermes_home, tmp_path):
        """The zip produced by pre-migration backup must be a valid Hermes
        backup — `hermes import` should accept it."""
        from hermes_cli.backup import create_pre_migration_backup, _validate_backup_zip
        out = create_pre_migration_backup(hermes_home=hermes_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            valid, _reason = _validate_backup_zip(zf)
        assert valid, "pre-migration zip failed _validate_backup_zip"




    def test_does_not_touch_pre_update_backups(self, hermes_home):
        """Pre-migration rotation must only prune pre-migration-*.zip files,
        leaving pre-update-*.zip backups untouched."""
        from hermes_cli.backup import create_pre_update_backup, create_pre_migration_backup
        update_backup = create_pre_update_backup(hermes_home=hermes_home, keep=5)
        assert update_backup is not None and update_backup.exists()
        # Spin up a lot of migration backups with keep=1
        for _ in range(3):
            out = create_pre_migration_backup(hermes_home=hermes_home, keep=1)
            assert out is not None
            _advance_backup_clock()
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




