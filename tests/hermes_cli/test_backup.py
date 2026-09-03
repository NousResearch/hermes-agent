"""Tests for hermes backup and import commands."""

import errno
import json
import os
import shutil
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


    def test_excludes_backups_dir(self):
        """backups/ is excluded so pre-update backups don't nest exponentially."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("backups/pre-update-2026-04-27-063400.zip"))

    def test_excludes_state_snapshots_dir(self):
        """state-snapshots/ is excluded for the same reason as backups/: every
        quick / pre-update snapshot holds its own copy of state.db, so zipping
        the tree would ship the DB once per retained snapshot."""
        from hermes_cli.backup import _EXCLUDED_DIRS, _QUICK_SNAPSHOTS_DIR, _should_exclude
        assert _QUICK_SNAPSHOTS_DIR in _EXCLUDED_DIRS
        assert _should_exclude(Path(_QUICK_SNAPSHOTS_DIR) / "20260814-203829-2026-08-15" / "state.db")
        assert _should_exclude(Path(_QUICK_SNAPSHOTS_DIR) / "20260814-203829-2026-08-15" / "manifest.json")
        # Named profiles accumulate snapshots too.
        assert _should_exclude(Path("profiles/coder") / _QUICK_SNAPSHOTS_DIR / "x" / "state.db")
        # The live DB is still backed up.
        assert not _should_exclude(Path("state.db"))

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

    def test_excludes_managed_runtime_trees_at_root(self):
        """models/, runtimes/, and node/ at a profile-home root hold
        re-downloadable GGUF weights and runtime binaries that reach
        hundreds of GB — zipping them is the 20-minute-hang symptom."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("models/Qwen3.6-27B-Q4_K_M.gguf"))
        assert _should_exclude(Path("models/assets/mmproj.gguf"))
        assert _should_exclude(Path("runtimes/llamacpp/b10362/cuda/ggml-cuda.dll"))
        assert _should_exclude(Path("node/node.exe"))
        # Named profiles download their own copies.
        assert _should_exclude(Path("profiles/clean/models/big.gguf"))
        assert _should_exclude(Path("profiles/clean/runtimes/llamacpp/x.dll"))

    def test_keeps_nested_dirs_named_like_runtime_trees(self):
        """A deeper directory that happens to be called models/ or node/ is
        user data (a skill's assets, project files) and must survive."""
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/mlops/models/notes.md"))
        assert not _should_exclude(Path("scratch/node/index.js"))
        assert not _should_exclude(Path("profiles/clean/skills/x/models/a.txt"))

    def test_excludes_desktop_emergency_state_db_baks(self):
        """The desktop updater's pre-flight drops timestamped
        state.db.pre-update-emergency-*.bak files at the HERMES_HOME root —
        backup artifacts in the same class as backups/, so a full backup
        must not re-ship them."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(
            Path("state.db.pre-update-emergency-2026-08-15T04-55-33-619Z.bak")
        )
        assert _should_exclude(
            Path("profiles/coder/state.db.pre-update-emergency-2026-08-15T04-55-33-619Z.bak")
        )
        # Other .bak files are user data and stay.
        assert not _should_exclude(Path("config.yaml.bak"))


# ---------------------------------------------------------------------------
# _iter_backup_files tests
# ---------------------------------------------------------------------------

class TestIterBackupFiles:
    def test_manual_and_automatic_paths_share_one_walk(self, tmp_path):
        """Both backup entry points must select the identical file set.

        Before the walks were unified, the automatic pre-update path pruned
        ``hermes-agent`` at ANY depth, silently dropping nested skill dirs
        like ``skills/autonomous-ai-agents/hermes-agent/`` that the manual
        path preserved. One shared iterator makes that drift impossible;
        this test pins the contract."""
        from hermes_cli.backup import _iter_backup_files

        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)

        # The case the old automatic walk got wrong: a nested dir named
        # hermes-agent holding real skill content.
        nested = root / "skills" / "autonomous-ai-agents" / "hermes-agent"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("# nested skill\n")

        # A root-level managed runtime tree that both paths must prune.
        (root / "models").mkdir()
        (root / "models" / "big.gguf").write_bytes(b"\x00" * 64)

        out_path = tmp_path / "out.zip"
        selected = {str(rel) for _, rel in _iter_backup_files(root, out_path)}

        rel_nested = str(Path("skills/autonomous-ai-agents/hermes-agent/SKILL.md"))
        assert rel_nested in selected
        assert str(Path("models/big.gguf")) not in selected
        assert not any(s.startswith("hermes-agent") for s in selected)

    def test_skipped_dirs_collected_for_summary(self, tmp_path):
        from hermes_cli.backup import _iter_backup_files

        root = tmp_path / ".hermes"
        root.mkdir()
        _make_hermes_tree(root)
        (root / "models").mkdir()
        (root / "models" / "big.gguf").write_bytes(b"\x00")

        skipped: set = set()
        list(_iter_backup_files(root, tmp_path / "out.zip", skipped))
        assert "models" in skipped
        assert "hermes-agent" in skipped


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

    def test_state_snapshots_not_nested_into_backup(self, tmp_path, monkeypatch):
        """A quick snapshot left under state-snapshots/ must not be re-shipped
        by the full backup — each snapshot already holds a copy of state.db, so
        nesting them multiplies the archive by (1 + retained snapshots)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)
        with sqlite3.connect(hermes_home / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO sessions VALUES ('s1')")

        from hermes_cli.backup import _QUICK_SNAPSHOTS_DIR, create_quick_snapshot, run_backup

        # Real producer, so the layout under state-snapshots/ is whatever the
        # code actually writes (manifest.json + state.db copy + ...).
        snap_id = create_quick_snapshot(hermes_home=hermes_home)
        assert snap_id and (hermes_home / _QUICK_SNAPSHOTS_DIR / snap_id / "state.db").exists()

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        out_zip = tmp_path / "backup.zip"
        run_backup(Namespace(output=str(out_zip)))

        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
        assert not any(n.startswith(_QUICK_SNAPSHOTS_DIR + "/") for n in names), names
        # Exactly one state.db in the archive: the live one.
        assert [n for n in names if n == "state.db" or n.endswith("/state.db")] == ["state.db"]


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

        ok, err = _safe_copy_db(src, dst)
        assert ok is True
        assert err is None

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT x FROM t").fetchall()
        conn.close()
        assert rows == [(42,)]

    def test_aborts_when_source_remains_busy_past_deadline(
        self, tmp_path, monkeypatch
    ):
        from hermes_cli import backup as backup_mod

        src = tmp_path / "locked.db"
        dst = tmp_path / "copy.db"
        src.touch()
        dst.write_bytes(b"partial")

        clock = iter((100.0, 100.5, 101.1))

        class FakeSourceConnection:
            def backup(self, _destination, *, pages, progress, sleep):
                assert pages > 0
                assert sleep > 0
                progress(sqlite3.SQLITE_BUSY, 0, 1)
                progress(sqlite3.SQLITE_BUSY, 0, 1)

            def close(self):
                pass

        destination_closed = []

        class FakeDestinationConnection:
            def close(self):
                destination_closed.append(True)

        connections = iter((FakeSourceConnection(), FakeDestinationConnection()))
        real_unlink = Path.unlink

        def assert_closed_before_unlink(path, *args, **kwargs):
            assert destination_closed
            return real_unlink(path, *args, **kwargs)

        connect_calls = []

        def fake_connect(*args, **kwargs):
            connect_calls.append((args, kwargs))
            return next(connections)

        monkeypatch.setattr(backup_mod.sqlite3, "connect", fake_connect)
        monkeypatch.setattr(backup_mod.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(Path, "unlink", assert_closed_before_unlink)

        ok, error = backup_mod._safe_copy_db(src, dst, timeout_seconds=1.0)
        assert ok is False
        assert error is not None
        assert connect_calls[0][1]["timeout"] == 0.0
        assert not dst.exists()

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



    def test_locked_source_fails_fast_not_hang(self, tmp_path):
        import subprocess
        import sys
        import time

        from hermes_cli.backup import _safe_copy_db
        src = tmp_path / "locked.db"
        dst = tmp_path / "copy.db"

        conn = sqlite3.connect(str(src))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()

        # Hold an EXCLUSIVE transaction in a separate process. POSIX file
        # locks only conflict across processes, so an in-process connection
        # cannot reproduce the "database is locked" condition.
        holder = (
            "import sqlite3, time\n"
            f"c = sqlite3.connect({str(src)!r})\n"
            "c.execute('BEGIN EXCLUSIVE')\n"
            "print('LOCKED', flush=True)\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", holder],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "LOCKED"
            started = time.monotonic()
            ok, err = _safe_copy_db(src, dst)
            elapsed = time.monotonic() - started
            assert ok is False
            assert err is not None and "locked" in err
            # The busy timeout is 5s, so a fast failure lands around there.
            # The regression this guards against is backup() retrying
            # SQLITE_BUSY forever, which would never return at all.
            assert elapsed < 30
        finally:
            proc.kill()
            proc.wait()


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
            # _safe_copy_db returns (ok, reason) on this branch so the
            # failure can be reported with its cause.
            return False, "simulated unreadable database"

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

    def test_restore_state_db_live_connection(self, hermes_home):
        """Restoring state.db must update data visible through a live connection.

        Regression test for #65942: when state.db is open with a live SQLite
        connection (as happens with the gateway, dashboard, or another CLI
        session), the restore must write pages through the backup API so the
        live connection sees the restored data instead of stale cached pages
        from a replaced inode.
        """
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot
        snap_id = create_quick_snapshot(hermes_home=hermes_home)

        # Open a live connection (simulating gateway/dashboard).
        live_conn = sqlite3.connect(str(hermes_home / "state.db"))
        live_conn.execute("PRAGMA journal_mode=wal")
        # Insert data AFTER the snapshot — this is what must be reverted.
        live_conn.execute("INSERT INTO sessions VALUES ('s2', 'new-data')")
        live_conn.commit()

        rows_before = live_conn.execute("SELECT * FROM sessions").fetchall()
        assert len(rows_before) == 2

        # Restore — the live connection stays open during restore.
        result = restore_quick_snapshot(snap_id, hermes_home=hermes_home)
        assert result is True

        # The live connection must see the restored (single-row) state.
        # A fresh connection would trivially work; the live one is the test.
        rows_after = live_conn.execute("SELECT * FROM sessions").fetchall()
        live_conn.close()
        assert len(rows_after) == 1, (
            f"Live connection still sees {len(rows_after)} rows after restore "
            f"(expected 1); the extra row 's2' should have been reverted."
        )










    def test_non_db_failure_is_incomplete_but_not_reported_as_a_db(
        self, hermes_home, monkeypatch, capsys
    ):
        """A non-DB protected file that fails to copy must not be reported as a
        failed DATABASE.

        #71223's ``failed_dbs`` means "present *.db that could not be
        snapshotted" and drives a DB-specific CRITICAL report. Recording every
        failure there made a failed ``.env`` print "could not snapshot DB
        file(s): .env". The failure still has to make the snapshot incomplete,
        which is what ``failed`` is for, so pruning must still be blocked.
        """
        import shutil as _shutil

        from hermes_cli import backup as backup_mod

        prior = hermes_home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text('{"id": "20200101-000000", "files": {}}')

        real_copy2 = _shutil.copy2

        def copy2_failing_on_env(src, dst, *a, **k):
            if str(src).endswith(".env"):
                raise OSError("simulated unreadable .env")
            return real_copy2(src, dst, *a, **k)

        monkeypatch.setattr(backup_mod.shutil, "copy2", copy2_failing_on_env)

        snap_id = backup_mod.create_quick_snapshot(hermes_home=hermes_home, keep=1)
        assert snap_id is not None
        out = capsys.readouterr().out

        with open(hermes_home / "state-snapshots" / snap_id / "manifest.json") as f:
            meta = json.load(f)
        # Recorded as a failure with its reason...
        assert ".env" in meta["failed"]
        # ...but NOT as a failed database.
        assert ".env" not in meta.get("failed_dbs", [])
        assert "could not snapshot DB file(s)" not in out
        # Still incomplete, so the prior complete snapshot survives keep=1.
        assert prior.is_dir(), "a non-DB failure let the prune evict the last snapshot"

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
        # A skipped *.db is reported through #71223's DB-specific line; the
        # generic protected-file line covers the non-DB case (see
        # test_oversized_file_inside_protected_dir_blocks_eviction).
        assert (
            "Skipping snapshot prune: DB file(s) skipped for size: state.db"
            in out
        )

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

    def test_all_size_skipped_snapshot_persists_manifest_and_keeps_prior(
        self, tmp_path
    ):
        """When EVERY present protected file is skipped for size, `manifest`
        and `failed` both stay empty -- only `size_skipped` is populated. The
        snapshot must still persist a manifest recording size_skipped and must
        NOT be deleted, and a prior complete snapshot must survive.

        Reproduces egilewski's report on #68907: an 8 KiB state.db as the only
        present protected file with a 4 KiB cap left no durable size_skipped
        record, because `if not manifest and not failed:` fired and deleted
        snap_dir before the manifest (and its size_skipped entries) could ever
        be written -- losing the forensic "these files existed but were
        skipped for size" record."""
        from hermes_cli.backup import create_quick_snapshot

        home = tmp_path / ".hermes"
        home.mkdir()
        # Only protected file present, and it is over the cap -- the
        # all-size-skipped path (manifest={}, failed={}, size_skipped={...}).
        (home / "state.db").write_bytes(b"x" * 8192)

        prior = home / "state-snapshots" / "20200101-000000"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text('{"id": "20200101-000000", "files": {}}')

        snap_id = create_quick_snapshot(hermes_home=home, max_file_size=4096, keep=1)

        assert snap_id is not None, (
            "an all-size-skipped snapshot must not return None -- it has a "
            "durable size_skipped record to persist"
        )
        snap_dir = home / "state-snapshots" / snap_id
        assert snap_dir.is_dir(), "snapshot dir must not be deleted"
        manifest_path = snap_dir / "manifest.json"
        assert manifest_path.exists(), (
            "manifest must persist even though manifest/failed are both empty"
        )
        with open(manifest_path) as f:
            meta = json.load(f)
        assert meta.get("files") == {}
        assert "state.db" in meta.get("size_skipped", {})
        # The prior complete snapshot must not be evicted by this incomplete run.
        assert prior.is_dir(), "prior complete snapshot was pruned by an incomplete run"

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
            # os.scandir accepts an int dir-fd on POSIX, and this patch is
            # process-global, so an unrelated fd-based scandir call (e.g.
            # from shutil.copy2's Linux sendfile fast path) must pass
            # through untouched; only intercept a real path to pairing/.
            if not isinstance(path, int) and Path(path) == pairing_dir:
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
            # os.scandir accepts an int dir-fd on POSIX, and this patch is
            # process-global, so an unrelated fd-based scandir call (e.g.
            # from shutil.copy2's Linux sendfile fast path) must pass
            # through untouched; only intercept a real path to pairing/.
            if not isinstance(path, int) and Path(path) == pairing_dir:
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
            # os.scandir accepts an int dir-fd on POSIX, and this patch is
            # process-global, so unrelated fd-based scandir calls must pass
            # through untouched; only intercept a real path to workspaces/.
            if not isinstance(path, int):
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
            # os.stat accepts an int fd on POSIX (equivalent to os.fstat),
            # and this patch is process-global, so an unrelated fd-based
            # stat call (e.g. from shutil.copy2's Linux sendfile fast
            # path, which stats the source fd for its size) must pass
            # through untouched; only intercept the real target path.
            if not isinstance(path, int) and Path(path) == target:
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
            # os.scandir accepts an int dir-fd on POSIX, and this patch is
            # process-global, so an unrelated fd-based scandir call (e.g.
            # from shutil.copy2's Linux sendfile fast path) must pass
            # through untouched; only intercept a real path to pairing/.
            if not isinstance(path, int):
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
            # os.scandir accepts an int dir-fd on POSIX, and this patch is
            # process-global, so an unrelated fd-based scandir call (e.g.
            # from shutil.copy2's Linux sendfile fast path) must pass
            # through untouched; only intercept a real path to pairing/.
            if not isinstance(path, int):
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
            # os.scandir accepts an int dir-fd on POSIX, and this patch is
            # process-global, so an unrelated fd-based scandir call (e.g.
            # from shutil.copy2's Linux sendfile fast path) must pass
            # through untouched; only intercept a real path to pairing/.
            if not isinstance(path, int) and Path(path) == pairing_dir:
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
            # os.scandir accepts an int dir-fd on POSIX, and this patch is
            # process-global, so an unrelated fd-based scandir call (e.g.
            # from shutil.copy2's Linux sendfile fast path) must pass
            # through untouched; only intercept a real path to pairing/.
            if not isinstance(path, int) and Path(path) == pairing_dir:
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

    def test_pre_update_zip_does_not_nest_the_pre_update_snapshot(self, hermes_home):
        """``hermes update`` in ``full`` mode takes the quick snapshot *before*
        the full zip, so the zip walk sees the snapshot it just made. It must
        skip it — otherwise every pre-update zip ships state.db twice."""
        from hermes_cli.backup import (
            _QUICK_SNAPSHOTS_DIR,
            create_pre_update_backup,
            create_quick_snapshot,
        )
        with sqlite3.connect(hermes_home / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")

        snap_id = create_quick_snapshot(label="pre-update", hermes_home=hermes_home)
        assert snap_id and (hermes_home / _QUICK_SNAPSHOTS_DIR / snap_id / "state.db").exists()

        out = create_pre_update_backup(hermes_home=hermes_home)
        assert out is not None
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "state.db" in names
        assert not any(n.startswith(_QUICK_SNAPSHOTS_DIR + "/") for n in names), names


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
# config.yaml model/provider + MoA auto-restore after silent update rewrite
# (issue #64160)
# ---------------------------------------------------------------------------

class TestRestoreConfigModelSettingsIfRewritten:
    """Desktop update/repair cycles have rewritten user-set model.provider /
    model.default and dropped the moa: section (#64160).
    `restore_config_model_settings_if_rewritten` is the post-update safety net
    that restores only the protected keys from the pre-update snapshot."""

    USER_CONFIG = (
        "_config_version: 39\n"
        "model:\n"
        "  provider: custom\n"
        "  default: zyphra/zamba-3-large\n"
        "  base_url: https://api.zyphra.example/v1\n"
        "moa:\n"
        "  enabled: true\n"
        "  presets:\n"
        "    council:\n"
        "      aggregator: {provider: custom, model: zyphra/zamba-3-large}\n"
        "custom_unknown_key:\n"
        "  hello: world\n"
    )

    def _make_snapshot(self, hermes_home: Path, label="pre-update"):
        from hermes_cli.backup import create_quick_snapshot
        return create_quick_snapshot(label=label, hermes_home=hermes_home, keep=5)

    def _seed(self, hermes_home: Path) -> Path:
        hermes_home.mkdir(parents=True, exist_ok=True)
        cfg = hermes_home / "config.yaml"
        cfg.write_text(self.USER_CONFIG, encoding="utf-8")
        return cfg

    def test_restores_rewritten_provider_and_dropped_moa(self, tmp_path):
        import yaml
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        hermes_home = tmp_path / ".hermes"
        cfg = self._seed(hermes_home)
        snap_id = self._make_snapshot(hermes_home)
        assert snap_id

        # The update flow rewrites config.yaml with defaults: provider flips
        # to deepseek, MoA section is gone (the #64160 field report).
        cfg.write_text(
            "_config_version: 39\nmodel:\n  provider: deepseek\n  default: deepseek-chat\n",
            encoding="utf-8",
        )

        result = restore_config_model_settings_if_rewritten(
            snap_id, hermes_home=hermes_home
        )
        assert result is not None
        assert result["restored"] is True
        assert result["snapshot_id"] == snap_id
        assert "model.provider" in result["keys"]
        assert "model.default" in result["keys"]
        assert "moa" in result["keys"]

        after = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert after["model"]["provider"] == "custom"
        assert after["model"]["default"] == "zyphra/zamba-3-large"
        assert after["model"]["base_url"] == "https://api.zyphra.example/v1"
        assert after["moa"]["enabled"] is True
        assert after["moa"]["presets"]["council"]["aggregator"]["model"] == (
            "zyphra/zamba-3-large"
        )

    def test_noop_when_config_untouched(self, tmp_path):
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        hermes_home = tmp_path / ".hermes"
        cfg = self._seed(hermes_home)
        snap_id = self._make_snapshot(hermes_home)
        before = cfg.read_text(encoding="utf-8")

        result = restore_config_model_settings_if_rewritten(
            snap_id, hermes_home=hermes_home
        )
        assert result is None
        assert cfg.read_text(encoding="utf-8") == before

    def test_preserves_legitimate_update_writes(self, tmp_path):
        """Only protected keys are restored — a version bump or a new section
        the migration legitimately wrote must survive the restore."""
        import yaml
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        hermes_home = tmp_path / ".hermes"
        cfg = self._seed(hermes_home)
        snap_id = self._make_snapshot(hermes_home)

        cfg.write_text(
            "_config_version: 40\n"      # legitimate migration bump
            "new_section:\n  added: true\n"  # legitimate new default
            "model:\n  provider: deepseek\n",  # illegitimate rewrite
            encoding="utf-8",
        )

        result = restore_config_model_settings_if_rewritten(
            snap_id, hermes_home=hermes_home
        )
        assert result is not None
        after = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert after["model"]["provider"] == "custom"          # restored
        assert after["_config_version"] == 40                   # kept
        assert after["new_section"] == {"added": True}          # kept

    def test_noop_when_user_never_set_protected_keys(self, tmp_path):
        """A config that never had model.provider/moa set gets no restore even
        if the update writes those keys fresh — nothing of the user's was lost."""
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        cfg = hermes_home / "config.yaml"
        cfg.write_text("_config_version: 39\nagent: {}\n", encoding="utf-8")
        snap_id = self._make_snapshot(hermes_home)

        cfg.write_text(
            "_config_version: 39\nmodel:\n  provider: deepseek\n", encoding="utf-8"
        )
        result = restore_config_model_settings_if_rewritten(
            snap_id, hermes_home=hermes_home
        )
        assert result is None

    def test_noop_on_unreadable_live_config(self, tmp_path):
        """An unparseable live config is a different failure the user must see;
        the safety net leaves it alone (mirrors the cron net's posture)."""
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        hermes_home = tmp_path / ".hermes"
        cfg = self._seed(hermes_home)
        snap_id = self._make_snapshot(hermes_home)

        cfg.write_text(": not [valid yaml", encoding="utf-8")
        result = restore_config_model_settings_if_rewritten(
            snap_id, hermes_home=hermes_home
        )
        assert result is None
        assert cfg.read_text(encoding="utf-8") == ": not [valid yaml"

    def test_noop_without_snapshot_id(self, tmp_path):
        from hermes_cli.backup import restore_config_model_settings_if_rewritten

        assert restore_config_model_settings_if_rewritten(
            "", hermes_home=tmp_path / ".hermes"
        ) is None







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


# ---------------------------------------------------------------------------
# run_import: HERMES_HOME override handling (issue #99839)
# ---------------------------------------------------------------------------

class TestImportHonorsHermesHomeOverride:
    """`hermes import` must restore into the home the command runs under.

    Resolving the target through get_default_hermes_root() maps a profile
    home (<root>/profiles/<name>) back to <root>: the import then overwrites
    the live root's config.yaml while the profile directory stays empty —
    exactly what "Target:" printed it would NOT do.
    """

    def _make_backup_zip(self, tmp_path):
        import zipfile

        src_root = tmp_path / "src-home"
        src_root.mkdir()
        (src_root / "config.yaml").write_text("model:\n  provider: anthropic\n")
        (src_root / ".env").write_text("ANTHROPIC_API_KEY=sk-test\n")
        zip_path = tmp_path / "backup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(src_root / "config.yaml", "config.yaml")
            zf.write(src_root / ".env", ".env")
        return zip_path

    def test_import_targets_named_profile_home(self, tmp_path, monkeypatch):
        """HERMES_HOME=<root>/profiles/<name> must restore INTO the profile,
        not into <root> (which would clobber the live root config)."""
        root = tmp_path / "hermes-root"
        profile = root / "profiles" / "coder"
        profile.mkdir(parents=True)
        # Live root config that must survive untouched.
        (root / "config.yaml").write_text("model:\n  provider: openai\n")

        monkeypatch.setenv("HERMES_HOME", str(profile))
        from hermes_constants import get_hermes_home

        assert get_hermes_home() == profile

        zip_path = self._make_backup_zip(tmp_path)

        import argparse

        from hermes_cli.backup import run_import

        args = argparse.Namespace(zipfile=str(zip_path), force=True)
        run_import(args)

        assert (profile / "config.yaml").read_text() == (
            "model:\n  provider: anthropic\n"
        )
        assert (root / "config.yaml").read_text() == "model:\n  provider: openai\n"

    def test_import_skips_gateway_install_for_non_default_home(
        self, tmp_path, monkeypatch
    ):
        """A restore into a sandbox must not silently start a second gateway
        pointed at it — the profile/sandbox gateway would shadow the default
        service installed by the primary install."""
        native_default = tmp_path / "native-default"
        sandbox = tmp_path / "sandbox-home"
        sandbox.mkdir(parents=True)
        # Live default install markers.
        native_default.mkdir()
        (native_default / "config.yaml").write_text("model:\n  provider: openai\n")

        monkeypatch.setenv("HERMES_HOME", str(sandbox))

        import argparse

        import hermes_constants
        from hermes_cli import backup as backup_mod

        monkeypatch.setattr(
            backup_mod,
            "_get_platform_default_hermes_home",
            lambda: native_default,
        )

        calls = []
        monkeypatch.setattr(
            "hermes_cli.gateway.ensure_gateway_service",
            lambda *a, **kw: calls.append(kw),
        )
        monkeypatch.setattr(
            "hermes_cli.gateway._is_service_running",
            lambda: False,
        )

        zip_path = self._make_backup_zip(tmp_path)
        args = argparse.Namespace(zipfile=str(zip_path), force=True)
        backup_mod.run_import(args)

        assert calls == [], "gateway must not be auto-installed for a sandbox restore"

    def test_import_installs_gateway_when_default_home_is_target(
        self, tmp_path, monkeypatch
    ):
        """Restoring into the default home keeps the auto-install behavior."""
        native_default = tmp_path / "native-default"
        native_default.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(native_default))

        import argparse

        from hermes_cli import backup as backup_mod

        monkeypatch.setattr(
            backup_mod,
            "_get_platform_default_hermes_home",
            lambda: native_default,
        )

        calls = []
        monkeypatch.setattr(
            "hermes_cli.gateway.ensure_gateway_service",
            lambda *a, **kw: calls.append(kw),
        )
        monkeypatch.setattr(
            "hermes_cli.gateway._is_service_running",
            lambda: False,
        )

        zip_path = self._make_backup_zip(tmp_path)
        args = argparse.Namespace(zipfile=str(zip_path), force=True)
        backup_mod.run_import(args)

        assert calls and calls[0].get("context") == "import"
