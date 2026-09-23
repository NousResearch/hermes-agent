"""Tests for hermes backup and import commands."""

import json
import os
import socket
import sqlite3
import stat
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.hermes_cli._backup_fixtures import (
    _advance_backup_clock, _make_hermes_tree,
    _no_real_gateway_service,  # noqa: F401 - autouse pytest fixture
    _symlink_file_or_skip,
)


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
        from hermes_cli.backup import _QUICK_SNAPSHOTS_DIR, _should_exclude
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

    def test_excludes_regenerable_cache_but_keeps_durable_artifacts(self):
        """Catalogs and live browser profiles are rebuilt on demand; delivered media and the
        citation ledger are not, so they stay in the archive."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("cache/model_catalog.json"))
        assert _should_exclude(Path("cache/chrome-debug/Default/Cookies"))
        assert _should_exclude(Path("profiles/sage/cache/chrome-debug/cache.db"))
        assert not _should_exclude(Path("cache/images/x.png"))
        assert not _should_exclude(Path("profiles/sage/cache/citations/ledger.json"))
        assert not _should_exclude(Path("skills/example/cache/notes.md"))

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

    def test_prunes_browser_use_cli_profiles_at_home_roots_only(self, tmp_path):
        """The Browser Use CLI backend writes ``HERMES_HOME/browser_profiles/`` (underscore) — a
        live Chromium user-data dir holding Login Data / Cookies. It must never enter an archive,
        at the root or under ``profiles/<name>/``; a skill's same-named dir is user data (#117346)."""
        from hermes_cli.backup import _iter_backup_files

        root = tmp_path / ".hermes"
        root.mkdir()
        files = {
            "browser_profiles/browser-use-default/Default/Login Data": False,
            "browser_profiles/browser-use-default/Default/Network/Cookies": False,
            "profiles/coder/browser_profiles/browser-use-default/Default/Cookies": False,
            "skills/example/browser_profiles/notes.md": True,
        }
        for rel in files:
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")
        selected = {str(rel) for _, rel in _iter_backup_files(root, tmp_path / "out.zip")}
        assert {rel for rel in files if str(Path(rel)) in selected} == {rel for rel, keep in files.items() if keep}

    def test_prunes_regenerable_caches_but_keeps_durable_and_nested(self, tmp_path):
        from hermes_cli.backup import _iter_backup_files

        root = tmp_path / ".hermes"
        root.mkdir()
        files = {
            "cache/model_catalog.json": False,
            "cache/chrome-debug/Default/Cookies": False,
            "profiles/sage/cache/chrome-debug/cache.db": False,
            "cache/images/x.png": True,
            "cache/citations/ledger.json": True,
            "profiles/sage/cache/images/y.png": True,
            "skills/example/cache/state.db": True,
        }
        for rel in files:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"x")

        skipped: set = set()
        selected = {str(rel) for _, rel in _iter_backup_files(root, tmp_path / "out.zip", skipped)}

        assert {rel for rel, keep in files.items() if keep} == {s.replace(os.sep, "/") for s in selected}
        assert str(Path("cache/chrome-debug")) in skipped
        assert str(Path("profiles/sage/cache/chrome-debug")) in skipped
        assert "cache" not in skipped

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

    @pytest.mark.linux_only
    def test_skips_unix_sockets(self, tmp_path, monkeypatch):
        from hermes_cli.backup import _iter_backup_files

        root = tmp_path / ".hermes"
        root.mkdir()
        # AF_UNIX paths are capped at ~108 bytes; pytest's tmp_path overflows that under the
        # test runner's deep temp root, so bind by a relative name from inside ``root``.
        monkeypatch.chdir(root)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as gateway_socket:
            gateway_socket.bind("gateway.sock")

            selected = {str(rel) for _, rel in _iter_backup_files(root, tmp_path / "out.zip")}

        assert "gateway.sock" not in selected


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

        assert (hermes_home / "config.yaml").read_text() == "model: test\n"







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

    def test_incomplete_archive_is_kept_but_reported_as_failure(self, tmp_path, monkeypatch, capsys):
        """A file that cannot be read is skipped, the zip still lands, and the CLI exits 1: a
        cron/systemd timer must never see a partial archive as success (#101096)."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        _make_hermes_tree(hermes_home)
        unreadable = hermes_home / "skills" / "locked.md"
        unreadable.write_text("secret\n")
        unreadable.chmod(0)
        if os.access(unreadable, os.R_OK):
            pytest.skip("running as root: chmod 0 does not make the file unreadable")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        from hermes_cli.backup import _RUN_BACKUP_PREFIX, run_backup
        from hermes_cli.main import cmd_backup

        out_dir = tmp_path / "b"
        out_dir.mkdir()
        good_old = out_dir / f"{_RUN_BACKUP_PREFIX}old.zip"
        good_old.write_bytes(b"PK")
        out_zip = out_dir / f"{_RUN_BACKUP_PREFIX}new.zip"

        assert run_backup(Namespace(output=str(out_zip), keep=1)) is False
        assert out_zip.exists()
        assert good_old.exists(), "an incomplete run must not rotate the last complete backup out"
        with pytest.raises(SystemExit) as exc:
            cmd_backup(Namespace(output=str(tmp_path / "out2.zip"), quick=False))
        assert exc.value.code == 1
        unreadable.chmod(0o600)
        assert run_backup(Namespace(output=str(tmp_path / "out4.zip"))) is True

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

        assert backup_mod._safe_copy_db(src, dst, timeout_seconds=1.0) is False
        assert connect_calls[0][1]["timeout"] == 0.0
        assert not dst.exists()




    def test_is_zeroed_sqlite_file_detects_nul_header(self, tmp_path):
        from hermes_cli.backup import is_zeroed_sqlite_file
        p = tmp_path / "state.db"
        p.write_bytes(bytes(4096))  # all NULs
        assert is_zeroed_sqlite_file(p) is True


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


# ---------------------------------------------------------------------------
# Live session database import (issue #100960)
# ---------------------------------------------------------------------------

def _write_session_db(path: Path, sessions: int, messages_per_session: int) -> None:
    """Create a minimal Hermes-shaped session database at *path*."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions "
            "(session_id TEXT PRIMARY KEY, message_count INTEGER)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages "
            "(id INTEGER PRIMARY KEY, session_id TEXT, content TEXT)"
        )
        for s in range(sessions):
            sid = f"sess-{s}"
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?)", (sid, messages_per_session)
            )
            for m in range(messages_per_session):
                conn.execute(
                    "INSERT INTO messages (session_id, content) VALUES (?, ?)",
                    (sid, f"{sid}-msg-{m}"),
                )
        conn.commit()
    finally:
        conn.close()


class TestImportLiveSessionDatabase:
    """`hermes import` must not swap the inode of a database Hermes holds open.

    Publishing state.db with a rename leaves any live gateway/dashboard/WebUI
    connection reading and writing the unlinked inode, so its sessions vanish
    from the database everyone else opens and nothing is logged (#100960).
    """

    def _zip_with_db(self, zip_path: Path, db_path: Path) -> None:
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(db_path, "state.db")

    def _prepare(self, tmp_path, monkeypatch, live=(3, 4), backup=(2, 2)):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        live_db = home / "state.db"
        _write_session_db(live_db, *live)

        staged = tmp_path / "backup-state.db"
        _write_session_db(staged, *backup)
        zip_path = tmp_path / "backup.zip"
        self._zip_with_db(zip_path, staged)
        return home, live_db, zip_path

    def test_live_holder_sees_imported_rows(self, tmp_path, monkeypatch):
        """A connection open across the import converges on the imported data."""
        from hermes_cli.backup import run_import

        home, live_db, zip_path = self._prepare(tmp_path, monkeypatch)

        holder = sqlite3.connect(str(live_db))
        # Read first so the connection has cached pages of the pre-import file.
        assert holder.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 12
        inode_before = os.stat(live_db).st_ino

        try:
            run_import(Namespace(zipfile=str(zip_path), force=True))
            assert holder.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4
        finally:
            holder.close()

        assert os.stat(live_db).st_ino == inode_before
        assert _count_rows(live_db) == (2, 4)



    def test_refused_restore_is_reported_and_leaves_db_intact(
        self, tmp_path, monkeypatch, capsys
    ):
        """A refused live-safe restore is a warning, not a counted success."""
        import hermes_cli.backup as backup_mod

        home, live_db, zip_path = self._prepare(tmp_path, monkeypatch)
        monkeypatch.setattr(backup_mod, "_safe_restore_db", lambda src, dst: False)

        backup_mod.run_import(Namespace(zipfile=str(zip_path), force=True))

        # The pre-import database is still the one on disk.
        assert _count_rows(live_db) == (3, 12)

    def test_sidecar_members_are_not_installed_beside_a_restored_db(
        self, tmp_path, monkeypatch
    ):
        """A `state.db-wal` member from an old/hand-built archive must not be
        os.replace'd next to the page-restored database: it describes a
        different image and SQLite would replay it on the next open."""
        from hermes_cli.backup import run_import

        home, live_db, zip_path = self._prepare(tmp_path, monkeypatch)
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.writestr("state.db-wal", b"foreign-wal-from-archive")
            zf.writestr("state.db-shm", b"foreign-shm")
            zf.writestr("state.db-journal", b"foreign-journal")

        run_import(Namespace(zipfile=str(zip_path), force=True))

        for suffix, payload in (
            ("-wal", b"foreign-wal-from-archive"),
            ("-shm", b"foreign-shm"),
            ("-journal", b"foreign-journal"),
        ):
            sidecar = live_db.with_name("state.db" + suffix)
            assert not sidecar.exists() or sidecar.read_bytes() != payload, suffix
        assert _count_rows(live_db) == (2, 4)

    def test_missing_target_takes_the_plain_publish(self, tmp_path, monkeypatch):
        """A fresh install has no inode to preserve; the member still lands."""
        from hermes_cli.backup import run_import

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        staged = tmp_path / "backup-state.db"
        _write_session_db(staged, 2, 3)
        zip_path = tmp_path / "backup.zip"
        self._zip_with_db(zip_path, staged)

        run_import(Namespace(zipfile=str(zip_path), force=True))
        assert _count_rows(home / "state.db") == (2, 6)


def _count_rows(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return (
            conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        )
    finally:
        conn.close()


def test_run_backup_prunes_older_default_named_zips_but_not_others(tmp_path, monkeypatch):
    """Hourly `hermes backup` callers accumulated 150+ zips; --keep bounds the default-named
    ones and leaves custom-named or foreign zips alone (#81317)."""
    from argparse import Namespace
    from hermes_cli import backup as backup_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: x\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for i in range(4):
        (tmp_path / f"hermes-backup-2026-01-0{i + 1}-000000.zip").write_bytes(b"old")
    (tmp_path / "my-archive.zip").write_bytes(b"mine")

    backup_mod.run_backup(Namespace(output=None, keep=2))

    kept = sorted(p.name for p in tmp_path.glob("hermes-backup-*.zip"))
    assert len(kept) == 2 and kept[0] == "hermes-backup-2026-01-04-000000.zip"
    assert (tmp_path / "my-archive.zip").exists()
