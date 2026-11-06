"""``*.lock`` runtime locks: left out of backups, skipped on import; dependency lockfiles stay."""

import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import hermes_cli.gateway_setup_service as service_setup


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    """run_import() may start an existing gateway service post-restore; never the host's."""
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(service_setup, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def _make_backup_zip(zip_path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


class TestShouldExcludeRuntimeLocks:
    def test_excludes_runtime_lock_files(self):
        """A ``*.lock`` file is a live process's lock (auth.lock, runtime/active_sessions.lock,
        gateway.lock, a scope lock). On Windows an ``msvcrt.locking`` owner makes it unreadable,
        so archiving it marks an otherwise complete backup incomplete; on another machine it
        names a PID that does not exist. Import skips the same files, for archives made before
        this exclusion."""
        from hermes_cli.backup import _should_exclude
        assert _should_exclude(Path("auth.lock"))
        assert _should_exclude(Path("gateway.lock"))
        assert _should_exclude(Path("runtime/active_sessions.lock"))
        assert _should_exclude(Path("profiles/coder/auth.lock"))

    def test_keeps_package_manager_lockfiles(self):
        """A dependency lockfile in a skill or plugin is user data, not a process lock."""
        from hermes_cli.backup import _should_exclude
        assert not _should_exclude(Path("skills/x/uv.lock"))
        assert not _should_exclude(Path("plugins/y/poetry.lock"))
        assert not _should_exclude(Path("skills/z/Cargo.lock"))
        assert not _should_exclude(Path("scratch/yarn.lock"))
        assert not _should_exclude(Path("skills/w/pubspec.lock"))
        assert not _should_exclude(Path("skills/v/mix.lock"))
        assert not _should_exclude(Path("skills/r/renv.lock"))
        assert not _should_exclude(Path("skills/c/conan.lock"))
        assert not _should_exclude(Path("skills/p/requirements.lock"))


class TestImportRuntimeLocks:
    def test_older_archive_runtime_locks_are_not_restored(self, tmp_path, monkeypatch):
        """An archive made before ``*.lock`` files were excluded from backups still carries them.
        Import must not land those foreign process locks; a dependency lockfile is user data
        and is restored."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        zip_path = tmp_path / "backup.zip"
        _make_backup_zip(zip_path, {
            "config.yaml": "model: test\n",
            "auth.lock": "7777",
            "cron/.tick.lock": "7777",
            "runtime/active_sessions.lock": "7777",
            "profiles/coder/auth.lock": "7777",
            "skills/demo/uv.lock": "version = 1\n",
        })

        from hermes_cli.backup import run_import
        run_import(Namespace(zipfile=str(zip_path), force=True))

        for rel in ("auth.lock", "cron/.tick.lock", "runtime/active_sessions.lock",
                    "profiles/coder/auth.lock"):
            assert not (hermes_home / rel).exists(), rel
        assert (hermes_home / "skills" / "demo" / "uv.lock").read_text() == "version = 1\n"
