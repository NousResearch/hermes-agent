"""Backup archives carry credentials: they must be owner-only regardless of the umask."""
import os
import stat
import zipfile

import pytest

from hermes_cli.backup import create_pre_update_backup
from hermes_cli.backup_atomic import _atomic_output_path


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.platforms("posix")
def test_partial_is_owner_only_before_first_byte_and_after_publish(tmp_path, umask_022):
    destination = tmp_path / "secret-backup.zip"
    with _atomic_output_path(destination) as partial_path:
        assert stat.S_IMODE(partial_path.stat().st_mode) == 0o600
        with zipfile.ZipFile(partial_path, "w") as zf:
            zf.writestr("auth.json", "credential-bearing-archive")
            assert stat.S_IMODE(partial_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


@pytest.mark.platforms("posix")
def test_partial_refuses_planted_symlink(tmp_path, umask_022, monkeypatch):
    destination = tmp_path / "secret-backup.zip"
    victim = tmp_path / "victim"
    real_open = os.open

    def plant_then_open(path, flags, mode=0o777, *args, **kwargs):
        if str(path).endswith(".partial"):
            os.symlink(victim, path)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", plant_then_open)
    with pytest.raises(FileExistsError):
        with _atomic_output_path(destination):
            pass
    monkeypatch.undo()
    assert not victim.exists()
    assert not destination.exists()


@pytest.mark.platforms("posix")
def test_pre_update_backup_owner_only_and_tightens_backups_dir(tmp_path, umask_022):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model:\n  provider: openrouter\n")
    (hermes_home / "auth.json").write_text('{"token": "sk-test"}\n')

    out = create_pre_update_backup(hermes_home=hermes_home)
    backup_dir = hermes_home / "backups"
    assert out is not None
    assert stat.S_IMODE(backup_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(out.stat().st_mode) == 0o600

    backup_dir.chmod(0o755)
    out.unlink()
    out2 = create_pre_update_backup(hermes_home=hermes_home)
    assert out2 is not None
    assert stat.S_IMODE(backup_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(out2.stat().st_mode) == 0o600
