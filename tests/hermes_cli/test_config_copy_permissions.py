"""config.yaml copies made outside the config writer drop world access too (#59726).

Restores, imports, clones and distribution installs publish a config.yaml with copy2 or an
atomic extract that keeps the old/source mode, so each must narrow it the way the writer does.
"""

import os
import stat
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import hermes_constants

pytestmark = pytest.mark.platforms("posix")  # POSIX permission bits


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_MANAGED", "HERMES_CONTAINER", "HERMES_SKIP_CHMOD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_detect_container", lambda: False)
    (home / "config.yaml").write_text("model:\n  default: old\n", encoding="utf-8")
    os.chmod(home / "config.yaml", 0o666)
    return home


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_snapshot_restore(home):
    from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot
    snap_id = create_quick_snapshot(hermes_home=home)
    snap_config = home / "state-snapshots" / snap_id / "config.yaml"
    os.chmod(snap_config, 0o666)  # a snapshot taken while the file was wide

    assert restore_quick_snapshot(snap_id, hermes_home=home)

    assert _mode(home / "config.yaml") == 0o660


def test_backup_import(home, tmp_path, monkeypatch):
    from hermes_cli import backup
    monkeypatch.setattr(backup, "_revive_gateway_after_import", lambda _root: None)
    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config.yaml", "model:\n  default: imported\n")

    backup.run_import(Namespace(zipfile=str(archive), force=True))

    assert "imported" in (home / "config.yaml").read_text(encoding="utf-8")
    assert _mode(home / "config.yaml") == 0o660


def test_profile_clone(home):
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.profiles import create_profile
    # Current version: no migration rewrite (save_config) runs on the clone.
    (home / "config.yaml").write_text(f"_config_version: {DEFAULT_CONFIG['_config_version']}\n", encoding="utf-8")

    profile_dir = create_profile("coder", clone_config=True, no_alias=True)

    assert _mode(profile_dir / "config.yaml") == 0o660


def test_distribution_install(home, tmp_path):
    from hermes_cli.profile_distribution import DistributionManifest, install_distribution, write_manifest
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "config.yaml").write_text("model:\n  default: dist\n", encoding="utf-8")
    os.chmod(staged / "config.yaml", 0o666)
    write_manifest(staged, DistributionManifest(name="dist"))

    plan = install_distribution(str(staged), name="dist")

    assert _mode(plan.target_dir / "config.yaml") == 0o660


def test_snapshot_restore_env_is_owner_only(home):
    from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot
    (home / ".env").write_text("OPENAI_API_KEY=" + "sk-" + "fake\n", encoding="utf-8")
    snap_id = create_quick_snapshot(hermes_home=home)
    os.chmod(home / "state-snapshots" / snap_id / ".env", 0o666)

    assert restore_quick_snapshot(snap_id, hermes_home=home)

    assert _mode(home / ".env") == 0o600


def test_profile_import(home, tmp_path):
    import tarfile
    from hermes_cli.profiles import import_profile
    src = tmp_path / "src" / "evil"
    src.mkdir(parents=True)
    for name, body in (("config.yaml", "model:\n  default: x\n"), (".env", "A=" + "b\n")):
        (src / name).write_text(body, encoding="utf-8")
        os.chmod(src / name, 0o666)
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname="evil")

    profile_dir = import_profile(str(archive))

    assert _mode(profile_dir / "config.yaml") == 0o660
    assert _mode(profile_dir / ".env") == 0o600
