"""Every dir Hermes creates for its own state must stay usable by the user that created it.

``hermes_dir_acl.make_user_dir`` is the one place Hermes state/lock/log directories are
created: an elevated Windows process that used a raw ``Path.mkdir(mode=0o700)`` leaves a
*protected* DACL naming only SYSTEM, Administrators and OWNER RIGHTS, and the same user's
non-elevated processes then get ``WinError 5`` inside that directory (#139845, #138872).
Three sites kept the raw call after the rollout: the quick-snapshot root and staging dir,
the shared-metrics private dir, and the Bot DM temp dir.

Elevation is a host fact the primitive takes as data (the grant selection is proved on a
POSIX host by ``test_hermes_dir_acl.py``; the ACL effect itself by
``test_hermes_dir_acl_windows.py`` on Windows CI), so each test below drives the production
call with ``process_is_elevated`` standing in for an elevated run and records the grant the
primitive requests. A raw ``mkdir`` requests none, which is the regression these prove.
"""
from pathlib import Path

import hermes_dir_acl


def _record_grants(monkeypatch) -> list[Path]:
    """Stand in for an elevated run: collect the dir objects the primitive grants."""
    granted: list[Path] = []
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "grant_user_access",
                        lambda path, **_kwargs: granted.append(Path(path)) or True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    return granted


def test_shared_metrics_private_dir_is_granted_to_the_creating_user(tmp_path, monkeypatch):
    from hermes_cli.observability import shared_metrics

    granted = _record_grants(monkeypatch)

    shared_metrics._ensure_private(tmp_path / "metrics", 0o700)

    assert granted == [tmp_path / "metrics"]


def test_bot_dm_temp_dir_is_granted_to_the_creating_user(tmp_path, monkeypatch):
    from tools import bot_mode_dm

    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    granted = _record_grants(monkeypatch)

    dm_dir = bot_mode_dm._dm_dir()

    assert granted == [dm_dir]


def test_quick_snapshot_dirs_are_granted_to_the_creating_user(tmp_path, monkeypatch):
    from hermes_cli import backup

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model: x\n", encoding="utf-8")
    granted = _record_grants(monkeypatch)

    snapshot_id = backup.create_quick_snapshot(hermes_home=home)

    assert snapshot_id
    assert backup._quick_snapshot_root(home) in granted
    staging = [path for path in granted if path.name.endswith(".partial")]
    assert len(staging) == 1, granted
