"""Config backups are secret-bearing copies: private mode at creation, legacy copies tightened.

A ``config.yaml`` commonly embeds live provider keys; on estate reports a fleet of
``config.yaml.good.*`` backups written 0644 (``shutil.copy2`` inheriting a 0644 source) put
those keys world-readable. The invariant is mode-at-creation (as ``auth.py`` keeps for token
files), plus a next-write repair for backups the old writer already stamped 0644.
"""
import os
import stat
from pathlib import Path

from hermes_cli.config_backups import backup_config


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_backup_created_private_even_from_world_readable_config(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("providers:\n  openai:\n    api_key: sk-livevalue\n")
    os.chmod(cfg, 0o644)  # the estate's actual pre-condition
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: "20261019-202800")

    dest = backup_config(cfg, "good")
    assert dest is not None
    assert _mode(dest) == 0o600, f"backup of a live-key config must not be group/world-readable, got {_mode(dest):#o}"


def test_backups_dir_created_private(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: "20261019-202801")

    backup_config(cfg, "good")
    assert _mode(tmp_path / "backups" / "config") == 0o700


def test_world_readable_legacy_backup_tightened_on_next_write(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(["20260101-000000", "20260102-000000"])
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    first = backup_config(cfg, "pre-setup")
    assert first is not None
    os.chmod(first, 0o644)  # simulate a copy the pre-fix writer stamped world-readable
    cfg.write_text("model: b\n")

    backup_config(cfg, "pre-setup")
    assert _mode(first) == 0o600, "a legacy world-readable copy must be tightened when the next backup is written"
