"""config.yaml backups: one dir, deduped, bounded — never a pile of siblings in HERMES_HOME."""
from pathlib import Path

import pytest

from hermes_cli.config_backups import backup_config, list_config_backups


@pytest.mark.parametrize("same_second", [False, True])
def test_repeat_backups_dedupe_and_rotate(tmp_path: Path, monkeypatch, same_second):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter("20260101-000000" if same_second else f"20260101-0000{i:02d}" for i in range(20))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    first = backup_config(cfg, "pre-setup", keep=2)
    assert first is not None and first.parent == tmp_path / "backups" / "config"
    # Backups made by an older Hermes release remain readable and participate in rotation.
    first.rename(first.parent / "config.yaml.pre-setup.20260101-000000")
    # Same bytes again → no new file (the hermes-setup-three-times case).
    assert backup_config(cfg, "pre-setup", keep=2) is None
    for i in range(12):
        cfg.write_text(f"model: {i}\n")
        assert backup_config(cfg, "pre-setup", keep=2) is not None
    kept = list_config_backups(cfg, "pre-setup")
    assert [p.read_text() for p in kept] == ["model: 11\n", "model: 10\n"]
    assert backup_config(cfg, "pre-setup", keep=2) is None
    # Nothing left beside config.yaml in the home root.
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.yaml"]


def test_legacy_siblings_move_but_user_named_copies_stay(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for name in ("config.yaml.bak.1778718391", "config.yaml.corrupt.20260729-093706.bak",
                 "config.yaml.bak-pre-migrate-xai-20260515-120000"):
        (tmp_path / name).write_text("old")
    (tmp_path / "config.yaml.bak-my-note").write_text("mine")

    backup_config(cfg, "corrupt")

    root = tmp_path / "backups" / "config"
    assert (root / "config.yaml.bak.1778718391").exists()
    assert (root / "config.yaml.corrupt.20260729-093706.bak").exists()
    assert (tmp_path / "config.yaml.bak-my-note").read_text() == "mine"
    assert not list(tmp_path.glob("config.yaml.bak.*")) and not list(tmp_path.glob("config.yaml.corrupt.*"))
