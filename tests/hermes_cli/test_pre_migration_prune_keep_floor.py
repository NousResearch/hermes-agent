"""keep<=0 must not wipe every pre-migration backup during prune."""

from __future__ import annotations

from hermes_cli.backup import _PRE_MIGRATION_PREFIX, _prune_pre_migration_backups


def _seed(backup_dir, names):
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (backup_dir / name).write_bytes(b"zip")


def test_prune_pre_migration_keep_zero_preserves_newest(tmp_path):
    root = tmp_path / "backups"
    _seed(
        root,
        [
            f"{_PRE_MIGRATION_PREFIX}20260101T000003.zip",
            f"{_PRE_MIGRATION_PREFIX}20260101T000002.zip",
            f"{_PRE_MIGRATION_PREFIX}20260101T000001.zip",
            "hand-made.zip",
        ],
    )

    deleted = _prune_pre_migration_backups(root, keep=0)

    remaining = sorted(p.name for p in root.iterdir())
    assert deleted == 2
    assert remaining == [
        "hand-made.zip",
        f"{_PRE_MIGRATION_PREFIX}20260101T000003.zip",
    ]


def test_prune_pre_migration_negative_keep_floors_to_one(tmp_path):
    root = tmp_path / "backups"
    _seed(
        root,
        [
            f"{_PRE_MIGRATION_PREFIX}20260101T000002.zip",
            f"{_PRE_MIGRATION_PREFIX}20260101T000001.zip",
        ],
    )

    deleted = _prune_pre_migration_backups(root, keep=-3)

    remaining = sorted(p.name for p in root.iterdir())
    assert deleted == 1
    assert remaining == [f"{_PRE_MIGRATION_PREFIX}20260101T000002.zip"]


def test_prune_pre_migration_invalid_keep_uses_default(tmp_path):
    root = tmp_path / "backups"
    names = [f"{_PRE_MIGRATION_PREFIX}20260101T{i:06d}.zip" for i in range(8)]
    _seed(root, names)

    deleted = _prune_pre_migration_backups(root, keep="nope")

    remaining = [p for p in root.iterdir() if p.name.startswith(_PRE_MIGRATION_PREFIX)]
    assert len(remaining) == 5  # _PRE_MIGRATION_DEFAULT_KEEP
    assert deleted == 3
