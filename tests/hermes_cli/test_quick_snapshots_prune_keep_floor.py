"""keep<=0 must not wipe every quick snapshot during prune."""

from __future__ import annotations

from hermes_cli.backup import _prune_quick_snapshots, prune_quick_snapshots


def _seed(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).mkdir()


def test_prune_quick_snapshots_keep_zero_preserves_newest(tmp_path):
    root = tmp_path / "state-snapshots"
    _seed(root, ["20260101T000003Z", "20260101T000002Z", "20260101T000001Z"])

    deleted = _prune_quick_snapshots(root, keep=0)

    remaining = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert deleted == 2
    assert remaining == ["20260101T000003Z"]


def test_prune_quick_snapshots_negative_keep_floors_to_one(tmp_path):
    root = tmp_path / "state-snapshots"
    _seed(root, ["20260101T000003Z", "20260101T000002Z", "20260101T000001Z"])

    deleted = _prune_quick_snapshots(root, keep=-5)

    remaining = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert deleted == 2
    assert remaining == ["20260101T000003Z"]


def test_prune_quick_snapshots_invalid_keep_uses_default(tmp_path):
    root = tmp_path / "state-snapshots"
    names = [f"20260101T{i:06d}Z" for i in range(25)]
    _seed(root, names)

    deleted = _prune_quick_snapshots(root, keep="nope")

    remaining = [p for p in root.iterdir() if p.is_dir()]
    assert len(remaining) == 20  # _QUICK_DEFAULT_KEEP
    assert deleted == 5


def test_prune_quick_snapshots_public_wrapper_keep_zero(tmp_path):
    root = tmp_path / "state-snapshots"
    _seed(root, ["20260101T000002Z", "20260101T000001Z"])

    deleted = prune_quick_snapshots(keep=0, hermes_home=tmp_path)

    remaining = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert deleted == 1
    assert remaining == ["20260101T000002Z"]
