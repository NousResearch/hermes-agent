"""Clamp list_quick_snapshots pagination before scanning manifests."""

from __future__ import annotations

import json

from hermes_cli.backup import list_quick_snapshots


def _seed_snapshots(home, n: int) -> None:
    root = home / "state-snapshots"
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = root / f"20260101T{i:06d}Z"
        d.mkdir()
        (d / "manifest.json").write_text(
            json.dumps({"id": d.name, "file_count": 1, "total_size": 1}),
            encoding="utf-8",
        )


def test_list_quick_snapshots_clamps_zero_and_negative(tmp_path):
    _seed_snapshots(tmp_path, 5)
    assert len(list_quick_snapshots(limit=0, hermes_home=tmp_path)) == 1
    assert len(list_quick_snapshots(limit=-5, hermes_home=tmp_path)) == 1


def test_list_quick_snapshots_clamps_excessive_limit(tmp_path):
    _seed_snapshots(tmp_path, 520)
    rows = list_quick_snapshots(limit=10_000_000, hermes_home=tmp_path)
    assert len(rows) == 500


def test_list_quick_snapshots_default_limit(tmp_path):
    _seed_snapshots(tmp_path, 30)
    assert len(list_quick_snapshots(hermes_home=tmp_path)) == 20


def test_list_quick_snapshots_invalid_limit_uses_default(tmp_path):
    _seed_snapshots(tmp_path, 25)
    assert len(list_quick_snapshots(limit="nope", hermes_home=tmp_path)) == 20
