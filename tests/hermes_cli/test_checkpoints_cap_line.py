"""``hermes checkpoints prune`` tells a retained history apart from a failed cleanup.

The prune summary printed ``Errors: 1`` for a store the size cap cannot reach without deleting a
project's last retained checkpoint — the reason only reached ``logger.warning``, which has no
console handler on that path, so the CLI user saw a bare error count. The floor now has its own
line and its own wording.
"""
from __future__ import annotations

import argparse

import pytest

_EMPTY_STATUS = {"projects": [], "pre_v2_projects": []}


def _ns(**kwargs) -> argparse.Namespace:
    values = {"retention_days": 7, "max_size_mb": 200, "keep_orphans": False, "force": True}
    values.update(kwargs)
    return argparse.Namespace(**values)


def _patch_prune(monkeypatch, result: dict) -> None:
    import tools.checkpoint_maintenance as ckpt_mgr

    monkeypatch.setattr(ckpt_mgr, "store_status", lambda *a, **k: _EMPTY_STATUS)
    monkeypatch.setattr(ckpt_mgr, "prune_checkpoints", lambda **kwargs: dict(result))


@pytest.mark.parametrize("cap_not_met,printed", [(1, True), (0, False)], ids=["floor", "reached"])
def test_the_summary_names_a_retained_history(monkeypatch, capsys, cap_not_met, printed):
    import hermes_cli.checkpoints as checkpoints_cli

    _patch_prune(monkeypatch, {"scanned": 6, "deleted_orphan": 0, "deleted_stale": 0,
                              "cap_not_met": cap_not_met, "errors": 0, "bytes_freed": 0})

    assert checkpoints_cli.cmd_prune(_ns()) == 0

    out = capsys.readouterr().out
    assert "Errors:          0" in out
    if printed:
        assert "Cap not met:     1 (history retained)" in out
    else:
        assert "Cap not met:" not in out
