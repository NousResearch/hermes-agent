"""Tests for `hermes checkpoints prune`'s orphan confirmation flow.

Covers the P1 raised on PR #69141: the confirmation preview must cover
BOTH v2 projects (`store_status()["projects"]`) and pre-v2 shadow repos
(`store_status()["pre_v2_projects"]`), since `prune_checkpoints()` deletes
orphans from both layouts. Exercises decline / accept / --force across
pre-v2-only and mixed (v2 + pre-v2) stores.
"""

from __future__ import annotations

import argparse

import pytest


def _ns(**kwargs) -> argparse.Namespace:
    defaults = {"retention_days": 7, "max_size_mb": 500, "keep_orphans": False, "force": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _prune_result(**kwargs) -> dict:
    result = {"scanned": 0, "deleted_orphan": 0, "deleted_stale": 0, "errors": 0, "bytes_freed": 0}
    result.update(kwargs)
    return result


_V2_ORPHAN_ONLY_STATUS = {
    "projects": [],
    "pre_v2_projects": [],
}

_PRE_V2_ONLY_STATUS = {
    "projects": [],
    "pre_v2_projects": [
        {"path": "/home/user/.hermes/checkpoints/deadbeefcafebabe", "workdir": None, "exists": False},
    ],
}

_MIXED_STATUS = {
    "projects": [
        {"hash": "abc123", "workdir": "/gone/v2-project", "exists": False, "commits": 4},
    ],
    "pre_v2_projects": [
        {"path": "/home/user/.hermes/checkpoints/deadbeefcafebabe", "workdir": "/gone/pre-v2-project", "exists": False},
    ],
}


def _patch_checkpoint_manager(monkeypatch, status: dict, prune_calls: list):
    import tools.checkpoint_manager as ckpt_mgr

    monkeypatch.setattr(ckpt_mgr, "store_status", lambda *a, **k: status)

    def _fake_prune(**kwargs):
        prune_calls.append(kwargs)
        return _prune_result(
            deleted_orphan=len(status["projects"]) + len(status["pre_v2_projects"]),
        )

    monkeypatch.setattr(ckpt_mgr, "prune_checkpoints", _fake_prune)


# ─── pre-v2-only store ──────────────────────────────────────────────────────




# ─── mixed store (v2 + pre-v2) ──────────────────────────────────────────────




# ─── --keep-orphans skips the prompt entirely, on either layout ───────────


@pytest.mark.parametrize("status", [_PRE_V2_ONLY_STATUS, _MIXED_STATUS], ids=["pre_v2_only", "mixed"])
def test_keep_orphans_skips_prompt(monkeypatch, capsys, status):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, status, prune_calls)

    def _unexpected_input(_prompt):
        raise AssertionError("input() must not be called when --keep-orphans is passed")

    monkeypatch.setattr("builtins.input", _unexpected_input)

    rc = checkpoints_cli.cmd_prune(_ns(keep_orphans=True))

    assert rc == 0
    assert len(prune_calls) == 1
    assert prune_calls[0]["delete_orphans"] is False


# ─── no orphans present: never prompts even without --force ───────────────


# ─── allowlist binding: preview set == deletion set, even when empty ───────


def test_empty_preview_binds_empty_allowlist(monkeypatch, capsys):
    """Zero-orphan-preview timing regression (PR #69141 review).

    When the non-force preview shows zero orphans, no prompt runs — but the
    later rescan inside prune_checkpoints() may discover a project that
    became orphaned *after* the preview. That undisplayed, unconfirmed orphan
    must not be deletable: the allowlist passed down must be the exact
    (empty) displayed set, never the unrestricted None sentinel.
    """
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _V2_ORPHAN_ONLY_STATUS, prune_calls)

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 0
    assert len(prune_calls) == 1
    assert prune_calls[0]["orphan_allowlist"] == set()




