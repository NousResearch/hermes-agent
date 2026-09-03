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
    result = {"scanned": 0, "deleted_orphan": 0, "deleted_stale": 0, "protected_unreachable": 0, "errors": 0, "bytes_freed": 0}
    result.update(kwargs)
    return result


_V2_ORPHAN_ONLY_STATUS = {
    "projects": [],
    "pre_v2_projects": [],
}

_PRE_V2_ONLY_STATUS = {
    "projects": [],
    "pre_v2_projects": [
        {"path": "/home/user/.hermes/checkpoints/deadbeefcafebabe", "workdir": None, "exists": False, "state": "orphan"},
    ],
}

_MIXED_STATUS = {
    "projects": [
        {"hash": "abc123", "workdir": "/gone/v2-project", "exists": False, "state": "orphan", "commits": 4},
    ],
    "pre_v2_projects": [
        {"path": "/home/user/.hermes/checkpoints/deadbeefcafebabe", "workdir": "/gone/pre-v2-project", "exists": False, "state": "orphan"},
    ],
}

_ORPHAN_AND_UNREACHABLE_STATUS = {
    "projects": [
        {
            "hash": "deletable",
            "workdir": "/gone/deletable",
            "exists": False,
            "state": "orphan",
            "commits": 2,
        },
        {
            "hash": "protected",
            "workdir": "/offline/protected",
            "exists": False,
            "state": "unreachable",
            "commits": 3,
        },
    ],
    "pre_v2_projects": [],
}


def _patch_checkpoint_manager(monkeypatch, status: dict, prune_calls: list):
    import tools.checkpoint_manager as ckpt_mgr

    monkeypatch.setattr(ckpt_mgr, "store_status", lambda *a, **k: status)

    def _fake_prune(**kwargs):
        prune_calls.append(kwargs)
        return _prune_result(
            deleted_orphan=sum(
                p.get("state") == "orphan"
                for p in status["projects"] + status["pre_v2_projects"]
            ),
            protected_unreachable=sum(
                p.get("state") == "unreachable"
                for p in status["projects"] + status["pre_v2_projects"]
            ),
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


def test_status_labels_protected_missing_workdir_as_unreachable(
    monkeypatch, capsys,
):
    import hermes_cli.checkpoints as checkpoints_cli
    import tools.checkpoint_manager as ckpt_mgr

    status = {
        **_ORPHAN_AND_UNREACHABLE_STATUS,
        "base": "/tmp/checkpoints",
        "total_size_bytes": 0,
        "store_size_bytes": 0,
        "legacy_size_bytes": 0,
        "project_count": 2,
        "legacy_archives": [],
    }
    monkeypatch.setattr(ckpt_mgr, "store_status", lambda: status)

    rc = checkpoints_cli.cmd_status(argparse.Namespace(limit=20))

    assert rc == 0
    output = capsys.readouterr().out
    assert "/offline/protected" in output
    assert "unreachable" in output


def test_force_reports_unreachable_projects_retained_for_safety(
    monkeypatch, capsys,
):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(
        monkeypatch, _ORPHAN_AND_UNREACHABLE_STATUS, prune_calls,
    )

    rc = checkpoints_cli.cmd_prune(_ns(force=True))

    assert rc == 0
    assert "Protected unreachable: 1" in capsys.readouterr().out


# ─── allowlist binding: preview set == deletion set, even when empty ───────


def test_preview_only_authorizes_deletable_orphans(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(
        monkeypatch, _ORPHAN_AND_UNREACHABLE_STATUS, prune_calls,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 0
    output = capsys.readouterr().out
    assert "permanently delete 1 orphan checkpoint project(s)" in output
    assert "/gone/deletable" in output
    assert "/offline/protected" not in output
    assert prune_calls[0]["orphan_allowlist"] == {"deletable"}


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




