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


def test_pre_v2_only_decline_aborts_without_deleting(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _PRE_V2_ONLY_STATUS, prune_calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 1
    assert prune_calls == []
    out = capsys.readouterr().out
    assert "pre-v2 shadow repo" in out
    assert "Aborted" in out


def test_pre_v2_only_accept_deletes(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _PRE_V2_ONLY_STATUS, prune_calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 0
    assert len(prune_calls) == 1
    assert prune_calls[0]["delete_orphans"] is True


def test_pre_v2_only_force_skips_prompt(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _PRE_V2_ONLY_STATUS, prune_calls)

    def _unexpected_input(_prompt):
        raise AssertionError("input() must not be called when --force is passed")

    monkeypatch.setattr("builtins.input", _unexpected_input)

    rc = checkpoints_cli.cmd_prune(_ns(force=True))

    assert rc == 0
    assert len(prune_calls) == 1


# ─── mixed store (v2 + pre-v2) ──────────────────────────────────────────────


def test_mixed_store_decline_aborts_without_deleting(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _MIXED_STATUS, prune_calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 1
    assert prune_calls == []
    out = capsys.readouterr().out
    # Both layouts must appear in the preview, not just the v2 one.
    assert "/gone/v2-project" in out
    assert "/gone/pre-v2-project" in out
    assert "This will permanently delete 2 orphan checkpoint project(s)" in out


def test_mixed_store_accept_deletes_both_layouts(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _MIXED_STATUS, prune_calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 0
    assert len(prune_calls) == 1
    out = capsys.readouterr().out
    assert "Deleted orphan:  2" in out


def test_mixed_store_force_skips_prompt_deletes_both(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _MIXED_STATUS, prune_calls)

    def _unexpected_input(_prompt):
        raise AssertionError("input() must not be called when --force is passed")

    monkeypatch.setattr("builtins.input", _unexpected_input)

    rc = checkpoints_cli.cmd_prune(_ns(force=True))

    assert rc == 0
    assert len(prune_calls) == 1
    assert prune_calls[0]["delete_orphans"] is True


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


def test_no_orphans_skips_prompt(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _V2_ORPHAN_ONLY_STATUS, prune_calls)

    def _unexpected_input(_prompt):
        raise AssertionError("input() must not be called when there are no orphans")

    monkeypatch.setattr("builtins.input", _unexpected_input)

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 0
    assert len(prune_calls) == 1
