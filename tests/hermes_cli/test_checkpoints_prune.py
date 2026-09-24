"""Tests for `hermes checkpoints prune`'s orphan confirmation flow.

Covers the P1 raised on PR #69141: the confirmation preview must cover
BOTH v2 projects (`store_status()["projects"]`) and pre-v2 shadow repos
(`store_status()["pre_v2_projects"]`), since `prune_checkpoints()` deletes
orphans from both layouts. Exercises decline / accept / --force across
pre-v2-only and mixed (v2 + pre-v2) stores.
"""

from __future__ import annotations

import argparse
import shutil

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


# ─── #109787: an unprovable orphan is not advertised as deletable ──────────


def _seed_deleted_workdir(tmp_path, monkeypatch, name, *, provable: bool):
    """Register a checkpoint project, then delete its workdir.

    ``provable=False`` leaves the parent directory empty and not a mount point —
    the #109787 shape, indistinguishable from a detached volume. ``provable=True``
    keeps a sibling entry, which positively witnesses the deletion.
    """
    import tools.checkpoint_manager as ckpt_mgr
    from tools.checkpoint_manager import CheckpointManager

    base = tmp_path / "checkpoints"
    monkeypatch.setattr(ckpt_mgr, "CHECKPOINT_BASE", base)
    parent = tmp_path / f"workspaces-{name}"
    work_dir = parent / name
    work_dir.mkdir(parents=True)
    (work_dir / "main.py").write_text("print('x')\n")
    if provable:
        (parent / "sibling-project").mkdir()
    assert CheckpointManager(enabled=True).ensure_checkpoint(str(work_dir), "initial") is True
    shutil.rmtree(work_dir)
    assert not work_dir.exists()
    return base, work_dir


def _project_meta(base, work_dir):
    from tools.checkpoint_manager import _project_hash, _project_meta_path, _store_path

    return _project_meta_path(_store_path(base), _project_hash(str(work_dir)))


def test_status_marks_unprovable_orphan_ambiguous(tmp_path, monkeypatch, capsys):
    """#109787: `checkpoints status` reported `orphan` for an entry prune will never delete."""
    import hermes_cli.checkpoints as checkpoints_cli

    _seed_deleted_workdir(tmp_path, monkeypatch, "checkpoint-mount-lab", provable=False)

    rc = checkpoints_cli.cmd_status(_ns(limit=100))

    out = capsys.readouterr().out
    assert rc == 0
    row = next(line for line in out.splitlines() if "checkpoint-mount-lab" in line)
    assert row.split()[-1] == "ambiguous", row


@pytest.mark.parametrize("force", [False, True], ids=["interactive", "force"])
def test_prune_explains_why_unprovable_orphan_is_kept(tmp_path, monkeypatch, capsys, force):
    """#109787: `Deleted orphan: 0` must come with a reason — prompt included or `--force`."""
    import hermes_cli.checkpoints as checkpoints_cli

    base, work_dir = _seed_deleted_workdir(
        tmp_path, monkeypatch, "checkpoint-mount-lab", provable=False,
    )
    meta = _project_meta(base, work_dir)

    def _unexpected_input(_prompt):
        raise AssertionError(
            "nothing is provably deletable — prune must not ask to confirm a deletion that cannot happen"
        )

    monkeypatch.setattr("builtins.input", _unexpected_input)

    rc = checkpoints_cli.cmd_prune(_ns(retention_days=0, max_size_mb=0, force=force))

    out = capsys.readouterr().out
    assert rc == 0
    assert "Deleted orphan:  0" in out
    assert "ambiguous" in out
    assert "not positively observed" in out
    assert meta.exists(), "history was deleted for a merely-unreachable project"


def test_prune_deletes_provable_orphan_and_names_the_kept_one(tmp_path, monkeypatch, capsys):
    """One store, both shapes: the provable orphan is deleted, the unprovable one is
    kept *and explained* — the two definitions of "orphan" agree within a single run."""
    import hermes_cli.checkpoints as checkpoints_cli

    base, provable = _seed_deleted_workdir(tmp_path, monkeypatch, "provably-gone", provable=True)
    _, ambiguous = _seed_deleted_workdir(tmp_path, monkeypatch, "ambiguous-gone", provable=False)
    provable_meta = _project_meta(base, provable)
    ambiguous_meta = _project_meta(base, ambiguous)
    assert provable_meta.exists() and ambiguous_meta.exists()

    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    rc = checkpoints_cli.cmd_prune(_ns(retention_days=0, max_size_mb=0))

    out = capsys.readouterr().out
    assert rc == 0
    assert "will permanently delete 1 orphan checkpoint project(s)" in out
    assert "ambiguous-gone" in out  # the kept project is named, not just counted
    assert "Deleted orphan:  1" in out
    assert "Retained (ambiguous): 1" in out
    assert not provable_meta.exists(), "a provably deleted workdir must still be pruned"
    assert ambiguous_meta.exists(), "an unprovable orphan must survive"




