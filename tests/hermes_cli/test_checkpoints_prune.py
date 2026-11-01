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
        {"path": "/home/user/.hermes/checkpoints/deadbeefcafebabe", "workdir": None,
         "exists": False, "orphan_confirmable": True, "unconfirmable_reason": None},
    ],
}

_MIXED_STATUS = {
    "projects": [
        {"hash": "abc123", "workdir": "/gone/v2-project", "exists": False, "commits": 4,
         "orphan_confirmable": True, "unconfirmable_reason": None},
    ],
    "pre_v2_projects": [
        {"path": "/home/user/.hermes/checkpoints/deadbeefcafebabe", "workdir": "/gone/pre-v2-project",
         "exists": False, "orphan_confirmable": True, "unconfirmable_reason": None},
    ],
}

# A pre-v2 shadow repo whose HERMES_WORKDIR marker is blank: store_status
# reports it as NOT orphan_confirmable (the P2 invariant), so the preview
# must not list it and the allowlist must not bind it.
_BLANK_MARKER_PRE_V2_STATUS = {
    "projects": [],
    "pre_v2_projects": [
        {"path": "/home/user/.hermes/checkpoints/blankbeefcafebabe", "workdir": "",
         "exists": False, "orphan_confirmable": False,
         "unconfirmable_reason": "blank workdir recorded"},
    ],
}


def _patch_checkpoint_manager(monkeypatch, status: dict, prune_calls: list):
    import tools.checkpoint_maintenance as ckpt_mgr

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


# ─── P2 invariant: a blank/unreadable marker is neither previewed nor deleted ─


def test_blank_marker_pre_v2_not_previewed_nor_allowlisted(monkeypatch, capsys):
    """P2 fix on PR #147152: a pre-v2 shadow repo whose HERMES_WORKDIR marker
    is blank is reported by store_status() as orphan_confirmable=False, so the
    confirmation preview must not list it and the allowlist must not bind it —
    otherwise the preview would show an orphan that execution skips.
    """
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _BLANK_MARKER_PRE_V2_STATUS, prune_calls)

    def _unexpected_input(_prompt):
        raise AssertionError("input() must not be called: no orphans in preview")

    monkeypatch.setattr("builtins.input", _unexpected_input)

    rc = checkpoints_cli.cmd_prune(_ns())

    assert rc == 0
    assert len(prune_calls) == 1
    # The blank-marker repo is not an orphan: not previewed, not allowlisted.
    assert prune_calls[0]["orphan_allowlist"] == set()


# ─── confirmation preview names every orphan it will delete ────────────────

# A v2 orphan with no recorded workdir is still confirmable (and deleted), so
# the preview must name it by its project hash instead of "(unknown)".
_V2_ORPHAN_NO_WORKDIR_STATUS = {
    "projects": [
        {"hash": "deadbeefcafebabe", "workdir": None, "exists": False, "commits": 2,
         "orphan_confirmable": True, "unconfirmable_reason": None},
    ],
    "pre_v2_projects": [],
}


def test_pre_v2_orphan_preview_names_shadow_repo_path(monkeypatch, capsys):
    """A pre-v2 orphan with no recorded workdir is named by its shadow repo
    path in the destructive preview.

    "(unknown)" gives the human no way to tell several such rows apart, and no
    path to inspect before confirming — the shadow repo path is also exactly
    what the allowlist binds, so showing it keeps the preview auditable.
    """
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _PRE_V2_ONLY_STATUS, prune_calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = checkpoints_cli.cmd_prune(_ns())

    out = capsys.readouterr().out
    assert rc == 1
    assert "/home/user/.hermes/checkpoints/deadbeefcafebabe" in out
    assert "(unknown)" not in out
    assert prune_calls == []


def test_v2_orphan_preview_falls_back_to_project_hash(monkeypatch, capsys):
    """Same rule for v2 records: a confirmable orphan whose workdir is gone is
    named by its project hash, not "(unknown)"."""
    import hermes_cli.checkpoints as checkpoints_cli

    prune_calls: list = []
    _patch_checkpoint_manager(monkeypatch, _V2_ORPHAN_NO_WORKDIR_STATUS, prune_calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = checkpoints_cli.cmd_prune(_ns())

    out = capsys.readouterr().out
    assert rc == 1
    assert "checkpoint deadbeefcafebabe" in out
    assert "(unknown)" not in out
    assert prune_calls == []


def test_cmd_status_names_v2_row_by_hash_when_workdir_missing(monkeypatch, capsys):
    """Independent-review observation ②: `cmd_status` still printed
    "(unknown)" for a v2 row whose workdir is missing, while the pre-v2 section
    falls back to the shadow repo path — asymmetric. A v2 record's identity is
    its project hash, so the row names it `checkpoint <hash>` instead."""
    import hermes_cli.checkpoints as checkpoints_cli

    status = {
        "base": "/home/user/.hermes/checkpoints",
        "total_size_bytes": 0, "store_size_bytes": 0, "legacy_size_bytes": 0,
        "project_count": 1,
        "projects": [
            {"hash": "deadbeefcafebabe", "workdir": None, "exists": False,
             "commits": 3, "orphan_confirmable": True, "unconfirmable_reason": None,
             "last_touch": 0},
        ],
        "pre_v2_projects": [], "legacy_archives": [],
    }
    monkeypatch.setattr(
        "tools.checkpoint_maintenance.store_status", lambda *a, **k: status,
    )

    rc = checkpoints_cli.cmd_status(_ns())

    assert rc == 0
    out = capsys.readouterr().out
    assert "checkpoint deadbeefcafebabe" in out
    assert "(unknown)" not in out


def test_orphan_label_defensive_fallback():
    """Independent-review observation ①: a record with neither a workdir nor a
    recorded id gets a neutral "(no recorded identity)" placeholder — never the
    old misleading "(unknown)" that named nothing."""
    import hermes_cli.checkpoints as checkpoints_cli

    label = checkpoints_cli._orphan_label
    # Defensive: nothing to name.
    assert label({}) == "(no recorded identity)"
    assert label({"workdir": None}, pre_v2=True) == "(no recorded identity)"
    # Recorded workdir always wins.
    assert label({"workdir": "/w", "hash": "abc"}, pre_v2=False) == "/w"
    # v2 identity is the project hash; pre-v2 identity is the shadow repo path.
    assert label({"hash": "abc"}, pre_v2=False) == "checkpoint abc"
    assert label({"path": "/p", "hash": "abc"}, pre_v2=True) == "/p"


def test_orphan_verdict_layout_rules():
    """Independent-review observation ① (deepseek pass): the orphan predicate
    was two mirrored implementations (store_status vs prune_checkpoints) that
    could drift. It is now one shared _orphan_verdict; this pins both layouts'
    rules so the single implementation stays correct."""
    import tools.checkpoint_maintenance as cm

    v = cm._orphan_verdict
    # Pre-v2 (no parent-identity channel, blank workdir kept).
    assert v("", require_parent_identity=False, blank_workdir_confirmable=False) == (
        False, "blank workdir recorded")
    assert v(None, require_parent_identity=False, blank_workdir_confirmable=False) == (
        True, None)
    assert v("/w", marker_unreadable=True, require_parent_identity=False,
             blank_workdir_confirmable=False) == (
        False, "shadow repo marker is unreadable")
    # v2 shared store: blank/missing workdir is confirmable.
    assert v("", require_parent_identity=True, blank_workdir_confirmable=True) == (
        True, None)
    assert v(None, require_parent_identity=True, blank_workdir_confirmable=True) == (
        True, None)


def test_unreachable_reason_parent_gone_wording(tmp_path):
    """Independent-review observation ⑤ (deepseek pass): pre-v2 has no
    parent-identity metadata channel, so "no parent identity recorded" is a
    category error there — it gets the same conservative wording as v2, while
    a v2 record that *could* have recorded one keeps the specific message."""
    import tools.checkpoint_maintenance as cm

    workdir = str(tmp_path / "gone" / "proj")
    assert cm._workdir_unreachable_reason(
        workdir, require_parent_identity=False
    ) == "parent path gone; deletion not corroborated (conservative)"
    assert cm._workdir_unreachable_reason(
        workdir, require_parent_identity=True
    ) == "parent path gone and no parent identity recorded"


# ─── cmd_status surfaces pre-v2 shadow repos with their reason ─────────────


def test_cmd_status_lists_pre_v2_shadow_repos(monkeypatch, capsys):
    """Finding #3 on PR #147152: `checkpoints status` previously only walked
    `projects`, so a pre-v2 shadow repo's "unreachable but not prunable"
    reason was invisible in the CLI (it only existed in the store_status()
    API). The pre-v2 rows must now be displayed with the same classification
    the prune preview binds to, falling back to the shadow repo path when the
    workdir is blank/missing so the row still names the directory.
    """
    import hermes_cli.checkpoints as checkpoints_cli

    blank_path = "/home/user/.hermes/checkpoints/blankbeefcafebabe"
    orphan_path = "/home/user/.hermes/checkpoints/deadbeefcafebabe"
    status = {
        "base": "/home/user/.hermes/checkpoints",
        "total_size_bytes": 0,
        "store_size_bytes": 0,
        "legacy_size_bytes": 0,
        "project_count": 0,
        "projects": [],
        "pre_v2_projects": [
            {"path": blank_path, "workdir": "",
             "exists": False, "orphan_confirmable": False,
             "unconfirmable_reason": "blank workdir recorded"},
            {"path": orphan_path, "workdir": None,
             "exists": False, "orphan_confirmable": True, "unconfirmable_reason": None},
        ],
        "legacy_archives": [],
    }
    monkeypatch.setattr(
        "tools.checkpoint_maintenance.store_status", lambda *a, **k: status,
    )

    rc = checkpoints_cli.cmd_status(_ns())

    assert rc == 0
    out = capsys.readouterr().out
    # Pre-v2 section is headed and shows both rows.
    assert "Pre-v2 shadow repos (2):" in out
    # Blank workdir row: falls back to the shadow repo path (not "(unknown)"),
    # displays unreachable + reason + the "won't prune" note.
    assert blank_path in out
    assert "unreachable" in out
    assert "blank workdir recorded" in out
    assert "pre-v2 shadow repo; prune won't treat it as an orphan" in out
    # Orphan_confirmable row (workdir None): its own row shows STATE=orphan.
    # Match on the row line so the weak `"orphan" in out` substring (which the
    # "won't treat it as an orphan" note already satisfies) can't false-pass.
    orphan_line = next(
        line for line in out.splitlines()
        if orphan_path in line
    )
    assert orphan_line.rstrip().endswith("orphan")




