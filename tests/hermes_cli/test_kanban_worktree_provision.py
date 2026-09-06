"""Worktree provisioning — nested toolchains and tenant-declared data (2026-09-07).

Why this exists
---------------
`_provision_worktree_toolchain` linked only TOP-LEVEL `venv`/`.venv`/`node_modules`.
BackupBrain's install is at `frontend/node_modules` and its 69MB corpus at the
gitignored `data/backupbrain.db`, so every fresh worktree was unable to build the
UI or answer a single query against real data. Two cards on 2026-09-07 had to be
hand-provisioned before they could check their own acceptance criteria — the exact
failure the toolchain links were added to prevent.

Every test here has a paired negative: a control at the end neuters the new
behaviour and asserts these go red, because a provisioning helper that silently
does nothing looks identical to one that works.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    (r / "frontend").mkdir(parents=True)
    (r / "data").mkdir()
    (r / "venv" / "bin").mkdir(parents=True)
    (r / "node_modules" / "pkg").mkdir(parents=True)
    (r / "frontend" / "node_modules" / "react").mkdir(parents=True)
    (r / "data" / "corpus.db").write_text("not really sqlite")
    return r


@pytest.fixture
def worktree(tmp_path):
    w = tmp_path / "proj" / ".worktrees" / "t_abc123"
    w.mkdir(parents=True)
    return w


def _tenant_map(tmp_path, monkeypatch, repo, provision):
    p = tmp_path / "kanban-tenants.json"
    p.write_text(json.dumps({
        "backupbrain": {
            "id": "p_1", "slug": "backupbrain", "name": "BackupBrain",
            "primary_path": str(repo),
            "provision": provision,
        }
    }))
    monkeypatch.setenv("HERMES_KANBAN_TENANTS", str(p))
    return p


# ── top-level dirs: the 2026-09-03 behaviour must not regress ────────


def test_top_level_toolchain_still_linked(repo, worktree, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TENANTS", raising=False)
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert "venv" in linked and "node_modules" in linked
    assert (worktree / "venv").is_symlink()
    assert (worktree / "node_modules" / "pkg").is_dir()


def test_existing_entry_is_never_clobbered(repo, worktree, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TENANTS", raising=False)
    (worktree / "venv").mkdir()
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert "venv" not in linked
    assert not (worktree / "venv").is_symlink()  # the real dir survived


# ── the actual 09-07 defect: one level down ──────────────────────────


def test_nested_node_modules_is_linked(repo, worktree, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TENANTS", raising=False)
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert "frontend/node_modules" in linked
    assert (worktree / "frontend" / "node_modules" / "react").is_dir()


def test_nested_scan_skips_dot_dirs_and_worktrees(repo, worktree, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TENANTS", raising=False)
    (repo / ".cache" / "node_modules").mkdir(parents=True)
    (repo / ".worktrees" / "other" / "node_modules").mkdir(parents=True)
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert not any(x.startswith(".cache/") for x in linked)
    assert not any(x.startswith(".worktrees/") for x in linked)


# ── tenant-declared data ─────────────────────────────────────────────


def test_tenant_provision_links_gitignored_data(repo, worktree, tmp_path, monkeypatch):
    _tenant_map(tmp_path, monkeypatch, repo, ["data/corpus.db"])
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert "data/corpus.db" in linked
    dst = worktree / "data" / "corpus.db"
    assert dst.is_symlink() and dst.read_text() == "not really sqlite"


def test_tenant_provision_creates_missing_parent_dirs(repo, worktree, tmp_path, monkeypatch):
    (repo / "deep" / "nest").mkdir(parents=True)
    (repo / "deep" / "nest" / "blob.bin").write_text("x")
    _tenant_map(tmp_path, monkeypatch, repo, ["deep/nest/blob.bin"])
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert "deep/nest/blob.bin" in linked
    assert (worktree / "deep" / "nest" / "blob.bin").is_symlink()


def test_tenant_provision_entry_absent_from_repo_is_skipped(repo, worktree, tmp_path, monkeypatch):
    _tenant_map(tmp_path, monkeypatch, repo, ["data/does-not-exist.db"])
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert "data/does-not-exist.db" not in linked
    assert not (worktree / "data" / "does-not-exist.db").exists()


@pytest.mark.parametrize("evil", ["../../../etc/passwd", "/etc/passwd", "data/../../escape"])
def test_tenant_provision_refuses_paths_that_escape_the_repo(
    repo, worktree, tmp_path, monkeypatch, evil
):
    """Entries are config data, i.e. untrusted. None of these may be linked."""
    _tenant_map(tmp_path, monkeypatch, repo, [evil])
    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert evil not in linked
    assert not (worktree / "etc").exists()


def test_unknown_repo_root_gets_no_tenant_entries(repo, worktree, tmp_path, monkeypatch):
    """The map is looked up by primary_path — a different repo must match nothing."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    _tenant_map(tmp_path, monkeypatch, other, ["data/corpus.db"])
    assert kb._tenant_provision_paths(repo) == []


def test_malformed_tenant_map_fails_open(repo, worktree, tmp_path, monkeypatch):
    p = tmp_path / "kanban-tenants.json"
    p.write_text("{ this is not json")
    monkeypatch.setenv("HERMES_KANBAN_TENANTS", str(p))
    assert kb._tenant_provision_paths(repo) == []
    # And the toolchain half still works — a bad map must not disarm everything.
    assert "venv" in kb._provision_worktree_toolchain(repo, worktree)


def test_provision_not_a_list_fails_open(repo, tmp_path, monkeypatch):
    _tenant_map(tmp_path, monkeypatch, repo, "data/corpus.db")  # a string, not a list
    assert kb._tenant_provision_paths(repo) == []


# ── negative control ─────────────────────────────────────────────────


def test_negative_control_neutered_helper_turns_these_red(
    repo, worktree, tmp_path, monkeypatch
):
    """Proof the tests above measure something.

    Replace the nested scan and the tenant lookup with no-ops and assert the two
    behaviours this change added disappear. Without this, a helper that silently
    returned early would pass every assertion above that only checks `not in`.
    """
    _tenant_map(tmp_path, monkeypatch, repo, ["data/corpus.db"])
    monkeypatch.setattr(kb, "_tenant_provision_paths", lambda _root: [])
    monkeypatch.setattr(kb, "_WORKTREE_NESTED_MAX_DIRS", 0)

    linked = kb._provision_worktree_toolchain(repo, worktree)
    assert "frontend/node_modules" not in linked   # nested scan disabled
    assert "data/corpus.db" not in linked          # tenant lookup disabled
    assert "venv" in linked                        # legacy path still alive
