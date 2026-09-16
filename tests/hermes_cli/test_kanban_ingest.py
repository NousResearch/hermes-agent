"""Isolation ingest gate: card-class matching and the .git-free diff path.

Regression coverage for adversarial-review findings:
- ``isolation_required`` must match ``card_class="single_blind"``, the only
  value ``BlindCard`` ever sets (the underscore/hyphen mismatch previously
  let BlindCard tasks skip the isolation gate entirely).
- ``_diff`` must be able to diff a projected (``.git``-free) workspace using
  its build-time manifest instead of assuming a git checkout.
- that manifest must live outside the worker-controlled workspace, or a
  worker could tamper with a file and its recorded hash together.
- a worker-introduced non-regular file (FIFO, device symlink) must never be
  opened for hashing, or the diff can hang forever instead of quarantining.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.kanban_ceiling import RoleCeiling, Scope, resolve_scope
from hermes_cli.kanban_ingest import _diff, _diff_projected, isolation_required
from hermes_cli.kanban_projection import build_projection


def test_isolation_required_matches_blind_card_canonical_class():
    assert isolation_required(SimpleNamespace(role=None, card_class="single_blind"))


def test_isolation_required_ignores_unknown_class():
    assert not isolation_required(SimpleNamespace(role=None, card_class="unscoped"))


def _write_manifest_sidecar(workspace: Path, manifest: dict) -> None:
    sidecar = workspace.parent / f".{workspace.name}.projection-manifest.json"
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")


def _build_projected_workspace(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "allowed.txt").write_text("v1")
    (repo / "other.txt").write_text("o1")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo, check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    scope = resolve_scope(
        SimpleNamespace(scope_paths=["allowed.txt", "other.txt"]),
        RoleCeiling(frozenset({"allowed.txt", "other.txt"}), frozenset({"allowed.txt", "other.txt"})),
        repo=repo, base_sha=sha,
    )
    dest = tmp_path / "workspace"
    handle = build_projection(sha, scope, dest, repo=repo)
    # Mirrors what kanban_db_dispatch._projected_workspace does: the manifest
    # sidecar lives next to, never inside, the workspace it baselines.
    _write_manifest_sidecar(dest, handle.manifest)
    return dest


def test_diff_projected_clean_workspace_has_no_changes(tmp_path):
    workspace = _build_projected_workspace(tmp_path)
    ok, paths, reason = _diff_projected(workspace)
    assert ok and paths == () and reason == "ok"


def test_diff_projected_detects_edit_add_and_delete(tmp_path):
    workspace = _build_projected_workspace(tmp_path)
    (workspace / "allowed.txt").write_text("v2")  # edit
    (workspace / "other.txt").unlink()  # delete
    (workspace / "new.txt").write_text("new")  # add
    ok, paths, reason = _diff_projected(workspace)
    assert ok and reason == "ok"
    assert set(paths) == {"allowed.txt", "other.txt", "new.txt"}


def test_diff_rejects_projected_workspace_missing_manifest(tmp_path):
    workspace = tmp_path / "no_manifest"
    workspace.mkdir()
    (workspace / "f.txt").write_text("x")
    task = SimpleNamespace(workspace_path=str(workspace), scope_manifest=json.dumps({"read": [], "write": ["*"]}))
    ok, paths, reason = _diff(task)
    assert not ok
    assert "manifest" in reason


def test_diff_projected_enforces_scope_manifest_write(tmp_path):
    workspace = _build_projected_workspace(tmp_path)
    (workspace / "other.txt").write_text("edited")
    task = SimpleNamespace(
        workspace_path=str(workspace),
        scope_manifest=json.dumps({"read": ["allowed.txt", "other.txt"], "write": ["allowed.txt"]}),
    )
    ok, paths, reason = _diff(task)
    assert not ok
    assert "other.txt" in reason


def test_diff_projected_manifest_lives_outside_workspace_so_worker_cannot_forge_it(tmp_path):
    """A worker with full control of ``workspace`` cannot hide a tamper by also
    rewriting an in-workspace manifest, because the manifest the diff trusts
    is not inside the workspace at all."""
    workspace = _build_projected_workspace(tmp_path)
    (workspace / "allowed.txt").write_text("tampered")
    # Even if the worker plants a forged manifest inside its own workspace
    # claiming everything still matches, the diff never reads it from there.
    (workspace / ".hermes_projection_manifest.json").write_text(
        json.dumps({"allowed.txt": "not-a-real-hash", "other.txt": "not-a-real-hash"}), encoding="utf-8",
    )
    ok, paths, reason = _diff_projected(workspace)
    assert ok and reason == "ok"
    # Both the tampered file and the forged manifest itself show up as
    # changes — the forged manifest is just another unauthorized file to the
    # diff, since it was never part of the baseline it's compared against.
    assert set(paths) == {"allowed.txt", ".hermes_projection_manifest.json"}


def test_diff_projected_never_opens_a_non_regular_file(tmp_path):
    """A worker-created FIFO must be flagged as changed, not opened for
    hashing — opening a FIFO with nothing writing to it blocks forever."""
    workspace = _build_projected_workspace(tmp_path)
    fifo_path = workspace / "pipe"
    os.mkfifo(fifo_path)
    ok, paths, reason = _diff_projected(workspace)
    assert ok and reason == "ok"
    assert "pipe" in paths
