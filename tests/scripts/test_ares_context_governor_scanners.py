"""Independent tooling contracts use controlled fixture secrets only."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scanner = _load("scan_ares_context_governor_candidate")
scope = _load("verify_ares_context_governor_scope")


@pytest.mark.parametrize(
    "payload",
    [
        b"FIXTURE_SECRET_0123456789",
        b"464958545552455f5345435245545f30313233343536373839",
        b"RklYVFVSRV9TRUNSRVRfMDEyMzQ1Njc4OQ==",
    ],
)
def test_controlled_fixture_secret_encodings_are_detected(
    tmp_path: Path, payload: bytes
):
    surface = tmp_path / "evidence.log"
    surface.write_bytes(payload)
    with pytest.raises(scanner.SecretMaterialDetected):
        scanner.scan([surface], fixture_secrets=[b"FIXTURE_SECRET_0123456789"])


def test_secret_reports_never_echo_controlled_secret(tmp_path: Path):
    surface = tmp_path / "config.json"
    surface.write_bytes(b'{"token":"highentropyfixturevalue0123456789"}')
    with pytest.raises(scanner.SecretMaterialDetected) as raised:
        scanner.scan([surface])
    assert "highentropyfixturevalue0123456789" not in str(raised.value)


def test_scope_verifier_rejects_undeclared_hunk(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=repo, check=True)
    (repo / "source.txt").write_text("one\n")
    subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
    baseline = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    (repo / "source.txt").write_text("two\n")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "source.txt").write_text("two\n")
    tree = [
        {"path": "source.txt", "mode": 0o644, "size": 4, "sha256": scope.sha(b"two\n")}
    ]
    manifest = {
        "repositories": [
            {
                "name": "fixture",
                "root": str(repo),
                "baseline_commit": baseline,
                "base_blob_ids": scope._base_blob_ids(repo, baseline),
                "required_hunk_ids": [],
                "unrelated_hunk_ids": [],
                "required_paths": [],
                "candidate_subtree": ".",
            }
        ],
        "candidate_tree_sha256": scope.sha(
            json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()
        ),
    }
    with pytest.raises(scope.UndeclaredHunk):
        scope.verify(manifest, candidate)


def test_scope_verifier_replays_declared_patch_and_rejects_tree_swap(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=repo, check=True)
    (repo / "source.txt").write_text("one\n")
    subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
    baseline = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    (repo / "source.txt").write_text("two\n")
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "source.txt").write_text("two\n")
    hunk_ids = sorted(scope.normalized_hunks(scope.complete_patch(repo)))
    manifest = {
        "repositories": [
            {
                "name": "fixture",
                "root": str(repo),
                "baseline_commit": baseline,
                "base_blob_ids": scope._base_blob_ids(repo, baseline),
                "required_hunk_ids": hunk_ids,
                "unrelated_hunk_ids": [],
                "required_paths": ["source.txt"],
                "candidate_subtree": ".",
            }
        ],
        "candidate_tree_sha256": scope.tree_digest(candidate_root),
    }
    assert scope.verify(manifest, candidate_root)["pass"] is True
    (candidate_root / "source.txt").write_text("swapped\n")
    with pytest.raises(scope.CandidateTreeMismatch):
        scope.verify(manifest, candidate_root)
