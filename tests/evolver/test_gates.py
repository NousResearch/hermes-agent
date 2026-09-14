"""Invariant tests for the three credit gates (evolver/gates.py).

Validity is tested against a real throwaway git repo; activation and
credit use injected seams (no subprocess, no network) so the gate logic —
not the harness — is what is under test.
"""
import subprocess
from pathlib import Path

import pytest

from evolver.gates import (CREDIT_MIN_PAIRS, Patch, activation_gate,
                           credit_gate, paired_bootstrap_ci, validity_gate)


@pytest.fixture()
def repo(tmp_path):
    """A minimal git repo with one committed Python file."""
    root = tmp_path / "repo"
    root.mkdir()
    env = {"GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root,
                   check=True, env=env)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True,
                   env=env)
    (root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True,
                   env=env)
    return root


def _diff_for(root: Path, rel: str, new_content: str) -> str:
    old = (root / rel).read_text(encoding="utf-8")
    import difflib
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{rel}", tofile=f"b/{rel}"))


def test_validity_accepts_clean_parseable_patch(repo):
    diff = _diff_for(repo, "mod.py", "VALUE = 2\n")
    res = validity_gate(Patch(diff=diff), repo)
    assert res.passed, res.detail
    assert res.detail["python_files_checked"] == 1


def test_validity_rejects_empty_diff(repo):
    res = validity_gate(Patch(diff=""), repo)
    assert not res.passed
    assert res.detail["reason"] == "empty_diff"


def test_validity_rejects_base_mismatch_unresolvable(repo):
    diff = _diff_for(repo, "mod.py", "VALUE = 2\n")
    res = validity_gate(
        Patch(diff=diff, base_sha="0" * 40), repo)
    assert not res.passed
    assert res.detail["reason"] == "base_mismatch"


def test_validity_rejects_base_mismatch_wrong_commit(repo):
    diff = _diff_for(repo, "mod.py", "VALUE = 2\n")
    (repo / "other.py").write_text("X = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "second"], cwd=repo, check=True)
    first = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=repo,
                           capture_output=True, text=True,
                           check=True).stdout.strip()
    res = validity_gate(Patch(diff=diff, base_sha=first), repo)
    assert not res.passed
    assert res.detail["reason"] == "base_mismatch"


def test_validity_accepts_matching_base_sha(repo):
    diff = _diff_for(repo, "mod.py", "VALUE = 2\n")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    res = validity_gate(Patch(diff=diff, base_sha=head), repo)
    assert res.passed, res.detail


def test_validity_rejects_non_applying_diff(repo):
    diff = ("--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n"
            "-WRONG_CONTEXT_LINE\n+VALUE = 2\n")
    res = validity_gate(Patch(diff=diff), repo)
    assert not res.passed
    assert res.detail["reason"] == "does_not_apply"


def test_validity_rejects_syntax_breaking_patch(repo):
    diff = _diff_for(repo, "mod.py", "VALUE = 1\ndef broken(:\n")
    res = validity_gate(Patch(diff=diff), repo)
    assert not res.passed
    assert res.detail["reason"] == "syntax_error"


def test_validity_ignores_non_python_files(repo):
    (repo / "notes.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "t"], cwd=repo, check=True)
    diff = ("--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-hi\n+hello\n")
    res = validity_gate(Patch(diff=diff), repo)
    assert res.passed
    assert res.detail["reason"] == "no_python_files_touched"


def _run(outcomes: dict):
    def run(checkout, _test_ref):
        return outcomes[str(checkout)]
    return run


def test_activation_passes_on_red_base_green_patched(tmp_path):
    base, patched = tmp_path / "base", tmp_path / "patched"
    res = activation_gate(Patch(diff="x"), _run({str(base): False,
                                                 str(patched): True}),
                          "t", base, patched)
    assert res.passed
    assert res.detail["red_on_base"] and res.detail["green_on_patched"]


def test_activation_fails_when_base_already_green(tmp_path):
    base, patched = tmp_path / "base", tmp_path / "patched"
    res = activation_gate(Patch(diff="x"), _run({str(base): True,
                                                 str(patched): True}),
                          "t", base, patched)
    assert not res.passed
    assert res.detail["reason"] == "no_repro_on_base"


def test_activation_fails_when_patch_does_not_fix(tmp_path):
    base, patched = tmp_path / "base", tmp_path / "patched"
    res = activation_gate(Patch(diff="x"), _run({str(base): False,
                                                 str(patched): False}),
                          "t", base, patched)
    assert not res.passed
    assert res.detail["reason"] == "not_fixed"


def test_activation_records_base_run_crash(tmp_path):
    def boom(checkout, _test_ref):
        raise RuntimeError("repro exploded")
    base, patched = tmp_path / "base", tmp_path / "patched"
    res = activation_gate(Patch(diff="x"), boom, "t", base, patched)
    assert not res.passed
    assert res.detail["reason"] == "base_run_crashed"
    assert "RuntimeError" in res.detail["error"]


def test_activation_records_patched_run_crash(tmp_path):
    def flaky(checkout, _test_ref):
        if str(checkout).endswith("patched"):
            raise RuntimeError("patched run exploded")
        return False  # red on base
    base, patched = tmp_path / "base", tmp_path / "patched"
    res = activation_gate(Patch(diff="x"), flaky, "t", base, patched)
    assert not res.passed
    assert res.detail["reason"] == "patched_run_crashed"
    assert res.detail["red_on_base"] is True


def test_credit_grants_on_clear_lift():
    res = credit_gate([0.5] * 8, [1.5] * 8)
    assert res.passed, res.detail
    assert res.detail["ci_lo"] > 0
    assert res.detail["mean_delta"] > 0


def test_credit_denies_on_noise():
    res = credit_gate([0.0, 1.0] * 4, [1.0, 0.0] * 4)
    assert not res.passed
    assert res.detail["reason"] == "no_significant_lift"
    assert res.detail["ci_lo"] <= 0 <= res.detail["ci_hi"]


def test_credit_denies_on_negative_lift():
    res = credit_gate([1.5] * 8, [0.5] * 8)
    assert not res.passed
    assert res.detail["ci_hi"] < 0


def test_credit_fails_closed_on_thin_data():
    res = credit_gate([0.0, 0.0], [1.0, 1.0])
    assert not res.passed
    assert res.detail["reason"] == "insufficient_data"


def test_credit_fails_closed_on_unpaired_data():
    res = credit_gate([0.0] * 5, [1.0] * 6)
    assert not res.passed
    assert res.detail["reason"] == "insufficient_data"


def test_credit_boundary_min_pairs():
    # Exactly CREDIT_MIN_PAIRS pairs is a measurement; one fewer is a guess.
    n = CREDIT_MIN_PAIRS
    res = credit_gate([0.5] * n, [1.5] * n)
    assert res.passed, res.detail
    assert res.detail["n"] == n
    res = credit_gate([0.5] * (n - 1), [1.5] * (n - 1))
    assert not res.passed
    assert res.detail["reason"] == "insufficient_data"


def test_bootstrap_ci_is_deterministic():
    a = paired_bootstrap_ci([1.0] * 6, [2.0] * 6)
    b = paired_bootstrap_ci([1.0] * 6, [2.0] * 6)
    assert a == b
