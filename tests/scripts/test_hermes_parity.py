from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path

from scripts.hermes_parity import bisect, buckets, cli, forkdelta, gates, gitops, lint_unbound, state


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def test_bucket_conflicts_groups_synthetic_unmerged_statuses(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "both.txt").write_text("base\n", encoding="utf-8")
    (repo / "ours_deleted.txt").write_text("base\n", encoding="utf-8")
    _commit(repo, "base")
    _git(repo, "checkout", "-b", "ours")
    (repo / "both.txt").write_text("ours\n", encoding="utf-8")
    (repo / "ours_deleted.txt").unlink()
    _commit(repo, "ours")
    _git(repo, "checkout", "-b", "theirs", "HEAD~1")
    (repo / "both.txt").write_text("theirs\n", encoding="utf-8")
    (repo / "ours_deleted.txt").write_text("theirs\n", encoding="utf-8")
    _commit(repo, "theirs")
    _git(repo, "checkout", "ours")
    _git(repo, "merge", "theirs", check=False)

    by_name = {bucket.name: bucket.files for bucket in buckets.bucket_conflicts(repo)}

    assert by_name["both_modified"] == ("both.txt",)
    assert by_name["deleted_by_us"] == ("ours_deleted.txt",)


def test_forkdelta_cross_references_manifest_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "tracked.py").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    _git(repo, "checkout", "-b", "fork-main")
    (repo / "tracked.py").write_text("fork\n", encoding="utf-8")
    (repo / "untracked_feature.py").write_text("fork\n", encoding="utf-8")
    _commit(repo, "fork")
    manifest = repo / "docs" / "sync" / "fork-features.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps([
            {
                "feature": "tracked feature",
                "tests": ["tests/example.py::test_contract"],
                "paths": ["tracked.py"],
                "why": "contract",
            }
        ]),
        encoding="utf-8",
    )

    report = forkdelta.compute_fork_delta(repo, base=base, fork_ref="fork-main")

    assert report.covered_paths == ("tracked.py",)
    assert report.uncovered_paths == ("untracked_feature.py",)
    assert report.covered_features == ("tracked feature",)


def test_unbound_linter_flags_name_after_only_binding_removed() -> None:
    issues = lint_unbound.lint_source(
        """
def run():
    return removed_name + 1
""".lstrip()
    )

    assert [(issue.name, issue.line) for issue in issues] == [("removed_name", 2)]


def test_unbound_linter_handles_common_binding_forms_without_false_positives() -> None:
    source = """
from somewhere import *

GLOBAL_VALUE = 1

def outer():
    local = 1
    values = [item for item in range(3)]
    if (named := local):
        pass
    def inner():
        nonlocal local
        global GLOBAL_VALUE
        return local + GLOBAL_VALUE + named + len(values)
    return inner()
""".lstrip()

    assert lint_unbound.lint_source(source) == []


def test_state_atomic_write_and_tree_sha_invalidation(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "file.txt").write_text("one\n", encoding="utf-8")
    _commit(repo, "one")
    original_tree = gitops.tree_sha(repo)
    saved = state.ParityState(tree_sha=original_tree, data={"step": "one"})

    state.save_state(repo, saved)

    loaded = state.load_state(repo)
    assert loaded == saved
    assert json.loads((repo / ".parity-state.json").read_text(encoding="utf-8"))["step"] == "one"
    leftovers = list(repo.glob(".parity-state.json.*"))
    assert leftovers == []

    (repo / "file.txt").write_text("two\n", encoding="utf-8")
    _commit(repo, "two")
    assert state.load_state(repo) is None
    assert state.load_state(repo, invalidate_on_tree_change=False) == saved


def test_bisect_classification_matrix() -> None:
    assert bisect.classify_results(baseline=True, merge=True) == bisect.Classification.ORDER_POLLUTION
    assert bisect.classify_results(baseline=True, merge=False) == bisect.Classification.REGRESSION
    assert bisect.classify_results(baseline=False, merge=True) == bisect.Classification.INHERITED_FLAKY
    assert bisect.classify_results(baseline=False, merge=False) == bisect.Classification.INHERITED_FLAKY
    assert bisect.classify_results(baseline=bisect.TestOutcome.ABSENT, merge=False) == bisect.Classification.UPSTREAM_TEST


def test_bisect_uses_injected_runner() -> None:
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def runner(repo: Path, tests: tuple[str, ...]) -> bool:
        calls.append((repo, tuple(tests)))
        return repo.name == "baseline"

    result = bisect.classify_baseline(
        baseline_repo=Path("/tmp/baseline"),
        merge_repo=Path("/tmp/merge"),
        tests=("tests/x.py::test_y",),
        runner=runner,
    )

    assert result.classification == bisect.Classification.REGRESSION
    assert sorted(calls, key=lambda item: str(item[0]))[:2] == [
        (Path("/tmp/baseline"), ("tests/x.py::test_y",)),
        (Path("/tmp/merge"), ("tests/x.py::test_y",)),
    ]


def test_bisect_demotes_regression_that_passes_on_rerun() -> None:
    merge_calls = 0

    def runner(repo: Path, tests: tuple[str, ...]) -> bisect.TestOutcome:
        nonlocal merge_calls
        if repo.name == "baseline":
            return bisect.TestOutcome.PASS
        merge_calls += 1
        return bisect.TestOutcome.FAIL if merge_calls == 1 else bisect.TestOutcome.PASS

    result = bisect.classify_baseline(
        baseline_repo=Path("/tmp/baseline"),
        merge_repo=Path("/tmp/merge"),
        tests=("tests/x.py::test_y",),
        runner=runner,
    )

    assert result.classification == bisect.Classification.FLAKY


def test_parse_pytest_outcome_detects_absent_baseline() -> None:
    assert bisect.parse_pytest_outcome(4, "ERROR: no tests ran") == bisect.TestOutcome.ABSENT
    assert bisect.parse_pytest_outcome(1, "not found: tests/x.py::missing") == bisect.TestOutcome.ABSENT


def test_from_file_parsing_stdin_and_path(tmp_path: Path) -> None:
    path = tmp_path / "nodes.txt"
    path.write_text("\n# comment\ntests/a.py::test_a\n\n", encoding="utf-8")

    assert bisect.parse_from_file(str(path)) == ["tests/a.py::test_a"]
    assert bisect.parse_from_file("-", stdin=StringIO("tests/b.py::test_b\n")) == ["tests/b.py::test_b"]


def test_bounded_jobs_caps_to_half_cpu() -> None:
    assert bisect.bounded_jobs(None, cpu_count=8) == 2
    assert bisect.bounded_jobs(99, cpu_count=8) == 4
    assert bisect.bounded_jobs(0, cpu_count=8) == 1


def test_resume_stage_skip_logic(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "file.txt").write_text("one\n", encoding="utf-8")
    _commit(repo, "one")
    state.save_state(
        repo,
        state.ParityState(
            tree_sha=gitops.tree_sha(repo),
            data={
                "gates": {
                    "markers": {"ok": True, "tree_sha": gitops.tree_sha(repo)},
                    "imports": {"ok": True, "tree_sha": gitops.tree_sha(repo)},
                    "traps": {"ok": False, "tree_sha": gitops.tree_sha(repo)},
                }
            },
        ),
    )

    assert gates.resume_plan(repo, gates.ORDERED_STAGE_NAMES[:4], resume=True) == ["traps", "manifest+forkdelta"]


def test_gitleaks_version_parse_and_failure_path(tmp_path: Path) -> None:
    assert gates.parse_gitleaks_version("uses: gitleaks/gitleaks-action@v2.3.4") == "v2.3.4"
    # Fleet form (fleet-secret-scan.yml): shell var feeding a release download,
    # name and VER= on different lines.
    fleet = "- name: Install gitleaks\n  run: |\n    VER=8.18.4\n    curl -fsSL something"
    assert gates.parse_gitleaks_version(fleet) == "8.18.4"
    # An unrelated VER= with no gitleaks nearby must NOT parse.
    assert gates.parse_gitleaks_version("VER=1.2.3\ncurl -fsSL something-else") is None
    repo = tmp_path / "repo"
    workflow = repo / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "ci.yml").write_text("name: ci\n", encoding="utf-8")

    reminders = gates.ci_reminders(repo)

    assert any("could not determine pinned gitleaks version" in item for item in reminders)


def test_status_fail_behind_exit_code(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli.gitops, "fetch", lambda repo, remote: None)
    monkeypatch.setattr(cli.gitops, "current_branch", lambda repo: "main")
    monkeypatch.setattr(cli.gitops, "ahead_behind", lambda repo, left, right: (0, 5))
    monkeypatch.setattr(cli.gitops, "merge_base", lambda repo, left, right: "abc")
    monkeypatch.setattr(cli.gitops, "conflict_entries", lambda repo: [])
    monkeypatch.setattr(cli.gitops, "conflict_marker_lines", lambda repo: [])
    monkeypatch.setattr(cli.buckets, "bucket_conflicts", lambda repo: [])

    assert cli._print_status(tmp_path, fail_behind=5) == 2
    assert cli._print_status(tmp_path, fail_behind=6) == 0


def test_forkdelta_ack_clears_uncovered_path(tmp_path: Path) -> None:
    """RC2: a reviewed-and-intentional ack clears the fork-delta gate without --force."""
    repo = _repo(tmp_path)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _commit(repo, "seed")
    st = state.ParityState(tree_sha=gitops.tree_sha(repo), data={"gates": {}})
    state.save_state(repo, st)

    assert state.acked_paths(repo) == set()
    state.record_ack(repo, "plugins/legacy/dead.py", "upstream removed feature; fork copy superseded")
    state.record_ack(repo, "plugins/legacy/dead.py", "updated reason")  # idempotent per path
    state.record_ack(repo, "other/file.py", "renamed upstream")

    acked = state.acked_paths(repo)
    assert acked == {"plugins/legacy/dead.py", "other/file.py"}
    loaded = state.load_state(repo, invalidate_on_tree_change=False)
    acks = loaded.data["forkdelta_acks"]
    assert len([a for a in acks if a["path"] == "plugins/legacy/dead.py"]) == 1
    assert [a["reason"] for a in acks if a["path"] == "plugins/legacy/dead.py"] == ["updated reason"]
    assert all(a.get("at") for a in acks)


def test_forkdelta_trigger_is_merge_touched_not_full_fork_delta(tmp_path: Path) -> None:
    """RC1: fork-only files the merge does NOT touch must not trip the gate;
    a fork-only file the merge deletes (upstream delete/rename case) must."""
    repo = _repo(tmp_path)
    (repo / "shared.py").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    _git(repo, "checkout", "-b", "fork-main")
    (repo / "fork_only_kept.py").write_text("fork\n", encoding="utf-8")
    (repo / "fork_only_dropped.py").write_text("fork\n", encoding="utf-8")
    _commit(repo, "fork adds two files")
    manifest = repo / "docs" / "sync" / "fork-features.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps([]), encoding="utf-8")
    # Simulate the merge deleting one fork-only file (upstream delete/rename),
    # leaving the other untouched.
    (repo / "fork_only_dropped.py").unlink()
    _git(repo, "add", "-A")

    touched = set(gitops.worktree_changed_files(repo, "fork-main"))
    report = forkdelta.compute_fork_delta(
        repo, base=base, fork_ref="fork-main", touched_paths=touched
    )

    assert "fork_only_dropped.py" in report.uncovered_paths
    assert "fork_only_kept.py" not in report.uncovered_paths
