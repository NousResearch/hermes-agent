from __future__ import annotations

import json
import subprocess
import threading
from io import StringIO
from pathlib import Path

from scripts.hermes_parity import bisect, buckets, catchup, cli, forkdelta, gates, gitops, lint_manifest, lint_merge_traps, lint_unbound, state


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
    assert bisect.classify_results(baseline=True, merge=True) == bisect.Classification.CLEAN
    assert bisect.classify_results(baseline=True, merge=False) == bisect.Classification.REGRESSION
    assert bisect.classify_results(baseline=False, merge=True) == bisect.Classification.FIXED_BY_MERGE
    assert bisect.classify_results(baseline=False, merge=False) == bisect.Classification.INHERITED
    assert bisect.classify_results(baseline=bisect.TestOutcome.ABSENT, merge=False) == bisect.Classification.UPSTREAM_TEST


def test_bisect_six_fixture_corpus_with_stability_and_dep_bump(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "merge").mkdir()
    baseline_repo = _repo(tmp_path / "base")
    merge_repo = _repo(tmp_path / "merge")
    counter = tmp_path / "counter.txt"
    counter_lock = threading.Lock()
    verdicts = {
        "clean": (bisect.TestOutcome.PASS, bisect.TestOutcome.PASS),
        "regression": (bisect.TestOutcome.PASS, bisect.TestOutcome.FAIL),
        "fixed": (bisect.TestOutcome.FAIL, bisect.TestOutcome.PASS),
        "inherited": (bisect.TestOutcome.FAIL, bisect.TestOutcome.FAIL),
        "dep": (bisect.TestOutcome.PASS, bisect.TestOutcome.FAIL),
    }

    def runner(repo: Path, tests: tuple[str, ...]) -> bisect.PytestRun:
        test = tests[0]
        if test == "nondeterministic":
            # PER-SIDE parity counter: a single shared counter lets the threaded
            # N=3 runs interleave so each SIDE sees a stable sequence (all-even /
            # all-odd) -> classified REGRESSION, flaking this test in CI. Keying
            # the counter by repo guarantees each side alternates P,F,P.
            # repo.name is "repo" for BOTH sides (_repo appends a fixed subdir);
            # key by the full resolved path hash so each side truly gets its own file.
            import hashlib
            side_key = hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:12]
            side_counter = counter.with_name(f"counter-{side_key}.txt")
            with counter_lock:
                current = int(side_counter.read_text(encoding="utf-8")) if side_counter.exists() else 0
                side_counter.write_text(str(current + 1), encoding="utf-8")
            outcome = bisect.TestOutcome.PASS if current % 2 == 0 else bisect.TestOutcome.FAIL
            return bisect.PytestRun(outcome, f"{repo} token=SECRET")
        baseline, merge = verdicts[test]
        return bisect.PytestRun(baseline if repo == baseline_repo else merge, f"{repo} output")

    def dep_runner(repo: Path, tests: tuple[str, ...]) -> bisect.PytestRun:
        return bisect.PytestRun(bisect.TestOutcome.FAIL, "dep bumped")

    expected = {
        "clean": bisect.Classification.CLEAN,
        "regression": bisect.Classification.REGRESSION,
        "fixed": bisect.Classification.FIXED_BY_MERGE,
        "inherited": bisect.Classification.INHERITED,
        "nondeterministic": bisect.Classification.NONDETERMINISTIC,
        "dep": bisect.Classification.REGRESSION_DEP,
    }
    results = {
        name: bisect.classify_one(
            baseline_repo=baseline_repo,
            merge_repo=merge_repo,
            test=name,
            runner=runner,
            dep_runner=dep_runner if name == "dep" else None,
        ).classification
        for name in expected
    }

    assert results == expected
    assert bisect.scrub_output(f"{tmp_path}/venv leaked", home=tmp_path) == ["$HOME/venv leaked"]


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
    assert calls.count((Path("/tmp/baseline"), ("tests/x.py::test_y",))) == 3
    assert calls.count((Path("/tmp/merge"), ("tests/x.py::test_y",))) == 3


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

    assert result.classification == bisect.Classification.REGRESSION_FLAKY


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


def test_lint_manifest_path_nodeid_and_vacuous_floor(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_pkg.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    base = _commit(repo, "base")
    _git(repo, "checkout", "-b", "fork-main")
    (repo / "fork_only.py").write_text("fork\n", encoding="utf-8")
    _commit(repo, "fork")
    manifest = repo / "docs" / "sync" / "fork-features.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps([
            {
                "feature": "broken",
                "tests": ["tests/test_pkg.py::test_ok"],
                "paths": ["missing/**/*.py"],
                "why": "contract",
            }
        ]),
        encoding="utf-8",
    )

    broken = lint_manifest.lint_manifest(repo, base=base, fork_ref="fork-main", touched_paths={"fork_only.py"})
    assert any("matches zero files" in error for error in broken.errors)

    manifest.write_text(
        json.dumps([
            {
                "feature": "vacuous",
                "tests": ["tests/test_pkg.py::test_ok"],
                "paths": ["pkg.py"],
                "why": "contract",
            }
        ]),
        encoding="utf-8",
    )
    vacuous = lint_manifest.lint_manifest(repo, base=base, fork_ref="fork-main", touched_paths={"fork_only.py"})
    assert any("vacuous forkdelta coverage" in error for error in vacuous.errors)
    assert lint_manifest.lint_manifest(repo, base=base, fork_ref="fork-main", touched_paths={"fork_only.py"}, vacuous_ok=True).ok


def test_catchup_selects_closure_tests_and_reports_coverage_map(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "app").mkdir()
    (repo / "tests").mkdir()
    (repo / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "app" / "api.py").write_text("def value(arg):\n    return arg\n", encoding="utf-8")
    (repo / "app" / "caller.py").write_text("from app.api import value\n\ndef call():\n    return value()\n", encoding="utf-8")
    (repo / "tests" / "test_caller.py").write_text("from app.caller import call\n\ndef test_call():\n    assert call() is None\n", encoding="utf-8")
    _commit(repo, "fixture")

    selected, coverage = catchup.select_tests(repo, ["app/api.py"], [])
    proc = subprocess.run(
        ["python3.11", "-m", "pytest", *selected, "-q", "-o", "addopts=", "-p", "no:randomly"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    report = catchup.format_report(
        catchup.CatchupReport(("app/api.py", "untested.py"), selected, {**coverage, "untested.py": ()}, ("untested.py",), ())
    )

    assert "tests/test_caller.py" in selected
    assert proc.returncode != 0
    assert "app/api.py: tests/test_caller.py" in report
    assert "untested.py: UNTESTED-BY-CATCHUP" in report


def test_catchup_resolves_init_reexport_closure(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("from .target import func\n", encoding="utf-8")
    (repo / "pkg" / "target.py").write_text("def func(arg):\n    return arg\n", encoding="utf-8")
    (repo / "caller.py").write_text("from pkg import func\n\ndef call():\n    return func()\n", encoding="utf-8")
    (repo / "tests" / "test_reexport.py").write_text("from caller import call\n\ndef test_call():\n    assert call() is None\n", encoding="utf-8")
    _commit(repo, "fixture")

    selected, _coverage = catchup.select_tests(repo, ["pkg/target.py"], [])

    assert "tests/test_reexport.py" in selected


def test_catchup_refuses_more_than_max_commits(monkeypatch, tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setattr(catchup.gitops, "fetch", lambda repo, remote: None)
    monkeypatch.setattr(catchup.gitops, "ahead_behind", lambda repo, left, right: (0, 51))

    try:
        catchup.run_catchup(repo, max_commits=50)
    except gitops.GitError as exc:
        assert "refuses 51 commits" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected refusal")


def test_ack_from_file_and_vacuous_ack_are_distinct(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _commit(repo, "seed")
    state.save_state(repo, state.ParityState(tree_sha=gitops.tree_sha(repo), data={"gates": {}}))
    ack_file = tmp_path / "acks.txt"
    ack_file.write_text("tabbed.py\tper-path reason\nbare.py\n", encoding="utf-8")

    assert cli.main(["ack", "--worktree", str(repo), "--from-file", str(ack_file)]) == 2
    assert cli.main(["ack", "--worktree", str(repo), "--from-file", str(ack_file), "--reason", "bulk reason"]) == 0
    assert state.acked_paths(repo) == {"tabbed.py", "bare.py"}
    assert not state.has_vacuous_ack(repo)
    assert cli.main(["ack", "--worktree", str(repo), "--vacuous-ok", "--reason", "reviewed empty coverage"]) == 0
    assert state.has_vacuous_ack(repo)


def test_forkdelta_emit_uncovered_is_snapshot_not_live_ack(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    (repo / "base.py").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    _git(repo, "checkout", "-b", "fork-main")
    (repo / "old.py").write_text("old\n", encoding="utf-8")
    _commit(repo, "fork old")
    manifest = repo / "docs" / "sync" / "fork-features.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps([]), encoding="utf-8")
    state.save_state(repo, state.ParityState(tree_sha=gitops.tree_sha(repo), data={"gates": {}}))
    assert cli.main(["forkdelta", "--worktree", str(repo), "--base", base, "--fork-ref", "fork-main", "--emit-uncovered"]) == 0
    snapshot = capsys.readouterr().out.strip().splitlines()
    (repo / "new.py").write_text("new\n", encoding="utf-8")
    _commit(repo, "fork new")
    review = tmp_path / "reviewed.txt"
    review.write_text("\n".join(snapshot), encoding="utf-8")

    assert cli.main(["ack", "--worktree", str(repo), "--from-file", str(review), "--reason", "reviewed snapshot"]) == 0
    assert state.acked_paths(repo) == {"old.py"}


def test_tests_stage_skip_exit_jsonl_and_finish_not_satisfied(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repo / "file.txt").write_text("one\n", encoding="utf-8")
    _commit(repo, "one")
    state.save_state(repo, state.ParityState(tree_sha=gitops.tree_sha(repo), data={"gates": {}}))

    results = gates.run_gates(repo, stage="tests")
    payload = json.loads((repo / "gates.jsonl").read_text(encoding="utf-8").splitlines()[-1])

    assert results[0].status == "SKIP"
    assert payload["gate"] == "tests"
    assert payload["status"] == "SKIP"
    assert not cli._all_gates_green(repo)


def test_merge_trap_dead_code_after_return() -> None:
    """The check_dangerous_command incident: merge concatenated fork+upstream
    bodies, an early return orphaned the fork's cron-gate block."""
    issues = lint_merge_traps.lint_source(
        """
def check(cmd):
    return _run_approval_gate(cmd)
    if is_cron_session():
        return "denied"
""".lstrip()
    )
    assert [(i.kind, i.line) for i in issues] == [("dead-code-after-return", 3)]

    clean = lint_merge_traps.lint_source(
        """
def check(cmd):
    if bad(cmd):
        return "denied"
    return _run_approval_gate(cmd)
""".lstrip()
    )
    assert clean == []


def test_merge_trap_getattr_lambda_fallback() -> None:
    """The should_defer_preflight incident: getattr with a callable fallback
    naming a method no side defines anymore silently returns the fallback."""
    issues = lint_merge_traps.lint_source(
        """
def guard(compressor, tokens):
    fn = getattr(compressor, "should_defer_preflight_to_real_usage", lambda _t: False)
    return fn(tokens)
""".lstrip()
    )
    assert [(i.kind, i.line) for i in issues] == [("getattr-lambda-fallback", 2)]

    # Defined somewhere in the scanned set -> no issue.
    clean = lint_merge_traps.lint_source(
        """
class C:
    def known_method(self):
        return 1

def guard(obj):
    fn = getattr(obj, "known_method", lambda: False)
    return fn()
""".lstrip()
    )
    assert clean == []

    # Non-callable fallback (plain default) is not the incident class.
    clean2 = lint_merge_traps.lint_source(
        """
def guard(obj):
    return getattr(obj, "whatever_setting", None)
""".lstrip()
    )
    assert clean2 == []


def test_merge_trap_duplicate_function_bodies() -> None:
    """The scheduler two-dispatch-bodies incident: merge kept both sides'
    structurally identical function bodies under different names."""
    body = """
    job = claim(x)
    if job is None:
        return None
    result = dispatch(job)
    record(result)
    return result
"""
    issues = lint_merge_traps.lint_source(
        f"def _process_one_job(x):{body}\ndef run_one_job(x):{body}"
    )
    assert [(i.kind,) for i in issues] == [("duplicate-function-bodies",)]

    # Different bodies -> no issue; trivial identical getters -> no issue.
    clean = lint_merge_traps.lint_source(
        """
def a(self):
    return self._x

def b(self):
    return self._x
""".lstrip()
    )
    assert clean == []


def test_merge_trap_repo_wide_defined_names(tmp_path: Path) -> None:
    """getattr-fallback must consider names defined in OTHER files."""
    (tmp_path / "impl.py").write_text(
        "class Compressor:\n    def cross_file_method(self):\n        return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "caller.py").write_text(
        "def guard(c):\n    return getattr(c, \"cross_file_method\", lambda: False)()\n",
        encoding="utf-8",
    )
    issues = lint_merge_traps.lint_paths(
        [tmp_path / "impl.py", tmp_path / "caller.py"], repo=tmp_path
    )
    assert issues == []


def test_merge_trap_empty_generator_idiom_not_flagged() -> None:
    """`return` followed by `yield` is the empty-async-generator idiom."""
    clean = lint_merge_traps.lint_source(
        """
def history(self):
    async def _empty():
        return
        yield
    return _empty()
""".lstrip()
    )
    assert clean == []
