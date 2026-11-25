"""Behavior tests for the changed-file verification gate."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts import verify_change


def _git(repo: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
    return proc.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "foo.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_changed_paths_includes_modified_and_untracked(tmp_path):
    repo = _repo(tmp_path)
    (repo / "pkg" / "foo.py").write_text("VALUE = 2\n", encoding="utf-8")
    test = repo / "tests" / "pkg" / "test_foo.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_foo(): pass\n", encoding="utf-8")

    changed = verify_change.changed_paths(repo, "HEAD")

    assert "pkg/foo.py" in changed
    assert "tests/pkg/test_foo.py" in changed


def test_discovery_excludes_suites_the_runner_skips_by_default():
    assert verify_change._is_discoverable_test_file("tests/hermes_cli/test_foo.py")
    assert not verify_change._is_discoverable_test_file("tests/integration/test_foo.py")
    assert not verify_change._is_discoverable_test_file("tests/e2e/test_foo.py")
    assert not verify_change._is_discoverable_test_file("tests/docker/test_foo.py")


def test_selects_named_and_importing_tests():
    all_tests = [
        "tests/pkg/test_foo.py",
        "tests/pkg/test_foo_extra.py",
        "tests/other/test_uses_foo.py",
        "tests/unrelated/test_bar.py",
    ]
    texts = {
        "tests/other/test_uses_foo.py": "from pkg import foo\n",
        "tests/unrelated/test_bar.py": "def test_bar(): pass\n",
    }

    selected, global_change = verify_change.select_tests(
        ["pkg/foo.py"], all_tests, lambda path: texts.get(path, ""),
    )

    assert selected == [
        "tests/other/test_uses_foo.py",
        "tests/pkg/test_foo.py",
        "tests/pkg/test_foo_extra.py",
    ]
    assert global_change is False


def test_direct_test_and_global_infrastructure_are_selected():
    all_tests = ["tests/a/test_one.py", "tests/b/test_two.py"]

    selected, global_change = verify_change.select_tests(
        ["tests/a/test_one.py", "tests/conftest.py"], all_tests, lambda _path: "",
    )

    assert selected == all_tests
    assert global_change is True


def test_parse_failures_reads_runner_summaries():
    output = """
FAILED tests/a/test_one.py::test_bad
  FAILED tests/a/test_two.py::test_worse
ERROR tests/b/test_three.py::test_setup
======================== 2 failed, 1 error
"""

    assert verify_change.parse_failures(output) == {
        "tests/a/test_one.py::test_bad",
        "tests/a/test_two.py::test_worse",
        "tests/b/test_three.py::test_setup",
    }


def test_baseline_compare_reports_new_and_fixed():
    new, fixed = verify_change.compare_baseline(
        {"a::new", "a::unchanged"}, {"a::unchanged", "a::fixed"},
    )

    assert new == ["a::new"]
    assert fixed == ["a::fixed"]


def test_baseline_round_trip(tmp_path):
    path = tmp_path / "baseline.json"

    verify_change.save_baseline(path, ["tests/b.py::test_b", "tests/a.py::test_a"], base="abc")

    assert verify_change.load_baseline(path) == {"tests/a.py::test_a", "tests/b.py::test_b"}
    assert path.read_text(encoding="utf-8").startswith('{\n  "version": 1')


def test_main_fails_only_on_failures_new_since_baseline(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    verify_change.save_baseline(baseline, ["tests/pkg/test_foo.py::test_known"], base="abc")
    monkeypatch.setattr(verify_change, "REPO", tmp_path)
    monkeypatch.setattr(verify_change, "_resolve_base", lambda _repo, _base: ("abc", "main"))
    monkeypatch.setattr(verify_change, "changed_paths", lambda _repo, _base: ["pkg/foo.py"])
    monkeypatch.setattr(verify_change, "_test_files", lambda _repo: ["tests/pkg/test_foo.py"])
    monkeypatch.setattr(verify_change, "_read_text", lambda _repo: lambda _path: "")
    observed = ["tests/pkg/test_foo.py::test_known"]
    monkeypatch.setattr(
        verify_change, "run_tests",
        lambda _repo, _tests, jobs=None: verify_change.RunResult(
            0, "FAILED tests/pkg/test_foo.py::test_known\n", frozenset(observed)),
    )

    assert verify_change.main(["--baseline", str(baseline)]) == 0

    observed.append("tests/pkg/test_foo.py::test_new")
    monkeypatch.setattr(
        verify_change, "run_tests",
        lambda _repo, _tests, jobs=None: verify_change.RunResult(
            1, "FAILED tests/pkg/test_foo.py::test_new\n", frozenset(observed)),
    )

    assert verify_change.main(["--baseline", str(baseline)]) == 1
