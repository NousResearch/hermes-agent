"""Tests for the fork-diff gate (`scripts/check_forkdiff.py`).

The gate has two jobs: pin `fork.yaml`'s base to the real merge-base with
upstream, and refuse a fork diff that any section fails to describe. Both are
exercised against throwaway git repositories so the tests never depend on the
network or on this checkout's history.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_forkdiff  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(repo: Path, message: str, **files: str) -> str:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def fork_repo(tmp_path: Path) -> tuple[Path, str]:
    """A repo with an 'upstream' line and a fork commit on top of it.

    Returns the repo and the upstream tip's hash. ``refs/remotes/upstream/main``
    points at the upstream tip, mirroring what CI fetches.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "upstream 1", **{"core/a.py": "a\n", "README.md": "up\n"})
    upstream_tip = _commit(repo, "upstream 2", **{"core/b.py": "b\n"})
    _git(repo, "update-ref", "refs/remotes/upstream/main", upstream_tip)
    _commit(
        repo,
        "fork feature",
        **{"gateway/wiki.py": "wiki\n", "tests/test_wiki.py": "t\n", "fork.yaml": "placeholder\n"},
    )
    return repo, upstream_tip


def _write_fork_yaml(repo: Path, base: str, globs: list[str], ignore: list[str] | None = None) -> None:
    spec = {
        "title": "t",
        "base": {"name": "up", "url": "https://x", "hash": base},
        "fork": {"name": "f", "url": "https://y", "ref": "refs/heads/main"},
        "def": {"title": "f", "sub": [{"title": "wiki", "globs": globs}]},
        "ignore": ignore or ["fork.yaml"],
    }
    (repo / "fork.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")


class TestGlobs:
    @pytest.mark.parametrize(
        "glob,path,expected",
        [
            ("a/b.py", "a/b.py", True),
            ("a/*.py", "a/b.py", True),
            ("a/*.py", "a/c/b.py", False),  # * never crosses a slash
            ("a/**", "a/c/b.py", True),
            ("a/**/b.py", "a/b.py", True),  # ** may match zero directories
            ("a/**/b.py", "a/c/d/b.py", True),
            ("**/*_test.py", "x/y/z_test.py", True),
            ("a/?.py", "a/b.py", True),
            ("a/?.py", "a/bb.py", False),
            ("a/*[!_test].py", "a/b_test.py", False),
            ("a/*[!_test].py", "a/bx.py", True),
            (".github/**", ".github/workflows/ci.yml", True),
        ],
    )
    def test_glob_semantics(self, glob: str, path: str, expected: bool) -> None:
        assert bool(check_forkdiff.glob_to_regex(glob).match(path)) is expected

    def test_collect_globs_walks_the_tree_and_counts_section_ignores(self) -> None:
        spec = {
            "title": "root",
            "globs": ["a"],
            "sub": [
                {"title": "one", "globs": ["b"], "ignore": ["c"]},
                {"title": "two", "sub": [{"title": "deep", "globs": ["d"]}]},
            ],
        }
        found = check_forkdiff.collect_globs(spec)
        assert [g for _, g in found] == ["a", "b", "c", "d"]
        assert found[3][0] == "root › two › deep"


class TestCoverage:
    def test_reports_uncovered_and_stale(self) -> None:
        globs = [("s", "gateway/*.py"), ("s", "docs/**"), ("s", "never/*")]
        uncovered, stale, hits = check_forkdiff.check_coverage(
            ["gateway/wiki.py", "tests/test_wiki.py", "docs/api/x.md"], globs
        )
        assert uncovered == ["tests/test_wiki.py"]
        assert stale == [("s", "never/*")]
        assert hits["gateway/*.py"] == 1 and hits["docs/**"] == 1


class TestEndToEnd:
    def test_passes_when_base_is_merge_base_and_everything_is_described(
        self, fork_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, upstream_tip = fork_repo
        _write_fork_yaml(repo, upstream_tip, ["gateway/wiki.py", "tests/test_wiki.py"])
        rc = check_forkdiff.main(
            ["--repo", str(repo), "--upstream-ref", "refs/remotes/upstream/main"]
        )
        assert rc == 0, capsys.readouterr().out

    def test_fails_when_a_fork_file_is_not_described(
        self, fork_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        repo, upstream_tip = fork_repo
        _write_fork_yaml(repo, upstream_tip, ["gateway/wiki.py"])
        status = tmp_path / "status.json"
        rc = check_forkdiff.main(["--repo", str(repo), "--review-status-out", str(status)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "tests/test_wiki.py" in out
        payload = yaml.safe_load(status.read_text())
        assert payload[0]["results"][0]["kind"] == "action_required"
        assert "tests/test_wiki.py" in payload[0]["results"][0]["detail"]

    def test_fails_on_stale_glob(self, fork_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
        repo, upstream_tip = fork_repo
        _write_fork_yaml(repo, upstream_tip, ["gateway/wiki.py", "tests/test_wiki.py", "gone/*.py"])
        rc = check_forkdiff.main(["--repo", str(repo)])
        assert rc == 1
        assert "gone/*.py" in capsys.readouterr().out

    def test_fails_after_a_rebase_until_base_is_bumped(
        self, fork_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, old_tip = fork_repo
        # Upstream moves on, and the fork is rebased onto it: the new upstream
        # commit becomes an ancestor of the fork while fork.yaml still names
        # the old tip.
        fork_commit = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "-q", old_tip)
        new_tip = _commit(repo, "upstream 3", **{"core/c.py": "c\n"})
        _git(repo, "update-ref", "refs/remotes/upstream/main", new_tip)
        _git(repo, "checkout", "-q", "main")
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "rebase", "-q", new_tip, "main")
        assert _git(repo, "rev-parse", "HEAD") != fork_commit
        _write_fork_yaml(repo, old_tip, ["gateway/wiki.py", "tests/test_wiki.py"])

        rc = check_forkdiff.main(["--repo", str(repo), "--upstream-ref", "refs/remotes/upstream/main"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "merge-base" in out and new_tip[:12] in out

        # Bumping base.hash is exactly what makes it green again — and with the
        # correct base, upstream's own core/c.py is no longer in the diff.
        _write_fork_yaml(repo, new_tip, ["gateway/wiki.py", "tests/test_wiki.py"])
        rc = check_forkdiff.main(["--repo", str(repo), "--upstream-ref", "refs/remotes/upstream/main"])
        assert rc == 0, capsys.readouterr().out

    def test_rejects_short_or_foreign_base(self, fork_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
        repo, upstream_tip = fork_repo
        _write_fork_yaml(repo, upstream_tip[:12], ["gateway/wiki.py", "tests/test_wiki.py"])
        assert check_forkdiff.main(["--repo", str(repo)]) == 1
        assert "40-hex" in capsys.readouterr().out

        # A fork commit is a real 40-hex id but not on upstream main.
        fork_commit = _git(repo, "rev-parse", "HEAD")
        _write_fork_yaml(repo, fork_commit, ["gateway/wiki.py", "tests/test_wiki.py"])
        rc = check_forkdiff.main(["--repo", str(repo), "--upstream-ref", "refs/remotes/upstream/main"])
        assert rc == 1
        assert "not an ancestor" in capsys.readouterr().out
