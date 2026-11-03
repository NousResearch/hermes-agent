"""Behaviour contracts for optional-skills/software-development/tech-debt-audit/scripts/churn_rank.py."""
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / "optional-skills/software-development/tech-debt-audit/scripts/churn_rank.py")


@pytest.fixture(scope="module")
def churn_rank():
    return SimpleNamespace(**runpy.run_path(str(SCRIPT)))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", "-C", str(repo), *args],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    for i in range(3):
        (tmp_path / "hot.py").write_text("x = 1\n" * (10 + i), encoding="utf-8")
        _git(tmp_path, "add", "hot.py")
        _git(tmp_path, "commit", "-q", "-m", f"hot {i}")
    (tmp_path / "big.py").write_text("y = 2\n" * 500, encoding="utf-8")
    (tmp_path / "gone.py").write_text("z\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\0" * 10_000)
    _git(tmp_path, "add", "big.py", "gone.py", "blob.bin")
    _git(tmp_path, "commit", "-q", "-m", "more")
    _git(tmp_path, "rm", "-q", "gone.py")
    _git(tmp_path, "commit", "-q", "-m", "drop gone")
    return tmp_path


def test_ranks_live_files_by_churn_and_size(churn_rank, repo):
    result = churn_rank.rank(repo, since=None, top=10, scope=None)
    assert result["by_churn"][0] == {"path": "hot.py", "commits": 3, "lines": 12}
    assert "gone.py" not in {r["path"] for r in result["by_churn"]}
    assert result["by_size"][0] == {"path": "big.py", "lines": 500, "commits": 1}
    assert "blob.bin" not in {r["path"] for r in result["by_size"]}


def test_non_ascii_paths_are_ranked_under_default_quotepath(churn_rank, repo):
    (repo / "café.py").write_text("c = 3\n" * 700, encoding="utf-8")
    _git(repo, "add", "café.py")
    _git(repo, "commit", "-q", "-m", "accent")
    result = churn_rank.rank(repo, since=None, top=10, scope=None)
    assert {"path": "café.py", "commits": 1, "lines": 700} in result["by_churn"]
    assert result["by_size"][0] == {"path": "café.py", "lines": 700, "commits": 1}


def test_not_a_repo_exits_2(churn_rank, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert churn_rank.main(["--repo", str(tmp_path)]) == 2
    assert "git failed" in capsys.readouterr().err
