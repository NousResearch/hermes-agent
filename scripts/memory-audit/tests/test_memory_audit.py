"""เทสต์ memory-audit ด้วย git repo ชั่วคราว (subprocess จริง)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "memory_audit.py"


def run_audit(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def init_repo(repo: Path) -> None:
    git(repo, "init")
    git(repo, "config", "user.email", "audit@test.local")
    git(repo, "config", "user.name", "Memory Audit Test")


def write_project(
    repo: Path,
    *,
    overview_extra: str = "",
    plan_extra: str = "",
    include_decisions: bool = True,
) -> str:
    project = repo / ".project"
    project.mkdir(parents=True, exist_ok=True)

    overview = (
        "> memory-schema: v1.2\n"
        "# Overview\n"
        "อัปเดต commit `PLACEHOLDER_SHA`\n"
        f"{overview_extra}\n"
    )
    plan = (
        "# Plan — TEST\n"
        "> memory-schema: v1.2 · **plan_id: GRD**\n"
        "งานตัวอย่าง\n"
        f"{plan_extra}\n"
    )
    (project / "OverviewProgress.md").write_text(overview, encoding="utf-8")
    (project / "plan.md").write_text(plan, encoding="utf-8")
    if include_decisions:
        (project / "decisions.md").write_text("# decisions\n", encoding="utf-8")
    return overview


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-m", message)
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def seed_healthy_repo(repo: Path) -> None:
    init_repo(repo)
    write_project(repo)
    sha = commit_all(repo, "init project memory")
    overview_path = repo / ".project" / "OverviewProgress.md"
    overview_path.write_text(
        overview_path.read_text(encoding="utf-8").replace("PLACEHOLDER_SHA", sha),
        encoding="utf-8",
    )
    git(repo, "add", ".project/OverviewProgress.md")
    git(repo, "commit", "-m", "record sha in overview")


@pytest.fixture()
def healthy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "healthy"
    repo.mkdir()
    seed_healthy_repo(repo)
    return repo


def test_healthy_repo_exit_zero(healthy_repo: Path) -> None:
    result = run_audit(healthy_repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "✅" in result.stdout


def test_nonexistent_sha_exit_one(tmp_path: Path) -> None:
    repo = tmp_path / "bad-sha"
    repo.mkdir()
    init_repo(repo)
    write_project(repo, overview_extra="งานที่ `deadbeef1` merge แล้ว\n")
    commit_all(repo, "memory with fake sha")

    result = run_audit(repo)
    assert result.returncode == 1
    assert "deadbeef1" in result.stdout or "ไม่มีใน git" in result.stdout


def test_reverted_sha_exit_nonzero(tmp_path: Path) -> None:
    repo = tmp_path / "reverted"
    repo.mkdir()
    init_repo(repo)

    feature = repo / "feature.txt"
    feature.write_text("v1\n", encoding="utf-8")
    git(repo, "add", "feature.txt")
    git(repo, "commit", "-m", "add feature for memory audit test")
    sha_proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    work_sha = sha_proc.stdout.strip()

    git(repo, "revert", "--no-edit", "HEAD")

    write_project(
        repo,
        overview_extra=f"ฟีเจอร์เสร็จแล้วที่ `{work_sha}`\n",
    )
    commit_all(repo, "memory still claims reverted work")

    result = run_audit(repo)
    assert result.returncode != 0
    assert "revert" in result.stdout.lower() or work_sha in result.stdout


def test_gitignore_hides_plan_exit_one(tmp_path: Path) -> None:
    repo = tmp_path / "ignored"
    repo.mkdir()
    seed_healthy_repo(repo)
    (repo / ".gitignore").write_text(".project/plan.md\n", encoding="utf-8")
    git(repo, "rm", "--cached", ".project/plan.md")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-m", "ignore plan.md")

    result = run_audit(repo)
    assert result.returncode == 1
    assert "gitignore" in result.stdout.lower() or "track" in result.stdout.lower()


def test_orphan_issue_id_exit_two(tmp_path: Path) -> None:
    repo = tmp_path / "orphan"
    repo.mkdir()
    seed_healthy_repo(repo)

    ledger_dir = repo / ".hermes" / "ai-relay"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / "calls-nobranch.md").write_text(
        "| timestamp | issue_id | tool | account_used | rotated_from | status | calls_used | output_ref |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-07-08T10:00:00 | P1-I1-orphan-test | codex | a1 | | ok | 1 | out |\n",
        encoding="utf-8",
    )
    git(repo, "add", ".hermes/ai-relay/calls-nobranch.md")
    git(repo, "commit", "-m", "add ledger with orphan id")

    result = run_audit(repo)
    assert result.returncode == 2
    assert "กำพร้า" in result.stdout or "P1-I1-orphan-test" in result.stdout


def test_json_output(healthy_repo: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(healthy_repo), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert '"status":"pass"' in result.stdout.replace(" ", "")


def test_not_a_repo_exit_three(tmp_path: Path) -> None:
    repo = tmp_path / "not-git"
    repo.mkdir()
    result = run_audit(repo)
    assert result.returncode == 3


def test_nonexistent_repo_path_exit_three(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert "ไม่พบโฟลเดอร์" in result.stderr or "ไม่พบโฟลเดอร์" in result.stdout


def test_digit_only_backtick_not_treated_as_sha(tmp_path: Path) -> None:
    repo = tmp_path / "digit-only"
    repo.mkdir()
    init_repo(repo)
    write_project(
        repo,
        overview_extra="ขนาด buffer `1048576` bytes และวันที่ `20260708`\n",
    )
    commit_all(repo, "memory with digit-only backticks")

    result = run_audit(repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_revert_prefix_false_positive_guard(tmp_path: Path) -> None:
    repo = tmp_path / "prefix-guard"
    repo.mkdir()
    init_repo(repo)

    first = repo / "first.txt"
    first.write_text("v1\n", encoding="utf-8")
    git(repo, "add", "first.txt")
    git(repo, "commit", "-m", "docs(grd): first change")
    kept_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    second = repo / "second.txt"
    second.write_text("v2\n", encoding="utf-8")
    git(repo, "add", "second.txt")
    git(repo, "commit", "-m", "docs(grd): second change")
    git(repo, "revert", "--no-edit", "HEAD")

    write_project(
        repo,
        overview_extra=f"งานที่ยังใช้อยู่ที่ `{kept_sha}`\n",
    )
    commit_all(repo, "memory claims kept sha after unrelated revert")

    result = run_audit(repo)
    assert result.returncode == 0, result.stdout + result.stderr