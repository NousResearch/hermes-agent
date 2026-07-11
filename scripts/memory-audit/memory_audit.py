#!/usr/bin/env python3
"""memory-audit — เทียบความจำ .project/ กับความเป็นจริงของ git (stdlib only)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "warn", "fail"]

PLAN_ID_RE = re.compile(r"plan_id:\s*([A-Za-z0-9_-]+)")
SHA_BACKTICK_RE = re.compile(r"`([0-9a-fA-F]{7,12})`")
MEMORY_SCHEMA_RE = re.compile(r"^>\s*memory-schema:")
EXTRA_PLAN_PREFIXES = ("jarvis",)

REQUIRED_PROJECT_FILES = (
    ".project/OverviewProgress.md",
    ".project/plan.md",
    ".project/decisions.md",
)

CHECK_LABELS = {
    "schema": "ป้าย schema / plan_id",
    "shas": "SHA ที่ความจำอ้าง",
    "git_tracking": "ไฟล์ความจำใน git",
    "orphan_ids": "เลขงานในสมุด relay",
}


@dataclass
class CheckResult:
    check_id: str
    status: Status
    message: str

    def to_dict(self) -> dict:
        return {"id": self.check_id, "status": self.status, "message": self.message}


class GitRunner:
    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=False,
        )

    def available(self) -> str | None:
        if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
            return "ไม่พบคำสั่ง git ในระบบ"
        probe = self.run(["git", "rev-parse", "--git-dir"])
        if probe.returncode != 0:
            return "โฟลเดอร์นี้ไม่ใช่ git repository"
        return None


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def first_line(text: str | None) -> str:
    if not text:
        return ""
    return text.splitlines()[0] if text.splitlines() else ""


def extract_shas(*texts: str | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for text in texts:
        if not text:
            continue
        for match in SHA_BACKTICK_RE.finditer(text):
            sha = match.group(1).lower()
            if not re.search(r"[a-f]", sha):
                continue
            if sha not in seen:
                seen.add(sha)
                ordered.append(sha)
    return ordered


def extract_plan_ids(plan_text: str | None) -> list[str]:
    if not plan_text:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for match in PLAN_ID_RE.finditer(plan_text):
        pid = match.group(1)
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


def parse_md_table(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        rows.append(cells)
    return rows


def issue_id_allowed(issue_id: str, plan_ids: list[str]) -> bool:
    if not issue_id or issue_id in {"issue_id", "?"}:
        return True
    prefixes = list(plan_ids) + list(EXTRA_PLAN_PREFIXES)
    return any(issue_id.startswith(f"{pid}-") or issue_id == pid for pid in prefixes)


def check_schema(project: Path, overview: str | None, plan: str | None) -> CheckResult:
    check_id = "schema"
    if overview is None:
        return CheckResult(check_id, "fail", "ไม่พบไฟล์ .project/OverviewProgress.md")
    if plan is None:
        return CheckResult(check_id, "fail", "ไม่พบไฟล์ .project/plan.md")
    line1 = first_line(overview)
    if not MEMORY_SCHEMA_RE.match(line1):
        return CheckResult(
            check_id,
            "fail",
            "บรรทัดแรกของ OverviewProgress.md ต้องขึ้นต้นด้วย '> memory-schema:'",
        )
    if not PLAN_ID_RE.search(plan):
        return CheckResult(check_id, "fail", "plan.md ไม่มี plan_id ที่อ่านได้")
    return CheckResult(check_id, "ok", "ป้าย memory-schema และ plan_id ครบ")


def sha_commit_exists(git: GitRunner, sha: str) -> bool:
    proc = git.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"])
    return proc.returncode == 0


def sha_reverted(git: GitRunner, sha: str) -> tuple[bool, str]:
    direct = git.run(
        [
            "git",
            "log",
            "--oneline",
            f"--grep={sha}",
            "--grep=Revert",
            "--all-match",
            "-n",
            "3",
        ]
    )
    if direct.stdout.strip():
        revert_sha = direct.stdout.strip().split()[0]
        return True, revert_sha
    return False, ""


def check_shas(git: GitRunner, overview: str | None, plan: str | None) -> CheckResult:
    check_id = "shas"
    shas = extract_shas(overview, plan)
    if not shas:
        return CheckResult(check_id, "ok", "ไม่พบ SHA ในความจำที่ต้องตรวจ")

    missing: list[str] = []
    reverted: list[str] = []
    for sha in shas:
        if not sha_commit_exists(git, sha):
            missing.append(sha)
            continue
        is_rev, rev_ref = sha_reverted(git, sha)
        if is_rev:
            reverted.append(f"{sha}→{rev_ref}")

    if missing:
        return CheckResult(
            check_id,
            "fail",
            f"SHA ที่อ้างไม่มีใน git: {', '.join(missing)}",
        )
    if reverted:
        return CheckResult(
            check_id,
            "fail",
            f"พบ SHA ที่ถูก revert แล้ว: {', '.join(reverted)}",
        )
    return CheckResult(check_id, "ok", f"SHA ทั้ง {len(shas)} ตัวมีจริงและยังไม่ถูก revert")


def list_project_files(project: Path) -> list[Path]:
    if not project.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(project.rglob("*")):
        if path.is_file():
            files.append(path)
    return files


def check_git_tracking(git: GitRunner, project: Path) -> CheckResult:
    check_id = "git_tracking"
    if not project.is_dir():
        return CheckResult(check_id, "fail", "ไม่พบโฟลเดอร์ .project/")

    ignored: list[str] = []
    for path in list_project_files(project):
        rel = path.relative_to(git.repo).as_posix()
        proc = git.run(["git", "check-ignore", "-v", "--", rel])
        if proc.returncode == 0:
            ignored.append(rel)

    if ignored:
        return CheckResult(
            check_id,
            "fail",
            f"ไฟล์ความจำถูก .gitignore ซ่อน: {', '.join(ignored[:5])}"
            + (" …" if len(ignored) > 5 else ""),
        )

    tracked_proc = git.run(["git", "ls-files", ".project/"])
    tracked = {line.strip() for line in (tracked_proc.stdout or "").splitlines() if line.strip()}

    missing_required = [p for p in REQUIRED_PROJECT_FILES if p not in tracked]
    if missing_required:
        return CheckResult(
            check_id,
            "fail",
            f"ไฟล์หลักยังไม่ถูก git track: {', '.join(missing_required)}",
        )

    on_disk = {
        p.relative_to(git.repo).as_posix()
        for p in list_project_files(project)
    }
    untracked = sorted(on_disk - tracked)
    if untracked:
        sample = ", ".join(untracked[:5])
        suffix = " …" if len(untracked) > 5 else ""
        return CheckResult(
            check_id,
            "warn",
            f"มีไฟล์ .project/ ที่ยังไม่ track ({len(untracked)}): {sample}{suffix}",
        )

    return CheckResult(check_id, "ok", "ไฟล์ความจำหลักถูก git เก็บครบ")


def check_orphan_ids(repo: Path, plan_text: str | None) -> CheckResult:
    check_id = "orphan_ids"
    ledger_dir = repo / ".hermes" / "ai-relay"
    if not ledger_dir.is_dir():
        return CheckResult(check_id, "ok", "ไม่พบสมุด relay — ข้ามการตรวจเลขงาน")

    call_files = sorted(ledger_dir.glob("calls-*.md"))
    if not call_files:
        return CheckResult(check_id, "ok", "ไม่พบสมุด relay — ข้ามการตรวจเลขงาน")

    plan_ids = extract_plan_ids(plan_text)
    orphans: list[str] = []
    for call_file in call_files:
        rows = parse_md_table(call_file)
        if not rows:
            continue
        header = rows[0]
        data_rows = rows[1:] if header and "issue_id" in [c.lower() for c in header] else rows
        issue_col = 1
        if header:
            lowered = [c.lower() for c in header]
            if "issue_id" in lowered:
                issue_col = lowered.index("issue_id")
        for row in data_rows:
            if len(row) <= issue_col:
                continue
            issue_id = row[issue_col].strip()
            if not issue_id_allowed(issue_id, plan_ids):
                orphans.append(issue_id)

    if not orphans:
        return CheckResult(check_id, "ok", "เลขงานในสมุด relay สังกัด plan_id ที่รู้จัก")

    unique = []
    seen: set[str] = set()
    for oid in orphans:
        if oid not in seen:
            seen.add(oid)
            unique.append(oid)
    examples = ", ".join(unique[:5])
    return CheckResult(
        check_id,
        "warn",
        f"พบเลขงานกำพร้า {len(unique)} รายการ (ตัวอย่าง: {examples})",
    )


def aggregate_status(checks: list[CheckResult]) -> Status:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "ok"


def exit_code_for(status: Status) -> int:
    if status == "fail":
        return 1
    if status == "warn":
        return 2
    return 0


def status_icon(status: Status) -> str:
    return {"ok": "✅", "warn": "⚠️", "fail": "❌"}[status]


def print_human_report(checks: list[CheckResult], overall: Status) -> None:
    for check in checks:
        label = CHECK_LABELS.get(check.check_id, check.check_id)
        print(f"{status_icon(check.status)} [{label}] {check.message}")
    summary = {"ok": "ผ่านทุกด่าน", "warn": "มีเตือน ไม่มี fail", "fail": "พบความจำไม่ตรง git"}[overall]
    print(f"\nสรุป: {summary}")


def run_audit(repo: Path) -> tuple[list[CheckResult], Status, str | None]:
    git = GitRunner(repo)
    err = git.available()
    if err:
        return [], "fail", err

    project = repo / ".project"
    overview = read_text(project / "OverviewProgress.md")
    plan = read_text(project / "plan.md")

    checks = [
        check_schema(project, overview, plan),
        check_shas(git, overview, plan),
        check_git_tracking(git, project),
        check_orphan_ids(repo, plan),
    ]
    return checks, aggregate_status(checks), None


def main() -> int:
    parser = argparse.ArgumentParser(description="ตรวจความจำ .project/ เทียบ git")
    parser.add_argument(
        "--repo",
        default=".",
        help="พาธไปยังราก git repository (ค่าเริ่มต้น = โฟลเดอร์ปัจจุบัน)",
    )
    parser.add_argument("--json", action="store_true", help="พิมพ์ผลเป็น JSON")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        message = f"ไม่พบโฟลเดอร์ repository: {repo}"
        if args.json:
            print(json.dumps({"status": "error", "message": message, "checks": []}, ensure_ascii=False))
        else:
            print(f"❌ ข้อผิดพลาด: {message}", file=sys.stderr)
        return 3

    checks, overall, error = run_audit(repo)

    if error:
        payload = {"status": "error", "message": error, "checks": []}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"❌ ข้อผิดพลาด: {error}", file=sys.stderr)
        return 3

    if args.json:
        payload = {
            "status": {"ok": "pass", "warn": "warn", "fail": "fail"}[overall],
            "checks": [c.to_dict() for c in checks],
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print_human_report(checks, overall)

    return exit_code_for(overall)


if __name__ == "__main__":
    sys.exit(main())