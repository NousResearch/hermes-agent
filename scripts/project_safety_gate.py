#!/usr/bin/env python3
"""Read-only safety gate for high-risk project migrations.

This script inspects WebEngine and ViberQC from outside their repositories.
It does not run package installs, builds, tests, deploys, or remote-changing git
commands. It also does not read `.env` files; it records only whether env-like
files are tracked or present.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


WORKSPACE = Path("/Users/rattanasak/Documents/Viber Project")
PROJECTS = {
    "webengine": {
        "name": "Master WebEngine",
        "path": WORKSPACE / "Office Project" / "Master WebEngine",
        "app_path": WORKSPACE / "Office Project" / "Master WebEngine" / "synerry-engine",
        "required_scripts": [
            "build",
            "lint",
            "test",
            "production:readiness",
            "migration:check",
        ],
        "required_files": [
            "CLAUDE.md",
            "MEMORY.md",
            ".gitlab-ci.yml",
            "synerry-engine/package.json",
            "synerry-engine/pnpm-lock.yaml",
            "synerry-engine/turbo.json",
        ],
    },
    "viberqc": {
        "name": "Master ViberQC",
        "path": WORKSPACE / "SaaS Project" / "Master ViberQC",
        "app_path": WORKSPACE / "SaaS Project" / "Master ViberQC",
        "required_scripts": [
            "build",
            "lint",
            "test",
            "verify:localhost",
            "verify:vps",
            "verify:release",
            "release:check",
            "phase:comply",
        ],
        "required_files": [
            "CLAUDE.md",
            "MEMORY.md",
            ".gitlab-ci.yml",
            "package.json",
            "package-lock.json",
            "next.config.ts",
            "scripts/verify-build.sh",
            "scripts/release-check.js",
        ],
    },
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    ".next",
    "dist",
    "coverage",
    "playwright-report",
    "test-results",
    ".turbo",
}

HEAVY_TRACKED_RE = re.compile(
    r"(^|/)(node_modules|\.next|dist|coverage|playwright-report|test-results|uploads)(/|$)"
)
ENV_FILE_RE = re.compile(r"(^|/)\.env($|\.)")
SECRET_DEFAULT_RE = re.compile(
    r"(PASSWORD|SECRET|TOKEN|API_KEY|DATABASE_URL|REDIS_PASSWORD|POSTGRES_PASSWORD|NEXTAUTH_SECRET)"
)
COMPOSE_DEFAULT_RE = re.compile(r"\$\{([A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|API_KEY|DATABASE_URL)[A-Z0-9_]*)\:-([^}]+)\}")
SECRET_ASSIGN_RE = re.compile(
    r"^\s*(?:const\s+|let\s+|var\s+)?([A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|API_KEY)[A-Z0-9_]*)\s*[:=]",
    re.IGNORECASE,
)
MANDATORY_CI_JOBS = {
    "secret-scan",
    "env-check",
    "install-and-build",
    "test-root",
    "test-engine",
    "lint-root",
    "lint-engine",
    "lint-module-pair",
    "lint-openapi-sync",
    "lint-migration-checklist",
    "quality-gate",
}


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    evidence: str = ""


@dataclass
class ProjectReport:
    key: str
    name: str
    path: str
    findings: list[Finding] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str, evidence: str = "") -> None:
        self.findings.append(Finding(severity, code, message, evidence))


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive for local env issues
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout.strip()


def safe_read_text(path: Path, max_bytes: int = 300_000) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def load_package_scripts(path: Path) -> dict[str, str]:
    try:
        package = json.loads(path.read_text())
    except Exception:
        return {}
    scripts = package.get("scripts", {})
    return scripts if isinstance(scripts, dict) else {}


def git_lines(path: Path, args: list[str]) -> list[str]:
    code, out = run(["git", *args], path)
    if code != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def iter_files(root: Path):
    for child in root.iterdir() if root.exists() else []:
        if child.name in SKIP_DIRS:
            continue
        if child.is_dir():
            yield from iter_files(child)
        else:
            yield child


def inspect_git(report: ProjectReport, root: Path) -> None:
    status_lines = git_lines(root, ["status", "--short", "--branch"])
    report.data["git_status"] = status_lines[:200]

    if not status_lines:
        report.add("critical", "git-status-unavailable", "Cannot read git status.", str(root))
        return

    branch_line = status_lines[0]
    report.data["branch_line"] = branch_line
    dirty = [line for line in status_lines[1:] if line.strip()]
    report.data["dirty_count"] = len(dirty)
    report.data["dirty_sample"] = dirty[:30]

    if dirty:
        report.add(
            "critical",
            "dirty-worktree",
            "Worktree has local changes. Do not push or deploy until reviewed.",
            f"{len(dirty)} changed/untracked entries",
        )

    if "behind" in branch_line or "ahead" in branch_line:
        report.add(
            "critical",
            "branch-diverged",
            "Local branch is ahead/behind its upstream. Resolve before deploy.",
            branch_line,
        )

    branch_lines = git_lines(root, ["branch", "-vv"])
    report.data["branches"] = branch_lines[:80]
    gone = [line for line in branch_lines if ": gone]" in line]
    if gone:
        report.add(
            "warning",
            "gone-upstream-branches",
            "Local branches point to removed upstream branches.",
            "; ".join(gone[:5]),
        )

    remotes = git_lines(root, ["remote", "-v"])
    report.data["remotes"] = remotes
    remote_branches = git_lines(root, ["branch", "-r"])
    report.data["remote_branches_sample"] = remote_branches[:80]
    if any(line.strip().endswith("/master") for line in remote_branches):
        report.add(
            "warning",
            "remote-master-present",
            "Remote still has a master branch. Confirm deploy branch source of truth.",
            "origin/master present",
        )

    worktrees = git_lines(root, ["worktree", "list", "--porcelain"])
    report.data["worktrees"] = worktrees[:120]
    if len([line for line in worktrees if line.startswith("worktree ")]) > 1:
        report.add(
            "info",
            "multiple-worktrees",
            "Multiple worktrees exist. Ensure the active worktree is the intended one.",
            str(len([line for line in worktrees if line.startswith("worktree ")])),
        )


def inspect_required_files(report: ProjectReport, root: Path, required: list[str]) -> None:
    missing = [rel for rel in required if not (root / rel).exists()]
    report.data["missing_required_files"] = missing
    if missing:
        report.add("critical", "missing-required-files", "Required project files are missing.", ", ".join(missing))


def inspect_package_scripts(report: ProjectReport, package_path: Path, required_scripts: list[str]) -> None:
    scripts = load_package_scripts(package_path)
    report.data["package_scripts"] = sorted(scripts)
    missing = [name for name in required_scripts if name not in scripts]
    if missing:
        report.add("warning", "missing-package-scripts", "Expected safety scripts are missing.", ", ".join(missing))


def inspect_tracked_files(report: ProjectReport, root: Path) -> None:
    tracked = git_lines(root, ["ls-files"])
    report.data["tracked_file_count"] = len(tracked)

    tracked_env = [f for f in tracked if ENV_FILE_RE.search(f) and not f.endswith(".example")]
    report.data["tracked_env_files"] = tracked_env
    if tracked_env:
        report.add("critical", "tracked-env-files", "Non-example env files are tracked by git.", ", ".join(tracked_env))

    tracked_heavy = [f for f in tracked if HEAVY_TRACKED_RE.search(f)]
    report.data["tracked_heavy_count"] = len(tracked_heavy)
    report.data["tracked_heavy_sample"] = tracked_heavy[:30]
    if tracked_heavy:
        report.add(
            "warning",
            "tracked-generated-artifacts",
            "Generated or heavy artifacts are tracked and can pollute merges/deploys.",
            f"{len(tracked_heavy)} tracked entries",
        )


def inspect_ci(report: ProjectReport, root: Path) -> None:
    ci = safe_read_text(root / ".gitlab-ci.yml")
    if not ci:
        report.add("warning", "missing-ci", "No .gitlab-ci.yml found.", "")
        return

    job_blocks = {}
    current_name = ""
    current_lines: list[str] = []
    for line in ci.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+):\s*$", line)
        if match and not line.startswith("."):
            if current_name:
                job_blocks[current_name] = "\n".join(current_lines)
            current_name = match.group(1)
            current_lines = []
            continue
        if current_name:
            current_lines.append(line)
    if current_name:
        job_blocks[current_name] = "\n".join(current_lines)

    disabled_mandatory = [
        job
        for job in sorted(MANDATORY_CI_JOBS)
        if job in job_blocks and ("fast-deploy-disabled-rules" in job_blocks[job] or "when: never" in job_blocks[job])
    ]
    report.data["disabled_mandatory_ci_jobs"] = disabled_mandatory
    if disabled_mandatory:
        report.add(
            "critical",
            "ci-gates-disabled",
            "Mandatory CI gates are disabled. Do not rely on CI as a production gate until fixed.",
            ", ".join(disabled_mandatory),
        )
    elif root.name == "Master WebEngine":
        report.add(
            "info",
            "ci-core-gates-enabled",
            "Mandatory WebEngine CI gates are enabled for merge requests and main.",
            ", ".join(sorted(MANDATORY_CI_JOBS)),
        )

    mandatory_allow_failure = [
        job for job in sorted(MANDATORY_CI_JOBS) if job in job_blocks and "allow_failure: true" in job_blocks[job]
    ]
    report.data["mandatory_allow_failure_jobs"] = mandatory_allow_failure
    if mandatory_allow_failure:
        report.add(
            "critical",
            "ci-allows-failure",
            "Mandatory CI jobs allow failure. They must block merge/deploy.",
            ", ".join(mandatory_allow_failure),
        )
    elif "allow_failure: true" in ci:
        report.add(
            "info",
            "ci-advisory-jobs-present",
            "Optional CI jobs still allow failure; mandatory gates are blocking.",
            ".gitlab-ci.yml",
        )

    if "sudo -u linux-nat" in ci or "deploy.sh" in ci:
        report.add(
            "info",
            "ci-production-deploy-path",
            "CI has production deploy path. Confirm manual approval and protected branch behavior.",
            ".gitlab-ci.yml",
        )


def inspect_secret_defaults(report: ProjectReport, root: Path) -> None:
    hits: list[str] = []
    tracked_files = git_lines(root, ["ls-files"])
    ignored_prefixes = (
        ".hermes/",
        "_demo-source/",
        "Docs/",
        "Docs_Project/",
        "QA-QC_Master/",
        "TOR_Projects/",
    )
    ignored_parts = ("/tests/", "/__tests__/", "/test/", "/coverage/")
    candidates = [
        root / rel
        for rel in tracked_files
        if not rel.endswith(".md")
        and not rel.endswith(".spec.ts")
        and not rel.endswith(".test.ts")
        and not rel.startswith(ignored_prefixes)
        and not any(part in rel for part in ignored_parts)
        and not rel.startswith("scripts/test-")
        and "/scripts/test-" not in rel
    ]
    for file_path in candidates:
        rel = file_path.relative_to(root).as_posix()
        if not re.search(r"(\.ya?ml|\.ts|\.tsx|\.js|\.mjs|\.sh|Dockerfile|\.md)$", rel):
            continue
        text = safe_read_text(file_path)
        if not text:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            default_match = COMPOSE_DEFAULT_RE.search(line)
            if default_match:
                fallback = default_match.group(2).strip().strip("\"'")
                if fallback not in {"", "0", "1", "true", "false"}:
                    hits.append(f"{rel}:{line_no}")
                    break
            assign_match = SECRET_ASSIGN_RE.search(line)
            if assign_match and SECRET_DEFAULT_RE.search(line):
                name = assign_match.group(1).upper()
                rhs = re.split(r"[:=]", line, maxsplit=1)[-1].strip()
                has_literal_secret = bool(re.search(r"[:=]\s*['\"][^'\"]{6,}", line))
                safe_reference = (
                    "${" in rhs
                    or "$" in rhs
                    or "process.env" in rhs
                    or rhs in {"''", '""', "``", ""}
                    or "_TO_" in name
                    or name.endswith("_MAP")
                    or "PATTERN" in name
                    or name in {"SECRETS", "FOUND_SECRETS"}
                    or name.startswith("TOKENS_PER")
                )
                if has_literal_secret and not safe_reference:
                    hits.append(f"{rel}:{line_no}")
                    break
    report.data["secret_like_locations"] = hits[:80]
    if hits:
        report.add(
            "warning",
            "secret-like-defaults",
            "Secret-like defaults or assignments exist in tracked files. Review and move real values to secret manager.",
            f"{len(hits)} files/locations",
        )


def inspect_viberqc_specific(report: ProjectReport, root: Path) -> None:
    nested = root / "viberqc-central" / "app"
    if nested.exists():
        report.add(
            "warning",
            "nested-app-present",
            "Nested ViberQC app exists. Decide whether it deploys separately before staging.",
            "viberqc-central/app",
        )
        nested_scripts = load_package_scripts(nested / "package.json")
        report.data["nested_app_scripts"] = sorted(nested_scripts)

    verify_build = safe_read_text(root / "scripts" / "verify-build.sh")
    if verify_build:
        if "set -e" in verify_build and "pipefail" not in verify_build:
            report.add(
                "warning",
                "verify-build-no-pipefail",
                "verify-build.sh uses pipelines without pipefail; some failures may be hidden.",
                "scripts/verify-build.sh",
            )
        if "tail -20 | grep" in verify_build:
            report.add(
                "warning",
                "verify-build-tail-grep",
                "Build failure detection relies on tail/grep. Replace with direct exit-code capture.",
                "scripts/verify-build.sh",
            )
        if "next dev" in verify_build and "start:standalone" not in verify_build:
            report.add(
                "warning",
                "verify-build-dev-server-only",
                "Smoke test uses dev server only. Add standalone smoke before production deploy.",
                "scripts/verify-build.sh",
            )


def inspect_webengine_specific(report: ProjectReport, root: Path) -> None:
    ci = safe_read_text(root / ".gitlab-ci.yml")
    if "GIT_CLEAN_FLAGS" in ci and ".next" in ci:
        if "chown -R" in ci and "fetch-cache" in ci:
            report.add(
                "info",
                "ci-next-cache-ownership-guard",
                "CI preserves .next caches but includes ownership and fetch-cache cleanup guards.",
                ".gitlab-ci.yml",
            )
        else:
            report.add(
                "warning",
                "ci-preserves-next-cache",
                "CI clean flags preserve .next caches. Confirm this cannot hide stale build/ownership issues.",
                ".gitlab-ci.yml",
            )

    wtg_scripts = [
        "synerry-engine/scripts/wtg-6-pre-migration-check.sh",
        "synerry-engine/scripts/wtg-6-backup.sh",
        "synerry-engine/scripts/wtg-6-rollback.sh",
    ]
    missing = [rel for rel in wtg_scripts if not (root / rel).exists()]
    report.data["missing_wtg_scripts"] = missing
    if missing:
        report.add("critical", "missing-wtg-safety-scripts", "WTG migration safety scripts are missing.", ", ".join(missing))


def inspect_project(key: str) -> ProjectReport:
    cfg = PROJECTS[key]
    root = cfg["path"]
    app_path = cfg["app_path"]
    report = ProjectReport(key=key, name=cfg["name"], path=str(root))

    if not root.exists():
        report.add("critical", "missing-project-path", "Project path does not exist.", str(root))
        return report

    inspect_git(report, root)
    inspect_required_files(report, root, cfg["required_files"])
    inspect_tracked_files(report, root)
    inspect_ci(report, root)
    inspect_secret_defaults(report, root)

    package_path = app_path / "package.json"
    inspect_package_scripts(report, package_path, cfg["required_scripts"])
    report.data["package_path"] = str(package_path)

    if key == "viberqc":
        inspect_viberqc_specific(report, root)
    elif key == "webengine":
        inspect_webengine_specific(report, root)

    return report


def severity_score(severity: str) -> int:
    return {"critical": 3, "warning": 2, "info": 1}.get(severity, 0)


def as_dict(report: ProjectReport) -> dict[str, Any]:
    counts = {"critical": 0, "warning": 0, "info": 0}
    for finding in report.findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return {
        "key": report.key,
        "name": report.name,
        "path": report.path,
        "counts": counts,
        "findings": [finding.__dict__ for finding in sorted(report.findings, key=lambda f: -severity_score(f.severity))],
        "data": report.data,
    }


def print_markdown(reports: list[ProjectReport]) -> None:
    print("# Project Safety Gate Report\n")
    print("Mode: read-only. No build, test, deploy, install, or secret reads were performed.\n")
    for report in reports:
        data = as_dict(report)
        counts = data["counts"]
        print(f"## {report.name}\n")
        print(f"- Path: `{report.path}`")
        print(f"- Critical: `{counts['critical']}`")
        print(f"- Warning: `{counts['warning']}`")
        print(f"- Info: `{counts['info']}`\n")
        print("| Severity | Code | Message | Evidence |")
        print("|---|---|---|---|")
        for finding in data["findings"]:
            evidence = finding["evidence"].replace("|", "\\|")
            print(f"| {finding['severity']} | `{finding['code']}` | {finding['message']} | {evidence} |")
        print()

        dirty = report.data.get("dirty_count")
        if dirty is not None:
            print(f"- Dirty entries: `{dirty}`")
        heavy = report.data.get("tracked_heavy_count")
        if heavy is not None:
            print(f"- Tracked generated/heavy entries: `{heavy}`")
        scripts = report.data.get("package_scripts", [])
        if scripts:
            print(f"- Package scripts detected: `{len(scripts)}`")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=[*PROJECTS.keys(), "all"], default="all")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    keys = list(PROJECTS) if args.project == "all" else [args.project]
    reports = [inspect_project(key) for key in keys]

    if args.format == "json":
        print(json.dumps([as_dict(report) for report in reports], ensure_ascii=False, indent=2))
    else:
        print_markdown(reports)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
