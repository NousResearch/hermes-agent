"""Work preflight and closeout gates for Hermes projects.

The command is intentionally local-shell friendly: it checks the current
worktree, optional user service, optional URLs, and an optional git push before
an agent reports work as complete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class CommandResult:
    cmd: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


@dataclass
class GateCheck:
    name: str
    ok: bool
    detail: str
    evidence: str = ""
    recovery: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "issue": self.name,
            "status": "done" if self.ok else "blocked",
            "done_percent": 100 if self.ok else 0,
            "remaining_percent": 0 if self.ok else 100,
            "detail": self.detail,
            "evidence": self.evidence,
            "recovery": list(self.recovery),
        }


@dataclass
class GateReport:
    project: str
    phase: str
    ok: bool
    checks: list[GateCheck]
    next_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        total = len(self.checks)
        passed = sum(1 for check in self.checks if check.ok)
        done_percent = int(round((passed / total) * 100)) if total else 100
        return {
            "project": self.project,
            "phase": self.phase,
            "status": "pass" if self.ok else "fail",
            "completion": {
                "done_percent": done_percent,
                "remaining_percent": 100 - done_percent,
                "passed_issues": passed,
                "total_issues": total,
            },
            "issues": [check.to_dict() for check in self.checks],
            "next_actions": list(self.next_actions),
        }


Runner = Callable[[Sequence[str], Path | None, int], CommandResult]


def run_command(
    cmd: Sequence[str], cwd: Path | None = None, timeout: int = 30
) -> CommandResult:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(tuple(cmd), proc.returncode, proc.stdout, proc.stderr)
    except FileNotFoundError as exc:
        return CommandResult(tuple(cmd), 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            tuple(cmd),
            124,
            exc.stdout or "",
            exc.stderr or f"command timed out after {timeout}s",
        )


def _trim(text: str, limit: int = 600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _check_cmd(name: str, result: CommandResult, ok_detail: str) -> GateCheck:
    return GateCheck(
        name=name,
        ok=result.returncode == 0,
        detail=ok_detail
        if result.returncode == 0
        else f"command failed: {' '.join(result.cmd)}",
        evidence=_trim(result.output),
    )


def https_github_to_ssh(url: str) -> str | None:
    url = url.strip()
    prefix = "https://github.com/"
    if not url.startswith(prefix):
        return None
    repo = url[len(prefix) :]
    if not repo.endswith(".git"):
        repo += ".git"
    if not repo or "/" not in repo:
        return None
    return f"git@github.com:{repo}"


def _github_ssh_auth_ok(result: CommandResult) -> bool:
    output = result.output.lower()
    return (
        "successfully authenticated" in output
        or "you've successfully authenticated" in output
    )


def check_git_worktree(cwd: Path, runner: Runner = run_command) -> list[GateCheck]:
    checks: list[GateCheck] = []
    root = runner(["git", "rev-parse", "--show-toplevel"], cwd, 30)
    checks.append(_check_cmd("git_root", root, f"git root: {root.stdout.strip()}"))
    if root.returncode != 0:
        return checks

    branch = runner(["git", "branch", "--show-current"], cwd, 30)
    checks.append(_check_cmd("git_branch", branch, f"branch: {branch.stdout.strip()}"))

    status = runner(["git", "status", "--porcelain"], cwd, 30)
    clean = status.returncode == 0 and status.stdout.strip() == ""
    checks.append(
        GateCheck(
            "git_clean",
            clean,
            "worktree clean" if clean else "worktree has uncommitted files",
            _trim(status.output),
        )
    )
    return checks


def check_git_remote(cwd: Path, remote: str, runner: Runner = run_command) -> GateCheck:
    result = runner(["git", "remote", "get-url", remote], cwd, 30)
    if result.returncode == 0:
        return GateCheck(
            "git_remote",
            True,
            f"{remote}: {result.stdout.strip()}",
            result.stdout.strip(),
        )
    return GateCheck(
        "git_remote", False, f"remote '{remote}' is missing", _trim(result.output)
    )


def _push_failure_next_action(output: str, remote_url: str = "") -> str:
    lower = output.lower()
    if "permission denied (publickey)" in lower:
        return "ตั้งค่า SSH deploy key หรือสิทธิ์ SSH ของ remote ให้ push ได้ แล้วรัน closeout ใหม่"
    if "could not read username" in lower or "authentication failed" in lower:
        return "เปลี่ยน remote จาก HTTPS เป็น SSH หรือเติม Git credential ที่ push ได้ แล้วรัน closeout ใหม่"
    if "non-fast-forward" in lower or "fetch first" in lower:
        return "ดึง remote ล่าสุดและแก้ conflict ก่อน push ใหม่"
    if remote_url:
        return f"ตรวจสิทธิ์ remote {remote_url} แล้วรัน git push ใหม่"
    return "ตรวจ error ของ git push แล้วรัน closeout ใหม่"


def git_push_with_recovery(
    cwd: Path,
    remote: str,
    branch: str,
    runner: Runner = run_command,
    auto_recover: bool = True,
) -> GateCheck:
    first = runner(["git", "push", remote, branch], cwd, 120)
    if first.returncode == 0:
        return GateCheck(
            "git_push", True, f"pushed {remote} {branch}", _trim(first.output)
        )

    recovery: list[str] = []
    remote_result = runner(["git", "remote", "get-url", remote], cwd, 30)
    remote_url = remote_result.stdout.strip() if remote_result.returncode == 0 else ""
    first_output = first.output
    ssh_url = https_github_to_ssh(remote_url)

    if (
        auto_recover
        and ssh_url
        and (
            "could not read Username" in first_output
            or "Authentication failed" in first_output
        )
    ):
        auth = runner(["ssh", "-T", "-o", "BatchMode=yes", "git@github.com"], cwd, 30)
        if auth.returncode == 0 or _github_ssh_auth_ok(auth):
            set_url = runner(["git", "remote", "set-url", remote, ssh_url], cwd, 30)
            if set_url.returncode == 0:
                recovery.append(f"remote {remote}: https -> ssh ({ssh_url})")
                retry = runner(["git", "push", remote, branch], cwd, 120)
                if retry.returncode == 0:
                    return GateCheck(
                        "git_push",
                        True,
                        f"pushed {remote} {branch}; recovered HTTPS credential failure",
                        _trim(retry.output),
                        recovery,
                    )
                recovery.append(f"retry failed: {_trim(retry.output)}")
                first_output = retry.output
            else:
                recovery.append(f"remote set-url failed: {_trim(set_url.output)}")
        else:
            recovery.append(f"ssh auth failed: {_trim(auth.output)}")

    next_action = _push_failure_next_action(first_output, remote_url)
    return GateCheck(
        "git_push",
        False,
        next_action,
        _trim(first_output),
        recovery,
    )


def check_systemd_user_service(
    service: str,
    runner: Runner = run_command,
) -> GateCheck:
    result = runner(["systemctl", "--user", "is-active", service], None, 30)
    active = result.returncode == 0 and result.stdout.strip() == "active"
    return GateCheck(
        "service",
        active,
        f"{service} active" if active else f"{service} is not active",
        _trim(result.output),
    )


def check_http_url(url: str) -> GateCheck:
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "hermes-closeout/1"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            status = getattr(response, "status", response.getcode())
            ok = 200 <= int(status) < 400
            return GateCheck("url", ok, f"HTTP {status} {url}", url)
    except urllib.error.HTTPError as exc:
        return GateCheck("url", False, f"HTTP {exc.code} {url}", _trim(str(exc)))
    except Exception as exc:  # noqa: BLE001 - surfaced as closeout evidence.
        return GateCheck("url", False, f"URL check failed: {url}", _trim(str(exc)))


def _next_actions(checks: Iterable[GateCheck]) -> list[str]:
    actions: list[str] = []
    for check in checks:
        if check.ok:
            continue
        if check.detail and check.detail not in actions:
            actions.append(check.detail)
        for recovery in check.recovery:
            if recovery and recovery not in actions:
                actions.append(recovery)
    return actions


def run_preflight(
    *,
    cwd: str | Path,
    project: str,
    service: str | None = None,
    urls: list[str] | None = None,
    remote: str | None = None,
    runner: Runner = run_command,
) -> GateReport:
    root = Path(cwd).expanduser().resolve()
    checks = check_git_worktree(root, runner)
    if remote:
        checks.append(check_git_remote(root, remote, runner))
    if service:
        checks.append(check_systemd_user_service(service, runner))
    for url in urls or []:
        checks.append(check_http_url(url))
    ok = all(check.ok for check in checks)
    return GateReport(
        project=project,
        phase="preflight",
        ok=ok,
        checks=checks,
        next_actions=_next_actions(checks),
    )


def run_closeout(
    *,
    cwd: str | Path,
    project: str,
    service: str | None = None,
    urls: list[str] | None = None,
    remote: str = "origin",
    branch: str | None = None,
    push: bool = False,
    runner: Runner = run_command,
) -> GateReport:
    root = Path(cwd).expanduser().resolve()
    checks = check_git_worktree(root, runner)
    checks.append(check_git_remote(root, remote, runner))
    if service:
        checks.append(check_systemd_user_service(service, runner))
    for url in urls or []:
        checks.append(check_http_url(url))

    branch_name = branch
    if not branch_name:
        branch_result = runner(["git", "branch", "--show-current"], root, 30)
        branch_name = (
            branch_result.stdout.strip() if branch_result.returncode == 0 else ""
        )
    if push:
        if branch_name:
            checks.append(git_push_with_recovery(root, remote, branch_name, runner))
        else:
            checks.append(GateCheck("git_push", False, "ไม่พบ branch สำหรับ push", ""))

    ok = all(check.ok for check in checks)
    return GateReport(
        project=project,
        phase="closeout",
        ok=ok,
        checks=checks,
        next_actions=_next_actions(checks),
    )


def _print_human(report: GateReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Hermes work {report.phase}: {status}")
    for check in report.checks:
        marker = "OK" if check.ok else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")
        if check.evidence:
            print(f"      evidence: {check.evidence}")
        for item in check.recovery:
            print(f"      recovery: {item}")
    if report.next_actions:
        print("Next actions:")
        for action in report.next_actions:
            print(f"- {action}")


def _emit(report: GateReport, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human(report)


def cmd_work(args: argparse.Namespace) -> None:
    action = getattr(args, "work_action", None) or "preflight"
    if action == "preflight":
        report = run_preflight(
            cwd=args.cwd,
            project=args.project,
            service=args.service,
            urls=args.urls,
            remote=args.remote,
        )
    elif action == "closeout":
        report = run_closeout(
            cwd=args.cwd,
            project=args.project,
            service=args.service,
            urls=args.urls,
            remote=args.remote,
            branch=args.branch,
            push=args.push,
        )
    else:
        raise SystemExit(f"unknown work action: {action}")

    _emit(report, args.as_json)
    raise SystemExit(0 if report.ok else 1)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        default="hermes-agent",
        help="Project label for the closeout report",
    )
    parser.add_argument("--cwd", default=".", help="Repository/worktree path to check")
    parser.add_argument(
        "--service", default=None, help="systemd --user service that must be active"
    )
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        default=[],
        help="HTTP URL that must return 2xx/3xx; repeatable",
    )
    parser.add_argument(
        "--remote", default="origin", help="Git remote to verify/push (default: origin)"
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit machine-readable JSON"
    )


def register_work_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "work",
        help="Run work preflight and closeout gates",
        description="Check git, service, URL, and optional push status before reporting work complete.",
    )
    _add_common_args(parser)
    parser.set_defaults(func=cmd_work, work_action="preflight")
    subs = parser.add_subparsers(dest="work_action", metavar="ACTION")

    preflight = subs.add_parser(
        "preflight", help="Check worktree, service, URLs, and remote"
    )
    _add_common_args(preflight)
    preflight.set_defaults(func=cmd_work, work_action="preflight")

    closeout = subs.add_parser(
        "closeout", help="Run final gate before handing work to the user"
    )
    _add_common_args(closeout)
    closeout.add_argument(
        "--branch", default=None, help="Branch to push; defaults to current branch"
    )
    closeout.add_argument(
        "--push", action="store_true", help="Require git push to succeed"
    )
    closeout.set_defaults(func=cmd_work, work_action="closeout")
    return parser


__all__ = [
    "CommandResult",
    "GateCheck",
    "GateReport",
    "check_git_remote",
    "check_git_worktree",
    "check_http_url",
    "check_systemd_user_service",
    "cmd_work",
    "git_push_with_recovery",
    "https_github_to_ssh",
    "register_work_subparser",
    "run_closeout",
    "run_command",
    "run_preflight",
]
