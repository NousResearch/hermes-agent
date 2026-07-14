#!/usr/bin/env python3
"""One-command New Chat bootstrap for team work on the Hermes VPS.

Given a staff id, project, and short task, this resolves the project's
reference repo, then creates one branch and one worktree following the locked
team standard:

    branch   : team/<staff>/<project>/<YYYYMMDD>-<task>
    worktree : /home/linux-nat/.worktree/<project>/<staff>/<YYYYMMDD>-<task>

It also reserves a free service port, sets the per-worktree git identity, and
prints a report. It is dry-run by default; pass --apply to make changes.

Run this ON the VPS (date and ports are read from the local machine). To test
remotely without changes:

    cat scripts/hermes_newchat.py | ssh linux-nat@<vps> \\
        'python3 - --staff nat --project lotto-reward --task dashboard-fix'
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess

WORKTREE_ROOT = "/home/linux-nat/.worktree"
REFERENCE_ROOTS = ("/home/linux-nat/projects", "/srv/projects")
PORT_RANGE = (8100, 8999)
EMAIL_DOMAIN = "jigsawgroups.work"


def slugify(value: str) -> str:
    """Lowercase kebab slug: 'Lotto Reward' -> 'lotto-reward'."""
    out = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return out.strip("-")


def norm(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def run_git(path: str, *args: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", path, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def is_git_toplevel(path: str) -> bool:
    top = run_git(path, "rev-parse", "--show-toplevel")
    return bool(top) and os.path.realpath(top) == os.path.realpath(path)


def resolve_reference_repo(project: str) -> str | None:
    """Find the project's reference clone. Prefer <dir>/main, else <dir>."""
    want = norm(project)
    for root in REFERENCE_ROOTS:
        if not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for name in entries:
            if name.startswith("."):
                continue
            if norm(name) != want:
                continue
            base = os.path.join(root, name)
            main = os.path.join(base, "main")
            if is_git_toplevel(main):
                return main
            if is_git_toplevel(base):
                return base
    return None


def default_branch(ref_repo: str) -> str:
    head = run_git(ref_repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if head and "/" in head:
        return head.split("/", 1)[1]
    for candidate in ("master", "main"):
        if run_git(ref_repo, "rev-parse", "--verify", f"origin/{candidate}") is not None:
            return candidate
    return "master"


def used_ports() -> set[int]:
    ports: set[int] = set()
    ss = subprocess.run(
        ["ss", "-H", "-ltn"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for line in ss.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        tail = fields[3].rsplit(":", 1)[-1]
        if tail.isdigit():
            ports.add(int(tail))
    docker = subprocess.run(
        ["docker", "ps", "--format", "{{.Ports}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for match in re.finditer(r":(\d+)->", docker.stdout):
        ports.add(int(match.group(1)))
    return ports


def pick_free_port() -> int | None:
    busy = used_ports()
    for port in range(PORT_RANGE[0], PORT_RANGE[1] + 1):
        if port not in busy:
            return port
    return None


def build_plan(staff: str, project: str, task: str) -> dict:
    staff_slug = slugify(staff)
    project_slug = slugify(project)
    task_slug = slugify(task)
    date = datetime.date.today().strftime("%Y%m%d")
    leaf = f"{date}-{task_slug}"
    branch = f"team/{staff_slug}/{project_slug}/{leaf}"
    worktree = os.path.join(WORKTREE_ROOT, project_slug, staff_slug, leaf)
    return {
        "staff": staff_slug,
        "project": project_slug,
        "task": task_slug,
        "date": date,
        "branch": branch,
        "worktree": worktree,
        "git_user_name": staff_slug,
        "git_user_email": f"{staff_slug}@{EMAIL_DOMAIN}",
    }


def apply_plan(plan: dict, ref_repo: str, base: str, port: int | None) -> list[str]:
    log: list[str] = []

    def git(*args: str) -> None:
        proc = subprocess.run(
            ["git", "-C", ref_repo, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.append(f"$ git -C {ref_repo} {' '.join(args)}\n{proc.stdout.strip()}")
        if proc.returncode != 0:
            raise SystemExit(f"git failed: {' '.join(args)}")

    git("fetch", "origin", base, "--prune")
    os.makedirs(os.path.dirname(plan["worktree"]), exist_ok=True)
    git("worktree", "add", "-b", plan["branch"], plan["worktree"], f"origin/{base}")

    wt = plan["worktree"]
    subprocess.run(["git", "-C", wt, "config", "user.name", plan["git_user_name"]])
    subprocess.run(["git", "-C", wt, "config", "user.email", plan["git_user_email"]])
    log.append(f"git identity set: {plan['git_user_name']} <{plan['git_user_email']}>")
    if port is not None:
        with open(os.path.join(wt, ".port"), "w", encoding="utf-8") as fh:
            fh.write(f"{port}\n")
        log.append(f".port written: {port}")
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a team branch + worktree.")
    parser.add_argument("--staff", required=True, help="Staff id, e.g. nat.")
    parser.add_argument("--project", required=True, help="Project, e.g. lotto-reward.")
    parser.add_argument("--task", required=True, help="Short task, e.g. dashboard-fix.")
    parser.add_argument("--task-id", help="Use WTL Manager with this stable task id instead of the legacy creator.")
    parser.add_argument("--root", default=WORKTREE_ROOT, help="Registered worktree root for WTL Manager.")
    parser.add_argument("--registry", help="Worktree Lifecycle registry JSON.")
    parser.add_argument("--machine-id", help="Stable Notebook/VPS machine id for WTL Manager.")
    parser.add_argument("--base", help="Base branch override (default: repo default).")
    parser.add_argument("--apply", action="store_true", help="Make changes (default dry-run).")
    args = parser.parse_args()

    plan = build_plan(args.staff, args.project, args.task)
    ref_repo = resolve_reference_repo(args.project)
    report: dict = {"plan": plan, "reference_repo": ref_repo}

    if ref_repo is None:
        report["error"] = "reference_repo_not_found"
        report["searched"] = list(REFERENCE_ROOTS)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    if args.task_id:
        # Compatibility bridge: existing callers may keep using this script,
        # while all new creation/state is delegated to the central Manager.
        from argparse import Namespace
        from hermes_cli.worktree_lifecycle import command_open

        managed = command_open(Namespace(
            project_id=plan["project"], staff_id=plan["staff"], task_id=args.task_id,
            slug=plan["task"], repo=ref_repo, root=args.root, registry=args.registry,
            machine_id=args.machine_id, remote="origin", base_branch=args.base or default_branch(ref_repo),
            lease_hours=12, allow_over_limit=False, apply=args.apply, as_json=True,
        ))
        managed["compatibility_entrypoint"] = "scripts/hermes_newchat.py"
        managed["manager"] = "hermes worktree open"
        print(json.dumps(managed, ensure_ascii=False, indent=2))
        return 0 if managed.get("ok") else 2

    base = args.base or default_branch(ref_repo)
    report["base_branch"] = base

    branch_exists = run_git(ref_repo, "rev-parse", "--verify", plan["branch"]) is not None
    report["branch_exists"] = branch_exists
    report["worktree_exists"] = os.path.exists(plan["worktree"])
    port = pick_free_port()
    report["reserved_port"] = port

    if not args.apply:
        report["mode"] = "dry-run"
        report["note"] = "nothing created; pass --apply to create branch + worktree"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if branch_exists or report["worktree_exists"]:
        report["error"] = "branch_or_worktree_already_exists"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    report["mode"] = "apply"
    report["log"] = apply_plan(plan, ref_repo, base, port)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
