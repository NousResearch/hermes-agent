#!/usr/bin/env python3
"""Exclusive write permit for one registered workspace.

The permit lives outside the repository so two chats using the same folder see
the same owner.  It is intentionally small and uses only the Python standard
library.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def state_paths(cwd: Path) -> tuple[Path, Path]:
    root = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    key = hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()[:20]
    folder = root / "write-permits"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.json", folder / f"{key}.lock"


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    expires = dt.datetime.fromisoformat(data["expires_at"])
    return None if expires <= now() else data


def write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normal_paths(values: list[str]) -> list[str]:
    result = []
    for value in values:
        item = Path(value)
        if item.is_absolute() or ".." in item.parts:
            raise ValueError(f"allowed path ต้องอยู่ภายใน repo: {value}")
        result.append(item.as_posix().rstrip("/"))
    return sorted(set(result))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("acquire", "check", "release", "status"))
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--branch")
    parser.add_argument("--base-sha")
    parser.add_argument("--allowed-path", action="append", default=[])
    parser.add_argument("--approval")
    parser.add_argument("--ttl-minutes", type=int, default=120)
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    actual_root = Path(git(cwd, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != cwd:
        cwd = actual_root
    actual_branch = git(cwd, "branch", "--show-current")
    actual_sha = git(cwd, "rev-parse", "HEAD")
    state_path, lock_path = state_paths(cwd)

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        active = load(state_path)

        if args.action == "status":
            print(json.dumps({"ok": True, "active": active}, ensure_ascii=False))
            return 0

        if args.action == "release":
            if active and active.get("task_id") != args.task_id:
                print(json.dumps({"ok": False, "reason": "owned_by_other_task", "active": active}, ensure_ascii=False))
                return 2
            state_path.unlink(missing_ok=True)
            print(json.dumps({"ok": True, "released": args.task_id}, ensure_ascii=False))
            return 0

        required = (args.task_id, args.branch, args.base_sha)
        if not all(required):
            parser.error("acquire/check ต้องมี --task-id --branch --base-sha")
        paths = normal_paths(args.allowed_path)
        if actual_branch != args.branch or actual_sha != args.base_sha:
            print(json.dumps({"ok": False, "reason": "git_state_changed", "actual_branch": actual_branch, "actual_sha": actual_sha}, ensure_ascii=False))
            return 2

        if args.action == "check":
            expected = {"task_id": args.task_id, "branch": args.branch, "base_sha": args.base_sha, "allowed_paths": paths}
            ok = bool(active) and all(active.get(k) == v for k, v in expected.items())
            print(json.dumps({"ok": ok, "active": active}, ensure_ascii=False))
            return 0 if ok else 2

        if not args.approval:
            parser.error("acquire ต้องมี --approval อ้างข้อความอนุมัติของเจ้าของ")
        if active and active.get("task_id") != args.task_id:
            print(json.dumps({"ok": False, "reason": "workspace_locked", "active": active}, ensure_ascii=False))
            return 2
        data = {
            "workspace": str(cwd),
            "task_id": args.task_id,
            "branch": args.branch,
            "base_sha": args.base_sha,
            "allowed_paths": paths,
            "approval": args.approval,
            "acquired_at": now().isoformat(),
            "expires_at": (now() + dt.timedelta(minutes=args.ttl_minutes)).isoformat(),
        }
        write(state_path, data)
        print(json.dumps({"ok": True, "permit": data}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
