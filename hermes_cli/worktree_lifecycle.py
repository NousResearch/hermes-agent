"""Hermes Worktree Lifecycle Manager.

คำสั่งนี้เป็นทางกลางสำหรับสร้าง ตรวจ ส่งต่อ ปิด และเก็บกวาด task
worktree โดยมีสมุดทะเบียนและด่านความปลอดภัยชุดเดียวกัน.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from hermes_constants import get_hermes_home

try:  # POSIX: macOS + VPS Linux
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback uses process-local write
    fcntl = None


SCHEMA_VERSION = "worktree-lifecycle-v1"
ACTIVE_STATES = {"CREATED", "ACTIVE", "PAUSED", "IN_REVIEW", "BLOCKED"}
CLEANUP_SOURCE_STATES = {"MERGED", "ABANDONED_BY_OWNER", "CLEANUP_READY", "QUARANTINED"}
DEFAULT_PORT_RANGE = (8100, 8999)
QUARANTINE_HOURS = 72
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,79}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
REMOTE_HOST_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")

_REMOTE_REGISTRY_SERVER = r'''
import fcntl, json, os, sys, tempfile
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
lock_path = path + ".lock"
with open(lock_path, "a+", encoding="utf-8") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as source:
                data = json.load(source)
        except (OSError, ValueError) as exc:
            print(json.dumps({"ready": False, "error": "registry_invalid", "detail": str(exc)}), flush=True)
            raise SystemExit(2)
    else:
        data = None
    print(json.dumps({"ready": True, "data": data}, separators=(",", ":")), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("action") == "close":
            print(json.dumps({"closed": True}), flush=True)
            break
        if message.get("action") != "save":
            print(json.dumps({"saved": False, "error": "bad_action"}), flush=True)
            continue
        payload = message["data"]
        fd, temp_name = tempfile.mkstemp(prefix=".wtl-", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump(payload, target, ensure_ascii=False, indent=2, sort_keys=True)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_name, path)
            os.chmod(path, 0o600)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        print(json.dumps({"saved": True}), flush=True)
'''

_REMOTE_SESSIONS: Dict[str, subprocess.Popen[str]] = {}


class WorktreeLifecycleError(RuntimeError):
    """ข้อผิดพลาดที่คืนให้ผู้ใช้โดยไม่แสดง traceback."""


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return utcnow().isoformat()


def parse_time(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    return dt.datetime.fromisoformat(value)


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git_text(cwd: Path, *args: str) -> str:
    return git(cwd, *args).stdout.strip()


def registry_path(explicit: Optional[str] = None):
    configured = explicit or os.environ.get("HERMES_WORKTREE_REGISTRY")
    if configured:
        if configured.startswith("ssh://"):
            parse_remote_registry(configured)
            return configured
        return Path(configured).expanduser().resolve()
    return (get_hermes_home() / "worktrees" / "registry.json").resolve()


def parse_remote_registry(value: str) -> Tuple[str, str]:
    parsed = urlparse(value)
    host = parsed.netloc
    path = parsed.path
    if parsed.scheme != "ssh" or not REMOTE_HOST_RE.match(host) or not REMOTE_PATH_RE.match(path):
        raise WorktreeLifecycleError("remote registry ต้องเป็น ssh://user@host/absolute/safe-path.json")
    return host, path


def remote_cache_path(value: str) -> Path:
    key = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return (get_hermes_home() / "worktrees" / "cache" / "{}.json".format(key)).resolve()


def save_remote_cache(value: str, data: Dict[str, Any]) -> None:
    cache = remote_cache_path(value)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_registry(cache, json.loads(json.dumps(data)))


def load_remote_cache(value: str) -> Dict[str, Any]:
    cache = remote_cache_path(value)
    if not cache.is_file():
        raise WorktreeLifecycleError("ไม่มีสำเนาทะเบียนสำหรับโหมด offline")
    return load_registry(cache)


def open_remote_registry(value: str) -> Tuple[subprocess.Popen[str], Dict[str, Any]]:
    host, path = parse_remote_registry(value)
    remote_command = "python3 -c {} {}".format(
        shlex.quote(_REMOTE_REGISTRY_SERVER), shlex.quote(path)
    )
    process = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote_command],
        text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=1,
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read().strip() if process.stderr else ""
        process.kill()
        raise WorktreeLifecycleError("เชื่อมสมุดทะเบียนกลางบน VPS ไม่สำเร็จ: {}".format(stderr or "no response"))
    try:
        ready = json.loads(line)
    except json.JSONDecodeError as exc:
        process.kill()
        raise WorktreeLifecycleError("VPS registry ตอบข้อมูลที่อ่านไม่ได้") from exc
    if not ready.get("ready"):
        process.kill()
        raise WorktreeLifecycleError("VPS registry เปิดไม่ได้: {}".format(ready.get("detail") or ready.get("error")))
    data = ready.get("data") or empty_registry()
    validate_registry(data)
    save_remote_cache(value, data)
    return process, data


def empty_registry() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": iso_now(),
        "tasks": {},
        "history": [],
    }


@contextlib.contextmanager
def locked_registry(path) -> Iterator[Dict[str, Any]]:
    if isinstance(path, str) and path.startswith("ssh://"):
        process, data = open_remote_registry(path)
        _REMOTE_SESSIONS[path] = process
        try:
            yield data
        finally:
            try:
                if process.stdin and process.stdout:
                    process.stdin.write(json.dumps({"action": "close"}) + "\n")
                    process.stdin.flush()
                    process.stdout.readline()
            except (BrokenPipeError, OSError):
                pass
            finally:
                _REMOTE_SESSIONS.pop(path, None)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = load_registry(path)
        try:
            yield data
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_registry(data: Dict[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise WorktreeLifecycleError(
            "รุ่นสมุดทะเบียนไม่ตรง: {}".format(data.get("schema_version"))
        )
    if not isinstance(data.get("tasks"), dict):
        raise WorktreeLifecycleError("สมุดทะเบียนไม่มี tasks mapping")
    data.setdefault("history", [])


def load_registry(path) -> Dict[str, Any]:
    if isinstance(path, str) and path.startswith("ssh://"):
        process, data = open_remote_registry(path)
        try:
            if process.stdin and process.stdout:
                process.stdin.write(json.dumps({"action": "close"}) + "\n")
                process.stdin.flush()
                process.stdout.readline()
        finally:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        return data
    if not path.exists():
        return empty_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorktreeLifecycleError("อ่านสมุดทะเบียนไม่ได้: {}".format(exc)) from exc
    validate_registry(data)
    return data


def save_registry(path, data: Dict[str, Any]) -> None:
    if isinstance(path, str) and path.startswith("ssh://"):
        process = _REMOTE_SESSIONS.get(path)
        if process is None or process.stdin is None or process.stdout is None:
            raise WorktreeLifecycleError("remote registry save ต้องอยู่ภายใน locked transaction")
        data["schema_version"] = SCHEMA_VERSION
        data["updated_at"] = iso_now()
        process.stdin.write(json.dumps({"action": "save", "data": data}, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        if not response.get("saved"):
            raise WorktreeLifecycleError("บันทึกสมุดทะเบียนกลางบน VPS ไม่สำเร็จ")
        save_remote_cache(path, data)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = iso_now()
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        temp_path.unlink(missing_ok=True)


def append_history(data: Dict[str, Any], task: Dict[str, Any], action: str, detail: str) -> None:
    event = {
        "at": iso_now(),
        "task_id": task["task_id"],
        "action": action,
        "detail": detail,
        "state": task.get("state"),
        "machine_id": task.get("machine_id"),
    }
    task.setdefault("history", []).append(event)
    data.setdefault("history", []).append(event)


def validate_id(label: str, value: str) -> str:
    if not ID_RE.match(value):
        raise WorktreeLifecycleError("{} ไม่ถูกต้อง: {}".format(label, value))
    return value


def validate_slug(value: str) -> str:
    if not SLUG_RE.match(value):
        raise WorktreeLifecycleError("slug ต้องเป็นตัวเล็ก ตัวเลข หรือขีดกลาง และยาวไม่เกิน 48")
    return value


def default_worktree_root(machine_id: Optional[str] = None, home: Optional[Path] = None) -> Path:
    machine = machine_id or machine_default()
    if machine.startswith("vps-") or Path("/home/linux-nat").exists():
        return Path("/home/linux-nat/.worktree")
    return (home or Path.home()) / "Documents" / "Worktrees"


def resolve_root(root: Optional[str], machine_id: Optional[str] = None) -> Path:
    path = (Path(root).expanduser() if root else default_worktree_root(machine_id)).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise WorktreeLifecycleError("path อยู่นอก registered root: {}".format(path)) from exc


def ensure_repo(path: Path) -> Path:
    try:
        root = Path(git_text(path, "rev-parse", "--show-toplevel")).resolve()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise WorktreeLifecycleError("ไม่ใช่ Git repository: {}".format(path)) from exc
    return root


def branch_exists(repo: Path, branch: str) -> bool:
    return git(repo, "show-ref", "--verify", "--quiet", "refs/heads/{}".format(branch), check=False).returncode == 0


def ref_exists(repo: Path, ref: str) -> bool:
    return git(repo, "rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0


def choose_base_ref(repo: Path, remote: str, base_branch: str) -> str:
    remote_ref = "{}/{}".format(remote, base_branch)
    if ref_exists(repo, remote_ref):
        return remote_ref
    if ref_exists(repo, base_branch):
        return base_branch
    raise WorktreeLifecycleError("ไม่พบ base branch: {} หรือ {}".format(remote_ref, base_branch))


def machine_default() -> str:
    host = socket.gethostname().split(".")[0].lower()
    kind = "vps" if Path("/home/linux-nat").exists() else "notebook"
    safe = re.sub(r"[^a-z0-9-]+", "-", host).strip("-") or "unknown"
    return "{}-{}".format(kind, safe)


def path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def task_path(root: Path, project_id: str, staff_id: str, task_id: str, slug: str) -> Path:
    result = (root / project_id / staff_id / "{}-{}".format(task_id, slug)).resolve()
    ensure_under_root(result, root)
    return result


def branch_name(staff_id: str, task_id: str, slug: str) -> str:
    return "task/{}/{}-{}".format(staff_id, task_id, slug)


def lease_id(task_id: str, machine_id: str) -> str:
    raw = "{}:{}:{}".format(task_id, machine_id, iso_now())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def used_ports(data: Dict[str, Any]) -> set[int]:
    result = set()
    for item in data.get("tasks", {}).values():
        if item.get("state") in {"IN_REVIEW", "MERGED", "ABANDONED_BY_OWNER", "CLEANUP_READY", "QUARANTINED", "ARCHIVED"}:
            continue
        port = (item.get("runtime") or {}).get("port")
        if isinstance(port, int):
            result.add(port)
    return result


def listening_ports() -> set[int]:
    ports = set()
    for port in range(DEFAULT_PORT_RANGE[0], DEFAULT_PORT_RANGE[1] + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.005)
        try:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                ports.add(port)
        finally:
            sock.close()
    return ports


def allocate_port(data: Dict[str, Any]) -> int:
    busy = used_ports(data) | listening_ports()
    for port in range(DEFAULT_PORT_RANGE[0], DEFAULT_PORT_RANGE[1] + 1):
        if port not in busy:
            return port
    raise WorktreeLifecycleError("ไม่มีพอร์ตว่างในช่วง 8100-8999")


def runtime_namespace(task_id: str, port: int, worktree: Path) -> Dict[str, Any]:
    safe = re.sub(r"[^a-z0-9]+", "-", task_id.lower()).strip("-")
    db = re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")
    return {
        "port": port,
        "container_project": safe,
        "database_name": "test_{}".format(db),
        "temp_dir": str((worktree / ".tmp" / safe).resolve()),
        "cache_namespace": safe,
        "processes": [],
    }


def disk_usage(path: Path) -> Dict[str, int]:
    buckets = {"code_bytes": 0, "dependency_bytes": 0, "cache_bytes": 0, "build_bytes": 0}
    if not path.exists():
        buckets["total_bytes"] = 0
        return buckets
    dependency_names = {"node_modules", ".venv", "venv", "vendor"}
    cache_names = {".cache", ".pytest_cache", ".mypy_cache", "__pycache__"}
    build_names = {"dist", "build", ".next", "coverage", "out"}
    for current, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name != ".git"]
        rel_parts = set(Path(current).relative_to(path).parts)
        if rel_parts & dependency_names:
            bucket = "dependency_bytes"
        elif rel_parts & cache_names:
            bucket = "cache_bytes"
        elif rel_parts & build_names:
            bucket = "build_bytes"
        else:
            bucket = "code_bytes"
        for name in files:
            try:
                buckets[bucket] += (Path(current) / name).stat().st_size
            except OSError:
                pass
    buckets["total_bytes"] = sum(buckets.values())
    return buckets


def disk_percent(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return (usage.used / usage.total * 100.0) if usage.total else 100.0


def disk_policy(path: Path) -> Dict[str, Any]:
    percent = disk_percent(path)
    if percent >= 90.0:
        level = "recovery"
        message = "พื้นที่ใช้ถึง 90% ต้องหยุดงานที่เพิ่มพื้นที่และกู้พื้นที่แบบตรวจสอบก่อน"
    elif percent >= 85.0:
        level = "stop_new"
        message = "พื้นที่ใช้ถึง 85% จึงหยุดสร้าง Worktree ใหม่"
    elif percent >= 70.0:
        level = "warning"
        message = "พื้นที่ใช้ถึง 70% ควรวางแผนเก็บกวาด"
    else:
        level = "normal"
        message = "พื้นที่อยู่ในเกณฑ์ปกติ"
    return {"percent": round(percent, 2), "level": level, "message": message}


def task_summary(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "project_id": task.get("project_id"),
        "state": task.get("state"),
        "staff_id": task.get("staff_id"),
        "machine_id": task.get("machine_id"),
        "branch": task.get("branch"),
        "worktree_path": task.get("worktree_path"),
        "write_owner": task.get("write_owner"),
        "last_seen_at": task.get("last_seen_at"),
        "disk": task.get("disk"),
    }


def emit(result: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(result.get("decision", "WTL_BLOCKED"))
    print(result.get("message", ""))
    task = result.get("task")
    if task:
        print("task: {} · state: {}".format(task.get("task_id"), task.get("state")))
        print("path: {}".format(task.get("worktree_path")))
        print("branch: {}".format(task.get("branch")))
    for item in result.get("reasons", []):
        print("- {}".format(item))


def get_task(data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    task = data.get("tasks", {}).get(task_id)
    if not task:
        raise WorktreeLifecycleError("ไม่พบ task ในสมุดทะเบียน: {}".format(task_id))
    return task


def inspect_git(task: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(task["worktree_path"])
    result = {
        "path_exists": path.exists(),
        "is_git": False,
        "branch_match": False,
        "clean": False,
        "unpushed": None,
        "head": None,
    }
    if not path.exists():
        return result
    try:
        ensure_repo(path)
        result["is_git"] = True
        result["head"] = git_text(path, "rev-parse", "HEAD")
        actual_branch = git_text(path, "branch", "--show-current")
        result["actual_branch"] = actual_branch
        result["branch_match"] = actual_branch == task.get("branch")
        result["clean"] = not bool(git_text(path, "status", "--porcelain"))
        remote = task.get("remote", "origin")
        branch = task.get("branch")
        remote_ref = "{}/{}".format(remote, branch)
        if ref_exists(path, remote_ref):
            result["unpushed"] = int(git_text(path, "rev-list", "--count", "{}..HEAD".format(remote_ref)))
        else:
            result["unpushed"] = int(git_text(path, "rev-list", "--count", "HEAD"))
    except (WorktreeLifecycleError, subprocess.CalledProcessError, ValueError):
        pass
    return result


def git_worktrees(repo: Path) -> List[Dict[str, Any]]:
    """Read `git worktree list --porcelain` without changing repository state."""
    records: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in git_text(repo, "worktree", "list", "--porcelain").splitlines() + [""]:
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = str(Path(value).resolve())
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"bare", "detached", "locked", "prunable"}:
            current[key] = value or True
    return records


def scan_records(repositories: List[Path], data: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_path = {
        path_key(Path(task["worktree_path"])): task
        for task in data.get("tasks", {}).values()
    }
    records: List[Dict[str, Any]] = []
    seen = set()
    for repository in repositories:
        canonical = ensure_repo(repository)
        if path_key(canonical) in seen:
            continue
        seen.add(path_key(canonical))
        for item in git_worktrees(canonical):
            worktree = Path(item["path"])
            task = by_path.get(path_key(worktree))
            broken = bool(item.get("prunable")) or not worktree.exists()
            state = "broken" if broken else ("managed" if task else "unknown")
            records.append({
                **item,
                "classification": state,
                "task_id": task.get("task_id") if task else None,
                "canonical_repo": str(canonical),
                "disk": disk_usage(worktree),
                "cleanup_candidate": bool(
                    task and task.get("state") in CLEANUP_SOURCE_STATES
                ),
            })
    return records


def cleanup_gates(task: Dict[str, Any]) -> Tuple[Dict[str, bool], List[str]]:
    state = task.get("state")
    check = inspect_git(task)
    runtime = task.get("runtime") or {}
    evidence = task.get("evidence") or []
    cleanup = task.get("cleanup") or {}
    dry_run_current = bool(cleanup.get("dry_run_at"))
    dry_run_current = dry_run_current and cleanup.get("dry_run_state") == state
    dry_run_current = dry_run_current and cleanup.get("dry_run_head") == check.get("head")
    gates = {
        "clean": bool(check.get("clean")) or state == "ABANDONED_BY_OWNER",
        "pushed": check.get("unpushed") == 0 or state == "ABANDONED_BY_OWNER",
        "no_writer_or_process": not task.get("lease_id") and not runtime.get("processes"),
        "merged_or_owner_abandoned": state in CLEANUP_SOURCE_STATES,
        "recovery_evidence": bool(evidence),
        "dry_run_recorded": dry_run_current,
    }
    reasons = [name for name, ok in gates.items() if not ok]
    return gates, reasons


def command_open(args: argparse.Namespace) -> Dict[str, Any]:
    project_id = validate_id("project_id", args.project_id)
    staff_id = validate_id("staff_id", args.staff_id)
    task_id = validate_id("task_id", args.task_id)
    slug = validate_slug(args.slug)
    machine_id = validate_id("machine_id", args.machine_id or machine_default())
    canonical = ensure_repo(Path(args.repo).expanduser().resolve())
    root = resolve_root(args.root, machine_id)
    target = task_path(root, project_id, staff_id, task_id, slug)
    branch = branch_name(staff_id, task_id, slug)
    ensure_under_root(target, root)
    if target == canonical:
        raise WorktreeLifecycleError("canonical repo กับ task worktree ห้ามเป็น path เดียวกัน")

    path = registry_path(args.registry)
    with locked_registry(path) as data:
        existing = data["tasks"].get(task_id)
        if existing:
            if path_key(Path(existing["worktree_path"])) == path_key(target):
                same_writer = existing.get("machine_id") == machine_id and bool(existing.get("lease_id"))
                ready = existing.get("state") == "ACTIVE" and same_writer
                return {
                    "ok": ready,
                    "decision": "WTL_READY" if ready else "WTL_BLOCKED",
                    "message": "task นี้มีอยู่แล้วและสิทธิ์ตรงเครื่อง" if ready else "task นี้มีอยู่แล้ว แต่เครื่อง/สิทธิ์เขียนไม่ตรง",
                    "task": task_summary(existing),
                }
            raise WorktreeLifecycleError("task_id นี้ถูกใช้กับ Worktree อื่นแล้ว")
        for item in data["tasks"].values():
            if path_key(Path(item["worktree_path"])) == path_key(target):
                raise WorktreeLifecycleError("ปลายทางถูกใช้โดย task {}".format(item["task_id"]))

        storage = disk_policy(root)
        if storage["percent"] >= 85.0:
            raise WorktreeLifecycleError(storage["message"] + " ({:.2f}%)".format(storage["percent"]))
        open_count = sum(
            1
            for item in data["tasks"].values()
            if item.get("project_id") == project_id
            and item.get("staff_id") == staff_id
            and item.get("state") in ACTIVE_STATES
        )
        if open_count >= 3 and not args.allow_over_limit:
            raise WorktreeLifecycleError("มี Worktree ที่เปิดอยู่ครบ 3 งานแล้ว ต้องอนุมัติ --allow-over-limit")

        base_ref = choose_base_ref(canonical, args.remote, args.base_branch)
        base_sha = git_text(canonical, "rev-parse", base_ref)
        port = allocate_port(data)
        task = {
            "schema_version": SCHEMA_VERSION,
            "project_id": project_id,
            "task_id": task_id,
            "staff_id": staff_id,
            "machine_id": machine_id,
            "canonical_repo": str(canonical),
            "registered_root": str(root),
            "worktree_path": str(target),
            "branch": branch,
            "base_branch": args.base_branch,
            "base_sha": base_sha,
            "remote": args.remote,
            "state": "CREATED",
            "write_owner": staff_id,
            "lease_id": lease_id(task_id, machine_id),
            "lease_expires_at": (utcnow() + dt.timedelta(hours=args.lease_hours)).isoformat(),
            "runtime": runtime_namespace(task_id, port, target),
            "disk": {"total_bytes": 0},
            "disk_policy": storage,
            "last_seen_at": iso_now(),
            "created_at": iso_now(),
            "cleanup": {"eligible": False, "gate_results": {}, "quarantine_until": None},
            "evidence": [],
            "history": [],
        }
        append_history(data, task, "open-plan", "เตรียมสร้างจาก {}".format(base_ref))

        if not args.apply:
            return {
                "ok": True,
                "decision": "WTL_OPEN_PROPOSED",
                "message": "แผนสร้างพร้อมแล้ว ยังไม่ได้แก้ Git หรือทะเบียน",
                "task": task_summary(task),
            }

        if target.exists():
            raise WorktreeLifecycleError("ปลายทางมีอยู่แล้ว: {}".format(target))
        if branch_exists(canonical, branch):
            raise WorktreeLifecycleError("branch มีอยู่แล้ว: {}".format(branch))

        task["state"] = "CREATING"
        data["tasks"][task_id] = task
        save_registry(path, data)
        created = git(
            canonical,
            "worktree",
            "add",
            "-b",
            branch,
            str(target),
            base_ref,
            check=False,
        )
        if created.returncode != 0:
            task["state"] = "BLOCKED"
            task["lease_id"] = None
            append_history(data, task, "open-failed", created.stderr.strip())
            save_registry(path, data)
            raise WorktreeLifecycleError("สร้าง Worktree ไม่สำเร็จ: {}".format(created.stderr.strip()))
        task["state"] = "ACTIVE"
        task["disk"] = disk_usage(target)
        task["evidence"].append({"kind": "git", "sha": git_text(target, "rev-parse", "HEAD")})
        append_history(data, task, "opened", "สร้าง branch และ Worktree สำเร็จ")
        save_registry(path, data)
        return {
            "ok": True,
            "decision": "WTL_READY",
            "message": "สร้าง Worktree และสิทธิ์เขียนแล้ว",
            "task": task_summary(task),
        }


def command_list(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    data = load_registry(path)
    tasks = [task_summary(item) for item in data.get("tasks", {}).values()]
    if args.project_id:
        tasks = [item for item in tasks if item.get("project_id") == args.project_id]
    return {"ok": True, "decision": "WTL_LIST", "message": "พบ {} งาน".format(len(tasks)), "tasks": tasks, "registry": str(path)}


def command_status(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    if getattr(args, "offline", False):
        if not isinstance(path, str) or not path.startswith("ssh://"):
            raise WorktreeLifecycleError("--offline ใช้กับ ssh:// registry เท่านั้น")
        data = load_remote_cache(path)
        task = get_task(data, args.task_id)
        machine = validate_id("machine_id", args.machine_id or machine_default())
        expires = parse_time(task.get("lease_expires_at"))
        check = inspect_git(task)
        ready = (
            task.get("state") == "ACTIVE"
            and task.get("machine_id") == machine
            and bool(task.get("lease_id"))
            and bool(expires and utcnow() < expires)
            and check.get("path_exists") and check.get("is_git") and check.get("branch_match")
        )
        return {
            "ok": ready,
            "decision": "WTL_READY_OFFLINE" if ready else "WTL_BLOCKED",
            "message": "ใช้สิทธิ์เดิมจาก cache ได้; ห้าม open/handoff/cleanup" if ready else "cache/lease/machine/Git ไม่พร้อมสำหรับ offline",
            "task": task_summary(task), "git": check, "cache": str(remote_cache_path(path)),
        }
    with locked_registry(path) as data:
        task = get_task(data, args.task_id)
        task["last_seen_at"] = iso_now()
        worktree = Path(task["worktree_path"])
        task["disk"] = disk_usage(worktree)
        check = inspect_git(task)
        task["git_check"] = check
        expires = parse_time(task.get("lease_expires_at"))
        lease_expired = bool(expires and utcnow() >= expires)
        drift = not check.get("path_exists") or not check.get("is_git") or not check.get("branch_match") or lease_expired
        if drift and task.get("state") != "ARCHIVED":
            task["state_before_block"] = task.get("state")
            task["state"] = "BLOCKED"
            if lease_expired:
                task["lease_id"] = None
                task["lease_expires_at"] = None
            append_history(data, task, "doctor-block", "lease หมดอายุ" if lease_expired else "path/git/branch ไม่ตรงทะเบียน")
        save_registry(path, data)
        return {
            "ok": not drift,
            "decision": "WTL_READY" if not drift else "WTL_BLOCKED",
            "message": "สถานะตรงทะเบียน" if not drift else "ของจริงไม่ตรงสมุดทะเบียน",
            "task": task_summary(task),
            "git": check,
        }


def command_enter(args: argparse.Namespace) -> Dict[str, Any]:
    data = load_registry(registry_path(args.registry))
    task = get_task(data, args.task_id)
    path = Path(task["worktree_path"])
    if not path.is_dir():
        raise WorktreeLifecycleError("Worktree ไม่มีอยู่: {}".format(path))
    return {
        "ok": True,
        "decision": "WTL_READY" if task.get("lease_id") else "WTL_READ_ONLY",
        "message": "ใช้คำสั่ง cd ที่คืนให้เพื่อเข้า Worktree",
        "task": task_summary(task),
        "shell_command": "cd {}".format(json.dumps(str(path))),
    }


def command_pause(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    with locked_registry(path) as data:
        task = get_task(data, args.task_id)
        if task.get("state") not in {"ACTIVE", "BLOCKED", "IN_REVIEW"}:
            raise WorktreeLifecycleError("สถานะนี้พักไม่ได้: {}".format(task.get("state")))
        if (task.get("runtime") or {}).get("processes"):
            raise WorktreeLifecycleError("พักไม่ได้เพราะยังมี process ใน runtime namespace")
        check = inspect_git(task)
        task["state"] = "PAUSED"
        task["lease_id"] = None
        task["lease_expires_at"] = None
        task["pause_snapshot"] = check
        append_history(data, task, "paused", args.reason or "พักงาน")
        save_registry(path, data)
        return {"ok": True, "decision": "WTL_READ_ONLY", "message": "พักงานและปลดสิทธิ์เขียนแล้ว", "task": task_summary(task), "git": check}


def command_handoff(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    with locked_registry(path) as data:
        task = get_task(data, args.task_id)
        check = inspect_git(task)
        if not check.get("clean") or check.get("unpushed") != 0:
            return {
                "ok": False,
                "decision": "WTL_BLOCKED",
                "message": "ส่งต่อไม่ได้ เพราะมีไฟล์หรือ commit ที่ยังไม่ส่ง",
                "task": task_summary(task),
                "git": check,
            }
        if not args.apply:
            return {
                "ok": True,
                "decision": "WTL_HANDOFF_PROPOSED",
                "message": "พร้อมส่งต่อ แต่ยังไม่ได้ปลดสิทธิ์เครื่องเดิม",
                "task": task_summary(task),
                "target_machine": args.to_machine,
            }
        task["state"] = "HANDOFF_READY"
        task["handoff_from_machine"] = task.get("machine_id")
        task["handoff_to_machine"] = validate_id("to_machine", args.to_machine)
        task["lease_id"] = None
        task["lease_expires_at"] = None
        append_history(data, task, "handoff-prepared", "ส่งต่อไป {}".format(args.to_machine))
        save_registry(path, data)
        return {"ok": True, "decision": "WTL_HANDOFF_READY", "message": "ปลดสิทธิ์เครื่องเดิมแล้ว เครื่องใหม่รับต่อได้", "task": task_summary(task)}


def command_accept(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    with locked_registry(path) as data:
        task = get_task(data, args.task_id)
        machine = validate_id("machine_id", args.machine_id or machine_default())
        if task.get("state") != "HANDOFF_READY" or task.get("handoff_to_machine") != machine:
            raise WorktreeLifecycleError("task ยังไม่พร้อมส่งต่อให้เครื่องนี้")
        canonical = ensure_repo(Path(args.repo).expanduser().resolve())
        new_path = Path(args.worktree_path).expanduser().resolve()
        root = resolve_root(args.root, machine)
        ensure_under_root(new_path, root)
        if new_path == canonical:
            raise WorktreeLifecycleError("canonical repo กับ task worktree ห้ามเป็น path เดียวกัน")
        remote_ref = "{}/{}".format(task.get("remote", "origin"), task.get("branch"))
        if not ref_exists(canonical, remote_ref):
            raise WorktreeLifecycleError("ไม่พบ branch ที่ส่งต่อบน remote: {}".format(remote_ref))
        if not new_path.exists():
            if not args.apply:
                return {
                    "ok": True,
                    "decision": "WTL_ACCEPT_PROPOSED",
                    "message": "พร้อมสร้าง Worktree ฝั่งรับ แต่ยังไม่ได้แก้ Git หรือทะเบียน",
                    "task": task_summary(task),
                    "target_path": str(new_path),
                    "remote_ref": remote_ref,
                }
            created = git(canonical, "worktree", "add", "-b", task["branch"], str(new_path), remote_ref, check=False)
            if created.returncode != 0 and branch_exists(canonical, task["branch"]):
                created = git(canonical, "worktree", "add", str(new_path), task["branch"], check=False)
            if created.returncode != 0:
                raise WorktreeLifecycleError("สร้าง Worktree ฝั่งรับไม่สำเร็จ: {}".format(created.stderr.strip()))
        elif not args.apply:
            return {
                "ok": True,
                "decision": "WTL_ACCEPT_PROPOSED",
                "message": "พบ Worktree ฝั่งรับแล้ว พร้อมตรวจและรับสิทธิ์เมื่อใส่ --apply",
                "task": task_summary(task),
                "target_path": str(new_path),
                "remote_ref": remote_ref,
            }
        new_repo = ensure_repo(new_path)
        actual_branch = git_text(new_repo, "branch", "--show-current")
        if actual_branch != task.get("branch"):
            raise WorktreeLifecycleError("branch เครื่องใหม่ไม่ตรงทะเบียน")
        task.setdefault("locations", []).append({
            "machine_id": task.get("machine_id"),
            "worktree_path": task.get("worktree_path"),
            "canonical_repo": task.get("canonical_repo"),
            "released_at": iso_now(),
        })
        task["machine_id"] = machine
        task["canonical_repo"] = str(canonical)
        task["worktree_path"] = str(new_repo)
        task["registered_root"] = str(root)
        task["state"] = "ACTIVE"
        task["lease_id"] = lease_id(task["task_id"], machine)
        task["lease_expires_at"] = (utcnow() + dt.timedelta(hours=args.lease_hours)).isoformat()
        task["runtime"] = runtime_namespace(task["task_id"], allocate_port(data), new_repo)
        append_history(data, task, "handoff-accepted", "เครื่อง {} รับสิทธิ์".format(machine))
        save_registry(path, data)
        return {"ok": True, "decision": "WTL_READY", "message": "รับงานและสิทธิ์เขียนบนเครื่องใหม่แล้ว", "task": task_summary(task)}


def command_close(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    with locked_registry(path) as data:
        task = get_task(data, args.task_id)
        check = inspect_git(task)
        if (task.get("runtime") or {}).get("processes"):
            return {"ok": False, "decision": "WTL_BLOCKED", "message": "ปิดงานไม่ได้ เพราะยังมี process ใน runtime namespace", "task": task_summary(task)}
        if not check.get("clean") or check.get("unpushed") != 0:
            return {"ok": False, "decision": "WTL_BLOCKED", "message": "ปิดงานไม่ได้ เพราะมีไฟล์หรือ commit ที่ยังไม่ส่ง", "task": task_summary(task), "git": check}
        task["lease_id"] = None
        task["lease_expires_at"] = None
        task["state"] = "MERGED" if args.merged else "IN_REVIEW"
        runtime = task.setdefault("runtime", {})
        runtime["released_port"] = runtime.get("port")
        runtime["port"] = None
        runtime["released_at"] = iso_now()
        if args.merge_sha:
            task.setdefault("evidence", []).append({"kind": "merge", "sha": args.merge_sha})
        append_history(data, task, "closed", "merged={}".format(args.merged))
        save_registry(path, data)
        return {"ok": True, "decision": "WTL_CLOSED", "message": "ปิดสิทธิ์งานแล้ว", "task": task_summary(task)}


def command_abandon(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.owner_approval:
        raise WorktreeLifecycleError("abandon ต้องมี --owner-approval")
    path = registry_path(args.registry)
    with locked_registry(path) as data:
        task = get_task(data, args.task_id)
        task["state"] = "ABANDONED_BY_OWNER"
        task["lease_id"] = None
        task["lease_expires_at"] = None
        task.setdefault("evidence", []).append({"kind": "owner_approval", "value": args.owner_approval})
        append_history(data, task, "abandoned", args.owner_approval)
        save_registry(path, data)
        return {"ok": True, "decision": "WTL_CLEANUP_REVIEW", "message": "เจ้าของยืนยันเลิกงานแล้ว รอตรวจ cleanup 6/6", "task": task_summary(task)}


def command_cleanup(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    with locked_registry(path) as data:
        task = get_task(data, args.task_id)
        task["disk"] = disk_usage(Path(task["worktree_path"]))
        cleanup = task.setdefault("cleanup", {"eligible": False, "gate_results": {}, "quarantine_until": None})
        if not args.apply:
            cleanup["dry_run_at"] = iso_now()
            cleanup["dry_run_state"] = task.get("state")
            cleanup["dry_run_head"] = inspect_git(task).get("head")
        gates, reasons = cleanup_gates(task)
        cleanup["gate_results"] = gates
        cleanup["eligible"] = not reasons
        if reasons:
            save_registry(path, data)
            return {"ok": False, "decision": "WTL_BLOCKED", "message": "ยังเก็บกวาดไม่ได้", "task": task_summary(task), "gates": gates, "reasons": reasons}

        if not args.apply:
            save_registry(path, data)
            return {"ok": True, "decision": "WTL_CLEANUP_PROPOSED", "message": "ผ่าน 6/6 และสร้างรายงานจำลองแล้ว ยังไม่ลบ", "task": task_summary(task), "gates": gates}

        if task.get("state") != "QUARANTINED":
            task["state"] = "QUARANTINED"
            cleanup["quarantine_until"] = (utcnow() + dt.timedelta(hours=QUARANTINE_HOURS)).isoformat()
            append_history(data, task, "quarantined", "กักพัก {} ชั่วโมง".format(QUARANTINE_HOURS))
            save_registry(path, data)
            return {"ok": True, "decision": "WTL_CLEANUP_READY", "message": "เข้ากักพักแล้ว ยังไม่ลบ Worktree", "task": task_summary(task), "gates": gates}

        until = parse_time(cleanup.get("quarantine_until"))
        if until and utcnow() < until:
            save_registry(path, data)
            return {"ok": False, "decision": "WTL_BLOCKED", "message": "ยังไม่พ้นช่วงกักพัก", "task": task_summary(task), "quarantine_until": until.isoformat()}

        gates, reasons = cleanup_gates(task)
        if reasons:
            save_registry(path, data)
            return {"ok": False, "decision": "WTL_BLOCKED", "message": "ตรวจซ้ำไม่ผ่าน", "task": task_summary(task), "gates": gates, "reasons": reasons}
        canonical = Path(task["canonical_repo"])
        target = Path(task["worktree_path"])
        removed = git(canonical, "worktree", "remove", str(target), check=False)
        if removed.returncode != 0:
            task["state"] = "BLOCKED"
            append_history(data, task, "cleanup-failed", removed.stderr.strip())
            save_registry(path, data)
            raise WorktreeLifecycleError("git worktree remove ไม่สำเร็จ: {}".format(removed.stderr.strip()))
        task["state"] = "ARCHIVED"
        task["archived_at"] = iso_now()
        task["disk"] = {"total_bytes": 0}
        append_history(data, task, "archived", "เอา Worktree ออกผ่าน Git แล้ว")
        save_registry(path, data)
        return {"ok": True, "decision": "WTL_ARCHIVED", "message": "เอา Worktree ออกและเก็บประวัติแล้ว", "task": task_summary(task)}


def command_doctor(args: argparse.Namespace) -> Dict[str, Any]:
    path = registry_path(args.registry)
    with locked_registry(path) as data:
        problems = []
        paths = {}
        branches = {}
        for task_id, task in data.get("tasks", {}).items():
            pkey = path_key(Path(task["worktree_path"]))
            if pkey in paths:
                problems.append("path ซ้ำ: {} กับ {}".format(paths[pkey], task_id))
            paths[pkey] = task_id
            bkey = "{}:{}".format(task.get("canonical_repo"), task.get("branch"))
            if bkey in branches:
                problems.append("branch ซ้ำ: {} กับ {}".format(branches[bkey], task_id))
            branches[bkey] = task_id
            check = inspect_git(task)
            if task.get("state") != "ARCHIVED" and (not check.get("path_exists") or not check.get("branch_match")):
                problems.append("{}: path/git/branch ไม่ตรง".format(task_id))
            if task.get("lease_id") and task.get("state") not in {"ACTIVE", "CREATING"}:
                problems.append("{}: มี lease ในสถานะ {}".format(task_id, task.get("state")))
        return {
            "ok": not problems,
            "decision": "WTL_READY" if not problems else "WTL_BLOCKED",
            "message": "สมุดทะเบียนตรงของจริง" if not problems else "พบปัญหา {} รายการ".format(len(problems)),
            "registry": str(path),
            "task_count": len(data.get("tasks", {})),
            "disk_policy": (
                {"level": "remote", "message": "พื้นที่ registry อยู่บน VPS; ตรวจด้วย report ฝั่ง VPS"}
                if isinstance(path, str) else disk_policy(path.parent)
            ),
            "reasons": problems,
        }


def command_scan(args: argparse.Namespace) -> Dict[str, Any]:
    """Inventory existing worktrees. This command is intentionally read-only."""
    data = load_registry(registry_path(args.registry))
    repositories = [Path(value).expanduser().resolve() for value in args.repo]
    records = scan_records(repositories, data)
    counts = {
        name: sum(1 for item in records if item["classification"] == name)
        for name in ("managed", "unknown", "broken")
    }
    return {
        "ok": counts["broken"] == 0,
        "decision": "WTL_SCAN_COMPLETE" if counts["broken"] == 0 else "WTL_SCAN_REVIEW",
        "message": "สำรวจแบบอ่านอย่างเดียว {} Worktree; ไม่ลบและไม่เปลี่ยน branch".format(len(records)),
        "counts": counts,
        "records": records,
        "cleanup_candidates": [item for item in records if item["cleanup_candidate"]],
    }


def command_import(args: argparse.Namespace) -> Dict[str, Any]:
    """Register an existing worktree without changing its Git branch or files."""
    if not args.owner_approval:
        raise WorktreeLifecycleError("import ต้องมี --owner-approval")
    project_id = validate_id("project_id", args.project_id)
    staff_id = validate_id("staff_id", args.staff_id)
    task_id = validate_id("task_id", args.task_id)
    machine_id = validate_id("machine_id", args.machine_id or machine_default())
    canonical = ensure_repo(Path(args.repo).expanduser().resolve())
    worktree = ensure_repo(Path(args.worktree_path).expanduser().resolve())
    root = resolve_root(args.root, machine_id)
    ensure_under_root(worktree, root)
    matching = [item for item in git_worktrees(canonical) if path_key(Path(item["path"])) == path_key(worktree)]
    if not matching:
        raise WorktreeLifecycleError("Worktree นี้ไม่อยู่ใน git worktree list ของ canonical repo")
    record = matching[0]
    if record.get("detached") or not record.get("branch"):
        raise WorktreeLifecycleError("ยังนำเข้า detached Worktree ไม่ได้ ต้องผูก branch ก่อน")

    path = registry_path(args.registry)
    with locked_registry(path) as data:
        if task_id in data["tasks"]:
            raise WorktreeLifecycleError("task_id นี้มีในทะเบียนแล้ว")
        for task in data["tasks"].values():
            if path_key(Path(task["worktree_path"])) == path_key(worktree):
                raise WorktreeLifecycleError("Worktree นี้อยู่ในทะเบียน task {} แล้ว".format(task["task_id"]))
        task = {
            "schema_version": SCHEMA_VERSION,
            "project_id": project_id,
            "task_id": task_id,
            "staff_id": staff_id,
            "machine_id": machine_id,
            "canonical_repo": str(canonical),
            "registered_root": str(root),
            "worktree_path": str(worktree),
            "branch": record["branch"],
            "base_branch": args.base_branch,
            "base_sha": record.get("head"),
            "remote": args.remote,
            "state": "PAUSED",
            "write_owner": staff_id,
            "lease_id": None,
            "lease_expires_at": None,
            "runtime": runtime_namespace(task_id, allocate_port(data), worktree),
            "disk": disk_usage(worktree),
            "disk_policy": disk_policy(root),
            "last_seen_at": iso_now(),
            "created_at": iso_now(),
            "imported_at": iso_now(),
            "cleanup": {"eligible": False, "gate_results": {}, "quarantine_until": None},
            "evidence": [{"kind": "import", "sha": record.get("head"), "approval": args.owner_approval}],
            "history": [],
        }
        append_history(data, task, "imported", "นำของเดิมเข้าทะเบียนแบบไม่แก้ Git")
        data["tasks"][task_id] = task
        save_registry(path, data)
        return {
            "ok": True,
            "decision": "WTL_IMPORTED_READ_ONLY",
            "message": "นำ Worktree เดิมเข้าทะเบียนในสถานะพักแล้ว; ยังไม่มีสิทธิ์เขียน",
            "task": task_summary(task),
        }


def command_report(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a read-only PDCA scorecard; optionally record that it ran."""
    path = registry_path(args.registry)
    manager = locked_registry(path) if args.record else contextlib.nullcontext(load_registry(path))
    with manager as data:
        now = utcnow()
        pdca = data.setdefault("pdca", {})
        last_light = parse_time(pdca.get("last_light_check_at"))
        last_cleanup = parse_time(pdca.get("last_cleanup_review_at"))
        due_light = last_light is None or now - last_light >= dt.timedelta(hours=24)
        due_cleanup = last_cleanup is None or now - last_cleanup >= dt.timedelta(hours=168)
        counts: Dict[str, int] = {}
        projects: Dict[str, Dict[str, Any]] = {}
        total_bytes = 0
        cleanup_candidates = []
        blocked = []
        for task in data.get("tasks", {}).values():
            state = task.get("state", "UNKNOWN")
            counts[state] = counts.get(state, 0) + 1
            size = int((task.get("disk") or {}).get("total_bytes") or 0)
            total_bytes += size
            project = projects.setdefault(task.get("project_id", "unknown"), {"tasks": 0, "bytes": 0, "states": {}})
            project["tasks"] += 1
            project["bytes"] += size
            project["states"][state] = project["states"].get(state, 0) + 1
            if state == "BLOCKED":
                blocked.append(task_summary(task))
            if state in CLEANUP_SOURCE_STATES:
                gates, reasons = cleanup_gates(task)
                cleanup_candidates.append({
                    "task": task_summary(task), "gates": gates,
                    "passed": sum(bool(value) for value in gates.values()), "reasons": reasons,
                })
        report = {
            "generated_at": now.isoformat(),
            "task_count": len(data.get("tasks", {})),
            "states": counts,
            "total_bytes": total_bytes,
            "projects": projects,
            "blocked": blocked,
            "cleanup_candidates": cleanup_candidates,
            "cadence": {
                "light_check_hours": 24, "cleanup_review_hours": 168,
                "last_light_check_at": pdca.get("last_light_check_at"),
                "last_cleanup_review_at": pdca.get("last_cleanup_review_at"),
                "light_check_due": due_light, "cleanup_review_due": due_cleanup,
            },
        }
        if args.record:
            pdca["last_light_check_at"] = now.isoformat()
            if args.cleanup_review:
                pdca["last_cleanup_review_at"] = now.isoformat()
            pdca["last_report"] = report
            save_registry(path, data)
        return {
            "ok": not blocked,
            "decision": "WTL_PDCA_HEALTHY" if not blocked else "WTL_PDCA_REVIEW",
            "message": "รายงาน PDCA {} task; blocked {}; cleanup review {}".format(
                report["task_count"], len(blocked), len(cleanup_candidates)
            ),
            "report": report,
            "recorded": bool(args.record),
        }


def cmd_worktree(args: argparse.Namespace) -> None:
    handlers = {
        "open": command_open,
        "list": command_list,
        "status": command_status,
        "enter": command_enter,
        "pause": command_pause,
        "handoff": command_handoff,
        "accept": command_accept,
        "close": command_close,
        "abandon": command_abandon,
        "cleanup": command_cleanup,
        "doctor": command_doctor,
        "scan": command_scan,
        "import": command_import,
        "report": command_report,
    }
    try:
        result = handlers[args.worktree_action](args)
    except WorktreeLifecycleError as exc:
        result = {"ok": False, "decision": "WTL_BLOCKED", "message": str(exc)}
    emit(result, getattr(args, "as_json", False))
    raise SystemExit(0 if result.get("ok") else 1)


def add_registry_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", help="ตำแหน่งสมุดทะเบียน; ค่าปริยายอยู่ใต้ HERMES_HOME")
    parser.add_argument("--json", dest="as_json", action="store_true", help="คืนผล JSON")


def register_worktree_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("worktree", help="จัดการวงจรชีวิต task worktree")
    subs = parser.add_subparsers(dest="worktree_action", required=True)

    open_p = subs.add_parser("open", help="วางแผนหรือสร้าง task worktree")
    for name in ("project-id", "staff-id", "task-id", "slug", "repo"):
        open_p.add_argument("--" + name, required=True)
    open_p.add_argument("--root", help="ราก Worktree; Notebook ใช้ ~/Documents/Worktrees โดยปริยาย")
    open_p.add_argument("--machine-id")
    open_p.add_argument("--remote", default="origin")
    open_p.add_argument("--base-branch", default="main")
    open_p.add_argument("--lease-hours", type=int, default=12)
    open_p.add_argument("--allow-over-limit", action="store_true")
    open_p.add_argument("--apply", action="store_true", help="สร้างจริง; ไม่ใส่คือ dry-run")
    add_registry_json(open_p)

    list_p = subs.add_parser("list", help="แสดง Worktree ในทะเบียน")
    list_p.add_argument("--project-id")
    add_registry_json(list_p)

    for action in ("status", "enter", "doctor"):
        item = subs.add_parser(action, help="{} Worktree".format(action))
        if action != "doctor":
            item.add_argument("--task-id", required=True)
        add_registry_json(item)
    subs.choices["status"].add_argument("--offline", action="store_true", help="อ่าน cache เพื่อทำ task เดิมเมื่อ VPS ขาดการเชื่อมต่อ")
    subs.choices["status"].add_argument("--machine-id", help="machine id ต้องตรง lease เดิมในโหมด offline")

    scan = subs.add_parser("scan", help="สำรวจ Worktree เดิมแบบอ่านอย่างเดียว")
    scan.add_argument("--repo", action="append", required=True, help="canonical repo; ใส่ซ้ำได้")
    add_registry_json(scan)

    imported = subs.add_parser("import", help="นำ Worktree เดิมเข้าทะเบียนโดยไม่เปลี่ยน Git")
    for name in ("project-id", "staff-id", "task-id", "repo", "worktree-path", "root"):
        imported.add_argument("--" + name, required=True)
    imported.add_argument("--machine-id")
    imported.add_argument("--remote", default="origin")
    imported.add_argument("--base-branch", default="main")
    imported.add_argument("--owner-approval", required=True)
    add_registry_json(imported)

    report = subs.add_parser("report", help="สรุป PDCA จำนวน พื้นที่ blocked และ cleanup")
    report.add_argument("--record", action="store_true", help="บันทึกเวลาตรวจ 24 ชั่วโมงลงทะเบียน")
    report.add_argument("--cleanup-review", action="store_true", help="เมื่อ --record ให้บันทึกรอบเสนอ cleanup 168 ชั่วโมง")
    add_registry_json(report)

    pause = subs.add_parser("pause", help="พักงานและปลดสิทธิ์เขียน")
    pause.add_argument("--task-id", required=True)
    pause.add_argument("--reason")
    add_registry_json(pause)

    handoff = subs.add_parser("handoff", help="เตรียมส่งงานไปอีกเครื่อง")
    handoff.add_argument("--task-id", required=True)
    handoff.add_argument("--to-machine", required=True)
    handoff.add_argument("--apply", action="store_true")
    add_registry_json(handoff)

    accept = subs.add_parser("accept", help="รับงานบนเครื่องใหม่")
    accept.add_argument("--task-id", required=True)
    accept.add_argument("--machine-id")
    accept.add_argument("--repo", required=True, help="canonical repo บนเครื่องรับ")
    accept.add_argument("--worktree-path", required=True)
    accept.add_argument("--root", help="ราก Worktree ของเครื่องรับ; Notebook ใช้ ~/Documents/Worktrees โดยปริยาย")
    accept.add_argument("--lease-hours", type=int, default=12)
    accept.add_argument("--apply", action="store_true", help="สร้าง/รับสิทธิ์จริง; ไม่ใส่คือ dry-run")
    add_registry_json(accept)

    close = subs.add_parser("close", help="ปิดสิทธิ์งานและบันทึกสถานะ")
    close.add_argument("--task-id", required=True)
    close.add_argument("--merged", action="store_true")
    close.add_argument("--merge-sha")
    add_registry_json(close)

    abandon = subs.add_parser("abandon", help="เจ้าของยืนยันเลิกทำ task")
    abandon.add_argument("--task-id", required=True)
    abandon.add_argument("--owner-approval", required=True)
    add_registry_json(abandon)

    cleanup = subs.add_parser("cleanup", help="ตรวจหรือเก็บกวาด Worktree")
    cleanup.add_argument("--task-id", required=True)
    cleanup.add_argument("--apply", action="store_true", help="เข้ากักพักหรือเอาออกเมื่อพ้นเวลา")
    add_registry_json(cleanup)

    for choice in subs.choices.values():
        choice.set_defaults(func=cmd_worktree)
    return parser


__all__ = [
    "SCHEMA_VERSION",
    "WorktreeLifecycleError",
    "cleanup_gates",
    "cmd_worktree",
    "default_worktree_root",
    "disk_usage",
    "load_registry",
    "register_worktree_subparser",
    "registry_path",
]
