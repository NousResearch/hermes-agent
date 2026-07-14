#!/usr/bin/env python3
"""Resolve the real VPS worktree for a staff id and project.

This is intentionally read-only. It helps New Chat startup decide where work
must happen before any agent edits files.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import textwrap
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_HOST = "linux-nat@103.142.150.185"
# Team work now lives under ~/.worktree/<project>/<staff>/<YYYYMMDD>-<task>
# (three levels). The first two roots keep the per-project reference clone and
# its 2-level legacy worktrees. The probe handles both depths.
DEFAULT_ROOTS = (
    "/home/linux-nat/projects",
    "/srv/projects",
    "/home/linux-nat/.worktree",
)

# --------------------------------------------------------------------------- #
# Claim store — team booking so two staff/AI do not edit the same paths.
# A claim is only "intent to edit". git status/branch stays the real source of
# truth for "already edited". Claims live in one shared JSON file guarded by an
# OS file lock, every claim carries expires_at so stale bookings self-expire.
# --------------------------------------------------------------------------- #
DEFAULT_EXPIRE_HOURS = 8


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _claim_store_path() -> Path:
    override = os.getenv("HERMES_CLAIM_STORE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hermes" / "worktree-claims.json"


@contextmanager
def _locked_store(path: Path):
    """Yield the store data while holding an exclusive lock on a separate .lock
    file. The JSON itself is replaced atomically on save, so the lock never sits
    on a file that gets renamed out from under it."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    lock_handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        data = {"claims": []}
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8").strip()
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict) and isinstance(parsed.get("claims"), list):
                    data = parsed
            except (OSError, json.JSONDecodeError):
                data = {"claims": []}
        yield data
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _atomic_save(path: Path, data: dict) -> None:
    """Write to a temp file then os.replace() so a crash mid-write never leaves
    a half-written (corrupt) claim store. fsync file + directory for durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".claims-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _norm_path(value: str) -> str:
    """Normalise a repo-relative path for overlap checks: drop '.', resolve
    '..', and strip empty segments so './a', 'a/../b', and 'a//b' compare cleanly."""
    segments: list[str] = []
    for seg in value.strip().replace("\\", "/").split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if segments:
                segments.pop()
            continue
        segments.append(seg)
    return "/".join(segments)


def _paths_overlap(a: str, b: str) -> bool:
    """Component-wise prefix overlap. 'gateway' overlaps 'gateway/run.py'."""
    pa = _norm_path(a)
    pb = _norm_path(b)
    if not pa or not pb:
        # An empty path means "the whole worktree" -> overlaps everything.
        return True
    sa = pa.split("/")
    sb = pb.split("/")
    shorter = min(len(sa), len(sb))
    return sa[:shorter] == sb[:shorter]


def _claim_is_expired(claim: dict, now: datetime) -> bool:
    expires = _parse_iso(claim.get("expires_at", ""))
    if expires is None:
        return False
    return now >= expires


def _annotate(claim: dict, now: datetime) -> dict:
    item = dict(claim)
    item["state"] = "STALE_CLAIM_REVIEW" if _claim_is_expired(claim, now) else "active"
    return item


def _project_norm(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _claims_for_project(claims: list[dict], project: str) -> list[dict]:
    target = _project_norm(project)
    return [c for c in claims if _project_norm(c.get("project", "")) == target]


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def claim_list(args: argparse.Namespace) -> int:
    now = _now()
    path = _claim_store_path()
    with _locked_store(path) as data:
        claims = data.get("claims", [])
        if args.project:
            claims = _claims_for_project(claims, args.project)
        annotated = [_annotate(claim, now) for claim in claims]
    _print_json({
        "ok": True,
        "action": "list",
        "store": str(path),
        "now": _iso(now),
        "count": len(annotated),
        "active": [c for c in annotated if c["state"] == "active"],
        "stale": [c for c in annotated if c["state"] != "active"],
    })
    return 0


def claim_acquire(args: argparse.Namespace) -> int:
    now = _now()
    if args.expires_at:
        expires = _parse_iso(args.expires_at)
        if expires is None:
            _print_json({"ok": False, "action": "acquire",
                         "error": "bad_expires_at", "value": args.expires_at})
            return 2
    else:
        expires = now + timedelta(hours=args.expires_hours)

    paths = [p for p in (args.paths.split(",") if args.paths else []) if p.strip()]
    if not paths:
        _print_json({"ok": False, "action": "acquire", "error": "no_paths",
                     "hint": "pass --paths a/,b/c.py (booking the whole worktree is not allowed)"})
        return 2

    path = _claim_store_path()
    with _locked_store(path) as data:
        claims = data.get("claims", [])
        # Drop fully expired claims so stale bookings never block forever.
        live = [c for c in claims if not _claim_is_expired(c, now)]

        conflicts = []
        for other in _claims_for_project(live, args.project):
            if str(other.get("staff_id", "")).lower() == args.staff_id.lower():
                continue
            for mine in paths:
                for theirs in other.get("paths", []):
                    if _paths_overlap(mine, theirs):
                        conflicts.append({
                            "staff_id": other.get("staff_id"),
                            "issue": other.get("issue"),
                            "their_path": theirs,
                            "your_path": _norm_path(mine),
                            "expires_at": other.get("expires_at"),
                        })
        if conflicts:
            data["claims"] = live
            _atomic_save(path, data)
            _print_json({
                "ok": False, "action": "acquire", "result": "CLAIM_CONFLICT",
                "staff_id": args.staff_id, "project": args.project,
                "conflicts": conflicts,
                "owner_action": "STOP — path ที่จะแก้ชนกับงานที่คนอื่นจองไว้ ห้ามแก้ทับ รายงานเจ้าของงาน",
            })
            return 2

        claim = {
            "id": uuid.uuid4().hex[:12],
            "staff_id": args.staff_id,
            "project": args.project,
            "worktree": args.worktree or "",
            "branch": args.branch or "",
            "issue": args.issue or "",
            "paths": [_norm_path(p) for p in paths],
            "created_at": _iso(now),
            "expires_at": _iso(expires),
        }
        live.append(claim)
        data["claims"] = live
        _atomic_save(path, data)
    _print_json({"ok": True, "action": "acquire", "result": "CLAIM_ACQUIRED", "claim": claim})
    return 0


def claim_release(args: argparse.Namespace) -> int:
    now = _now()
    path = _claim_store_path()
    with _locked_store(path) as data:
        claims = data.get("claims", [])
        removed, kept = [], []
        for claim in claims:
            match = (
                str(claim.get("staff_id", "")).lower() == args.staff_id.lower()
                and _project_norm(claim.get("project", "")) == _project_norm(args.project)
            )
            if match and args.id:
                match = claim.get("id") == args.id
            if match and args.issue:
                match = str(claim.get("issue", "")) == args.issue
            (removed if match else kept).append(claim)
        data["claims"] = kept
        _atomic_save(path, data)
    _print_json({
        "ok": bool(removed), "action": "release",
        "result": "CLAIM_RELEASED" if removed else "NO_MATCHING_CLAIM",
        "now": _iso(now), "released": removed, "remaining": len(kept),
    })
    return 0 if removed else 2


def _build_claim_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes_worktree_route.py claim",
        description="Team booking for worktree paths (acquire/release/list).",
    )
    sub = parser.add_subparsers(dest="claim_action", required=True)

    p_list = sub.add_parser("list", help="List active and stale claims.")
    p_list.add_argument("--project", default="", help="Filter by project slug.")
    p_list.set_defaults(func=claim_list)

    p_acq = sub.add_parser("acquire", help="Book paths before editing.")
    p_acq.add_argument("--staff-id", required=True)
    p_acq.add_argument("--project", required=True)
    p_acq.add_argument("--worktree", default="")
    p_acq.add_argument("--branch", default="")
    p_acq.add_argument("--issue", default="")
    p_acq.add_argument("--paths", required=True,
                       help="Comma-separated paths to book, e.g. scripts/,skills/devops/x.py")
    p_acq.add_argument("--expires-hours", type=float, default=DEFAULT_EXPIRE_HOURS)
    p_acq.add_argument("--expires-at", default="", help="ISO time; overrides --expires-hours.")
    p_acq.set_defaults(func=claim_acquire)

    p_rel = sub.add_parser("release", help="Release your claim after commit/handoff.")
    p_rel.add_argument("--staff-id", required=True)
    p_rel.add_argument("--project", required=True)
    p_rel.add_argument("--issue", default="", help="Release only the claim for this issue.")
    p_rel.add_argument("--id", default="", help="Release only the claim with this id.")
    p_rel.set_defaults(func=claim_release)
    return parser


def claim_main(argv: list[str]) -> int:
    parser = _build_claim_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _remote_probe_script(staff_id: str, project: str, roots: list[str]) -> str:
    return textwrap.dedent(
        r"""
        import json
        import os
        import subprocess
        import sys

        staff_id = sys.argv[1].strip().lower()
        project = sys.argv[2].strip().lower()
        roots = sys.argv[3:]

        def norm(value):
            return "".join(ch for ch in value.lower() if ch.isalnum())

        project_norm = norm(project)
        staff_norm = norm(staff_id)
        candidates = []
        project_candidates = []
        seen = set()

        def run_git(path, *args):
            proc = subprocess.run(
                ["git", "-C", path, *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                return None
            return proc.stdout.strip()

        def redact_remote(line):
            for marker in ("https://oauth2:", "https://x-token-auth:", "https://gitlab-ci-token:"):
                if marker in line:
                    before, after = line.split(marker, 1)
                    if "@" in after:
                        return before + marker + "***@" + after.split("@", 1)[1]
            return line

        def record(path):
            real = os.path.realpath(path)
            if real in seen or not os.path.isdir(real):
                return
            seen.add(real)

            top = run_git(real, "rev-parse", "--show-toplevel")
            if not top or os.path.realpath(top) != real:
                return

            branch = run_git(real, "branch", "--show-current") or ""
            head = run_git(real, "rev-parse", "HEAD") or ""
            status = run_git(real, "status", "--short", "--branch") or ""
            remotes = run_git(real, "remote", "-v") or ""

            haystack = norm(" ".join([real, branch, remotes]))
            staff_match = (
                norm(os.path.basename(real)) == staff_norm
                or norm(branch) == staff_norm
                or ("/" + staff_id + "/") in real.lower()
                or real.lower().endswith("/" + staff_id)
            )
            project_match = project_norm in haystack

            item = {
                "path": real,
                "branch": branch,
                "head": head,
                "status": status.splitlines(),
                "remote_summary": [redact_remote(line) for line in remotes.splitlines()[:4]],
                "staff_match": staff_match,
                "project_match": project_match,
                "score": (10 if project_match else 0) + (10 if staff_match else 0),
            }
            if project_match:
                project_candidates.append(item)
            if project_match and staff_match:
                candidates.append(item)

        def variants(value):
            raw = value.strip()
            simple = raw.lower().replace(" ", "-").replace("_", "-")
            compact = norm(raw)
            values = {raw, raw.lower(), simple, compact}
            values.update({
                simple.replace("-", "_"),
                simple.replace("-", ""),
                compact.replace("-", ""),
            })
            return [v for v in values if v]

        project_variants = variants(project)
        staff_variants = variants(staff_id)

        # Fast path: known VPS team layouts. New Chat must route by project
        # first, then staff. Do not accept root-level staff folders such as
        # /home/linux-nat/projects/nat because they do not scale across projects.
        for root in roots:
            if not os.path.isdir(root):
                continue
            for pv in project_variants:
                for sv in staff_variants:
                    base = os.path.join(root, pv, sv)
                    # 2-level layout: <root>/<project>/<staff> is itself a worktree.
                    record(base)
                    # 3-level layout: <root>/<project>/<staff>/<YYYYMMDD>-<task>.
                    # Each child task folder is its own worktree/branch.
                    try:
                        for task in sorted(os.listdir(base)):
                            if task.startswith("."):
                                continue
                            record(os.path.join(base, task))
                    except OSError:
                        pass

            # Shallow fallback only. Do not walk dependency folders or bare repo
            # object stores during chat startup.
            try:
                first_level = [
                    os.path.join(root, name)
                    for name in os.listdir(root)
                    if not name.startswith(".")
                ]
            except OSError:
                first_level = []
            project_norm_set = {norm(v) for v in project_variants}
            for path in first_level:
                if norm(os.path.basename(path)) not in project_norm_set:
                    continue
                # <root>/<project>/<staff> — record, then descend one more level
                # for <root>/<project>/<staff>/<task> worktrees. Case-insensitive
                # so PascalCase folders (LottoReward) still resolve.
                try:
                    staff_dirs = os.listdir(path)
                except OSError:
                    staff_dirs = []
                for child in staff_dirs:
                    child_path = os.path.join(path, child)
                    record(child_path)
                    if not os.path.isdir(child_path):
                        continue
                    try:
                        for task in os.listdir(child_path):
                            if task.startswith("."):
                                continue
                            record(os.path.join(child_path, task))
                    except OSError:
                        pass

        candidates.sort(key=lambda item: (-item["score"], len(item["path"])))
        project_candidates.sort(key=lambda item: (-item["score"], len(item["path"])))

        print(json.dumps({
            "ok": bool(candidates),
            "staff_id": staff_id,
            "project": project,
            "selected": candidates[0] if candidates else None,
            "matches": candidates[:10],
            "project_candidates": project_candidates[:10],
            "searched_roots": roots,
        }, ensure_ascii=False, indent=2))
        """
    )


def main() -> int:
    # Subcommand dispatch: `claim list|acquire|release ...` runs the team
    # booking flow; anything else keeps the original read-only resolve mode so
    # existing `--staff-id X --project Y` callers stay unchanged.
    if len(sys.argv) > 1 and sys.argv[1] == "claim":
        return claim_main(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="Resolve a staff/project worktree on the Hermes VPS.",
    )
    parser.add_argument("--staff-id", required=True, help="Staff id, e.g. nat, may, mind.")
    parser.add_argument("--project", required=True, help="Project slug/name, e.g. hermes-agent.")
    parser.add_argument("--task-id", help="WTL task id; when supplied, resolve exact task from lifecycle registry first.")
    parser.add_argument(
        "--registry",
        default=os.getenv("HERMES_WORKTREE_REGISTRY"),
        help="Worktree Lifecycle registry JSON (or HERMES_WORKTREE_REGISTRY).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HERMES_VPS_HOST", DEFAULT_HOST),
        help=f"SSH host, or 'local' when already on the VPS. Defaults to {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Scan local filesystem instead of SSH. Use when already inside the VPS.",
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Remote root to scan. Can be passed multiple times.",
    )
    args = parser.parse_args()

    if args.registry:
        registry_path = Path(args.registry).expanduser().resolve()
        if not registry_path.is_file():
            print(json.dumps({
                "ok": False, "error": "wtl_registry_not_found", "registry": str(registry_path),
                "owner_action": "ตรวจตำแหน่งสมุดทะเบียนกลาง; ห้าม fallback ไป Worktree คนอื่น",
            }, ensure_ascii=False, indent=2))
            return 2
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"ok": False, "error": "wtl_registry_invalid", "detail": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        matches = []
        for task in (registry.get("tasks") or {}).values():
            if _project_norm(task.get("project_id", "")) != _project_norm(args.project):
                continue
            if str(task.get("staff_id", "")).lower() != args.staff_id.lower():
                continue
            if args.task_id and task.get("task_id") != args.task_id:
                continue
            matches.append(task)
        if len(matches) == 1:
            task = matches[0]
            selected = {
                "path": task.get("worktree_path"), "branch": task.get("branch"),
                "task_id": task.get("task_id"), "machine_id": task.get("machine_id"),
                "state": task.get("state"), "writer_lease": bool(task.get("lease_id")),
            }
            selected["cd_command"] = "cd {}".format(shlex.quote(selected["path"] or ""))
            print(json.dumps({
                "ok": True, "source": "wtl_registry", "registry": str(registry_path),
                "staff_id": args.staff_id, "project": args.project, "selected": selected,
            }, ensure_ascii=False, indent=2))
            return 0
        if args.task_id or len(matches) > 1:
            print(json.dumps({
                "ok": False, "error": "wtl_task_not_unique", "match_count": len(matches),
                "task_ids": [task.get("task_id") for task in matches],
                "owner_action": "ระบุ --task-id ที่ตรงทะเบียน; ห้ามเดาเลือก Worktree",
            }, ensure_ascii=False, indent=2))
            return 2

    roots = args.roots or list(DEFAULT_ROOTS)
    remote = _remote_probe_script(args.staff_id, args.project, roots)
    is_vps_hostname = socket.gethostname().split(".", 1)[0] == "linux-nat"
    local_scan = args.local or args.host in {"local", "localhost", "127.0.0.1"} or is_vps_hostname
    if local_scan:
        proc = _run(["python3", "-c", remote, args.staff_id, args.project, *roots])
    else:
        remote_cmd = " ".join(
            [
                "python3",
                "-c",
                shlex.quote(remote),
                shlex.quote(args.staff_id),
                shlex.quote(args.project),
                *[shlex.quote(root) for root in roots],
            ]
        )
        proc = _run(["ssh", args.host, remote_cmd])
    if proc.returncode != 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "local_probe_failed" if local_scan else "ssh_probe_failed",
                    "host": args.host,
                    "stderr": proc.stderr.strip(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return proc.returncode

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return 1

    selected = payload.get("selected")
    if selected:
        if local_scan:
            selected["cd_command"] = f"cd {shlex.quote(selected['path'])}"
        else:
            selected["ssh_cd_command"] = (
                f"ssh {shlex.quote(args.host)} "
                f"{shlex.quote('cd ' + selected['path'] + ' && exec $SHELL -l')}"
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
