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
import textwrap


DEFAULT_HOST = "linux-nat@103.142.150.185"
# Team work now lives under ~/.worktree/<project>/<staff>/<YYYYMMDD>-<task>
# (three levels). The first two roots keep the per-project reference clone and
# its 2-level legacy worktrees. The probe handles both depths.
DEFAULT_ROOTS = (
    "/home/linux-nat/projects",
    "/srv/projects",
    "/home/linux-nat/.worktree",
)


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
    parser = argparse.ArgumentParser(
        description="Resolve a staff/project worktree on the Hermes VPS.",
    )
    parser.add_argument("--staff-id", required=True, help="Staff id, e.g. nat, may, mind.")
    parser.add_argument("--project", required=True, help="Project slug/name, e.g. hermes-agent.")
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
