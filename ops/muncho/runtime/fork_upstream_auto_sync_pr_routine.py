#!/usr/bin/env python3
"""Create a fork-only upstream sync PR when the Muncho fork is behind.

Safety contract:
- pulls NousResearch/hermes-agent main into lomliev/hermes-agent only;
- creates one fork-only branch/PR when no sync PR is already open;
- never opens an upstream PR;
- may auto-merge only automation-owned clean/green upstream sync PRs;
- may auto-deploy only the exact auto-merge fork/main SHA via a restricted helper;
- never force-pushes main.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from auto_sync_hardening import (
    blocker_fingerprint,
    classify_stale_sync_pr,
    clear_blocker_delivery_state,
    decide_blocker_delivery,
)

FORK_REPO = "lomliev/hermes-agent"
UPSTREAM_REPO = "NousResearch/hermes-agent"
FORK_BRANCH = "main"
UPSTREAM_BRANCH = "main"
BRANCH_PREFIX = "codex/upstream-sync-auto-"
WORKTREE_DIR_PREFIX = BRANCH_PREFIX.replace("/", "-")
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/opt/adventico-ai-platform/hermes-home"))
GH = HERMES_HOME / "bin" / "gh-hermes"
STATE_DIR = Path("/opt/adventico-ai-platform/canonical-brain/state/private/upstream_sync_monitor")
WORKTREE_ROOT = Path("/opt/adventico-ai-platform/canonical-brain/state/private/upstream_sync_worktrees")
REPORT_DIR = Path("/opt/adventico-ai-platform/canonical-brain/state/reports")
MONITOR_LATEST = STATE_DIR / "fork-upstream-drift-latest.json"
AUTO_STATE = STATE_DIR / "auto-sync-pr-state.json"
BLOCKER_DEDUPE_STATE = STATE_DIR / "auto-sync-blocker-dedupe.json"
EXECUTE_ENV = "FORK_UPSTREAM_AUTO_SYNC_EXECUTE_APPROVED"
AUTO_MERGE_DEPLOY_ENV = "FORK_UPSTREAM_AUTO_SYNC_AUTO_MERGE_DEPLOY_APPROVED"
POST_CREATE_WAIT_SECONDS_ENV = "FORK_UPSTREAM_AUTO_SYNC_POST_CREATE_WAIT_SECONDS"
POST_CREATE_POLL_SECONDS_ENV = "FORK_UPSTREAM_AUTO_SYNC_POST_CREATE_POLL_SECONDS"
WORKTREE_RETENTION_ENV = "FORK_UPSTREAM_AUTO_SYNC_WORKTREE_RETENTION"
AUTO_DEPLOY_HELPER = Path("/usr/local/sbin/muncho-auto-deploy-release")
AUTO_DEPLOY_QUEUE_DIR = STATE_DIR / "deploy_queue"
RELEASE_AUTHOR_MAP_PATH = "scripts/release.py"
DISCORD_TOOL_PATH = "tools/discord_tool.py"
GATEWAY_RUN_PATH = "gateway/run.py"
ALLOWED_CHECK_CONCLUSIONS = {"SUCCESS", "SKIPPED", "NEUTRAL"}
WAITABLE_AUTO_MERGE_BLOCKERS = {
    "checks_missing",
    "checks_pending_or_active",
    "mergeable_UNKNOWN",
    "merge_state_UNSTABLE",
    "merge_state_UNKNOWN",
}
AUTHOR_ENTRY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<quote>['\"])(?P<key>[^'\"]+)(?P=quote)\s*:\s*"
    r"(?P<value_quote>['\"])(?P<value>[^'\"]+)(?P=value_quote)\s*,(?:\s*#.*)?$"
)
CONFLICT_RE = re.compile(r"^<<<<<<< [^\n]*\n(?P<ours>.*?)^=======\n(?P<theirs>.*?)^>>>>>>> [^\n]*\n?", re.M | re.S)


@dataclass
class CmdResult:
    cmd: list[str]
    rc: int
    stdout: str
    stderr: str


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, timeout: int | None = None) -> CmdResult:
    cp = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    result = CmdResult(cmd=cmd, rc=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)
    if check and cp.returncode != 0:
        raise RuntimeError(f"command failed rc={cp.returncode}: {' '.join(cmd)}\n{cp.stderr}")
    return result


def gh_json(args: list[str]) -> Any:
    if not GH.exists():
        raise RuntimeError(f"gh-hermes missing at {GH}")
    cp = run([str(GH), *args], timeout=120)
    return json.loads(cp.stdout or "null")


def load_monitor() -> dict[str, Any]:
    if not MONITOR_LATEST.exists():
        return {"status": "missing_monitor_state", "behind_by": None}
    return json.loads(MONITOR_LATEST.read_text(encoding="utf-8"))


def ref_sha(repo: str, branch: str) -> str:
    data = gh_json(["api", f"repos/{repo}/git/ref/heads/{branch}"])
    return data["object"]["sha"]


def compare_refs() -> dict[str, Any]:
    comp = gh_json(["api", f"repos/{UPSTREAM_REPO}/compare/{UPSTREAM_BRANCH}...lomliev:{FORK_BRANCH}"])
    return {
        "fork_main_ref": ref_sha(FORK_REPO, FORK_BRANCH),
        "upstream_main_ref": ref_sha(UPSTREAM_REPO, UPSTREAM_BRANCH),
        "merge_base": (comp.get("merge_base_commit") or {}).get("sha"),
        "ahead_by": int(comp.get("ahead_by") or 0),
        "behind_by": int(comp.get("behind_by") or 0),
        "compare_status": comp.get("status"),
        "compare_url": comp.get("html_url"),
    }


def branch_name(ts: str) -> str:
    stamp = ts.replace("-", "").replace(":", "").replace("Z", "").replace("T", "-")
    return f"{BRANCH_PREFIX}{stamp[:13]}"


def list_open_sync_prs() -> list[dict[str, Any]]:
    prs = gh_json([
        "pr",
        "list",
        "--repo",
        FORK_REPO,
        "--base",
        FORK_BRANCH,
        "--state",
        "open",
        "--json",
        "number,title,url,headRefName,headRefOid,baseRefName,isDraft,labels,createdAt,body",
    ])
    return [p for p in prs if str(p.get("headRefName", "")).startswith(("codex/upstream-sync-", BRANCH_PREFIX))]


def compare_sha(repo: str, base: str, head: str) -> dict[str, Any] | None:
    if not base or not head:
        return None
    try:
        return gh_json(["api", f"repos/{repo}/compare/{base}...{head}"])
    except Exception:
        return None


def compare_shows_head_contains_base(repo: str, base: str, head: str) -> bool:
    comp = compare_sha(repo, base, head)
    if not comp:
        return False
    return int(comp.get("behind_by") or 0) == 0 and comp.get("status") in {"identical", "ahead"}


def upstream_sha_from_pr_body(pr: dict[str, Any]) -> str | None:
    body = str(pr.get("body") or "")
    match = re.search(r"Upstream main:\s*`?([0-9a-f]{40})`?", body)
    return match.group(1) if match else None


def is_auto_owned_sync_pr(pr: dict[str, Any]) -> bool:
    head = str(pr.get("headRefName") or "")
    body = str(pr.get("body") or "")
    return head.startswith(BRANCH_PREFIX) and "Automated fork-only upstream sync PR" in body


def pr_view(number: int | str) -> dict[str, Any]:
    return gh_json(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            FORK_REPO,
            "--json",
            "number,title,url,state,isDraft,headRefName,headRefOid,baseRefName,mergeable,mergeStateStatus,statusCheckRollup,labels,body",
        ]
    )


def check_rollup_summary(pr: dict[str, Any]) -> dict[str, Any]:
    rollup = pr.get("statusCheckRollup") or []
    summary: dict[str, Any] = {
        "total": len(rollup),
        "success": 0,
        "skipped": 0,
        "neutral": 0,
        "active": 0,
        "failure_like": 0,
        "failure_like_checks": [],
        "active_checks": [],
    }
    for item in rollup:
        status = str(item.get("status") or "").upper()
        conclusion = str(item.get("conclusion") or "").upper()
        name = item.get("name")
        if status != "COMPLETED":
            summary["active"] += 1
            summary["active_checks"].append(name)
            continue
        if conclusion == "SUCCESS":
            summary["success"] += 1
        elif conclusion == "SKIPPED":
            summary["skipped"] += 1
        elif conclusion == "NEUTRAL":
            summary["neutral"] += 1
        elif conclusion not in ALLOWED_CHECK_CONCLUSIONS:
            summary["failure_like"] += 1
            summary["failure_like_checks"].append({"name": name, "conclusion": conclusion})
    summary["ready"] = (
        summary["total"] > 0
        and summary["success"] > 0
        and summary["active"] == 0
        and summary["failure_like"] == 0
    )
    return summary


def evaluate_auto_merge_deploy_pr(pr: dict[str, Any]) -> dict[str, Any]:
    view = pr_view(pr["number"])
    checks = check_rollup_summary(view)
    blockers: list[str] = []
    if os.environ.get(AUTO_MERGE_DEPLOY_ENV) != "1":
        blockers.append(f"missing_{AUTO_MERGE_DEPLOY_ENV}")
    if not is_auto_owned_sync_pr(view):
        blockers.append("not_auto_owned_sync_pr")
    if view.get("state") != "OPEN":
        blockers.append("pr_not_open")
    if view.get("isDraft"):
        blockers.append("pr_is_draft")
    if view.get("baseRefName") != FORK_BRANCH:
        blockers.append("base_not_fork_main")
    if view.get("mergeable") != "MERGEABLE":
        blockers.append(f"mergeable_{view.get('mergeable')}")
    if view.get("mergeStateStatus") != "CLEAN":
        blockers.append(f"merge_state_{view.get('mergeStateStatus')}")
    if not checks["ready"]:
        if checks["active"]:
            blockers.append("checks_pending_or_active")
        if checks["failure_like"]:
            blockers.append("checks_failed")
        if checks["total"] == 0:
            blockers.append("checks_missing")
    return {"ready": not blockers, "blockers": blockers, "pr": view, "checks": checks}


def verify_new_main_contains_head(new_main: str, expected_head: str) -> bool:
    if new_main == expected_head:
        return True
    commit = gh_json(["api", f"repos/{FORK_REPO}/commits/{new_main}"])
    parents = [p.get("sha") for p in commit.get("parents", [])]
    if expected_head in parents:
        return True
    return compare_shows_head_contains_base(FORK_REPO, expected_head, new_main)


def queue_auto_deploy_request(target_sha: str, pr_number: str) -> CmdResult:
    """Request deploy without sudo so gateway/cron no-new-privileges contexts can proceed."""
    if not re.fullmatch(r"[0-9a-f]{40}", target_sha):
        return CmdResult(cmd=["queue_auto_deploy_request"], rc=2, stdout="", stderr="invalid target sha")
    if not re.fullmatch(r"[0-9]+", pr_number):
        return CmdResult(cmd=["queue_auto_deploy_request"], rc=2, stdout="", stderr="invalid pr number")

    AUTO_DEPLOY_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "muncho-auto-deploy-request.v1",
        "target_commit": target_sha,
        "pr_number": int(pr_number),
        "requested_at": now_utc(),
        "requested_by": "fork_upstream_auto_sync_pr_routine",
        "source": "post_auto_merge",
    }
    target = AUTO_DEPLOY_QUEUE_DIR / f"deploy-{target_sha[:12]}-pr{pr_number}.json"
    tmp = AUTO_DEPLOY_QUEUE_DIR / f".{target.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    try:
        target.chmod(0o640)
    except PermissionError:
        pass
    return CmdResult(
        cmd=["queue_auto_deploy_request", str(target)],
        rc=0,
        stdout=f"queued {target}\n",
        stderr="",
    )


def auto_merge_sync_pr_and_start_deploy(pr: dict[str, Any]) -> dict[str, Any]:
    evaluation = evaluate_auto_merge_deploy_pr(pr)
    if not evaluation["ready"]:
        return {"merged": False, "deploy_started": False, **evaluation}
    view = evaluation["pr"]
    number = str(view["number"])
    expected_head = str(view["headRefOid"])
    before_main = ref_sha(FORK_REPO, FORK_BRANCH)
    merge = run(
        [
            str(GH),
            "pr",
            "merge",
            number,
            "--repo",
            FORK_REPO,
            "--merge",
            "--delete-branch",
            "--subject",
            f"Merge auto upstream sync PR #{number}",
            "--body",
            "Auto-merged by Muncho fork upstream sync routine after CLEAN merge state and green checks. "
            "No upstream action was performed.",
        ],
        check=False,
        timeout=180,
    )
    if merge.rc != 0:
        return {
            "merged": False,
            "deploy_started": False,
            **evaluation,
            "merge_rc": merge.rc,
            "merge_stdout_tail": merge.stdout[-2000:],
            "merge_stderr_tail": merge.stderr[-2000:],
        }

    after_main = ref_sha(FORK_REPO, FORK_BRANCH)
    contains_head = verify_new_main_contains_head(after_main, expected_head)
    if after_main == before_main or not contains_head:
        return {
            "merged": False,
            "deploy_started": False,
            **evaluation,
            "merge_rc": merge.rc,
            "before_main": before_main,
            "after_main": after_main,
            "expected_head": expected_head,
            "reason": "post_merge_sha_verification_failed",
        }

    deploy = queue_auto_deploy_request(after_main, number)
    return {
        "merged": True,
        "deploy_started": deploy.rc == 0,
        **evaluation,
        "merge_rc": merge.rc,
        "before_main": before_main,
        "after_main": after_main,
        "expected_head": expected_head,
        "deploy_rc": deploy.rc,
        "deploy_start_mode": "systemd_queue",
        "deploy_stdout_tail": deploy.stdout[-2000:],
        "deploy_stderr_tail": deploy.stderr[-2000:],
    }


def wait_for_pr_auto_merge_deploy(pr_number: int | str) -> dict[str, Any]:
    """Poll a newly opened automation PR so green sync PRs do not wait for the next cron tick."""
    wait_seconds = env_int(POST_CREATE_WAIT_SECONDS_ENV, 20 * 60, minimum=0, maximum=45 * 60)
    poll_seconds = env_int(POST_CREATE_POLL_SECONDS_ENV, 30, minimum=10, maximum=120)
    deadline = time.monotonic() + wait_seconds
    attempts: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None

    while True:
        result = auto_merge_sync_pr_and_start_deploy({"number": pr_number})
        last_result = result
        attempts.append(
            {
                "checked_at_utc": now_utc(),
                "ready": result.get("ready"),
                "merged": result.get("merged"),
                "deploy_started": result.get("deploy_started"),
                "blockers": result.get("blockers") or [],
                "checks": result.get("checks"),
            }
        )
        if result.get("merged") and result.get("deploy_started"):
            return {
                "status": "merged_deploy_started",
                "wait_seconds": wait_seconds,
                "poll_seconds": poll_seconds,
                "attempts": attempts,
                "result": result,
            }
        if result.get("merged") and not result.get("deploy_started"):
            return {
                "status": "merged_deploy_start_failed",
                "wait_seconds": wait_seconds,
                "poll_seconds": poll_seconds,
                "attempts": attempts,
                "result": result,
            }

        blockers = set(result.get("blockers") or [])
        if not blockers.issubset(WAITABLE_AUTO_MERGE_BLOCKERS):
            return {
                "status": "blocked_non_waitable",
                "wait_seconds": wait_seconds,
                "poll_seconds": poll_seconds,
                "attempts": attempts,
                "result": result,
            }
        if time.monotonic() >= deadline:
            return {
                "status": "timed_out_waiting_for_clean_green",
                "wait_seconds": wait_seconds,
                "poll_seconds": poll_seconds,
                "attempts": attempts,
                "result": result,
            }
        time.sleep(poll_seconds)


def stale_sync_reason(pr: dict[str, Any], fresh: dict[str, Any]) -> str | None:
    automation_owned = is_auto_owned_sync_pr(pr)
    if not automation_owned:
        return None
    head_sha = str(pr.get("headRefOid") or "")
    fork_sha = str(fresh.get("fork_main_ref") or "")
    head_already_in_fork_main = compare_shows_head_contains_base(
        FORK_REPO, head_sha, fork_sha
    )
    upstream_sha = upstream_sha_from_pr_body(pr)
    merge_base = str(fresh.get("merge_base") or "")
    upstream_snapshot_in_fork_merge_base = bool(
        upstream_sha
        and compare_shows_head_contains_base(UPSTREAM_REPO, upstream_sha, merge_base)
    )
    current_upstream_sha = str(fresh.get("upstream_main_ref") or "")
    current_upstream_contains_snapshot = bool(
        upstream_sha
        and current_upstream_sha
        and upstream_sha != current_upstream_sha
        and compare_shows_head_contains_base(
            UPSTREAM_REPO, upstream_sha, current_upstream_sha
        )
    )
    return classify_stale_sync_pr(
        automation_owned=automation_owned,
        head_already_in_fork_main=head_already_in_fork_main,
        upstream_snapshot_sha=upstream_sha,
        upstream_snapshot_in_fork_merge_base=upstream_snapshot_in_fork_merge_base,
        current_upstream_sha=current_upstream_sha,
        current_upstream_contains_snapshot=current_upstream_contains_snapshot,
    )


def apply_blocker_notification_dedupe(
    report: dict[str, Any], pr: dict[str, Any]
) -> bool:
    """Return True only when this blocker should reach the cron notifier."""

    evaluation = report.get("auto_merge_deploy") or {}
    checks = evaluation.get("checks") or {}
    fingerprint = blocker_fingerprint(
        status=str(report.get("status") or "blocked_unknown"),
        pr_number=pr.get("number"),
        head_sha=str(pr.get("headRefOid") or "") or None,
        blockers=evaluation.get("blockers") or [],
        failed_checks=checks.get("failure_like_checks") or [],
    )
    decision = decide_blocker_delivery(
        BLOCKER_DEDUPE_STATE, fingerprint=fingerprint
    )
    report["blocker_notification"] = {
        "emit": decision["emit"],
        "reason": decision["reason"],
        "suppressed_runs": decision["suppressed_runs"],
        "repeat_after_seconds": decision["repeat_after_seconds"],
        "fingerprint_prefix": fingerprint[:12],
    }
    return bool(decision["emit"])


def cleanup_stale_sync_prs(open_prs: list[dict[str, Any]], fresh: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"enabled": True, "closed": [], "kept": []}
    for pr in open_prs:
        reason = stale_sync_reason(pr, fresh)
        if not reason:
            result["kept"].append(
                {
                    "number": pr.get("number"),
                    "headRefName": pr.get("headRefName"),
                    "reason": "not_provably_auto_owned_stale",
                }
            )
            continue
        number = str(pr["number"])
        comment = (
            "Auto-closing stale fork upstream sync PR.\n\n"
            f"Reason: `{reason}`.\n"
            "This PR is automation-owned and superseded by the current fork/main state. "
            "No upstream action, merge, deploy, or force-push was performed."
        )
        close = run(
            [str(GH), "pr", "close", number, "--repo", FORK_REPO, "--comment", comment, "--delete-branch"],
            check=False,
            timeout=120,
        )
        if close.rc != 0:
            close = run([str(GH), "pr", "close", number, "--repo", FORK_REPO, "--comment", comment], check=False, timeout=120)
        bucket = "closed" if close.rc == 0 else "kept"
        result[bucket].append(
            {
                "number": pr.get("number"),
                "headRefName": pr.get("headRefName"),
                "headRefOid": pr.get("headRefOid"),
                "reason": reason if close.rc == 0 else "close_failed",
                "stale_reason": reason,
                "close_rc": close.rc,
                "close_stdout_tail": close.stdout[-1000:],
                "close_stderr_tail": close.stderr[-1000:],
            }
        )
    return result


def marker_scan(root: Path) -> list[str]:
    """Return real Git conflict markers, avoiding decorative separator false positives."""
    exclude = {".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__", ".pytest_cache"}
    marker_prefixes = ("<<<<<<< ", ">>>>>>> ", "||||||| ")
    markers: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file() or any(part in exclude for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if line.startswith(marker_prefixes):
                markers.append(f"{p.relative_to(root)}:{i}:{line[:140]}")
                if len(markers) >= 100:
                    return markers
    return markers


def disk_free_bytes(path: Path) -> int:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def cleanup_old_auto_sync_worktrees() -> list[str]:
    retain = env_int(WORKTREE_RETENTION_ENV, 2, minimum=0, maximum=20)
    if not WORKTREE_ROOT.exists():
        return []

    candidates = [
        path
        for path in WORKTREE_ROOT.iterdir()
        if path.is_dir() and path.name.startswith(WORKTREE_DIR_PREFIX)
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    deleted: list[str] = []
    for old_worktree in candidates[retain:]:
        safe_rmtree(old_worktree)
        deleted.append(str(old_worktree))
    return deleted


def changed_python_files(repo: Path, base_ref: str) -> list[str]:
    cp = run(["git", "diff", "--name-only", f"{base_ref}..HEAD"], cwd=repo)
    return [line for line in cp.stdout.splitlines() if line.endswith(".py") and (repo / line).exists()]


def author_map_conflict_start_is_inside_map(text: str, conflict_start: int) -> bool:
    map_start = text.rfind("AUTHOR_MAP", 0, conflict_start)
    if map_start < 0:
        return False
    brace_start = text.find("{", map_start, conflict_start)
    if brace_start < 0:
        return False
    return "}" not in text[brace_start:conflict_start]


def parse_author_entry_lines(block: str) -> tuple[list[tuple[str, str]], str | None]:
    entries: list[tuple[str, str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = AUTHOR_ENTRY_RE.match(line)
        if not match:
            return [], f"unsupported_author_map_line:{line[:160]}"
        entries.append((match.group("key"), line))
    return entries, None


def merge_author_map_conflict_text(text: str) -> tuple[str, dict[str, Any]]:
    matches = list(CONFLICT_RE.finditer(text))
    if not matches:
        return text, {"resolved": False, "reason": "no_conflict_block"}
    if any(not author_map_conflict_start_is_inside_map(text, match.start()) for match in matches):
        return text, {"resolved": False, "reason": "conflict_outside_AUTHOR_MAP"}

    merged_parts: list[str] = []
    last = 0
    resolved_blocks: list[dict[str, Any]] = []
    for match in matches:
        ours_entries, ours_error = parse_author_entry_lines(match.group("ours"))
        theirs_entries, theirs_error = parse_author_entry_lines(match.group("theirs"))
        if ours_error or theirs_error:
            return text, {"resolved": False, "reason": ours_error or theirs_error}

        seen: set[str] = set()
        merged_lines: list[str] = []
        for key, line in ours_entries:
            if key not in seen:
                merged_lines.append(line)
                seen.add(key)
        added_from_theirs = 0
        for key, line in theirs_entries:
            if key not in seen:
                merged_lines.append(line)
                seen.add(key)
                added_from_theirs += 1

        merged_parts.append(text[last:match.start()])
        merged_parts.append("\n".join(merged_lines) + "\n")
        last = match.end()
        resolved_blocks.append(
            {
                "ours_entries": len(ours_entries),
                "theirs_entries": len(theirs_entries),
                "added_from_theirs": added_from_theirs,
            }
        )

    merged_parts.append(text[last:])
    return "".join(merged_parts), {"resolved": True, "blocks": resolved_blocks}


def resolve_release_author_map_conflict(repo: Path) -> dict[str, Any]:
    path = repo / RELEASE_AUTHOR_MAP_PATH
    text = path.read_text(encoding="utf-8")
    merged, result = merge_author_map_conflict_text(text)
    if not result.get("resolved"):
        return {"file": RELEASE_AUTHOR_MAP_PATH, **result}
    path.write_text(merged, encoding="utf-8")
    markers = marker_scan(repo)
    release_markers = [m for m in markers if m.startswith(f"{RELEASE_AUTHOR_MAP_PATH}:")]
    if release_markers:
        return {"file": RELEASE_AUTHOR_MAP_PATH, "resolved": False, "reason": "markers_remain", "markers": release_markers[:20]}
    compile_result = run([sys.executable, "-m", "py_compile", RELEASE_AUTHOR_MAP_PATH], cwd=repo, check=False, timeout=120)
    if compile_result.rc != 0:
        return {
            "file": RELEASE_AUTHOR_MAP_PATH,
            "resolved": False,
            "reason": "py_compile_failed",
            "stderr_tail": compile_result.stderr[-2000:],
        }
    run(["git", "add", RELEASE_AUTHOR_MAP_PATH], cwd=repo, timeout=120)
    return {"file": RELEASE_AUTHOR_MAP_PATH, **result, "py_compile": "pass"}


def parse_simple_import_lines(block: str) -> tuple[list[str], str | None]:
    imports: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not re.fullmatch(r"import [A-Za-z_][A-Za-z0-9_]*", stripped):
            return [], f"unsupported_import_conflict_line:{line[:160]}"
        imports.append(stripped)
    return imports, None


def merge_simple_import_conflict_text(text: str) -> tuple[str, dict[str, Any]]:
    matches = list(CONFLICT_RE.finditer(text))
    if not matches:
        return text, {"resolved": False, "reason": "no_conflict_block"}

    merged_parts: list[str] = []
    last = 0
    resolved_blocks: list[dict[str, Any]] = []
    for match in matches:
        ours_imports, ours_error = parse_simple_import_lines(match.group("ours"))
        theirs_imports, theirs_error = parse_simple_import_lines(match.group("theirs"))
        if ours_error or theirs_error:
            return text, {"resolved": False, "reason": ours_error or theirs_error}

        seen: set[str] = set()
        merged_imports: list[str] = []
        for line in [*ours_imports, *theirs_imports]:
            if line not in seen:
                merged_imports.append(line)
                seen.add(line)

        merged_parts.append(text[last:match.start()])
        merged_parts.append("\n".join(merged_imports) + "\n")
        last = match.end()
        resolved_blocks.append(
            {
                "ours_imports": len(ours_imports),
                "theirs_imports": len(theirs_imports),
                "merged_imports": len(merged_imports),
            }
        )

    merged_parts.append(text[last:])
    return "".join(merged_parts), {"resolved": True, "blocks": resolved_blocks}


def resolve_discord_tool_import_conflict(repo: Path) -> dict[str, Any]:
    path = repo / DISCORD_TOOL_PATH
    text = path.read_text(encoding="utf-8")
    merged, result = merge_simple_import_conflict_text(text)
    if not result.get("resolved"):
        return {"file": DISCORD_TOOL_PATH, **result}
    path.write_text(merged, encoding="utf-8")
    markers = marker_scan(repo)
    discord_markers = [m for m in markers if m.startswith(f"{DISCORD_TOOL_PATH}:")]
    if discord_markers:
        return {"file": DISCORD_TOOL_PATH, "resolved": False, "reason": "markers_remain", "markers": discord_markers[:20]}
    compile_result = run([sys.executable, "-m", "py_compile", DISCORD_TOOL_PATH], cwd=repo, check=False, timeout=120)
    if compile_result.rc != 0:
        return {
            "file": DISCORD_TOOL_PATH,
            "resolved": False,
            "reason": "py_compile_failed",
            "stderr_tail": compile_result.stderr[-2000:],
        }
    run(["git", "add", DISCORD_TOOL_PATH], cwd=repo, timeout=120)
    return {"file": DISCORD_TOOL_PATH, **result, "py_compile": "pass"}


def normalized_conflict_lines(block: str) -> list[str]:
    return [line.strip() for line in block.splitlines() if line.strip()]


def merge_gateway_startup_resume_conflict_text(text: str) -> tuple[str, dict[str, Any]]:
    """Merge the one proven startup-resume sync/async migration conflict.

    The fork adds exact active-session recovery before the historical timestamp
    heuristic. Upstream moved that heuristic onto ``async_session_store``. The
    safe union preserves the exact recovery and adopts the awaited async call.
    Any other shape is rejected instead of being guessed.
    """
    matches = list(CONFLICT_RE.finditer(text))
    if len(matches) != 1:
        return text, {"resolved": False, "reason": f"expected_one_conflict_block_got:{len(matches)}"}

    match = matches[0]
    ours = normalized_conflict_lines(match.group("ours"))
    theirs = normalized_conflict_lines(match.group("theirs"))
    expected_ours = [
        "exact_marked = self._mark_runtime_status_active_sessions_resume_pending()",
        "heuristic_marked = self.session_store.suspend_recently_active()",
        "total_marked = exact_marked + heuristic_marked",
        "if total_marked:",
        "logger.info(",
        '"Marked %d in-flight session(s) as resumable from previous run "',
        '"(exact=%d, heuristic=%d)",',
        "total_marked,",
        "exact_marked,",
        "heuristic_marked,",
        ")",
    ]
    expected_theirs = [
        "suspended = await self.async_session_store.suspend_recently_active()",
        "if suspended:",
        'logger.info("Marked %d in-flight session(s) as resumable from previous run", suspended)',
    ]
    if ours != expected_ours or theirs != expected_theirs:
        return text, {
            "resolved": False,
            "reason": "unsupported_gateway_startup_resume_conflict_shape",
            "ours_lines": ours[:20],
            "theirs_lines": theirs[:20],
        }

    old_call = "heuristic_marked = self.session_store.suspend_recently_active()"
    new_call = "heuristic_marked = await self.async_session_store.suspend_recently_active()"
    ours_block = match.group("ours")
    if ours_block.count(old_call) != 1:
        return text, {"resolved": False, "reason": "expected_single_sync_heuristic_call"}
    merged_block = ours_block.replace(old_call, new_call, 1)
    merged = text[:match.start()] + merged_block + text[match.end():]
    return merged, {"resolved": True, "blocks": 1, "strategy": "fork_exact_plus_upstream_async_heuristic"}


def resolve_gateway_startup_resume_conflict(repo: Path) -> dict[str, Any]:
    path = repo / GATEWAY_RUN_PATH
    text = path.read_text(encoding="utf-8")
    merged, result = merge_gateway_startup_resume_conflict_text(text)
    if not result.get("resolved"):
        return {"file": GATEWAY_RUN_PATH, **result}
    path.write_text(merged, encoding="utf-8")
    gateway_markers = [m for m in marker_scan(repo) if m.startswith(f"{GATEWAY_RUN_PATH}:")]
    if gateway_markers:
        return {
            "file": GATEWAY_RUN_PATH,
            "resolved": False,
            "reason": "markers_remain",
            "markers": gateway_markers[:20],
        }
    compile_result = run(
        [sys.executable, "-m", "py_compile", GATEWAY_RUN_PATH],
        cwd=repo,
        check=False,
        timeout=120,
    )
    if compile_result.rc != 0:
        return {
            "file": GATEWAY_RUN_PATH,
            "resolved": False,
            "reason": "py_compile_failed",
            "stderr_tail": compile_result.stderr[-2000:],
        }
    run(["git", "add", GATEWAY_RUN_PATH], cwd=repo, timeout=120)
    return {"file": GATEWAY_RUN_PATH, **result, "py_compile": "pass"}


def try_known_conflict_auto_resolvers(repo: Path, conflicted: list[str]) -> dict[str, Any]:
    known_paths = {RELEASE_AUTHOR_MAP_PATH, DISCORD_TOOL_PATH, GATEWAY_RUN_PATH}
    supported = sorted(known_paths)
    if not conflicted or any(path not in known_paths for path in conflicted):
        return {
            "resolved": False,
            "reason": "unsupported_conflict_set",
            "supported_conflict_paths": supported,
        }

    results: list[dict[str, Any]] = []
    if RELEASE_AUTHOR_MAP_PATH in conflicted:
        result = resolve_release_author_map_conflict(repo)
        result["resolver"] = "scripts/release.py:AUTHOR_MAP"
        results.append(result)
    if DISCORD_TOOL_PATH in conflicted:
        result = resolve_discord_tool_import_conflict(repo)
        result["resolver"] = "tools/discord_tool.py:import_union"
        results.append(result)
    if GATEWAY_RUN_PATH in conflicted:
        result = resolve_gateway_startup_resume_conflict(repo)
        result["resolver"] = "gateway/run.py:exact_resume_plus_async_heuristic"
        results.append(result)

    return {
        "resolved": all(result.get("resolved") for result in results),
        "results": results,
        "supported_conflict_paths": supported,
    }


def safe_rmtree(path: Path) -> None:
    root = WORKTREE_ROOT.resolve()
    target = path.resolve()
    if root not in target.parents and target != root:
        raise RuntimeError(f"refusing to remove path outside worktree root: {target}")
    if path.exists():
        shutil.rmtree(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_summary(report: dict[str, Any]) -> str:
    status = report.get("status")
    lines = [
        "# Fork Upstream Auto Sync PR Routine",
        "",
        f"Status: `{status}`",
        "",
        f"Time: `{report.get('created_at_utc')}`",
        "",
        "Boundaries: fork-only branch/PR; no upstream PR/push; auto-merge/deploy only for auto-owned clean/green sync PRs.",
    ]
    fresh = report.get("fresh_refs") or {}
    if fresh:
        lines += [
            "",
            "## Refs",
            f"- fork_main: `{fresh.get('fork_main_ref')}`",
            f"- upstream_main: `{fresh.get('upstream_main_ref')}`",
            f"- merge_base: `{fresh.get('merge_base')}`",
            f"- ahead_by: `{fresh.get('ahead_by')}`",
            f"- behind_by: `{fresh.get('behind_by')}`",
        ]
    if report.get("pr_url"):
        lines += ["", f"PR: {report['pr_url']}"]
    auto_merge = report.get("auto_merge_deploy")
    if auto_merge:
        lines += [
            "",
            "## Auto Merge/Deploy",
            f"- ready: `{auto_merge.get('ready')}`",
            f"- merged: `{auto_merge.get('merged')}`",
            f"- deploy_started: `{auto_merge.get('deploy_started')}`",
        ]
        if auto_merge.get("blockers"):
            lines += ["- blockers: `" + ", ".join(auto_merge.get("blockers") or []) + "`"]
    if report.get("conflicted_files"):
        lines += ["", "Conflicts:", *[f"- `{p}`" for p in report["conflicted_files"]]]
    if report.get("message"):
        lines += ["", str(report["message"])]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any]) -> None:
    ts = report["created_at_utc"].replace("-", "").replace(":", "")
    write_json(STATE_DIR / "auto-sync-pr-latest.json", report)
    write_json(STATE_DIR / f"auto-sync-pr-{ts}.json", report)
    (REPORT_DIR / "fork-upstream-auto-sync-pr-latest-public-summary.md").write_text(
        render_summary(report),
        encoding="utf-8",
    )


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    ts = now_utc()
    monitor = load_monitor()
    fresh = compare_refs()
    prs = list_open_sync_prs()
    status = "dry_run_plan"
    if fresh["behind_by"] == 0:
        status = "no_drift_no_action"
    elif len(prs) == 1:
        status = "open_sync_pr_exists_no_action"
    elif len(prs) > 1:
        status = "blocked_multiple_open_sync_prs"
    return {
        "created_at_utc": ts,
        "status": status,
        "mode": "execute" if args.execute else "dry_run",
        "monitor_state": monitor,
        "fresh_refs": fresh,
        "open_sync_prs": prs,
        "proposed_branch": branch_name(ts),
        "hard_boundaries": {
            "auto_merge": "auto_owned_clean_green_sync_pr_only",
            "merge_into_fork_main": "auto_owned_clean_green_sync_pr_only",
            "upstream_pr_or_push": False,
            "runtime_deploy": "after_auto_merge_exact_sha_only",
            "dashboard_update": False,
            "gateway_restart": False,
            "conflict_auto_resolution": "known_safe_only",
            "known_conflict_auto_resolvers": [
                "scripts/release.py:AUTHOR_MAP",
                "tools/discord_tool.py:import_union",
                "gateway/run.py:exact_resume_plus_async_heuristic",
            ],
            "stale_sync_pr_cleanup": True,
            "auto_merge_green_sync_pr": os.environ.get(AUTO_MERGE_DEPLOY_ENV) == "1",
            "auto_deploy_after_auto_merge": os.environ.get(AUTO_MERGE_DEPLOY_ENV) == "1",
        },
    }


def execute(args: argparse.Namespace) -> int:
    if os.environ.get(EXECUTE_ENV) != "1":
        report = build_plan(args)
        report["status"] = "blocked_execute_env_missing"
        report["message"] = f"Missing {EXECUTE_ENV}=1"
        write_report(report)
        print(render_summary(report).rstrip())
        return 3

    report = build_plan(args)
    fresh = report["fresh_refs"]
    open_prs = report["open_sync_prs"]

    cleanup = cleanup_stale_sync_prs(open_prs, fresh)
    if cleanup["closed"]:
        open_prs = list_open_sync_prs()
        report["stale_sync_pr_cleanup"] = cleanup
        report["open_sync_prs_after_cleanup"] = open_prs
    elif cleanup["kept"]:
        report["stale_sync_pr_cleanup"] = cleanup

    report["old_worktrees_deleted"] = cleanup_old_auto_sync_worktrees()

    if fresh["behind_by"] == 0:
        report["status"] = "no_drift_no_action"
        clear_blocker_delivery_state(BLOCKER_DEDUPE_STATE)
        write_report(report)
        return 0
    if len(open_prs) == 1:
        auto_merge_deploy = auto_merge_sync_pr_and_start_deploy(open_prs[0])
        public_result = {
            k: v
            for k, v in auto_merge_deploy.items()
            if k not in {"pr"}
        }
        report["auto_merge_deploy"] = public_result
        if auto_merge_deploy.get("merged") and auto_merge_deploy.get("deploy_started"):
            report["status"] = "sync_pr_auto_merged_deploy_started"
            report["pr_url"] = (auto_merge_deploy.get("pr") or {}).get("url")
            report["pr_number"] = (auto_merge_deploy.get("pr") or {}).get("number")
            report["head"] = auto_merge_deploy.get("after_main")
            report["message"] = "Auto-owned upstream sync PR was clean/green, merged into fork main, and a detached Cloud deploy unit was started."
            clear_blocker_delivery_state(BLOCKER_DEDUPE_STATE)
            write_report(report)
            print(render_summary(report).rstrip())
            return 0
        if auto_merge_deploy.get("merged") and not auto_merge_deploy.get("deploy_started"):
            report["status"] = "blocked_deploy_start_failed_after_auto_merge"
            report["pr_url"] = (auto_merge_deploy.get("pr") or {}).get("url")
            report["pr_number"] = (auto_merge_deploy.get("pr") or {}).get("number")
            report["head"] = auto_merge_deploy.get("after_main")
            report["message"] = (
                "Auto-owned upstream sync PR was clean/green and merged into fork main, "
                "but the detached Cloud deploy unit did not start. Manual deploy reconciliation is required."
            )
            write_report(report)
            print(render_summary(report).rstrip())
            return 2
        blockers = auto_merge_deploy.get("blockers") or []
        if set(blockers).issubset(WAITABLE_AUTO_MERGE_BLOCKERS):
            report["status"] = "open_sync_pr_exists_waiting_checks_no_action"
            report["message"] = "Existing auto sync PR is not ready yet; waiting for checks/merge state."
            write_report(report)
            return 0
        report["status"] = "blocked_auto_merge_deploy_gate"
        report["message"] = "Existing sync PR did not satisfy the standing auto-merge/deploy safety gate."
        emit_blocker = apply_blocker_notification_dedupe(report, open_prs[0])
        write_report(report)
        if emit_blocker:
            print(render_summary(report).rstrip())
            return 2
        return 0
    if len(open_prs) > 1:
        report["status"] = "blocked_multiple_open_sync_prs"
        write_report(report)
        print(render_summary(report).rstrip())
        return 2

    branch = report["proposed_branch"]
    worktree = WORKTREE_ROOT / branch.replace("/", "-")
    safe_rmtree(worktree)
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)

    free_bytes = disk_free_bytes(WORKTREE_ROOT)
    report["disk_free_bytes"] = free_bytes
    if free_bytes < 5 * 1024 * 1024 * 1024:
        report.update({
            "status": "blocked_disk_space_low",
            "branch": branch,
            "worktree": str(worktree),
            "message": "Less than 5 GiB free before cloning upstream sync worktree.",
        })
        write_report(report)
        print(render_summary(report).rstrip())
        return 2

    try:
        run(["git", "clone", "https://github.com/lomliev/hermes-agent.git", str(worktree)], timeout=300)
        run(["git", "remote", "add", "upstream", "https://github.com/NousResearch/hermes-agent.git"], cwd=worktree)
        run(["git", "fetch", "origin", FORK_BRANCH], cwd=worktree, timeout=300)
        run(["git", "fetch", "upstream", UPSTREAM_BRANCH], cwd=worktree, timeout=300)
        run(["git", "checkout", "-B", branch, f"origin/{FORK_BRANCH}"], cwd=worktree)
        merge = run(["git", "merge", "--no-commit", "--no-ff", f"upstream/{UPSTREAM_BRANCH}"], cwd=worktree, check=False, timeout=300)
        if merge.rc != 0:
            conflicted = run(["git", "diff", "--name-only", "--diff-filter=U"], cwd=worktree, check=False).stdout.splitlines()
            resolver_result = try_known_conflict_auto_resolvers(worktree, conflicted)
            report["known_conflict_auto_resolvers"] = resolver_result
            remaining_conflicted = run(["git", "diff", "--name-only", "--diff-filter=U"], cwd=worktree, check=False).stdout.splitlines()
            if merge.rc != 0 and (not resolver_result.get("resolved") or remaining_conflicted):
                report.update(
                    {
                        "status": "blocked_merge_conflicts",
                        "branch": branch,
                        "worktree": str(worktree),
                        "conflicted_files": remaining_conflicted or conflicted,
                        "merge_stdout_tail": merge.stdout[-4000:],
                        "merge_stderr_tail": merge.stderr[-4000:],
                        "conflict_markers": marker_scan(worktree),
                    }
                )
                run(["git", "merge", "--abort"], cwd=worktree, check=False)
                write_report(report)
                print(render_summary(report).rstrip())
                return 2

        markers = marker_scan(worktree)
        if markers:
            report.update({"status": "blocked_conflict_markers_after_clean_merge", "branch": branch, "worktree": str(worktree), "conflict_markers": markers})
            write_report(report)
            print(render_summary(report).rstrip())
            return 2

        py_files = changed_python_files(worktree, f"origin/{FORK_BRANCH}")
        if py_files:
            run([sys.executable, "-m", "py_compile", *py_files], cwd=worktree, timeout=300)

        title = f"chore: sync fork with upstream main {report['created_at_utc'][:10]}"
        body = (
            "Automated fork-only upstream sync PR.\n\n"
            f"- Base fork_main: `{fresh['fork_main_ref']}`\n"
            f"- Upstream main: `{fresh['upstream_main_ref']}`\n"
            f"- behind_by before sync: `{fresh['behind_by']}`\n"
            f"- ahead_by before sync: `{fresh['ahead_by']}`\n\n"
            "Boundaries: no upstream PR/push; may be auto-merged/deployed only by the "
            "standing Muncho auto-sync gate after CLEAN merge state, green checks, and exact SHA verification.\n"
        )
        run(["git", "commit", "-m", title, "-m", body], cwd=worktree, timeout=300)
        head = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
        run([
            "git",
            "-c",
            f"credential.https://github.com.helper=!{GH} auth git-credential",
            "push",
            "origin",
            f"HEAD:refs/heads/{branch}",
        ], cwd=worktree, timeout=300)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            fh.write(body)
            body_file = fh.name
        try:
            pr_url = run(
                [
                    str(GH),
                    "pr",
                    "create",
                    "--repo",
                    FORK_REPO,
                    "--base",
                    FORK_BRANCH,
                    "--head",
                    branch,
                    "--title",
                    title,
                    "--body-file",
                    body_file,
                ],
                cwd=worktree,
                timeout=120,
            ).stdout.strip()
        finally:
            Path(body_file).unlink(missing_ok=True)

        pr_number = None
        if pr_url.rstrip("/").split("/")[-1].isdigit():
            pr_number = int(pr_url.rstrip("/").split("/")[-1])
        state = {
            "automation_owned": True,
            "branch": branch,
            "head": head,
            "pr_url": pr_url,
            "pr_number": pr_number,
            "created_at_utc": report["created_at_utc"],
        }
        write_json(AUTO_STATE, state)

        post_create_wait: dict[str, Any] | None = None
        if pr_number is not None:
            post_create_wait = wait_for_pr_auto_merge_deploy(pr_number)

        report.update(
            {
                "status": "sync_pr_opened_no_merge",
                "branch": branch,
                "head": head,
                "pr_url": pr_url,
                "pr_number": pr_number,
                "worktree": str(worktree),
                "py_compile_files": len(py_files),
                "post_create_wait": post_create_wait,
            }
        )
        if post_create_wait:
            result = post_create_wait.get("result") or {}
            if result.get("merged") and result.get("deploy_started"):
                report["status"] = "sync_pr_created_auto_merged_deploy_started"
                report["head"] = result.get("after_main")
                report["auto_merge_deploy"] = {k: v for k, v in result.items() if k not in {"pr"}}
                report["message"] = (
                    "Newly opened auto-owned upstream sync PR became clean/green inside the bounded "
                    "post-create wait window, was merged into fork main, and a detached Cloud deploy unit was started."
                )
            elif result.get("merged") and not result.get("deploy_started"):
                report["status"] = "blocked_post_create_deploy_start_failed_after_merge"
                report["head"] = result.get("after_main")
                report["auto_merge_deploy"] = {k: v for k, v in result.items() if k not in {"pr"}}
                report["message"] = (
                    "Newly opened auto sync PR was merged after clean/green validation, but the detached Cloud deploy "
                    "unit did not start. Manual deploy reconciliation is required."
                )
            elif post_create_wait.get("status") == "timed_out_waiting_for_clean_green":
                report["status"] = "sync_pr_opened_waiting_followup"
                report["message"] = (
                    "Newly opened auto sync PR was not clean/green before the bounded wait window ended; "
                    "the next cron tick will continue the auto-merge/deploy gate."
                )
            elif post_create_wait.get("status") == "blocked_non_waitable":
                report["status"] = "blocked_post_create_auto_merge_deploy_gate"
                report["auto_merge_deploy"] = {k: v for k, v in result.items() if k not in {"pr"}}
                report["message"] = "Newly opened sync PR did not satisfy the standing auto-merge/deploy safety gate."

        emit_blocker = True
        if report["status"] == "blocked_post_create_auto_merge_deploy_gate":
            emit_blocker = apply_blocker_notification_dedupe(
                report,
                {"number": pr_number, "headRefOid": head},
            )
        write_report(report)
        if not report["status"].startswith("blocked_") or emit_blocker:
            print(render_summary(report).rstrip())
        return 2 if report["status"].startswith("blocked_") and emit_blocker else 0
    except Exception as exc:
        report.update({"status": "blocked_execute_exception", "error_type": type(exc).__name__, "error": str(exc), "branch": branch, "worktree": str(worktree)})
        write_report(report)
        print(render_summary(report).rstrip())
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Fork-only upstream auto-sync PR routine")
    parser.add_argument("--execute", action="store_true", help="Create a fork-only sync PR if drift exists")
    parser.add_argument("--output", default=str(STATE_DIR / "auto-sync-pr-dry-run-latest.json"))
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.execute:
        return execute(args)

    plan = build_plan(args)
    write_json(Path(args.output), plan)
    if plan["status"] == "no_drift_no_action":
        return 0
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
