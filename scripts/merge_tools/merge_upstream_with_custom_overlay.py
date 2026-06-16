#!/usr/bin/env python3
"""Merge upstream/main with custom overlay support for overlapping files.

Flow:
  1) inventory and classify upstream overlap paths
  2) merge upstream/main with upstream-first conflict preference
  3) auto-resolve conflicts:
     - upstream: keep upstream side
     - preserve_custom: keep current branch side
     - official_with_overlay/manual_api_followup: keep upstream and re-apply
       current-branch delta from merge-base to HEAD
  4) report remaining unresolved paths

The script intentionally keeps manual intervention points explicit so we can
review and rerun quickly when strategy evolves.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_REF = "upstream/main"
DEFAULT_STRATEGY = ROOT / "scripts" / "merge_tools" / "hermes-merge-conflict-strategies.json"
REPORT_DIR = ROOT / "_docs" / "merge-reports"


@dataclass(frozen=True)
class ClassifiedPath:
    path: str
    action: str
    note: str


def run(cmd: list[str], *, check: bool = True, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def run_text(cmd: list[str], **kwargs) -> str:
    return run(cmd, **kwargs).stdout


def resolve_path(path: str, ref: str) -> None:
    run(["git", "checkout", ref, "--", path], check=False)


def git_add(path: str) -> None:
    run(["git", "add", "--", path], check=False)


def list_unresolved() -> list[str]:
    return [line.strip() for line in run_text(["git", "diff", "--name-only", "--diff-filter=U"]).splitlines() if line.strip()]


def list_unmerged_files_from_paths(classified: list[ClassifiedPath]) -> list[str]:
    unresolved = set(list_unresolved())
    return [item for item in classified if item.path in unresolved]


def load_action_map(inventory_path: Path, dirty_paths_file: Path | None = None) -> dict[str, ClassifiedPath]:
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    classifications = payload.get("classifications", [])
    action_map: dict[str, ClassifiedPath] = {}
    for item in classifications:
        path = item.get("path")
        if not path:
            continue
        action_map[path] = ClassifiedPath(
            path=path,
            action=item.get("action", ""),
            note=item.get("note", ""),
        )

    if dirty_paths_file and dirty_paths_file.exists():
        for line in dirty_paths_file.read_text(encoding="utf-8").splitlines():
            normalized = line.strip().replace("\\", "/")
            if normalized:
                action_map.pop(normalized, None)
    return action_map


def merge_file_overlay(target_path: str, upstream_ref: str, base_sha: str, old_head: str) -> bool:
    run(["git", "checkout", upstream_ref, "--", target_path], check=False)
    tmpdir = tempfile.mkdtemp(prefix="hermes-merge-overlay-")
    patch_file = Path(tmpdir) / "overlay.diff"

    patch_cmd = [
        "git",
        "diff",
        f"{base_sha}..{old_head}",
        "--",
        target_path,
    ]
    patch_payload = run_text(patch_cmd, check=False)
    if not patch_payload.strip():
        git_add(target_path)
        return True
    patch_file.write_text(patch_payload, encoding="utf-8")

    # 1) Overlay custom diff using 3-way apply.
    apply_cmd = ["git", "apply", "--3way", "--whitespace=nowarn", str(patch_file)]
    apply_res = run(apply_cmd, check=False)
    if apply_res.returncode != 0:
        # 2) Keep upstream file if overlay cannot be applied cleanly.
        return False

    # Keep only overlay success when patch changed the file.
    changed = run_text(["git", "diff", "--", target_path], check=False).strip()
    git_add(target_path)
    return True if apply_res.returncode == 0 else False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge upstream/main with custom overlay strategy.")
    parser.add_argument("--upstream-ref", default=DEFAULT_UPSTREAM_REF)
    parser.add_argument(
        "--strategy-file",
        default=str(DEFAULT_STRATEGY),
    )
    parser.add_argument("--commit", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="merge: sync upstream with custom overlay resolution",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy_file = Path(args.strategy_file)
    if not strategy_file.is_absolute():
        strategy_file = (ROOT / strategy_file).resolve()

    if run(["git", "diff-index", "--quiet", "HEAD", "--"], check=False).returncode != 0:
        print("Working tree is dirty. Commit or stash first.", file=sys.stderr)
        return 2

    upstream_ref = args.upstream_ref
    old_head = run_text(["git", "rev-parse", "HEAD"], check=False).strip()
    base_sha = run_text(["git", "merge-base", "HEAD", upstream_ref], check=False).strip()
    if not base_sha:
        print("Could not determine merge base.")
        return 2

    # Refresh inventory once and load the path classifications.
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "merge_tools" / "upstream_diff_inventory.py"),
            "--upstream-ref",
            upstream_ref,
            "--strategy-file",
            str(strategy_file),
        ]
    )
    inventory = ROOT / "_docs" / "upstream-main-diff-inventory.json"
    action_map = load_action_map(inventory, None)

    if args.dry_run:
        print(f"[DRY] base_sha={base_sha} old_head={old_head}")
        print(f"[DRY] inventory loaded={inventory}")
        return 0

    merge = run(["git", "merge", "-X", "theirs", "--no-commit", "--no-edit", upstream_ref], check=False)
    if merge.returncode == 0:
        if args.commit:
            run(["git", "commit", "-m", args.commit_message])
        return 0

    unresolved = list_unresolved()
    if not unresolved:
        if args.commit:
            run(["git", "commit", "-m", args.commit_message])
        return 0

    resolved = []
    blocked: list[str] = []
    for path in unresolved:
        classification = action_map.get(path)
        action = (classification.action if classification else "manual_api_followup")
        note = classification.note if classification else ""
        ok = False
        if action == "upstream":
            resolve_path(path, upstream_ref)
            git_add(path)
            ok = True
        elif action == "preserve_custom":
            resolve_path(path, "HEAD")
            git_add(path)
            ok = True
        elif action in {"manual_api_followup", "official_with_overlay"}:
            ok = merge_file_overlay(path, upstream_ref, base_sha, old_head)
        if ok:
            resolved.append((path, action, note))
        else:
            blocked.append(path)

    unresolved = list_unresolved()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = REPORT_DIR / f"merge-overlay-{stamp}.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "upstream_ref": upstream_ref,
                "base_sha": base_sha,
                "old_head": old_head,
                "resolved": [
                    {"path": p, "action": a, "note": n}
                    for p, a, n in resolved
                ],
                "blocked": blocked,
                "remaining_unresolved": unresolved,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if unresolved:
        print("Merge blocked. Remaining unresolved files:")
        for path in unresolved:
            print(f"  - {path}")
        print(f"Report: {report}")
        run(["git", "merge", "--abort"], check=False)
        return 1

    if args.commit:
        run(["git", "commit", "-m", args.commit_message])
        print(f"Committed upstream overlay merge. Report: {report}")
    else:
        print(f"Auto-resolved merge prepared (not committed). Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
