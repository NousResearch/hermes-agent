#!/usr/bin/env python3
"""Unified sync: NousResearch upstream + layered OpenClaw (openclaw-sync + clawdbot-main).

Pipeline:
  1. Inventory + dry-run classify (policy)
  2. Optional: sync OpenClaw vendor extensions
  3. Merge upstream/main with custom-first conflict preference
  4. Auto-resolve conflicts per hermes-merge-conflict-strategies.json
  5. Emit reports under _docs/merge-reports/

Examples:
  py -3 scripts/sync_all.py --dry-run
  py -3 scripts/sync_all.py --openclaw-vendor --dry-run
  py -3 scripts/sync_all.py --openclaw-vendor --execute --port-cli-tools
  py -3 scripts/sync_all.py --merge --target main
  py -3 scripts/sync_all.py --inventory-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_TOOLS = REPO_ROOT / "scripts" / "merge_tools"
REPORT_DIR = REPO_ROOT / "_docs" / "merge-reports"
DEFAULT_REMOTE = "upstream"
DEFAULT_UPSTREAM_REF = f"{DEFAULT_REMOTE}/main"
DEFAULT_STRATEGY = MERGE_TOOLS / "hermes-merge-conflict-strategies.json"
INVENTORY_JSON = REPO_ROOT / "_docs" / "upstream-main-diff-inventory.json"
DEFAULT_CLAW_ROOT = REPO_ROOT.parent / "clawdbot-main3"


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def _python_script(rel: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, str(REPO_ROOT / rel), *args], check=check)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], check=check)


def _write_report(name: str, payload: dict[str, object]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"{name}-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _working_tree_clean() -> bool:
    return _git("diff-index", "--quiet", "HEAD", "--", check=False).returncode == 0


def _unmerged() -> list[str]:
    proc = _git("diff", "--name-only", "--diff-filter=U", check=False)
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _collect_blockers(report_path: Path) -> list[str]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return list(payload.get("blocked_paths") or [])


def stage_inventory(upstream_ref: str, strategy_file: Path) -> dict[str, object]:
    _python_script(
        "scripts/merge_tools/upstream_diff_inventory.py",
        "--upstream-ref",
        upstream_ref,
        "--strategy-file",
        str(strategy_file),
    )
    return json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))


def stage_dry_run_resolver(upstream_ref: str, strategy_file: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = REPO_ROOT / "_docs" / f"merge-conflict-resolution-dry-run-{stamp}.md"
    report_path = REPORT_DIR / f"upstream-dry-run-{stamp}.json"
    proc = _python_script(
        "scripts/merge_tools/resolve_merge_conflicts.py",
        "--upstream-ref",
        upstream_ref,
        "--strategy-file",
        str(strategy_file),
        "--paths-file",
        str(INVENTORY_JSON),
        "--dry-run",
        "--log-file",
        str(log_path),
        "--report-json",
        str(report_path),
        "--strict",
        check=False,
    )
    if proc.returncode != 0 and report_path.exists():
        return report_path
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return report_path


def stage_merge(upstream_ref: str, strategy_file: Path, *, conflict_preference: str) -> str:
    old_head = _git("rev-parse", "HEAD").stdout.strip()
    merge_x = "ours" if conflict_preference == "custom-first" else "theirs"
    proc = _git("merge", "-X", merge_x, "--no-edit", "--no-commit", upstream_ref, check=False)
    if proc.returncode == 0:
        return old_head

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = REPO_ROOT / "_docs" / f"merge-conflict-resolution-{stamp}.md"
    report_path = REPORT_DIR / f"merge-conflict-resolution-{stamp}.json"
    resolve_proc = _python_script(
        "scripts/merge_tools/resolve_merge_conflicts.py",
        "--upstream-ref",
        upstream_ref,
        "--strategy-file",
        str(strategy_file),
        "--paths-file",
        str(INVENTORY_JSON),
        "--only-unresolved",
        "--old-head",
        old_head,
        "--log-file",
        str(log_path),
        "--report-json",
        str(report_path),
        "--strict",
        check=False,
    )
    blockers = _collect_blockers(report_path) if report_path.exists() else []
    unresolved = _unmerged()
    if resolve_proc.returncode != 0 or blockers or unresolved:
        _git("merge", "--abort", check=False)
        raise RuntimeError(
            "Merge blocked. "
            f"blockers={blockers[:10]} unresolved={unresolved[:10]} "
            f"report={report_path}"
        )
    return old_head


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync upstream Hermes + OpenClaw vendor.")
    parser.add_argument("--dry-run", action="store_true", help="Inventory + classify only.")
    parser.add_argument("--inventory-only", action="store_true", help="Only write diff inventory.")
    parser.add_argument("--merge", action="store_true", help="Perform git merge after inventory.")
    parser.add_argument("--target", default="", help="Checkout target branch before merge.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--upstream-ref", default=DEFAULT_UPSTREAM_REF)
    parser.add_argument("--strategy-file", default=str(DEFAULT_STRATEGY))
    parser.add_argument(
        "--conflict-policy",
        choices=["custom-first", "official-first"],
        default="official-first",
    )
    parser.add_argument(
        "--openclaw-vendor",
        action="store_true",
        help="Run scripts/sync_openclaw_vendor.py before merge.",
    )
    parser.add_argument("--openclaw-execute", action="store_true", help="Apply vendor copy (not dry-run).")
    parser.add_argument(
        "--port-cli-tools",
        action="store_true",
        help="With --openclaw-vendor: port clawdbot-main scripts/tools to scripts/openclaw_ports/.",
    )
    parser.add_argument("--claw-root", default=str(DEFAULT_CLAW_ROOT))
    parser.add_argument("--clawdbot-main", default=None, help="Override clawdbot-main path.")
    parser.add_argument("--commit", action="store_true", help="Commit merge after successful resolution.")
    parser.add_argument(
        "--commit-message",
        default="merge: sync upstream/main + OpenClaw vendor with fork policy",
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument(
        "--allow-preflight-blockers",
        action="store_true",
        help=(
            "Continue into --merge when the policy dry-run reports manual overlay "
            "blockers. The blocker list is still recorded in the report."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not any((args.dry_run, args.inventory_only, args.merge, args.openclaw_vendor)):
        print("Nothing to do. Use --dry-run, --inventory-only, --merge, and/or --openclaw-vendor.", file=sys.stderr)
        return 2

    strategy_file = Path(args.strategy_file)
    if not strategy_file.is_absolute():
        strategy_file = (REPO_ROOT / strategy_file).resolve()

    report: dict[str, object] = {
        "started_at": datetime.now(UTC).isoformat(),
        "upstream_ref": args.upstream_ref,
        "strategy_file": str(strategy_file),
        "dry_run": args.dry_run,
        "steps": [],
    }

    try:
        if args.merge and not _working_tree_clean():
            raise RuntimeError("Working tree must be clean before --merge (commit or stash).")

        if args.target:
            _git("checkout", args.target)
            report["steps"].append(f"checkout {args.target}")

        if not args.skip_fetch:
            _git("fetch", args.remote, "--prune")
            report["steps"].append(f"fetch {args.remote}")

        if args.openclaw_vendor:
            vendor_args = ["--claw-root", args.claw_root]
            if args.clawdbot_main:
                vendor_args.extend(["--clawdbot-main", args.clawdbot_main])
            if args.openclaw_execute:
                vendor_args.append("--execute")
            else:
                vendor_args.append("--dry-run")
            if args.port_cli_tools:
                vendor_args.append("--port-cli-tools")
            _python_script("scripts/sync_openclaw_vendor.py", *vendor_args)
            report["steps"].append("openclaw layered vendor sync (openclaw-sync + clawdbot-main)")

        inventory = stage_inventory(args.upstream_ref, strategy_file)
        report["inventory"] = {
            "counts": inventory.get("counts"),
            "action_counts": inventory.get("action_counts"),
        }
        report["steps"].append("upstream inventory")

        if args.inventory_only:
            path = _write_report("sync-all-inventory", report)
            print(f"Inventory only. Report: {path}")
            return 0

        dry_report = stage_dry_run_resolver(args.upstream_ref, strategy_file)
        blockers = _collect_blockers(dry_report)
        report["dry_run_report"] = str(dry_report)
        report["preflight_blockers"] = blockers
        report["steps"].append("dry-run classify")

        if blockers and not args.allow_preflight_blockers:
            path = _write_report("sync-all-blocked", report)
            print(f"Preflight blockers ({len(blockers)}). Manual overlay required.")
            print(f"Report: {path}")
            for item in blockers[:25]:
                print(f"  - {item}")
            return 2
        if blockers:
            report["steps"].append("preflight blockers allowed for explicit merge")

        if args.dry_run and not args.merge:
            path = _write_report("sync-all-dry-run-ok", report)
            print(f"Dry-run OK. Report: {path}")
            return 0

        if args.merge:
            old_head = stage_merge(args.upstream_ref, strategy_file, conflict_preference=args.conflict_policy)
            report["steps"].append("merge + policy resolve")
            overlay_proc = _python_script(
                "scripts/merge_tools/apply_post_merge_overlay.py",
                "--upstream-ref",
                args.upstream_ref,
                "--old-head",
                old_head,
                "--strategy-file",
                str(strategy_file),
                check=False,
            )
            report["overlay_exit_code"] = overlay_proc.returncode
            if overlay_proc.returncode != 0:
                raise RuntimeError(
                    "Post-merge overlay failed. "
                    f"stdout={overlay_proc.stdout[-2000:]} "
                    f"stderr={overlay_proc.stderr[-2000:]}"
                )
            report["steps"].append("post-merge overlay")
            if args.commit:
                _git("commit", "-m", args.commit_message)
                report["steps"].append("commit")

        path = _write_report("sync-all-ok", report)
        print(f"Sync complete. Report: {path}")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        path = _write_report("sync-all-failed", report)
        print(f"Failed: {exc}\nReport: {path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
