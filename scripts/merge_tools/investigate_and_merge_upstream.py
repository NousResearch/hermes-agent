#!/usr/bin/env python3
"""Investigate upstream vs fork, then merge with equivalence-aware policy.

Policy:
  - upstream-only / security / lockfiles → take official
  - custom-only → preserve fork
  - equivalent (fork blob == upstream blob) → take official
  - near-equivalent / overlapping → official base + re-apply fork advantages
    (official_with_overlay / three-way overlay)

Designed for Windows worktrees so the live ``main`` checkout is never touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for dry environments
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else range(kwargs.get("total", 0))


REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_TOOLS = Path(__file__).resolve().parent
DEFAULT_STRATEGY = MERGE_TOOLS / "hermes-merge-conflict-strategies.json"
REPORT_DIR = REPO_ROOT / "_docs" / "merge-reports"
INVENTORY_JSON = REPO_ROOT / "_docs" / "upstream-main-diff-inventory.json"


def run(
    cmd: list[str],
    *,
    check: bool = True,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_show(ref: str, path: str) -> bytes | None:
    proc = run(["git", "show", f"{ref}:{path}"], check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.encode("utf-8", errors="replace") if isinstance(proc.stdout, str) else proc.stdout


def git_show_text(ref: str, path: str) -> str | None:
    proc = run(["git", "show", f"{ref}:{path}"], check=False)
    return proc.stdout if proc.returncode == 0 else None


def bucket_commit(line: str) -> str:
    low = line.lower()
    if any(k in low for k in ("cve", "security", "ssrf", "harden", "xss", "auth", "credential")):
        return "security"
    if " fix" in f" {low}" or low.split(" ", 1)[-1].startswith("fix"):
        return "bugfix"
    if any(k in low for k in ("feat", "perf")):
        return "feature_perf"
    return "other"


def probe_equivalence(
    classifications: list[dict],
    *,
    merge_base: str,
    upstream_ref: str,
    head_ref: str = "HEAD",
) -> list[dict]:
    rows: list[dict] = []
    overlap = [
        item
        for item in classifications
        if item.get("touched_upstream") and item.get("touched_custom")
    ]
    for item in tqdm(overlap, desc="Equivalence probe", unit="file"):
        path = item["path"]
        base = git_show_text(merge_base, path)
        head = git_show_text(head_ref, path)
        up = git_show_text(upstream_ref, path)
        if base is None or head is None or up is None:
            decision = "missing_version"
            recommended = item.get("action", "manual_api_followup")
        elif head == up:
            decision = "equivalent_use_upstream"
            recommended = "upstream"
        elif abs(len(head) - len(up)) < 64 and head.count("\n") == up.count("\n"):
            # Near-identical size/line count: prefer official with fork overlay.
            decision = "near_equivalent_overlay"
            recommended = "official_with_overlay"
        else:
            decision = "divergent_overlay"
            recommended = (
                "official_with_overlay"
                if item.get("action") in {"manual_api_followup", "official_with_overlay"}
                else item.get("action", "official_with_overlay")
            )
        rows.append(
            {
                "path": path,
                "prior_action": item.get("action"),
                "decision": decision,
                "recommended_action": recommended,
                "fork_delta_bytes": (len(head) - len(base)) if base and head else None,
                "upstream_delta_bytes": (len(up) - len(base)) if base and up else None,
                "note": (
                    "Latest and custom are equivalent; keep official and retain "
                    "fork advantages only via non-overlapping preserve_custom paths."
                    if decision == "equivalent_use_upstream"
                    else "Take official latest, then re-apply fork advantages as overlay."
                ),
            }
        )
    return rows


def write_strategy_overrides(probe_rows: list[dict], strategy_path: Path) -> Path:
    """Emit a session strategy that promotes equivalent/near-equivalent decisions."""
    payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    existing = {(rule.get("pattern"), rule.get("context", "always")) for rule in payload.get("rules", [])}
    added = 0
    for row in probe_rows:
        pattern = row["path"]
        action = row["recommended_action"]
        key = (pattern, "overlap_only")
        if key in existing:
            continue
        if row["decision"] not in {"equivalent_use_upstream", "near_equivalent_overlay", "divergent_overlay"}:
            continue
        payload["rules"].insert(
            0,
            {
                "pattern": pattern,
                "action": action,
                "note": f"auto:{row['decision']} — {row['note']}",
                "context": "overlap_only",
            },
        )
        existing.add(key)
        added += 1

    out = REPORT_DIR / f"strategy-equivalence-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Strategy overrides written ({added} rules): {out}")
    return out


def summarize_upstream(merge_base: str, upstream_ref: str) -> dict:
    log = run(["git", "log", "--oneline", f"{merge_base}..{upstream_ref}"]).stdout.splitlines()
    buckets: dict[str, list[str]] = defaultdict(list)
    for line in log:
        buckets[bucket_commit(line)].append(line)
    return {
        "commit_count": len(log),
        "buckets": {k: {"count": len(v), "samples": v[:12]} for k, v in buckets.items()},
        "commits": log,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument("--strategy-file", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--investigate-only", action="store_true")
    parser.add_argument("--merge", action="store_true", help="Run sync_all merge after investigation.")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="merge: sync upstream/main (features/security/bugfixes) with fork overlays",
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--allow-preflight-blockers", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy_file = Path(args.strategy_file)
    if not strategy_file.is_absolute():
        strategy_file = (REPO_ROOT / strategy_file).resolve()

    if not args.skip_fetch:
        print("Fetching upstream...")
        fetch = run(["git", "fetch", "upstream", "--prune"], check=False)
        if fetch.returncode != 0:
            print(fetch.stderr or fetch.stdout, file=sys.stderr)
            return 2

    print("Building inventory...")
    inv_proc = run(
        [
            sys.executable,
            str(MERGE_TOOLS / "upstream_diff_inventory.py"),
            "--upstream-ref",
            args.upstream_ref,
            "--strategy-file",
            str(strategy_file),
        ],
        check=False,
    )
    if inv_proc.returncode != 0:
        print(inv_proc.stderr or inv_proc.stdout, file=sys.stderr)
        return 2

    inventory = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    merge_base = inventory["refs"]["merge_base"]
    upstream_summary = summarize_upstream(merge_base, args.upstream_ref)
    probe_rows = probe_equivalence(
        inventory.get("classifications", []),
        merge_base=merge_base,
        upstream_ref=args.upstream_ref,
    )
    decision_counts = Counter(row["decision"] for row in probe_rows)
    session_strategy = write_strategy_overrides(probe_rows, strategy_file)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": run(["git", "branch", "--show-current"], check=False).stdout.strip(),
        "head": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "upstream_ref": args.upstream_ref,
        "upstream_sha": run(["git", "rev-parse", args.upstream_ref]).stdout.strip(),
        "merge_base": merge_base,
        "inventory_counts": inventory.get("counts"),
        "action_counts": inventory.get("action_counts"),
        "upstream_summary": upstream_summary,
        "equivalence": {
            "decision_counts": dict(decision_counts),
            "rows": probe_rows,
        },
        "session_strategy": str(session_strategy),
        "policy": {
            "equivalent": "use official latest",
            "divergent_or_near": "official latest + fork advantage overlay",
            "custom_only": "preserve_custom",
        },
    }
    report_path = REPORT_DIR / f"investigate-upstream-{stamp}.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Investigation report: {report_path}")
    print(f"Equivalence decisions: {dict(decision_counts)}")
    print(
        "Upstream incoming: "
        f"{upstream_summary['commit_count']} commits "
        f"({ {k: v['count'] for k, v in upstream_summary['buckets'].items()} })"
    )

    if args.investigate_only or not args.merge:
        return 0

    sync_args = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sync_all.py"),
        "--merge",
        "--skip-fetch",
        "--upstream-ref",
        args.upstream_ref,
        "--strategy-file",
        str(session_strategy),
        "--conflict-policy",
        "official-first",
        "--allow-preflight-blockers",
        "--commit-message",
        args.commit_message,
    ]
    if args.commit:
        sync_args.append("--commit")

    print("Merging with official-first + fork overlay...")
    merge_proc = run(sync_args, check=False)
    print(merge_proc.stdout)
    if merge_proc.stderr:
        print(merge_proc.stderr, file=sys.stderr)
    report["merge_exit_code"] = merge_proc.returncode
    report["merge_stdout_tail"] = (merge_proc.stdout or "")[-4000:]
    report["merge_stderr_tail"] = (merge_proc.stderr or "")[-4000:]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merge_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
