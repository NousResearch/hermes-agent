#!/usr/bin/env python3
"""Re-apply fork deltas on official_with_overlay paths after an upstream merge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY = REPO_ROOT / "scripts" / "merge_tools" / "hermes-merge-conflict-strategies.json"


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def overlay_path(path: str, upstream_ref: str, base_sha: str, old_head: str, *, sanitizers: dict) -> tuple[str, str]:
    from apply_three_way_overlay import three_way_merge

    code, merged = three_way_merge(path, base_sha, upstream_ref, old_head, sanitizers=sanitizers)
    if code == 2:
        return path, f"failed: missing version for {path}"
    if "<<<<<<<" in merged:
        target = REPO_ROOT / path
        target.write_text(merged, encoding="utf-8", newline="\n")
        run(["git", "add", "--", path])
        return path, "conflict-markers"

    target = REPO_ROOT / path
    target.write_text(merged, encoding="utf-8", newline="\n")
    run(["git", "add", "--", path])
    return path, "applied"


def load_overlay_paths(strategy_file: Path) -> list[str]:
    payload = json.loads(strategy_file.read_text(encoding="utf-8"))
    paths: list[str] = []
    for rule in payload.get("rules", []):
        if rule.get("action") != "official_with_overlay":
            continue
        pattern = rule.get("pattern", "")
        if "*" in pattern or "?" in pattern:
            continue
        paths.append(pattern)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply post-merge custom overlays.")
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument("--old-head", required=True)
    parser.add_argument("--merge-base", default="")
    parser.add_argument("--strategy-file", default=str(DEFAULT_STRATEGY))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy_file = Path(args.strategy_file)
    if not strategy_file.is_absolute():
        strategy_file = (REPO_ROOT / strategy_file).resolve()

    merge_base = args.merge_base.strip() or run(
        ["git", "merge-base", args.old_head, args.upstream_ref],
    ).stdout.strip()
    if not merge_base:
        print("Could not resolve merge-base", file=sys.stderr)
        return 2

    paths = load_overlay_paths(strategy_file)
    strategy_payload = json.loads(strategy_file.read_text(encoding="utf-8"))
    from overlay_sanitize import load_overlay_sanitizers

    sanitizers = load_overlay_sanitizers(strategy_payload)
    failures: list[tuple[str, str]] = []
    for path in paths:
        result_path, status = overlay_path(
            path,
            args.upstream_ref,
            merge_base,
            args.old_head,
            sanitizers=sanitizers,
        )
        print(f"{result_path}: {status}")
        if status.startswith("failed") or status == "conflict-markers":
            failures.append((result_path, status))

    if failures:
        print(f"\nOverlay failures: {len(failures)}", file=sys.stderr)
        return 1
    print(f"\nOverlay complete ({len(paths)} paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
