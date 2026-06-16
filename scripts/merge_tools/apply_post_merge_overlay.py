#!/usr/bin/env python3
"""Re-apply fork deltas on official_with_overlay paths after an upstream merge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

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


def overlay_path(path: str, upstream_ref: str, base_sha: str, old_head: str) -> tuple[str, str]:
    run(["git", "checkout", upstream_ref, "--", path])
    diff = run(["git", "diff", f"{base_sha}..{old_head}", "--", path])
    if not diff.stdout.strip():
        run(["git", "add", "--", path])
        return path, "no-delta"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".diff",
        delete=False,
    ) as handle:
        handle.write(diff.stdout)
        patch_file = handle.name

    apply_res = run(["git", "apply", "--3way", "--whitespace=nowarn", patch_file])
    Path(patch_file).unlink(missing_ok=True)
    if apply_res.returncode != 0:
        detail = (apply_res.stderr or apply_res.stdout or "apply failed").strip()
        return path, f"failed: {detail[:300]}"

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
    failures: list[tuple[str, str]] = []
    for path in paths:
        result_path, status = overlay_path(path, args.upstream_ref, merge_base, args.old_head)
        print(f"{result_path}: {status}")
        if status.startswith("failed"):
            failures.append((result_path, status))

    if failures:
        print(f"\nOverlay failures: {len(failures)}", file=sys.stderr)
        return 1
    print(f"\nOverlay complete ({len(paths)} paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
