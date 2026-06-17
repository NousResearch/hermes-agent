#!/usr/bin/env python3
"""Apply 3-way merge (upstream + fork delta) for official_with_overlay paths."""

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


def git_show(ref: str, path: str) -> str | None:
    proc = run(["git", "show", f"{ref}:{path}"], cwd=REPO_ROOT)
    if proc.returncode != 0:
        return None
    return proc.stdout


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


def three_way_merge(
    path: str,
    base_sha: str,
    upstream_ref: str,
    fork_sha: str,
    *,
    sanitizers: dict[str, dict[str, object]] | None = None,
) -> tuple[int, str]:
    from overlay_sanitize import sanitize_fork_overlay_text

    base_text = git_show(base_sha, path)
    up_text = git_show(upstream_ref, path)
    fork_text = git_show(fork_sha, path)
    if up_text is None:
        return 2, f"missing upstream version: {path}"
    if fork_text is None:
        return 2, f"missing fork version: {path}"
    if base_text is None:
        base_text = ""

    fork_text = sanitize_fork_overlay_text(path, fork_text, up_text, sanitizers or {})

    with tempfile.TemporaryDirectory(prefix="hermes-3way-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "base").write_text(base_text, encoding="utf-8", newline="\n")
        (tmp_path / "up").write_text(up_text, encoding="utf-8", newline="\n")
        (tmp_path / "fork").write_text(fork_text, encoding="utf-8", newline="\n")
        proc = run(
            [
                "git",
                "merge-file",
                "-p",
                str(tmp_path / "up"),
                str(tmp_path / "base"),
                str(tmp_path / "fork"),
            ],
        )
        return proc.returncode, proc.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-way overlay merge for fork custom features.")
    parser.add_argument("--merge-base", required=True)
    parser.add_argument("--fork-ref", default="44f30816e445aa26ed92ea002a7fde33e761b6b9")
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument("--strategy-file", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--write", action="store_true", help="Write merged output to working tree.")
    parser.add_argument("--paths", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy_file = Path(args.strategy_file)
    if not strategy_file.is_absolute():
        strategy_file = (REPO_ROOT / strategy_file).resolve()

    paths = args.paths or load_overlay_paths(strategy_file)
    strategy_payload = json.loads(strategy_file.read_text(encoding="utf-8"))
    from overlay_sanitize import load_overlay_sanitizers

    sanitizers = load_overlay_sanitizers(strategy_payload)
    clean: list[str] = []
    conflicted: list[str] = []
    failed: list[tuple[str, str]] = []

    for path in paths:
        code, merged = three_way_merge(
            path,
            args.merge_base,
            args.upstream_ref,
            args.fork_ref,
            sanitizers=sanitizers,
        )
        if code == 2:
            failed.append((path, merged))
            continue
        if "<<<<<<<" in merged:
            conflicted.append(path)
        else:
            clean.append(path)
        if args.write:
            target = REPO_ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(merged, encoding="utf-8", newline="\n")

    print(f"clean={len(clean)} conflicted={len(conflicted)} failed={len(failed)}")
    for path in clean:
        print(f"  OK {path}")
    for path in conflicted:
        print(f"  CONFLICT {path}")
    for path, reason in failed:
        print(f"  FAIL {path}: {reason}")

    if args.write and clean:
        run(["git", "add", "--", *clean])

    return 1 if failed or conflicted else 0


if __name__ == "__main__":
    raise SystemExit(main())
