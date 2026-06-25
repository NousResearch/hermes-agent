#!/usr/bin/env python3
"""Sync SakanaAI/AI-Scientist upstream into vendor/openclaw-mirror/AI-Scientist.

Strategy:
  1. Shallow-fetch upstream (SakanaAI/AI-Scientist@main)
  2. Copy upstream tree into vendor target (skip results/cache dirs)
  3. Re-apply local overlay paths (fork templates, _overlay/, etc.)
  4. Copy tracked overlay_source from scripts/merge_tools/overlays/ai-scientist/

Usage:
  py -3 scripts/sync_ai_scientist_vendor.py --dry-run
  py -3 scripts/sync_ai_scientist_vendor.py --execute
  py -3 scripts/sync_ai_scientist_vendor.py --execute --ref v2.0.0
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_TOOLS = REPO_ROOT / "scripts" / "merge_tools"
DEFAULT_CONFIG = MERGE_TOOLS / "ai_scientist_vendor_layers.json"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "ai-scientist-upstream"

sys.path.insert(0, str(MERGE_TOOLS))
from openclaw_layered_sync import _file_hash, _should_skip  # noqa: E402


@dataclass(frozen=True)
class OverlayRule:
    pattern: str
    regex: re.Pattern[str]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _glob_to_regex(pattern: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**", index):
            parts.append("(?:.*/)?")
            index += 2
            if index < len(pattern) and pattern[index] == "/":
                index += 1
            continue
        if pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
            continue
        ch = pattern[index]
        if ch in ".^$+?{}[]|()\\":
            parts.append("\\" + ch)
        else:
            parts.append(ch)
        index += 1
    return "^" + "".join(parts) + "$"


def compile_overlay_rules(patterns: list[str]) -> list[OverlayRule]:
    return [OverlayRule(pattern=p, regex=re.compile(_glob_to_regex(p))) for p in patterns]


def matches_overlay(rel: str, rules: list[OverlayRule]) -> bool:
    return any(rule.regex.match(rel) for rule in rules)


def collect_files(
    root: Path,
    *,
    skip_dirs: set[str],
    skip_globs: tuple[str, ...],
) -> dict[str, str]:
    files: dict[str, str] = {}
    if not root.is_dir():
        return files
    for path in root.rglob("*"):
        if not path.is_file() or _should_skip(path, skip_dirs, skip_globs):
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = _file_hash(path)
    return files


def _rmtree_robust(path: Path) -> None:
    def _onerror(func, p, _exc_info) -> None:
        if not os.access(p, os.W_OK):
            os.chmod(p, stat.S_IWUSR)
            func(p)
        else:
            raise

    shutil.rmtree(path, onerror=_onerror)


def ensure_upstream(cache_dir: Path, url: str, ref: str) -> tuple[Path, str]:
    """Return (upstream_tree_path, resolved_sha)."""
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / ".git").is_dir():
        if cache_dir.exists():
            _rmtree_robust(cache_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, url, str(cache_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        fetch = subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", ref],
            cwd=str(cache_dir),
            capture_output=True,
            text=True,
        )
        if fetch.returncode != 0:
            _rmtree_robust(cache_dir)
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", ref, url, str(cache_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                ["git", "checkout", "FETCH_HEAD"],
                cwd=str(cache_dir),
                check=True,
                capture_output=True,
                text=True,
            )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cache_dir),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return cache_dir, sha


def snapshot_overlay(source: Path, rules: list[OverlayRule], tmp: Path) -> dict[str, Path]:
    """Copy preserve_paths from source into tmp; return rel -> copied path."""
    saved: dict[str, Path] = {}
    if not source.is_dir():
        return saved
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        if not matches_overlay(rel, rules):
            continue
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        saved[rel] = dest
    return saved


def restore_overlay(saved: dict[str, Path], target: Path) -> list[str]:
    restored: list[str] = []
    for rel, src in sorted(saved.items()):
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        restored.append(rel)
    return restored


def apply_overlay_source(source: Path, target: Path) -> list[str]:
    """Copy git-tracked Hermes overlay tree into vendor target."""
    applied: list[str] = []
    if not source.is_dir():
        return applied
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        applied.append(rel)
    return applied


def overlay_source_files(source: Path) -> list[str]:
    if not source.is_dir():
        return []
    return sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file())


def copy_upstream_tree(source: Path, target: Path, skip_dirs: set[str]) -> None:
    def _ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in skip_dirs}

    if target.exists():
        _rmtree_robust(target)
    shutil.copytree(source, target, ignore=_ignore)


def build_plan(
    upstream: Path,
    target: Path,
    *,
    skip_dirs: set[str],
    skip_globs: tuple[str, ...],
    overlay_rules: list[OverlayRule],
) -> dict[str, object]:
    up_files = collect_files(upstream, skip_dirs=skip_dirs, skip_globs=skip_globs)
    cur_files = collect_files(target, skip_dirs=skip_dirs, skip_globs=skip_globs) if target.exists() else {}

    added = sorted(set(up_files) - set(cur_files))
    removed = sorted(set(cur_files) - set(up_files))
    changed = sorted(rel for rel in set(up_files) & set(cur_files) if up_files[rel] != cur_files[rel])

    overlay_kept = sorted(
        rel for rel in cur_files if matches_overlay(rel, overlay_rules) and rel not in up_files
    )
    overlay_overridden = sorted(
        rel for rel in changed if matches_overlay(rel, overlay_rules)
    )

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "overlay_preserved_local_only": overlay_kept,
        "overlay_will_restore_after_sync": overlay_overridden,
    }


def write_report(payload: dict[str, object]) -> Path:
    out_dir = REPO_ROOT / "_docs" / "merge-reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"ai-scientist-vendor-sync-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync SakanaAI/AI-Scientist vendor with local overlays.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ref", type=str, default=None, help="Upstream git ref (default: config upstream_ref).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.execute:
        print("Specify --dry-run and/or --execute", file=sys.stderr)
        return 2

    config = load_config(args.config.resolve())
    target = (REPO_ROOT / config["vendor_target"]).resolve()
    skip_dirs = set(config.get("skip_dir_names", []))
    skip_globs = tuple(config.get("skip_globs", []))
    overlay_rules = compile_overlay_rules(config.get("preserve_paths", []))
    upstream_ref = args.ref or config.get("upstream_ref", "main")
    upstream_url = config.get("upstream_url", "https://github.com/SakanaAI/AI-Scientist.git")

    overlay_source_rel = config.get("overlay_source", "")
    overlay_source = (REPO_ROOT / overlay_source_rel).resolve() if overlay_source_rel else None
    overlay_source_list = overlay_source_files(overlay_source) if overlay_source else []

    upstream_path, sha = ensure_upstream(args.cache_dir.resolve(), upstream_url, upstream_ref)
    plan = build_plan(
        upstream_path,
        target,
        skip_dirs=skip_dirs,
        skip_globs=skip_globs,
        overlay_rules=overlay_rules,
    )

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream_url": upstream_url,
        "upstream_ref": upstream_ref,
        "upstream_sha": sha,
        "target": str(target),
        "dry_run": args.dry_run,
        "executed": False,
        "plan": plan,
        "overlay_source": str(overlay_source) if overlay_source else None,
        "overlay_source_files": overlay_source_list,
    }

    print("AI-Scientist vendor sync:")
    print(
        f"  upstream {sha[:12]} → {target.name}:"
        f" +{len(plan['added'])} ~{len(plan['changed'])} -{len(plan['removed'])}"
    )
    print(f"  overlay restore: {len(plan['overlay_will_restore_after_sync'])} paths")
    print(f"  overlay local-only: {len(plan['overlay_preserved_local_only'])} paths")
    if overlay_source_list:
        print(f"  overlay_source apply: {len(overlay_source_list)} tracked file(s)")

    if args.execute:
        with tempfile.TemporaryDirectory(prefix="ai-scientist-overlay-") as tmp_name:
            tmp = Path(tmp_name)
            saved = snapshot_overlay(target, overlay_rules, tmp)
            copy_upstream_tree(upstream_path, target, skip_dirs)
            restored = restore_overlay(saved, target)
            applied = apply_overlay_source(overlay_source, target) if overlay_source else []
            report["executed"] = True
            report["overlay_restored"] = restored
            report["overlay_source_applied"] = applied
            print(f"Applied sync; restored {len(restored)} overlay file(s).")
            if applied:
                print(f"Applied overlay_source: {len(applied)} file(s).")

    report_path = write_report(report)
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
