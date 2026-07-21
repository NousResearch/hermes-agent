#!/usr/bin/env python3
"""Investigate official upstream vs fork, then merge with Ebbinghaus preserved.

Policy (matches AGENTS.md / fork/harness):
  - upstream-only, security, lockfiles → official
  - equivalent blobs → official
  - divergent overlap → official base + fork overlay
  - custom-only (including Ebbinghaus) → preserve_custom

Ebbinghaus is confirmed absent from NousResearch/hermes-agent plugins/memory;
bounded dream memory stays on the fork path and is never overwritten by upstream.

Designed so a live ``main`` checkout can stay running: prefer creating/using
branch ``agent/ebbinghaus-bounded-dream-memory`` before merge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else range(kwargs.get("total", 0))


REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_TOOLS = Path(__file__).resolve().parent
INVESTIGATE = MERGE_TOOLS / "investigate_and_merge_upstream.py"
STRATEGY = MERGE_TOOLS / "hermes-merge-conflict-strategies.json"
REPORT_DIR = REPO_ROOT / "_docs" / "merge-reports"
BRANCH = "agent/ebbinghaus-bounded-dream-memory"
EBBINGHAUS_PATHS = (
    "plugins/memory/ebbinghaus/__init__.py",
    "plugins/memory/ebbinghaus/policies.py",
    "plugins/memory/ebbinghaus/store.py",
    "plugins/memory/ebbinghaus/plugin.yaml",
    "skills/autonomous-ai-agents/ebbinghaus-memory/SKILL.md",
    "tests/plugins/test_ebbinghaus_plugin.py",
)


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def ensure_branch() -> str:
    current = run(["git", "branch", "--show-current"], check=False).stdout.strip()
    if current == BRANCH:
        return current
    exists = run(["git", "rev-parse", "--verify", BRANCH], check=False)
    if exists.returncode == 0:
        run(["git", "switch", BRANCH])
    else:
        run(["git", "switch", "-c", BRANCH])
    return BRANCH


def upstream_has_ebbinghaus(upstream_ref: str) -> bool:
    proc = run(
        ["git", "cat-file", "-e", f"{upstream_ref}:plugins/memory/ebbinghaus/__init__.py"],
        check=False,
    )
    return proc.returncode == 0


def fork_advantage_summary(upstream_ref: str) -> dict:
    rows = []
    for path in tqdm(EBBINGHAUS_PATHS, desc="Ebbinghaus fork probe", unit="file"):
        up = run(["git", "cat-file", "-e", f"{upstream_ref}:{path}"], check=False)
        local = (REPO_ROOT / path).exists()
        rows.append(
            {
                "path": path,
                "local": local,
                "upstream": up.returncode == 0,
                "decision": (
                    "preserve_custom_fork_only"
                    if local and up.returncode != 0
                    else (
                        "official_with_overlay"
                        if local and up.returncode == 0
                        else "missing"
                    )
                ),
            }
        )
    return {
        "upstream_has_ebbinghaus_plugin": upstream_has_ebbinghaus(upstream_ref),
        "paths": rows,
        "policy": (
            "Ebbinghaus is fork-only; keep official latest for shared core, "
            "preserve_custom for ebbinghaus paths, and merge fork advantages "
            "(bounded capacity, rumination caps, archive prune_mode, dream "
            "preview/apply) into the fork plugin rather than inventing a "
            "parallel official equivalent."
        ),
    }


def write_markdown_report(payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = REPORT_DIR / f"{stamp}_ebbinghaus-upstream-investigate.md"
    buckets = payload.get("upstream_summary", {}).get("buckets", {})
    lines = [
        f"# {stamp} Ebbinghaus upstream investigate + merge plan",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- branch: `{payload.get('branch')}`",
        f"- head: `{payload.get('head')}`",
        f"- upstream_ref: `{payload.get('upstream_ref')}`",
        f"- upstream_sha: `{payload.get('upstream_sha')}`",
        f"- merge_base: `{payload.get('merge_base')}`",
        "",
        "## Upstream incoming",
        "",
        f"- commit_count: {payload.get('upstream_summary', {}).get('commit_count')}",
    ]
    for name, info in buckets.items():
        lines.append(f"- {name}: {info.get('count')}")
        for sample in info.get("samples", [])[:5]:
            lines.append(f"  - `{sample}`")
    lines.extend(
        [
            "",
            "## Ebbinghaus equivalence",
            "",
            f"- upstream_has_ebbinghaus_plugin: "
            f"**{payload.get('ebbinghaus', {}).get('upstream_has_ebbinghaus_plugin')}**",
            f"- policy: {payload.get('ebbinghaus', {}).get('policy')}",
            "",
        ]
    )
    for row in payload.get("ebbinghaus", {}).get("paths", []):
        lines.append(
            f"- `{row['path']}` local={row['local']} upstream={row['upstream']} → {row['decision']}"
        )
    lines.extend(
        [
            "",
            "## Equivalence probe (overlap files)",
            "",
            f"- decision_counts: `{payload.get('equivalence', {}).get('decision_counts')}`",
            "",
            "## Merge",
            "",
            f"- investigate_exit: {payload.get('investigate_exit')}",
            f"- merge_exit: {payload.get('merge_exit')}",
            f"- session_strategy: `{payload.get('session_strategy')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument("--investigate-only", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-branch", action="store_true")
    parser.add_argument("--allow-preflight-blockers", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_branch:
        print(f"Ensuring branch {BRANCH}...")
        ensure_branch()

    cmd = [
        sys.executable,
        str(INVESTIGATE),
        "--upstream-ref",
        args.upstream_ref,
        "--strategy-file",
        str(STRATEGY),
    ]
    if args.skip_fetch:
        cmd.append("--skip-fetch")
    if args.investigate_only or not args.merge:
        cmd.append("--investigate-only")
    else:
        cmd.extend(["--merge", "--allow-preflight-blockers"])

    print("Running investigate_and_merge_upstream...")
    proc = run(cmd, check=False)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    # Load newest investigate report if present
    reports = sorted(REPORT_DIR.glob("investigate-upstream-*.json")) if REPORT_DIR.exists() else []
    payload: dict = {"generated_at": datetime.now(UTC).isoformat()}
    if reports:
        payload.update(json.loads(reports[-1].read_text(encoding="utf-8")))
    payload["investigate_exit"] = proc.returncode
    payload["merge_exit"] = proc.returncode if args.merge else None
    payload["branch"] = run(["git", "branch", "--show-current"], check=False).stdout.strip()
    payload["head"] = run(["git", "rev-parse", "HEAD"], check=False).stdout.strip()
    payload["upstream_ref"] = args.upstream_ref
    up_sha = run(["git", "rev-parse", args.upstream_ref], check=False)
    payload["upstream_sha"] = up_sha.stdout.strip() if up_sha.returncode == 0 else None
    payload["ebbinghaus"] = fork_advantage_summary(args.upstream_ref)
    md = write_markdown_report(payload)
    print(f"Markdown report: {md}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
