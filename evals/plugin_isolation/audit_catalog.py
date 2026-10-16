"""Run the plugin-host readiness audit over every plugin-catalog entry at its pinned commit.

    python evals/plugin_isolation/audit_catalog.py --out <dir> [--jobs 12] [--limit N]

Fetches each entry's repo at its pinned sha (shallow, blob-filtered) into ``<dir>/src/``, runs
:func:`hermes_cli.plugin_isolation_audit.audit_plugin_dir` on the entry's subdir, and writes
``<dir>/results.json`` plus a markdown summary on stdout: how many catalog plugins run in the
plugin host unchanged, and the reasons the rest need in-process loading.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hermes_cli.plugin_catalog import load_catalog
from hermes_cli.plugin_isolation_audit import audit_plugin_dir


def _git(args, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=180,
                   stdin=subprocess.DEVNULL)


def _fetch(entry, src_root: Path) -> Path:
    dest = src_root / re.sub(r"[^A-Za-z0-9_.-]", "_", entry.name)
    marker = dest / ".audited_sha"
    if marker.is_file() and marker.read_text(encoding="utf-8-sig").strip() == entry.sha:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        _git(["init", "-q"], dest)
        _git(["remote", "add", "origin", entry.repo if "://" in entry.repo else f"https://github.com/{entry.repo}"], dest)
    _git(["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", entry.sha], dest)
    _git(["checkout", "-q", "--force", "FETCH_HEAD"], dest)
    marker.write_text(entry.sha, encoding="utf-8")
    return dest


def _audit(entry, src_root: Path) -> dict:
    row = {"name": entry.name, "repo": entry.repo, "sha": entry.sha, "tier": entry.tier, "category": entry.category}
    try:
        checkout = _fetch(entry, src_root)
    except Exception as exc:
        return {**row, "verdict": "fetch_failed", "reasons": [str(exc)[:300]], "notes": []}
    plugin_dir = checkout / entry.subdir if entry.subdir else checkout
    if not plugin_dir.is_dir():
        return {**row, "verdict": "fetch_failed", "reasons": [f"subdir {entry.subdir!r} missing"], "notes": []}
    return {**row, **audit_plugin_dir(plugin_dir).to_dict()}


def _reason_class(reason: str) -> str:
    if "patches" in reason or "setattr()" in reason:
        return "patches a Hermes module attribute"
    if "directly instead of through ctx" in reason:
        return "mutates a Hermes registry directly (not through ctx)"
    match = re.search(r"ctx\.(\w+)\(\)", reason)
    if match:
        return f"ctx.{match.group(1)}()"
    if reason.startswith("kind "):
        return reason.split(":")[0]
    if "create_client()" in reason:
        return "model-provider builds its own SDK client (create_client)"
    if reason.startswith("dashboard backend"):
        return "dashboard backend API streams (SSE/websocket)"
    return reason[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plugin-host readiness audit over the plugin catalog")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    entries = load_catalog()
    if args.limit:
        entries = entries[: args.limit]
    src_root = args.out / "src"
    src_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda e: _audit(e, src_root), entries))
    (args.out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    verdicts = collections.Counter(r["verdict"] for r in rows)
    audited = len(rows) - verdicts.get("fetch_failed", 0)
    ready = verdicts.get("host", 0) + verdicts.get("portable", 0)
    blockers = collections.Counter()
    for r in rows:
        if r["verdict"] == "in_process":
            for cls in {_reason_class(x) for x in r["reasons"]}:
                blockers[cls] += 1
    print(f"**{ready}/{audited} catalog plugins run in the plugin host unchanged** "
          f"({100 * ready / max(audited, 1):.1f}%; {len(rows)} entries, {verdicts.get('fetch_failed', 0)} unfetchable)\n")
    print("| verdict | count |\n|---|---|")
    for verdict, count in verdicts.most_common():
        print(f"| {verdict} | {count} |")
    print("\n| needs in-process because | plugins |\n|---|---|")
    for cls, count in blockers.most_common():
        print(f"| {cls} | {count} |")
    print("\nin-process entries:")
    for r in sorted((r for r in rows if r["verdict"] == "in_process"), key=lambda r: r["name"]):
        print(f"- {r['name']}: " + "; ".join(sorted({_reason_class(x) for x in r['reasons']})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
