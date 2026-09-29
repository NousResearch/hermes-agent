#!/usr/bin/env python3
"""What changed in upstream Hermes that Agent Czesiek has not looked at yet?

Agent Czesiek is a fork whose git history is unrelated to upstream's, so there is no merge base to
diff against. ``docs/upstream/WATCH.json`` records where the last review stopped instead: an
upstream commit SHA once one has been marked, the snapshot date until then.

The report lists upstream commits since that point, grouped by area, and flags the files that BOTH
sides changed — those are where a merge or cherry-pick will conflict, so they are what to read first.
Read-only: it never changes the repository. Output is Markdown (stdout, or ``--out``).

    python scripts/upstream_watch.py                      # report
    python scripts/upstream_watch.py --mark-reviewed SHA  # record that review reached SHA
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "docs" / "upstream" / "WATCH.json"
API = "https://api.github.com"
MAX_COMMITS = 200
MAX_DETAIL_CALLS = 80  # per-commit file lookups, only needed when comparing by date

# Areas in the order a maintainer should care about them; anything else lands in "other".
AREAS = ("run_agent.py", "agent", "tools", "gateway", "hermes_cli", "tui_gateway", "plugins", "skills",
         "cron", "apps", "web", "tests", "docs", "website")


def area_of(path: str) -> str:
    head = path.split("/", 1)[0]
    return head if head in AREAS else "other"


def group_by_area(commits: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each commit goes under every area its files touch (a commit with no file list under 'other')."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for commit in commits:
        areas = sorted({area_of(f) for f in commit.get("files", [])}) or ["other"]
        for area in areas:
            groups[area].append(commit)
    return dict(sorted(groups.items(), key=lambda kv: (AREAS.index(kv[0]) if kv[0] in AREAS else len(AREAS), kv[0])))


def conflict_files(upstream_files: Iterable[str], fork_files: Iterable[str]) -> list[str]:
    """Files changed on both sides since the review point — the likely merge conflicts."""
    return sorted(set(upstream_files) & set(fork_files))


def render_report(*, config: dict[str, Any], commits: list[dict[str, Any]], fork_files: set[str],
                  truncated: bool, head_sha: Optional[str]) -> str:
    upstream = config["upstream"]
    point = (f"commit `{config['reviewed_sha'][:10]}`" if config.get("reviewed_sha")
             else f"{config.get('since', '?')} (snapshot date)")
    lines = [f"# Upstream {upstream}: {len(commits)} change(s) since {point}", ""]
    if not commits:
        return "\n".join(lines + ["Nothing new to review."])

    upstream_files = {f for c in commits for f in c.get("files", [])}
    both = conflict_files(upstream_files, fork_files)
    if truncated:
        lines += [f"> Only the newest {MAX_COMMITS} commits are listed; review in smaller steps.", ""]
    if both:
        lines += [f"## Changed on both sides ({len(both)}) — read these first", ""]
        lines += [f"- `{path}`" for path in both[:60]]
        lines += ([f"- … and {len(both) - 60} more"] if len(both) > 60 else []) + [""]

    for area, group in group_by_area(commits).items():
        lines += [f"## {area} ({len(group)})", ""]
        for c in group:
            marker = " ⚠️" if set(c.get("files", [])) & set(both) else ""
            lines.append(f"- [`{c['sha'][:10]}`]({c['url']}) {c['title']}{marker}")
        lines.append("")

    if head_sha:
        lines += ["---", f"When done reviewing: `python scripts/upstream_watch.py --mark-reviewed {head_sha}`"]
    return "\n".join(lines)


# ── GitHub access ────────────────────────────────────────────────────────────────────────────────

def _get(url: str, token: Optional[str]) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "agent-czesiek-upstream-watch"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _summarise(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha": raw["sha"],
        "title": (raw.get("commit", {}).get("message") or "").split("\n", 1)[0],
        "url": raw.get("html_url", ""),
        "files": [f["filename"] for f in raw.get("files", [])],
    }


def fetch_commits(config: dict[str, Any], token: Optional[str],
                  get: Callable[[str, Optional[str]], Any] = _get) -> tuple[list[dict[str, Any]], bool, Optional[str]]:
    """(oldest-first commits with file lists, truncated?, upstream main tip SHA)."""
    repo = config["upstream"]
    tip = get(f"{API}/repos/{repo}/commits/main", token)["sha"]
    if config.get("reviewed_sha"):
        cmp = get(f"{API}/repos/{repo}/compare/{config['reviewed_sha']}...main", token)
        commits = [_summarise(c) for c in cmp.get("commits", [])]
        # compare returns the aggregate file list, not per commit: attribute it to a single synthetic
        # entry only when commits carry none, so the conflict list still works.
        if commits and not any(c["files"] for c in commits):
            commits[-1]["files"] = [f["filename"] for f in cmp.get("files", [])]
        return commits[-MAX_COMMITS:], len(commits) > MAX_COMMITS, tip

    since = urllib.parse.quote(config.get("since", "2026-09-26T00:00:00Z"))
    listed: list[dict[str, Any]] = []
    for page in (1, 2):
        chunk = get(f"{API}/repos/{repo}/commits?sha=main&since={since}&per_page=100&page={page}", token)
        listed += chunk
        if len(chunk) < 100:
            break
    listed = listed[:MAX_COMMITS]
    commits = []
    for index, raw in enumerate(reversed(listed)):
        entry = _summarise(raw)
        if index < MAX_DETAIL_CALLS:
            entry = _summarise(get(f"{API}/repos/{repo}/commits/{raw['sha']}", token))
        commits.append(entry)
    return commits, len(listed) >= MAX_COMMITS, tip


def fork_changed_files(since: str) -> set[str]:
    out = subprocess.run(["git", "log", f"--since={since}", "--name-only", "--format=", "HEAD"],
                         cwd=ROOT, capture_output=True, text=True, check=False).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def mark_reviewed(sha: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["reviewed_sha"] = sha
    CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="write the Markdown report here instead of stdout")
    parser.add_argument("--mark-reviewed", metavar="SHA", help="record that upstream was reviewed up to SHA")
    args = parser.parse_args(argv)

    if args.mark_reviewed:
        mark_reviewed(args.mark_reviewed)
        print(f"Recorded reviewed_sha={args.mark_reviewed} in {CONFIG.relative_to(ROOT)}")
        return 0

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    try:
        commits, truncated, tip = fetch_commits(config, token)
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"Could not read upstream: {exc}", file=sys.stderr)
        return 1
    report = render_report(config=config, commits=commits, fork_files=fork_changed_files(config.get("since", "")),
                           truncated=truncated, head_sha=tip)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
