#!/usr/bin/env python3
"""
Deterministic GitHub PR -> release watcher example.

Usage:
  python github_pr_release_watcher.py /path/to/config.json

Behavior:
- Prints nothing if not ready.
- Prints a Telegram-ready success message if ready.
- Exits non-zero only for hard configuration/check failures.

Config keys:
  repo: "owner/repo"
  pr: 123
  allow_prerelease: false
  require_release_after_merge: true
  success_action: "Install/update ..."
  cron_job_id: "optional"
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def github_get(path: str) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Hermes release-feature-watcher",
    }
    req = urllib.request.Request(f"https://api.github.com{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def latest_eligible_release(repo: str, allow_prerelease: bool) -> dict[str, Any] | None:
    releases = github_get(f"/repos/{repo}/releases?per_page=20")
    for rel in releases:
        if rel.get("draft"):
            continue
        if rel.get("prerelease") and not allow_prerelease:
            continue
        return rel
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: github_pr_release_watcher.py CONFIG.json", file=sys.stderr)
        return 2

    cfg = json.loads(Path(sys.argv[1]).read_text())
    repo = cfg["repo"]
    pr_number = int(cfg["pr"])
    allow_prerelease = bool(cfg.get("allow_prerelease", False))
    require_after_merge = bool(cfg.get("require_release_after_merge", True))

    pr = github_get(f"/repos/{repo}/pulls/{pr_number}")
    merged_at = parse_time(pr.get("merged_at"))
    if merged_at is None:
        return 0

    rel = latest_eligible_release(repo, allow_prerelease)
    if not rel:
        return 0

    published_at = parse_time(rel.get("published_at"))
    if require_after_merge and (published_at is None or published_at <= merged_at):
        return 0

    tag = rel.get("tag_name") or rel.get("name") or "latest release"
    release_url = rel.get("html_url")
    pr_url = pr.get("html_url")
    action = cfg.get("success_action", "Install or update from the latest release.")
    job_id = cfg.get("cron_job_id")

    print(f"Ready: {cfg.get('name', repo + ' release watcher')} appears shipped.\n")
    print("Evidence:")
    print(f"- PR #{pr_number} merged at {merged_at.isoformat()}")
    print(f"- Release {tag} published at {published_at.isoformat() if published_at else 'unknown'}")
    print(f"- PR: {pr_url}")
    print(f"- Release: {release_url}\n")
    print(f"Next step: {action}")
    if job_id:
        print(f"Watcher job: {job_id}. Remove/self-remove this watcher after delivery.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as e:
        print(f"GitHub API error {e.code}: {e.reason}", file=sys.stderr)
        raise SystemExit(1)
