#!/usr/bin/env python3
"""
GitRadar Upstream Monitor - Weekly check on seeded repos.
Reads repo pages from the shared GitRadar runbook root
(<repo>/runbooks/github-radar/repos/), queries GitHub API for upstream
status, reports: new releases, dormancy, star growth.
Falls back to ~/wiki/repos if the shared runbook dir doesn't exist yet.
Output: JSON to stdout + optional kanban task creation.

C030: aligned to the same runbook root as github-radar-discover.py
(producer) so both share a single source of truth.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone, timedelta

# ── Config ──────────────────────────────────────────────────────────
# Shared runbook root (C030): aligned with github-radar-discover.py
# producer, which writes to <repo>/runbooks/github-radar/. The consumer
# reads seeded repo pages from the same runbook tree so producer and
# consumer share a single root instead of diverging across ~/wiki and
# <repo>/runbooks.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUNBOOK_ROOT = os.path.join(_REPO_ROOT, "runbooks", "github-radar")
REPOS_DIR = os.path.join(_RUNBOOK_ROOT, "repos")
META_INDEX = os.path.join(_RUNBOOK_ROOT, "repo-index.md")
LOG_FILE = os.path.join(_RUNBOOK_ROOT, "log.md")

# Backward-compat: if the shared runbook dirs don't exist yet, fall back
# to the legacy ~/wiki paths so the monitor still works during migration.
if not os.path.isdir(REPOS_DIR):
    _legacy_repos = os.path.expanduser("~/wiki/repos")
    if os.path.isdir(_legacy_repos):
        REPOS_DIR = _legacy_repos
        META_INDEX = os.path.expanduser("~/wiki/_meta/repo-index.md")
        LOG_FILE = os.path.expanduser("~/wiki/log.md")
DORMANCY_THRESHOLD_DAYS = 90
STAR_GROWTH_THRESHOLD_PCT = 50  # alert if stars grew by >50% since last check

# ── Helpers ─────────────────────────────────────────────────────────

def gh_auth_token():
    result = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        print(f"WARN: gh auth failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    return result.stdout.strip()


def parse_repo_page(path):
    """Parse a wiki repo page, return frontmatter as dict."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Extract YAML frontmatter
    m = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not m:
        return None

    frontmatter = {}
    for line in m.group(1).split('\n'):
        if ':' in line:
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            frontmatter[key] = val
    return frontmatter


def update_repo_page(path, updates):
    """Update specific frontmatter fields in a repo page."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    m = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not m:
        return False

    old_fm = m.group(1)
    new_fm = old_fm
    for key, value in updates.items():
        pattern = rf'^{key}:.*$'
        replacement = f'{key}: {value}'
        if re.search(pattern, new_fm, re.MULTILINE):
            new_fm = re.sub(pattern, replacement, new_fm, count=1, flags=re.MULTILINE)
        else:
            new_fm += f'\n{key}: {value}'

    new_content = content.replace(old_fm, new_fm)
    with open(path, 'w', encoding="utf-8") as f:
        f.write(new_content)
    return True


def query_github(repo_full_name, token):
    """Query GitHub API for a repo. Returns dict or None."""
    if not token:
        return None

    url = f"https://api.github.com/repos/{repo_full_name}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "KENSEI-GitRadar-Upstream/1.0")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return {
                "exists": True,
                "full_name": data.get("full_name", repo_full_name),
                "stars": data.get("stargazers_count", 0),
                "language": data.get("language", ""),
                "pushed_at": data.get("pushed_at", ""),
                "open_issues": data.get("open_issues_count", 0),
                "description": (data.get("description") or "")[:100],
            }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"exists": False, "full_name": repo_full_name}
        print(f"WARN: GitHub API error {e.code}: {e.reason}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"WARN: GitHub API exception: {e}", file=sys.stderr)
        return None


def check_latest_release(repo_full_name, token):
    """Check latest release tag. Returns dict or None."""
    if not token:
        return None

    url = f"https://api.github.com/repos/{repo_full_name}/releases/latest"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "KENSEI-GitRadar-Upstream/1.0")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return {
                "tag_name": data.get("tag_name", ""),
                "published_at": data.get("published_at", ""),
                "name": data.get("name", ""),
            }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # No releases
        return None
    except Exception:
        return None


# ── Main ────────────────────────────────────────────────────────────

def main():
    findings = {
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repos_checked": 0,
        "new_releases": [],
        "dormant": [],
        "star_growth": [],
        "vanished": [],
        "up_to_date": 0,
    }

    if not os.path.isdir(REPOS_DIR):
        print(json.dumps({"error": f"REPOS_DIR not found: {REPOS_DIR}"}))
        return

    token = gh_auth_token()

    for filename in sorted(os.listdir(REPOS_DIR)):
        if not filename.endswith(".md"):
            continue

        path = os.path.join(REPOS_DIR, filename)
        fm = parse_repo_page(path)
        if not fm:
            continue

        repo_name = fm.get("repo_full_name", "")
        if not repo_name or repo_name == "unknown/unknown":
            continue

        # Skip archived repos
        if fm.get("adoption_status") == "archived":
            continue

        findings["repos_checked"] += 1

        # Query GitHub API
        gh_data = query_github(repo_name, token)
        if gh_data is None:
            continue  # API error, skip

        if not gh_data.get("exists"):
            findings["vanished"].append({
                "repo": repo_name,
                "note": "Repo no longer exists on GitHub (404)"
            })
            continue

        # Parse dates
        pushed_str = gh_data.get("pushed_at", "")
        today = datetime.now(timezone.utc)

        # Check dormancy
        if pushed_str:
            pushed = datetime.fromisoformat(pushed_str.replace("Z", "+00:00"))
            days_since_push = (today - pushed).days
            if days_since_push > DORMANCY_THRESHOLD_DAYS:
                findings["dormant"].append({
                    "repo": repo_name,
                    "days_since_push": days_since_push,
                    "stars": gh_data["stars"],
                })

        # Check star growth
        prev_stars_str = fm.get("stars", "0")
        try:
            prev_stars = int(prev_stars_str)
        except ValueError:
            prev_stars = 0
        current_stars = gh_data.get("stars", 0)
        if prev_stars > 0 and current_stars > prev_stars:
            growth_pct = ((current_stars - prev_stars) / prev_stars) * 100
            if growth_pct > STAR_GROWTH_THRESHOLD_PCT:
                findings["star_growth"].append({
                    "repo": repo_name,
                    "previous_stars": prev_stars,
                    "current_stars": current_stars,
                    "growth_pct": round(growth_pct, 1),
                })

        # Check latest release
        release = check_latest_release(repo_name, token)
        prev_version = fm.get("upstream_latest_version", "")
        if release:
            new_tag = release.get("tag_name", "")
            if new_tag and new_tag != prev_version:
                findings["new_releases"].append({
                    "repo": repo_name,
                    "previous_version": prev_version,
                    "new_version": new_tag,
                    "released_at": release.get("published_at", ""),
                })

        # Update repo page with latest data
        today_str = today.strftime("%Y-%m-%d")
        updates = {
            "upstream_last_checked": today_str,
        }
        if current_stars > 0:
            updates["stars"] = str(current_stars)
        if release and release.get("tag_name"):
            updates["upstream_latest_version"] = release["tag_name"]

        update_repo_page(path, updates)

    # Categorise repos with no issues
    total_with_findings = (
        len(findings["new_releases"])
        + len(findings["dormant"])
        + len(findings["star_growth"])
        + len(findings["vanished"])
    )
    findings["up_to_date"] = findings["repos_checked"] - total_with_findings

    print(json.dumps(findings, indent=2))
    return findings


if __name__ == "__main__":
    main()
