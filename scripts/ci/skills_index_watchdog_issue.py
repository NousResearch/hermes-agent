#!/usr/bin/env python3
"""Open or update the watchdog issue for a failed probe in skills-index-freshness.yml.

Each watchdog owns a title prefix. When an open issue already carries that prefix, the script
appends a comment instead of opening a new issue each tick, so a stuck probe does not spam the
tracker. The prefixes differ so a plugin-catalog CORS report never lands on a skills-index issue.

Reads ``STATUS`` and ``DETAIL`` (probe outputs), ``GITHUB_REPOSITORY``, and leaves ``GH_TOKEN`` in
the environment for ``gh``; the token never appears in argv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

GH_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class Watchdog:
    title_prefix: str
    title: str
    body: str


WATCHDOGS = {
    "skills-index": Watchdog(
        title_prefix="[skills-index-watchdog]",
        title="Skills index is stale or degraded ({status})",
        body="""Automated freshness probe failed.

**Status:** `{status}`
**Detail:** {detail}

The Skills Hub at /docs/skills depends on `/docs/api/skills-index.json`.
The unified index is rebuilt by `.github/workflows/skills-index.yml` (cron 6/18 UTC)
and `.github/workflows/deploy-site.yml` (on every push affecting website/skills).
If this issue keeps reopening, check the latest runs:

- https://github.com/{repository}/actions/workflows/skills-index.yml
- https://github.com/{repository}/actions/workflows/deploy-site.yml

This issue was opened by `.github/workflows/skills-index-freshness.yml`. Close it once the underlying problem is fixed; the next probe will reopen if it's still broken.""",
    ),
    "plugin-catalog-cors": Watchdog(
        title_prefix="[plugin-catalog-cors-watchdog]",
        title="Desktop cannot read the plugin catalog ({status})",
        body="""Hermes Desktop cannot read the plugin catalog.

**Status:** `{status}`
**Detail:** {detail}

Desktop's `hermes://plugin/install?catalog=` links fetch `/docs/api/plugins.json` cross-origin.
A browser rejects that fetch when any response in the redirect chain, including the docs host's
redirect to GitHub Pages, lacks `Access-Control-Allow-Origin`. Every catalog install link then
shows "Could not load the Hermes plugin catalog". Background: #136483.

This issue was opened by `.github/workflows/skills-index-freshness.yml` (job `check-catalog-cors`). Close it once the header is back; the next probe will reopen it if it's still broken.""",
    ),
}


def find_open_issue(issues: list[dict], title_prefix: str) -> int | None:
    """First listed issue whose title starts with the prefix (GitHub search also matches mid-title)."""
    for issue in issues:
        if issue["title"].startswith(title_prefix):
            return issue["number"]
    return None


def plan_commands(
    watchdog: Watchdog,
    existing: int | None,
    *,
    repository: str,
    status: str,
    detail: str,
    now: datetime,
) -> tuple[str, list[str]]:
    """Return the log line and the ``gh`` argv that reports this failure."""
    if existing is not None:
        stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"Appending to existing issue #{existing}", [
            "gh", "issue", "comment", str(existing), "--repo", repository,
            "--body", f"Probe still failing at {stamp}: `{status}` — {detail}",
        ]
    fields = {"repository": repository, "status": status, "detail": detail}
    return "Opening new watchdog issue", [
        "gh", "issue", "create", "--repo", repository,
        "--title", f"{watchdog.title_prefix} {watchdog.title.format(**fields)}",
        "--body", watchdog.body.format(**fields),
    ]


def list_open_issues(repository: str, title_prefix: str) -> list[dict]:
    """Open issues matching the prefix; a failed listing counts as none, so a new issue is filed."""
    listed = subprocess.run(
        [
            "gh", "issue", "list", "--repo", repository, "--state", "open",
            "--search", f'in:title "{title_prefix}"', "--json", "number,title",
        ],
        check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, timeout=GH_TIMEOUT_SECONDS,
    )
    if listed.returncode != 0:
        return []
    return json.loads(listed.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open or update a probe watchdog issue.")
    parser.add_argument("watchdog", choices=sorted(WATCHDOGS))
    args = parser.parse_args(argv)
    watchdog = WATCHDOGS[args.watchdog]
    repository = os.environ["GITHUB_REPOSITORY"]

    existing = find_open_issue(list_open_issues(repository, watchdog.title_prefix), watchdog.title_prefix)
    message, command = plan_commands(
        watchdog,
        existing,
        repository=repository,
        status=os.environ.get("STATUS", ""),
        detail=os.environ.get("DETAIL", ""),
        now=datetime.now(UTC),
    )
    print(message, flush=True)
    return subprocess.run(command, check=False, stdin=subprocess.DEVNULL, timeout=GH_TIMEOUT_SECONDS).returncode


if __name__ == "__main__":
    sys.exit(main())
