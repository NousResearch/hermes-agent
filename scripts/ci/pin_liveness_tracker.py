"""Keep ONE tracking issue in step with the scheduled pin-liveness census.

Run by .github/workflows/lock-liveness.yml after the census produces its JSON.
A census with any DEAD pin (a source that answered 404/410 for the exact
object a pin names) opens the issue labelled `pin-liveness`, or rewrites the
body of the one already open, so the issue always names the current dead rows.
A census with no dead rows closes it. Never a second issue and never a comment
per run: subscribers hear about the open and the close, and the body is the
live state in between — the same shape install-e2e-red uses for the E2E
matrix.

Unknown rows (401/403/429/transport failures) are reported in the body but
neither open nor hold the issue: a regional edge denial is not evidence that a
pin has rotated.

    python3 -m scripts.ci.pin_liveness_tracker --report liveness.json [--dry-run]

Needs GH_TOKEN (issues: write) and GITHUB_REPOSITORY.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LABEL = "pin-liveness"
MAX_ROWS = 60


def plan(report: dict, open_issue: dict | None) -> dict:
    """Decide what the tracker does for one census report. Pure.

    ``report`` is the JSON produced by ``scripts.ci.lock_liveness --format
    json``: {alive, dead, unknown, total, rows: [{scope, name, role, url,
    status, detail}]}. ``open_issue`` is {"number": n} or None.

    Two repair paths, kept apart on purpose:

    * a DEAD PRIMARY (role=primary) is a retired pin — the supplier moved on
      and the lock must be re-pinned;
    * a DEAD MIRROR (role=mirror) is an unseeded content-addressed copy —
      `archive-inputs.yml` seeds it on the next main push touching a pin file
      (or via workflow_dispatch when R2 is provisioned).

    Either one opens/holds the issue; neither is a false positive, because
    the fallback ladder only works when both tiers exist.
    """
    rows = report.get("rows") or []
    dead_primary = [row for row in rows if row.get("status") == "dead" and row.get("role") != "mirror"]
    dead_mirror = [row for row in rows if row.get("status") == "dead" and row.get("role") == "mirror"]
    unknown = [row for row in rows if row.get("status") == "unknown"]

    if not dead_primary and not dead_mirror:
        if open_issue:
            return {
                "action": "close",
                "body": (
                    f"Every pinned source and its mirror object answer again: {report.get('alive', 0)} alive, "
                    f"0 dead, {len(unknown)} unknown of {report.get('total', 0)}. Closing; "
                    "the next dead pin reopens a fresh tracker."
                ),
            }
        return {"action": "none"}

    lines = [
        f"**{len(dead_primary)} pin(s) name a retired source; {len(dead_mirror)} mirror object(s) are unseeded** "
        f"({report.get('alive', 0)} alive / {len(dead_primary) + len(dead_mirror)} dead / {len(unknown)} unknown "
        f"of {report.get('total', 0)}).",
        "",
        "The census HEADs every pinned source AND its content-addressed mirror. A `404`/`410` on a primary means "
        "the supplier retired the exact object the pin names; a `404` on a mirror means the archiver has not "
        "seeded it. Both break the fallback ladder the installs rely on.",
        "",
        "This issue is rewritten in place by `.github/workflows/lock-liveness.yml` after every scheduled census "
        "and closed by the first all-alive one.",
        "",
        "### Retired pins (primary dead)",
        "",
        "| scope | pin | url | evidence |",
        "|---|---|---|---|",
    ]
    for row in dead_primary[:MAX_ROWS]:
        lines.append(f"| {row.get('scope', '')} | `{row.get('name', '')}` | {row.get('url', '')} | {row.get('detail', '')} |")
    if not dead_primary:
        lines.append("| — | none | | |")
    if len(dead_primary) > MAX_ROWS:
        lines.append(f"| … | and {len(dead_primary) - MAX_ROWS} more | see the run's artifact | |")

    lines += [
        "",
        "### Unseeded mirrors (mirror dead)",
        "",
        "| scope | pin | url | evidence |",
        "|---|---|---|---|",
    ]
    for row in dead_mirror[:MAX_ROWS]:
        lines.append(f"| {row.get('scope', '')} | `{row.get('name', '')}` | {row.get('url', '')} | {row.get('detail', '')} |")
    if not dead_mirror:
        lines.append("| — | none | | |")
    if len(dead_mirror) > MAX_ROWS:
        lines.append(f"| … | and {len(dead_mirror) - MAX_ROWS} more | see the run's artifact | |")

    lines += [
        "",
        "### Repair",
        "",
        "- Termux-pool rows (`uv@linux-arm64-bionic`, `ffmpeg@linux-arm64-bionic`, and the runtime-lib table "
        "rows): `python -m pm update --termux --check` lists them and `--termux` re-pins them with "
        "download-verified hashes. Re-run the census after the repin.",
        "- Any other retired primary: bump the pin the way the last re-pin did; the lockfile row carries the "
        "resolved URL, so a repair is a reviewed lock change with the new digest.",
        "- Unseeded mirrors: `.github/workflows/archive-inputs.yml` seeds them automatically on the next main "
        "push touching a pin file; a `workflow_dispatch` run seeds them on demand (needs the release-signing "
        "R2 secrets).",
        "",
        "Repairs are commits; the census never rewrites a pin itself.",
    ]
    if unknown:
        lines += ["", "### Unknown (not counted as dead)", ""]
        for row in unknown[:MAX_ROWS]:
            lines.append(f"- {row.get('scope', '')} {row.get('role', '')} `{row.get('name', '')}` — {row.get('detail', '')}")
        if len(unknown) > MAX_ROWS:
            lines.append(f"- …and {len(unknown) - MAX_ROWS} more")
    if dead_primary:
        title = f"Pin liveness: {len(dead_primary)} retired pinned source{'s' if len(dead_primary) != 1 else ''}"
    else:
        title = f"Pin liveness: {len(dead_mirror)} unseeded mirror{'s' if len(dead_mirror) != 1 else ''}"
    return {"action": "update" if open_issue else "open", "title": title, "body": "\n".join(lines)}


def _gh(args: list[str]) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="census JSON from scripts.ci.lock_liveness")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2

    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    issues = json.loads(_gh(["api", f"repos/{repo}/issues?labels={LABEL}&state=open&per_page=5"]))
    open_issue = next((issue for issue in issues if "pull_request" not in issue), None)
    change = plan(report, {"number": open_issue["number"]} if open_issue else None)
    print(f"census: {report.get('alive')} alive / {report.get('dead')} dead / {report.get('unknown')} unknown "
          f"of {report.get('total')} -> {change['action']}"
          + (f" (#{open_issue['number']})" if open_issue else ""))

    if args.dry_run or change["action"] == "none":
        if change.get("body"):
            print(f"\n--- {change.get('title', 'comment')} ---\n{change['body']}")
        return 0

    if change["action"] == "open":
        _gh(["label", "create", LABEL, "--repo", repo, "--force", "--color", "B60205",
             "--description", "Pinned sources that have retired the exact object a pin names (managed by lock-liveness.yml)"])
        url = _gh(["issue", "create", "--repo", repo, "--label", LABEL,
                   "--title", change["title"], "--body", change["body"]])
        print(f"opened {url.strip()}")
    elif change["action"] == "update" and open_issue:
        _gh(["api", "-X", "PATCH", f"repos/{repo}/issues/{open_issue['number']}",
             "-f", f"title={change['title']}", "-f", f"body={change['body']}"])
        print(f"updated #{open_issue['number']} in place")
    elif change["action"] == "close" and open_issue:
        _gh(["issue", "comment", str(open_issue["number"]), "--repo", repo, "--body", change["body"]])
        _gh(["issue", "close", str(open_issue["number"]), "--repo", repo])
        print(f"closed #{open_issue['number']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
