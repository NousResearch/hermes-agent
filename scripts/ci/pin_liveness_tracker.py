"""Maintain one upstream pin incident from a complete validated census.

Retired URLs on a recoverable pin are informational. Unknown recovery holds an
existing incident; malformed/partial input cannot mutate GitHub. Dry-run is
fully offline and reports the plan without authentication or issue discovery.
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
    """Only an unrecoverable pin opens an incident; recovery must be observed."""
    from scripts.ci.lock_liveness import ALIVE, DEAD, UNKNOWN, validate_report

    pins = validate_report(report)
    broken = [pin for pin in pins if pin["status"] == DEAD]
    uncertain = [pin for pin in pins if pin["status"] == UNKNOWN]
    if not broken:
        if not open_issue or uncertain:
            # Preserve yesterday's proof until every pin has an observed live source.
            return {"action": "none"}
        return {"action": "close", "body": f"Recovery observed: every one of {len(pins)} pins has a live source in its download ladder. "
                "Retired individual URLs remain informational; HEAD availability does not certify byte integrity."}

    lines = [f"**{len(broken)} unrecoverable pin(s); {len(uncertain)} inconclusive pin(s) of {len(pins)}.**", "",
             "A pin is unrecoverable only when every source in its primary → content-addressed mirror → historical ladder "
             "answers 404/410. A live fallback preserves availability without a repin. Unknown observations cannot prove "
             "recovery or justify closing an existing incident. HEAD checks establish availability; downloads still verify SHA256.",
             "", "### Unrecoverable pins", "", "| scope | pin | digest | source | evidence |", "|---|---|---|---|---|"]
    for pin in broken[:MAX_ROWS]:
        for source in pin["sources"]:
            detail = str(source.get("detail", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {pin['scope']} | `{pin['name']}` | `{pin['sha256'][:12]}` | {source['role']}: {source['url']} | {detail} |")
    if len(broken) > MAX_ROWS:
        lines.append(f"| … | {len(broken) - MAX_ROWS} more | see census artifact | | |")
    lines += ["", "### Repair", "",
              "Restore a hash-identical archived source first. Repin only when the complete ladder is unrecoverable: "
              "`python -m pm update --termux` verifies replacements for Termux pool pins; other package owners resolve their reviewed pins. "
              "Seed missing content-addressed objects with `archive-inputs.yml` before relying on a new digest.", "",
              "Individual retired URLs and missing mirrors on recoverable pins are redundancy observations, not installation failures."]
    if uncertain:
        lines += ["", "### Inconclusive pins", ""]
        for pin in uncertain[:MAX_ROWS]:
            lines.append(f"- {pin['scope']} `{pin['name']}` (`{pin['sha256'][:12]}`): no observed live source; at least one probe was unknown")
    return {"action": "update" if open_issue else "open",
            "title": f"Pin liveness: {len(broken)} unrecoverable pin{'s' if len(broken) != 1 else ''}",
            "body": "\n".join(lines)}


def _gh(args: list[str]) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _open_issue(repo: str) -> dict | None:
    # The issues endpoint includes PRs. A page full of labeled PRs must not
    # hide an older incident and cause this job to open a duplicate.
    page = 1
    while True:
        issues = json.loads(_gh(["api", f"repos/{repo}/issues?labels={LABEL}&state=open&per_page=100&page={page}"]))
        if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
            raise ValueError("Invalid GitHub issue inventory")
        found = next((issue for issue in issues if "pull_request" not in issue), None)
        if found is not None:
            return found
        if len(issues) < 100:
            return None
        page += 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="census JSON from scripts.ci.lock_liveness")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2], help="checkout whose full inventory the report must cover")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo != "NousResearch/hermes-agent" and not args.dry_run:
        print("Tracker writes require GITHUB_REPOSITORY=NousResearch/hermes-agent", file=sys.stderr)
        return 2

    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    from scripts.ci.lock_liveness import pinned_inputs, validate_report
    validate_report(report, expected_inventory=pinned_inputs(args.repo.resolve()))
    if args.dry_run:
        change = plan(report, None)
        print(json.dumps(change, indent=2))
        return 0

    open_issue = _open_issue(repo)
    change = plan(report, {"number": open_issue["number"]} if open_issue else None)
    print(f"census: {report.get('alive')} alive / {report.get('dead')} dead / {report.get('unknown')} unknown "
          f"of {report.get('total')} -> {change['action']}"
          + (f" (#{open_issue['number']})" if open_issue else ""))

    if change["action"] == "none":
        if change.get("body"):
            print(f"\n--- {change.get('title', 'comment')} ---\n{change['body']}")
        return 0

    if change["action"] == "open":
        _gh(["label", "create", LABEL, "--repo", repo, "--force", "--color", "B60205",
             "--description", "Unrecoverable pinned artifacts (managed by lock-liveness.yml)"])
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
