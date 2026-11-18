"""Run repository UX triage, release gating, or an auditable data refresh."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.ux_blockers.github import GitHub
from scripts.ux_blockers.service import ensure_labels, gate_report, sync_issue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("triage", "gate", "report"))
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--issue", type=int)
    parser.add_argument("--candidate")
    parser.add_argument("--event", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Audited historical sample to refresh (report only)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = GitHub(args.repository or "")
    if args.action == "gate":
        result = gate_report(api, args.candidate or "")
    elif args.action == "report":
        if args.snapshot:
            from scripts.ux_blockers.audit import refresh
            result = refresh(api, args.snapshot)
        else:
            result = gate_report(api, args.candidate or "")
    else:
        ensure_labels(api)
        numbers = set()
        if args.issue:
            numbers.add(args.issue)
        elif args.event:
            event = json.loads(args.event.read_text(encoding="utf-8"))
            issue = event.get("issue")
            if issue and "pull_request" not in issue:
                numbers.add(issue["number"])
            elif not issue:
                # Refresh known UX reports when fixing PRs change; relationships
                # are read from GitHub, never inferred from PR prose.
                for label in ("impact%3Aux-release-blocker", "impact%3Aux-friction", "ux%3Aconfirmed"):
                    numbers.update(row["number"] for row in api.pages(
                        f"{api.prefix}/issues?state=all&labels={label}&per_page=100")
                                   if "pull_request" not in row)
        result = {"issues": [sync_issue(api, number) for number in sorted(numbers)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write("## Hermes UX blockers\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
    if args.action == "gate" and result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
