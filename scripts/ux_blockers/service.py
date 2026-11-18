"""Triage proposals, linked fixes and release decisions using GitHub as the ledger."""
from __future__ import annotations

import hashlib
import json
import re

from scripts.ux_blockers.github import GitHub
from scripts.ux_blockers.policy import (
    BLOCKER, BOT, CONFIRMED, FRICTION, LABELS, MARKER, RECEIPT, REVIEW,
    classify, decision, latest_commands, sections,
)


def ensure_labels(api: GitHub) -> None:
    import subprocess
    from urllib.parse import quote

    present = {row["name"] for row in api.pages(f"{api.prefix}/labels?per_page=100")}
    for name, (color, description) in LABELS.items():
        if name not in present:
            try:
                api.write(f"{api.prefix}/labels", "POST", name=name, color=color, description=description)
            except subprocess.CalledProcessError:
                # Another issue's first run may have created the label in parallel.
                if api.get(f"{api.prefix}/labels/{quote(name, safe='')}").get("name") != name:
                    raise


def sync_issue(api: GitHub, number: int) -> dict:
    issue = api.issue(number)
    comments = api.comments(number)
    receipts = [row for row in api.receipts() if row["issue"] == number]
    commands = latest_commands(comments, api.trusted)
    labels = {row["name"] for row in issue["labels"]}
    # Confirmation is indexed before any optional API read or comment write.
    # Closing or editing a report never removes this index.
    if (receipts or any(command.action == "confirm" for _, command in commands)) and CONFIRMED not in labels:
        api.write(f"{api.prefix}/issues/{number}/labels", "POST", labels=[CONFIRMED])
        labels.add(CONFIRMED)
    for comment, command in commands:
        if command.action == "confirm":
            receipt = RECEIPT + "\n" + json.dumps({"issue": number, "id": comment["id"], "sha256":
                hashlib.sha256(comment["body"].encode()).hexdigest()}, sort_keys=True)
            if not any(row["id"] == comment["id"] for row in receipts):
                api.write(f"{api.prefix}/issues/{api.ledger_issue}/comments", "POST", body=receipt)
                # Other issues in this run must see newly acknowledged records.
                api._receipts = None
            if not any(row["user"].get("login") == BOT and row.get("body") == receipt for row in comments):
                api.write(f"{api.prefix}/issues/{number}/comments", "POST", body=receipt)
    assessment_body = issue.get("body") or ""
    for comment in comments:
        if (comment.get("body") or "").startswith("/ux assess\n") and (
                comment["user"].get("type") == "User"
                and comment["updated_at"] == comment["created_at"]
                and api.trusted(comment["user"]["login"])):
            assessment_body = comment["body"]
    assessment = classify(assessment_body)
    if ("UX surface" not in sections(assessment_body) and CONFIRMED not in labels
            and not ({BLOCKER, FRICTION, REVIEW} & labels)):
        return {"issue": number, "skipped": "No UX assessment requested"}
    wanted = {REVIEW}
    if assessment.priority in ("P0", "P1"):
        wanted.add(BLOCKER if assessment.priority == "P0" else FRICTION)
    # Confirmed issues retain their P0 signal even if reporter input changes.
    if CONFIRMED in labels:
        wanted = {BLOCKER}
    for name in sorted(wanted - labels):
        api.write(f"{api.prefix}/issues/{number}/labels", "POST", labels=[name])
    for name in sorted(({BLOCKER, FRICTION, REVIEW} & labels) - wanted):
        from urllib.parse import quote
        api.write(f"{api.prefix}/issues/{number}/labels/{quote(name, safe='')}", "DELETE")
    fixes = api.links(number)
    links = "\n".join(f"- {row['url']} ({row['state'].lower()})" for row in fixes) or "No GitHub closing PR is linked yet."
    proposed = f"proposed UX {assessment.priority}" if assessment.priority != "unknown" else "UX evidence needed"
    status = "Maintainer confirmation is indexed." if CONFIRMED in labels else "Awaiting maintainer review."
    body = (f"{MARKER}\n## Hermes UX assessment\n\n{proposed}. {assessment.reason}\n\n"
            f"{status} Engineering priority is assessed independently.\n\n"
            f"### Linked fixing PRs\n\n{links}\n\n"
            "A merged fix is evaluated against the release candidate and the reported recovery.\n\n"
            "Maintainers can use `/ux confirm <40-character candidate SHA> <evidence URL> <reason>`. "
            "Candidate dispositions use `/ux resolve fixed|recovered|unaffected|exception <SHA> <evidence URL> <reason and recovery>`. "
            "For related confirmed reports, use `/ux duplicate #<canonical issue>`. "
            "Issue authors can fill the UX fields. Maintainers can assess older reports with `/ux assess` followed by the same fields.")
    own = [row for row in comments if row["user"]["login"] == BOT
           and row["user"].get("type") == "Bot"
           and (row.get("body") or "").startswith(MARKER)]
    if own:
        if own[0]["body"].strip() != body.strip():
            api.write(f"{api.prefix}/issues/comments/{own[0]['id']}", "PATCH", body=body)
    else:
        api.write(f"{api.prefix}/issues/{number}/comments", "POST", body=body)
    return {"issue": number, "priority": assessment.priority, "confirmed": CONFIRMED in labels, "fixes": fixes}


def gate_report(api: GitHub, candidate: str) -> dict:
    if re.fullmatch(r"[a-f0-9]{40}", candidate) is None:
        raise ValueError("UX gate requires the admitted 40-character candidate SHA")
    indexed = api.indexed()
    decisions = {}
    for issue in indexed:
        if "pull_request" in issue:
            raise ValueError("UX confirmation index contains a pull request")
        number = issue["number"]
        decisions[number] = {"issue": number, "url": issue["html_url"], "state": issue["state"],
                             **decision([*api.comments(number), *issue.get("_ux_receipts", [])], candidate, api.trusted),
                             "fixes": api.links(number)}
        row = decisions[number]
        if row.get("disposition") == "fixed":
            contained = [fix["number"] for fix in row["fixes"]
                         if fix["state"] == "MERGED" and fix.get("mergeCommit")
                         and api.contains(candidate, fix["mergeCommit"]["oid"])]
            row["contained_fixes"] = contained
            if not contained:
                row.update(status="blocked", reason="No linked merged fix is contained in this candidate.")
    # A duplicate clears only through a confirmed canonical issue's disposition.
    results = []
    for number, original in decisions.items():
        row = original.copy()
        visited = {number}
        resolved = row
        while resolved["status"] == "duplicate":
            canonical = resolved["canonical"]
            if canonical in visited or canonical not in decisions:
                resolved = {"status": "blocked", "reason": "Canonical issue is missing, unconfirmed or cyclic."}
                break
            visited.add(canonical)
            resolved = decisions[canonical]
        if row["status"] == "duplicate":
            row["status"] = resolved["status"]
            row["canonical_result"] = resolved.copy()
        results.append(row)
    blocked = [row["issue"] for row in results if row["status"] != "cleared"]
    return {"candidate": candidate, "status": "blocked" if blocked else "passed",
            "blocking_issues": blocked, "issues": results}


def require_candidate(repository: str, candidate: str, *, run=None) -> dict:
    api = GitHub(repository, **({"run": run} if run else {}))
    report = gate_report(api, candidate)
    if report["status"] != "passed":
        numbers = ", ".join(f"#{number}" for number in report["blocking_issues"])
        raise ValueError(f"Stable promotion held by confirmed UX blockers for {candidate}: {numbers}")
    return report
