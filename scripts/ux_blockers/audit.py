"""Refresh lifecycle data for an explicitly curated audit without reclassifying reports."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.ux_blockers.github import GitHub


def refresh(api: GitHub, snapshot: Path) -> dict:
    sample = json.loads(snapshot.read_text(encoding="utf-8"))
    if sample["repository"] != api.repository or sample["schema_version"] != 1:
        raise ValueError("Audit snapshot must match this repository and schema")
    issue_cache, fix_cache = {}, {}
    clusters = []
    for cluster in sample["clusters"]:
        issues, fixes = [], []
        for prior in cluster["issues"]:
            number = prior["number"]
            if number not in issue_cache:
                current = api.issue(number)
                issue_cache[number] = {"number": number, "url": current["html_url"],
                                       "state": current["state"]}
            issues.append(issue_cache[number])
        for prior in cluster["audited_fix_pull_requests"]:
            number = prior["number"]
            if number not in fix_cache:
                current = api.get(f"{api.prefix}/pulls/{number}")
                fix_cache[number] = {"number": number, "url": current["html_url"],
                                     "state": "merged" if current["merged_at"] else current["state"],
                                     "head_sha": current["head"]["sha"],
                                     "merge_commit_sha": current["merge_commit_sha"]}
            fixes.append(fix_cache[number])
        clusters.append({"id": cluster["id"], "historical_disposition": cluster["historical_disposition"],
                         "issues": issues, "audited_fixes": fixes,
                         "current_release_blocker_assessment": "not_assessed"})
    return {"schema_version": 1, "captured_at_utc": datetime.now(UTC).isoformat(),
            "repository": api.repository, "scope": "Lifecycle refresh of the curated historical sample",
            "historical_counts": sample["counts"], "clusters": clusters,
            "current_counts": {
                "open_issue_records": sum(row["state"] == "open" for row in issue_cache.values()),
                "historically_open_clusters_still_all_open": sum(
                    row["historical_disposition"] in ("open_report_evidenced", "pre_window_carry_in")
                    and all(issue["state"] == "open" for issue in row["issues"]) for row in clusters),
                "merged_audited_fix_prs": sum(row["state"] == "merged" for row in fix_cache.values()),
            },
            "not_proven": "New classifications, a complete census, current RC failure or native recovery."}
