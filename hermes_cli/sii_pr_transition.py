"""Consume a structured SII Developer PASS into one guarded merge and closeout handoff."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Callable


_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ReviewVerdict:
    pr: int
    url: str
    head: str
    check_name: str
    check_job: int


def parse_review_verdict(metadata: dict) -> ReviewVerdict:
    if metadata.get("review_outcome") != "pass":
        raise ValueError("Developer PASS is required")
    blockers = metadata.get("unresolved_blocking_findings")
    if blockers is None or not isinstance(blockers, list) or blockers:
        raise ValueError("Developer PASS must explicitly report zero unresolved blocking findings")
    if metadata.get("check_conclusion") != "success":
        raise ValueError("Developer PASS must cite successful CI")
    head = metadata.get("head")
    if not isinstance(head, str) or not _SHA.fullmatch(head):
        raise ValueError("Developer PASS must cite an exact 40-character head SHA")
    return ReviewVerdict(
        pr=int(metadata["pr"]), url=str(metadata["url"]), head=head,
        check_name=str(metadata["check_name"]), check_job=int(metadata["check_job"]),
    )


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=90, check=True,
    )
    return completed.stdout


def _pr_state(verdict: ReviewVerdict, run: Callable[[list[str]], str]) -> dict:
    return json.loads(run([
        "gh", "pr", "view", str(verdict.pr), "--repo", _repo(verdict.url),
        "--json", "state,headRefOid,mergeable,mergeStateStatus,mergedAt,mergeCommit",
    ]))


def _repo(url: str) -> str:
    match = re.fullmatch(r"https://github\.com/([^/]+/[^/]+)/pull/[1-9][0-9]*", url)
    if not match:
        raise ValueError("Developer PASS must cite an exact GitHub PR URL")
    return match[1]


def _verify_exact_head_ci(verdict: ReviewVerdict, receipt: dict) -> None:
    checks = receipt.get("checks") if isinstance(receipt, dict) else None
    exact = [check for check in checks or [] if (
        check.get("head_sha") == verdict.head
        and check.get("classification") == "success"
        and check.get("id") == verdict.check_job
        and check.get("name") == verdict.check_name
    )]
    if not (receipt.get("ok") is True and receipt.get("classification") == "success"
            and receipt.get("head_sha") == verdict.head and exact):
        raise ValueError("exact-head successful CI evidence is required")


def consume_pass(
    verdict: ReviewVerdict, *, receipt: dict, board: str, developer: str,
    workspace: str, run: Callable[[list[str]], str] = _run,
) -> dict:
    """Merge one reviewed head, verify GitHub state, and enqueue only continuity closeout."""
    _verify_exact_head_ci(verdict, receipt)
    before = _pr_state(verdict, run)
    if before.get("state") != "OPEN" or before.get("headRefOid") != verdict.head:
        raise ValueError("PR is not open at the Developer-reviewed exact head")
    if before.get("mergeable") != "MERGEABLE" or before.get("mergeStateStatus") != "CLEAN":
        raise ValueError("PR is not mergeable and clean")

    run([
        "gh", "pr", "merge", str(verdict.pr), "--repo", _repo(verdict.url),
        "--merge", "--match-head-commit", verdict.head,
    ])
    after = _pr_state(verdict, run)
    merge_sha = ((after.get("mergeCommit") or {}).get("oid"))
    if (after.get("state") != "MERGED" or after.get("headRefOid") != verdict.head
            or not after.get("mergedAt") or not isinstance(merge_sha, str) or not _SHA.fullmatch(merge_sha)):
        raise RuntimeError("GitHub merge read-back did not verify merged state, reviewed head, and merge SHA")

    body = (
        f"Bounded continuity closeout only for PR #{verdict.pr} ({verdict.url}). GitHub read-back verifies "
        f"MERGED from reviewed exact head {verdict.head} as merge commit {merge_sha}. Required check "
        f"{verdict.check_name} succeeded on that exact head as check/job {verdict.check_job}. Append PR_MERGED "
        f"with pr:{verdict.pr} commit:{merge_sha} ci:green and preserve the exact-head check evidence, then append "
        "ACTIVITY_CLOSED for the currently open bounded activity only. Regenerate "
        "docs/engineering/SII-ENG-008_Project_State.md with python -m sii.continuity --write, verify "
        "python -m sii.continuity --check reports zero open activities and zero inconsistencies, commit and push "
        "the closeout directly to main, then stop. Do not start the next product slice, create a Worker card, edit "
        "canon, or touch PR #82, #83, or #74."
    )
    created = json.loads(run([
        "hermes", "kanban", "--board", board, "create",
        f"SII PR #{verdict.pr} bounded continuity closeout",
        "--assignee", developer, "--workspace", f"dir:{workspace}",
        "--idempotency-key", f"sii-closeout-{verdict.pr}-{verdict.head}",
        "--body", body, "--json",
    ]))
    return {"pr": verdict.pr, "head": verdict.head, "merge_sha": merge_sha,
            "merged_at": after["mergedAt"], "closeout_task": created["id"]}
