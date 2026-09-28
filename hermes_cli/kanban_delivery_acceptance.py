"""Read-only GitHub delivery verification for issue-backed Kanban tasks."""
from __future__ import annotations

import json
import re
import subprocess
from urllib.parse import quote

_SOURCE = re.compile(r"github:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+):issue:([1-9][0-9]*):intake")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_SHA = re.compile(r"[0-9a-f]{40}")


def is_source_issue_intake(source_key: str | None) -> bool:
    return bool(_SOURCE.fullmatch(source_key or ""))


def _api(endpoint: str, *, query: str | None = None):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=30, check=True)
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


def collect_delivery_acceptance(source_key: str | None, published_pr: str | None,
                                *, expected_head_sha: str | None = None) -> dict:
    """Verify the source issue is closed by the exact PR merged into its default branch."""
    receipt = {"ok": False, "classification": "missing", "source_issue": source_key,
               "pr_url": published_pr, "head_sha": None,
               "recovery": "Keep the delivery task open; merge the linked PR to the current default branch, "
                           "verify GitHub closed the source issue, then retry completion."}
    source = _SOURCE.fullmatch(source_key or "")
    pr_match = _PR.fullmatch(published_pr or "")
    if not source or not pr_match:
        receipt["detail"] = "A source issue intake key and exact published PR URL are required."
        return receipt
    repo, issue_number = source[1], int(source[2])
    pr_repo, pr_number = pr_match[1], int(pr_match[2])
    receipt.update(repository=repo, issue_number=issue_number, pr_number=pr_number)
    if pr_repo != repo:
        receipt.update(classification="pr_mismatch", detail="Published PR is not in the source issue repository.")
        return receipt

    try:
        owner, name = repo.split("/", 1)
        query = '''{repository(owner:%s,name:%s){defaultBranchRef{name}
            pullRequest(number:%d){headRefOid baseRefName state mergedAt body
                closingIssuesReferences(first:100){nodes{number repository{nameWithOwner}} pageInfo{hasNextPage}}}
            issue(number:%d){state}}}''' % (
                json.dumps(owner), json.dumps(name), pr_number, issue_number)
        data = _api("graphql", query=query)["data"]["repository"]
        if not data or not data.get("defaultBranchRef") or not data.get("pullRequest") or not data.get("issue"):
            receipt.update(classification="missing", detail="GitHub could not resolve the repository, PR, or source issue.")
            return receipt
        default_branch = data["defaultBranchRef"].get("name")
        pr = data["pullRequest"]
        issue = data["issue"]
        sha = pr.get("headRefOid")
        receipt.update(default_branch=default_branch, base_branch=pr.get("baseRefName"),
                       head_sha=sha, pr_state=pr.get("state"), merged_at=pr.get("mergedAt"),
                       issue_state=issue.get("state"))
        if not _SHA.fullmatch(sha or ""):
            receipt.update(classification="stale", detail="PR head SHA is unavailable.")
            return receipt
        if expected_head_sha and sha != expected_head_sha:
            receipt.update(classification="stale", detail="PR head changed between validation and delivery verification.")
            return receipt
        if pr.get("state") != "MERGED" or not pr.get("mergedAt"):
            receipt.update(classification="not_merged", detail="Published PR is not merged.")
            return receipt
        if not default_branch or pr.get("baseRefName") != default_branch:
            receipt.update(classification="wrong_base", detail="PR was not merged into the current default branch.")
            return receipt

        refs = ((pr.get("closingIssuesReferences") or {}).get("nodes") or [])
        linked = any(ref.get("number") == issue_number and
                     ((ref.get("repository") or {}).get("nameWithOwner", "").casefold() == repo.casefold())
                     for ref in refs)
        if not linked:
            if (pr.get("closingIssuesReferences") or {}).get("pageInfo", {}).get("hasNextPage"):
                receipt.update(classification="infra", detail="GitHub issue-link references were incomplete.")
                return receipt
            # GitHub's closingIssuesReferences can lag; a same-repository closing keyword is a safe fallback.
            keyword = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#" +
                                 re.escape(str(issue_number)) + r"\b", re.IGNORECASE)
            linked = bool(keyword.search(pr.get("body") or ""))
        if not linked:
            receipt.update(classification="issue_unlinked", detail="PR does not declare the source issue as a closing reference.")
            return receipt
        if issue.get("state") != "CLOSED":
            receipt.update(classification="issue_open", detail="GitHub has not closed the source issue.")
            return receipt
        receipt.update(ok=True, classification="delivered", detail="Merged PR and closed linked issue verified on the default branch.")
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Do not expose gh stderr or transport details; receipts must not leak credentials/host data.
        receipt.update(classification="infra", detail="GitHub delivery evidence unavailable or incomplete; retry after access recovers.")
        return receipt
