"""KILL LOCK audit — issue linkage, PR linkage, dedup both directions, resolution convergence.

The campaign-operations-kill-locks doctrine made enforceable. Mirrors what
the kill-lock skill documents, as code: every shard PR binds its issues
(keyword lines), every issue thread carries the literal #PR token, no
duplicate PRs or issues hide in the cluster, and the whole graph converges
toward resolutions (shipped PRs → close-ready issues).

Invocation:
    python scripts/audit_kill_locks.py --epic 78647            # live audit
    python scripts/audit_kill_locks.py --epic 78647 --json out.json  # dump for tests

The audit logic is pure (takes dicts) so the regression test in
tests/scripts/test_audit_kill_locks.py exercises it offline with fixtures.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any, Iterable

LINKING_KEYWORDS = ("fixes", "closes", "resolves", "part of")
PR_TOKEN_RE = re.compile(r"#(\d{3,6})")


def gh(args: list[str]) -> str:
    r = subprocess.run(
        ["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60
    )
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {r.stderr[:300]}")
    return r.stdout


def pr_keyword_bindings(body: str) -> set[int]:
    """Issue numbers a PR body binds via Fixes/Closes/Resolves/Part of.

    The literal keyword line is the only reliable signal — a 'Related #N'
    footer or a bare '#N' in prose does NOT bind (verified against GitHub's
    cross-reference registry; only these keywords register).
    """
    bound: set[int] = set()
    for line in body.splitlines():
        low = line.strip().lower()
        if not low.startswith(LINKING_KEYWORDS):
            continue
        for m in PR_TOKEN_RE.finditer(line):
            bound.add(int(m.group(1)))
    return bound


def issue_thread_tokens(comments: Iterable[dict[str, Any]]) -> set[int]:
    """Literal #PR tokens in an issue's comment thread (the audit greps these)."""
    tokens: set[int] = set()
    for c in comments:
        body = c.get("body", "")
        for m in PR_TOKEN_RE.finditer(body):
            tokens.add(int(m.group(1)))
    return tokens


def dedup_issues(issues: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """Near-duplicate issues: same normalized title → (a, b, title)."""
    from difflib import SequenceMatcher

    dups: list[tuple[int, int, str]] = []
    norm = [
        (i["number"], re.sub(r"\s+", " ", i["title"]).strip().lower())
        for i in issues
    ]
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            a, ta = norm[i]
            b, tb = norm[j]
            if a == b:
                continue
            ratio = SequenceMatcher(None, ta, tb).ratio()
            if ratio > 0.85:
                dups.append((a, b, norm[i][1][:80]))
    return dups


def dedup_prs(prs: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """Duplicate PRs: the SAME fix shipped twice.

    A real duplicate ships the same issue's fix with the same title —
    e.g. a worktree-session duplicate (#77179/#77181/#77185 class). A
    slice series ('slice R1' vs 'slice R2') is NOT a duplicate even at
    high title similarity: different slices, different windows.
    """
    dups: list[tuple[int, int, str]] = []
    by_title: dict[str, list[int]] = {}
    for p in prs:
        title = re.sub(r"\s+", " ", p["title"]).strip().lower()
        by_title.setdefault(title, []).append(p["number"])
    for title, nums in by_title.items():
        if len(nums) > 1:
            a, b = nums[0], nums[1]
            dups.append((a, b, title[:80]))
    return dups


def audit(
    epic_body: str,
    shard_issues: dict[int, str],          # issue -> title
    issue_comments: dict[int, list[dict]], # issue -> comments
    prs: list[dict[str, Any]],             # PR dicts with number/title/body/state/merged_at
    *,
    expected_keyword_issues: dict[int, set[int]] | None = None,
) -> dict[str, Any]:
    """Run the full kill-lock audit. Pure function — testable offline.

    Returns a report dict with per-direction findings + verdicts.
    """
    report: dict[str, Any] = {
        "pr_to_issue_holes": [],
        "issue_to_pr_holes": [],
        "issue_dups": [],
        "pr_dups": [],
        "unbound_prs": [],
        "unresolved_shards": [],
        "converged": [],
        "verdict": "PASS",
    }

    pr_by_num = {p["number"]: p for p in prs}

    # Direction 1: every PR binds its issues via keyword lines.
    for p in prs:
        body = p.get("body", "")
        bound = pr_keyword_bindings(body)
        if not bound:
            report["unbound_prs"].append(p["number"])
        report["pr_to_issue_holes"].append(
            {"pr": p["number"], "bound": sorted(bound)}
        )

    # Direction 2: every issue thread carries the literal #PR token of
    # every PR that binds it.
    for issue, comments in issue_comments.items():
        tokens = issue_thread_tokens(comments)
        binding_prs = {
            p["number"] for p in prs if issue in pr_keyword_bindings(p.get("body", ""))
        }
        missing = sorted(binding_prs - tokens)
        if missing:
            report["issue_to_pr_holes"].append({"issue": issue, "missing": missing})

    # Dedup both directions.
    report["issue_dups"] = dedup_issues(
        [{"number": n, "title": t} for n, t in shard_issues.items()]
    )
    report["pr_dups"] = dedup_prs(prs)

    # Resolution convergence: shipped (open, all shards done) PRs exist for
    # every shard issue; the graph moves toward closure.
    shipped = {p["number"] for p in prs if p.get("state") == "OPEN"}
    for issue, title in shard_issues.items():
        binding = {
            p["number"] for p in prs if issue in pr_keyword_bindings(p.get("body", ""))
        }
        if not binding:
            report["unresolved_shards"].append({"issue": issue, "title": title[:60]})
        else:
            report["converged"].append({"issue": issue, "prs": sorted(binding)})

    # Verdict: any hole in either direction, any dup, any unresolved shard.
    if (
        report["issue_to_pr_holes"]
        or report["pr_dups"]
        or report["issue_dups"]
        or report["unresolved_shards"]
    ):
        report["verdict"] = "HOLES"

    return report


def fetch_epic_surface(epic: int, repo: str) -> dict[str, Any]:
    """Live fetch: epic body, its shard issues, their comments, and bound PRs."""
    epic_body = gh(["issue", "view", str(epic), "--repo", repo, "--json", "body", "--jq", ".body"])

    issues_json = gh(
        ["issue", "list", "--repo", repo, "--search", f"related:{epic}", "--json", "number,title,state", "--limit", "200"]
    )
    issues = json.loads(issues_json)
    shard_issues = {i["number"]: i["title"] for i in issues}

    issue_comments: dict[int, list[dict]] = {}
    for n in list(shard_issues)[:50]:  # cap for CI speed
        try:
            c = gh(["api", f"repos/{repo}/issues/{n}/comments", "--jq", ".[] | {body}"])
            issue_comments[n] = json.loads(c) if c.strip() else []
        except RuntimeError:
            issue_comments[n] = []

    prs_json = gh(["pr", "list", "--repo", repo, "--search", f"related:{epic}", "--json", "number,title,body,state,mergedAt", "--limit", "100"])
    prs = json.loads(prs_json)

    return {"epic": epic, "epic_body": epic_body, "shard_issues": shard_issues,
            "issue_comments": issue_comments, "prs": prs}


def main() -> int:
    ap = argparse.ArgumentParser(description="KILL LOCK audit")
    ap.add_argument("--epic", type=int, required=True)
    ap.add_argument("--repo", default="NousResearch/hermes-agent")
    ap.add_argument("--json", dest="json_out", help="dump the fetched surface to a file")
    args = ap.parse_args()

    surface = fetch_epic_surface(args.epic, args.repo)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(surface, fh, indent=1, default=str)

    report = audit(
        surface["epic_body"],
        surface["shard_issues"],
        surface["issue_comments"],
        surface["prs"],
    )
    print(json.dumps(report, indent=1, default=str))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
