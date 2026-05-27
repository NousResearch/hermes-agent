#!/usr/bin/env python3
"""Audit open Dependabot PRs for stale-maintenance hygiene.

Dry-run by default. The script emits JSON describing which Dependabot PRs are
stale, behind the base branch, conflicting, superseded by newer dependency
bumps, or ready to merge. With ``--update-branch`` it can safely refresh a
small capped batch of stale PRs whose only clear blocker is that they are
behind the base branch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"
DEPENDABOT_LOGINS = {"dependabot[bot]", "dependabot-preview[bot]"}
SUCCESS_CONCLUSIONS = {None, "success", "neutral", "skipped"}
VERSION_SUFFIX_RE = re.compile(r"-(?:v)?\d[\w.+-]*$")
TITLE_PATTERNS = (
    re.compile(r"^Bump\s+(.+?)\s+from\s+", re.IGNORECASE),
    re.compile(r"^build\(deps(?:-[^)]+)?\):\s*bump\s+(.+?)\s+from\s+", re.IGNORECASE),
)
RECOMMENDATION_ORDER = {
    "close_superseded": 0,
    "update_branch": 1,
    "ready_to_merge": 2,
    "recreate_or_manual_rebase": 3,
    "manual_conflict_review": 4,
    "manual_review": 5,
    "wait_for_checks": 6,
}


class GitHubApiError(RuntimeError):
    """Raised when the GitHub API returns a non-success response."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"GitHub API returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/repo format")
    parser.add_argument(
        "--stale-days",
        type=int,
        default=7,
        help="Age or idle threshold, in days, for treating a Dependabot PR as stale (default: 7)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of open PRs to inspect before filtering to Dependabot (default: 100)",
    )
    parser.add_argument(
        "--update-branch",
        action="store_true",
        help="Call GitHub's update-branch endpoint for stale Dependabot PRs that are merely behind base",
    )
    parser.add_argument(
        "--close-superseded",
        action="store_true",
        help="Close older superseded Dependabot PRs when a newer PR for the same dependency is already open",
    )
    parser.add_argument(
        "--autofix",
        action="store_true",
        help="Convenience flag: enable both --update-branch and --close-superseded",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=3,
        help="Maximum number of stale behind-branch PRs to refresh in one run (default: 3)",
    )
    parser.add_argument(
        "--max-closes",
        type=int,
        default=10,
        help="Maximum number of superseded PRs to close in one run (default: 10)",
    )
    parser.add_argument(
        "--watchdog",
        action="store_true",
        help="Emit cron-friendly text and stay silent when the Dependabot backlog is healthy",
    )
    return parser.parse_args()


def _read_env_file_token() -> str:
    """Read a GitHub token from ~/.hermes/.env when present."""
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text().splitlines():
        if line.startswith("GITHUB_TOKEN=") or line.startswith("GH_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _read_git_credentials_token() -> str:
    """Read a GitHub token from ~/.git-credentials when present."""
    creds_path = Path.home() / ".git-credentials"
    if not creds_path.is_file():
        return ""
    for line in creds_path.read_text().splitlines():
        if "github.com" not in line:
            continue
        match = re.search(r"https://[^:]+:([^@]+)@", line)
        if match:
            return urllib.parse.unquote(match.group(1))
    return ""


def _read_gh_cli_token() -> str:
    """Read a GitHub token from gh auth when available."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def get_github_token() -> str:
    """Resolve a GitHub token from env, Hermes env file, credentials, or gh auth."""
    for env_name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return token

    for token in (_read_env_file_token(), _read_git_credentials_token(), _read_gh_cli_token()):
        if token:
            return token

    sys.stderr.write(
        "ERROR: no GitHub token found. Export GITHUB_TOKEN, add it to ~/.hermes/.env, "
        "or authenticate gh with `gh auth login`.\n"
    )
    sys.exit(2)


def parse_github_datetime(value: str) -> datetime:
    """Parse a GitHub ISO timestamp into a timezone-aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def age_in_days(now: datetime, value: str) -> float:
    """Return fractional age in days for a GitHub timestamp."""
    return round((now - parse_github_datetime(value)).total_seconds() / 86400, 1)


def is_dependabot_pr(pr: dict[str, Any]) -> bool:
    """Return True when the PR clearly belongs to Dependabot."""
    login = ((pr.get("user") or {}).get("login") or "").lower()
    if login in DEPENDABOT_LOGINS:
        return True

    head_ref = ((pr.get("head") or {}).get("ref") or "").lower()
    return head_ref.startswith("dependabot/")


def dependency_key_from_pr(pr: dict[str, Any]) -> str:
    """Extract a stable dependency key from a Dependabot PR title or branch."""
    title = pr.get("title") or ""
    for pattern in TITLE_PATTERNS:
        match = pattern.match(title)
        if match:
            return match.group(1).strip().lower()

    head_ref = (pr.get("head") or {}).get("ref") or ""
    if not head_ref.lower().startswith("dependabot/"):
        return ""

    parts = head_ref.split("/")
    if len(parts) < 3:
        return ""

    dependency_parts = parts[2:]
    dependency_parts[-1] = VERSION_SUFFIX_RE.sub("", dependency_parts[-1])
    return "/".join(part for part in dependency_parts if part).lower()


def summarize_checks(check_bundle: dict[str, Any]) -> dict[str, Any]:
    """Collapse commit-status and check-run payloads into one summary."""
    commit_status = check_bundle.get("commit_status") or {}
    check_runs = check_bundle.get("check_runs") or []

    failed: list[str] = []
    pending: list[str] = []
    passed: list[str] = []

    overall = commit_status.get("state")
    if overall in {"failure", "error"}:
        failed.append(f"commit_status:{overall}")
    elif overall == "pending":
        pending.append("commit_status")
    elif overall == "success":
        passed.append("commit_status")

    for run in check_runs:
        name = run.get("name") or "unnamed-check"
        status = run.get("status")
        conclusion = run.get("conclusion")
        if status != "completed":
            pending.append(name)
            continue
        if conclusion in SUCCESS_CONCLUSIONS:
            passed.append(name)
            continue
        failed.append(f"{name}:{conclusion or 'unknown'}")

    if failed:
        summary = "failure"
    elif pending:
        summary = "pending"
    elif passed:
        summary = "success"
    else:
        summary = "none"

    return {
        "summary": summary,
        "failed": failed,
        "pending": pending,
        "passed": passed,
    }


def _issue_labels(issue: dict[str, Any]) -> set[str]:
    """Return normalized label names from a PR's issue payload."""
    return {
        (label.get("name") or "").strip().lower()
        for label in issue.get("labels") or []
        if label.get("name")
    }


def classify_dependabot_pr(
    pr: dict[str, Any],
    issue: dict[str, Any],
    checks: dict[str, Any],
    stale_days: int,
    now: datetime,
    superseded_by: int | None = None,
) -> dict[str, Any]:
    """Classify one Dependabot PR for stale-maintenance triage."""
    labels = _issue_labels(issue)
    dependency = dependency_key_from_pr(pr)
    age_days = age_in_days(now, pr["created_at"])
    idle_days = age_in_days(now, pr.get("updated_at") or pr["created_at"])
    has_stale_label = "stale" in labels
    is_old = age_days >= stale_days
    is_idle = idle_days >= stale_days
    is_stale = has_stale_label or is_old or is_idle
    mergeable_state = (pr.get("mergeable_state") or "unknown").lower()
    mergeable = pr.get("mergeable")
    is_draft = bool(pr.get("draft"))
    checks_summary = summarize_checks(checks)

    reasons: list[str] = []
    if is_old:
        reasons.append(f"open {age_days}d")
    if is_idle:
        reasons.append(f"idle {idle_days}d")
    if has_stale_label:
        reasons.append("stale label")

    recommendation = "manual_review"
    if superseded_by is not None:
        recommendation = "close_superseded"
        reasons.append(f"superseded by newer PR #{superseded_by}")
    elif is_draft:
        recommendation = "manual_review"
        reasons.append("draft PR")
    elif mergeable_state == "behind":
        recommendation = "update_branch"
        reasons.append("head branch behind base")
    elif mergeable_state == "dirty" or mergeable is False:
        recommendation = (
            "recreate_or_manual_rebase" if is_stale else "manual_conflict_review"
        )
        reasons.append("merge conflict")
    elif checks_summary["summary"] == "failure":
        recommendation = "manual_review"
        reasons.append("required checks failing")
    elif checks_summary["summary"] == "pending":
        recommendation = "wait_for_checks"
        reasons.append("required checks pending")
    elif mergeable is True and mergeable_state in {"clean", "unstable", "has_hooks"}:
        recommendation = "ready_to_merge"
        reasons.append("mergeable and checks not failing")
    else:
        reasons.append("mergeability unclear")

    return {
        "number": pr["number"],
        "title": pr.get("title"),
        "url": pr.get("html_url"),
        "author": (pr.get("user") or {}).get("login"),
        "dependency": dependency,
        "base_ref": (pr.get("base") or {}).get("ref"),
        "head_ref": (pr.get("head") or {}).get("ref"),
        "head_sha": (pr.get("head") or {}).get("sha"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "age_days": age_days,
        "idle_days": idle_days,
        "is_stale": is_stale,
        "is_old": is_old,
        "is_idle": is_idle,
        "has_stale_label": has_stale_label,
        "labels": sorted(labels),
        "mergeable": mergeable,
        "mergeable_state": mergeable_state,
        "outdated_base_branch": mergeable_state == "behind",
        "checks": checks_summary,
        "superseded_by": superseded_by,
        "recommendation": recommendation,
        "can_update_branch": (
            mergeable_state == "behind" and not is_draft and superseded_by is None
        ),
        "checks_outdated": mergeable_state == "behind" and checks_summary["summary"] == "success",
        "reasons": reasons,
    }


def build_audit(
    prs: list[dict[str, Any]],
    issues_by_number: dict[int, dict[str, Any]],
    checks_by_number: dict[int, dict[str, Any]],
    stale_days: int,
    now: datetime,
) -> list[dict[str, Any]]:
    """Build a sorted stale-maintenance audit for open Dependabot PRs."""
    dependabot_prs = [pr for pr in prs if is_dependabot_pr(pr)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    superseded: dict[int, int] = {}

    for pr in dependabot_prs:
        dependency = dependency_key_from_pr(pr)
        if not dependency:
            continue
        grouped.setdefault(dependency, []).append(pr)

    for group in grouped.values():
        if len(group) < 2:
            continue
        group.sort(
            key=lambda pr: (
                parse_github_datetime(pr.get("created_at") or pr.get("updated_at") or "1970-01-01T00:00:00Z"),
                pr["number"],
            ),
            reverse=True,
        )
        newest_number = group[0]["number"]
        for pr in group[1:]:
            superseded[pr["number"]] = newest_number

    records = [
        classify_dependabot_pr(
            pr,
            issues_by_number.get(pr["number"], {}),
            checks_by_number.get(pr["number"], {}),
            stale_days,
            now,
            superseded_by=superseded.get(pr["number"]),
        )
        for pr in dependabot_prs
    ]
    records.sort(
        key=lambda record: (
            RECOMMENDATION_ORDER.get(record["recommendation"], 99),
            -record["age_days"],
            record["number"],
        )
    )
    return records


class GitHubClient:
    """Minimal GitHub REST client for the Dependabot hygiene helper."""

    def __init__(self, token: str):
        self.token = token

    def request_json(
        self,
        method: str,
        path_or_url: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, str]]:
        """Execute one JSON GitHub API request and return body plus headers."""
        url = path_or_url
        if not path_or_url.startswith("https://"):
            url = f"{API_ROOT}/{path_or_url.lstrip('/')}"

        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "hermes-agent-dependabot-hygiene/1.0",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", "replace")
            raise GitHubApiError(exc.code, body_text) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error while calling GitHub: {exc}") from exc

        if not raw:
            return {}, headers
        return json.loads(raw), headers

    def list_open_prs(self, repo: str, limit: int) -> list[dict[str, Any]]:
        """List open pull requests with pagination, capped by limit."""
        items: list[dict[str, Any]] = []
        url = f"repos/{repo}/pulls?state=open&per_page=100"
        while url and len(items) < limit:
            page_items, headers = self.request_json("GET", url)
            if isinstance(page_items, list):
                items.extend(page_items)
            else:
                break
            url = next_link(headers.get("Link", ""))
        return items[:limit]

    def get_pr(self, repo: str, number: int) -> dict[str, Any]:
        """Get detailed PR metadata, including mergeability fields."""
        data, _ = self.request_json("GET", f"repos/{repo}/pulls/{number}")
        return data

    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        """Get issue metadata for label-based stale detection."""
        data, _ = self.request_json("GET", f"repos/{repo}/issues/{number}")
        return data

    def get_checks(self, repo: str, sha: str) -> dict[str, Any]:
        """Collect commit-status and check-run state for one PR head SHA."""
        commit_status, _ = self.request_json("GET", f"repos/{repo}/commits/{sha}/status")
        check_runs, _ = self.request_json("GET", f"repos/{repo}/commits/{sha}/check-runs")
        return {
            "commit_status": commit_status,
            "check_runs": check_runs.get("check_runs", []),
        }

    def update_branch(self, repo: str, number: int, expected_head_sha: str) -> dict[str, Any]:
        """Request GitHub to update a PR branch against its base branch."""
        response, _ = self.request_json(
            "PUT",
            f"repos/{repo}/pulls/{number}/update-branch",
            payload={"expected_head_sha": expected_head_sha},
        )
        return response if isinstance(response, dict) else {"response": response}

    def close_pr(self, repo: str, number: int) -> dict[str, Any]:
        """Close a pull request via the REST API."""
        response, _ = self.request_json(
            "PATCH",
            f"repos/{repo}/pulls/{number}",
            payload={"state": "closed"},
        )
        return response if isinstance(response, dict) else {"response": response}


def next_link(link_header: str) -> str:
    """Extract the next-page URL from a GitHub Link header."""
    for chunk in link_header.split(","):
        chunk = chunk.strip()
        if 'rel="next"' not in chunk:
            continue
        match = re.search(r"<([^>]+)>", chunk)
        if match:
            return match.group(1)
    return ""


def summarize_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count recommendations across the audit."""
    summary: dict[str, int] = {}
    for record in records:
        summary[record["recommendation"]] = summary.get(record["recommendation"], 0) + 1
    return summary


def apply_update_branch_actions(
    client: GitHubClient,
    repo: str,
    records: list[dict[str, Any]],
    max_updates: int,
) -> None:
    """Safely update a capped set of stale Dependabot PR branches."""
    applied = 0
    for record in records:
        if not record["can_update_branch"] or not record["is_stale"]:
            continue
        if applied >= max_updates:
            record["update_branch"] = {
                "attempted": False,
                "skipped": "max_updates_reached",
            }
            continue
        try:
            response = client.update_branch(repo, record["number"], record["head_sha"])
        except Exception as exc:  # noqa: BLE001 - surface GitHub response directly
            record["update_branch"] = {
                "attempted": True,
                "success": False,
                "error": str(exc),
            }
            applied += 1
            continue

        record["update_branch"] = {
            "attempted": True,
            "success": True,
            "response": response,
        }
        applied += 1


def apply_close_actions(
    client: GitHubClient,
    repo: str,
    records: list[dict[str, Any]],
    max_closes: int,
) -> None:
    """Safely close capped superseded Dependabot PRs."""
    applied = 0
    for record in records:
        if record["recommendation"] != "close_superseded":
            continue
        if applied >= max_closes:
            record["close_pr"] = {
                "attempted": False,
                "skipped": "max_closes_reached",
            }
            continue
        try:
            response = client.close_pr(repo, record["number"])
        except Exception as exc:  # noqa: BLE001 - surface GitHub response directly
            record["close_pr"] = {
                "attempted": True,
                "success": False,
                "error": str(exc),
            }
            applied += 1
            continue

        record["close_pr"] = {
            "attempted": True,
            "success": True,
            "response": response,
        }
        applied += 1


def _action_counts(records: list[dict[str, Any]], key: str) -> tuple[int, int, int]:
    """Return attempted, successful, and failed counts for one action key."""
    attempted = 0
    successful = 0
    failed = 0
    for record in records:
        action = record.get(key)
        if not action or not action.get("attempted"):
            continue
        attempted += 1
        if action.get("success"):
            successful += 1
        else:
            failed += 1
    return attempted, successful, failed


def render_watchdog_report(report: dict[str, Any]) -> str:
    """Render a no-agent cron summary; return empty string when healthy."""
    records = report["prs"]
    update_attempted, update_ok, update_failed = _action_counts(records, "update_branch")
    close_attempted, close_ok, close_failed = _action_counts(records, "close_pr")
    unresolved = [
        record
        for record in records
        if record["recommendation"] in {"manual_review", "manual_conflict_review", "recreate_or_manual_rebase"}
    ]
    waiting = [record for record in records if record["recommendation"] == "wait_for_checks"]

    if not records:
        return ""
    if not any((update_attempted, close_attempted, unresolved, waiting)):
        return ""

    lines = [f"Dependabot hygiene: {report['repo']}"]
    if close_attempted:
        lines.append(f"- closed superseded PRs: {close_ok}/{close_attempted}" + (f" ({close_failed} failed)" if close_failed else ""))
    if update_attempted:
        lines.append(f"- refreshed stale branches: {update_ok}/{update_attempted}" + (f" ({update_failed} failed)" if update_failed else ""))
    if unresolved:
        lines.append("- manual backlog:")
        for record in unresolved[:10]:
            lines.append(
                f"  - #{record['number']} {record['recommendation']} — {record['title']}"
            )
    if waiting:
        lines.append("- waiting for checks:")
        for record in waiting[:10]:
            lines.append(f"  - #{record['number']} — {record['title']}")
    return "\n".join(lines)


def main() -> int:
    """Entry point for CLI usage."""
    args = parse_args()
    if args.autofix:
        args.update_branch = True
        args.close_superseded = True
    now = datetime.now(timezone.utc)
    client = GitHubClient(get_github_token())

    open_prs = client.list_open_prs(args.repo, args.limit)
    detailed_prs: list[dict[str, Any]] = []
    issues_by_number: dict[int, dict[str, Any]] = {}
    checks_by_number: dict[int, dict[str, Any]] = {}

    for pr in open_prs:
        if not is_dependabot_pr(pr):
            continue
        number = pr["number"]
        detailed = client.get_pr(args.repo, number)
        detailed_prs.append(detailed)
        issues_by_number[number] = client.get_issue(args.repo, number)
        checks_by_number[number] = client.get_checks(args.repo, (detailed.get("head") or {}).get("sha", ""))

    records = build_audit(detailed_prs, issues_by_number, checks_by_number, args.stale_days, now)

    if args.close_superseded:
        apply_close_actions(client, args.repo, records, args.max_closes)
    if args.update_branch:
        apply_update_branch_actions(client, args.repo, records, args.max_updates)

    report = {
        "repo": args.repo,
        "generated_at": now.isoformat(),
        "stale_days": args.stale_days,
        "open_dependabot_prs": len(records),
        "recommendations": summarize_records(records),
        "prs": records,
    }
    if args.watchdog:
        watchdog_text = render_watchdog_report(report)
        if watchdog_text:
            print(watchdog_text)
        return 0

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())