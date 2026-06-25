#!/usr/bin/env python3
"""Read-only GitHub repository inventory for Hermes skills.

The script prefers an authenticated `gh` session when one is available. It only
falls back to REST when `gh` is unavailable or not authenticated, so a missing
GITHUB_TOKEN environment variable is not misreported as an auth failure on
machines where `gh auth status` is already valid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MINIMAL_REPO_FIELDS = [
    "nameWithOwner",
    "description",
    "defaultBranchRef",
    "isPrivate",
    "pushedAt",
    "updatedAt",
    "url",
]

ISSUE_FIELDS = [
    "number",
    "title",
    "state",
    "createdAt",
    "updatedAt",
    "url",
    "labels",
    "assignees",
]

PR_FIELDS = [
    "number",
    "title",
    "state",
    "createdAt",
    "updatedAt",
    "url",
    "headRefName",
    "baseRefName",
    "isDraft",
]


def run_command(
    args: list[str], env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    proc = subprocess.run(args, text=True, capture_output=True, check=False, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def _append_unique(paths: list[Path], path: Path) -> None:
    expanded = path.expanduser()
    if expanded not in paths:
        paths.append(expanded)


def _real_home_from_hermes_path(path: Path) -> Path | None:
    text = str(path.expanduser())
    marker = "/.hermes/profiles/"
    if marker in text:
        return Path(text.split(marker, 1)[0])
    if text.endswith("/.hermes"):
        return Path(text).parent
    return None


def candidate_home_dirs() -> list[Path]:
    """Return homes that may contain gh/git credentials."""
    homes: list[Path] = []
    _append_unique(homes, Path.home())

    for key in ("HOME", "HERMES_HOME"):
        value = os.environ.get(key, "").strip()
        if not value:
            continue
        path = Path(value)
        _append_unique(homes, path)
        real_home = _real_home_from_hermes_path(path)
        if real_home is not None:
            _append_unique(homes, real_home)

    return homes


def candidate_gh_config_dirs() -> list[Path]:
    dirs: list[Path] = []
    configured = os.environ.get("GH_CONFIG_DIR", "").strip()
    if configured:
        _append_unique(dirs, Path(configured))
    for home in candidate_home_dirs():
        _append_unique(dirs, home / ".config" / "gh")
    return dirs


def env_with_gh_config(config_dir: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if config_dir:
        env["GH_CONFIG_DIR"] = config_dir
    return env


def load_token() -> str:
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value

    for home in candidate_home_dirs():
        env_path = home / ".hermes" / ".env"
        if env_path.exists():
            for line in env_path.read_text(errors="ignore").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() in {"GH_TOKEN", "GITHUB_TOKEN"}:
                    return value.strip().strip("'\"")

        credentials = home / ".git-credentials"
        if credentials.exists():
            for line in credentials.read_text(errors="ignore").splitlines():
                if "github.com" not in line:
                    continue
                parsed = urllib.parse.urlparse(line.strip())
                if parsed.password:
                    return urllib.parse.unquote(parsed.password)

    return ""


def detect_auth() -> dict[str, Any]:
    gh_path = shutil.which("gh")
    if gh_path:
        for config_dir in candidate_gh_config_dirs():
            config = str(config_dir)
            env = env_with_gh_config(config)
            code, _, _ = run_command(["gh", "auth", "status"], env=env)
            if code != 0:
                continue
            user = ""
            user_code, user_out, _ = run_command(
                [
                    "gh",
                    "api",
                    "user",
                    "--jq",
                    ".login",
                ],
                env=env,
            )
            if user_code == 0:
                user = user_out.strip()
            return {
                "method": "gh",
                "gh_available": True,
                "gh_authenticated": True,
                "gh_config_dir": config,
                "user": user,
            }

    token = load_token()
    if token:
        return {
            "method": "rest_token",
            "gh_available": bool(gh_path),
            "gh_authenticated": False,
            "user": "",
        }

    return {
        "method": "rest_unauthenticated",
        "gh_available": bool(gh_path),
        "gh_authenticated": False,
        "user": "",
    }


def parse_json(stdout: str, fallback: Any) -> Any:
    try:
        return json.loads(stdout or "null")
    except json.JSONDecodeError:
        return fallback


def normalize_issue(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": item.get("number"),
        "title": item.get("title"),
        "state": item.get("state"),
        "createdAt": item.get("createdAt"),
        "updatedAt": item.get("updatedAt"),
        "url": item.get("url"),
        "labels": [
            label.get("name")
            for label in item.get("labels", [])
            if isinstance(label, dict)
        ],
        "assignees": [
            assignee.get("login")
            for assignee in item.get("assignees", [])
            if isinstance(assignee, dict)
        ],
    }


def normalize_pr(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": item.get("number"),
        "title": item.get("title"),
        "state": item.get("state"),
        "createdAt": item.get("createdAt"),
        "updatedAt": item.get("updatedAt"),
        "url": item.get("url"),
        "headRefName": item.get("headRefName"),
        "baseRefName": item.get("baseRefName"),
        "isDraft": item.get("isDraft"),
    }


def collect_with_gh(repo: str, config_dir: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"repo": repo, "status": "ok"}
    env = env_with_gh_config(config_dir)

    code, stdout, stderr = run_command(
        [
            "gh",
            "repo",
            "view",
            repo,
            "--json",
            ",".join(MINIMAL_REPO_FIELDS),
        ],
        env=env,
    )
    if code != 0:
        result.update({
            "status": "repo_error",
            "error": stderr.strip() or stdout.strip(),
        })
        return result

    repo_data = parse_json(stdout, {})
    result["summary"] = {
        "nameWithOwner": repo_data.get("nameWithOwner"),
        "description": repo_data.get("description"),
        "defaultBranch": (repo_data.get("defaultBranchRef") or {}).get("name"),
        "isPrivate": repo_data.get("isPrivate"),
        "pushedAt": repo_data.get("pushedAt"),
        "updatedAt": repo_data.get("updatedAt"),
        "url": repo_data.get("url"),
    }

    issue_code, issue_stdout, issue_stderr = run_command(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            ",".join(ISSUE_FIELDS),
        ],
        env=env,
    )
    if issue_code == 0:
        issues = parse_json(issue_stdout, [])
        result["openIssues"] = [
            normalize_issue(item) for item in issues if isinstance(item, dict)
        ]
    else:
        result["openIssues"] = []
        result["issuesError"] = issue_stderr.strip() or issue_stdout.strip()

    pr_code, pr_stdout, pr_stderr = run_command(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            ",".join(PR_FIELDS),
        ],
        env=env,
    )
    if pr_code == 0:
        prs = parse_json(pr_stdout, [])
        result["openPullRequests"] = [
            normalize_pr(item) for item in prs if isinstance(item, dict)
        ]
    else:
        result["openPullRequests"] = []
        result["pullRequestsError"] = pr_stderr.strip() or pr_stdout.strip()

    return result


def rest_get(path: str, token: str) -> tuple[int, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hermes-github-readonly-inventory",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"message": raw[:500]}
        return exc.code, body


def collect_with_rest(repo: str, token: str) -> dict[str, Any]:
    result: dict[str, Any] = {"repo": repo}

    code, repo_data = rest_get(f"/repos/{repo}", token)
    if code != 200:
        result.update({
            "status": "repo_error",
            "httpStatus": code,
            "error": repo_data.get("message", "GitHub repo request failed")
            if isinstance(repo_data, dict)
            else "GitHub repo request failed",
        })
        return result

    result["status"] = "ok"
    result["summary"] = {
        "nameWithOwner": repo_data.get("full_name"),
        "description": repo_data.get("description"),
        "defaultBranch": repo_data.get("default_branch"),
        "isPrivate": repo_data.get("private"),
        "pushedAt": repo_data.get("pushed_at"),
        "updatedAt": repo_data.get("updated_at"),
        "url": repo_data.get("html_url"),
    }

    issue_code, issues = rest_get(
        f"/repos/{repo}/issues?state=open&per_page=100", token
    )
    if issue_code == 200 and isinstance(issues, list):
        result["openIssues"] = [
            {
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state"),
                "createdAt": item.get("created_at"),
                "updatedAt": item.get("updated_at"),
                "url": item.get("html_url"),
                "labels": [
                    label.get("name")
                    for label in item.get("labels", [])
                    if isinstance(label, dict)
                ],
                "assignees": [
                    assignee.get("login")
                    for assignee in item.get("assignees", [])
                    if isinstance(assignee, dict)
                ],
            }
            for item in issues
            if isinstance(item, dict) and "pull_request" not in item
        ]
    else:
        result["openIssues"] = []
        if isinstance(issues, dict) and issues.get("message"):
            result["issuesError"] = issues["message"]

    pr_code, prs = rest_get(f"/repos/{repo}/pulls?state=open&per_page=100", token)
    if pr_code == 200 and isinstance(prs, list):
        result["openPullRequests"] = [
            {
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state"),
                "createdAt": item.get("created_at"),
                "updatedAt": item.get("updated_at"),
                "url": item.get("html_url"),
                "headRefName": (item.get("head") or {}).get("ref"),
                "baseRefName": (item.get("base") or {}).get("ref"),
                "isDraft": item.get("draft"),
            }
            for item in prs
            if isinstance(item, dict)
        ]
    else:
        result["openPullRequests"] = []
        if isinstance(prs, dict) and prs.get("message"):
            result["pullRequestsError"] = prs["message"]

    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Collect read-only GitHub repo, issue, and PR inventory."
    )
    parser.add_argument("repos", nargs="+", help="Repository names in owner/repo form.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any repo cannot be read.",
    )
    args = parser.parse_args(argv)

    auth = detect_auth()
    token = load_token() if auth["method"] == "rest_token" else ""
    gh_config_dir = auth.get("gh_config_dir") if auth["method"] == "gh" else None
    repos = [
        collect_with_gh(repo, gh_config_dir)
        if auth["method"] == "gh"
        else collect_with_rest(repo, token)
        for repo in args.repos
    ]
    output = {"auth": auth, "repos": repos}
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))

    if args.strict and any(repo.get("status") != "ok" for repo in repos):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
