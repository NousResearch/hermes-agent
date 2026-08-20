"""GitHub connector tools for Hermes (registered via plugins/github).

Every tool resolves credentials bot-first (GitHub App installation token
when configured), and every successful result includes an attribution
block — ``auth_method`` + ``actor`` — so the agent and the user can always
tell whether an action was performed by the bot (``<slug>[bot]``) or by a
human account. This is the connector's core value: Hermes's GitHub actions
are visibly distinct from the user's.

Requires one of:
- GitHub App credentials: ``GITHUB_APP_ID`` + ``GITHUB_APP_PRIVATE_KEY_PATH``
  + ``GITHUB_APP_INSTALLATION_ID`` (recommended — bot identity)
- ``GITHUB_TOKEN`` / ``GH_TOKEN`` (PAT — human identity)
- ``gh auth login`` (gh CLI — human identity)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from plugins.github.client import (
    GitHubClient,
    GitHubError,
    parse_repo,
)
from tools.registry import tool_error, tool_result


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_github_available() -> bool:
    try:
        from agent.github_auth import GitHubAppAuth
        from agent.secret_scope import get_secret

        if GitHubAppAuth().credentials_configured():
            return True
        if get_secret("GITHUB_TOKEN") or get_secret("GH_TOKEN"):
            return True
        import shutil

        return shutil.which("gh") is not None
    except Exception:
        return False


def _github_client() -> GitHubClient:
    return GitHubClient()


def _github_tool_error(exc: Exception) -> str:
    if isinstance(exc, GitHubError):
        return tool_error(exc.message, status_code=exc.status_code)
    return tool_error(f"GitHub tool failed: {type(exc).__name__}: {exc}")


def _with_attribution(client: GitHubClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the action result with who performed it."""
    payload["attribution"] = client.attribution()
    return payload


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

GITHUB_IDENTITY_SCHEMA = {
    "name": "github_identity",
    "description": (
        "Verify which GitHub identity Hermes currently acts as: the GitHub App bot "
        "(<slug>[bot]) or a human account. Returns auth method, actor login, app slug "
        "when applicable, and accessible repositories. Use this first to confirm bot "
        "identity before commenting/reviewing, so agent actions are distinguishable "
        "from the user's."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GITHUB_CREATE_ISSUE_SCHEMA = {
    "name": "github_create_issue",
    "description": (
        "Create a GitHub issue in a repository as the resolved identity (bot when a "
        "GitHub App is configured). Returns the new issue (number, url, state) plus "
        "attribution of who created it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name format, e.g. himanusia/plus1"},
            "title": {"type": "string", "description": "Issue title"},
            "body": {"type": "string", "description": "Issue body (markdown)"},
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Label names to apply"},
            "assignees": {"type": "array", "items": {"type": "string"}, "description": "GitHub logins to assign"},
        },
        "required": ["repo", "title"],
    },
}

GITHUB_COMMENT_ISSUE_SCHEMA = {
    "name": "github_comment_issue",
    "description": (
        "Comment on a GitHub issue or pull request as the resolved identity. Works for "
        "both issues and PRs (PR comments land in the issue thread). Returns the comment "
        "plus attribution of who wrote it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name format"},
            "number": {"type": "integer", "description": "Issue or PR number"},
            "body": {"type": "string", "description": "Comment body (markdown)"},
        },
        "required": ["repo", "number", "body"],
    },
}

GITHUB_LIST_ISSUES_SCHEMA = {
    "name": "github_list_issues",
    "description": "List GitHub issues in a repository with filters (state, labels, assignee, creator, sort). Read-only.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name format"},
            "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Issue state (default open)"},
            "labels": {"type": "string", "description": "Comma-separated label names to filter by"},
            "assignee": {"type": "string", "description": "Login of the assignee"},
            "creator": {"type": "string", "description": "Login of the creator"},
            "sort": {"type": "string", "enum": ["created", "updated", "comments"], "description": "Sort key (default created)"},
            "direction": {"type": "string", "enum": ["asc", "desc"], "description": "Sort direction (default desc)"},
            "per_page": {"type": "integer", "description": "Results per page, max 100 (default 30)"},
        },
        "required": ["repo"],
    },
}

GITHUB_GET_ISSUE_SCHEMA = {
    "name": "github_get_issue",
    "description": "Fetch a GitHub issue (or PR) detail, optionally including its comments. Read-only.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name format"},
            "number": {"type": "integer", "description": "Issue or PR number"},
            "include_comments": {"type": "boolean", "description": "Also fetch the comment thread (default false)"},
        },
        "required": ["repo", "number"],
    },
}

GITHUB_REVIEW_PR_SCHEMA = {
    "name": "github_review_pr",
    "description": (
        "Submit a pull request review as the resolved identity: approve, request changes, "
        "or comment. Optionally attach inline comments to specific diff lines. Returns the "
        "review plus attribution of who submitted it. Use for PR review workflows where the "
        "agent's review must be attributed to the bot, not the human account."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name format"},
            "number": {"type": "integer", "description": "Pull request number"},
            "event": {
                "type": "string",
                "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"],
                "description": "Review event: APPROVE, REQUEST_CHANGES, or COMMENT",
            },
            "body": {"type": "string", "description": "Review summary (markdown)"},
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path in the diff"},
                        "line": {"type": "integer", "description": "Line number in the diff"},
                        "body": {"type": "string", "description": "Inline comment text"},
                    },
                    "required": ["path", "line", "body"],
                },
                "description": "Optional inline review comments",
            },
        },
        "required": ["repo", "number", "event"],
    },
}

GITHUB_MERGE_PR_SCHEMA = {
    "name": "github_merge_pr",
    "description": (
        "Merge a GitHub pull request as the resolved identity. Default merge method is "
        "squash. Returns the merge result plus attribution of who merged it. Use when the "
        "agent is authorized to merge (e.g. after CI + review gates pass)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name format"},
            "number": {"type": "integer", "description": "Pull request number"},
            "method": {
                "type": "string",
                "enum": ["squash", "merge", "rebase"],
                "description": "Merge method (default squash)",
            },
            "commit_title": {"type": "string", "description": "Custom merge commit title"},
        },
        "required": ["repo", "number"],
    },
}

_TOOLS = (
    ("github_identity", GITHUB_IDENTITY_SCHEMA, "🆔"),
    ("github_create_issue", GITHUB_CREATE_ISSUE_SCHEMA, "🐛"),
    ("github_comment_issue", GITHUB_COMMENT_ISSUE_SCHEMA, "💬"),
    ("github_list_issues", GITHUB_LIST_ISSUES_SCHEMA, "📋"),
    ("github_get_issue", GITHUB_GET_ISSUE_SCHEMA, "🔍"),
    ("github_review_pr", GITHUB_REVIEW_PR_SCHEMA, "👁️"),
    ("github_merge_pr", GITHUB_MERGE_PR_SCHEMA, "🔀"),
)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_github_identity(args: Dict[str, Any], **kw) -> str:
    try:
        client = _github_client()
        info = client.verify_identity()
        return tool_result(info)
    except Exception as exc:
        return _github_tool_error(exc)


def _handle_github_create_issue(args: Dict[str, Any], **kw) -> str:
    try:
        owner, repo = parse_repo(args.get("repo", ""))
        client = _github_client()
        issue = client.create_issue(
            owner=owner,
            repo=repo,
            title=args.get("title", ""),
            body=args.get("body", ""),
            labels=args.get("labels"),
            assignees=args.get("assignees"),
        )
        return tool_result(
            _with_attribution(
                client,
                {
                    "success": True,
                    "action": "create_issue",
                    "issue_number": issue.get("number"),
                    "url": issue.get("html_url"),
                    "state": issue.get("state"),
                    "title": issue.get("title"),
                },
            )
        )
    except Exception as exc:
        return _github_tool_error(exc)


def _handle_github_comment_issue(args: Dict[str, Any], **kw) -> str:
    try:
        owner, repo = parse_repo(args.get("repo", ""))
        client = _github_client()
        comment = client.comment_issue(
            owner=owner,
            repo=repo,
            number=int(args.get("number", 0)),
            body=args.get("body", ""),
        )
        return tool_result(
            _with_attribution(
                client,
                {
                    "success": True,
                    "action": "comment_issue",
                    "comment_id": comment.get("id"),
                    "url": comment.get("html_url"),
                    "author": (comment.get("user") or {}).get("login"),
                },
            )
        )
    except Exception as exc:
        return _github_tool_error(exc)


def _handle_github_list_issues(args: Dict[str, Any], **kw) -> str:
    try:
        owner, repo = parse_repo(args.get("repo", ""))
        client = _github_client()
        issues = client.list_issues(
            owner=owner,
            repo=repo,
            state=args.get("state", "open"),
            labels=args.get("labels"),
            assignee=args.get("assignee"),
            creator=args.get("creator"),
            sort=args.get("sort", "created"),
            direction=args.get("direction", "desc"),
            per_page=int(args.get("per_page") or 30),
        )
        summary = [
            {
                "number": i.get("number"),
                "title": i.get("title"),
                "state": i.get("state"),
                "user": (i.get("user") or {}).get("login"),
                "labels": [l.get("name") for l in (i.get("labels") or [])],
                "created_at": i.get("created_at"),
            }
            for i in issues
        ]
        return tool_result({"success": True, "action": "list_issues", "count": len(summary), "issues": summary})
    except Exception as exc:
        return _github_tool_error(exc)


def _handle_github_get_issue(args: Dict[str, Any], **kw) -> str:
    try:
        owner, repo = parse_repo(args.get("repo", ""))
        client = _github_client()
        issue = client.get_issue(
            owner=owner,
            repo=repo,
            number=int(args.get("number", 0)),
            include_comments=bool(args.get("include_comments")),
        )
        return tool_result({"success": True, "action": "get_issue", "issue": issue})
    except Exception as exc:
        return _github_tool_error(exc)


def _handle_github_review_pr(args: Dict[str, Any], **kw) -> str:
    try:
        owner, repo = parse_repo(args.get("repo", ""))
        event = str(args.get("event", "")).upper()
        if event not in {"APPROVE", "REQUEST_CHANGES", "COMMENT"}:
            return tool_error("event must be one of: APPROVE, REQUEST_CHANGES, COMMENT")
        client = _github_client()
        review = client.review_pull_request(
            owner=owner,
            repo=repo,
            number=int(args.get("number", 0)),
            event=event,
            body=args.get("body", ""),
            comments=args.get("comments"),
        )
        return tool_result(
            _with_attribution(
                client,
                {
                    "success": True,
                    "action": "review_pr",
                    "review_id": review.get("id"),
                    "state": review.get("state"),
                    "url": review.get("html_url"),
                },
            )
        )
    except Exception as exc:
        return _github_tool_error(exc)


def _handle_github_merge_pr(args: Dict[str, Any], **kw) -> str:
    try:
        owner, repo = parse_repo(args.get("repo", ""))
        method = str(args.get("method") or "squash").lower()
        if method not in {"squash", "merge", "rebase"}:
            return tool_error("method must be one of: squash, merge, rebase")
        client = _github_client()
        result = client.merge_pull_request(
            owner=owner,
            repo=repo,
            number=int(args.get("number", 0)),
            method=method,
            commit_title=args.get("commit_title", ""),
        )
        return tool_result(
            _with_attribution(
                client,
                {
                    "success": True,
                    "action": "merge_pr",
                    "merged": result.get("merged"),
                    "message": result.get("message"),
                    "sha": result.get("sha"),
                    "url": result.get("html_url"),
                },
            )
        )
    except Exception as exc:
        return _github_tool_error(exc)


_HANDLERS = {
    "github_identity": _handle_github_identity,
    "github_create_issue": _handle_github_create_issue,
    "github_comment_issue": _handle_github_comment_issue,
    "github_list_issues": _handle_github_list_issues,
    "github_get_issue": _handle_github_get_issue,
    "github_review_pr": _handle_github_review_pr,
    "github_merge_pr": _handle_github_merge_pr,
}
