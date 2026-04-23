"""GitLab MR context tools for the gitlab-review plugin.

Registers 2 context-related tools:

- gitlab_mr_context    — Fetch related issues and branch comparison
- gitlab_mr_discussions — List existing discussions/comments on an MR
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from plugins.gitlab_review.gitlab_client import (
    GitLabAPIError,
    gitlab_get,
    gitlab_get_paginated,
    is_available,
    project_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_context
# ---------------------------------------------------------------------------

GITLAB_MR_CONTEXT_SCHEMA = {
    "name": "gitlab_mr_context",
    "description": (
        "Fetch related context for a GitLab Merge Request: issues that "
        "the MR closes, and a comparison between source and target "
        "branches. Useful for understanding the full scope before "
        "reviewing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project path (e.g. 'group/project') or numeric project ID.",
            },
            "mr_iid": {
                "type": "integer",
                "description": "The MR internal ID (iid).",
            },
        },
        "required": ["project", "mr_iid"],
    },
}


def _handle_mr_context(args: dict, **kw) -> str:
    """Fetch related issues and branch comparison for an MR."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    if not project or mr_iid is None:
        return _error("Missing required parameters: project and mr_iid")

    result: Dict[str, Any] = {"closes_issues": [], "compare": None}

    try:
        # First, fetch the MR to get branch names and diff_refs
        mr_path = f"{project_path(project)}/merge_requests/{mr_iid}"
        mr_data = gitlab_get(mr_path)
        source_branch = mr_data.get("source_branch", "")
        target_branch = mr_data.get("target_branch", "")

        # Fetch issues that this MR closes
        try:
            issues_path = f"{project_path(project)}/merge_requests/{mr_iid}/closes_issues"
            issues = gitlab_get_paginated(issues_path, max_pages=3)
            result["closes_issues"] = [
                {
                    "iid": issue.get("iid"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "labels": issue.get("labels", []),
                    "web_url": issue.get("web_url"),
                }
                for issue in issues
            ]
        except GitLabAPIError as e:
            logger.debug("Could not fetch closes_issues: %s", e)

        # Fetch branch comparison
        if source_branch and target_branch:
            try:
                compare_path = (
                    f"{project_path(project)}/repository/compare"
                    f"?from={target_branch}&to={source_branch}"
                )
                compare = gitlab_get(compare_path)
                commits = compare.get("commits", [])
                result["compare"] = {
                    "from": target_branch,
                    "to": source_branch,
                    "commit_count": len(commits),
                    "commits": [
                        {
                            "id": c.get("id"),
                            "short_id": c.get("short_id"),
                            "title": c.get("title"),
                            "author_name": c.get("author_name"),
                            "created_at": c.get("created_at"),
                        }
                        for c in commits[:20]  # Cap at 20 to avoid huge output
                    ],
                }
            except GitLabAPIError as e:
                logger.debug("Could not fetch comparison: %s", e)

        return json.dumps({"result": result})

    except GitLabAPIError as e:
        return _error(f"Failed to fetch MR context: {e}")
    except Exception as e:
        logger.error("gitlab_mr_context error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_discussions
# ---------------------------------------------------------------------------

GITLAB_MR_DISCUSSIONS_SCHEMA = {
    "name": "gitlab_mr_discussions",
    "description": (
        "List existing discussions (comment threads) on a GitLab Merge "
        "Request. Use this before reviewing to avoid duplicating feedback "
        "that was already provided."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project path (e.g. 'group/project') or numeric project ID.",
            },
            "mr_iid": {
                "type": "integer",
                "description": "The MR internal ID (iid).",
            },
        },
        "required": ["project", "mr_iid"],
    },
}


def _handle_mr_discussions(args: dict, **kw) -> str:
    """List discussions on an MR."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    if not project or mr_iid is None:
        return _error("Missing required parameters: project and mr_iid")

    try:
        path = f"{project_path(project)}/merge_requests/{mr_iid}/discussions"
        items = gitlab_get_paginated(path, max_pages=5)

        discussions = []
        for d in items:
            notes = d.get("notes", [])
            discussion_entry = {
                "id": d.get("id"),
                "noteable_type": d.get("noteable_type"),
                "notes": [
                    {
                        "id": n.get("id"),
                        "type": n.get("type"),
                        "body": n.get("body", ""),
                        "author": n.get("author", {}).get("username", ""),
                        "created_at": n.get("created_at"),
                        "resolvable": n.get("resolvable", False),
                        "resolved": n.get("resolved", False),
                    }
                    for n in notes
                ],
            }
            # Include position info for inline comments
            for n in notes:
                if n.get("position"):
                    discussion_entry["position"] = {
                        "new_path": n["position"].get("new_path"),
                        "new_line": n["position"].get("new_line"),
                        "old_path": n["position"].get("old_path"),
                        "old_line": n["position"].get("old_line"),
                    }
                    break

            discussions.append(discussion_entry)

        return json.dumps({
            "result": {
                "count": len(discussions),
                "discussions": discussions,
            },
        })
    except GitLabAPIError as e:
        return _error(f"Failed to fetch discussions: {e}")
    except Exception as e:
        logger.error("gitlab_mr_discussions error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(msg: str) -> str:
    """Return a JSON error result."""
    return json.dumps({"error": msg})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ALL_CONTEXT_SCHEMAS = [
    (GITLAB_MR_CONTEXT_SCHEMA, _handle_mr_context, "🔗"),
    (GITLAB_MR_DISCUSSIONS_SCHEMA, _handle_mr_discussions, "💭"),
]


def register_context_tools(ctx) -> None:
    """Register all context-related tools with the plugin context."""
    for schema, handler, emoji in ALL_CONTEXT_SCHEMAS:
        ctx.register_tool(
            name=schema["name"],
            toolset="gitlab_review",
            schema=schema,
            handler=handler,
            check_fn=is_available,
            requires_env=["GITLAB_TOKEN"],
            emoji=emoji,
        )
