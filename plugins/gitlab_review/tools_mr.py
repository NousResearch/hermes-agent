"""GitLab Merge Request tools for the gitlab-review plugin.

Registers 7 MR-related tools:

- gitlab_mr_view        — Fetch MR metadata
- gitlab_mr_diff        — Get full diff of an MR
- gitlab_mr_list_files  — List changed files with stats
- gitlab_mr_comments    — Post a general comment on an MR
- gitlab_mr_inline_comment — Post an inline comment on a specific line
- gitlab_mr_review      — Submit a formal review (approve / request changes / comment)
- gitlab_mr_list        — List open MRs with optional filters
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from plugins.gitlab_review.gitlab_client import (
    GitLabAPIError,
    gitlab_delete,
    gitlab_get,
    gitlab_get_paginated,
    gitlab_post,
    is_available,
    project_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_view
# ---------------------------------------------------------------------------

GITLAB_MR_VIEW_SCHEMA = {
    "name": "gitlab_mr_view",
    "description": (
        "Fetch metadata for a GitLab Merge Request: title, description, "
        "author, source/target branch, state, labels, milestones, and "
        "approval status. Requires GITLAB_TOKEN env var."
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
                "description": "The MR internal ID (iid, not global id).",
            },
        },
        "required": ["project", "mr_iid"],
    },
}


def _handle_mr_view(args: dict, **kw) -> str:
    """Fetch and summarize MR metadata."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    if not project or mr_iid is None:
        return _error("Missing required parameters: project and mr_iid")

    try:
        path = f"{project_path(project)}/merge_requests/{mr_iid}"
        data = gitlab_get(path)
        result = {
            "iid": data.get("iid"),
            "title": data.get("title"),
            "description": data.get("description", ""),
            "state": data.get("state"),
            "author": data.get("author", {}).get("username", ""),
            "source_branch": data.get("source_branch"),
            "target_branch": data.get("target_branch"),
            "labels": data.get("labels", []),
            "milestone": (data.get("milestone") or {}).get("title"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "merged_at": data.get("merged_at"),
            "web_url": data.get("web_url"),
            "draft": data.get("draft", False),
            "merge_status": data.get("merge_status"),
            "detailed_merge_status": data.get("detailed_merge_status"),
            "user_notes_count": data.get("user_notes_count"),
        }
        return json.dumps({"result": result})
    except GitLabAPIError as e:
        return _error(f"Failed to fetch MR: {e}")
    except Exception as e:
        logger.error("gitlab_mr_view error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_diff
# ---------------------------------------------------------------------------

GITLAB_MR_DIFF_SCHEMA = {
    "name": "gitlab_mr_diff",
    "description": (
        "Get the full diff of a GitLab Merge Request. Returns the unified diff "
        "for all changed files. For large MRs, use gitlab_mr_list_files first "
        "to see the scope, then request the diff."
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


def _handle_mr_diff(args: dict, **kw) -> str:
    """Fetch MR diff (changes endpoint returns diffs in each file entry)."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    if not project or mr_iid is None:
        return _error("Missing required parameters: project and mr_iid")

    try:
        path = f"{project_path(project)}/merge_requests/{mr_iid}/changes"
        data = gitlab_get(path)
        changes = data.get("changes", [])
        diffs = []
        for change in changes:
            diffs.append({
                "old_path": change.get("old_path"),
                "new_path": change.get("new_path"),
                "diff": change.get("diff", ""),
            })
        return json.dumps({
            "result": {
                "source_branch": data.get("source_branch"),
                "target_branch": data.get("target_branch"),
                "diffs": diffs,
            },
        })
    except GitLabAPIError as e:
        return _error(f"Failed to fetch MR diff: {e}")
    except Exception as e:
        logger.error("gitlab_mr_diff error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_list_files
# ---------------------------------------------------------------------------

GITLAB_MR_LIST_FILES_SCHEMA = {
    "name": "gitlab_mr_list_files",
    "description": (
        "List changed files in a GitLab Merge Request with additions/deletions "
        "stats per file. Use this to understand the scope of an MR before "
        "reading the full diff."
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


def _handle_mr_list_files(args: dict, **kw) -> str:
    """List changed files with stats."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    if not project or mr_iid is None:
        return _error("Missing required parameters: project and mr_iid")

    try:
        path = f"{project_path(project)}/merge_requests/{mr_iid}/changes"
        data = gitlab_get(path)
        changes = data.get("changes", [])
        files = []
        for change in changes:
            files.append({
                "old_path": change.get("old_path"),
                "new_path": change.get("new_path"),
                "new_file": change.get("new_file", False),
                "renamed_file": change.get("renamed_file", False),
                "deleted_file": change.get("deleted_file", False),
            })
        return json.dumps({
            "result": {
                "source_branch": data.get("source_branch"),
                "target_branch": data.get("target_branch"),
                "files": files,
                "total_changes": len(files),
            },
        })
    except GitLabAPIError as e:
        return _error(f"Failed to list MR files: {e}")
    except Exception as e:
        logger.error("gitlab_mr_list_files error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_comments
# ---------------------------------------------------------------------------

GITLAB_MR_COMMENTS_SCHEMA = {
    "name": "gitlab_mr_comments",
    "description": (
        "Post a general (top-level) comment on a GitLab Merge Request. "
        "Use for overall feedback or summary reviews."
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
            "body": {
                "type": "string",
                "description": "Comment text. Markdown is supported.",
            },
        },
        "required": ["project", "mr_iid", "body"],
    },
}


def _handle_mr_comments(args: dict, **kw) -> str:
    """Post a general comment on an MR."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    body = args.get("body", "")
    if not project or mr_iid is None or not body:
        return _error("Missing required parameters: project, mr_iid, and body")

    try:
        path = f"{project_path(project)}/merge_requests/{mr_iid}/notes"
        result = gitlab_post(path, json_body={"body": body})
        return json.dumps({
            "result": {
                "id": result.get("id"),
                "noteable_iid": result.get("noteable_iid"),
                "created_at": result.get("created_at"),
            },
        })
    except GitLabAPIError as e:
        return _error(f"Failed to post comment: {e}")
    except Exception as e:
        logger.error("gitlab_mr_comments error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_inline_comment
# ---------------------------------------------------------------------------

GITLAB_MR_INLINE_COMMENT_SCHEMA = {
    "name": "gitlab_mr_inline_comment",
    "description": (
        "Post an inline comment on a specific line of a changed file in a "
        "GitLab Merge Request. Creates a new discussion thread. Requires "
        "the head commit SHA (available from gitlab_mr_view output)."
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
            "file_path": {
                "type": "string",
                "description": "Path of the file to comment on (e.g. 'src/auth/login.py').",
            },
            "line": {
                "type": "integer",
                "description": "Line number in the new version of the file to comment on.",
            },
            "body": {
                "type": "string",
                "description": "Comment text. Markdown is supported.",
            },
            "head_sha": {
                "type": "string",
                "description": "SHA of the head commit. Required by GitLab for position. Get from gitlab_mr_view.",
            },
            "base_sha": {
                "type": "string",
                "description": "SHA of the base commit. Optional — GitLab can resolve it if omitted.",
            },
            "start_sha": {
                "type": "string",
                "description": "SHA of the start commit (branch point). Optional — GitLab can resolve it if omitted.",
            },
            "line_type": {
                "type": "string",
                "description": "'new' (default) for added/modified lines, 'old' for deleted lines.",
                "enum": ["new", "old"],
            },
        },
        "required": ["project", "mr_iid", "file_path", "line", "body", "head_sha"],
    },
}


def _handle_mr_inline_comment(args: dict, **kw) -> str:
    """Post an inline comment on a specific line."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    file_path = args.get("file_path", "")
    line = args.get("line")
    body = args.get("body", "")
    head_sha = args.get("head_sha", "")
    base_sha = args.get("base_sha", "")
    start_sha = args.get("start_sha", "")
    line_type = args.get("line_type", "new")

    if not all([project, mr_iid is not None, file_path, line is not None, body, head_sha]):
        return _error("Missing required parameters: project, mr_iid, file_path, line, body, head_sha")

    # If base_sha and start_sha are not provided, fetch them from the MR
    if not base_sha or not start_sha:
        try:
            mr_path = f"{project_path(project)}/merge_requests/{mr_iid}"
            mr_data = gitlab_get(mr_path)
            if not base_sha:
                base_sha = mr_data.get("diff_refs", {}).get("base_sha", "")
            if not start_sha:
                start_sha = mr_data.get("diff_refs", {}).get("start_sha", "")
        except GitLabAPIError:
            pass  # Try with what we have

    if not base_sha or not start_sha:
        return _error(
            "Cannot resolve base_sha/start_sha. Provide them explicitly or "
            "ensure the MR has diff_refs (may be empty for merged MRs)."
        )

    position = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "start_sha": start_sha,
        "position_type": "text",
        "new_path": file_path,
        "new_line": line if line_type == "new" else None,
        "old_path": file_path,
        "old_line": line if line_type == "old" else None,
    }
    # Remove None values — GitLab rejects null position fields
    position = {k: v for k, v in position.items() if v is not None}

    try:
        path = f"{project_path(project)}/merge_requests/{mr_iid}/discussions"
        result = gitlab_post(path, json_body={
            "body": body,
            "position": position,
        })
        return json.dumps({
            "result": {
                "id": result.get("id"),
                "notes": [
                    {"id": n.get("id"), "type": n.get("type")}
                    for n in result.get("notes", [])
                ],
            },
        })
    except GitLabAPIError as e:
        return _error(f"Failed to post inline comment: {e}")
    except Exception as e:
        logger.error("gitlab_mr_inline_comment error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_review
# ---------------------------------------------------------------------------

GITLAB_MR_REVIEW_SCHEMA = {
    "name": "gitlab_mr_review",
    "description": (
        "Submit a formal review on a GitLab Merge Request. Actions: "
        "'approve' — approve the MR; 'request_changes' — unapprove "
        "(or leave a comment requesting changes); 'comment' — leave a "
        "general comment without changing approval status."
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
            "action": {
                "type": "string",
                "description": "Review action: 'approve', 'request_changes', or 'comment'.",
                "enum": ["approve", "request_changes", "comment"],
            },
            "body": {
                "type": "string",
                "description": "Review comment text. Markdown supported. Used for all actions.",
            },
        },
        "required": ["project", "mr_iid", "action"],
    },
}


def _handle_mr_review(args: dict, **kw) -> str:
    """Submit a formal review (approve / request changes / comment)."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    action = args.get("action", "")
    body = args.get("body", "")

    if not project or mr_iid is None or not action:
        return _error("Missing required parameters: project, mr_iid, and action")

    try:
        results = {}

        # Post the comment first (if provided)
        if body:
            note_path = f"{project_path(project)}/merge_requests/{mr_iid}/notes"
            note_result = gitlab_post(note_path, json_body={"body": body})
            results["comment"] = {
                "id": note_result.get("id"),
                "created_at": note_result.get("created_at"),
            }

        # Then apply the action
        if action == "approve":
            approve_path = f"{project_path(project)}/merge_requests/{mr_iid}/approve"
            approve_result = gitlab_post(approve_path, json_body={"sha": ""} if not body else {})
            results["approval"] = {
                "state": "approved",
                "approved_by": approve_result.get("approved_by", []),
            }
        elif action == "request_changes":
            # GitLab doesn't have a native "request changes" action.
            # We unapprove if currently approved, and post a comment.
            unapprove_path = f"{project_path(project)}/merge_requests/{mr_iid}/approvals"
            try:
                gitlab_delete(unapprove_path)
                results["approval"] = {"state": "unapproved"}
            except GitLabAPIError:
                # May not be approved — that's fine
                results["approval"] = {"state": "not_previously_approved"}
        # 'comment' action only posts the note (already done above)

        results["action"] = action
        return json.dumps({"result": results})

    except GitLabAPIError as e:
        return _error(f"Failed to submit review: {e}")
    except Exception as e:
        logger.error("gitlab_mr_review error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_list
# ---------------------------------------------------------------------------

GITLAB_MR_LIST_SCHEMA = {
    "name": "gitlab_mr_list",
    "description": (
        "List merge requests in a GitLab project with optional filters. "
        "Returns a summary of each MR: iid, title, author, state, labels."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project path (e.g. 'group/project') or numeric project ID.",
            },
            "state": {
                "type": "string",
                "description": "Filter by MR state: 'opened', 'closed', 'merged', 'all'. Default: 'opened'.",
                "enum": ["opened", "closed", "merged", "all"],
            },
            "labels": {
                "type": "string",
                "description": "Comma-separated label names to filter by (e.g. 'bug,review-needed').",
            },
            "author_username": {
                "type": "string",
                "description": "Filter by author username.",
            },
            "milestone": {
                "type": "string",
                "description": "Filter by milestone title.",
            },
            "search": {
                "type": "string",
                "description": "Search MR titles and descriptions.",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum pages to fetch (default: 3). Each page returns up to 100 MRs.",
            },
        },
        "required": ["project"],
    },
}


def _handle_mr_list(args: dict, **kw) -> str:
    """List MRs with optional filters."""
    project = args.get("project", "")
    if not project:
        return _error("Missing required parameter: project")

    params: Dict[str, Any] = {}
    if args.get("state"):
        params["state"] = args["state"]
    if args.get("labels"):
        params["labels"] = args["labels"]
    if args.get("author_username"):
        params["author_username"] = args["author_username"]
    if args.get("milestone"):
        params["milestone"] = args["milestone"]
    if args.get("search"):
        params["search"] = args["search"]

    max_pages = min(int(args.get("max_pages", 3)), 10)

    try:
        path = f"{project_path(project)}/merge_requests"
        items = gitlab_get_paginated(path, params=params, max_pages=max_pages)

        mrs = []
        for mr in items:
            mrs.append({
                "iid": mr.get("iid"),
                "title": mr.get("title"),
                "author": mr.get("author", {}).get("username", ""),
                "state": mr.get("state"),
                "labels": mr.get("labels", []),
                "draft": mr.get("draft", False),
                "web_url": mr.get("web_url"),
                "updated_at": mr.get("updated_at"),
            })

        return json.dumps({"result": {"count": len(mrs), "merge_requests": mrs}})
    except GitLabAPIError as e:
        return _error(f"Failed to list MRs: {e}")
    except Exception as e:
        logger.error("gitlab_mr_list error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(msg: str) -> str:
    """Return a JSON error result consistent with tool_error format."""
    return json.dumps({"error": msg})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ALL_MR_SCHEMAS = [
    (GITLAB_MR_VIEW_SCHEMA, _handle_mr_view, "🔍"),
    (GITLAB_MR_DIFF_SCHEMA, _handle_mr_diff, "📝"),
    (GITLAB_MR_LIST_FILES_SCHEMA, _handle_mr_list_files, "📁"),
    (GITLAB_MR_COMMENTS_SCHEMA, _handle_mr_comments, "💬"),
    (GITLAB_MR_INLINE_COMMENT_SCHEMA, _handle_mr_inline_comment, "📍"),
    (GITLAB_MR_REVIEW_SCHEMA, _handle_mr_review, "✅"),
    (GITLAB_MR_LIST_SCHEMA, _handle_mr_list, "📋"),
]


def register_mr_tools(ctx) -> None:
    """Register all MR-related tools with the plugin context."""
    for schema, handler, emoji in ALL_MR_SCHEMAS:
        ctx.register_tool(
            name=schema["name"],
            toolset="gitlab_review",
            schema=schema,
            handler=handler,
            check_fn=is_available,
            requires_env=["GITLAB_TOKEN"],
            emoji=emoji,
        )
