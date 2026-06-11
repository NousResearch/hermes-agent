"""Secret-safe Linear GraphQL client and native tool handlers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

API_URL = "https://api.linear.app/graphql"
_UNSUPPORTED_LINK_REASON = "Linear relation/link mutation shape deferred for a later gate."


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _error(message: str) -> str:
    # Deliberately compact and sanitized: no headers, token values, token length,
    # env paths, config paths, logs, sessions, cookies, cache, or database detail.
    return _json({"ok": False, "error": message})


def _require_str(args: dict[str, Any], key: str) -> str:
    value = str(args.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def _limit(args: dict[str, Any], default: int = 10, maximum: int = 25) -> int:
    try:
        value = int(args.get("limit", default))
    except (TypeError, ValueError):
        value = default
    return max(1, min(maximum, value))


def gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a Linear GraphQL operation.

    The API key is used only as the Authorization header and is never returned.
    Raised errors are sanitized by the public tool handlers.
    """
    key = os.environ.get("LINEAR_API_KEY", "")
    if not key:
        raise RuntimeError("Linear API key unavailable")

    body = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": key,
            "User-Agent": "hermes-agent-linear-tools/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Do not include body: Linear/API errors may echo request details.
        raise RuntimeError(f"Linear API request failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Linear API request failed") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Linear API returned invalid JSON") from exc

    if payload.get("errors"):
        raise RuntimeError("Linear API returned GraphQL errors")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Linear API returned no data")
    return data


def _compact_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    compact = {"name": state.get("name")}
    if state.get("type") is not None:
        compact["type"] = state.get("type")
    return {k: v for k, v in compact.items() if v is not None}


def _compact_team(team: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(team, dict):
        return {}
    return {k: team.get(k) for k in ("key", "name") if team.get(k) is not None}


def _compact_issue(issue: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(issue, dict):
        return {}
    compact = {
        "id": issue.get("id"),
        "identifier": issue.get("identifier"),
        "title": issue.get("title"),
        "url": issue.get("url"),
        "state": _compact_state(issue.get("state")),
        "team": _compact_team(issue.get("team")),
    }
    return {k: v for k, v in compact.items() if v not in (None, {}, [])}


ISSUE_QUERY = """query($id: String!) {
  issue(id: $id) {
    id identifier title url
    state { name type }
    team { id key name }
  }
}"""

ISSUE_WITH_COMMENTS_QUERY = """query($id: String!) {
  issue(id: $id) {
    id identifier title url
    state { name type }
    team { id key name }
    comments(first: 100) { nodes { id body createdAt } }
  }
}"""


def _get_issue(identifier: str, *, comments: bool = False) -> dict[str, Any]:
    query = ISSUE_WITH_COMMENTS_QUERY if comments else ISSUE_QUERY
    issue = gql(query, {"id": identifier}).get("issue")
    if not isinstance(issue, dict):
        raise RuntimeError("Linear issue not found")
    return issue


def handle_get_issue(args: dict[str, Any], **_: Any) -> str:
    try:
        identifier = _require_str(args, "identifier")
        return _json({"ok": True, "issue": _compact_issue(_get_issue(identifier))})
    except Exception as exc:
        return _error(str(exc))


def handle_search_issues(args: dict[str, Any], **_: Any) -> str:
    try:
        query = _require_str(args, "query")
        first = _limit(args)
        data = gql(
            """query($term: String!, $first: Int!) {
              searchIssues(term: $term, first: $first) {
                nodes { id identifier title url state { name type } team { id key name } }
              }
            }""",
            {"term": query, "first": first},
        )
        nodes = data.get("searchIssues", {}).get("nodes", [])
        return _json({"ok": True, "issues": [_compact_issue(node) for node in nodes]})
    except Exception as exc:
        return _error(str(exc))


def _create_comment(issue_id: str, body: str) -> dict[str, Any]:
    data = gql(
        """mutation($input: CommentCreateInput!) {
          commentCreate(input: $input) {
            success comment { id createdAt }
          }
        }""",
        {"input": {"issueId": issue_id, "body": body}},
    )
    payload = data.get("commentCreate", {})
    if not payload.get("success"):
        raise RuntimeError("Linear comment create failed")
    comment = payload.get("comment")
    if not isinstance(comment, dict):
        raise RuntimeError("Linear comment create returned no comment")
    return {"id": comment.get("id"), "createdAt": comment.get("createdAt")}


def handle_add_comment(args: dict[str, Any], **_: Any) -> str:
    try:
        identifier = _require_str(args, "identifier")
        body = _require_str(args, "body")
        issue = _get_issue(identifier)
        comment = _create_comment(issue["id"], body)
        return _json({"ok": True, "issue": _compact_issue(issue), "comment": comment})
    except Exception as exc:
        return _error(str(exc))


def handle_ensure_comment(args: dict[str, Any], **_: Any) -> str:
    try:
        identifier = _require_str(args, "identifier")
        body = _require_str(args, "body")
        issue = _get_issue(identifier, comments=True)
        comments = issue.get("comments", {}).get("nodes", [])
        for comment in comments:
            if isinstance(comment, dict) and comment.get("body") == body:
                return _json(
                    {
                        "ok": True,
                        "created": False,
                        "issue": _compact_issue(issue),
                        "comment": {
                            "id": comment.get("id"),
                            "createdAt": comment.get("createdAt"),
                        },
                    }
                )
        comment = _create_comment(issue["id"], body)
        return _json({"ok": True, "created": True, "issue": _compact_issue(issue), "comment": comment})
    except Exception as exc:
        return _error(str(exc))


def _workflow_states_for_team(team_id: str) -> list[dict[str, Any]]:
    data = gql(
        """query($teamId: ID!) {
          workflowStates(filter: { team: { id: { eq: $teamId } } }, first: 100) {
            nodes { id name type }
          }
        }""",
        {"teamId": team_id},
    )
    nodes = data.get("workflowStates", {}).get("nodes", [])
    return [node for node in nodes if isinstance(node, dict)]


def handle_update_status(args: dict[str, Any], **_: Any) -> str:
    try:
        identifier = _require_str(args, "identifier")
        state_name = _require_str(args, "state")
        issue = _get_issue(identifier)
        team = issue.get("team") or {}
        team_id = team.get("id")
        if not team_id:
            raise RuntimeError("Linear issue has no team id")
        matches = [s for s in _workflow_states_for_team(team_id) if s.get("name") == state_name]
        if len(matches) != 1:
            raise RuntimeError("Linear workflow state not found or ambiguous")
        before = _compact_state(issue.get("state"))
        data = gql(
            """mutation($id: String!, $input: IssueUpdateInput!) {
              issueUpdate(id: $id, input: $input) {
                success issue { id identifier title url state { name type } team { id key name } }
              }
            }""",
            {"id": issue["id"], "input": {"stateId": matches[0]["id"]}},
        )
        payload = data.get("issueUpdate", {})
        if not payload.get("success"):
            raise RuntimeError("Linear issue update failed")
        updated = payload.get("issue")
        return _json({"ok": True, "issue": _compact_issue(updated), "before": before, "after": _compact_state(updated.get("state") if isinstance(updated, dict) else None)})
    except Exception as exc:
        return _error(str(exc))


def _resolve_team_id(team_key: str) -> str:
    data = gql("query { teams(first: 100) { nodes { id key name } } }")
    for team in data.get("teams", {}).get("nodes", []):
        if isinstance(team, dict) and team.get("key") == team_key:
            return team["id"]
    raise RuntimeError("Linear team not found")


def handle_create_issue(args: dict[str, Any], **_: Any) -> str:
    try:
        team_key = _require_str(args, "team")
        title = _require_str(args, "title")
        issue_input: dict[str, Any] = {"teamId": _resolve_team_id(team_key), "title": title}
        description = str(args.get("description", "")).strip()
        if description:
            issue_input["description"] = description
        if args.get("priority") is not None:
            issue_input["priority"] = int(args["priority"])
        parent = str(args.get("parent", "")).strip()
        if parent:
            issue_input["parentId"] = parent
        data = gql(
            """mutation($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success issue { id identifier title url state { name type } team { id key name } }
              }
            }""",
            {"input": issue_input},
        )
        payload = data.get("issueCreate", {})
        if not payload.get("success"):
            raise RuntimeError("Linear issue create failed")
        return _json({"ok": True, "issue": _compact_issue(payload.get("issue"))})
    except Exception as exc:
        return _error(str(exc))


def handle_link_issues(args: dict[str, Any], **_: Any) -> str:
    return _json({"ok": False, "unsupported": True, "reason": _UNSUPPORTED_LINK_REASON})
