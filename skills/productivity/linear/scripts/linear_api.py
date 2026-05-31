#!/usr/bin/env python3
"""Linear GraphQL API CLI — zero dependencies, stdlib only.

Usage:
  linear_api.py <command> [args...]

Commands:
  whoami                                  Show authenticated user
  list-teams                              List all teams
  list-projects [--team KEY]              List projects (optionally filter by team)
  list-states [--team KEY]                List workflow states
  list-issues [filters]                   List issues
    --team KEY                            Filter by team key (e.g. ENG)
    --status NAME                         Filter by workflow state name
    --assignee NAME                       Filter by assignee name (exact)
    --label NAME                          Filter by label name
    --limit N                             Max results (default: 25)
  get-issue <IDENTIFIER>                  Full issue details (e.g. ENG-42)
  search-issues <query>                   Full-text search across issues
  create-issue [options]                  Create a new issue
    --title TITLE                         Required
    --team KEY                            Required
    --description DESC
    --priority 0-4                        0=none, 1=urgent, 4=low
    --label NAME
    --assignee NAME
    --parent IDENTIFIER                   Parent issue ID for sub-issues
  update-issue <IDENTIFIER> [options]     Update existing issue (same options as create)
  update-status <IDENTIFIER> <STATE>      Move issue to workflow state (by state name)
  add-comment <IDENTIFIER> <body>         Add comment to issue

  list-documents [--limit N]              List documents (docs, not issues)
  get-document <SLUG_OR_ID>               Fetch a document by slugId (from URL) or UUID
  search-documents <query>                Search documents by title

  raw <graphql_query> [variables_json]    Run an arbitrary GraphQL query
                                          Use --vars '{"key":"value"}' for variables

Auth:
  Set LINEAR_API_KEY environment variable (from Linear Settings -> API).
  Uses the personal API key header format: `Authorization: <KEY>` (no Bearer prefix).

Output:
  JSON to stdout. Errors to stderr with non-zero exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ATTR_RE = re.compile(r"^attr:\s.*$", re.MULTILINE)


def _runtime_attr(args: argparse.Namespace, role: str = "builder") -> str:
    model = (getattr(args, "model", None) or os.getenv("HERMES_MODEL") or "unknown").strip()
    provider = (getattr(args, "provider", None) or os.getenv("HERMES_PROVIDER") or os.getenv("HERMES_INFERENCE_PROVIDER") or "unknown").strip()
    effort = (getattr(args, "reasoning_effort", None) or os.getenv("HERMES_REASONING_EFFORT") or "unknown").strip()
    session_id = (getattr(args, "session_id", None) or os.getenv("HERMES_SESSION_ID") or "unknown").strip()
    source = "runtime" if model != "unknown" and provider != "unknown" else "manual-unverified"
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"attr: a=Alfred; r={role}; m={model}; e={effort}; p={provider}; s={session_id}; t={timestamp}; fb=none; src={source}"


def _with_attribution(text: str | None, args: argparse.Namespace, role: str = "builder") -> str:
    body = text or ""
    body = _ATTR_RE.sub("", body).rstrip()
    if body:
        body += "\n"
    return body + _runtime_attr(args, role=role)

API_URL = "https://api.linear.app/graphql"


def _offline_receipt_dir() -> Path:
    base = os.getenv("LINEAR_OFFLINE_RECEIPT_DIR")
    if base:
        return Path(base).expanduser()
    hermes_home = Path(os.getenv("HERMES_HOME") or Path.home() / ".hermes")
    return hermes_home / "linear-offline-receipts"


def _write_offline_receipt(operation: str, payload: dict[str, Any]) -> Path:
    directory = _offline_receipt_dir()
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {
        "id": uuid.uuid4().hex,
        "operation": operation,
        "createdAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "payload": payload,
    }
    path = directory / f"{receipt['createdAt'].replace(':', '').replace('-', '')}-{receipt['id']}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _mutation_with_receipt(operation: str, query: str, variables: dict[str, Any], *, queue_on_failure: bool) -> dict[str, Any]:
    try:
        return gql(query, variables)
    except SystemExit:
        if not queue_on_failure:
            raise
        path = _write_offline_receipt(operation, {"query": query, "variables": variables})
        return {"offlineReceiptQueued": {"path": str(path), "operation": operation}}


def replay_offline_receipts(directory: Path | None = None) -> dict[str, Any]:
    directory = directory or _offline_receipt_dir()
    if not directory.exists():
        return {"replayed": 0, "failed": 0, "remaining": 0, "receipts": []}

    results: list[dict[str, Any]] = []
    replayed = 0
    failed = 0
    for path in sorted(directory.glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        payload = receipt.get("payload") or {}
        try:
            data = gql(payload["query"], payload.get("variables") or {})
        except SystemExit as exc:
            failed += 1
            results.append({"path": str(path), "ok": False, "exit_code": exc.code})
            continue
        path.unlink()
        replayed += 1
        results.append({"path": str(path), "ok": True, "data": data})
    remaining = len(list(directory.glob("*.json"))) if directory.exists() else 0
    return {"replayed": replayed, "failed": failed, "remaining": remaining, "receipts": results}


def _get_key() -> str:
    key = os.environ.get("LINEAR_API_KEY", "").strip()
    if not key:
        sys.stderr.write(
            "ERROR: LINEAR_API_KEY not set.\n"
            "Create one at https://linear.app/settings/api and export it,\n"
            "or add `LINEAR_API_KEY=lin_api_...` to ~/.hermes/.env\n"
        )
        sys.exit(2)
    return key


def gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a GraphQL query against Linear. Raises on HTTP error or GraphQL errors."""
    key = _get_key()
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": key,  # Personal API key — NO `Bearer` prefix
            "User-Agent": "hermes-agent-linear-skill/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}\n")
        sys.exit(1)
    except urllib.error.URLError as e:
        sys.stderr.write(f"Network error: {e}\n")
        sys.exit(1)

    result = json.loads(body)
    if "errors" in result and result["errors"]:
        sys.stderr.write(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}\n")
        # Still return data if partial success; let caller decide
        if not result.get("data"):
            sys.exit(1)
    return result.get("data", {}) or {}


def emit(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------- Commands ----------

def cmd_whoami(_args: argparse.Namespace) -> None:
    q = "query { viewer { id name email displayName } }"
    emit(gql(q).get("viewer"))


def cmd_list_teams(_args: argparse.Namespace) -> None:
    q = "query { teams(first: 100) { nodes { id key name description } } }"
    emit(gql(q).get("teams", {}).get("nodes", []))


def _resolve_team_id(key_or_name: str) -> str | None:
    """Map a team key (ENG) or name to UUID."""
    q = "query { teams(first: 100) { nodes { id key name } } }"
    teams = gql(q).get("teams", {}).get("nodes", [])
    kl = key_or_name.lower()
    for t in teams:
        if t["key"].lower() == kl or t["name"].lower() == kl:
            return t["id"]
    return None


def cmd_list_projects(args: argparse.Namespace) -> None:
    if args.team:
        tid = _resolve_team_id(args.team)
        if not tid:
            sys.stderr.write(f"Team not found: {args.team}\n")
            sys.exit(1)
        q = """query($id: String!) {
          team(id: $id) { projects(first: 100) { nodes { id name description state } } }
        }"""
        data = gql(q, {"id": tid})
        emit(data.get("team", {}).get("projects", {}).get("nodes", []))
    else:
        q = "query { projects(first: 100) { nodes { id name description state } } }"
        emit(gql(q).get("projects", {}).get("nodes", []))


def cmd_list_states(args: argparse.Namespace) -> None:
    if args.team:
        tid = _resolve_team_id(args.team)
        if not tid:
            sys.stderr.write(f"Team not found: {args.team}\n")
            sys.exit(1)
        q = """query($id: String!) {
          team(id: $id) { states(first: 100) { nodes { id name type color } } }
        }"""
        emit(gql(q, {"id": tid}).get("team", {}).get("states", {}).get("nodes", []))
    else:
        q = "query { workflowStates(first: 200) { nodes { id name type team { key } } } }"
        emit(gql(q).get("workflowStates", {}).get("nodes", []))


def cmd_list_issues(args: argparse.Namespace) -> None:
    filt: dict[str, Any] = {}
    if args.team:
        filt["team"] = {"key": {"eq": args.team}}
    if args.status:
        filt["state"] = {"name": {"eq": args.status}}
    if args.assignee:
        filt["assignee"] = {"name": {"eq": args.assignee}}
    if args.label:
        filt["labels"] = {"name": {"eq": args.label}}

    q = """query($filter: IssueFilter, $first: Int!) {
      issues(filter: $filter, first: $first, orderBy: updatedAt) {
        nodes {
          id identifier title
          state { name } priority
          assignee { name }
          team { key }
          updatedAt url
        }
      }
    }"""
    data = gql(q, {"filter": filt or None, "first": args.limit})
    emit(data.get("issues", {}).get("nodes", []))


def cmd_get_issue(args: argparse.Namespace) -> None:
    q = """query($id: String!) {
      issue(id: $id) {
        id identifier title description
        state { name type }
        priority priorityLabel
        assignee { name email }
        creator { name }
        team { key name }
        project { name }
        labels { nodes { name } }
        parent { identifier title }
        children { nodes { identifier title state { name } } }
        comments { nodes { user { name } body createdAt } }
        createdAt updatedAt url
      }
    }"""
    emit(gql(q, {"id": args.identifier}).get("issue"))


def cmd_search_issues(args: argparse.Namespace) -> None:
    q = """query($term: String!, $first: Int!) {
      searchIssues(term: $term, first: $first) {
        nodes { id identifier title state { name } url }
      }
    }"""
    emit(gql(q, {"term": args.query, "first": args.limit}).get("searchIssues", {}).get("nodes", []))


def cmd_create_issue(args: argparse.Namespace) -> None:
    tid = _resolve_team_id(args.team)
    if not tid:
        sys.stderr.write(f"Team not found: {args.team}\n")
        sys.exit(1)
    inp: dict[str, Any] = {"title": args.title, "teamId": tid}
    if args.description:
        inp["description"] = _with_attribution(args.description, args, role=args.role)
    if args.priority is not None:
        inp["priority"] = args.priority
    if args.parent:
        inp["parentId"] = args.parent
    # TODO: label + assignee name->id lookup (omitted for v1 brevity)

    q = """mutation($input: IssueCreateInput!) {
      issueCreate(input: $input) {
        success issue { id identifier title url }
      }
    }"""
    data = _mutation_with_receipt("create-issue", q, {"input": inp}, queue_on_failure=args.queue_offline)
    emit(data.get("issueCreate") or data.get("offlineReceiptQueued"))


def cmd_update_issue(args: argparse.Namespace) -> None:
    inp: dict[str, Any] = {}
    if args.title:
        inp["title"] = args.title
    if args.description:
        inp["description"] = _with_attribution(args.description, args, role=args.role)
    if args.priority is not None:
        inp["priority"] = args.priority
    if not inp:
        sys.stderr.write("No update fields provided.\n")
        sys.exit(1)
    q = """mutation($id: String!, $input: IssueUpdateInput!) {
      issueUpdate(id: $id, input: $input) {
        success issue { identifier title url }
      }
    }"""
    data = _mutation_with_receipt("update-issue", q, {"id": args.identifier, "input": inp}, queue_on_failure=args.queue_offline)
    emit(data.get("issueUpdate") or data.get("offlineReceiptQueued"))


def cmd_update_status(args: argparse.Namespace) -> None:
    # Resolve state name -> id within the issue's team
    get_q = """query($id: String!) {
      issue(id: $id) { team { id states(first: 100) { nodes { id name } } } }
    }"""
    issue = gql(get_q, {"id": args.identifier}).get("issue")
    if not issue:
        sys.stderr.write(f"Issue not found: {args.identifier}\n")
        sys.exit(1)
    sl = args.state.lower()
    match = next((s for s in issue["team"]["states"]["nodes"] if s["name"].lower() == sl), None)
    if not match:
        sys.stderr.write(
            f"State '{args.state}' not found. Available: "
            f"{[s['name'] for s in issue['team']['states']['nodes']]}\n"
        )
        sys.exit(1)

    q = """mutation($id: String!, $stateId: String!) {
      issueUpdate(id: $id, input: { stateId: $stateId }) {
        success issue { identifier state { name } url }
      }
    }"""
    emit(gql(q, {"id": args.identifier, "stateId": match["id"]}).get("issueUpdate"))


def cmd_add_comment(args: argparse.Namespace) -> None:
    q = """mutation($input: CommentCreateInput!) {
      commentCreate(input: $input) {
        success comment { id body createdAt }
      }
    }"""
    body = _with_attribution(args.body, args, role=args.role)
    data = _mutation_with_receipt("add-comment", q, {"input": {"issueId": args.identifier, "body": body}}, queue_on_failure=args.queue_offline)
    emit(data.get("commentCreate") or data.get("offlineReceiptQueued"))


# ---- Documents ----

def cmd_list_documents(args: argparse.Namespace) -> None:
    q = """query($first: Int!) {
      documents(first: $first, orderBy: updatedAt) {
        nodes { id title slugId updatedAt url project { name } creator { name } }
      }
    }"""
    emit(gql(q, {"first": args.limit}).get("documents", {}).get("nodes", []))


def cmd_get_document(args: argparse.Namespace) -> None:
    """Fetch a document by slugId (from URL) OR full UUID.

    Linear document URLs look like:
      https://linear.app/<workspace>/document/<slug>-<shortid>
    The part we want is the final hex segment (the slugId).
    """
    ref = args.ref
    # If it looks like a UUID, query by id. Otherwise, assume slugId.
    is_uuid = len(ref) == 36 and ref.count("-") == 4
    if is_uuid:
        q = """query($id: String!) {
          document(id: $id) {
            id title content contentState slugId
            createdAt updatedAt url
            creator { name } project { name }
          }
        }"""
        emit(gql(q, {"id": ref}).get("document"))
    else:
        # Query the collection and filter by slugId — the doc() query only accepts UUIDs.
        q = """query($slug: String!) {
          documents(filter: { slugId: { eq: $slug } }, first: 1) {
            nodes {
              id title content contentState slugId
              createdAt updatedAt url
              creator { name } project { name }
            }
          }
        }"""
        nodes = gql(q, {"slug": ref}).get("documents", {}).get("nodes", [])
        emit(nodes[0] if nodes else None)


def cmd_search_documents(args: argparse.Namespace) -> None:
    # Linear doesn't have a first-class searchDocuments — use title filter as a fallback.
    q = """query($term: String!, $first: Int!) {
      documents(filter: { title: { containsIgnoreCase: $term } }, first: $first) {
        nodes { id title slugId url updatedAt }
      }
    }"""
    emit(gql(q, {"term": args.query, "first": args.limit}).get("documents", {}).get("nodes", []))


def cmd_raw(args: argparse.Namespace) -> None:
    variables = json.loads(args.vars) if args.vars else None
    emit(gql(args.query, variables))


# ---------- Arg parsing ----------

def _add_attribution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Runtime model for compact attr footer")
    parser.add_argument("--provider", help="Runtime provider for compact attr footer")
    parser.add_argument("--reasoning-effort", dest="reasoning_effort", help="Runtime reasoning effort for compact attr footer")
    parser.add_argument("--session-id", dest="session_id", help="Runtime/session id for compact attr footer")
    parser.add_argument("--role", default="builder", help="Attribution role, e.g. builder/reviewer/verifier")


def _add_offline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queue-offline", action="store_true", help="Queue mutation receipt instead of failing when Linear is unreachable")


def cmd_replay_offline_receipts(args: argparse.Namespace) -> None:
    directory = Path(args.dir).expanduser() if args.dir else None
    emit(replay_offline_receipts(directory))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="linear_api.py", description="Linear GraphQL CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami").set_defaults(func=cmd_whoami)
    sub.add_parser("list-teams").set_defaults(func=cmd_list_teams)

    lp = sub.add_parser("list-projects")
    lp.add_argument("--team")
    lp.set_defaults(func=cmd_list_projects)

    ls = sub.add_parser("list-states")
    ls.add_argument("--team")
    ls.set_defaults(func=cmd_list_states)

    li = sub.add_parser("list-issues")
    li.add_argument("--team")
    li.add_argument("--status")
    li.add_argument("--assignee")
    li.add_argument("--label")
    li.add_argument("--limit", type=int, default=25)
    li.set_defaults(func=cmd_list_issues)

    gi = sub.add_parser("get-issue")
    gi.add_argument("identifier")
    gi.set_defaults(func=cmd_get_issue)

    si = sub.add_parser("search-issues")
    si.add_argument("query")
    si.add_argument("--limit", type=int, default=25)
    si.set_defaults(func=cmd_search_issues)

    ci = sub.add_parser("create-issue")
    ci.add_argument("--title", required=True)
    ci.add_argument("--team", required=True)
    ci.add_argument("--description")
    ci.add_argument("--priority", type=int, choices=[0, 1, 2, 3, 4])
    ci.add_argument("--label")
    ci.add_argument("--assignee")
    ci.add_argument("--parent")
    _add_attribution_args(ci)
    _add_offline_args(ci)
    ci.set_defaults(func=cmd_create_issue)

    ui = sub.add_parser("update-issue")
    ui.add_argument("identifier")
    ui.add_argument("--title")
    ui.add_argument("--description")
    ui.add_argument("--priority", type=int, choices=[0, 1, 2, 3, 4])
    _add_attribution_args(ui)
    _add_offline_args(ui)
    ui.set_defaults(func=cmd_update_issue)

    us = sub.add_parser("update-status")
    us.add_argument("identifier")
    us.add_argument("state")
    us.set_defaults(func=cmd_update_status)

    ac = sub.add_parser("add-comment")
    ac.add_argument("identifier")
    ac.add_argument("body")
    _add_attribution_args(ac)
    _add_offline_args(ac)
    ac.set_defaults(func=cmd_add_comment)

    rr = sub.add_parser("replay-offline-receipts")
    rr.add_argument("--dir", help="Receipt directory (defaults to $LINEAR_OFFLINE_RECEIPT_DIR or ~/.hermes/linear-offline-receipts)")
    rr.set_defaults(func=cmd_replay_offline_receipts)

    ld = sub.add_parser("list-documents")
    ld.add_argument("--limit", type=int, default=50)
    ld.set_defaults(func=cmd_list_documents)

    gd = sub.add_parser("get-document")
    gd.add_argument("ref", help="slugId (hex suffix from URL) or full UUID")
    gd.set_defaults(func=cmd_get_document)

    sd = sub.add_parser("search-documents")
    sd.add_argument("query")
    sd.add_argument("--limit", type=int, default=25)
    sd.set_defaults(func=cmd_search_documents)

    r = sub.add_parser("raw")
    r.add_argument("query")
    r.add_argument("--vars", help="JSON string of variables")
    r.set_defaults(func=cmd_raw)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
