#!/usr/bin/env python3
"""Authenticated HTTP client for the Commander dashboard API.

Named subcommands cover the handful of reads an agent reaches for daily.
Everything else in Commander's ~155-route surface is reachable through the
generic `call` subcommand, backed by references/endpoints.md (or the live
`spec` subcommand, which dumps Commander's own /openapi.json).

Safety: every non-GET request is refused unless --confirm is passed, and
routes on the HIGH_RISK list print a loud banner even with --confirm. This
script never decides "is it OK to do this" on its own — the agent must have
gotten explicit, specific confirmation from the user in chat first; --confirm
only records that the agent did so. No implicit retries on mutating calls.
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

MAX_LIST_ITEMS = 15
MAX_BODY_CHARS = 6000

# Path substrings where a mismatched confirmation would be expensive or
# irreversible: spawning paid agent runs, deleting data, merging branches,
# deploying. Simple substring checks, not full pattern matching — false
# positives (warn when unneeded) are fine; false negatives are not.
HIGH_RISK_MARKERS = [
    "/sprints/run",
    "/sprint-run",
    "/bulk-complete",
    "/finish",
    "/sprint-branch-merge",
    "/resolve-branch-conflict",
    "/deploy",
    "/environments/",  # covers start/stop/restart/deploy under an env
    "/maintenance/",
]

# name -> (path_template, path_params, query_params[(name, required)], help)
SHORTCUTS = {
    "health": ("/api/health", [], [], "Overall health snapshot"),
    "board": ("/api/board", [], [("project", True)], "Kanban board data"),
    "running": ("/api/running", [], [("project", True)], "Currently-running agents/jobs"),
    "sprints": ("/api/sprints", [], [], "List sprints"),
    "sprint_state": (
        "/api/sprints/{sprint_label}/state",
        ["sprint_label"],
        [("project", True)],
        "Sprint state snapshot",
    ),
    "sprint_progress": (
        "/api/sprint-progress",
        [],
        [("project", False), ("repo", False)],
        "Sprint progress bar data",
    ),
    "issues": ("/api/issues", [], [("sprint", False)], "List issues"),
    "running_sprint": (
        "/api/projects/{project}/running-sprint",
        ["project"],
        [],
        "Currently running sprint for a project (dispatch-guard check)",
    ),
    "todos": ("/api/projects/{project}/todos", ["project"], [], "List todos for a project"),
    "advisor_suggestions": (
        "/api/projects/{project}/advisor/suggestions",
        ["project"],
        [],
        "Pre-computed advisor suggestions for a project",
    ),
    "mis_sizing_flags": (
        "/api/sprints/{sprint_label}/mis-sizing-flags",
        ["sprint_label"],
        [("project", True)],
        "Pre-computed mis-sizing flags for a sprint",
    ),
    "preflight": (
        "/api/sprints/{sprint_label}/preflight",
        ["sprint_label"],
        [("project", True)],
        "Full preflight report before dispatching a sprint",
    ),
    "home": ("/api/home", [], [], "Cross-project rollup: running/awaiting-UAT/backlog counts"),
    "nav_status": (
        "/api/sprint-nav-status",
        [],
        [("repo", True)],  # API marks this optional but omitting it silently
        # returns an arbitrary project's data instead of erroring — require
        # it here so that footgun can't happen through this script.
        "Current sprint + ticket-column breakdown for one project (repo must be full 'owner/repo')",
    ),
    "rerun_preview": (
        "/api/sprints/{sprint_label}/rerun-preview",
        ["sprint_label"],
        [("project", True)],
        "Preview what a re-run of a sprint would do (SAFE, no side effects)",
    ),
    "milestones": (
        "/api/projects/{slug}/milestones",
        ["slug"],
        [],
        "List milestones + which one is active (bare repo name) — check before plan_next",
    ),
}


def _confirm_gate(method, path, confirmed):
    method = method.upper()
    if method in ("GET", "HEAD"):
        return
    if not confirmed:
        print(
            f"REFUSED: {method} {path} is a mutating call. Pass --confirm "
            "only after the user has explicitly approved this exact action "
            "in chat.",
            file=sys.stderr,
        )
        sys.exit(2)
    is_high_risk = method == "DELETE" or any(m in path for m in HIGH_RISK_MARKERS)
    if is_high_risk:
        print(
            f"!!! HIGH-RISK CALL: {method} {path} !!!\n"
            "This can spawn paid agent runs, merge/delete code, or deploy. "
            "Proceeding because --confirm was passed — this assumes the "
            "user approved *this specific action*, not a generic 'go ahead'.",
            file=sys.stderr,
        )


def _shrink(obj):
    """Cap list length and overall size so one call can't blow the context."""
    if isinstance(obj, list):
        shrunk = [_shrink(item) for item in obj[:MAX_LIST_ITEMS]]
        if len(obj) > MAX_LIST_ITEMS:
            shrunk.append(f"... ({len(obj) - MAX_LIST_ITEMS} more items truncated)")
        return shrunk
    if isinstance(obj, dict):
        return {k: _shrink(v) for k, v in obj.items()}
    return obj


def _fetch_raw(base_url, token, method, path, body=None, query=None, timeout=20):
    """Issue the HTTP call and return (status, parsed_body) with no shrinking."""
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        status = e.code
    except urllib.error.URLError as e:
        return None, {"error": f"connection failed: {e.reason}", "url": url}

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {"raw": raw[:MAX_BODY_CHARS]}
    return status, parsed


def request(base_url, token, method, path, body=None, query=None, timeout=20):
    status, parsed = _fetch_raw(base_url, token, method, path, body, query, timeout)
    if status is None:
        return parsed  # connection error dict
    result = {"status": status, "body": _shrink(parsed)}
    if len(json.dumps(result)) > MAX_BODY_CHARS:
        result["body"] = {"truncated": True, "preview": str(parsed)[:MAX_BODY_CHARS]}
    return result


def status_overview(base_url, token):
    """One-shot cross-project sprint status, sourced entirely from
    GET /api/home's `projects` array — the one place Commander's own
    dashboard computes a live, authoritative per-project `status`
    (idle/uat-pending/running) plus the real project-wide backlog count.

    Earlier versions of this command used /api/sprint-nav-status per
    project instead. Don't go back to that: its `state` field is derived
    from GitHub issue labels ("has a Sprint N Executive Summary issue been
    posted"), cached 30s, and answers "has this sprint been formally
    closed out" — NOT "is an agent actively working right now". It can
    read "running" for a project that finished its actual work hours ago
    and just hasn't been through the finish step yet, which reads as a
    false positive against the real running state /api/home reports.
    Similarly nav-status's `columns.backlog` is the tiny in-sprint kanban
    backlog column (0-2 tickets), not the project's real untriaged
    backlog pool (dozens) — /api/home's `backlog_count` is the right one.
    """
    status, home_body = _fetch_raw(base_url, token, "GET", "/api/home")
    if status != 200:
        return {"error": "could not fetch /api/home", "status": status, "body": home_body}

    results = []
    for proj in home_body.get("projects", []):
        entry = {
            "repo": proj.get("repo"),
            "status": proj.get("status"),  # "idle" | "uat-pending" | "running"
            "uat_count": proj.get("uat_count"),
            "backlog_open": proj.get("backlog_count"),
            "last_sprint_num": (proj.get("last_sprint") or {}).get("sprint_num"),
        }
        sprint_running = proj.get("sprint_running")
        if sprint_running:
            entry["running_sprint_label"] = sprint_running.get("label")
            entry["running_elapsed_sec"] = sprint_running.get("elapsed_sec")
        results.append(entry)
    return {"projects": results}


def stream(base_url, token, path, max_seconds):
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    lines = []
    # One connection-level timeout, one read attempt at a time, no retry
    # after a timeout: CPython's http.client marks the underlying socket
    # file object unusable after its first read timeout ("cannot read from
    # timed out object"), so retrying on the same connection just raises
    # again. A single bounded pass keeps this safe for an idle SSE stream
    # in an unattended cron run without needing that retry.
    try:
        with urllib.request.urlopen(req, timeout=max_seconds) as resp:
            for raw_line in resp:
                lines.append(raw_line.decode(errors="replace").rstrip())
                if len(lines) >= 200:
                    break
    except urllib.error.URLError as e:
        return {"error": f"stream failed: {e.reason}", "url": url, "lines": lines}
    except OSError as e:
        # Read timed out mid-stream (idle SSE) — return whatever arrived.
        return {"lines": lines, "capped_at_seconds": max_seconds, "note": str(e)}
    return {"lines": lines, "capped_at_seconds": max_seconds}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000, help="8000=prd, 8001=uat")
    p.add_argument("--token", default="", help="COMMANDER_API_TOKEN, or blank")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, (path_template, path_params, query_params, help_text) in SHORTCUTS.items():
        sp = sub.add_parser(name, help=help_text)
        for param in path_params:
            sp.add_argument(param)
        for qname, required in query_params:
            sp.add_argument(f"--{qname}", required=required, default=None)

    sub.add_parser(
        "status",
        help="Cross-project sprint status in one call (running/finished/UAT counts per project)",
    )

    sp = sub.add_parser("spec", help="Dump Commander's live OpenAPI schema")
    sp.add_argument("--path", default="", help="Optional path filter substring")

    sp = sub.add_parser("stream", help="Read an SSE endpoint for a capped duration")
    sp.add_argument("path")
    sp.add_argument("--max-seconds", type=int, default=20)

    sp = sub.add_parser("call", help="Generic request to any Commander route")
    sp.add_argument("method")
    sp.add_argument("path")
    sp.add_argument("--json", dest="json_body", default=None, help="JSON request body")
    sp.add_argument(
        "--confirm",
        action="store_true",
        help="Required for any non-GET call; only pass after explicit user approval",
    )

    args = p.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    if args.cmd == "status":
        print(json.dumps(status_overview(base_url, args.token), indent=2))
        return

    if args.cmd == "spec":
        status, parsed = _fetch_raw(base_url, args.token, "GET", "/openapi.json")
        if status is None:
            print(json.dumps(parsed, indent=2))
            return
        if isinstance(parsed, dict):
            # Drop $ref schema definitions — bulky and rarely needed for a
            # basic method/path/param lookup, which is what this is for.
            parsed = {k: v for k, v in parsed.items() if k != "components"}
            if args.path:
                parsed["paths"] = {
                    k: v for k, v in parsed.get("paths", {}).items() if args.path in k
                }
        result = {"status": status, "body": _shrink(parsed)}
        if len(json.dumps(result)) > MAX_BODY_CHARS:
            result["body"] = {"truncated": True, "preview": str(parsed)[:MAX_BODY_CHARS]}
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "stream":
        result = stream(base_url, args.token, args.path, args.max_seconds)
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "call":
        body = json.loads(args.json_body) if args.json_body else None
        _confirm_gate(args.method, args.path, args.confirm)
        result = request(base_url, args.token, args.method, args.path, body)
        print(json.dumps(result, indent=2))
        return

    # Named GET shortcuts
    path_template, path_params, query_params, _ = SHORTCUTS[args.cmd]
    path = path_template
    for param in path_params:
        path = path.replace(f"{{{param}}}", getattr(args, param))
    query = {qname: getattr(args, qname) for qname, _ in query_params}
    result = request(base_url, args.token, "GET", path, query=query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
