#!/usr/bin/env python3
"""Sokosumi API helper: hire agents, create coworker tasks, poll to completion.

Stdlib only. Auth via SOKOSUMI_API_KEY (Bearer). Base URL via SOKOSUMI_API_URL
(default https://api.sokosumi.com; preprod https://api.preprod.sokosumi.com --
keys are environment-specific). Every response unwraps the {data, meta}
envelope; list commands paginate with meta.pagination.nextCursor (no page
param exists). Only 429 responses are retried (exponential backoff); never
blind-retry a timed-out POST -- read state back first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_API_URL = "https://api.sokosumi.com"
REQUEST_TIMEOUT_S = 30
MAX_429_RETRIES = 4

# Three distinct status vocabularies -- never conflate them.
# Job.status (lowercase):
JOB_TERMINAL = {"completed", "failed", "payment_failed", "refund_resolved", "dispute_resolved"}
JOB_BLOCKED = {"input_required"}
# Job *event* status is a different, UPPERCASE enum:
# INITIATED | AWAITING_PAYMENT | AWAITING_INPUT | RUNNING | COMPLETED | FAILED
# Task.status (UPPERCASE):
TASK_TERMINAL = {"COMPLETED", "FAILED", "CANCELED"}
TASK_BLOCKED = {"INPUT_REQUIRED", "AUTHENTICATION_REQUIRED", "OUT_OF_CREDITS"}

KEY_HELP = (
    "Set SOKOSUMI_API_KEY (create one at https://app.sokosumi.com/connections; "
    "mainnet and preprod keys are separate)."
)

# Indirection point so hermetic tests can patch a single symbol.
_urlopen = urllib.request.urlopen


class ApiError(Exception):
    def __init__(self, status: int | None, message: str, body: object = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _base_url() -> str:
    return os.environ.get("SOKOSUMI_API_URL", DEFAULT_API_URL).rstrip("/")


def _api_key() -> str:
    key = os.environ.get("SOKOSUMI_API_KEY", "")
    if not key:
        raise ApiError(None, f"No API key found. {KEY_HELP}")
    return key


def request(method: str, path: str, body: dict | None = None, query: dict | None = None,
            raw: bool = False) -> dict:
    """Call /v1/<path>, unwrap the {data, meta} envelope, retry only on 429.

    raw=True returns the whole envelope (pagination lives in meta)."""
    url = f"{_base_url()}/v1/{path.lstrip('/')}"
    if query:
        pairs = [(k, v) for k, vals in query.items() if vals is not None
                 for v in (vals if isinstance(vals, list) else [vals])]
        if pairs:
            url += "?" + urllib.parse.urlencode(pairs)
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    for attempt in range(MAX_429_RETRIES + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with _urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                text = resp.read().decode("utf-8")
                try:
                    payload = json.loads(text)
                except ValueError:
                    raise ApiError(None, f"Non-JSON response from {url}: {text[:200]!r}") from None
                if raw:
                    return payload
                return payload.get("data", payload) if isinstance(payload, dict) else payload
        except urllib.error.HTTPError as err:
            err_text = err.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(err_text)
            except ValueError:
                parsed = err_text
            message = parsed.get("message", err_text) if isinstance(parsed, dict) else err_text
            if err.code == 429:
                if attempt < MAX_429_RETRIES:
                    time.sleep(2**attempt)
                    continue
                message = f"Rate limited after {MAX_429_RETRIES} retries: {message}"
            if err.code == 401:
                message = f"Unauthorized: {message}. {KEY_HELP}"
            raise ApiError(err.code, message, parsed) from None
        except urllib.error.URLError as err:
            raise ApiError(None, f"Request to {url} failed: {err.reason}") from None
        except OSError as err:  # incl. bare TimeoutError from a stalled read
            raise ApiError(None, f"Request to {url} failed: {err}") from None
    raise AssertionError("unreachable")


def paginate(path: str, query: dict | None = None, limit: int | None = None) -> list:
    """Follow meta.pagination.nextCursor... simplified: single page unless --all.

    The API caps limit at 100 per page; callers wanting everything pass
    limit=None and we loop cursors until exhausted.
    """
    items: list = []
    cursor = None
    while True:
        q = dict(query or {})
        q["limit"] = min(limit - len(items), 100) if limit else 100
        if cursor:
            q["cursor"] = cursor
        page = request("GET", path, query=q, raw=True)
        batch = page.get("data") or [] if isinstance(page, dict) else page
        items.extend(batch)
        prev_cursor, cursor = cursor, _next_cursor(page)
        if not cursor or cursor == prev_cursor or not batch or (limit and len(items) >= limit):
            return items[:limit] if limit else items


def _next_cursor(page: object) -> str | None:
    if isinstance(page, dict):
        pagination = (page.get("meta") or {}).get("pagination") or {}
        return pagination.get("nextCursor")
    return None


def whoami() -> dict:
    try:
        return {"kind": "user", "identity": request("GET", "users/me")}
    except ApiError as err:
        if err.status in (401, 403, 404):
            # Coworker bearer keys identify via /coworkers/me instead.
            try:
                return {"kind": "coworker", "identity": request("GET", "coworkers/me")}
            except ApiError:
                raise err from None
        raise


def _fetch_input_schema(agent_id: str) -> dict:
    """GET the agent's input schema, retrying once on a transient server
    failure (observed live: intermittent 422 'Failed to parse input schema'
    that succeeds on immediate retry)."""
    try:
        return request("GET", f"agents/{agent_id}/input-schema")
    except ApiError as err:
        if err.status in (422, 500, 502, 503):
            return request("GET", f"agents/{agent_id}/input-schema")
        raise


def hire(agent_id: str, input_data: dict, max_credits: float | None,
         name: str | None, task_id: str | None = None) -> dict:
    """Hire an agent. The API requires inputSchema echoed verbatim from
    GET /agents/{id}/input-schema; inputData is keyed by schema field id."""
    body: dict = {"inputSchema": _fetch_input_schema(agent_id), "inputData": input_data}
    if max_credits is not None:
        body["maxCredits"] = max_credits
    if name:
        body["name"] = name
    path = f"agents/{agent_id}/jobs"
    if task_id:
        body["agentId"] = agent_id
        path = f"tasks/{task_id}/jobs"
    try:
        return request("POST", path, body=body)
    except ApiError as err:
        if err.status == 422:
            # The stored schema may have changed between fetch and post; a 422
            # creates no job, so refetch once and retry.
            body["inputSchema"] = _fetch_input_schema(agent_id)
            return request("POST", path, body=body)
        raise


def wait(kind: str, item_id: str, interval: int, timeout: int, out=None) -> int:
    """Poll a job or task until terminal/blocked. Exit codes:
    0 success-terminal, 1 failure-terminal or timeout, 2 blocked on input."""
    out = out if out is not None else sys.stdout
    terminal = JOB_TERMINAL if kind == "job" else TASK_TERMINAL
    blocked = JOB_BLOCKED if kind == "job" else TASK_BLOCKED
    success = {"completed"} if kind == "job" else {"COMPLETED"}
    deadline = time.monotonic() + timeout
    last_status = None
    while True:
        item = request("GET", f"{kind}s/{item_id}")
        status = item.get("status")
        if status != last_status:
            print(json.dumps({"id": item_id, "status": status}), file=out, flush=True)
            last_status = status
        if status in terminal or status in blocked:
            print(json.dumps({"final": item}, indent=2), file=out)
            if status in success:
                return 0
            return 2 if status in blocked else 1
        if time.monotonic() >= deadline:
            print(json.dumps({"timeout": True, "id": item_id, "status": status}), file=out)
            return 1
        time.sleep(interval)


def job_details(job_id: str) -> dict:
    details = {"job": request("GET", f"jobs/{job_id}")}
    for part in ("events", "files", "links", "input-request"):
        try:
            details[part.replace("-", "_")] = request("GET", f"jobs/{job_id}/{part}")
        except ApiError as err:
            details.setdefault("errors", {})[part] = {"status": err.status, "message": str(err)}
    return details


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _json_arg(value: str) -> dict:
    try:
        return json.loads(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"not valid JSON: {err}") from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sokosumi_api.py", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("whoami", help="verify the key; user or coworker identity")
    sub.add_parser("credits", help="current credit balance")

    agents = sub.add_parser("agents", help="list hireable agents")
    agents.add_argument("--category", action="append", help="category slug filter (repeatable)")
    agents.add_argument("--limit", type=_positive_int, default=20)

    agent = sub.add_parser("agent", help="agent detail incl. credits price")
    agent.add_argument("agent_id")

    schema = sub.add_parser("input-schema", help="input schema for an agent")
    schema.add_argument("agent_id")

    hire_p = sub.add_parser("hire", help="hire an agent (fetches + echoes input schema)")
    hire_p.add_argument("agent_id")
    hire_p.add_argument("--input-json", type=_json_arg, required=True,
                    help="inputData JSON keyed by schema field id")
    hire_p.add_argument("--max-credits", type=float)
    hire_p.add_argument("--name")
    hire_p.add_argument("--task-id", help="attach the job to an existing task instead")

    coworkers = sub.add_parser("coworkers", help="list coworkers")
    coworkers.add_argument("--limit", type=_positive_int, default=20)
    coworkers.add_argument("--scope", choices=["whitelisted", "all", "archived"],
                           help="default: whitelisted")
    coworkers.add_argument("--capability", choices=["chat", "tasks"])

    task_c = sub.add_parser("create-task", help="create a coworker task (DRAFT unless --ready)")
    task_c.add_argument("--name", required=True)
    task_c.add_argument("--description")
    task_c.add_argument("--coworker-id")
    task_c.add_argument("--ready", action="store_true", help="start now (spends credits)")

    for kind in ("job", "task"):
        g = sub.add_parser(kind, help=f"fetch one {kind}")
        g.add_argument(f"{kind}_id")
        if kind == "job":
            g.add_argument("--details", action="store_true",
                           help="aggregate events, files, links, input-request")

    events = sub.add_parser("task-events", help="task activity feed")
    events.add_argument("task_id")

    comment = sub.add_parser("comment", help="post a comment on a task")
    comment.add_argument("task_id")
    comment.add_argument("--text", required=True)

    wait_p = sub.add_parser("wait", help="poll until terminal or input needed")
    wait_p.add_argument("kind", choices=["job", "task"])
    wait_p.add_argument("item_id")
    wait_p.add_argument("--interval", type=int, default=60, help="seconds between polls")
    wait_p.add_argument("--timeout", type=int, default=3600, help="max seconds to wait")

    inreq = sub.add_parser("input-request", help="pending input request for a job")
    inreq.add_argument("job_id")

    provide = sub.add_parser("provide-input", help="answer a job's input request")
    provide.add_argument("job_id")
    provide.add_argument("--event-id", required=True)
    provide.add_argument("--input-json", type=_json_arg, required=True)

    return p


def run(args: argparse.Namespace) -> tuple[int, object]:
    cmd = args.command
    if cmd == "whoami":
        return 0, whoami()
    if cmd == "credits":
        return 0, request("GET", "users/me/credits")
    if cmd == "agents":
        return 0, paginate("agents", query={"category": args.category}, limit=args.limit)
    if cmd == "agent":
        return 0, request("GET", f"agents/{args.agent_id}")
    if cmd == "input-schema":
        return 0, request("GET", f"agents/{args.agent_id}/input-schema")
    if cmd == "hire":
        return 0, hire(args.agent_id, args.input_json, args.max_credits,
                       args.name, args.task_id)
    if cmd == "coworkers":
        # GET /coworkers has no limit/cursor params (only scope and capability);
        # truncate client-side.
        query = {"scope": args.scope, "capability": args.capability}
        listed = request("GET", "coworkers", query=query)
        return 0, listed[: args.limit] if isinstance(listed, list) else listed
    if cmd == "create-task":
        body = {"name": args.name, "status": "READY" if args.ready else "DRAFT"}
        if args.description:
            body["description"] = args.description
        if args.coworker_id:
            body["coworkerId"] = args.coworker_id
        return 0, request("POST", "tasks", body=body)
    if cmd == "job":
        return 0, job_details(args.job_id) if args.details else request("GET", f"jobs/{args.job_id}")
    if cmd == "task":
        return 0, request("GET", f"tasks/{args.task_id}")
    if cmd == "task-events":
        return 0, request("GET", f"tasks/{args.task_id}/events")
    if cmd == "comment":
        return 0, request("POST", f"tasks/{args.task_id}/events", body={"comment": args.text})
    if cmd == "wait":
        return wait(args.kind, args.item_id, args.interval, args.timeout), None
    if cmd == "input-request":
        return 0, request("GET", f"jobs/{args.job_id}/input-request")
    if cmd == "provide-input":
        body = {"eventId": args.event_id, "inputData": args.input_json}
        return 0, request("POST", f"jobs/{args.job_id}/inputs", body=body)
    raise ApiError(None, f"Unknown command: {cmd}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, result = run(args)
    except ApiError as err:
        print(json.dumps({"error": {"status": err.status, "message": str(err)}}),
              file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
