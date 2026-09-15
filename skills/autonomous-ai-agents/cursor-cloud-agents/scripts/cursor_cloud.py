#!/usr/bin/env python3
"""Drive Cursor Cloud Agents (https://api.cursor.com) from a Hermes terminal.

Stdlib only. The API key is read from the ``CURSOR_API_KEY`` environment
variable and is never written to disk, echoed, or embedded in a prompt. The
API host is fixed at ``BASE_URL`` — Cloud Agents are a remote service, so a
caller-supplied host would only ever be a way to leak the key to a third party.

Usage:
    python cursor_cloud.py models
    python cursor_cloud.py launch --prompt "<task>" --repo owner/name [--ref main]
    python cursor_cloud.py followup <agent-id> --prompt "<task>"
    python cursor_cloud.py status <agent-id>
    python cursor_cloud.py watch <agent-id> --run <run-id>
    python cursor_cloud.py cancel <agent-id> --run <run-id>
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.cursor.com"
ENV_VAR = "CURSOR_API_KEY"
USER_AGENT = "hermes-cursor-cloud-agents/1.0"
DEFAULT_TIMEOUT = 60.0

# Run.status enum from the Cloud Agents OpenAPI spec: everything else is
# in-flight and worth polling.
TERMINAL_STATUSES = frozenset({"FINISHED", "ERROR", "CANCELLED", "EXPIRED"})

_HTTPS_REPO = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9._-]+)/(?P<name>[A-Za-z0-9._-]+?)"
    r"(?:\.git)?/?$"
)
_SSH_REPO = re.compile(
    r"^git@github\.com:(?P<owner>[A-Za-z0-9._-]+)/(?P<name>[A-Za-z0-9._-]+?)(?:\.git)?$"
)
_SLUG_REPO = re.compile(
    r"^(?P<owner>[A-Za-z0-9._-]+)/(?P<name>[A-Za-z0-9._-]+?)(?:\.git)?$"
)


class CursorCloudError(RuntimeError):
    """User-facing failure: bad input, missing credential, or API error."""


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def auth_header(api_key: str) -> str:
    """Cloud Agents accepts the key as Basic user with an empty password."""
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def normalize_repo(value: str) -> str:
    """Accept owner/name, an HTTPS URL, or an SSH URL; emit the canonical URL."""
    for pattern in (_HTTPS_REPO, _SSH_REPO, _SLUG_REPO):
        match = pattern.match(value.strip())
        if match:
            return f"https://github.com/{match.group('owner')}/{match.group('name')}"
    raise CursorCloudError(
        f"not a GitHub repository: {value!r} (use owner/name or a github.com URL)"
    )


def api_key_from_env() -> str:
    key = os.environ.get(ENV_VAR, "").strip()
    if not key:
        raise CursorCloudError(
            f"{ENV_VAR} is not set. Create a key at https://cursor.com/dashboard "
            "and export it in the environment that launches the agent."
        )
    return key


def _urlopen(req, timeout):  # pragma: no cover - thin seam, stubbed in tests
    return urllib.request.urlopen(req, timeout=timeout)


def api_request(
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    headers = {
        "Authorization": auth_header(api_key or api_key_from_env()),
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        base_url.rstrip("/") + path, data=body, headers=headers, method=method
    )
    try:
        with _urlopen(req, timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise CursorCloudError(f"{method} {path} failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise CursorCloudError(f"{method} {path} failed: {exc.reason}") from exc
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CursorCloudError(f"{method} {path} returned non-JSON: {raw[:200]!r}") from exc


def build_launch_payload(
    prompt: str,
    *,
    repo: str | None = None,
    ref: str | None = None,
    model: str | None = None,
    name: str | None = None,
    mode: str | None = None,
    work_on_current_branch: bool = False,
    auto_create_pr: bool = False,
) -> dict:
    payload: dict = {"prompt": {"text": prompt}}
    if repo:
        entry = {"url": normalize_repo(repo)}
        if ref:
            entry["startingRef"] = ref
        payload["repos"] = [entry]
    elif ref:
        raise CursorCloudError("--ref only applies with --repo")
    if model:
        payload["model"] = {"id": model}
    if name:
        payload["name"] = name
    if mode:
        payload["mode"] = mode
    # Both defaults are spelled out because the docs' default (a new cursor/...
    # branch, no PR) is what a caller usually wants, not a silent server default.
    payload["workOnCurrentBranch"] = bool(work_on_current_branch)
    payload["autoCreatePR"] = bool(auto_create_pr)
    return payload


def build_followup_payload(prompt: str, *, mode: str | None = None) -> dict:
    payload: dict = {"prompt": {"text": prompt}}
    if mode:
        payload["mode"] = mode
    return payload


def print_run(run: dict) -> None:
    for label, value in (
        ("run", run.get("id")),
        ("status", run.get("status")),
    ):
        if value:
            print(f"{label}\t{value}")
    git = run.get("git") or {}
    for branch in git.get("branches") or []:
        print(f"branch\t{branch}")
    for pr in git.get("prs") or []:
        print(f"pr\t{pr}")
    if run.get("result"):
        print(f"result\t{run['result']}")


# --- commands --------------------------------------------------------------


def cmd_models(args) -> int:
    data = api_request("GET", "/v1/models")
    for item in data.get("items") or []:
        label = item.get("displayName") or item.get("name") or ""
        print(f"{item.get('id', '?')}\t{label}".rstrip())
    return 0


def cmd_me(args) -> int:
    data = api_request("GET", "/v1/me")
    print(f"apiKeyName\t{data.get('apiKeyName', '?')}")
    print(f"createdAt\t{data.get('createdAt', '?')}")
    return 0


def cmd_repos(args) -> int:
    data = api_request("GET", "/v1/repositories")
    for item in data.get("items") or []:
        print(item.get("url", "?"))
    return 0


def cmd_list(args) -> int:
    data = api_request("GET", f"/v1/agents?limit={args.limit}")
    for item in data.get("items") or []:
        print(f"{item.get('id', '?')}\t{item.get('status', '')}\t{item.get('name', '')}".rstrip())
    if data.get("nextCursor"):
        print(f"nextCursor\t{data['nextCursor']}")
    return 0


def cmd_launch(args) -> int:
    payload = build_launch_payload(
        args.prompt,
        repo=args.repo,
        ref=args.ref,
        model=args.model,
        name=args.name,
        mode=args.mode,
        work_on_current_branch=args.work_on_current_branch,
        auto_create_pr=args.auto_create_pr,
    )
    data = api_request("POST", "/v1/agents", payload)
    agent = data.get("agent") or {}
    run = data.get("run") or {}
    if agent.get("id"):
        print(f"agent\t{agent['id']}")
    print_run(run)
    return 0


def cmd_followup(args) -> int:
    payload = build_followup_payload(args.prompt, mode=args.mode)
    data = api_request("POST", f"/v1/agents/{args.agent}/runs", payload)
    print_run(data.get("run") or {})
    return 0


def cmd_status(args) -> int:
    agent = api_request("GET", f"/v1/agents/{args.agent}")
    runs = api_request("GET", f"/v1/agents/{args.agent}/runs?limit=1")
    print(f"agent\t{agent.get('id', args.agent)}\t{agent.get('name', '')}".rstrip())
    items = runs.get("items") or []
    if not items:
        print("run\t(none)")
        return 0
    print_run(items[0])
    return 0


def cmd_watch(args) -> int:
    deadline = time.monotonic() + args.timeout
    while True:
        run = api_request("GET", f"/v1/agents/{args.agent}/runs/{args.run}")
        status = run.get("status", "?")
        print(f"status\t{status}", flush=True)
        if is_terminal(status):
            print_run(run)
            return 0
        if time.monotonic() >= deadline:
            print(f"timed out after {args.timeout}s with status {status}", file=sys.stderr)
            return 1
        if args.interval:
            time.sleep(args.interval)


def cmd_cancel(args) -> int:
    data = api_request("POST", f"/v1/agents/{args.agent}/runs/{args.run}/cancel")
    print(f"cancelled\t{data.get('id', args.run)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cursor_cloud.py",
        description="Launch and manage Cursor Cloud Agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("models", help="list models usable with launch --model")
    p.set_defaults(handler=cmd_models)

    p = subparsers.add_parser("me", help="show the API key identity")
    p.set_defaults(handler=cmd_me)

    p = subparsers.add_parser("repos", help="list GitHub repos visible to the key")
    p.set_defaults(handler=cmd_repos)

    p = subparsers.add_parser("list", help="list agents, newest first")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(handler=cmd_list)

    p = subparsers.add_parser("launch", help="create an agent and enqueue its first run")
    p.add_argument("--prompt", required=True)
    p.add_argument("--repo", help="owner/name or a github.com URL")
    p.add_argument("--ref", help="starting branch or commit SHA (requires --repo)")
    p.add_argument("--model", help="model id from the models command")
    p.add_argument("--name", help="display name for the agent")
    p.add_argument("--mode", choices=("agent", "plan"))
    p.add_argument("--work-on-current-branch", action="store_true")
    p.add_argument("--auto-create-pr", action="store_true")
    p.set_defaults(handler=cmd_launch)

    p = subparsers.add_parser("followup", help="send a follow-up prompt to an agent")
    p.add_argument("agent")
    p.add_argument("--prompt", required=True)
    p.add_argument("--mode", choices=("agent", "plan"))
    p.set_defaults(handler=cmd_followup)

    p = subparsers.add_parser("status", help="show an agent and its latest run")
    p.add_argument("agent")
    p.set_defaults(handler=cmd_status)

    p = subparsers.add_parser("watch", help="poll a run until it reaches a terminal status")
    p.add_argument("agent")
    p.add_argument("--run", required=True)
    p.add_argument("--interval", type=float, default=10.0, help="seconds between polls")
    p.add_argument("--timeout", type=float, default=900.0, help="give up after this many seconds")
    p.set_defaults(handler=cmd_watch)

    p = subparsers.add_parser("cancel", help="cancel the active run of an agent")
    p.add_argument("agent")
    p.add_argument("--run", required=True)
    p.set_defaults(handler=cmd_cancel)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except CursorCloudError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
