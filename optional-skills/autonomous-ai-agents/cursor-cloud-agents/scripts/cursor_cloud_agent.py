#!/usr/bin/env python3
"""Small, dependency-free CLI for Cursor Cloud Agents API v1."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE = "https://api.cursor.com"
TERMINAL_STATUSES = {"FINISHED", "ERROR", "CANCELLED", "EXPIRED"}


class ApiError(RuntimeError):
    """A user-actionable Cursor API or input error."""


def api_request(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    accept: str = "application/json",
) -> Any:
    key = os.environ.get("CURSOR_API_KEY")
    if not key:
        raise ApiError("CURSOR_API_KEY is not set")
    base = DEFAULT_BASE
    token = base64.b64encode((key + ":").encode("utf-8")).decode("ascii")
    body = None
    headers = {"Authorization": f"Basic {token}", "Accept": accept}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(base + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ApiError(f"Cursor API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ApiError(f"Cursor API connection failed: {exc.reason}") from exc
    if not raw:
        return {}
    if "json" in content_type or raw.lstrip().startswith((b"{", b"[")):
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError("Cursor API returned invalid JSON") from exc
    return raw.decode("utf-8", "replace")


def require_https_github_repo(value: str) -> str:
    prefix = "https://github.com/"
    parts = value.removeprefix(prefix).rstrip("/").split("/")
    if not value.startswith(prefix) or len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            "--repo must be an HTTPS GitHub repository URL"
        )
    return value.rstrip("/")


def prompt_value(args: argparse.Namespace) -> str:
    text = args.prompt or ""
    if args.prompt_file:
        try:
            file_text = Path(args.prompt_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ApiError(f"cannot read --prompt-file: {exc}") from exc
        text = f"{text}\n\n{file_text}" if text else file_text
    if not text.strip():
        raise ApiError("a non-empty --prompt or --prompt-file is required")
    return text.strip()


def print_result(value: Any) -> None:
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


def create_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": {"text": prompt_value(args)},
        "repos": [{"url": args.repo, "startingRef": args.ref}],
    }
    if args.name:
        payload["name"] = args.name
    if args.model:
        payload["model"] = {"id": args.model}
    if args.auto_create_pr:
        payload["autoCreatePR"] = True
    if args.work_on_current_branch:
        payload["workOnCurrentBranch"] = True
    if args.mode:
        payload["mode"] = args.mode
    return payload


def run_object(response: Dict[str, Any]) -> Dict[str, Any]:
    run = response.get("run")
    if not isinstance(run, dict) or not run.get("id") or not run.get("agentId"):
        raise ApiError("Cursor response did not contain an agent run")
    return run


def get_run(agent_id: str, run_id: str) -> Dict[str, Any]:
    value = api_request("GET", f"/v1/agents/{agent_id}/runs/{run_id}")
    if not isinstance(value, dict):
        raise ApiError("Cursor returned a non-object run response")
    return value


def wait_for_run(
    agent_id: str,
    run_id: str,
    poll_seconds: float,
    wait_timeout: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + wait_timeout if wait_timeout > 0 else None
    while True:
        run = get_run(agent_id, run_id)
        status = str(run.get("status", "")).upper()
        if status in TERMINAL_STATUSES:
            return run
        if deadline is not None and time.monotonic() >= deadline:
            raise ApiError(f"timed out waiting for run {run_id}")
        time.sleep(poll_seconds)


def command_launch(args: argparse.Namespace) -> Dict[str, Any]:
    response = api_request("POST", "/v1/agents", create_payload(args))
    run = run_object(response)
    result: Dict[str, Any] = {"agent": response.get("agent"), "run": run}
    if args.wait:
        result["finalRun"] = wait_for_run(
            run["agentId"], run["id"], args.poll_seconds, args.wait_timeout
        )
    return result


def command_follow_up(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": {"text": prompt_value(args)}}
    if args.mode:
        payload["mode"] = args.mode
    response = api_request("POST", f"/v1/agents/{args.agent_id}/runs", payload)
    run = run_object({"run": response.get("run")})
    result: Dict[str, Any] = {"run": run}
    if args.wait:
        result["finalRun"] = wait_for_run(
            args.agent_id, run["id"], args.poll_seconds, args.wait_timeout
        )
    return result


def iter_sse(response: Any) -> Iterable[Dict[str, Any]]:
    event = "message"
    data_lines = []
    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data_lines:
                text = "\n".join(data_lines)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {"raw": text}
                yield {"event": event, "data": data}
            event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


def command_stream(args: argparse.Namespace) -> None:
    key = os.environ.get("CURSOR_API_KEY")
    if not key:
        raise ApiError("CURSOR_API_KEY is not set")
    base = DEFAULT_BASE
    token = base64.b64encode((key + ":").encode("utf-8")).decode("ascii")
    request = Request(
        f"{base}/v1/agents/{args.agent_id}/runs/{args.run_id}/stream",
        headers={"Authorization": f"Basic {token}", "Accept": "text/event-stream"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            for event in iter_sse(response):
                print(json.dumps(event, sort_keys=True), flush=True)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ApiError(f"Cursor stream HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ApiError(f"Cursor stream connection failed: {exc.reason}") from exc


def add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")


def add_wait_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", action="store_true", help="poll until a terminal status")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=3600.0,
        help="maximum seconds to wait; 0 disables the timeout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="create an agent and enqueue its first run")
    launch.add_argument("--repo", required=True, type=require_https_github_repo)
    launch.add_argument("--ref", default="main")
    add_prompt_arguments(launch)
    launch.add_argument("--name")
    launch.add_argument("--model")
    launch.add_argument("--mode", choices=("agent", "plan"))
    launch.add_argument("--auto-create-pr", action="store_true")
    launch.add_argument("--work-on-current-branch", action="store_true")
    add_wait_arguments(launch)
    launch.set_defaults(handler=command_launch)

    follow = sub.add_parser("follow-up", help="send a prompt to an existing agent")
    follow.add_argument("--agent-id", required=True)
    add_prompt_arguments(follow)
    follow.add_argument("--mode", choices=("agent", "plan"))
    add_wait_arguments(follow)
    follow.set_defaults(handler=command_follow_up)

    status = sub.add_parser("status", help="read one run")
    status.add_argument("--agent-id", required=True)
    status.add_argument("--run-id", required=True)
    status.set_defaults(handler=lambda a: get_run(a.agent_id, a.run_id))

    stream = sub.add_parser("stream", help="print Server-Sent Events for one run")
    stream.add_argument("--agent-id", required=True)
    stream.add_argument("--run-id", required=True)
    stream.add_argument("--timeout", type=float, default=900)
    stream.set_defaults(handler=command_stream)

    cancel = sub.add_parser("cancel", help="cancel an active run")
    cancel.add_argument("--agent-id", required=True)
    cancel.add_argument("--run-id", required=True)
    cancel.set_defaults(
        handler=lambda a: api_request(
            "POST", f"/v1/agents/{a.agent_id}/runs/{a.run_id}/cancel"
        )
    )

    models = sub.add_parser("models", help="list available models")
    models.set_defaults(handler=lambda a: api_request("GET", "/v1/models"))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        value = args.handler(args)
        if value is not None:
            print_result(value)
        return 0
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
