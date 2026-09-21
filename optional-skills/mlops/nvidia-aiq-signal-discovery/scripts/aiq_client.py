#!/usr/bin/env python3
"""Small, dependency-free client for the NVIDIA AI-Q HTTP job contract."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Allow redirects only when their target meets the same transport policy."""

    def redirect_request(self, request, file, code, message, headers, new_url):
        parsed = urlparse(new_url)
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise URLError("AI-Q redirect target must use HTTPS except for loopback development")
        return super().redirect_request(request, file, code, message, headers, new_url)


_OPENER = build_opener(_SafeRedirectHandler)


def request_json(base_url: str, path: str, method: str = "GET", payload: object | None = None) -> object:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise SystemExit("AI-Q server must use HTTPS except for loopback development")
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method=method, headers={"Accept": "application/json"})
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with _OPENER.open(request, timeout=30) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"AI-Q request failed: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit, poll, and download an AI-Q job")
    parser.add_argument("--server", default=os.environ.get("AIQ_SERVER_URL"), required=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="POST a JSON request to the job endpoint")
    submit.add_argument("request", type=Path)
    submit.add_argument("--path", default="/v1/jobs")

    status = subparsers.add_parser("status", help="GET a job status")
    status.add_argument("job_id")
    status.add_argument("--path-template", default="/v1/jobs/{job_id}")

    download = subparsers.add_parser("download", help="GET a completed job artifact")
    download.add_argument("job_id")
    download.add_argument("output", type=Path)
    download.add_argument("--path-template", default="/v1/jobs/{job_id}/result")

    args = parser.parse_args()
    if not args.server:
        parser.error("--server or AIQ_SERVER_URL is required")
    if args.command == "submit":
        result = request_json(
            args.server,
            args.path,
            "POST",
            json.loads(args.request.read_text(encoding="utf-8")),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "status":
        result = request_json(args.server, args.path_template.format(job_id=quote(args.job_id, safe="")))
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = request_json(args.server, args.path_template.format(job_id=quote(args.job_id, safe="")))
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
