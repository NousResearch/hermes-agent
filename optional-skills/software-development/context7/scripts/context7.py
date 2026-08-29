#!/usr/bin/env python3
"""Query Context7's public HTTP API without an MCP client."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_BASE = "https://context7.com/api/v2"
DEFAULT_TIMEOUT = 30.0


class _NoRedirectHandler(HTTPRedirectHandler):
    """Surface HTTP redirects as errors instead of forwarding secrets."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _open_without_redirects(request: Request, timeout: float):
    return build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


class Context7Error(RuntimeError):
    """A structured error returned by the Context7 API."""

    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self.error = str(payload.get("error", "http_error"))
        self.redirect_url = payload.get("redirectUrl")
        message = str(payload.get("message", f"Context7 request failed with HTTP {status}"))
        super().__init__(message)


def _request(
    path: str,
    params: dict[str, str],
    *,
    api_key: str | None = None,
    opener: Callable[..., Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[str, str]:
    url = f"{API_BASE}/{path}?{urlencode(params)}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers)
    open_request = opener or _open_without_redirects
    try:
        with open_request(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"message": raw or str(exc)}
        raise Context7Error(exc.code, payload) from exc
    except URLError as exc:
        raise Context7Error(0, {"message": str(exc.reason)}) from exc
    return body, content_type


def search_libraries(
    library_name: str,
    query: str,
    *,
    fast: bool = False,
    api_key: str | None = None,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    body, _ = _request(
        "libs/search",
        {
            "libraryName": library_name,
            "query": query,
            "fast": str(fast).lower(),
        },
        api_key=api_key,
        opener=opener,
    )
    return json.loads(body)


def get_context(
    library_id: str,
    query: str,
    *,
    response_type: str = "txt",
    fast: bool = False,
    api_key: str | None = None,
    opener: Callable[..., Any] | None = None,
    _redirects_remaining: int = 1,
) -> str | dict[str, Any]:
    try:
        body, _ = _request(
            "context",
            {
                "libraryId": library_id,
                "query": query,
                "type": response_type,
                "fast": str(fast).lower(),
            },
            api_key=api_key,
            opener=opener,
        )
    except Context7Error as exc:
        if exc.status == 301 and exc.redirect_url and _redirects_remaining > 0:
            return get_context(
                str(exc.redirect_url),
                query,
                response_type=response_type,
                fast=fast,
                api_key=api_key,
                opener=opener,
                _redirects_remaining=_redirects_remaining - 1,
            )
        raise
    return json.loads(body) if response_type == "json" else body


def lookup(
    library_name: str,
    query: str,
    *,
    response_type: str = "txt",
    fast: bool = False,
    api_key: str | None = None,
    opener: Callable[..., Any] | None = None,
) -> str | dict[str, Any]:
    """Resolve a library name and return context for the best match."""
    api_key = api_key or os.getenv("CONTEXT7_API_KEY")
    search_result = search_libraries(
        library_name,
        query,
        fast=fast,
        api_key=api_key,
        opener=opener,
    )
    results = search_result.get("results", [])
    if not results:
        raise RuntimeError(f"No Context7 library matched {library_name!r}")
    return get_context(
        results[0]["id"],
        query,
        response_type=response_type,
        fast=fast,
        api_key=api_key,
        opener=opener,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search current library documentation through Context7's HTTP API. "
            "Anonymous access works with a lower quota; CONTEXT7_API_KEY is optional."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="resolve a library name")
    search_parser.add_argument("library_name")
    search_parser.add_argument("query")
    search_parser.add_argument("--fast", action="store_true")

    context_parser = subparsers.add_parser("context", help="fetch context by library ID")
    context_parser.add_argument("library_id")
    context_parser.add_argument("query")
    context_parser.add_argument("--type", choices=("txt", "json"), default="txt")
    context_parser.add_argument("--fast", action="store_true")

    lookup_parser = subparsers.add_parser("lookup", help="resolve and fetch in one command")
    lookup_parser.add_argument("library_name")
    lookup_parser.add_argument("query")
    lookup_parser.add_argument("--type", choices=("txt", "json"), default="txt")
    lookup_parser.add_argument("--fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    api_key = os.getenv("CONTEXT7_API_KEY")
    try:
        if args.command == "search":
            result = search_libraries(
                args.library_name,
                args.query,
                fast=args.fast,
                api_key=api_key,
            )
        elif args.command == "context":
            result = get_context(
                args.library_id,
                args.query,
                response_type=args.type,
                fast=args.fast,
                api_key=api_key,
            )
        else:
            result = lookup(
                args.library_name,
                args.query,
                response_type=args.type,
                fast=args.fast,
                api_key=api_key,
            )
    except (Context7Error, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Context7 error: {exc}", file=sys.stderr)
        return 1

    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
