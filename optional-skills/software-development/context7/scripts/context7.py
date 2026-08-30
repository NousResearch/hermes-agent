#!/usr/bin/env python3
"""Query Context7's public HTTP API without an MCP client."""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.client import HTTPException
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_BASE = "https://context7.com/api/v2"
DEFAULT_TIMEOUT = 30.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ERROR_MESSAGE_CHARS = 500


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

    def __init__(self, status: int, payload: Any) -> None:
        if not isinstance(payload, dict):
            payload = {
                "error": "invalid_error_payload",
                "message": "Context7 returned an unexpected error payload",
            }
        self.status = status
        self.error = str(payload.get("error", "http_error"))
        self.redirect_url = payload.get("redirectUrl")
        message = str(payload.get("message", f"Context7 request failed with HTTP {status}"))
        if len(message) > MAX_ERROR_MESSAGE_CHARS:
            message = message[:MAX_ERROR_MESSAGE_CHARS] + "..."
        super().__init__(message)


def _read_body(response: Any, *, status: int = 0) -> str:
    try:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    except TimeoutError as exc:
        raise Context7Error(
            status,
            {"message": f"Context7 response read timed out: {exc}"},
        ) from exc
    except (HTTPException, OSError) as exc:
        raise Context7Error(
            status,
            {"message": f"Context7 response read failed: {exc}"},
        ) from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise Context7Error(
            status,
            {"message": f"Context7 response exceeds {MAX_RESPONSE_BYTES} bytes"},
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Context7Error(status, {"message": "Context7 returned invalid UTF-8"}) from exc


def _parse_json_object(body: str, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise Context7Error(0, {"message": f"Context7 {label} returned invalid JSON"}) from exc
    if not isinstance(payload, dict):
        raise Context7Error(0, {"message": f"Context7 {label} must be a JSON object"})
    return payload


def _require_json_content_type(content_type: str, *, label: str) -> None:
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise Context7Error(
            0,
            {"message": f"Context7 {label} returned unexpected content type {content_type!r}"},
        )


def _validate_library_id(value: Any, *, status: int, label: str) -> str:
    if not isinstance(value, str):
        raise Context7Error(status, {"message": f"Context7 {label} is not a valid library ID"})
    library_id = value.strip()
    segments = library_id.split("/")[1:]
    has_forbidden_character = any(
        character.isspace() or ord(character) < 32 or character in "?#\\"
        for character in library_id
    )
    if (
        len(library_id) > 512
        or not library_id.startswith("/")
        or library_id.startswith("//")
        or "://" in library_id
        or has_forbidden_character
        or len(segments) < 2
        or any(not segment or segment in {".", ".."} for segment in segments)
    ):
        raise Context7Error(status, {"message": f"Context7 {label} is not a valid library ID"})
    return library_id


def _redact_secret(payload: Any, secret: str, *, depth: int = 0) -> Any:
    if not secret:
        return payload
    if depth >= 32:
        return "[nested value omitted]"
    if isinstance(payload, dict):
        return {
            _redact_secret(key, secret, depth=depth + 1): _redact_secret(
                value,
                secret,
                depth=depth + 1,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_redact_secret(value, secret, depth=depth + 1) for value in payload]
    if isinstance(payload, str):
        return payload.replace(secret, "[REDACTED]")
    return payload


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
    clean_api_key = api_key.strip() if api_key else ""
    if clean_api_key:
        headers["Authorization"] = f"Bearer {clean_api_key}"
    request = Request(url, headers=headers)
    open_request = opener or _open_without_redirects
    try:
        with open_request(request, timeout=timeout) as response:
            body = _read_body(response)
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        raw = _read_body(exc, status=exc.code)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"message": raw or str(exc)}
        except RecursionError:
            payload = {"message": "Context7 returned invalid error JSON"}
        payload = _redact_secret(payload, clean_api_key)
        raise Context7Error(exc.code, payload) from exc
    except TimeoutError as exc:
        raise Context7Error(0, {"message": f"Context7 request timed out: {exc}"}) from exc
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
    body, content_type = _request(
        "libs/search",
        {
            "libraryName": library_name,
            "query": query,
            "fast": str(fast).lower(),
        },
        api_key=api_key,
        opener=opener,
    )
    _require_json_content_type(content_type, label="search response")
    return _parse_json_object(body, label="search response")


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
    library_id = _validate_library_id(library_id, status=0, label="library ID")
    try:
        body, content_type = _request(
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
            redirect_id = _validate_library_id(
                exc.redirect_url,
                status=301,
                label="redirect",
            )
            if redirect_id == library_id.strip():
                raise Context7Error(301, {"message": "Context7 redirect self-loop detected"}) from exc
            return get_context(
                redirect_id,
                query,
                response_type=response_type,
                fast=fast,
                api_key=api_key,
                opener=opener,
                _redirects_remaining=_redirects_remaining - 1,
            )
        raise
    if response_type == "json":
        _require_json_content_type(content_type, label="context response")
        return _parse_json_object(body, label="context response")
    return body


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
    if not isinstance(results, list):
        raise Context7Error(0, {"message": "Context7 search results must be a list"})
    if not results:
        raise RuntimeError(f"No Context7 library matched {library_name!r}")
    first_result = results[0]
    library_id = first_result.get("id") if isinstance(first_result, dict) else None
    library_id = _validate_library_id(
        library_id,
        status=0,
        label="search result library ID",
    )
    return get_context(
        library_id,
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
