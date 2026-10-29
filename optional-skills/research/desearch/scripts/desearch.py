#!/usr/bin/env python3
"""Call Desearch tools on the hosted MCP server (stdlib only).

Sends JSON-RPC ``tools/call`` requests to https://mcp.desearch.ai/mcp with
``Authorization: Bearer $DESEARCH_API_KEY`` and prints the tool result.

Usage:
  desearch.py web QUERY [--start N]
  desearch.py ai PROMPT [--sources web,reddit] [--date-filter PAST_WEEK] [--links-only]
  desearch.py x QUERY [--count N] [--user NAME] [--lang en] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
  desearch.py links PROMPT [--count N]
  desearch.py extract URL [--format text|html]
  desearch.py call TOOL '{"arg": "value"}'
  desearch.py tools
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://mcp.desearch.ai/mcp"
TIMEOUT = 90
USER_AGENT = "hermes-desearch-skill/1.0.0"


class DesearchError(RuntimeError):
    """Raised for missing credentials, transport failures and tool errors."""


def _api_key() -> str:
    key = os.environ.get("DESEARCH_API_KEY", "").strip()
    if not key:
        raise DesearchError(
            "DESEARCH_API_KEY is not set. Create a key at "
            "https://console.desearch.ai/api-keys and set it with hermes setup"
        )
    return key


def _parse_body(raw: str) -> dict:
    """Accept a plain JSON body or a text/event-stream body with data: lines."""
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload.startswith("{"):
                return json.loads(payload)
    raise DesearchError("unexpected response from Desearch MCP server")


def rpc(method: str, params: dict | None = None, *, opener=urllib.request.urlopen) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with opener(request, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code in (401, 403):
            raise DesearchError(f"HTTP {exc.code}: check DESEARCH_API_KEY ({detail})") from None
        raise DesearchError(f"HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise DesearchError(f"network error: {exc.reason}") from None
    message = _parse_body(raw)
    if "error" in message:
        err = message["error"]
        raise DesearchError(f"MCP error {err.get('code')}: {err.get('message')}")
    return message.get("result", {})


def call_tool(name: str, arguments: dict, *, opener=urllib.request.urlopen) -> str:
    result = rpc("tools/call", {"name": name, "arguments": arguments}, opener=opener)
    text = "\n".join(
        block.get("text", "") for block in result.get("content", []) if block.get("type") == "text"
    )
    if result.get("isError"):
        raise DesearchError(f"{name} failed: {text[:500]}")
    return text


def list_tools(*, opener=urllib.request.urlopen) -> list[str]:
    return [tool["name"] for tool in rpc("tools/list", opener=opener).get("tools", [])]


def build_call(args: argparse.Namespace) -> tuple[str, dict]:
    """Map a parsed subcommand to (MCP tool name, arguments)."""
    if args.cmd == "web":
        arguments = {"query": args.query}
        if args.start is not None:
            arguments["start"] = args.start
        return "web-search", arguments
    if args.cmd == "ai":
        arguments = {"prompt": args.prompt}
        if args.sources:
            arguments["tools"] = [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.date_filter:
            arguments["date_filter"] = args.date_filter
        if args.links_only:
            arguments["result_type"] = "ONLY_LINKS"
        return "ai-search", arguments
    if args.cmd == "x":
        arguments = {"query": args.query, "count": args.count}
        for key, value in (("user", args.user), ("lang", args.lang),
                           ("start_date", args.since), ("end_date", args.until)):
            if value:
                arguments[key] = value
        return "x-search", arguments
    if args.cmd == "links":
        arguments = {"prompt": args.prompt}
        if args.count is not None:
            arguments["count"] = args.count
        return "web-links-search", arguments
    if args.cmd == "extract":
        return "extract", {"url": args.url, "format": args.format}
    if args.cmd == "call":
        try:
            arguments = json.loads(args.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise DesearchError(f"arguments must be a JSON object: {exc}") from None
        if not isinstance(arguments, dict):
            raise DesearchError("arguments must be a JSON object")
        return args.tool, arguments
    raise DesearchError(f"unknown command {args.cmd!r}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Desearch search tools over the hosted MCP server.")
    sub = p.add_subparsers(dest="cmd", required=True)

    web = sub.add_parser("web", help="ranked web results (titles, links, snippets)")
    web.add_argument("query")
    web.add_argument("--start", type=int, help="result offset for the next page")

    ai = sub.add_parser("ai", help="AI search with a summary across selected sources")
    ai.add_argument("prompt")
    ai.add_argument("--sources", help="comma list: web,twitter,arxiv,wikipedia,hackernews,reddit")
    ai.add_argument("--date-filter", help="PAST_24_HOURS, PAST_WEEK, PAST_MONTH, PAST_YEAR, ...")
    ai.add_argument("--links-only", action="store_true", help="return links without a summary")

    x = sub.add_parser("x", help="search X posts")
    x.add_argument("query")
    x.add_argument("--count", type=int, default=10)
    x.add_argument("--user", help="only posts from this username")
    x.add_argument("--lang", help="language code, for example en")
    x.add_argument("--since", help="start date YYYY-MM-DD")
    x.add_argument("--until", help="end date YYYY-MM-DD")

    links = sub.add_parser("links", help="web links that match a prompt")
    links.add_argument("prompt")
    links.add_argument("--count", type=int, help="10 to 200")

    extract = sub.add_parser("extract", help="fetch a public URL as text or HTML")
    extract.add_argument("url")
    extract.add_argument("--format", choices=["text", "html"], default="text")

    call = sub.add_parser("call", help="call any Desearch MCP tool with JSON arguments")
    call.add_argument("tool")
    call.add_argument("arguments", nargs="?", default="{}")

    sub.add_parser("tools", help="list available tool names")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.cmd == "tools":
            print("\n".join(list_tools()))
            return 0
        name, arguments = build_call(args)
        print(call_tool(name, arguments))
        return 0
    except DesearchError as exc:
        print(f"desearch: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
