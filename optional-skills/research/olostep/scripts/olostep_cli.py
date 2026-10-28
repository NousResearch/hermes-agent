#!/usr/bin/env python3
"""Search, scrape and answer through the Olostep Python SDK.

    python olostep_cli.py search "query" [--limit N] [--include-domain D] [--exclude-domain D]
    python olostep_cli.py scrape URL [--format markdown|html]
    python olostep_cli.py answer "task" [--json-format '{"field": ""}']

Needs OLOSTEP_API_KEY and `pip install olostep`. Output is JSON, except `scrape`
which prints the page content.
"""

import argparse
import json
import os
import sys


def get_client():
    key = os.environ.get("OLOSTEP_API_KEY")
    if not key:
        sys.exit("OLOSTEP_API_KEY is not set. Create a key at https://www.olostep.com/dashboard")
    try:
        from olostep import Olostep
    except ImportError:
        sys.exit("The olostep SDK is not installed. Run: pip install -U olostep")
    return Olostep(api_key=key)


def sdk_errors():
    try:
        from olostep import Olostep_BaseError
    except ImportError:
        return ()
    return (Olostep_BaseError,)


def search(client, query, limit=None, include_domains=None, exclude_domains=None):
    result = client.searches.create(
        query,
        limit=limit,
        include_domains=include_domains or None,
        exclude_domains=exclude_domains or None,
    )
    return [
        {
            "title": link.get("title"),
            "url": link.get("url"),
            "description": link.get("description"),
        }
        for link in result.links
    ]


def scrape(client, url, fmt="markdown"):
    result = client.scrapes.create(url_to_scrape=url, formats=[fmt])
    content = getattr(result, f"{fmt}_content", None)
    if not content:
        raise RuntimeError(f"Olostep returned no {fmt} content for {url}")
    return content


def answer(client, task, json_format=None):
    kwargs = {"json_format": json_format} if json_format else {}
    result = client.answers.create(task=task, **kwargs)
    content = result.json_content
    try:
        content = json.loads(content)
    except (TypeError, ValueError):
        pass
    return {"answer": content, "sources": list(result.sources)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="web search, returns title/url/description")
    p.add_argument("query")
    p.add_argument("--limit", type=int)
    p.add_argument("--include-domain", action="append", default=[])
    p.add_argument("--exclude-domain", action="append", default=[])

    p = sub.add_parser("scrape", help="fetch one URL as markdown or html")
    p.add_argument("url")
    p.add_argument("--format", choices=["markdown", "html"], default="markdown")

    p = sub.add_parser("answer", help="web-sourced answer with sources")
    p.add_argument("task")
    p.add_argument("--json-format", help='JSON object with empty values, e.g. \'{"title": ""}\'')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    client = get_client()
    try:
        if args.command == "search":
            out = search(client, args.query, args.limit, args.include_domain, args.exclude_domain)
        elif args.command == "scrape":
            print(scrape(client, args.url, args.format))
            return 0
        else:
            fmt = json.loads(args.json_format) if args.json_format else None
            out = answer(client, args.task, fmt)
    except (*sdk_errors(), RuntimeError, ValueError) as exc:
        sys.exit(f"Olostep request failed: {type(exc).__name__}: {exc}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
