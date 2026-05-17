#!/usr/bin/env python3
"""
Patent Research CLI — search and analyze patents via free public APIs.

Usage:
    python3 patent_search.py search "blockchain consensus" --num 5 --language ENGLISH
    python3 patent_search.py assignee "Apple Inc." --since 2024
    python3 patent_search.py cpc G06N20/00 --num 10
    python3 patent_search.py inventor "John Doe"
    python3 patent_search.py detail US11074495B2
    python3 patent_search.py citations US11074495B2
    python3 patent_search.py landscape "quantum computing" --companies "IBM,Google,Microsoft"

No API keys required. Uses Google Patents public XHR API.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class PatentHTMLParser(HTMLParser):
    """Minimal HTML parser to extract patent details from Google Patents pages."""

    def __init__(self):
        super().__init__()
        self.in_meta = False
        self.meta_name = None
        self.abstract = None
        self.current_tag = None
        self.in_claims = False
        self.claims_text = []
        self.in_assignee = False
        self.assignee = None
        self.in_inventor = False
        self.inventors = []

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)
        if tag == "meta":
            if attrs_dict.get("name") == "DC.description":
                self.in_meta = True
        elif tag == "div" and "claims" in attrs_dict.get("class", ""):
            self.in_claims = True
        elif tag == "tr" and "assignee" in attrs_dict.get("class", ""):
            self.in_assignee = True
        elif tag == "tr" and "inventor" in attrs_dict.get("class", ""):
            self.in_inventor = True

    def handle_endtag(self, tag):
        if tag == "div" and self.in_claims:
            self.in_claims = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        if self.in_meta:
            self.abstract = data
            self.in_meta = False
        if self.in_claims:
            self.claims_text.append(data)
        if self.in_assignee and self.current_tag == "td":
            self.assignee = data
            self.in_assignee = False
        if self.in_inventor and self.current_tag == "td":
            self.inventors.append(data)
            self.in_inventor = False


API_BASE = "https://patents.google.com/xhr/query"
PATENT_BASE = "https://patents.google.com/patent"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def build_url(params: dict) -> str:
    """Build the URL-encoded query string for the Google Patents XHR API."""
    query_parts = []
    for k, v in params.items():
        if v is not None:
            query_parts.append(f"{k}={urllib.parse.quote(str(v), safe='')}")
    url_param = "&".join(query_parts)
    return f"{API_BASE}?url={urllib.parse.quote(url_param, safe='')}"


def api_request(url: str) -> dict:
    """Make a request to the Google Patents XHR API."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def cmd_search(args):
    """Search patents by keyword."""
    query_parts = [args.query.replace(" ", "+")]
    if args.assignee:
        query_parts.append(f'assignee:%22{args.assignee.replace(" ", "+")}%22')
    if args.inventor:
        query_parts.append(f'inventor:%22{args.inventor.replace(" ", "+")}%22')
    if args.cpc:
        query_parts.append(f"cpc:{args.cpc}")
    if args.office:
        query_parts.append(f"office:{args.office}")
    if args.status:
        query_parts.append(f"status:{args.status}")

    params = {"q": "+".join(query_parts), "num": str(args.num), "language": args.language}
    if args.since:
        params["date"] = f"{args.since}0101000000/20260101000000"

    url = build_url(params)
    data = api_request(url)

    results = data.get("results", {}).get("cluster", [{}])[0].get("result", [])
    total = data.get("results", {}).get("total_num_results", 0)

    output = {"total_results": total, "page": 0, "results": []}
    for r in results:
        patent = r.get("patent", {})
        output["results"].append({
            "id": r.get("id", "").replace("patent/", ""),
            "title": patent.get("title", "").replace("&hellip;", "...").replace("<b>", "").replace("</b>", ""),
            "snippet": patent.get("snippet", "").replace("<b>", "").replace("</b>", ""),
            "rank": r.get("rank"),
        })

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_detail(args):
    """Fetch full patent details from the patent page."""
    url = f"{PATENT_BASE}/{args.patent_id}/en"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode()

    parser = PatentHTMLParser()
    parser.feed(html)

    output = {
        "patent_id": args.patent_id,
        "abstract": parser.abstract,
        "assignee": parser.assignee,
        "inventors": parser.inventors if parser.inventors else None,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_citations(args):
    """Extract patent IDs from the citation section of a patent page."""
    url = f"{PATENT_BASE}/{args.patent_id}/en"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode()

    import re
    patent_ids = re.findall(r"(US|EP|WO|JP|CN|KR|DE|FR|GB)\d{7,12}[A-Z]\d?", html)
    unique_ids = sorted(set(patent_ids))

    output = {"source_patent": args.patent_id, "cited_patents_found": unique_ids[:30]}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_assignee(args):
    """Search patents by assignee (company)."""
    params = {
        "q": f'assignee:%22{args.name.replace(" ", "+")}%22',
        "num": str(args.num),
        "language": "ENGLISH",
    }
    if args.since:
        params["date"] = f"{args.since}0101000000/20260101000000"

    url = build_url(params)
    data = api_request(url)
    results = data.get("results", {}).get("cluster", [{}])[0].get("result", [])
    total = data.get("results", {}).get("total_num_results", 0)

    output = {"assignee": args.name, "total_patents": total, "results": []}
    for r in results:
        patent = r.get("patent", {})
        output["results"].append({
            "id": r.get("id", "").replace("patent/", ""),
            "title": patent.get("title", "").replace("<b>", "").replace("</b>", ""),
        })

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_cpc(args):
    """Search patents by CPC classification."""
    params = {"q": f"cpc:{args.code}", "num": str(args.num), "language": "ENGLISH"}
    url = build_url(params)
    data = api_request(url)
    results = data.get("results", {}).get("cluster", [{}])[0].get("result", [])
    total = data.get("results", {}).get("total_num_results", 0)

    output = {"cpc_class": args.code, "total_patents": total, "results": []}
    for r in results:
        patent = r.get("patent", {})
        output["results"].append({
            "id": r.get("id", "").replace("patent/", ""),
            "title": patent.get("title", "").replace("<b>", "").replace("</b>", ""),
        })

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_landscape(args):
    """Patent landscape analysis — search by topic, optionally compare assignees."""
    # Basic landscape query
    companies = [c.strip() for c in args.companies.split(",")] if args.companies else []

    output = {"topic": args.topic, "landscape": {}}

    # Overall count
    params = {"q": args.topic.replace(" ", "+"), "num": "1", "language": "ENGLISH"}
    url = build_url(params)
    data = api_request(url)
    output["landscape"]["total_patents"] = data.get("results", {}).get("total_num_results", 0)

    # Per-company counts
    if companies:
        output["landscape"]["by_assignee"] = {}
        for company in companies:
            params["q"] = f'{args.topic.replace(" ", "+")}+assignee:%22{company.replace(" ", "+")}%22'
            url = build_url(params)
            data = api_request(url)
            output["landscape"]["by_assignee"][company] = data.get("results", {}).get("total_num_results", 0)

    print(json.dumps(output, indent=2, ensure_ascii=False))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "search":
        import argparse
        p = argparse.ArgumentParser(description="Search patents by keyword")
        p.add_argument("query", help="Search query (e.g. 'blockchain consensus')")
        p.add_argument("--num", type=int, default=10, help="Results per page (max 100)")
        p.add_argument("--language", default="ENGLISH", help="Language filter")
        p.add_argument("--assignee", help="Filter by assignee (company)")
        p.add_argument("--inventor", help="Filter by inventor name")
        p.add_argument("--cpc", help="Filter by CPC classification")
        p.add_argument("--office", choices=["US", "EP", "WO", "JP", "CN", "KR", "DE", "FR", "GB"], help="Patent office filter")
        p.add_argument("--status", choices=["GRANTED", "PENDING", "EXPIRED"], help="Legal status filter")
        p.add_argument("--since", type=int, help="Year to search from (e.g. 2023)")
        args = p.parse_args(sys.argv[2:])
        cmd_search(args)

    elif command == "detail":
        import argparse
        p = argparse.ArgumentParser(description="Get patent details")
        p.add_argument("patent_id", help="Patent ID (e.g. US11074495B2)")
        args = p.parse_args(sys.argv[2:])
        cmd_detail(args)

    elif command == "citations":
        import argparse
        p = argparse.ArgumentParser(description="Get citation data for a patent")
        p.add_argument("patent_id", help="Patent ID (e.g. US11074495B2)")
        args = p.parse_args(sys.argv[2:])
        cmd_citations(args)

    elif command == "assignee":
        import argparse
        p = argparse.ArgumentParser(description="Search patents by assignee")
        p.add_argument("name", help="Assignee/company name")
        p.add_argument("--num", type=int, default=10, help="Results per page")
        p.add_argument("--since", type=int, help="Year to search from")
        args = p.parse_args(sys.argv[2:])
        cmd_assignee(args)

    elif command == "cpc":
        import argparse
        p = argparse.ArgumentParser(description="Search patents by CPC classification")
        p.add_argument("code", help="CPC code (e.g. G06N20/00)")
        p.add_argument("--num", type=int, default=10, help="Results per page")
        args = p.parse_args(sys.argv[2:])
        cmd_cpc(args)

    elif command == "landscape":
        import argparse
        p = argparse.ArgumentParser(description="Patent landscape analysis")
        p.add_argument("topic", help="Research topic")
        p.add_argument("--companies", help="Comma-separated company names to compare")
        args = p.parse_args(sys.argv[2:])
        cmd_landscape(args)

    elif command in ("--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()