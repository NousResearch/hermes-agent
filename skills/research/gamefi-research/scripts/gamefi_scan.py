#!/usr/bin/env python3
"""gamefi_scan.py — starter GitHub scanner for the gamefi-research skill.

A lightweight prototype that searches GitHub for early-stage Web3 gaming
("GameFi") repositories, filters to recently created ones, removes duplicates,
and prints the results in the terminal.

This is a *discovery* helper for the gamefi-research workflow. It collects
public repository signals only. It does NOT score, summarize, or recommend
anything — scoring arrives in a later milestone.

Research only — not financial advice, not a trading tool, not an investment
recommendation. All results are unverified public signals; confirm manually.

Usage:
    python gamefi_scan.py
    python gamefi_scan.py --days 14 --limit 25 --per-keyword 30

Auth (optional but recommended to avoid low rate limits):
    Set GITHUB_TOKEN in your environment or in a .env file next to this script.
    See .env.example.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit(
        "Missing dependency 'requests'. Install it with:\n"
        "    pip install -r requirements.txt"
    )

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"

# Keywords used to discover candidate game projects. Multi-word phrases are
# quoted in the query so GitHub treats them as a phrase.
KEYWORDS: list[str] = [
    "gamefi",
    "web3 game",
    "onchain game",
    "crypto game",
    "blockchain game",
    "unity blockchain game",
    "solana game",
    "base game",
    "ronin game",
    "abstract game",
]

# Fields we keep from each repository result.
FIELDS = (
    "name",
    "full_name",
    "html_url",
    "description",
    "created_at",
    "stargazers_count",
    "forks_count",
    "language",
    "topics",
)


def load_dotenv(script_dir: Path) -> None:
    """Load KEY=VALUE pairs from a .env file next to this script, if present.

    Kept dependency-free on purpose. Existing environment variables win.
    """
    env_path = script_dir / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        print(f"warning: could not read .env ({exc})", file=sys.stderr)


def build_headers() -> dict[str, str]:
    """Build request headers, adding auth if a token is available."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "gamefi-research-scanner",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def search_keyword(
    keyword: str,
    since: str,
    per_keyword: int,
    headers: dict[str, str],
) -> list[dict]:
    """Search GitHub for one keyword. Returns a list of raw repo dicts.

    Returns an empty list on any error so one failed keyword does not abort
    the whole scan.
    """
    phrase = f'"{keyword}"' if " " in keyword else keyword
    query = f"{phrase} created:>={since}"
    params = {
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": min(per_keyword, 100),
    }
    try:
        resp = requests.get(
            GITHUB_SEARCH_URL, headers=headers, params=params, timeout=30
        )
    except requests.RequestException as exc:
        print(f"  ! request failed for '{keyword}': {exc}", file=sys.stderr)
        return []

    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        print(
            f"  ! rate limited on '{keyword}'. Set GITHUB_TOKEN to raise the "
            "limit.",
            file=sys.stderr,
        )
        return []
    if resp.status_code != 200:
        print(
            f"  ! '{keyword}' returned HTTP {resp.status_code}: "
            f"{resp.text[:200]}",
            file=sys.stderr,
        )
        return []

    try:
        items = resp.json().get("items", [])
    except ValueError:
        print(f"  ! could not parse JSON for '{keyword}'", file=sys.stderr)
        return []
    return items


def slim(repo: dict) -> dict:
    """Reduce a raw repo dict to the fields we care about."""
    return {field: repo.get(field) for field in FIELDS}


def scan(days: int, per_keyword: int, headers: dict[str, str]) -> list[dict]:
    """Run the scan across all keywords and return deduplicated results."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%d"
    )
    print(f"Scanning GitHub for projects created since {since} ...\n")

    seen: set[str] = set()
    results: list[dict] = []
    for keyword in KEYWORDS:
        items = search_keyword(keyword, since, per_keyword, headers)
        print(f"  '{keyword}': {len(items)} result(s)")
        for repo in items:
            full_name = repo.get("full_name")
            if not full_name or full_name in seen:
                continue
            seen.add(full_name)
            results.append(slim(repo))
    return results


def print_results(results: list[dict], limit: int) -> None:
    """Print the top results to the terminal, sorted by stars (desc)."""
    if not results:
        print("\nNo repositories found. Try a longer --days window.")
        return

    ranked = sorted(
        results, key=lambda r: r.get("stargazers_count") or 0, reverse=True
    )[:limit]

    print(f"\nFound {len(results)} unique repositories. Top {len(ranked)}:\n")
    for i, repo in enumerate(ranked, start=1):
        topics = ", ".join(repo.get("topics") or []) or "none"
        desc = (repo.get("description") or "").strip() or "(no description)"
        print(f"{i}. {repo.get('full_name')}")
        print(f"   {repo.get('html_url')}")
        print(
            f"   stars: {repo.get('stargazers_count')} | "
            f"forks: {repo.get('forks_count')} | "
            f"lang: {repo.get('language') or 'unknown'} | "
            f"created: {repo.get('created_at')}"
        )
        print(f"   topics: {topics}")
        print(f"   {desc}\n")

    print(
        "Disclaimer: neutral research signals only — not financial advice. "
        "All results are unverified; confirm manually."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Starter GitHub scanner for early-stage Web3 game projects."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Only include repos created within the last N days (default: 30).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of top results to print (default: 25).",
    )
    parser.add_argument(
        "--per-keyword",
        type=int,
        default=30,
        help="Max results to fetch per keyword, 1-100 (default: 30).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.days < 1:
        print("error: --days must be >= 1", file=sys.stderr)
        return 2

    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir)
    headers = build_headers()
    if "Authorization" not in headers:
        print(
            "note: no GITHUB_TOKEN found — using unauthenticated requests "
            "(lower rate limit).\n",
            file=sys.stderr,
        )

    results = scan(args.days, args.per_keyword, headers)
    print_results(results, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
