#!/usr/bin/env python3
"""gamefi_scan.py — starter GitHub scanner for the game research skill.

A lightweight prototype that searches GitHub for early-stage game projects
(including Web3 / GameFi games), filters to recently created ones, removes
duplicates, fetches README content where available, computes a neutral
Game Research Signal Score (0-100) from public repository signals, and
classifies each project as WATCH / TEST / CONTACT / SKIP.

It collects public repository signals only. It does NOT generate a Markdown
report yet and does NOT recommend anything.

Research only — not financial advice. All results are unverified public
signals; confirm manually.

Usage:
    python gamefi_scan.py
    python gamefi_scan.py --days 14 --limit 10 --per-keyword 10
    python gamefi_scan.py --fetch-cap 40

Auth (optional but recommended to avoid low rate limits):
    Set GITHUB_TOKEN in your environment or in a .env file next to this script.
    README fetching uses the GitHub core API (60 requests/hour without a token),
    so a token is strongly recommended for scoring. See .env.example.
"""

from __future__ import annotations

import argparse
import os
import re
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

# Keywords used to discover candidate game projects. Kept unchanged for now.
# Multi-word phrases are quoted in the query so GitHub treats them as a phrase.
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

# --- Signal keyword sets (lowercase). Used only to detect public signals. ---

# Setup / "how to run" markers in a README.
HOWTO_KEYWORDS = (
    "getting started", "installation", "install", "how to play", "how to run",
    "npm install", "yarn", "pnpm", "cargo", "pip install", "docker", "build",
    "setup", "quick start", "quickstart", "usage",
)

# Early-access / testnet markers.
EARLY_ACCESS_KEYWORDS = (
    "testnet", "devnet", "early access", "whitelist", "waitlist",
    "closed beta", "open beta", "beta", "playtest", "points",
)

# Demo / playability markers.
PLAYABILITY_KEYWORDS = (
    "playable", "play now", "live demo", "demo", "gameplay", "download",
    "itch.io", "webgl", "web build", "apk", "client", "try it",
    "play the game", "launch the game",
)

# Languages commonly seen in relevant game projects.
RELEVANT_LANGUAGES = {
    "Solidity", "Rust", "TypeScript", "Python", "C#", "JavaScript",
    "C++", "Go", "Move", "Cairo",
}

# Promotional markers (used only together with a lack of technical markers).
PROMO_KEYWORDS = (
    "to the moon", "100x", "1000x", "guaranteed", "huge gains", "next big",
    "presale", "get rich", "don't miss", "dont miss", "pump",
)

# Technical markers that indicate real substance.
TECH_KEYWORDS = (
    "install", "build", "run", "code", "function", "contract", "api", "sdk",
    "npm", "cargo", "docker", "compile", "commit", "class", "module",
)

# Web3 / GameFi domain terms. A repo is considered relevant only if its public
# text mentions at least one of these. Matched with word boundaries so generic
# names like "gamefinal" or "gamefinder" do NOT match "gamefi".
DOMAIN_KEYWORDS = (
    "web3", "blockchain", "nft", "token", "crypto", "gamefi", "play to earn",
    "onchain", "wallet", "smart contract", "testnet", "mainnet",
)
DOMAIN_PATTERNS = [
    (term, re.compile(r"\b" + re.escape(term) + r"\b"))
    for term in DOMAIN_KEYWORDS
]

# Penalty applied (and SKIP forced) when no domain term is found.
RELEVANCE_PENALTY = 25


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


def fetch_readme(full_name: str, headers: dict[str, str]) -> tuple[str | None, bool]:
    """Fetch a repo's README as raw text.

    Returns (content, available):
      - (text, True)  README present
      - (None, True)  confirmed no README (HTTP 404)
      - (None, False) could not retrieve (network error / rate limit)
    """
    url = f"https://api.github.com/repos/{full_name}/readme"
    raw_headers = dict(headers)
    raw_headers["Accept"] = "application/vnd.github.raw"
    try:
        resp = requests.get(url, headers=raw_headers, timeout=30)
    except requests.RequestException:
        return None, False

    if resp.status_code == 200:
        return resp.text, True
    if resp.status_code == 404:
        return None, True
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        return None, False
    return None, False


def repo_age_days(created_at: str | None) -> int | None:
    """Return the repository age in days, or None if the date is unparseable."""
    if not created_at:
        return None
    try:
        created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - created).days


def relevant_terms(repo: dict, readme: str | None) -> list[str]:
    """Return the Web3/GameFi domain terms found in the repo's public text.

    Searches name, full_name, description, topics, and README using
    word-boundary matching to avoid false matches inside generic names.
    """
    parts = [
        repo.get("name") or "",
        repo.get("full_name") or "",
        repo.get("description") or "",
        " ".join(repo.get("topics") or []),
        readme or "",
    ]
    hay = " ".join(parts).lower()
    return [term for term, pat in DOMAIN_PATTERNS if pat.search(hay)]


def score_project(
    repo: dict, readme: str | None, readme_known: bool
) -> dict:
    """Compute a neutral Game Research Signal Score (0-100) from public signals.

    The score reflects research signal strength (how much there is to look at,
    how active/early the project appears) — not financial merit. Returns a dict
    with the score, a component breakdown, detected signals, risk notes, and a
    few booleans used for classification.
    """
    components: list[str] = []
    signals: list[str] = []
    risks: list[str] = []
    score = 0.0

    # Base.
    score += 20
    components.append("base +20")

    # Repository activity (stars + forks), capped.
    stars = repo.get("stargazers_count") or 0
    forks = repo.get("forks_count") or 0
    activity = min(20.0, stars * 0.3 + forks * 1.5)
    score += activity
    components.append(f"activity +{activity:.0f} (stars {stars}, forks {forks})")

    # Freshness: newer is higher; 0 at ~30 days old.
    days = repo_age_days(repo.get("created_at"))
    if days is None:
        freshness = 0.0
        risks.append("unknown creation date")
    else:
        freshness = max(0.0, 15.0 - days * 0.5)
    score += freshness
    age_note = f" ({days}d old)" if days is not None else ""
    components.append(f"freshness +{freshness:.0f}{age_note}")

    # Documentation quality.
    text = readme or ""
    low = text.lower()
    readme_present = bool(text.strip())
    doc = 0.0
    if readme_present:
        doc += 10
        signals.append("README present")
        if len(text) >= 800:
            doc += 10
            signals.append("detailed README")
        elif len(text) >= 200:
            doc += 4
        if any(k in low for k in HOWTO_KEYWORDS):
            doc += 5
            signals.append("setup/run instructions")
    elif readme_known:
        risks.append("no README")
    else:
        risks.append("README not retrieved (set GITHUB_TOKEN)")
    doc = min(25.0, doc)
    score += doc
    components.append(f"docs +{doc:.0f}")

    # Testing / demo / playability indicators.
    has_test_signal = any(k in low for k in EARLY_ACCESS_KEYWORDS)
    has_play_signal = any(k in low for k in PLAYABILITY_KEYWORDS)
    play = 0.0
    if has_test_signal:
        play += 10
        signals.append("early-access/testnet signal")
    if has_play_signal:
        play += 10
        signals.append("demo/playable signal")
    play = min(20.0, play)
    score += play
    components.append(f"testing/demo +{play:.0f}")

    # Project clarity (description + relevant language).
    desc = (repo.get("description") or "").strip()
    clarity = 0.0
    if desc:
        clarity += 5
    else:
        risks.append("empty description")
    lang = repo.get("language")
    if lang in RELEVANT_LANGUAGES:
        clarity += 5
        signals.append(f"relevant language ({lang})")
    score += clarity
    components.append(f"clarity +{clarity:.0f}")

    # Domain relevance filter (reduce false positives).
    domain_terms = relevant_terms(repo, readme)
    relevant = bool(domain_terms)
    if relevant:
        signals.append("relevance: " + ", ".join(domain_terms[:5]))
        components.append("relevance ok (" + ", ".join(domain_terms[:5]) + ")")
    else:
        score -= RELEVANCE_PENALTY
        risks.append("low web3/gamefi relevance — no domain keywords found")
        components.append(f"relevance -{RELEVANCE_PENALTY} (no domain terms)")

    # Risk deductions for missing / unclear information.
    deductions = 0.0
    if readme_known and not readme_present:
        deductions += 10
    if not desc:
        deductions += 10
    unclear_purpose = not desc and len(text) < 200
    if unclear_purpose:
        deductions += 15
        risks.append("unclear purpose")
    promo_no_substance = (
        any(k in low for k in PROMO_KEYWORDS)
        and not any(k in low for k in TECH_KEYWORDS)
    )
    if promo_no_substance:
        deductions += 15
        risks.append("promotional language without technical substance")
    if deductions:
        score -= deductions
        components.append(f"risk -{deductions:.0f}")

    score = max(0.0, min(100.0, score))
    return {
        "score": round(score),
        "components": components,
        "signals": signals,
        "risks": risks,
        "readme_present": readme_present,
        "readme_known": readme_known,
        "has_desc": bool(desc),
        "has_test_signal": has_test_signal,
        "has_play_signal": has_play_signal,
        "unclear_purpose": unclear_purpose,
        "relevant": relevant,
        "domain_terms": domain_terms,
    }


def classify(s: dict) -> str:
    """Map a score result to WATCH / TEST / CONTACT / SKIP.

    Mirrors the decision rules in SKILL.md: apply the first match, top to
    bottom, preferring the more conservative category when in doubt.
    """
    # SKIP: off-topic — no Web3/GameFi domain terms in the public text.
    if not s["relevant"]:
        return "SKIP"
    # SKIP: confirmed missing docs, unclear purpose, or too little signal.
    if (s["readme_known"] and not s["readme_present"]) or s["unclear_purpose"]:
        return "SKIP"
    if s["score"] < 35:
        return "SKIP"
    # TEST: a concrete way to test/review exists, with at least some docs.
    if (s["has_play_signal"] or s["has_test_signal"]) and s["readme_present"]:
        return "TEST"
    # CONTACT: enough substance for outreach, but no open test yet.
    if s["score"] >= 55:
        return "CONTACT"
    # WATCH: interesting but needs more observation.
    return "WATCH"


def score_and_rank(
    repos: list[dict], headers: dict[str, str], fetch_cap: int
) -> list[dict]:
    """Fetch READMEs for a bounded candidate pool, score, classify, and rank.

    To bound the number of core-API requests, only the top `fetch_cap` repos
    (by stars) have their README fetched and scored.
    """
    pool = sorted(
        repos, key=lambda r: r.get("stargazers_count") or 0, reverse=True
    )[:fetch_cap]
    if len(repos) > fetch_cap:
        print(
            f"\nNote: scoring the top {fetch_cap} of {len(repos)} repos "
            "(use --fetch-cap to change).",
            file=sys.stderr,
        )

    scored: list[dict] = []
    for repo in pool:
        readme, available = fetch_readme(repo["full_name"], headers)
        result = score_project(repo, readme, available)
        result["repo"] = repo
        result["classification"] = classify(result)
        scored.append(result)

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored


def print_scored(scored: list[dict], limit: int) -> None:
    """Print the top scored projects with their classification."""
    if not scored:
        print("\nNo repositories found. Try a longer --days window.")
        return

    top = scored[:limit]
    print(f"\nScored {len(scored)} repositories. Top {len(top)}:\n")
    for i, s in enumerate(top, start=1):
        repo = s["repo"]
        signals = "; ".join(s["signals"]) or "none"
        risks = "; ".join(s["risks"]) or "none"
        print(f"{i}. {repo.get('full_name')}  —  {s['score']}/100  [{s['classification']}]")
        print(f"   {repo.get('html_url')}")
        print(
            f"   stars: {repo.get('stargazers_count')} | "
            f"forks: {repo.get('forks_count')} | "
            f"lang: {repo.get('language') or 'unknown'} | "
            f"created: {repo.get('created_at')}"
        )
        print(f"   signals: {signals}")
        print(f"   risks: {risks}")
        print(f"   score: {', '.join(s['components'])}\n")

    penalized = sum(1 for s in scored if not s.get("relevant", True))
    print(
        f"\nRelevance filter: {penalized} of {len(scored)} repositories "
        "penalized (no web3/gamefi domain terms)."
    )
    print(
        "Game Research Signal Score = research signal strength, not financial "
        "merit.\nDisclaimer: neutral research signals only — not financial "
        "advice. All results are unverified; confirm manually."
    )


def reason_for(s: dict) -> str:
    """Return a short, neutral reason for the classification."""
    cls = s["classification"]
    if cls == "SKIP":
        if not s["relevant"]:
            return "Low Web3/GameFi relevance — no domain keywords found in repo text."
        if s["readme_known"] and not s["readme_present"]:
            return "No README; not enough public information to assess."
        if s["unclear_purpose"]:
            return "Unclear purpose and minimal documentation."
        return "Weak overall signals for now."
    if cls == "TEST":
        bits = []
        if s["has_play_signal"]:
            bits.append("demo/playable signal")
        if s["has_test_signal"]:
            bits.append("early-access/testnet signal")
        detail = ", ".join(bits) or "testable signal"
        return f"Concrete way to try it ({detail}) with basic documentation."
    if cls == "CONTACT":
        return "Substantial signals and docs, but no open test yet — outreach candidate."
    return "Interesting signals; needs more observation before action."


def build_markdown_report(
    scored: list[dict], total_unique: int, days: int, generated: str, limit: int
) -> str:
    """Build a lightweight Markdown report string from scored results."""
    top = scored[:limit]
    counts = {"WATCH": 0, "TEST": 0, "CONTACT": 0, "SKIP": 0}
    for s in scored:
        counts[s["classification"]] = counts.get(s["classification"], 0) + 1
    penalized = sum(1 for s in scored if not s["relevant"])

    lines: list[str] = []
    lines.append("# Game Research — Scan Report")
    lines.append("")
    lines.append(f"**Generated:** {generated}")
    lines.append(f"**Scan window:** last {days} days")
    lines.append(f"**Projects scanned (unique):** {total_unique}")
    lines.append(f"**Projects scored:** {len(scored)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- WATCH: {counts['WATCH']} | TEST: {counts['TEST']} | "
        f"CONTACT: {counts['CONTACT']} | SKIP: {counts['SKIP']}"
    )
    lines.append(
        f"- Penalized by relevance filter (no web3/gamefi domain terms): "
        f"{penalized}"
    )
    lines.append("")
    lines.append(f"## Top {len(top)} projects (by Game Research Signal Score)")
    lines.append("")

    for i, s in enumerate(top, start=1):
        repo = s["repo"]
        full_name = repo.get("full_name") or "unknown"
        url = repo.get("html_url") or ""
        signals = ", ".join(s["signals"]) or "none"
        risks = ", ".join(s["risks"]) or "none"
        breakdown = ", ".join(s["components"])
        if s["readme_present"]:
            readme_src = f"{url}#readme"
        elif s["readme_known"]:
            readme_src = "none found"
        else:
            readme_src = "not retrieved"

        lines.append(f"### {i}. {full_name} — {s['score']}/100 [{s['classification']}]")
        lines.append("")
        lines.append(f"- **Recommendation:** {s['classification']} — {reason_for(s)}")
        lines.append(
            f"- **Stats:** stars {repo.get('stargazers_count')} | "
            f"forks {repo.get('forks_count')} | "
            f"language {repo.get('language') or 'unknown'} | "
            f"created {repo.get('created_at')}"
        )
        lines.append(f"- **Detected signals:** {signals}")
        lines.append(f"- **Risk notes:** {risks}")
        lines.append(f"- **Score breakdown:** {breakdown}")
        lines.append("- **Sources:**")
        lines.append(f"  - Repository: {url}")
        lines.append(f"  - README: {readme_src}")
        lines.append("")

    lines.append("## Manual verification")
    lines.append("")
    lines.append(
        "All signals above are automated and **unverified**. Before acting on "
        "any entry, open the repository, read the README, and confirm the "
        "detected signals (testnet / demo / early access, etc.) are real and "
        "current."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Game Research Signal Score reflects research signal strength, not "
        "financial merit. Neutral research summary — not financial advice. "
        "Verify all project claims manually.*"
    )
    lines.append("")
    return "\n".join(lines)


def write_report(
    scored: list[dict],
    total_unique: int,
    days: int,
    limit: int,
    reports_dir: Path,
) -> Path:
    """Write the Markdown report to a timestamped file and return its path."""
    now = datetime.now(timezone.utc)
    generated = now.strftime("%Y-%m-%d %H:%M UTC")
    stamp = now.strftime("%Y%m%d-%H%M%S")
    content = build_markdown_report(scored, total_unique, days, generated, limit)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"game-research-report-{stamp}.md"
    path.write_text(content, encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Starter GitHub scanner + scorer for early-stage game projects."
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
        default=10,
        help="Number of top results to print (default: 10).",
    )
    parser.add_argument(
        "--per-keyword",
        type=int,
        default=30,
        help="Max results to fetch per keyword, 1-100 (default: 30).",
    )
    parser.add_argument(
        "--fetch-cap",
        type=int,
        default=40,
        help="Max repos to fetch README + score, to bound API use (default: 40).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also write a Markdown report to the reports/ folder.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for the report (default: ../reports next to this script).",
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
            "(lower rate limit; README scoring may be limited).\n",
            file=sys.stderr,
        )

    results = scan(args.days, args.per_keyword, headers)
    scored = score_and_rank(results, headers, args.fetch_cap)
    print_scored(scored, args.limit)

    if args.report:
        reports_dir = (
            Path(args.report_dir)
            if args.report_dir
            else script_dir.parent / "reports"
        )
        path = write_report(scored, len(results), args.days, args.limit, reports_dir)
        print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
