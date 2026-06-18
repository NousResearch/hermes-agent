"""Primary-source backfill — official-domain search + GitHub provenance (defensive OSINT).

Uses public APIs and site-constrained search (e-Gov first, then DDGS ``site:`` queries).
Does NOT implement bot-evasion or stealth crawling — only polite, standards-compliant HTTP.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import milspec_prose

# Official GitHub repos for methodology / toolchain provenance (primary for pipeline metadata)
GITHUB_TOOLCHAIN_REPOS: list[dict[str, str]] = [
    {
        "repo": "koala73/worldmonitor",
        "role": "World Monitor OSINT data plane",
        "url": "https://github.com/koala73/worldmonitor",
    },
    {
        "repo": "takurot/egov-law-mcp",
        "role": "e-Gov Law API MCP（日本法令一次資料）",
        "url": "https://github.com/takurot/egov-law-mcp",
    },
    {
        "repo": "NousResearch/hermes-agent",
        "role": "Hermes PDB / worldmonitor-osint plugin",
        "url": "https://github.com/NousResearch/hermes-agent",
    },
]

PRIMARY_SITE_SUFFIXES = (
    "site:go.jp",
    "site:gov",
    "site:mil",
    "site:int",
    "site:europa.eu",
)

HEADLINE_QUERY_MAX_WORDS = 10
BACKFILL_PER_HEADLINE = 2


def _ddgs_site_search(query: str, *, limit: int = 3) -> list[dict[str, str]]:
    """Site-constrained search via ddgs (no API key)."""
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        return []

    safe_limit = max(1, min(limit, 5))
    hits: list[dict[str, str]] = []
    try:
        with DDGS() as client:
            for i, row in enumerate(client.text(query, max_results=safe_limit)):
                if i >= safe_limit:
                    break
                url = str(row.get("href") or row.get("url") or "")
                if not url:
                    continue
                hits.append(
                    {
                        "title": str(row.get("title") or ""),
                        "url": url,
                        "description": str(row.get("body") or "")[:240],
                    }
                )
    except Exception:
        return []
    return hits


def _headline_search_terms(title: str) -> str:
    words = re.findall(r"[\w\u3040-\u30ff\u4e00-\u9fff]+", title or "")
    return " ".join(words[:HEADLINE_QUERY_MAX_WORDS])


def backfill_headline_primary(headline: dict[str, Any]) -> dict[str, Any]:
    """Try to find a PRIMARY official URL for a secondary WM headline."""
    title = (headline.get("title") or "").strip()
    original_url = (headline.get("url") or "").strip()
    original_tier = milspec_prose.classify_source_tier(original_url)

    if original_tier == milspec_prose.SOURCE_TIER_PRIMARY:
        return {
            "title": title,
            "original_url": original_url,
            "primary_url": original_url,
            "primary_title": title,
            "source_tier": milspec_prose.SOURCE_TIER_PRIMARY,
            "backfill_method": "already_primary",
        }

    terms = _headline_search_terms(title)
    if not terms:
        return {
            "title": title,
            "original_url": original_url,
            "primary_url": "",
            "source_tier": milspec_prose.SOURCE_TIER_UNVERIFIED,
            "backfill_method": "no_terms",
        }

    site_clause = " OR ".join(PRIMARY_SITE_SUFFIXES)
    query = f"{terms} ({site_clause})"
    candidates = _ddgs_site_search(query, limit=BACKFILL_PER_HEADLINE)

    for hit in candidates:
        url = hit.get("url") or ""
        tier = milspec_prose.classify_source_tier(url)
        if tier == milspec_prose.SOURCE_TIER_PRIMARY:
            return {
                "title": title,
                "original_url": original_url,
                "primary_url": url,
                "primary_title": hit.get("title") or title,
                "source_tier": tier,
                "backfill_method": "ddgs_site_primary",
                "search_query": query,
            }

    # Best secondary from official-domain search (still not PRIMARY tier)
    if candidates:
        hit = candidates[0]
        url = hit.get("url") or ""
        return {
            "title": title,
            "original_url": original_url,
            "primary_url": url,
            "primary_title": hit.get("title") or title,
            "source_tier": milspec_prose.classify_source_tier(url),
            "backfill_method": "ddgs_site_best_effort",
            "search_query": query,
        }

    return {
        "title": title,
        "original_url": original_url,
        "primary_url": "",
        "source_tier": milspec_prose.SOURCE_TIER_UNVERIFIED,
        "backfill_method": "not_found",
        "search_query": query,
    }


def github_toolchain_provenance(*, topic: str = "") -> dict[str, Any]:
    """Fetch GitHub metadata for known toolchain repos (defensive provenance)."""
    refs: list[dict[str, Any]] = []
    for entry in GITHUB_TOOLCHAIN_REPOS:
        repo = entry["repo"]
        api_url = f"https://api.github.com/repos/{repo}"
        req = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "hermes-worldmonitor-osint-pdb",
            },
            method="GET",
        )
        row: dict[str, Any] = {
            "repo": repo,
            "role": entry.get("role"),
            "html_url": entry.get("url") or f"https://github.com/{repo}",
            "source_tier": "PRIMARY",
            "source_type": "github_official_repo",
        }
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            row.update(
                {
                    "description": (data.get("description") or "")[:200],
                    "default_branch": data.get("default_branch"),
                    "updated_at": data.get("updated_at"),
                    "citation": f"[出典: GitHub {repo} @ {data.get('updated_at')}] {row['html_url']}",
                }
            )
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as exc:
            row["error"] = str(exc)[:200]
            row["citation"] = f"[出典: GitHub {repo}] {row['html_url']}"
        refs.append(row)

    # Optional topic search on GitHub (public API, rate-limited)
    topic_hits: list[dict[str, Any]] = []
    if topic.strip():
        q = urllib.parse.quote(f"{topic} japan security in:name,description")
        search_url = f"https://api.github.com/search/repositories?q={q}&sort=updated&per_page=3"
        req = urllib.request.Request(
            search_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "hermes-worldmonitor-osint-pdb",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            for item in (payload.get("items") or [])[:3]:
                full_name = item.get("full_name") or ""
                html_url = item.get("html_url") or ""
                topic_hits.append(
                    {
                        "repo": full_name,
                        "html_url": html_url,
                        "description": (item.get("description") or "")[:180],
                        "updated_at": item.get("updated_at"),
                        "citation": f"[出典: GitHub search — {full_name}] {html_url}",
                        "source_tier": "SECONDARY",
                        "source_type": "github_search",
                    }
                )
        except Exception:
            pass

    return {
        "success": True,
        "toolchain_repos": refs,
        "topic_search_hits": topic_hits,
        "api_docs": "https://docs.github.com/en/rest",
    }


def _fetch_gov_feeds_digest(*, hours: int = 24, max_per_feed: int = 8) -> dict[str, Any]:
    """Load scrapling-feeds sibling plugin and digest enabled government RSS."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    plugin_dir = Path(__file__).resolve().parents[1] / "scrapling-feeds"
    if not (plugin_dir / "gov_digest.py").is_file():
        return {"skipped": True, "reason": "scrapling-feeds plugin not installed"}

    pkg_name = "hermes_scrapling_feeds"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
        for stem in ("feeds_catalog", "fetcher", "rss_parse", "gov_digest"):
            mod_name = f"{pkg_name}.{stem}"
            spec = importlib.util.spec_from_file_location(
                mod_name, plugin_dir / f"{stem}.py"
            )
            if spec is None or spec.loader is None:
                return {"skipped": True, "reason": f"import failed: {stem}"}
            module = importlib.util.module_from_spec(spec)
            module.__package__ = pkg_name
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

    digest_mod = sys.modules.get(f"{pkg_name}.gov_digest")
    if digest_mod is None:
        return {"skipped": True, "reason": "gov_digest module missing"}
    return digest_mod.digest_feeds(hours=hours, max_per_feed=max_per_feed)


def enrich_primary_sources(
    threats: dict[str, Any],
    *,
    topic: str = "",
    max_headline_backfill: int = 5,
    fetch_egov: bool = True,
    fetch_github: bool = True,
    fetch_gov_feeds: bool = True,
    gov_feed_hours: int = 24,
    gov_feed_max_per_feed: int = 8,
) -> dict[str, Any]:
    """Full primary-source enrichment pass for PDB generation."""
    from . import egov_primary

    result: dict[str, Any] = {
        "egov": {"skipped": True},
        "headline_backfill": [],
        "github": {"skipped": True},
        "gov_feeds": {"skipped": True},
        "stats": {},
    }

    if fetch_egov:
        result["egov"] = egov_primary.fetch_security_law_citations(max_entries=5)

    headlines = threats.get("high_threat_headlines") or []
    backfill_rows: list[dict[str, Any]] = []
    primary_found = 0
    for item in headlines[: max(1, min(max_headline_backfill, 8))]:
        if not isinstance(item, dict):
            continue
        row = backfill_headline_primary(item)
        backfill_rows.append(row)
        if row.get("source_tier") == milspec_prose.SOURCE_TIER_PRIMARY and row.get("primary_url"):
            primary_found += 1

    result["headline_backfill"] = backfill_rows

    if fetch_github:
        result["github"] = github_toolchain_provenance(topic=topic)

    if fetch_gov_feeds:
        result["gov_feeds"] = _fetch_gov_feeds_digest(
            hours=gov_feed_hours,
            max_per_feed=gov_feed_max_per_feed,
        )

    egov_ok = (result.get("egov") or {}).get("fetched") or 0
    gov = result.get("gov_feeds") or {}
    gov_entries = gov.get("total_entries") or 0
    result["stats"] = {
        "egov_citations_ok": egov_ok,
        "headlines_backfilled": len(backfill_rows),
        "headlines_primary_resolved": primary_found,
        "gov_feed_entries": gov_entries,
        "gov_feeds_ok": gov.get("feeds_ok") or 0,
        "methodology": (
            "e-Gov Law API v2 (PRIMARY) + government RSS (scrapling-feeds) + "
            "site:-constrained DDGS backfill + GitHub REST provenance"
        ),
    }
    result["success"] = bool(
        egov_ok or primary_found or backfill_rows or gov_entries
    )
    return result


def apply_enrichment_to_threats(
    threats: dict[str, Any],
    enrichment: dict[str, Any],
) -> dict[str, Any]:
    """Merge PRIMARY backfill URLs into headline list for KEY DEVELOPMENTS."""
    backfill_by_title = {
        (row.get("title") or ""): row
        for row in (enrichment.get("headline_backfill") or [])
        if isinstance(row, dict)
    }
    merged: list[dict[str, Any]] = []
    for item in threats.get("high_threat_headlines") or []:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or ""
        row = backfill_by_title.get(title)
        copy = dict(item)
        if row:
            copy["backfill_method"] = row.get("backfill_method")
            if row.get("primary_url"):
                copy["url"] = row["primary_url"]
                copy["original_wm_url"] = item.get("url") or ""
        merged.append(copy)
    out = dict(threats)
    out["high_threat_headlines"] = merged
    return out
