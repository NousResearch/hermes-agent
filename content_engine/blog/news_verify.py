"""Web-grounding check for named-event claims in the AI blog stream.

Before the AI generator asserts a named current event (acquisition, regulation,
product launch), this module confirms it with a web lookup and returns grounding
snippets or an ``unverified`` flag so the generator can reframe to the durable
pattern/economics instead of stating an unverified event as fact.

Degrades to ``unverified`` (never fabricates) when the web tool is unavailable
or the network is unreachable.
"""

from __future__ import annotations

from typing import Any


def _tavily_key() -> str:
    """TAVILY_API_KEY from env; fall back to reading ~/.hermes/.env once."""
    import os
    from pathlib import Path

    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if key:
        return key
    env_path = Path.home() / ".hermes" / ".env"
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("TAVILY_API_KEY=") and "export " not in line[:8]:
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    os.environ["TAVILY_API_KEY"] = key
                    return key
    except Exception:
        pass
    return ""


def _web_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Run a web search for `query`, return up to `max_results` snippets.

    Primary: Tavily API (house key in ~/.hermes/.env) — reliable and the same
    backend the agent's web_search tool uses. Fallback: DuckDuckGo lite HTML
    scrape (may be rate-limited/blocked; these days often returns an anti-bot
    page). On any failure returns [] so the caller never gets fabricated
    evidence. Never prints credentials.
    """
    import re

    import requests

    results: list[dict[str, str]] = []

    # --- Primary: Tavily ---
    key = _tavily_key()
    if key:
        try:
            session = requests.Session()
            session.trust_env = False  # bypass stale HTTP(S)_PROXY env
            resp = session.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": max_results,
                      "search_depth": "basic"},
                headers={"Authorization": f"Bearer {key}"},
                timeout=20,
            )
            if resp.status_code == 200:
                for r in (resp.json().get("results") or [])[:max_results]:
                    results.append({
                        "title": str(r.get("title") or "").strip(),
                        "snippet": str(r.get("content") or "").strip()[:500],
                        "url": str(r.get("url") or "").strip(),
                    })
                if results:
                    return results
        except Exception:
            pass

    # --- Fallback: DuckDuckGo lite scrape (best-effort) ---
    url = "https://lite.duckduckgo.com/lite/"
    data = {"q": query}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    try:
        resp = requests.post(url, data=data, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        # Parse the HTML table rows for results.
        # Look for result rows: <tr class="result">...</tr>
        for row in re.findall(
            r'<tr class="result".*?</tr>', resp.text, re.DOTALL
        ):
            title_m = re.search(
                r'<a[^>]*class="result-link"[^>]*>(.*?)</a>', row, re.DOTALL
            )
            snippet_m = re.search(
                r'<td class="result-snippet">(.*?)</td>', row, re.DOTALL
            )
            url_m = re.search(r'href="(https?://[^"]+)"', row)
            if title_m:
                results.append({
                    "title": re.sub(r"<[^>]+>", "", title_m.group(1)).strip(),
                    "snippet": re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip()
                    if snippet_m else "",
                    "url": url_m.group(1) if url_m else "",
                })
            if len(results) >= max_results:
                break
        return results
    except Exception:
        return []


def verify_event(claim: str) -> dict[str, Any]:
    """Verify a claimed named event with a web search.

    Args:
        claim: A short description of the event, e.g. "a frontier-model
               company acquiring an AI coding tool".

    Returns:
        ``{"verified": True/False, "snippets": [...], "query": claim}``.
        ``verified`` is True when at least one result was found.
        ``snippets`` contains up to 3 result dicts with title/snippet/url.
        Never fabricates — on network failure returns unverified.
    """
    try:
        hits = _web_search(claim) or []
    except Exception:
        return {"verified": False, "snippets": [], "query": claim}
    return {
        "verified": bool(hits),
        "snippets": hits[:3],
        "query": claim,
    }
