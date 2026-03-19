"""
Cosmos Web Search Tool — DuckDuckGo Integration
================================================
Gives Cosmos the ability to search the web and ground responses in real data.

Uses DuckDuckGo HTML API (no API key required).
Falls back to DuckDuckGo Lite if the main endpoint fails.

Usage:
    results = await search_web("latest AI news", max_results=5)
    context = format_search_context(results)
"""

import re
import logging
import html
from typing import List, Dict, Optional
from urllib.parse import quote_plus

logger = logging.getLogger("COSMOS_SEARCH")

# Lazy import httpx to avoid startup cost
_httpx = None

def _get_httpx():
    global _httpx
    if _httpx is None:
        import httpx
        _httpx = httpx
    return _httpx


async def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo HTML.
    
    Returns list of dicts: [{"title": ..., "snippet": ..., "url": ...}, ...]
    """
    httpx = _get_httpx()
    results = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            
            if resp.status_code != 200:
                logger.warning(f"DuckDuckGo returned {resp.status_code}")
                return []
            
            body = resp.text
            
            # Parse results from HTML
            # DuckDuckGo HTML returns results in <a class="result__a"> tags
            result_blocks = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                body,
                re.DOTALL
            )
            
            for url_raw, title_raw, snippet_raw in result_blocks[:max_results]:
                # Clean HTML entities and tags
                title = _clean_html(title_raw).strip()
                snippet = _clean_html(snippet_raw).strip()
                
                # DuckDuckGo wraps URLs in a redirect — extract the real URL
                real_url = _extract_url(url_raw)
                
                if title and snippet:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": real_url
                    })
            
            if not results:
                # Fallback: try simpler regex for DuckDuckGo Lite format
                lite_blocks = re.findall(
                    r'<td[^>]*class="result-link"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                    r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
                    body,
                    re.DOTALL
                )
                for url_raw, title_raw, snippet_raw in lite_blocks[:max_results]:
                    title = _clean_html(title_raw).strip()
                    snippet = _clean_html(snippet_raw).strip()
                    real_url = _extract_url(url_raw)
                    if title and snippet:
                        results.append({
                            "title": title,
                            "snippet": snippet,
                            "url": real_url
                        })
        
        logger.info(f"[SEARCH] '{query}' → {len(results)} results")
        
    except Exception as e:
        logger.error(f"[SEARCH] Error: {e}")
    
    return results


def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _extract_url(url_raw: str) -> str:
    """Extract real URL from DuckDuckGo redirect wrapper."""
    # DuckDuckGo HTML wraps URLs: //duckduckgo.com/l/?uddg=ENCODED_URL&...
    match = re.search(r'uddg=([^&]+)', url_raw)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1))
    # Already a direct URL
    if url_raw.startswith('http'):
        return url_raw
    return url_raw


def format_search_context(results: List[Dict[str, str]], max_chars: int = 2000) -> str:
    """
    Format search results into a context block for the LLM synthesis prompt.
    
    Returns a concise block like:
    
    WEB SEARCH RESULTS:
    [1] Title (url)
        Snippet text...
    [2] Title (url)
        Snippet text...
    """
    if not results:
        return ""
    
    lines = ["WEB SEARCH RESULTS:"]
    total = 0
    
    for i, r in enumerate(results, 1):
        entry = f"[{i}] {r['title']} ({r['url']})\n    {r['snippet']}"
        total += len(entry)
        if total > max_chars:
            break
        lines.append(entry)
    
    return "\n".join(lines)


def should_search(message: str) -> bool:
    """
    Detect if a user message would benefit from a web search.
    
    Returns True for:
    - Questions about current events, news, prices
    - "Search for", "look up", "find out"
    - "What is", "Who is", "When did" (factual questions)
    - "Latest", "recent", "today", "current"
    """
    msg_lower = message.lower()
    
    # Explicit search requests
    explicit_triggers = [
        "search for", "look up", "find out", "google",
        "search the web", "web search", "look online",
    ]
    if any(t in msg_lower for t in explicit_triggers):
        return True
    
    # Current events / time-sensitive
    current_triggers = [
        "latest", "recent", "today", "current", "right now",
        "this week", "this month", "breaking", "news about",
        "what happened", "update on",
    ]
    if any(t in msg_lower for t in current_triggers):
        return True
    
    # Factual questions that benefit from search
    factual_patterns = [
        r"^(what|who|when|where|how|why)\s+(is|are|was|were|did|does|do|can|will)\s+",
        r"^(tell me about|explain|describe)\s+",
        r"(price of|cost of|worth of)",
        r"(how to|how do i|steps to)",
    ]
    for pattern in factual_patterns:
        if re.search(pattern, msg_lower):
            return True
    
    return False
