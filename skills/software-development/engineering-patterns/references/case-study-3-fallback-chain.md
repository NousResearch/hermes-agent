# Case Study 3: Fallback Chain — Chain of Responsibility

> Source: ReconIQ project (`~/Documents/ReconIQ/`) — implemented 2026-05-20

## Problem

`FirecrawlSearchProvider` failed with "Insufficient credits" (a billing error).
`discover_competitors()` had no fallback — one provider, one shot. When Firecrawl
ran out of credits, the entire competitor module returned empty results silently.

## Solution

`research/search_provider.py` — `FallbackSearchProvider` wraps primary + fallback:

```python
class FallbackSearchProvider(SearchProvider):
    """Chain-of-Responsibility: try primary, fall back to secondary on failure."""

    def __init__(self, primary: SearchProvider, fallback: SearchProvider):
        self._primary = primary
        self._fallback = fallback

    def _is_fallback_worthy(self, result: dict[str, Any]) -> bool:
        if not result.get("results") and not result.get("accounts"):
            return True
        for msg in result.get("data_limitations") or []:
            lower = msg.lower()
            if any(kw in lower for kw in (
                "insufficient credits", "payment required",
                "api error", "rate limit", "unauthorized", "timeout",
            )):
                return True
        return False

    def discover_competitors(self, company_profile, target_url):
        primary_result = self._primary.discover_competitors(company_profile, target_url)
        if not self._is_fallback_worthy(primary_result):
            return primary_result
        fallback_result = self._fallback.discover_competitors(company_profile, target_url)
        fallback_result["data_limitations"] = (
            (primary_result.get("data_limitations") or [])
            + (fallback_result.get("data_limitations") or [])
        )
        fallback_result["provider"] = f"{self._primary.name}+{self._fallback.name}"
        return fallback_result
```

## Configuration

Fallback chain is data, not code. In `config.yaml`:

```yaml
search:
  provider: "firecrawl"
  firecrawl:
    api_key: "${FIRECRAWL_API_KEY}"
  serpapi:
    api_key: "${SERPAPI_API_KEY}"
  fallback_chains:
    firecrawl:
      - serpapi   # SerpAPI tried when Firecrawl fails
```

The factory reads `fallback_chains` and builds the chain at startup. Adding a
third fallback is one line in config, zero lines in Python code.

## Chain of Responsibility Pattern

`FallbackSearchProvider` is the textbook Chain of Responsibility: a request
passes through handlers until one succeeds. The key insight is that the
**trigger condition** (`_is_fallback_worthy`) is centralized in one place rather
than scattered across provider classes. Any provider can be primary or fallback
— the chain composition is determined entirely by config data.

## SerpAPI Provider

Fixed-cost model ($10/month / 5,000 searches) vs Firecrawl's credits model.
Uses `urllib.request` directly, no external library dependency. Returns real
Google results including local map listings:

```python
def _search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
    import urllib.request, urllib.parse, json
    params = {"q": query, "api_key": self._api_key, "num": limit, "engine": "google"}
    url = f"https://serpapi.com/search?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode())
    organic = data.get("search_results", {}).get("organic_results", [])
    return [
        {"title": item.get("title", "") or "",
         "url": item.get("link", "") or "",
         "snippet": item.get("snippet", "") or ""}
        for item in organic[:limit] if item.get("link")
    ]
```

## Composable Chains

Three-provider chain:
```python
FallbackSearchProvider(
    FirecrawlSearchProvider(firecrawl_key),
    FallbackSearchProvider(SerpAPISearchProvider(serpapi_key), DuckDuckGoProvider())
)
```

## Design Patterns Applied

1. **Chain of Responsibility** — request propagates through handlers until one succeeds;
   trigger condition centralized; chain order from config
2. **Strategy** — concrete providers implement the same `SearchProvider` interface,
   fully interchangeable
3. **Factory** — `get_search_provider()` reads config and builds the provider graph;
   configuration format (`fallback_chains`) is data-driven, not hardcoded in Python

## Files

- `research/search_provider.py` — `SerpAPISearchProvider`, `FallbackSearchProvider`
- `tests/test_search_provider.py` — 22 tests
- `config.yaml` — `serpapi` config + `fallback_chains` section
- `.env` / `.env.example` — `SERPAPI_API_KEY` field