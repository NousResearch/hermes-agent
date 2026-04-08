# Brave Search Tool

Hermes now includes a dedicated `brave_search` tool for querying Brave Search directly, without overloading the generic `web_search` / `web_extract` backend abstraction.

## Why this exists

Brave Search fits cleanly as a search-only capability.

Instead of pretending Brave is a full replacement for Hermes' generic web backend stack, Hermes exposes it as its own first-class tool:

- `brave_search` for Brave-native search
- `web_search` / `web_extract` remain backed by the existing multi-provider web stack

This keeps capability boundaries honest and avoids implying that Brave supports arbitrary page extraction in the same way as Firecrawl, Tavily, Exa, or Parallel.

## Requirements

Set this environment variable in `~/.hermes/.env`:

```env
BRAVE_API_KEY=your_brave_search_api_key
```

Brave Search uses the API documented at:
- https://brave.com/search/api/

## Tool name

`brave_search`

## Parameters

- `query` (string, required): search query
- `count` (integer, optional): number of results to return, default `5`, capped to `10` at the tool schema layer

## Return shape

The tool returns Hermes-standard search result JSON:

```json
{
  "success": true,
  "data": {
    "web": [
      {
        "title": "Example title",
        "url": "https://example.com",
        "description": "Example description",
        "position": 1
      }
    ]
  }
}
```

## Implementation notes

Implementation lives in:
- `tools/web_tools.py`

Registration details:
- tool name: `brave_search`
- toolset: `web`
- env requirement: `BRAVE_API_KEY`

The tool currently:
- calls Brave's web search endpoint directly
- authenticates with `X-Subscription-Token`
- normalizes Brave results into Hermes' standard search result shape

## Example usage

```python
brave_search({
  "query": "Hermes Agent Brave Search integration",
  "count": 5
})
```

## Testing

Dedicated tests live at:
- `tests/tools/test_brave_search_tool.py`

Covered behavior includes:
- API key gating
- Brave request header and parameter construction
- response normalization
- tool registration
- toolset inclusion

## Design rationale

This tool was intentionally added as a separate capability instead of a new generic web backend because that design is cleaner:

- avoids conflating search with extraction
- avoids misleading tool availability reporting
- preserves the existing generic web backend model
- leaves room for future Brave-specific tools such as:
  - `brave_news_search`
  - `brave_image_search`
  - `brave_llm_context`
