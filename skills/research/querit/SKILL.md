---
name: querit
description: Search using Querit AI API — a powerful search service for retrieving relevant information from various sources.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, Search, API, Querit, Information Retrieval]
    related_skills: [arxiv, web-search]
---

# Querit AI Search

Search using Querit AI API to retrieve relevant information from various sources. This skill requires a Querit API key to function.

## When to Use

- When you need to search for specific information using the Querit AI search service
- When you want aggregated search results from multiple sources
- When you explicitly want to use Querit instead of other search methods



## Quick Reference

| Action | Command |
|--------|---------|
| Basic search | `python scripts/search_querit.py "your query"` |
| Search with limit | `python scripts/search_querit.py "your query" --count 20` |

## Setup

This skill requires a Querit API key. The key is entered interactively and stored only in memory for the current session.

### Getting an API Key

Get your API key from https://querit.ai

### Usage

On first use in each session, the script will interactively prompt for your API key:

```bash
python scripts/search_querit.py "your search query"
# Will prompt: "Please enter your Querit API key:"
```

The key is stored only in memory and will be lost when the process exits.

### Script Location

The script is located at `scripts/search_querit.py` relative to this skill directory.

## Searching

### Basic search

```bash
python scripts/search_querit.py "artificial intelligence trends"
```

### Search with custom result count

```bash
python scripts/search_querit.py "machine learning" --count 20
```



## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query string |
| `count` | integer | 10 | Number of results to return |

## Helper Script

The `scripts/search_querit.py` script provides a convenient interface:

```bash
# Basic usage
python scripts/search_querit.py "search query"

# With result count
python scripts/search_querit.py "search query" --count 15

# Pretty-printed JSON output
python scripts/search_querit.py "search query" --json
```

## Response Format

The API returns JSON. Typical response structure:

```json
{
  "took": "150ms",
  "error_code": 200,
  "error_msg": "success",
  "search_id": 12345,
  "query_context": {
    "query": "your search query"
  },
  "results": {
    "result": [
      {
        "url": "https://example.com/page",
        "page_age": "2024-01-15T08:30:00Z",
        "title": "Result Title",
        "snippet": "Description or snippet of the result...",
        "site_name": "Example Site",
        "site_icon": "https://example.com/favicon.ico"
      }
    ]
  }
}
```

### Response Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `took` | string | Required | Server-side response time |
| `error_code` | integer | Required | Error code (200 for success) |
| `error_msg` | string | Required | Error message |
| `search_id` | integer | Required | Unique request reference for support |
| `query_context` | object | Required | Information about the search query |
| `query_context.query` | string | Required | The search query that was executed |
| `results` | object | Required | Search result set |
| `results.result` | array | Required | Array of search result objects |
| `results.result[].url` | string | Optional | The URL of the search result |
| `results.result[].title` | string | Optional | The title of the search result |
| `results.result[].snippet` | string | Optional | Brief snippet from the web page |
| `results.result[].page_age` | string | Optional | Age of result in UTC+0 |
| `results.result[].site_name` | string | Optional | Website name |
| `results.result[].site_icon` | string | Optional | Favicon URL |

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Invalid API key | Enter a valid API key when prompted |
| 429 Too Many Requests | Rate limit exceeded | Wait before retrying |
| Empty results | No matches found | Try different search terms |

## Verification

To verify the skill is working:

```bash
python scripts/search_querit.py "test" --count 5
```

If configured correctly, you should see search results formatted in the output.

## Notes

- **API Key Required**: This skill requires a valid Querit API key (interactive input or environment variable)
- **Session-only Storage**: API key entered via interactive prompt is stored only in memory and lost after process exit
- Results may vary based on the search query complexity
- Use `--compressed` flag with curl for faster transfers
- JSON responses can be piped through `python3 -m json.tool` for readability
