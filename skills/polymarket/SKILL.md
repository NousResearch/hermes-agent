---
name: polymarket
description: Polymarket prediction market data - query markets, search topics, and monitor predictions
metadata:
  author: NousResearch (contributed by @Rainhoole)
  version: "1.0.0"
  tags: [finance, prediction-markets, data, polling]
  requires:
    - requests
    - click
---

# Polymarket Skill

Access [Polymarket](https://polymarket.com) prediction market data directly from Hermes Agent.

## Features

- List active markets
- Search markets by topic
- Get market details
- Monitor market changes
- Category filtering

## Usage

### List Markets

```
List top 10 active markets
```

### Search Markets

```
Search for election markets
```

### Get Market Details

```
Show details for will-trump-win-2024-election market
```

### Filter by Category

```
Show politics markets
```

## API Reference

### get_markets(limit: int = 20, category: str = None) -> List[dict]

Get active markets.

**Parameters:**
- `limit`: Number of markets to return (default: 20)
- `category`: Filter by category (optional)

**Returns:** List of market dictionaries

### search_markets(query: str, limit: int = 20) -> List[dict]

Search markets by keyword.

**Parameters:**
- `query`: Search query
- `limit`: Number of results (default: 20)

**Returns:** List of matching markets

### get_market(slug: str) -> dict

Get market details by slug.

**Parameters:**
- `slug`: Market slug

**Returns:** Market dictionary or None

## Examples

### Example 1: List Top Markets

```
Show me the top 5 prediction markets by volume
```

### Example 2: Search Elections

```
Find markets related to 2024 election
```

### Example 3: Monitor Crypto Markets

```
List cryptocurrency prediction markets
```

## Data Fields

| Field | Description |
|-------|-------------|
| slug | Market identifier |
| question | Market question |
| last_price | Current Yes price (0-100%) |
| volume_24h | 24h trading volume |
| liquidity | Market liquidity |
| category | Market category |
| url | Polymarket link |

## Rate Limits

- Public API: No authentication required
- Rate limit: ~100 requests/minute
- Recommended: Add delays between requests

## Troubleshooting

### No Data Returned

- Check internet connection
- Verify API endpoint is accessible
- Try increasing limit parameter

### Slow Response

- API may be rate limited
- Reduce limit parameter
- Add request delays

## Contributing

This skill was contributed by @Rainhoole as part of the Polymarket Signal Engine project.

## License

MIT License - See Hermes Agent LICENSE
