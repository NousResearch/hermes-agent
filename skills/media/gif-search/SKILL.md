---
name: gif-search
description: Search and download GIFs from Giphy using curl. No dependencies beyond curl and jq. Useful for finding reaction GIFs, creating visual content, and sending GIFs in chat.
version: 2.0.0
author: Hermes Agent
license: MIT
required_environment_variables:
  - name: GIPHY_API_KEY
    prompt: Giphy API key
    help: Get a free API key from https://developers.giphy.com
metadata:
  hermes:
    tags: [GIF, Media, Search, Giphy, API]
---

# GIF Search (Giphy API)

Search and download GIFs directly via the Giphy API using curl. No extra tools needed.

## Setup

Set your Giphy API key in your environment (add to `~/.hermes/.env`):

```bash
GIPHY_API_KEY=your_key_here
```

Get a free API key at https://developers.giphy.com — the Giphy API has a free tier with generous rate limits (42 requests per hour).

## Prerequisites

- `curl` and `jq` (both standard on macOS/Linux)
- `GIPHY_API_KEY` environment variable

## Search for GIFs

```bash
# Search and get GIF URLs
curl -s "https://api.giphy.com/v1/gifs/search?q=thumbs+up&limit=5&api_key=${GIPHY_API_KEY}" | jq -r '.data[].images.original.url'

# Get smaller/preview versions
curl -s "https://api.giphy.com/v1/gifs/search?q=nice+work&limit=3&api_key=${GIPHY_API_KEY}" | jq -r '.data[].images.fixed_height.url'
```

## Download a GIF

```bash
# Search and download the top result
URL=$(curl -s "https://api.giphy.com/v1/gifs/search?q=celebration&limit=1&api_key=${GIPHY_API_KEY}" | jq -r '.data[0].images.original.url')
curl -sL "$URL" -o celebration.gif
```

## Get Full Metadata

```bash
curl -s "https://api.giphy.com/v1/gifs/search?q=cat&limit=3&api_key=${GIPHY_API_KEY}" | jq '.data[] | {title: .title, url: .images.original.url, preview: .images.fixed_height.url, dimensions: .images.original}'
```

## API Parameters

| Parameter | Description |
|-----------|-------------|
| `q` | Search query (URL-encode spaces as `+`) |
| `limit` | Max results (1-50, default 25) |
| `api_key` | API key (from `$GIPHY_API_KEY` env var) |
| `rating` | Content rating: `g`, `pg`, `pg-13`, `r` |
| `lang` | Language: `en`, `es`, `fr`, etc. |
| `offset` | Pagination offset |

## Available Image Formats

Each result has multiple formats under `.images`:

| Format | Use case |
|--------|----------|
| `original` | Full quality GIF |
| `fixed_height` | Small preview GIF (fixed height) |
| `fixed_width` | Small preview GIF (fixed width) |
| `downsized` | Optimized smaller GIF |
| `preview_gif` | Very small preview GIF |
| `preview_webp` | WebP preview |

## Notes

- URL-encode the query: spaces as `+`, special chars as `%XX`
- For sending in chat, `fixed_height` URLs are lighter weight
- GIF URLs can be used directly in markdown: `![alt](url)`
- Rate limit: 42 requests per hour on the free tier
