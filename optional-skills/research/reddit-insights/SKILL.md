---
name: reddit-insights
description: |
  Search and analyze Reddit content using semantic AI search via reddapi.dev HTTP API.
  Use when you need to: (1) Find user pain points and frustrations for product ideas, (2) Discover niche markets or underserved needs, (3) Research what people really think about products/topics, (4) Find content inspiration from real discussions, (5) Analyze sentiment and trends on Reddit, (6) Validate business ideas with real user feedback.
  Triggers: reddit search, find pain points, market research, user feedback, what do people think about, reddit trends, niche discovery, product validation.
---

# Reddit Insights

Semantic search across millions of Reddit posts. Unlike keyword search, this understands intent and meaning.

**Powered by [reddapi.dev](https://reddapi.dev)** — AI-powered semantic search and vector similarity across 1000+ subreddits with millions of indexed posts, updated continuously.

**Key Advantage:**
- ✅ **Two search modes** - Semantic (AI summary) + Vector (fast similarity)
- ✅ **Full Reddit archive** - Access historical and real-time discussions
- ✅ **AI summaries** - Semantic search generates comprehensive summaries
- ✅ **MCP support** - Direct integration with Claude Desktop, Cursor, etc.

## Setup

### Get API Key
1. Create an account at https://reddapi.dev
2. Subscribe to a paid plan (Lite $9.90/mo, Starter $49/mo, Pro $99/mo, or Enterprise)
3. Go to https://reddapi.dev/account to view or generate your API key

### Environment Variable
```bash
export REDDAPI_API_KEY="your_api_key"
```

### Rate Limits

| Plan | Monthly API Calls | Per Minute |
|------|-------------------|------------|
| Lite | 500 | 50 |
| Starter | 5,000 | 50 |
| Pro | 15,000 | 100 |
| Enterprise | Unlimited | 1,000 |

## HTTP API Reference

**Base URL:** `https://reddapi.dev`

**Authentication:** All requests require header:
```
Authorization: Bearer YOUR_API_KEY
```

---

### POST /api/v1/search/semantic

AI-powered semantic search with keyword extraction, vector search, and AI summary generation.

```bash
curl -X POST "https://reddapi.dev/api/v1/search/semantic" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What do developers think about Rust vs Go for backend services?", "limit": 20}'
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Natural language question |
| limit | number | No | Results to return (default: 20, max: 100) |

**Response:**
```json
{
  "success": true,
  "data": {
    "query": "What do developers think about Rust vs Go for backend services?",
    "results": [
      {
        "id": "1abc234",
        "title": "Switched our microservices from Go to Rust - here's what happened",
        "content": "After 6 months of running Go in production...",
        "subreddit": "rust",
        "upvotes": 847,
        "comments": 234,
        "created": "2026-02-15T10:30:00.000Z",
        "relevance": 0.92,
        "sentiment": "Discussion",
        "url": "https://reddit.com/r/rust/comments/1abc234"
      }
    ],
    "total": 20,
    "processing_time_ms": 12450,
    "ai_summary": "Developers are divided on Rust vs Go for backend services..."
  }
}
```

---

### POST /api/v1/search/vector

Fast vector similarity search. No LLM processing, returns results in seconds.

```bash
curl -X POST "https://reddapi.dev/api/v1/search/vector" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "best productivity apps for ADHD", "limit": 30, "start_date": "2026-01-01", "end_date": "2026-03-18"}'
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Search query |
| limit | number | No | Results to return (default: 30, max: 30) |
| start_date | string | No | Start date filter (YYYY-MM-DD) |
| end_date | string | No | End date filter (YYYY-MM-DD) |

**Response:**
```json
{
  "success": true,
  "data": {
    "query": "best productivity apps for ADHD",
    "results": [
      {
        "id": "2def567",
        "title": "Finally found an app that works for my ADHD brain",
        "content": "I've tried everything from Todoist to Notion...",
        "subreddit": "ADHD",
        "upvotes": 1203,
        "comments": 456,
        "created": "2026-03-01T14:22:00.000Z",
        "similarity_score": 0.89,
        "url": "https://reddit.com/r/ADHD/comments/2def567"
      }
    ],
    "total": 30,
    "processing_time_ms": 3200
  }
}
```

---

### GET /api/v1/subreddits

List available subreddits with metadata, sorted by subscribers.

```bash
curl "https://reddapi.dev/api/v1/subreddits?search=programming&limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| search | string | No | Filter by name/title/description |
| page | number | No | Page number (default: 1) |
| limit | number | No | Results per page (default: 50, max: 100) |
| sort | string | No | Sort by "subscribers" or "created" (default: subscribers) |
| order | string | No | "asc" or "desc" (default: desc) |

---

### GET /api/v1/subreddits/{name}

Get detailed information about a specific subreddit with recent posts.

```bash
curl "https://reddapi.dev/api/v1/subreddits/webdev" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Subreddit name without `r/` prefix |

---

### POST /api/v1/trends

Get trending topics from Reddit based on post engagement.

```bash
curl -X POST "https://reddapi.dev/api/v1/trends" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2026-03-11", "end_date": "2026-03-18", "limit": 20}'
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| start_date | string | No | Start date (default: today) |
| end_date | string | No | End date (default: today) |
| limit | number | No | Topics to return (default: 20, max: 100) |

---

## MCP Server

reddapi.dev also provides a Model Context Protocol (MCP) server for direct integration with AI clients.

**Endpoint:** `https://reddapi.dev/api/mcp`
**Protocol:** MCP Streamable HTTP Transport

```json
{
  "mcpServers": {
    "reddit-search-api": {
      "transport": {
        "type": "http",
        "url": "https://reddapi.dev/api/mcp",
        "headers": {
          "Authorization": "Bearer YOUR_API_KEY"
        }
      }
    }
  }
}
```

MCP tools: `reddit_semantic_search`, `reddit_vector_search`, `reddit_list_subreddits`, `reddit_get_subreddit`, `reddit_get_trends`

---

## Choosing Between Semantic and Vector Search

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Exploratory research | Semantic | LLM extracts keywords, generates summary |
| Known-topic monitoring | Vector | Faster, direct matching, no LLM overhead |
| Batch processing | Vector | 5x faster per request |
| Answering complex questions | Semantic | AI summary synthesizes insights |
| Real-time dashboards | Vector | Low latency (~5s vs ~20s) |

## Best Use Cases (Tested)

| Use Case | Effectiveness | Why |
|----------|--------------|-----|
| Product comparisons (A vs B) | ⭐⭐⭐⭐⭐ | Reddit loves debates |
| Tool/app recommendations | ⭐⭐⭐⭐⭐ | High-intent discussions |
| Side hustle/money topics | ⭐⭐⭐⭐⭐ | Engaged communities |
| Pain point discovery | ⭐⭐⭐⭐ | Emotional posts rank well |
| Health questions | ⭐⭐⭐⭐ | Active health subreddits |
| Technical how-to | ⭐⭐⭐ | Better to search specific subreddits |
| Abstract market research | ⭐⭐ | Too vague for semantic search |
| Non-English queries | ⭐ | Reddit is English-dominant |

## Query Strategies

### ✅ Excellent Queries (relevance 0.70+)

**Product Comparisons** (best results!):
```
"Notion vs Obsidian for note taking which one should I use"
→ Relevance: 0.72-0.81 | Found: Detailed comparison discussions, user experiences

"why I switched from Salesforce to HubSpot honest experience"  
→ Relevance: 0.70-0.73 | Found: Migration stories, feature comparisons
```

**Side Hustle/Money Topics:**
```
"side hustle ideas that actually make money not scams"
→ Relevance: 0.70-0.77 | Found: Real experiences, specific suggestions
```

### ✅ Good Queries (relevance 0.60-0.69)

**Pain Point Discovery:**
```
"I hate my current CRM it is so frustrating"
→ Relevance: 0.60-0.64 | Found: Specific CRM complaints, feature wishlists
```

**Tool Evaluation:**
```
"AI tools that actually save time not just hype"
→ Relevance: 0.64-0.65 | Found: Real productivity gains, tool recommendations
```

### ❌ Weak Queries (avoid these patterns)

**Too Abstract:** "business opportunity growth potential" → 0.52-0.58
**Non-English:** "学习编程最好的方法" → 0.45-0.51

### Query Formula Cheat Sheet

| Goal | Pattern | Relevance |
|------|---------|-----------|
| Compare products | "[A] vs [B] which should I use" | 0.70-0.81 |
| Find switchers | "why I switched from [A] to [B]" | 0.70-0.73 |
| Money/hustle topics | "[topic] that actually [works/makes money] not [scam/hype]" | 0.70-0.77 |
| App recommendations | "[category] apps which one is [accurate/best] and why" | 0.67-0.72 |
| Pain points | "I hate my current [tool] it is so [frustrating/slow]" | 0.60-0.64 |
| Solutions seeking | "[problem] tried everything what actually works" | 0.60-0.63 |

## Example Workflows

**Market Research:**
```bash
curl -X POST https://reddapi.dev/api/v1/search/semantic \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "frustrated with project management tools switching from Jira", "limit": 50}'
```

**Brand Monitoring (fast):**
```bash
curl -X POST https://reddapi.dev/api/v1/search/vector \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "your-brand-name reviews complaints", "limit": 30}'
```

**Deep Research (combine both):**
1. Vector search first to quickly scope the landscape
2. Semantic search for deep analysis on specific angles

**Time-Filtered Sentiment Tracking:**
```bash
curl -X POST https://reddapi.dev/api/v1/search/vector \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "ChatGPT quality", "start_date": "2026-01-01", "end_date": "2026-03-18"}'
```

## Tips

1. **Natural language works best** - Ask questions like a human would
2. **Include context** - "for small business" or "as a developer" improves results
3. **Combine emotion words** - "frustrated", "love", "hate", "wish" find stronger opinions
4. **Filter by engagement** - High upvotes/comments = validated pain points
5. **Use vector search for speed** - ~5s vs ~20s for semantic
6. **Use date filters** - Track sentiment changes over time periods

## Error Handling

All endpoints return consistent error responses:
```json
{
  "success": false,
  "error": "Error description",
  "message": {
    "title": "Human-readable title",
    "message": "Detailed explanation",
    "cta": "Suggested action",
    "ctaLink": "/pricing"
  }
}
```

Common status codes: `400` (invalid params), `401` (bad API key), `403` (plan limit), `429` (rate limit), `500` (server error)
