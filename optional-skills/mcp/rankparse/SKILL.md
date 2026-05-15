---
name: rankparse
description: Fetch SEO signals for any domain via RankParse MCP.
version: 1.0.0
author: abhibavishi
license: MIT-0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [seo, backlinks, mcp, domain-authority, link-building]
    category: research
    related_skills: [duckduckgo-search, scrapling]
---

# RankParse Skill

RankParse exposes 18 SEO tools over MCP — backlinks, domain authority, tech stack, referring domains, competitor overlap, and more. All data is sourced from pre-processed Common Crawl snapshots. No subscription required; credits from $0.009/call.

## When to Use

Use this skill when the user asks about:

- Backlinks to a domain or page
- Domain authority or trust score
- Referring domains and anchor text distribution
- Which pages on a site earn the most inbound links
- Domains linking to multiple competitors (link gap / overlap)
- Shared linkers between two domains
- Similar domains for outreach prospecting
- Tech stack a site is running
- Page metadata (title, description, canonical, OG tags)
- Crawl history, status codes, or content types

## Prerequisites

1. Get a free API key (100 credits) at https://rankparse.com/signup
2. Add the MCP server to your Hermes config:

```json
{
  "mcpServers": {
    "rankparse": {
      "url": "https://mcp.rankparse.com/mcp",
      "headers": {
        "X-API-Key": "rp_your_key_here"
      }
    }
  }
}
```

Or add via CLI:

```bash
hermes mcp add rankparse --url https://mcp.rankparse.com/mcp --header "X-API-Key: rp_your_key_here"
```

No local install required. All tools run server-side.

## How to Run

Once the MCP server is connected, call tools directly:

```
get_domain_authority(domain: "example.com")
get_backlinks(domain: "example.com", limit: 50)
get_domain_overlap(domains: ["competitor-a.com", "competitor-b.com"])
```

The agent selects the right tool based on the user's request. No manual tool selection needed.

## Quick Reference

| Tool | Input | Credits |
|---|---|---|
| `get_domain_authority` | `domain` | 1 |
| `get_backlinks` | `domain`, `limit` | 2 |
| `get_referring_domains` | `domain`, `limit` | 2 |
| `get_anchor_text` | `domain` | 2 |
| `get_top_pages` | `domain`, `limit` | 2 |
| `get_outbound_links` | `domain`, `limit` | 2 |
| `get_domain_rank` | `domain` | 2 |
| `get_site_explorer` | `domain` | 10 |
| `get_page_meta` | `url` | 2 |
| `get_tech_stack` | `domain` | 2 |
| `get_url_index` | `domain`, `limit` | 2 |
| `get_crawl_history` | `domain` | 2 |
| `get_status_codes` | `domain` | 2 |
| `get_content_types` | `domain` | 2 |
| `get_language` | `domain` | 2 |
| `get_domain_overlap` | `domains[]` | 5 |
| `get_link_intersect` | `domain_a`, `domain_b` | 5 |
| `get_similar_domains` | `domain`, `limit` | 5 |

## Procedure

### Competitor link gap analysis

Find domains linking to competitors but not to you:

1. Call `get_backlinks` for your domain
2. Call `get_backlinks` for each competitor
3. Call `get_domain_overlap` with all domains as input
4. Call `get_domain_authority` on gap domains to rank by authority

### Backlink profile audit

1. Call `get_referring_domains` — unique domains linking in
2. Call `get_anchor_text` — check for over-optimized anchors
3. Call `get_backlinks` — inspect top individual links

### Outreach prospecting

1. Call `get_similar_domains` on a target site
2. Call `get_link_intersect` between two competitors
3. Call `get_domain_authority` to filter by DA threshold

### Site technical audit

1. Call `get_tech_stack` — CMS, CDN, analytics, frameworks
2. Call `get_page_meta` on key URLs — title, canonical, OG tags
3. Call `get_crawl_history` — first seen, last crawled

## Pitfalls

- Data is sourced from Common Crawl snapshots (~quarterly). It is not real-time — very recently acquired or lost links may not appear.
- `get_domain_overlap` and `get_link_intersect` cost 5 credits each. For large competitor lists, call `get_referring_domains` first and intersect manually to save credits.
- `get_site_explorer` costs 10 credits — use for full overviews only, not for single-metric lookups.
- Domains with no crawl history return `200` with an empty `data` array, not a 404.
- The `limit` parameter defaults to 100, max 1000. Large limits on high-authority domains can be slow.

## Verification

Check that the MCP server is connected and tools are available:

```
get_domain_authority(domain: "rankparse.com")
```

Expected: returns a JSON object with `authority_score`, `referring_domains`, `backlinks`, and `registered_at`. If the tool is not found, verify the MCP server URL and API key in your config.
