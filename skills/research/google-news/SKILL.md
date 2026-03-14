---
name: google-news
description: Full Google News access with homepage, all categories, subcategories, and search. Returns structured JSON with title, source, URL. Uses Scrapling CLI for browser-based fetching.
version: 5.0.0
author: Hermes Agent
tags: [news, google-news, scraping, research, current-events, categories]
---

# Google News — Complete Category and Subcategory Access

## Overview
Full Google News access. Homepage, 10 categories with subcategories, and search.
Returns structured JSON. Categories have subcategory tabs (client-side, dynamic loading).

## Files
- Script: `SKILL_DIR/scripts/google-news-search.py`
- Config: `SKILL_DIR/scripts/google-news-categories.json`

`SKILL_DIR` is the directory containing this SKILL.md file.

## Usage
```bash
# Homepage
python3 SKILL_DIR/scripts/google-news-search.py homepage US en 10

# Category
python3 SKILL_DIR/scripts/google-news-search.py technology US en 10

# Search
python3 SKILL_DIR/scripts/google-news-search.py search "AI agents" US en 10

# List all categories with subcategories
python3 SKILL_DIR/scripts/google-news-search.py categories
```

## Complete Category Structure

### Categories WITHOUT subcategories
- **Home** (homepage) — Top stories, breaking news
- **U.S.** — US national news
- **World** — International news
- **Local** — Location-based news

### Categories WITH subcategories (tabs at top of page)

**Business** (6 subcategories)
Latest | Economy | Markets | Jobs | Personal finance | Entrepreneurship

**Technology** (7 subcategories)
Latest | Mobile | Gadgets | Internet | Virtual reality | Artificial intelligence | Computing

**Entertainment** (7 subcategories)
Latest | Movies | Music | TV | Books | Arts & design | Celebrities

**Sports** (12 subcategories)
Latest | NFL | NBA | MLB | NHL | NCAA Football | NCAA Basketball | Soccer | NASCAR | Golf | Tennis | WNBA

**Science** (6 subcategories)
Latest | Environment | Space | Physics | Genetics | Wildlife

**Health** (6 subcategories)
Latest | Medication | Health care | Mental health | Nutrition | Fitness

### NOT Categories
- **Following** — Personal feed of followed topics, requires sign-in, NOT a news category
- **For You** — Personalized feed, requires sign-in, NOT a browsable category

## Subcategory Technical Notes
Subcategory tabs are CLIENT-SIDE — they filter content within the parent category page.
No separate URLs exist for subcategories. Content loads dynamically when tab is clicked.
To get subcategory-specific content, browser interaction (click tab + wait) is required.

## Aliases
home, top, top_stories → homepage
tech → technology
biz → business
ent → entertainment
sci → science

## Regional Presets
US: region=US, language=en
UK: region=GB, language=en
HK: region=HK, language=en
DE: region=DE, language=de

## JSON Output
```json
{
  "category": "technology",
  "query": null,
  "region": "US",
  "language": "en",
  "total": 10,
  "articles": [
    {"title": "...", "source": "...", "url": "..."}
  ]
}
```

## When to Use vs ddgs
ddgs: PRIMARY — faster, broader, no browser
Google News: SUPPLEMENT — category browsing, Google curation, subcategory filtering
