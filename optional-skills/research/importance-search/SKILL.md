---
name: importance-search
description: Rank news, X, and YouTube results by importance signals.
version: 1.1.0
author: Su Ham (su-record), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [search, news, youtube, x-twitter, ranking, briefing]
    category: research
    related_skills: [duckduckgo-search]
---

# Importance Search Skill

Ranks multi-source search results by importance signals instead of raw search order: news by cross-outlet coverage + recency, X posts by configured influential accounts, YouTube by view velocity (views per day since upload). It produces a short ranked briefing per configured domain — it does not crawl the whole web and is not a replacement for `web_search` on one-off lookups.

## When to Use

- The user wants "what actually matters right now" in a field (AI, finance, general news, or a custom domain), not a flat result list.
- Recurring daily/weekly briefings driven by a config file.
- Not for single factual lookups — use `web_search` directly for those.

## Prerequisites

- Codex OAuth provider authenticated (ChatGPT subscription). Discovery and synthesis go through `agent.auxiliary_client.call_llm` with the `web_search` tool — no paid search API.
- `yt-dlp` installed in the Python environment running the script (`pip install yt-dlp`).
- Optional: `IMPORTANCE_FETCH_DIR` — path to an insane-search-style deep-fetch engine for full article bodies. Without it, a plain HTTP fallback is used (URL-safety-checked, every redirect target re-validated).

## How to Run

Run with `terminal` from the hermes-agent root, so the script can import `agent.*` and `tools.*` from the package:

```bash
python optional-skills/research/importance-search/scripts/importance_search.py ai-tech
```

The argument is a domain key from `search_domains.json` (ships with `ai-tech`, `finance`, `general`). An unknown key logs a warning and falls back to the first configured domain.

## Quick Reference

| Source | Importance signal | Backend |
|---|---|---|
| News | cross-outlet coverage + recency | Codex OAuth `web_search` |
| X | recent posts from configured influential accounts | Codex OAuth `web_search` |
| YouTube | view velocity = views ÷ days since upload | `yt-dlp` flat search |

## Procedure

1. Codex OAuth `web_search` discovers today's top news and influential-account posts for the domain.
2. `yt-dlp` flat search (with `--extractor-args youtubetab:approximate_date`) returns `view_count` plus an approximate upload timestamp — without the bot-gate that blocks full extraction on datacenter IPs. Velocity = views per day; undated items are treated as ~30 days old.
3. Each source is scored by its signal; the top 3 per source are kept.
4. The top news URL is deep-fetched to enrich a final 2-line synthesis — only after `is_safe_url` and website-policy checks pass, with each HTTP redirect target re-validated against the same checks.

## Configuration

Edit `search_domains.json` at the skill root. Each domain entry has `label`, `keywords` (news), `x_influencers`, and `youtube_queries`. Add a new domain by adding one JSON object — the same engine then serves that field.

## Pitfalls

- Codex OAuth not authenticated → news/X sections come back empty. Failures are logged as warnings; the script still prints whatever sources succeeded.
- YouTube full extraction is bot-gated on datacenter IPs — the skill deliberately uses flat search only. Don't "upgrade" it to full extraction.
- `approximate_date` timestamps are best-effort; treat velocity as a ranking signal, not an exact metric.

## Verification

```bash
python optional-skills/research/importance-search/scripts/importance_search.py ai-tech
# expect: "# Importance briefing — AI/Tech" with News / X / YouTube sections and a 💡 insight
scripts/run_tests.sh tests/skills/test_importance_search_skill.py -q
```
