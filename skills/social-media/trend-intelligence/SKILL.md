---
name: trend-intelligence
description: "Predictive social trend analysis engine. Scrapes multiple platforms (Bluesky, Reddit, Danbooru, Fediverse, news), scores topic velocity, detects trend phases (early_rise, accelerating, peaking, declining), and generates actionable content recommendations. Use when users want to: identify trending topics, predict viral content, time social posts, find rising hashtags, analyze engagement patterns, or front-run content trends."
version: 1.0.0
author: ENERGENAI LLC
author_url: https://tiamat.live
---

# Predictive Trend Intelligence Engine

## What This Does

Monitors social platforms in real-time, builds time-series velocity profiles, and predicts which topics are about to blow up — before they peak. Instead of reacting to trends, you front-run them.

**Core insight:** A topic's 24h velocity relative to its 7d baseline is a stronger signal than raw volume. Early-rise detection (velocity accelerating but volume still low) gives a 6-24 hour lead on trending topics.

## Architecture

```
SCRAPE (multi-source) → SCORE (velocity engine) → PREDICT (phase detection) → RECOMMEND (content brief)
```

### Phase Detection

| Phase | Signal | Action |
|-------|--------|--------|
| `early_rise` | 24h velocity > 2x baseline, volume < 50th percentile | **Generate now** — first-mover window |
| `accelerating` | Velocity increasing, volume growing | **Publish now** — wave is building |
| `peaking` | Volume high, velocity flattening | **Last chance** — ride the tail |
| `declining` | Velocity negative | **Skip** — you missed it |
| `dormant` | No significant activity | **Ignore** |

### Confidence Scoring

Each prediction gets a confidence score (0-1) based on:
- Number of independent sources confirming the trend
- Velocity consistency across platforms
- Historical accuracy for this topic category
- Event calendar correlation (anime season, game releases, holidays)

## Prerequisites

```bash
pip install aiohttp beautifulsoup4 sqlite3
```

Optional for expanded source coverage:
```bash
pip install atproto  # Bluesky firehose
```

## Usage

### Quick scan — what's trending right now?
```
Scan social trends and tell me what's rising
```

### Predictive mode — what's about to trend?
```
What topics are in early_rise phase right now? I want to create content before they peak.
```

### Platform-specific
```
What's trending on Bluesky in the AI art community?
```

### Content timing
```
I have a blog post about [topic]. When should I publish it based on current trend data?
```

### Historical analysis
```
Show me the trend history for "vtuber" over the past week. Is it rising or falling?
```

## Sources

The engine scrapes from configurable sources. Default set:

| Source | Type | What it catches |
|--------|------|-----------------|
| Bluesky trending | Social | Real-time hashtag velocity |
| Reddit rising | Social | Subreddit momentum |
| Danbooru hot tags | Creative | Character/fandom trends |
| HackerNews | Tech | Developer/startup trends |
| Google Trends | Search | Mainstream interest |
| Fediverse relays | Social | Decentralized platform signals |

Sources are modular — add custom scrapers via `scripts/sources/` directory.

## Data Model

Trends stored in SQLite with 30-day retention:

```sql
trend_scans(id, topic, source, volume, velocity_24h, velocity_3d, velocity_7d, phase, confidence, scanned_at)
```

The prediction engine reads the latest scan, computes velocity derivatives, cross-references the event calendar, and outputs ranked recommendations.

## Output Format

```json
{
  "timestamp": "2026-04-06T19:00:00Z",
  "predictions": [
    {
      "topic": "creature",
      "phase": "early_rise",
      "confidence": 0.82,
      "velocity24h": 2.4,
      "velocity7d": 1.1,
      "sources": ["bluesky", "danbooru", "reddit"],
      "recommendation": "Generate creature content NOW — 3 platforms showing acceleration, 6-12hr window before peak"
    }
  ]
}
```

## Configuration

Create `trend-config.yaml` in your Hermes home directory:

```yaml
trend_intelligence:
  scan_interval: 3600  # seconds between scans
  sources:
    - bluesky
    - reddit
    - danbooru
    - hackernews
  topics:
    - ai
    - gaming
    - anime
    - crypto
    - vtuber
  min_confidence: 0.6
  db_path: ~/.hermes/trend-history.db
  retention_days: 30
```

## Credits

Built by [ENERGENAI LLC](https://tiamat.live) — autonomous AI infrastructure.
Extracted from TIAMAT's production trend engine (7,000+ cycles, 46 Bluesky tags, 54 topics, 17 source types).

## License

MIT — same as hermes-agent.
