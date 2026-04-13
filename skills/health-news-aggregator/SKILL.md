---
name: health-news-aggregator
description: >
  Hunts for the most exciting global health and longevity research breakthroughs
  from leading journals and produces a curated news briefing suitable for audio TTS podcasting.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [health, longevity, news, podcast, audio]
    category: health
    related_skills: [orchestrator-skill, daily-market-briefing]
---

# Longevity & Medical News Audio Factory

This skill aggregates health and longevity news from RSS feeds and journals, curates them via an LLM pass for validity and impact, and formats the output linearly as a script. This script is intended to be picked up by the Louis TTS engine for podcast synthesis.

## Features
- **RSS Aggregation**: Fetches feeds from Cell, Nature Medicine, NIH, etc.
- **LLM Curation**: Filters out low-impact or poorly executed studies.
- **Podcast Scripting**: Transitions facts into a conversational narrative.
