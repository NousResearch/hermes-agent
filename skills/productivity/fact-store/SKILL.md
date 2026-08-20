---
name: fact-store
description: SQLite-backed long-term fact store for persistent knowledge. Search, add, update, delete facts with metadata tags, categories, and auto-cleanup. Replaces limited in-memory storage with an unlimited, searchable database.
version: 1.1.0
author: SamimKonjicija
license: MIT
metadata:
  hermes:
    tags: [memory, sqlite, facts, knowledge, persistence, search]
    homepage: https://github.com/SamimKonjicija/fact-store
    related_skills: []
---

# Long-Term Fact Store

A lightweight, zero-dependency SQLite knowledge base for AI agents. Store persistent facts with metadata tags, categories, and automatic usage tracking — unlimited storage that survives across sessions.

## Why?

AI agent memory systems are typically limited to a few KB of injected context per turn. This fact store provides **unlimited persistent storage** for reference data that doesn't need to be in every conversation but must be available on demand: device configs, IP addresses, known bugs, procedures, credentials references, and any long-lived knowledge.

## Quick Start

```bash
# Copy the script to your Hermes scripts directory
cp scripts/fact_store.py ~/.hermes/scripts/fact_store.py
chmod +x ~/.hermes/scripts/fact_store.py

# The database is created automatically on first use.
# Or initialize it manually:
sqlite3 ~/.hermes/facts.db < templates/schema.sql
```

## Database Location

Default: `~/.hermes/facts.db`

Override with the `FACT_STORE_DB` environment variable:
```bash
export FACT_STORE_DB=/path/to/custom.db
python3 ~/.hermes/scripts/fact_store.py stats
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment ID |
| fact | TEXT | The fact/knowledge string |
| meta_tags | TEXT (JSON array) | Searchable tags for categorization |
| category | TEXT | Grouping (e.g., "devices", "presence", "system") |
| date_created | TEXT (ISO 8601) | When the fact was added |
| last_used | TEXT (ISO 8601) | When the fact was last accessed via search |
| use_count | INTEGER | Times the fact was returned in search results |

Indexes on `meta_tags`, `category`, `last_used`, and `date_created` for fast queries.

## Commands

### Add a fact
```bash
python3 ~/.hermes/scripts/fact_store.py add "Server has passwordless sudo for deploy user" --tags server sudo system --category system
```

### Search facts
```bash
python3 ~/.hermes/scripts/fact_store.py search "sudo" [--category system] [--limit 10]
```
Search uses LIKE matching on `fact`, `meta_tags`, and `category`. Auto-updates `last_used` and increments `use_count` for accessed facts.

### List facts
```bash
python3 ~/.hermes/scripts/fact_store.py list [--category system] [--limit 50] [--offset 0] [--order last_used|date_created|use_count|id]
```

### Delete a fact
```bash
python3 ~/.hermes/scripts/fact_store.py delete 42
```

### Update a fact
```bash
python3 ~/.hermes/scripts/fact_store.py update 42 --fact "Updated text" --tags new tags --category category
```

### Cleanup stale facts
```bash
python3 ~/.hermes/scripts/fact_store.py cleanup --days 730
```
Removes facts not accessed for more than N days. Default: 365. Recommended: 730 (2 years).

### Statistics
```bash
python3 ~/.hermes/scripts/fact_store.py stats [--json]
```
Shows total facts, categories breakdown, oldest/newest, most-used facts.

### JSON output
All commands support `--json` for machine-readable output.

## Auto-Cleanup (Optional)

Set up a cron job to periodically remove stale facts:

```bash
# Via Hermes cron (recommended):
hermes cron create "every 10d" --name "fact-store-cleanup" \
  --script "python3 ~/.hermes/scripts/fact_store.py cleanup --days 730" \
  --no-agent

# Or via system crontab:
# 0 3 */10 * * python3 ~/.hermes/scripts/fact_store.py cleanup --days 730 >> ~/.hermes/logs/fact_store_cleanup.log 2>&1
```

The `--no-agent` flag means no LLM invocation — the script runs directly and its stdout is delivered as a notification. Adjust `--days` to your needs.

## Example Categories

- `devices` — hardware, cameras, TV, network devices
- `camera` — camera config, alerts, YOLO detection
- `presence` — detection rules, MAC addresses, lag notes
- `system` — host info, logging, logrotate, Gateway
- `network` — IP assignments, network config
- `bugs` — known bugs and pitfalls
- `home` — home automation, sensors, smart devices
- `project` — project-specific conventions and configs

## When to Use: Fact Store vs. Agent Memory

| | Agent Memory | Fact Store |
|---|---|---|
| **Size limit** | 8–16 KB | Unlimited |
| **Availability** | Injected every turn | Searched on demand |
| **Best for** | Rules, preferences, persona | Reference data, configs, procedures |
| **Context cost** | Always paid | Only when searched |
| **Persistence** | Across sessions | Across sessions |

**Rule of thumb:** If you need it in *every* conversation → Agent Memory. If you need it *when asked* → Fact Store.

## Integration Pattern

In your agent's configuration or system prompt, add:

> When searching for information and not finding it in Agent Memory, always check the Fact Store as a fallback.

This ensures the agent falls back to `fact_store.py search` for reference data that isn't worth burning context tokens on every turn.

## Requirements

- Python 3.8+
- SQLite3 (bundled with Python)
- No external dependencies

## Files

- `scripts/fact_store.py` — Main script (add, search, list, delete, update, cleanup, stats)
- `templates/schema.sql` — Database schema (DDL only, no data). Auto-created by the script on first run.