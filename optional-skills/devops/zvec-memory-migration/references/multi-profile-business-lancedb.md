# Multi-Profile Business LanceDB Patterns

> Captured 2026-06-04 from a real restore session. Coexists with the
> `vec_memory_*` memory system but is **logically independent** — it's
> domain data (finance news, chip database, etc.) using LanceDB as a
> standalone vector store.

## Two-Track LanceDB Usage

In a typical multi-profile setup, LanceDB gets used for **two completely
different things** that share the same engine but have separate data paths:

| Track | Storage | Plugin that writes | Tools that read |
|---|---|---|---|
| **Memory** (session) | `~/.hermes/lance_memory/memories.lance` + per-profile `lance_memory/memories.lance` | `memory-lancedb` (independent, git-installed) | `vec_memory_add/search/list/delete/stats` |
| **Business / domain** | Anything else with `.lance` extension (e.g. `lancedb_news/financial_news.lance`, `lancedb_analysis/analysis_reports.lance`, `sqlite_holdings_journal/holdings.db`) | Domain skills / cron jobs / agents directly via `lancedb` Python lib | Domain-specific search scripts; NOT exposed via `vec_memory_*` |

## Business LanceDB Inventory (real example, 2026-06-04)

financial_expert profile's domain data:
- `lancedb_analysis/analysis_reports.lance` — analysis reports corpus (20K)
- `lancedb_news/financial_news.lance` — financial news feed (1.5M)
- `lancedb_news/financial_telegraph.lance` — financial telegraph feed (492K)
- `sqlite_holdings_journal/holdings.db` — actual holdings table (1.1M SQLite, not lancedb)

chip_expert profile:
- `skills/hardware/chip-database/scripts/chip_compare.db` — chip compare
  database (332K SQLite)

zunhunfan profile:
- `data/negative-news-history/history.db` — negative news history (44K SQLite)

**All of these are NOT touched by `vec_memory_*` tools.** They are accessed
by domain skills via direct `lancedb.connect()` or `sqlite3` calls.

## Discovery Commands

```bash
# 1. Find all .lance directories under ~/.hermes
find ~/.hermes -type d -name "*.lance"

# 2. Find all .db / .sqlite* (including inside skills/)
find ~/.hermes \( -name "*.db" -o -name "*.sqlite*" \) -type f 2>/dev/null \
  | grep -v venv | grep -v __pycache__

# 3. Per-profile business data (top-level dirs that look like data)
for p in default chip_expert financial_expert zunhunfan; do
  echo "=== $p ==="
  ls -la ~/.hermes/${p:+/profiles/$p/} 2>/dev/null \
    | grep -E "(lancedb|sqlite|holdings|docs|data|history)" \
    | grep -v PRE_REBUILD
done
```

## Why This Matters for Restores

When a user says "restore my data" or "restore my settings", they may
mean **all three of these** — and they have **different restore strategies**:

| Data type | mtime-sensitive? | Backup-override safe? | Strategy |
|---|---|---|---|
| **Skills** (in `skills/`) | No (idempotent) | Yes (cp -a) | Auto-restore missing ones |
| **state.db** (session DB) | **Yes — never overwrite** | ❌ No | Migrate-from-current, not from-backup |
| **Memory `.lance`** | Sometimes (older might be fine) | Sometimes | Ask per-profile |
| **Business `.lance` / `*.db`** | Usually yes (data accumulates) | **Never** | Compare mtime, ask user |

The session 2026-06-04 conversation confirmed: business databases
(`lancedb_news`, `lancedb_analysis`, `sqlite_holdings_journal`, `chip_compare.db`,
`history.db`) had **current mtime >= backup mtime** for everything —
**all preserved intact, no restore needed**.

## Migration Tooling

If a user wants to migrate a business `.lance` to a new schema or
rebuild it after a lancedb version bump, the pattern is the same as
for `memories.lance`:

```bash
# Backup first
mv ~/.hermes/profiles/<p>/lancedb_news/financial_news.lance \
   ~/.hermes/profiles/<p>/lancedb_news/financial_news_PRE_REBUILD_$(date +%Y%m%d)

# Rebuild using the same script as memory
~/.hermes/venv/bin/python \
  ~/.hermes/skills/lancedb-memory-migration/scripts/lancedb_rebuild_table.py \
  --path ~/.hermes/profiles/<p>/lancedb_news/financial_news.lance
```

(Note: the rebuild script's `--profile` flag targets the
`lance_memory/memories.lance` location. For arbitrary `.lance` paths,
use `--path` instead — verify your local version supports it.)

## When to Use This Reference

Load this file when:
- User mentions "domain lancedb", "business database", "financial news db",
  "chip database", "negative news db"
- User asks to restore / migrate / rebuild a non-memory `.lance` database
- The standard `lancedb-memory-migration` skill triggers but the target
  isn't a `memories.lance` under `lance_memory/`
- An agent confuses `vec_memory_*` (memory track) with domain `.lance`
  files (business track) — usually diagnosed by the user saying
  "this is my X database, not memory"
