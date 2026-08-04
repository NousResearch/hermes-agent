# v2 → v3 Data Migration Tool

## When to Use

When upgrading from MemoryCore v1.x or v0.x to v2.0.0+ (data format v3), the database schema and file layout have changed. Run this migration script to upgrade existing data to v3 format before starting the new Gateway.

**Affected data:**

- `vectors.db` — new tenant isolation columns: `team_id`, `task_id`, `user_id`, `agent_id`, `version`
- `scene_blocks/`, `persona.md`, `.metadata/` — L2/L3 files migrated to scoped paths under `profiles/`

**⚠️ Back up your entire data directory before migration** to prevent accidental data loss.

## Prerequisites

- Python 3.8+
- Data directory path (default: `~/.memory-tencentdb/memory-tdai/`)

## Usage

```bash
# 1. Dry-run first to inspect without making changes
python v2-to-v3-migrate.py /path/to/memory-tdai --dry-run

# 2. Run the migration
python v2-to-v3-migrate.py /path/to/memory-tdai

# 3. Database only (skip L2/L3 file migration)
python v2-to-v3-migrate.py /path/to/memory-tdai --db-only
```

### Options

| Option | Description |
|---|---|
| `/path/to/memory-tdai` | Data directory path (required) |
| `--dry-run` | Inspect only, no changes |
| `--db-only` | Migrate `vectors.db` schema only, skip L2/L3 files |
| `--no-backup` | Skip automatic backup (a `.bak` file is created by default) |

## What Gets Migrated

### 1. Database Schema Upgrade

| Table | Changes |
|---|---|
| `l1_records` | Added `team_id`, `task_id`, `user_id`, `agent_id`, `version` |
| `l0_conversations` | Added `team_id`, `task_id`, `user_id`, `agent_id` |
| `l1_fts` / `l0_fts` | Rebuilt FTS5 indexes with tenant isolation columns |
| `memory_audit` | New audit table |
| `skills` | New skills table |
| `skill_fts` | New skills full-text index table |

### 2. L2/L3 File Migration

| Source | Destination |
|---|---|
| `{data_dir}/scene_blocks/` | `{data_dir}/profiles/team%3Adefault%7Cagent%3Adefault/scene_blocks/` |
| `{data_dir}/persona.md` | `{data_dir}/profiles/team%3Adefault%7Cagent%3Adefault/persona.md` |
| `{data_dir}/.metadata/` | `{data_dir}/profiles/team%3Adefault%7Cagent%3Adefault/.metadata/` |

## Examples

```bash
# Hermes
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai --dry-run
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai

# OpenClaw
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai --dry-run
python scripts/migrate-v2-to-v3/v2-to-v3-migrate.py ~/.memory-tencentdb/memory-tdai
```

## FAQ

**Q: What if migration fails?**

The script automatically backs up `vectors.db` before migration (creates a `.bak.{timestamp}` file). L2/L3 files are copied rather than moved — source files are never deleted. Simply restore from backup if anything goes wrong.

**Q: Can I run it multiple times?**

Yes. The script is idempotent — existing columns and files are skipped.

**Q: Do I need to run this on a fresh install?**

No. The migration script is only for upgrading existing v1.x data to v2.0.0+. Fresh installations of the new Gateway automatically create v3-format data.
