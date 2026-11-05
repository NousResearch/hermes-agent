---
name: db-schema-introspect
description: "Database schema introspection — auto-discover tables, columns, foreign keys, indexes, generate ER diagrams (Mermaid), export to JSON/Markdown."
version: "0.1.0"
author: "kenya"
license: "MIT"
platforms: [linux, macos, windows]
category: "data-engineering"
tags: [database, schema, introspection, postgresql, mysql, sqlite, mermaid, er-diagram]
depends_on: [sql-query-executor]
compatibility:
  hermes: ">=0.18.0"
  claude-code: ">=1.0.0"
  codex: ">=1.0.0"
  opencode: ">=0.5.0"
maturity: "beta"
homepage: "https://github.com/NousResearch/hermes-agent"
repository: "https://github.com/NousResearch/hermes-agent"
---

# DB Schema Introspect

> Database schema introspection — auto-discover tables, columns, foreign keys, indexes, generate ER diagrams (Mermaid), export to JSON/Markdown.

## Prerequisites
- sql-query-executor skill installed
- Target database accessible
- DATABASE_DSN environment variable set

## Installation
```bash
hermes skill install db-schema-introspect
# Dependencies auto-installed via sql-query-executor
```

## Configuration
| Environment Variable | Required | Description | Example |
|----------------------|----------|-------------|---------|
| `DATABASE_DSN` | Yes | Database DSN | `postgresql://user:pass@host:5432/db` |

## Usage
### db_schema
Introspect database schema and output in various formats.

```bash
# Full schema as JSON
hermes skill run db-schema-introspect db_schema --format json

# Tables only as Markdown table
hermes skill run db-schema-introspect db_schema --format markdown --tables-only

# Generate Mermaid ER diagram
hermes skill run db-schema-introspect db_schema --format mermaid --output schema.er.md

# Specific table details
hermes skill run db-schema-introspect db_schema --table users --format json
```

## API / Tools
| Tool | Description | Parameters |
|------|-------------|------------|
| `db_schema` | Introspect schema | `format: enum[json,markdown,mermaid], tables_only: bool, table: str, output: str` |

## Examples
```bash
# Generate documentation-ready Markdown
hermes skill run db-schema-introspect db_schema --format markdown > schema.md

# ER diagram for docs
hermes skill run db-schema-introspect db_schema --format mermaid > schema.er.md

# CI/CD schema validation - export JSON for diffing
hermes skill run db-schema-introspect db_schema --format json > schema.json
```

## Troubleshooting
| Symptom | Cause | Solution |
|---------|-------|----------|
| `No tables found` | Wrong database/schema | Check DATABASE_DSN points to correct DB |
| `Permission denied` | Insufficient grants | Grant SELECT on information_schema / pg_catalog |

## Changelog
### v0.1.0 (2026-08-15)
- Initial release with PostgreSQL, MySQL, SQLite support
- JSON, Markdown, Mermaid ER diagram output
- Foreign key and index detection