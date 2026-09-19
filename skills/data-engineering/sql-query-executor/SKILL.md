---
name: sql-query-executor
description: "Universal SQL executor across PostgreSQL, MySQL, SQLite, ClickHouse — run queries, export CSV/JSON/Markdown, parameterized queries with safety limits."
version: "0.1.0"
author: "kenya"
license: "MIT"
platforms: [linux, macos, windows]
category: "data-engineering"
tags: [sql, database, postgresql, mysql, sqlite, clickhouse, csv, json]
depends_on: []
compatibility:
  hermes: ">=0.18.0"
  claude-code: ">=1.0.0"
  codex: ">=1.0.0"
  opencode: ">=0.5.0"
maturity: "beta"
homepage: "https://github.com/NousResearch/hermes-agent"
repository: "https://github.com/NousResearch/hermes-agent"
---

# SQL Query Executor

> Universal SQL executor across PostgreSQL, MySQL, SQLite, ClickHouse — run queries, export CSV/JSON/Markdown, parameterized queries with safety limits.

## Prerequisites
- Python 3.10+ with packages: psycopg2-binary, pymysql, clickhouse-connect
- Target database accessible via network/Unix socket
- DSN connection string for the database

## Installation
```bash
# Standard installation
hermes skill install sql-query-executor
# Or manual
pip install psycopg2-binary pymysql clickhouse-connect tabulate
```

## Configuration
| Environment Variable | Required | Description | Example |
|----------------------|----------|-------------|---------|
| `DATABASE_DSN` | Yes | Database DSN (postgresql://, mysql://, sqlite://, clickhouse://) | `postgresql://user:pass@host:5432/db` |
| `SQL_DEFAULT_LIMIT` | No | Default row limit for safety | `1000` |

## Usage
### sql_exec
Execute SQL query and output results in table/CSV/JSON/Markdown format.

```bash
# Execute inline query
hermes skill run sql-query-executor sql_exec --dsn "$DATABASE_DSN" --query "SELECT * FROM users LIMIT 10" --format table

# Execute from file
hermes skill run sql-query-executor sql_exec --dsn "$DATABASE_DSN" --file query.sql --format csv

# Parameterized query
hermes skill run sql-query-executor sql_exec --dsn "$DATABASE_DSN" --query "SELECT * FROM users WHERE id = %(id)s" --params '{"id": 42}' --format json
```

## API / Tools
| Tool | Description | Parameters |
|------|-------------|------------|
| `sql_exec` | Execute SQL query | `dsn: str, query: str, file: str, format: enum[table,csv,json,markdown], limit: int, params: dict` |

## Examples
```bash
# Export users table to CSV
hermes skill run sql-query-executor sql_exec --dsn "$DATABASE_DSN" --query "SELECT * FROM users" --format csv > users.csv

# Run parameterized query with JSON output
hermes skill run sql-query-executor sql_exec --dsn "$DATABASE_DSN" --query "SELECT * FROM orders WHERE user_id = %(uid)s AND created_at > %(since)s" --params '{"uid": 123, "since": "2024-01-01"}' --format json

# Markdown table for documentation
hermes skill run sql-query-executor sql_exec --dsn "$DATABASE_DSN" --query "SELECT table_name, table_rows FROM information_schema.tables WHERE table_schema = 'public'" --format markdown
```

## Troubleshooting
| Symptom | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: psycopg2` | Missing driver | `pip install psycopg2-binary` |
| `Connection refused` | DB not reachable | Check host/port/firewall, test with `psql`/`mysql` CLI |
| `SSL SYSCALL error` | SSL required | Add `?sslmode=require` to DSN |
| `Query timeout` | Long running query | Add `LIMIT` or increase timeout in DSN params |

## Changelog
### v0.1.0 (2026-08-15)
- Initial release with PostgreSQL, MySQL, SQLite, ClickHouse support
- Table, CSV, JSON, Markdown output formats
- Parameterized queries with JSON params
- Safety row limit