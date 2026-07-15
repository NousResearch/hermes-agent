# pg_dump Compatibility with Hindsight

Known version compatibility issues when migrating Hindsight database dumps.

## pg_dump 18 + psql 16

pg_dump 18 emits `\restrict` directives that psql 16 cannot parse.
Result: `ERROR: \restrict: command not found` during import.

## NULL Markers in COPY Blocks

COPY data blocks use `\N` as NULL markers. Without `\restrict` protection,
psql 16 may misparse these, causing column alignment errors.

## Workaround: Python Re-Import

```python
import psycopg2

conn = psycopg2.connect("dbname=hindsight user=postgres")
cur = conn.cursor()

with open("dump.sql", "r") as f:
    for line in f:
        if line.startswith("COPY "):
            # Parse COPY block manually
            pass
        elif line.strip() and not line.startswith("\\"):
            cur.execute(line)

conn.commit()
```

## Prerequisites

- Install `postgresql-16-pgvector` before importing (Hindsight tables use `public.vector` type)
- Always test on a staging database before production import
