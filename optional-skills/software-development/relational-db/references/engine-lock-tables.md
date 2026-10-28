# Engine Lock Tables

Lock behaviour differs by engine and version. State both before reading a row; when the version is
unknown, plan for the most restrictive row.

## 1. PostgreSQL: lock taken per statement

| Statement | Lock taken | Blocks | Online alternative |
| --- | --- | --- | --- |
| `CREATE INDEX` | `SHARE` | writes for the whole build | `CREATE INDEX CONCURRENTLY` (`SHARE UPDATE EXCLUSIVE`; no transaction block; leaves an `INVALID` index on failure that must be dropped) |
| `DROP INDEX` | `ACCESS EXCLUSIVE` | reads and writes | `DROP INDEX CONCURRENTLY` |
| `REINDEX` | `SHARE` on the table + `ACCESS EXCLUSIVE` on the index | writes, and reads that would use the index | `REINDEX ... CONCURRENTLY` (12+) |
| `ADD COLUMN` (nullable, or constant default on 11+) | `ACCESS EXCLUSIVE`, brief | everything, while queued behind long transactions | set `lock_timeout` and retry |
| `ADD COLUMN ... DEFAULT volatile()` | `ACCESS EXCLUSIVE` + table rewrite | everything for the rewrite | add nullable, backfill in batches, then set the default |
| `ALTER COLUMN ... SET NOT NULL` | `ACCESS EXCLUSIVE` + full scan | everything for the scan | `ADD CONSTRAINT ... CHECK (col IS NOT NULL) NOT VALID`, `VALIDATE CONSTRAINT`, then `SET NOT NULL` (12+ skips the scan) |
| `ADD FOREIGN KEY` | `SHARE ROW EXCLUSIVE` on both tables + scan | writes on both | `ADD CONSTRAINT ... NOT VALID`, then `VALIDATE CONSTRAINT` (`SHARE UPDATE EXCLUSIVE`) |
| `ADD CHECK` constraint | `ACCESS EXCLUSIVE` + scan | everything for the scan | `NOT VALID`, then `VALIDATE CONSTRAINT` |
| `ALTER COLUMN TYPE` | `ACCESS EXCLUSIVE` + rewrite (most types; binary-coercible changes like widening `varchar(n)` skip it) | everything | new column, dual write, backfill, switch, drop |
| `RENAME COLUMN` / `RENAME TABLE` | `ACCESS EXCLUSIVE`, brief | everything, briefly | fine with `lock_timeout`, but old code breaks: expand/contract with a view or dual columns |
| `VACUUM FULL`, `CLUSTER` | `ACCESS EXCLUSIVE` + rewrite | everything | `pg_repack`, or plain `VACUUM` |

Every DDL on a live table runs under `SET lock_timeout = '2s'` (or the service's tolerance) with
a retry. Without it, a statement queued behind one long transaction blocks every query that arrives
after it.

## 2. MySQL (InnoDB) online DDL

| Operation | `ALGORITHM` | `LOCK` | Note |
| --- | --- | --- | --- |
| Add a secondary index | `INPLACE` | `NONE` | concurrent DML allowed; brief metadata lock at start and end |
| Add a column, last position (8.0.12+) | `INSTANT` | n/a | no rebuild; any position from 8.0.29 |
| Add a column on older versions | `INPLACE` with rebuild | `NONE` | rebuilds the table |
| Change a column's data type | `COPY` | `SHARED` | blocks writes; use `gh-ost` or `pt-online-schema-change` |
| Drop the primary key without adding one | `COPY` | `SHARED` | blocks writes |

State `ALGORITHM=..., LOCK=...` explicitly so the server refuses instead of silently falling back
to a blocking copy. Every DDL still waits for a metadata lock; a long transaction holds it up and
queues everything behind it, so bound it with `lock_wait_timeout`.

## 3. Index selection and sizing

| Need | Index type |
| --- | --- |
| Equality and range on scalar columns | B-tree: equality columns first (most selective first), range column last |
| Arrays, JSONB containment, full-text | GIN |
| Geometric, range overlap, nearest neighbour | GiST |
| Huge append-only table ordered by time | BRIN |
| Query touches a known subset | partial index with that `WHERE` |
| Query reads few columns | covering index (`INCLUDE`) for an index-only scan |

Size before proposing: rows × (key width + ~20 bytes per-entry overhead), plus fill-factor slack.
Every index adds a write per insert and per update of its columns; name that cost next to the read
it saves. Cite the observed plan the index changes (`EXPLAIN (ANALYZE, BUFFERS)`), and re-read the
plan after the build.
