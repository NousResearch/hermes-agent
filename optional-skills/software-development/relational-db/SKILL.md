---
name: relational-db
description: Postgres/MySQL migrations, locks, indexes, and N+1 fixes.
version: 0.1.0
author: hope (rlaope), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [database, postgres, postgresql, mysql, innodb, migrations, ddl, indexes, locks, n-plus-one]
    category: software-development
    related_skills: [systematic-debugging]
---

# Relational DB Skill

Plan and carry out database work on a live PostgreSQL or MySQL/InnoDB system so that every
statement's lock, timeout and rollback is known before it runs: online schema migrations, index
builds, N+1 query fixes and partition/shard decisions. It does not cover analysing the data
itself or a whole-system performance goal.

## When to Use

- Adding/altering columns, constraints or foreign keys on a table that serves traffic.
- Adding or dropping an index; "this query is slow, should we index it?".
- A page or endpoint whose query count grows with the number of rows (N+1).
- An ORM migration file that needs review before it reaches production.
- Deciding whether a growing table should be partitioned or the database sharded.

Don't use for: reports or metrics over the data, a fresh schema with no live traffic (just design
it), or SQLite-only work (different locking model entirely).

## Prerequisites

- The engine **and version** (`SELECT version();`). Lock behaviour changes across versions; when
  unknown, plan for the most restrictive row in `references/engine-lock-tables.md`.
- Read access to the target environment, or the user running statements and pasting output.
- Never place connection strings, passwords or customer rows into a plan, a file or a reply.

## How to Run

Run read-only probes (plain `EXPLAIN`, catalog queries, row counts) through `terminal` with the
project's client (`psql`, `mysql`) or ORM shell. DDL, backfills and anything that writes to a
shared or production database run only after the user approves the specific plan; read the
migration files with `read_file` and `search_files` first.

## Quick Reference

- Postgres lock-safe DDL: `SET lock_timeout = '2s';` before the statement, retry on timeout.
- Postgres index on a live table: `CREATE INDEX CONCURRENTLY ...` (not inside a transaction).
- Postgres NOT NULL on a big table: `ADD CONSTRAINT c CHECK (col IS NOT NULL) NOT VALID` →
  `VALIDATE CONSTRAINT c` → `SET NOT NULL` (12+ skips the scan) → drop the check.
- Postgres FK on a big table: `ADD CONSTRAINT ... FOREIGN KEY ... NOT VALID` → `VALIDATE CONSTRAINT`.
- MySQL: write `ALGORITHM=INPLACE, LOCK=NONE` (or `INSTANT`) explicitly so the server refuses
  instead of silently falling back to a blocking copy; bound waits with `lock_wait_timeout`.
- Plan evidence: `EXPLAIN (ANALYZE, BUFFERS) <query>` (Postgres), `EXPLAIN ANALYZE <query>` (MySQL 8.0.18+).
  `ANALYZE` executes the statement: on an `INSERT`/`UPDATE`/`DELETE` it performs the write. Use plain
  `EXPLAIN` for write statements until the user approves, or wrap the analyzed run in
  `BEGIN; ... ROLLBACK;` (Postgres) after that approval.
- Who is blocking whom (Postgres): `SELECT pid, pg_blocking_pids(pid), query FROM pg_stat_activity WHERE cardinality(pg_blocking_pids(pid)) > 0;`

## Procedure

1. **State engine, version and table size.** Rows, on-disk size, write rate, and the longest
   transaction the workload normally holds. Done when each is a measured number or marked unknown.
2. **Look up every statement's lock** in `references/engine-lock-tables.md`. Done when each
   statement in the plan has: the lock it takes, what that lock blocks, for how long, its
   timeout, and its rollback. A step missing any of these means the plan is not ready; say which.
3. **Order a live migration as expand → dual-write → backfill → switch → contract.**
   - *Expand:* add the new column/table/index without breaking current code.
   - *Dual-write:* deploy code (or a trigger) writing both shapes, and confirm every old writer
     has stopped before backfilling; a row updated after its batch was copied otherwise stays stale.
   - *Backfill:* in batches keyed on the primary key, bounded rows per batch, a pause between
     batches, progress recorded so it restarts where it stopped. Done when a comparison query
     finds no row where the two shapes disagree.
   - *Switch:* deploy code reading the new shape; keep dual-writing while both exist.
   - *Contract:* drop the old shape only after the switch is observed in production.
   Each phase is its own deploy; a rollback across a contract step is a restore, so say so.
4. **Indexes only from evidence.** Cite the observed plan the index changes; otherwise mark the
   proposal unverified. Pick the type from the reference's selection table, size it (rows ×
   (key width + ~20 bytes) plus fill-factor slack), and name its write cost next to the read it
   saves. After the build, re-run `EXPLAIN` and confirm the plan actually uses it. On Postgres,
   check for a leftover `INVALID` index after a failed `CONCURRENTLY` build and drop it.
5. **N+1:** confirm from the query log or the ORM's statement counter, not by reading code. Fix
   with an eager load (`JOIN`, `IN (...)` batch, ORM preload) and pin it with a test asserting the
   statement count for a page of N rows. Done when the test fails on the old code and passes on
   the new.
6. **Partition or shard?** Project rows, bytes and write rate 12–24 months out. Partition when one
   table's maintenance (vacuum, index builds, retention deletes) outgrows its window. Shard only
   when one primary can't hold the write rate or working set after vertical scaling, replicas and
   partitioning; name the shard key, the cross-shard queries it breaks, and the rebalancing plan.
7. **Apply with approval, then verify** (see Verification). Keep planned, run and verified as
   separate states per statement when you report.

## Pitfalls

- **No `lock_timeout` is how a sub-second `ALTER` becomes an outage:** the DDL queues behind one
  long transaction and every query arriving after it queues behind the DDL.
- `CREATE INDEX CONCURRENTLY` cannot run inside a transaction block; many migration frameworks
  wrap each migration in one (Django `atomic = False`, Rails `disable_ddl_transaction!`).
- A failed `CONCURRENTLY` build leaves an `INVALID` index that still costs writes.
- `ADD COLUMN ... DEFAULT <volatile expr>` (e.g. `gen_random_uuid()`, `clock_timestamp()`) rewrites
  the whole table even on 11+; a constant default does not.
- MySQL DDL still takes a metadata lock at start and end even with `LOCK=NONE`; a long transaction
  blocks it, and everything queues behind it.
- Backfilling in one `UPDATE` bloats the table, holds row locks for the whole run, and floods
  replicas with lag.
- An index proposed from reading SQL, without a plan, is a guess.

## Verification

- Every statement in the plan lists lock, blocked operations, timeout and rollback.
- After applying: the schema matches the plan (`\d table` / `SHOW CREATE TABLE`), no `INVALID`
  indexes remain, and the target query's `EXPLAIN` shows the expected change.
- N+1 fixes have a statement-count test that is red before and green after.
- Nothing in the transcript or files contains credentials or customer rows.

Ported from rlaope/oh-my-hermes (MIT), `skills/omh-relational-db`.
