---
name: database
description: Manage PostgreSQL, MySQL, and SQLite from a terminal.
version: 1.1.0
author: Tugrul Guner (@tugrulguner), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Database, PostgreSQL, MySQL, SQLite, SQL, Data, DevOps]
    related_skills: [systematic-debugging]
---

# Database Skill

Manage PostgreSQL, MySQL, and SQLite with their standard command-line clients. This skill covers inspection, queries, backup, restore, and guarded administration; it does not replace schema migration tools or authorize destructive production changes.

## When to Use

- Inspect a database schema, indexes, sizes, or active connections.
- Run a reviewed SQL statement or SQL file.
- Export, import, back up, or restore data.
- Diagnose database availability, integrity, or slow queries.
- Perform an explicitly approved administrative operation.

Do not use this skill to invent credentials, bypass access controls, or run destructive SQL without confirming the target and obtaining approval.

## Prerequisites

Use `terminal` to identify the host platform and check for the required client before installing anything:

```bash
python3 -c "import platform; print(platform.system())"
psql --version
mysql --version
sqlite3 --version
```

Install only the client needed for the current database, using the package manager available on that platform:

| Platform | PostgreSQL client | MySQL client | SQLite |
|---|---|---|---|
| Debian/Ubuntu | `sudo apt-get install postgresql-client` | `sudo apt-get install default-mysql-client` | `sudo apt-get install sqlite3` |
| macOS | `brew install libpq` | `brew install mysql-client` | bundled or `brew install sqlite` |

This skill is gated to Linux and macOS because its runnable examples use POSIX
shell syntax. On Windows, use the same database clients through their native
PowerShell conventions rather than copying these commands unchanged.

Prefer PostgreSQL service files plus `.pgpass`, MySQL interactive password
prompts, or approved secret injection. Never embed a password in a connection
URI passed to a CLI process, command, committed file, or response.

## How to Run

1. Confirm the database engine and target host/database.
2. Start with a read-only connectivity or schema command.
3. Show the exact target and SQL before any write, restore, termination, or privilege change.
4. Obtain approval for destructive or availability-affecting operations.
5. Run the command through `terminal` and inspect its real exit status and output.
6. Run a verification query that proves the intended result.

## Quick Reference

| Task | PostgreSQL | MySQL | SQLite |
|---|---|---|---|
| Connect | `psql` with `PGSERVICE` | `mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME"` | `sqlite3 app.db` |
| Test availability | `pg_isready -h "$DB_HOST"` | `mysqladmin -h "$DB_HOST" -u "$DB_USER" -p ping` | `sqlite3 app.db "PRAGMA quick_check;"` |
| List tables | `psql -c '\dt'` | `mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" -e 'SHOW TABLES;'` | `sqlite3 app.db '.tables'` |
| Describe table | `psql -c '\d+ users'` | `mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" -e 'DESCRIBE users;'` | `sqlite3 app.db 'PRAGMA table_info(users);'` |
| Run SQL file | `psql -f change.sql` | `mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" < change.sql` | `sqlite3 app.db < change.sql` |
| Read-only open | transaction `READ ONLY` | transaction `READ ONLY` | `sqlite3 -readonly app.db` |

## Procedure

### 1. Establish a Safe Connection

Keep credentials outside command history:

```bash
# PostgreSQL: ~/.pg_service.conf holds host/user/database (not the password),
# and ~/.pgpass supplies the password with mode 0600.
export PGSERVICE="project"
psql -c "\conninfo" \
  -c "SELECT inet_server_addr(), inet_server_port(), current_database(), current_user;"

# MySQL: -p prompts without exposing the password
mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" \
  -e "SELECT @@hostname, @@port, DATABASE(), CURRENT_USER();"

# SQLite: open read-only while investigating
sqlite3 -readonly app.db "SELECT sqlite_version();"
```

Stop if the reported database, user, or host is not the intended target.

### 2. Inspect Schema and Storage

PostgreSQL:

```bash
psql -c "\dt"
psql -c "\d+ users"
psql -c "\di"
psql -c "SELECT relname,
  pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;"
```

MySQL:

```bash
mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" -e "SHOW TABLES;"
mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" -e "SHOW CREATE TABLE users\G"
mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" -e "SHOW INDEX FROM users;"
```

SQLite:

```bash
sqlite3 -readonly app.db ".schema users"
sqlite3 -readonly app.db ".indexes users"
sqlite3 -readonly app.db "PRAGMA foreign_key_list(users);"
```

### 3. Run Queries Conservatively

Use a read-only transaction when supported:

```sql
-- PostgreSQL
BEGIN TRANSACTION READ ONLY;
SELECT * FROM users ORDER BY id LIMIT 20;
COMMIT;
```

```sql
-- MySQL
START TRANSACTION READ ONLY;
SELECT * FROM users ORDER BY id LIMIT 20;
COMMIT;
```

For SQLite, use `sqlite3 -readonly`. Add a deterministic `ORDER BY` and a practical `LIMIT` during exploration. Use engine-native parameter binding from application code instead of interpolating user input into SQL.

### 4. Export and Back Up

Confirm available disk space and the destination path before exporting.

```bash
# PostgreSQL
pg_dump --format=custom --file=backup.dump
pg_restore --list backup.dump

# MySQL
mysqldump -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" > backup.sql

# SQLite: online backup avoids copying a database mid-write
sqlite3 app.db ".backup 'backup.db'"
sqlite3 backup.db "PRAGMA integrity_check;"
```

Treat backup files as sensitive because they can contain credentials, personal data, and deleted records.

### 5. Restore With an Explicit Target

Never restore over an unverified production target. Create or identify a disposable target first, then verify the backup before promoting it.

```bash
# PostgreSQL custom-format restore
pg_restore --clean --if-exists --dbname="service=restore-target" backup.dump

# MySQL restore; the password is prompted
mysql -h "$RESTORE_DB_HOST" -u "$RESTORE_DB_USER" -p "$RESTORE_DB_NAME" < backup.sql

# SQLite restore the binary .backup into a new file
sqlite3 restored.db ".restore 'backup.db'"
sqlite3 restored.db "PRAGMA integrity_check;"
```

### 6. Perform Guarded Administration

List sessions before terminating anything:

```bash
psql -c "SELECT pid, usename, datname, state,
  now() - query_start AS duration, left(query, 120) AS query
FROM pg_stat_activity
WHERE pid <> pg_backend_pid()
ORDER BY query_start NULLS LAST;"
```

After a human reviews and approves one numeric PID, substitute that exact number into a guarded statement:

```bash
psql -c "SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE pid = 12345
  AND datname = current_database()
  AND pid <> pg_backend_pid();"
```

Do not terminate all matching sessions with a broad predicate. For MySQL, inspect `SHOW PROCESSLIST;`, approve one numeric process ID, then run `KILL QUERY 42;` rather than `KILL 42;` when only the statement should stop.

### 7. Check Health

```bash
# PostgreSQL
pg_isready -h "$DB_HOST"
psql -c "SELECT count(*) AS connections
FROM pg_stat_activity WHERE datname = current_database();"

# MySQL
mysqladmin -h "$DB_HOST" -u "$DB_USER" -p ping
mysql -h "$DB_HOST" -u "$DB_USER" -p "$DB_NAME" \
  -e "SHOW GLOBAL STATUS LIKE 'Threads_connected';"

# SQLite
sqlite3 -readonly app.db "PRAGMA quick_check;"
```

`pg_stat_statements` is optional. Check that the extension exists before querying it.

## Pitfalls

- **Wrong target:** Print the current database, user, and host before writes or restores.
- **Leaked passwords:** Use prompts or approved secret injection, never inline passwords.
- **Unbounded reads:** Add `ORDER BY` and `LIMIT` while investigating large tables.
- **PostgreSQL `COPY` confusion:** Server-side `COPY` differs from client-side `\copy`.
- **Version mismatch:** Use a `pg_dump` version equal to or newer than the PostgreSQL server.
- **SQLite locking:** Use `.backup` for a live database instead of copying a file mid-write.
- **Broad termination:** Select and approve one PID; keep database and self-session guards.
- **Unverified restore:** Restore into a disposable target and run integrity checks first.
- **Optional extensions:** Do not assume `pg_stat_statements` is installed.

## Verification

- [ ] The command exited successfully and its output was inspected.
- [ ] The reported host, database, and user match the intended target.
- [ ] No password or connection secret appears in output, history, or changed files.
- [ ] Read operations returned plausible, bounded results.
- [ ] A backup can be listed, opened, or integrity-checked.
- [ ] A restore was tested against a disposable target before promotion.
- [ ] Administrative changes were approved and verified with a follow-up query.
- [ ] SQLite reports `ok` for `PRAGMA integrity_check;` when integrity was tested.
