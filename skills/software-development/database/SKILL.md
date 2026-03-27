---
name: database
description: Connect to and manage PostgreSQL, MySQL, and SQLite databases. Run queries, inspect schemas, export data, and perform common admin tasks from the terminal.
version: 1.0.0
author: Tugrul Guner
license: MIT
metadata:
  hermes:
    tags: [Database, PostgreSQL, MySQL, SQLite, SQL, Data, DevOps]
    related_skills: [docker, systematic-debugging]
---

# Database Management

Terminal-first guide for managing PostgreSQL, MySQL, and SQLite databases. All commands use standard CLI tools — no GUI needed.

## Prerequisites

Install the CLI client for your database:

```bash
# PostgreSQL
which psql || sudo apt-get install -y postgresql-client  # Debian/Ubuntu
which psql || brew install libpq && brew link --force libpq  # macOS

# MySQL
which mysql || sudo apt-get install -y mysql-client
which mysql || brew install mysql-client

# SQLite (usually pre-installed)
which sqlite3 || sudo apt-get install -y sqlite3
```

---

## 1. Connecting

### PostgreSQL

```bash
# Connection string format
psql "postgresql://user:password@host:5432/dbname"

# Environment variable (avoids password in command)
export PGPASSWORD="***"
psql -h localhost -U postgres -d mydb

# Via Docker
docker exec -it my-postgres psql -U postgres -d mydb

# Quick connection test
pg_isready -h localhost -U postgres
```

### MySQL

```bash
# Direct connection
mysql -h localhost -u root -p mydb

# With password inline (scripts only — not for interactive use)
mysql -h localhost -u root -pMyPassword mydb

# Via Docker
docker exec -it my-mysql mysql -u root -p mydb
```

### SQLite

```bash
# Open or create a database file
sqlite3 mydb.sqlite

# Open in read-only mode
sqlite3 -readonly mydb.sqlite

# Execute a query directly
sqlite3 mydb.sqlite "SELECT count(*) FROM users;"
```

---

## 2. Schema Inspection

### PostgreSQL

```bash
# List databases
psql -c "\l"

# List tables in current database
psql -d mydb -c "\dt"

# Describe a table (columns, types, constraints)
psql -d mydb -c "\d+ users"

# List indexes
psql -d mydb -c "\di"

# Show table sizes
psql -d mydb -c "SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC;"

# Show foreign keys
psql -d mydb -c "SELECT conname, conrelid::regclass, confrelid::regclass
FROM pg_constraint WHERE contype = 'f';"
```

### MySQL

```bash
# List databases
mysql -e "SHOW DATABASES;"

# List tables
mysql mydb -e "SHOW TABLES;"

# Describe table
mysql mydb -e "DESCRIBE users;"

# Show create table (full DDL)
mysql mydb -e "SHOW CREATE TABLE users\G"

# Show indexes
mysql mydb -e "SHOW INDEX FROM users;"

# Table sizes
mysql mydb -e "SELECT table_name, ROUND(data_length/1024/1024, 2) AS size_mb
FROM information_schema.tables WHERE table_schema='mydb' ORDER BY data_length DESC;"
```

### SQLite

```bash
# List tables
sqlite3 mydb.sqlite ".tables"

# Show schema for a table
sqlite3 mydb.sqlite ".schema users"

# Show all schemas
sqlite3 mydb.sqlite ".schema"

# Show indexes
sqlite3 mydb.sqlite ".indexes"

# Table info (columns and types)
sqlite3 mydb.sqlite "PRAGMA table_info(users);"
```

---

## 3. Running Queries

### Outputting Results

```bash
# PostgreSQL — expanded output for wide tables
psql -d mydb -c "\x" -c "SELECT * FROM users LIMIT 5;"

# PostgreSQL — CSV output
psql -d mydb -c "COPY (SELECT * FROM users) TO STDOUT WITH CSV HEADER;"

# MySQL — tab-separated (default) or vertical
mysql mydb -e "SELECT * FROM users LIMIT 5;"
mysql mydb -e "SELECT * FROM users LIMIT 5\G"  # vertical format

# SQLite — with headers and column mode
sqlite3 -header -column mydb.sqlite "SELECT * FROM users LIMIT 5;"

# SQLite — CSV mode
sqlite3 -header -csv mydb.sqlite "SELECT * FROM users;" > users.csv
```

### Running SQL Files

```bash
# PostgreSQL
psql -d mydb -f migration.sql

# MySQL
mysql mydb < migration.sql

# SQLite
sqlite3 mydb.sqlite < migration.sql
```

---

## 4. Data Export and Import

### PostgreSQL

```bash
# Dump entire database
pg_dump -h localhost -U postgres mydb > backup.sql

# Dump specific table
pg_dump -h localhost -U postgres -t users mydb > users.sql

# Custom format (compressed, supports parallel restore)
pg_dump -Fc -h localhost -U postgres mydb > backup.dump

# Restore
psql -h localhost -U postgres mydb < backup.sql
pg_restore -h localhost -U postgres -d mydb backup.dump
```

### MySQL

```bash
# Dump entire database
mysqldump -u root -p mydb > backup.sql

# Dump specific table
mysqldump -u root -p mydb users > users.sql

# Restore
mysql -u root -p mydb < backup.sql
```

### SQLite

```bash
# Dump to SQL
sqlite3 mydb.sqlite ".dump" > backup.sql

# Dump specific table
sqlite3 mydb.sqlite ".dump users" > users.sql

# Restore
sqlite3 newdb.sqlite < backup.sql

# Copy entire database file (simplest backup)
cp mydb.sqlite mydb_backup.sqlite
```

---

## 5. Common Admin Tasks

### PostgreSQL

```bash
# Create database
psql -c "CREATE DATABASE mydb;"

# Create user with password
psql -c "CREATE USER myuser WITH PASSWORD 'secret';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;"

# Kill long-running queries
psql -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC;"
psql -c "SELECT pg_terminate_backend(PID);"

# Vacuum (reclaim space)
psql -d mydb -c "VACUUM ANALYZE;"

# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'mydb';"
```

### MySQL

```bash
# Create database
mysql -e "CREATE DATABASE mydb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# Create user
mysql -e "CREATE USER 'myuser'@'%' IDENTIFIED BY 'secret';"
mysql -e "GRANT ALL PRIVILEGES ON mydb.* TO 'myuser'@'%';"
mysql -e "FLUSH PRIVILEGES;"

# Show running queries
mysql -e "SHOW PROCESSLIST;"

# Kill a query
mysql -e "KILL QUERY 42;"
```

---

## 6. Quick Health Checks

```bash
# PostgreSQL — database size, connections, slowest queries
psql -d mydb -c "
SELECT pg_size_pretty(pg_database_size(current_database())) AS db_size;
SELECT count(*) AS active_connections FROM pg_stat_activity;
SELECT query, calls, mean_exec_time::int AS avg_ms
FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 5;
"

# MySQL — status overview
mysql -e "SHOW GLOBAL STATUS LIKE 'Threads_connected';"
mysql -e "SHOW GLOBAL STATUS LIKE 'Slow_queries';"

# SQLite — integrity check
sqlite3 mydb.sqlite "PRAGMA integrity_check;"
sqlite3 mydb.sqlite "PRAGMA quick_check;"
```

## Pitfalls

- **PostgreSQL COPY vs \copy**: `COPY` runs server-side (needs file access on server). `\copy` runs client-side (use this from your terminal).
- **MySQL root without password**: Modern MySQL uses auth_socket by default. Use `sudo mysql` or set a password.
- **SQLite locking**: SQLite uses file-level locking. Only one writer at a time. Not suitable for concurrent web apps.
- **pg_dump version mismatch**: `pg_dump` version must be >= server version. Check with `pg_dump --version`.
- **Character encoding**: Always specify UTF-8 explicitly when creating databases to avoid encoding issues.
- **Connection limits**: PostgreSQL default is 100 connections. Check with `SHOW max_connections;`.
