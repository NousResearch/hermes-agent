# External-Content Session FTS Implementation Plan

**Status:** Implemented and validated in `perf/session-fts-external-content`; not deployed. See [`../benchmarks/2026-07-17-external-content-fts.md`](../benchmarks/2026-07-17-external-content-fts.md) for the checksum-verified snapshot benchmark.

> **For Hermes:** Use strict test-driven development and review each migration safety boundary before implementation.

**Goal:** Remove duplicated message text from the `messages_fts*_content` shadow tables while preserving standard, tool-field, CJK/trigram, ranking, and snippet behavior.

**Architecture:** Expose the existing canonical search document (`content || ' ' || tool_name || ' ' || tool_calls`) through a zero-storage SQL view named `messages_fts_source`. The non-obvious name is required: `messages_fts_content` is reserved by FTS5 as the legacy inline table's physical content-shadow name and cannot safely be reused as a migration view. New FTS5 tables reference the source view with `content='messages_fts_source'` and `content_rowid='id'`. Existing inline FTS tables remain supported and are migrated only by an explicit transactional maintenance method; startup never performs the large rebuild automatically.

**Tech Stack:** Python 3.11, SQLite/FTS5 3.37+, pytest.

---

### Task 1: Specify external-content behavior

**Objective:** Prove the desired new-database schema and search behavior before implementation.

**Files:**
- Create: `tests/test_fts_external_content_migration.py`
- Modify later: `hermes_state.py`

**Steps:**
1. Create a new `SessionDB` in `tmp_path`.
2. Assert both FTS virtual-table SQL definitions include `content='messages_fts_source'` and `content_rowid='id'`.
3. Assert `messages_fts_source` is a view and neither `messages_fts_source` nor `messages_fts_trigram_content` exists as a physical FTS shadow table.
4. Insert messages whose search terms occur in message content, `tool_name`, `tool_calls`, and CJK text.
5. Assert result rowids and snippets preserve current behavior.
6. Run the focused test and observe the expected RED failure against inline FTS.

Run: `venv/bin/python -m pytest tests/test_fts_external_content_migration.py::test_new_database_uses_external_content_fts -v`
Expected RED: FTS SQL lacks `content=` and physical `_content` shadow tables exist.

### Task 2: Make new databases external-content by default

**Objective:** Add the content view and correct FTS5 trigger semantics without migrating existing databases.

**Files:**
- Modify: `hermes_state.py` near `FTS_SQL`, `FTS_TRIGRAM_SQL`, and `_ensure_fts_schema()`.
- Test: `tests/test_fts_external_content_migration.py`

**Steps:**
1. Define one canonical SQL expression for the indexed document.
2. Add `CREATE VIEW IF NOT EXISTS messages_fts_source AS SELECT id, <expression> AS content FROM messages`.
3. Define external-content base and trigram FTS DDL using the view.
4. Change delete/update triggers to FTS5's special `INSERT ... VALUES('delete', old.id, <old expression>)` form.
5. Keep existing inline tables and triggers untouched when they already exist (`IF NOT EXISTS` compatibility).
6. Run the focused test until GREEN.

### Task 3: Preserve legacy inline compatibility and repair behavior

**Objective:** Ensure upgraded code opens and repairs legacy inline databases without silently changing storage mode.

**Files:**
- Modify: `hermes_state.py` around `_rebuild_fts_indexes()` and FTS schema inspection.
- Test: `tests/test_fts_external_content_migration.py`
- Regression: `tests/test_hermes_state.py`, `tests/test_state_db_malformed_repair.py`

**Steps:**
1. Add a schema-inspection helper returning `inline`, `external`, `missing`, or `mixed`.
2. For external tables, rebuild with `INSERT INTO <table>(<table>) VALUES('rebuild')`.
3. For legacy inline tables, retain the existing delete/backfill rebuild path.
4. Build a legacy inline fixture and assert opening it does not auto-migrate it.
5. Drop triggers, reopen, and verify trigger repair plus search backfill for both storage modes.
6. Run focused and repair regression tests.

### Task 4: Implement explicit transactional migration

**Objective:** Rebuild and atomically swap the FTS indexes while preserving the old schema on any failure.

**Files:**
- Modify: `hermes_state.py`
- Test: `tests/test_fts_external_content_migration.py`

**Steps:**
1. Add `SessionDB.migrate_fts_to_external_content()`; reject read-only connections.
2. Acquire `BEGIN IMMEDIATE` so canonical messages cannot change during rebuild.
3. Create uniquely named staging FTS tables referencing `messages_fts_source`.
4. Rebuild each staging table from the view.
5. Run `integrity-check` with `rank=1` so SQLite compares each staged index against the external content view.
6. Drop old FTS triggers/tables only after staged validation.
7. Rename staged tables to canonical names and create external-content triggers.
8. Re-run `integrity-check` with `rank=1`, then commit.
9. On any error, roll back the entire transaction; the legacy tables/triggers remain operational.
10. Return a report containing previous mode, final mode, migrated tables, and whether the call was a no-op.
11. Test successful dual-inline migration, idempotence, rollback after destructive swap/final-validation failure, subprocess kill recovery after staged validation, and write/search behavior after migration.

### Task 5: Keep migration opt-in operationally

**Objective:** Expose a maintenance path only after the core migration is proven safe.

**Files:**
- Prefer create: `hermes_cli/subcommands/sessions_fts.py` if command extraction is warranted.
- Otherwise modify narrowly: `hermes_cli/main.py`, `hermes_cli/console_engine.py`.
- Test: focused CLI parser/handler tests.

**Safety requirements:**
- Never migrate during `SessionDB.__init__`.
- Require an explicit flag/command and confirmation.
- State that a gateway stop/maintenance window is required.
- Create a verified SQLite online backup by default; allow backup suppression only with an explicit unsafe flag.
- Refuse before taking the write lock unless filesystem free space is at least `backup_bytes + current_fts_used_bytes + max(2 GiB, 25% of database_bytes)`. Re-check after the backup. This reserves coexistence space for old/staged indexes plus WAL/transaction overhead; report every term to the operator.
- Persist and verify `state_meta.fts_storage_revision=external-content-v1`; state that downgrading to a Hermes release without external-content-aware repair paths is unsupported.
- Do not run `VACUUM` implicitly as part of the core migration. Reclamation is a separate, explicit step because it needs additional free disk and an exclusive rewrite; perform a second free-space preflight sized for a complete replacement database.

### Task 6: Verify equivalence and storage savings

**Objective:** Quantify correctness and savings without touching production.

**Files:**
- Create if useful: `scripts/benchmark_fts_external_content.py`

**Steps:**
1. Run all focused state/FTS/repair tests plus Ruff and `py_compile`.
2. Generate a representative disposable database containing content, tool fields, updates, deletes, and CJK rows.
3. Capture ordered search results and snippets before migration.
4. Migrate, then assert byte-for-byte result/snippet equivalence for the query corpus.
5. Compare `dbstat` used bytes by FTS shadow object before/after.
6. Run `VACUUM` only on the disposable copy and compare file sizes.
7. If a verified production snapshot is available locally with adequate free space, repeat there; never benchmark the live production file.

### Task 7: Independent review

**Objective:** Catch corruption, compatibility, and operational hazards before any deployment proposal.

**Review checklist:**
- Transactional DDL rollback on injected failures.
- Correct old-value delete commands in update/delete triggers.
- `rank=1` external-content integrity checks.
- Compatibility with SQLite 3.37.2 and builds lacking trigram.
- Legacy inline repair path remains green.
- No startup migration or implicit production `VACUUM`.
- No secrets, snapshots, or generated databases in git.
