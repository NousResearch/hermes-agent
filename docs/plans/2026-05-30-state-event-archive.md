# State Event Archive / SessionDB Decoupling Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add an append-only, portable session event archive layer that lets Hermes export/import/repair conversation evidence without syncing `state.db` as a SQLite file.

**Architecture:** Keep `hermes_state.SessionDB` and local SQLite as the default runtime store. Add a canonical event manifest/receipt format plus additive export/import/verify commands around existing sessions and messages. Treat SQLite FTS/trigram tables as rebuildable indexes over durable event evidence, not as the cross-machine source of truth.

**Tech Stack:** Python stdlib (`json`, `hashlib`, `sqlite3`, `pathlib`), `hermes_state.SessionDB`, `hermes_cli` sessions commands, pytest temp `HERMES_HOME`, JSONL manifests, optional gzip in a later PR.

---

## Context

`state.db` currently combines multiple responsibilities:

- local runtime session metadata (`sessions` table),
- raw message transcript evidence (`messages` table),
- rebuildable full-text indexes (`messages_fts`, `messages_fts_trigram`),
- small framework metadata (`state_meta`), and
- inputs for CLI resume/history/search, gateway routing, dashboard analytics, MCP event polling, and repair flows.

That makes the SQLite file a poor synchronization primitive for multi-machine setups. WAL/locking semantics, local file mtimes, FTS virtual tables, and process-local locks are useful locally but unsafe as a shared cross-device truth layer.

This plan intentionally does **not** replace SQLite first. It introduces a durable event/evidence layer that can later support a `SessionDBProvider` or remote store while keeping the first PR additive and reviewable.

## Non-goals for the first implementation PR

- Do not add PostgreSQL/MySQL/remote service support.
- Do not change the default `state.db` path or schema semantics.
- Do not remove SQLite FTS/trigram search.
- Do not migrate gateway routing metadata out of SQLite in this PR.
- Do not make semantic memory providers store full raw transcripts.
- Do not synchronize SQLite/WAL files across machines.

---

## Proposed manifest shape

Start with line-delimited JSON so exports can stream and imports can be deduplicated record-by-record.

```json
{"type":"manifest","schema_version":1,"exported_at":"2026-05-30T00:00:00Z","producer":"hermes-agent","source":{"profile":"default","machine_id":"..."},"record_count":2}
{"type":"session","session_id":"20260530_abc","source":"discord","started_at":1770000000.0,"title":"...","parent_session_id":null,"metadata_hash":"sha256:..."}
{"type":"message","session_id":"20260530_abc","local_message_id":12,"message_index":0,"role":"user","timestamp":1770000001.0,"content_sha256":"...","payload_sha256":"...","payload":{"content":"hi","tool_call_id":null,"tool_calls":null,"tool_name":null,"finish_reason":null,"reasoning":null,"platform_message_id":"...","observed":0}}
{"type":"receipt","schema_version":1,"record_count":2,"content_sha256":"sha256-of-prior-lines","finished_at":"2026-05-30T00:00:01Z"}
```

Design constraints:

- `local_message_id` preserves SQLite row provenance but is not globally authoritative.
- `message_index` is stable per exported session and drives import ordering.
- `payload_sha256` deduplicates exact message payloads across repeated exports.
- `content_sha256` supports transcript integrity checks without always reading full payload text.
- Import should be idempotent: importing the same archive twice must not duplicate messages.
- The manifest must be usable as repair input even when FTS tables or parts of `state.db` are broken.

---

### Task 1: Document current SessionDB boundaries

**Objective:** Create a short architecture note that distinguishes local runtime store, durable event archive, derived search indexes, semantic memory, and gateway routing.

**Files:**
- Create: `docs/session-state-architecture.md`

**Steps:**
1. Read `hermes_state.py` schema (`sessions`, `messages`, `state_meta`, FTS/trigram tables).
2. Search direct `SessionDB()` construction sites in `cli.py`, `gateway/session.py`, `gateway/run.py`, `mcp_serve.py`, `cron/scheduler.py`, `tui_gateway/server.py`, `acp_adapter/session.py`, and dashboard plugins.
3. Write the note with a table:
   - state layer,
   - current storage,
   - durability,
   - sync semantics,
   - rebuildability,
   - future provider boundary.
4. Explicitly say `state.db` is still the default local store.

**Verification:**

```bash
python -m pytest tests/test_hermes_state.py -q -o 'addopts='
```

Expected: existing SessionDB tests still pass; docs-only task should not affect behavior.

---

### Task 2: Add pure event archive data model helpers

**Objective:** Add schema-versioned helpers that can turn SessionDB rows into canonical JSON-serializable archive records without writing files yet.

**Files:**
- Create: `hermes_cli/session_archive.py`
- Test: `tests/hermes_cli/test_session_archive.py`

**Implementation outline:**

```python
SCHEMA_VERSION = 1

MESSAGE_PAYLOAD_FIELDS = (
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "finish_reason",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "platform_message_id",
    "observed",
)

def stable_json(data: dict[str, object]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

def message_record(session_id: str, row: Mapping[str, object], message_index: int) -> dict[str, object]:
    payload = {field: row.get(field) for field in MESSAGE_PAYLOAD_FIELDS}
    return {
        "type": "message",
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "local_message_id": row.get("id"),
        "message_index": message_index,
        "role": row.get("role"),
        "timestamp": row.get("timestamp"),
        "content_sha256": sha256_text(str(row.get("content") or "")),
        "payload_sha256": sha256_text(stable_json(payload)),
        "payload": payload,
    }
```

**Tests:**
- stable JSON ordering is deterministic,
- content hash changes when content changes,
- payload hash includes tool/reasoning/platform fields,
- generated records contain `schema_version == 1`,
- helper handles `None` optional fields.

**Verification:**

```bash
python -m pytest tests/hermes_cli/test_session_archive.py -q -o 'addopts='
```

---

### Task 3: Export one session to JSONL with receipt

**Objective:** Export a single local session into a manifest JSONL file with manifest, session, message, and receipt records.

**Files:**
- Modify: `hermes_cli/session_archive.py`
- Modify: `hermes_cli/sessions.py` or the existing sessions subcommand module
- Test: `tests/hermes_cli/test_session_archive.py`

**Implementation outline:**

Add a pure function first:

```python
def export_session_jsonl(db: SessionDB, session_id: str, out_path: Path, *, profile: str | None = None) -> ArchiveReceipt:
    session = db.get_session(session_id)
    messages = db.get_messages(session_id)
    # write manifest + session + message records + receipt
```

Then expose a CLI command such as:

```bash
hermes sessions export-archive SESSION_ID --output /tmp/session.jsonl
```

If command naming conflicts with the existing `hermes sessions export OUT`, prefer a conservative hidden/experimental flag first:

```bash
hermes sessions export OUT --format archive-jsonl --session SESSION_ID
```

**Tests:**
- create temp `HERMES_HOME`, insert one session with two messages,
- export to JSONL,
- assert first line is `manifest`, last line is `receipt`,
- assert receipt count/hash matches prior lines,
- assert file is UTF-8 and line-delimited JSON.

**Verification:**

```bash
python -m pytest tests/hermes_cli/test_session_archive.py tests/test_hermes_state.py -q -o 'addopts='
```

---

### Task 4: Verify archive integrity

**Objective:** Add a verifier that checks schema version, record ordering, required fields, receipt hash, and duplicate message identities before import exists.

**Files:**
- Modify: `hermes_cli/session_archive.py`
- Test: `tests/hermes_cli/test_session_archive.py`

**Implementation outline:**

```python
def verify_archive(path: Path) -> ArchiveVerification:
    # stream lines
    # validate first manifest, last receipt
    # recompute receipt hash over prior lines
    # ensure session records appear before their message records
    # ensure no duplicate (session_id, message_index, payload_sha256)
```

**Tests:**
- valid export verifies,
- tampered content fails,
- missing receipt fails,
- duplicate message identity fails,
- unsupported `schema_version` fails with a clear error.

**Verification:**

```bash
python -m pytest tests/hermes_cli/test_session_archive.py -q -o 'addopts='
```

---

### Task 5: Add idempotent import into local SQLite

**Objective:** Import a verified archive into a local `SessionDB` without duplicating existing messages.

**Files:**
- Modify: `hermes_cli/session_archive.py`
- Test: `tests/hermes_cli/test_session_archive.py`

**Approach:**
- Use `session_id` from the archive unless an explicit `--prefix-session-id` / `--remap-session-id` is requested later.
- Create missing sessions with existing SessionDB APIs.
- For each message, deduplicate on `(session_id, message_index, payload_sha256)` using a small archive import ledger.
- If adding a table is too much for the first code PR, start with a sidecar import receipt file under a Hermes-controlled cache directory and leave DB ledger as the next PR.

Preferred DB table for stable idempotency:

```sql
CREATE TABLE IF NOT EXISTS session_archive_imports (
    archive_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    payload_sha256 TEXT NOT NULL,
    imported_message_id INTEGER,
    imported_at REAL NOT NULL,
    PRIMARY KEY (archive_id, session_id, message_index, payload_sha256)
);
```

**Tests:**
- import archive into empty DB creates session/messages,
- importing same archive twice does not duplicate messages,
- import preserves roles/content/tool metadata,
- FTS search sees imported messages after insert triggers run.

**Verification:**

```bash
python -m pytest tests/hermes_cli/test_session_archive.py tests/test_hermes_state.py -q -o 'addopts='
```

---

### Task 6: Add repair-oriented dry run

**Objective:** Let users compare archive contents against local `state.db` before mutating it.

**Files:**
- Modify: `hermes_cli/session_archive.py`
- Modify: sessions CLI module
- Test: `tests/hermes_cli/test_session_archive.py`

**CLI shape:**

```bash
hermes sessions import-archive /tmp/session.jsonl --dry-run
```

Dry-run output should include:

- sessions to create,
- messages to insert,
- messages already present,
- conflicting records,
- unsupported schema version / verification errors.

**Verification:**

```bash
python -m pytest tests/hermes_cli/test_session_archive.py -q -o 'addopts='
```

---

### Task 7: Add user-facing docs for safe sync patterns

**Objective:** Explain what can be synced with files/Git/vFS and what should use archive/import instead.

**Files:**
- Create or modify: `website/docs/user-guide/features/session-archive.md`
- Modify docs nav if needed.

**Content requirements:**
- `config.yaml` and skills can be synced cautiously.
- `state.db`, `state.db-wal`, and `state.db-shm` should not be multi-writer synced.
- Use export/import archive for cross-machine transcript evidence.
- Semantic memory providers are for durable meaning, not raw transcript transport.
- SQLite remains the default local runtime store.

**Verification:**

```bash
cd website && npm run build
```

---

### Task 8: Decide the next provider boundary PR

**Objective:** After archive export/import is working, decide whether the next PR should introduce `SearchProvider` or `SessionProvider` first.

**Recommendation:** Do `SearchProvider` first because FTS is already a derived index and easier to isolate than write-path `SessionDB`.

Candidate follow-up PRs:

1. `refactor(session-search): define SearchProvider protocol`
2. `refactor(session-search): wrap SQLite FTS behind SQLiteSearchProvider`
3. `feat(sessions): rebuild search indexes from session archive`
4. `refactor(session-db): introduce SessionEventStore protocol`

---

## First PR acceptance criteria

A good first implementation PR should satisfy all of these:

- Adds docs/plan or architecture docs that make the state-layer boundaries explicit.
- Adds canonical archive helpers with deterministic hashes.
- Adds export + verify for at least one session.
- Includes tests with isolated temp `HERMES_HOME`.
- Does not alter default runtime behavior when archive commands are unused.
- Does not require any external service.
- Does not sync or copy SQLite WAL files.

## Risk notes

- Hashing full content helps integrity but not privacy; archive files may contain raw transcripts and must be treated as sensitive.
- Import must be conservative by default: verify first, dry-run supported, fail closed on conflicts.
- Existing `messages.id` row IDs are local provenance only; do not treat them as cross-machine identity.
- FTS tables should be rebuilt/updated through normal inserts, not copied from archive.
- Gateway/session routing should not depend on archive import side effects until a later PR defines logical session mapping.
