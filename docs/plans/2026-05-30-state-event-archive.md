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

Start with line-delimited JSON so imports can be deduplicated record-by-record. Export records are first serialized in deterministic order to a temporary file while the evidence-level `archive_id` is computed; only then is `manifest + records + receipt` written atomically to the final path. This buffering is required because the manifest is line 1 but its `archive_id` depends on all session and message records.

```json
{"type":"manifest","schema_version":1,"archive_id":"sha256:...","generation":null,"evidence_scope":"complete-repair","exported_at":"2026-05-30T00:00:00Z","producer":"hermes-agent","source":{"profile":"default","machine_id":"..."},"record_count":2,"message_payload_fields":["role","content","tool_call_id","tool_calls","tool_name","effect_disposition","timestamp","token_count","finish_reason","reasoning","reasoning_content","reasoning_details","codex_reasoning_items","codex_message_items","platform_message_id","observed","active","compacted","api_content","display_kind","display_metadata"]}
{"type":"session","schema_version":1,"session_id":"20260530_abc","source":"discord","started_at":1770000000.0,"title":"...","parent_session_id":null,"metadata_hash":"sha256:..."}
{"type":"message","schema_version":1,"session_id":"20260530_abc","local_message_id":12,"message_index":0,"content_sha256":"sha256:...","payload_sha256":"sha256:...","payload":{"role":"user","content":"hi","tool_call_id":null,"tool_calls":null,"tool_name":null,"effect_disposition":null,"timestamp":1770000001.0,"token_count":null,"finish_reason":null,"reasoning":null,"reasoning_content":null,"reasoning_details":null,"codex_reasoning_items":null,"codex_message_items":null,"platform_message_id":"...","observed":0,"active":1,"compacted":0,"api_content":null,"display_kind":null,"display_metadata":null}}
{"type":"receipt","schema_version":1,"record_count":2,"content_sha256":"sha256-of-prior-lines","finished_at":"2026-05-30T00:00:01Z"}
```

### Archive identity and repeated exports

Define `archive_id = "sha256:" + sha256(canonical_body)`. The `canonical_body` is the concatenation of every `session` and `message` record serialized with `stable_json` (`sort_keys=True`, `separators=(",", ":")`), with each record newline-terminated. Sessions are ordered by `session_id`, followed by messages ordered by `(session_id, message_index)`.

The canonical body excludes the `manifest` and `receipt` records and strips `local_message_id` from each message before canonical serialization. The emitted message record still carries `local_message_id` as informational provenance, but the canonical message record does not: `messages.id` is local provenance only, and hashing it would assign different archive identities to the same conversation on different machines. Likewise, `exported_at`, `producer`, `source.machine_id`, and receipt timestamps describe an export run rather than its evidence.

Two exports of identical evidence therefore have the same `archive_id` even when their timestamps and literal file bytes differ. JSON key order, whitespace, and manifest/receipt-only metadata do not affect evidence identity; a changed record set or hashed field does. `receipt.content_sha256` remains a separate file-level integrity check over the literal preceding lines, so it detects truncation or byte-level tampering. `verify_archive` recomputes both values and reports archive-identity failure separately from receipt-integrity failure; neither check substitutes for the other.

For an intentional new generation of otherwise identical evidence, the CLI accepts `--archive-generation LABEL`. The manifest's `generation` defaults to `null`; when non-null, canonicalization prepends a stable `{"type":"generation","value":...}` line. A schema-version bump or widened record field set also changes canonical records and therefore `archive_id`. Comparing `generation` and `schema_version` distinguishes an intentional new generation from changed evidence.

Design constraints:

- `local_message_id` preserves SQLite row provenance but is not globally authoritative.
- `message_index` is stable per exported session and drives import ordering.
- `payload_sha256` deduplicates exact message payloads across repeated exports.
- `content_sha256` supports transcript integrity checks without always reading full payload text.
- Import should be idempotent: importing the same archive twice must not duplicate messages.
- The manifest must be usable as repair input even when FTS tables or parts of `state.db` are broken.
- The archive boundary is **complete repair evidence**, not live-only replay. A repair artifact must retain rows a damaged database may have lost, including rewind history (`active=0, compacted=0`) and compaction history (`active=0, compacted=1`); otherwise it cannot satisfy the repair goal above.
- Mechanically, `payload` carries every persisted `messages` column except `id` and `session_id`, represented in the record envelope as `local_message_id` and `session_id`. The source schema is `hermes_state_common.py:192-216`; deriving the field set from that rule avoids a hand-maintained partial boundary.

---

### Task 1: Document current SessionDB boundaries

**Objective:** Create a short architecture note that distinguishes local runtime store, durable event archive, derived search indexes, semantic memory, and gateway routing.

**Files:**
- Create: `docs/session-state-architecture.md`

**Steps:**
1. Read the persisted schema in `hermes_state_common.py` (`sessions`, `messages`, `state_meta`, FTS/trigram tables).
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
scripts/run_tests.sh tests/test_hermes_state.py -q
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
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "effect_disposition",
    "timestamp",
    "token_count",
    "finish_reason",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "platform_message_id",
    "observed",
    "active",
    "compacted",
    "api_content",
    "display_kind",
    "display_metadata",
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
        "content_sha256": sha256_text(str(row.get("content") or "")),
        "payload_sha256": sha256_text(stable_json(payload)),
        "payload": payload,
    }
```

`MESSAGE_PAYLOAD_FIELDS` is not an independently chosen allowlist: it implements the mechanical boundary above, every persisted `messages` column except `id` and `session_id`. A schema test should compare it to the persisted column set so future schema additions cannot silently fall outside the repair archive.

**Tests:**
- stable JSON ordering is deterministic,
- content hash changes when content changes,
- payload hash includes tool/reasoning/platform fields,
- payload contains role, timestamp, lifecycle state, API/display metadata, and every other persisted message field except the two envelope identifiers,
- generated records contain `schema_version == 1`,
- helper handles `None` optional fields.

**Verification:**

```bash
scripts/run_tests.sh tests/hermes_cli/test_session_archive.py -q
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
def export_session_jsonl(db: SessionDB, session_id: str, out_path: Path, *, profile: str | None = None, generation: str | None = None) -> ArchiveReceipt:
    session = db.get_session(session_id)
    messages = db.get_messages(session_id, include_inactive=True)
    # Serialize deterministic records to a temp file and compute archive_id.
    # Then atomically write manifest + records + receipt to out_path.
```

The explicit `include_inactive=True` is required for the complete-repair boundary: `SessionDB.get_messages()` otherwise adds `active = 1` (`hermes_state.py:6111-6161`, especially `active_clause` at line 6134) and irreversibly omits rewind and compaction history from the export.

Then expose a CLI command such as:

```bash
hermes sessions export-archive SESSION_ID --output /tmp/session.jsonl [--archive-generation LABEL]
```

The existing session-shaped JSONL path is `hermes_state_portability.py:221-259`, its import entry point is `hermes_state_portability.py:331`, and the current `hermes sessions export OUT` CLI is implemented at `hermes_cli/console_engine.py:1401-1437`. If command naming conflicts with that CLI, prefer a conservative hidden/experimental flag first:

```bash
hermes sessions export OUT --format archive-jsonl --session SESSION_ID
```

**Tests:**
- create temp `HERMES_HOME`, insert one session with two messages,
- export to JSONL,
- assert first line is `manifest`, last line is `receipt`,
- assert receipt count/hash matches prior lines,
- assert repeated exports of identical evidence share an `archive_id` despite export-run metadata changes,
- assert `--archive-generation` changes `archive_id`,
- assert inactive rewind and compacted rows are exported with their lifecycle flags,
- assert file is UTF-8 and line-delimited JSON.

**Verification:**

```bash
scripts/run_tests.sh tests/hermes_cli/test_session_archive.py tests/test_hermes_state.py -q
```

---

### Task 4: Verify archive integrity

**Objective:** Add a verifier that checks schema version, record ordering, required fields, evidence-level archive identity, file-level receipt integrity, and duplicate message identities before import exists.

**Files:**
- Modify: `hermes_cli/session_archive.py`
- Test: `tests/hermes_cli/test_session_archive.py`

**Implementation outline:**

```python
def verify_archive(path: Path) -> ArchiveVerification:
    # stream lines
    # validate first manifest, last receipt
    # recompute archive_id from canonical evidence records
    # recompute receipt hash over literal prior lines
    # ensure session records appear before their message records
    # ensure no duplicate (session_id, message_index, payload_sha256)
```

Verification may establish well-formedness and both hash contracts, but it may claim **complete repair evidence** only for the complete-repair export path above: the exporter must have queried `get_messages(..., include_inactive=True)`, emitted the full mechanical payload field set, and marked `evidence_scope` accordingly. Hash integrity alone cannot prove that the source query did not omit inactive rows.

**Tests:**
- valid export verifies,
- tampered content fails,
- canonical archive-identity mismatch and literal receipt-integrity mismatch are distinct failures,
- missing receipt fails,
- duplicate message identity fails,
- unsupported `schema_version` fails with a clear error.

**Verification:**

```bash
scripts/run_tests.sh tests/hermes_cli/test_session_archive.py -q
```

---

### Task 5: Add idempotent import into local SQLite

**Objective:** Import a verified archive into a local `SessionDB` without duplicating existing messages.

**Files:**
- Modify: `hermes_state.py`
- Modify: `hermes_cli/session_archive.py`
- Test: `tests/hermes_cli/test_session_archive.py`
- Test: `tests/test_hermes_state.py`

**Approach:**
- Use `session_id` from the archive unless an explicit `--prefix-session-id` / `--remap-session-id` is requested later.
- Create missing sessions with existing SessionDB APIs.
- For each message, deduplicate on `(session_id, message_index, payload_sha256)` using a small archive import ledger.
- Add one transaction-capable `SessionDB` archive-import primitive in `hermes_state.py`. For each archive message, one `_execute_write` callback must perform the ledger lookup, message insertion, restoration of `active` / `compacted`, all corresponding session-counter updates, and ledger insertion on the same connection. The unique ledger key remains the concurrency backstop; a duplicate key means the message is already imported rather than authorizing a second insert.
- Reuse append's transaction-local SQL by extracting an internal helper that accepts the callback's `sqlite3.Connection`, or implement the dedicated archive insert inside that callback. Do **not** call public `append_message()` from inside the import transaction: it currently owns a separate `_execute_write` and commits when that callback returns (`hermes_state.py:5612-5614`, `_execute_write` at `hermes_state.py:2317-2368`), so it is neither a transaction-sharing nor safely nestable API. `append_message()` also has no `active` or `compacted` parameter (`hermes_state.py:5450-5473`), so preserving lifecycle state remains a real write-path addition.
- The DB ledger and the atomic import primitive are required in the first idempotent-import code PR. A sidecar receipt cannot atomically commit with SQLite message state and must not be used as the idempotence boundary.

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
- parameterized fault injection after message insertion and after lifecycle restoration (before the ledger write) raises from inside the import transaction; each failure rolls back the message row, ledger row, and session counters, and retrying the same archive then restores exactly one message and one ledger row (with a further retry remaining a no-op),
- import preserves roles/content/tool metadata,
- FTS assertions distinguish lifecycle state: restored `compacted=1` rows are included by `search_messages()` by default, while rewind rows (`active=0, compacted=0`) remain hidden (`hermes_state.py:6039-6045`). Do not promise blanket FTS visibility for every restored inactive row.

**Verification:**

```bash
scripts/run_tests.sh tests/hermes_cli/test_session_archive.py tests/test_hermes_state.py -q
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
scripts/run_tests.sh tests/hermes_cli/test_session_archive.py -q
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

- Hashing full content helps integrity but not privacy. Complete-repair archives contain rewind history and compacted-away text that users may believe was discarded, so they are strictly more sensitive than live-only exports and must be treated accordingly.
- Import must be conservative by default: verify first, dry-run supported, fail closed on conflicts.
- Existing `messages.id` row IDs are local provenance only; do not treat them as cross-machine identity.
- FTS tables should be rebuilt/updated through normal inserts, not copied from archive.
- Gateway/session routing should not depend on archive import side effects until a later PR defines logical session mapping.
