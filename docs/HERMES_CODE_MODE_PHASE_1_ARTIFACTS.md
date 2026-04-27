# Hermes Code Mode — Fase 1: Artifacts/Diffs

## 1. Resumo

Phase 1 stabilizes the artifacts/diffs-per-session foundation for Hermes Code Mode. All core work was already present on the branch; this phase validated correctness, fixed a pre-existing test that was testing stale API behavior, and confirmed the full test suite passes.

Implemented:
- Dedicated `artifacts` SQLite table (schema v8) with safe idempotent migration
- `count_diff_changes()` — correct unified-diff counting, ignores headers
- `SessionDB.create_artifact()` — persist artifacts with auto diff-counting
- `SessionDB.get_artifacts_by_session()` — queries artifacts table first, falls back to legacy message extraction for old sessions
- `write_file` artifact creation — reads old content pre-write, generates `difflib.unified_diff`, stores status (`added`/`modified`)
- `patch` artifact creation — per-file artifacts with real unified diff from patch result
- WebSocket `artifact.created` event via `_configure_artifact_ws_callback` at startup
- `GET /api/sessions/{session_id}/artifacts` endpoint — returns `{session_id, artifacts, total}`

## 2. Arquivos alterados

| File | Why |
|------|-----|
| `hermes_state.py` | Added `count_diff_changes()`, `create_artifact()`, `get_artifacts_by_session()`, `_get_artifacts_from_messages()`, schema v8 migration, artifacts table in `SCHEMA_SQL` |
| `tools/file_tools.py` | Added `set_artifact_created_callback()`, `_emit_artifact_event()`, `_persist_write_file_artifact()`, `_persist_patch_artifacts()`, artifact creation hooks in `write_file_tool()` and `patch_tool()` |
| `hermes_cli/web_server.py` | Added `_configure_artifact_ws_callback()` startup event, `GET /api/sessions/{session_id}/artifacts` endpoint |
| `tests/test_artifacts.py` | New — 19 tests covering counting, persistence, legacy fallback, schema |
| `tests/hermes_cli/test_web_server.py` | Fixed `test_post_chat_returns_turn_payload` → `test_post_chat_returns_queued_response` to match async `post_chat` (returns `{run_id, status: "queued"}`, not `assistant_message`) |

## 3. Banco/migration

Schema version bumped from 7 → 8.

```sql
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,
    diff TEXT DEFAULT '',
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    timestamp REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artifacts_session_id ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_timestamp ON artifacts(timestamp);
```

Migration is idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`). Existing sessions are unaffected — the legacy fallback extracts artifacts from `messages` table when no rows exist in `artifacts` for that session.

## 4. Endpoint validado

```
GET /api/sessions/{session_id}/artifacts
Authorization: Bearer <token>
```

**Session without artifacts:**
```json
{
  "session_id": "abc123",
  "artifacts": [],
  "total": 0
}
```

**Session with artifacts:**
```json
{
  "session_id": "abc123",
  "artifacts": [
    {
      "id": "d4e5f6a7b8c9",
      "tool_call_id": "call_xyz",
      "tool_name": "patch",
      "path": "src/example.ts",
      "status": "modified",
      "diff": "--- a/src/example.ts\n+++ b/src/example.ts\n@@ ...",
      "additions": 10,
      "deletions": 2,
      "timestamp": 1745500000.0
    }
  ],
  "total": 1
}
```

- Returns 404 if session not found
- Returns `{artifacts: [], total: 0}` (never 500) if DB error — error is logged
- Auth same as all other `/api/*` endpoints

## 5. WebSocket event

Emitted when `create_artifact()` is called after a successful `patch` or `write_file`.

```json
{
  "type": "artifact.created",
  "payload": {
    "session_id": "abc123",
    "artifact": {
      "id": "d4e5f6a7b8c9",
      "tool_call_id": "call_xyz",
      "tool_name": "patch",
      "path": "src/example.ts",
      "status": "modified",
      "diff": "--- a/src/example.ts\n+++ b/src/example.ts\n@@...",
      "additions": 10,
      "deletions": 2,
      "timestamp": 1745500000.0
    }
  },
  "timestamp": "2026-04-24T..."
}
```

Broadcast is filtered by `session_id` via `_REALTIME_HUB.broadcast(..., session_id=session_id)`. Existing chat/session/approval events are unchanged.

## 6. Diff counting

`count_diff_changes(diff: str) -> tuple[int, int]` in `hermes_state.py`:

- Skips lines starting with `+++` or `---` (file headers)
- Counts `+` lines as additions
- Counts `-` lines as deletions
- Ignores `diff --git`, `index`, `@@`, `new file mode`, context lines

```python
for line in diff.splitlines():
    if line.startswith("+++") or line.startswith("---"):
        continue
    if line.startswith("+"):
        additions += 1
    elif line.startswith("-"):
        deletions += 1
```

## 7. write_file diff

`write_file_tool` reads old content via `file_ops.read_file_raw()` before writing. After write:

```python
diff = "".join(
    difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
)
status = "modified" if file_existed else "added"
```

- New file → diff against empty string, status `added`
- Existing file → real unified diff, status `modified`
- Binary/unreadable files → artifact stored with `diff=""`, additions/deletions = 0

## 8. Testes executados

```
uv run pytest tests/test_artifacts.py -v
→ 19 passed

uv run pytest tests/test_hermes_state.py tests/hermes_cli/test_web_server.py -v
→ 252 passed

uv run pytest tests/test_artifacts.py tests/test_hermes_state.py tests/hermes_cli/test_web_server.py
→ 271 passed, 0 failed
```

## 9. Compatibilidade

No breaking changes to HermesWeb:

- `GET /api/sessions/{session_id}/artifacts` endpoint shape unchanged
- Legacy sessions (no rows in `artifacts` table) still return data via message fallback
- WebSocket event schema is additive — `artifact.created` is a new event type
- `post_chat` now returns `{run_id, status: "queued"}` (async mode) — updated the test that incorrectly expected the old sync `assistant_message` response

## 10. Próximos passos — Fase 2

Recommended: **CodeWorkspaceService**

- Open workspace/project from HermesWeb
- Detect stack (package.json, pyproject.toml, Cargo.toml, etc.)
- Read current git branch, dirty status, recent commits
- Enumerate available build/test/lint commands
- Expose via `GET /api/workspace` endpoint
- Emit `workspace.opened` WebSocket event

This gives HermesWeb the context panel it needs to show project state alongside the artifact diff feed from Phase 1.
