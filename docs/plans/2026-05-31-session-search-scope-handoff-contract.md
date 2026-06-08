# Session Search Scope + Previous/Handoff Contract Implementation Plan

> **For Hermes:** This plan is the implementation contract for fixing gateway session-search cross-contamination after `/new`.

**Goal:** Prevent `/new` handoff/previous-session requests from retrieving unrelated sessions, especially across QQ users or adjacent projects.

**Architecture:** Store stable session scope metadata in `sessions`, propagate current gateway scope into `session_search`, and add an explicit `previous`/`handoff` retrieval path that does not rely on keyword search. Default gateway recall is scoped; explicit global search remains available for debug/admin use.

**Tech Stack:** Python, SQLite SessionDB, Hermes gateway `SessionSource`, tool executor, `session_search` tool.

---

## Merged Review Contract

This merges Codex design review and the user's follow-up review.

### 1. Dedicated previous/handoff mode

`session_search(mode="previous" | "handoff", scope="current")`

Behavior:

- Only uses the current platform/chat scope.
- Excludes the current session lineage.
- Picks the most recent ended session first (`ended_at DESC`).
- Falls back to last message activity only after `ended_at` ordering.
- Never performs keyword/global discovery.
- If scoped lookup finds nothing, returns an empty result; it must not silently fall back to global search.

This is the core fix for “刚才那个会话 / 交接信息”.

### 2. Stable scope fields

`session_key` may be persisted and used as auxiliary evidence, but default isolation is based on stable business fields:

- `source` — canonical platform/source (`qqbot`, `telegram`, `cli`, `webui`, `cron`, etc.)
- `chat_type`
- `chat_id`
- `thread_id`
- `user_id`
- `session_key`

QQ DM default scope is `source + chat_type + chat_id`; `user_id` is stored but not the primary QQ DM isolation key.

### 3. Scope propagation chain

The current scope must be available from gateway to tool execution:

- gateway adapter creates `SessionSource`
- gateway `SessionStore` persists scope on new/reset session creation
- `run_agent.AIAgent` receives `platform/user_id/chat_id/chat_type/thread_id/gateway_session_key`
- `agent/tool_executor.py` passes those as hidden current-scope kwargs to `session_search`
- `tools/session_search_tool.py` applies scope defaults and filters

### 4. Legacy compatibility with strict fallback

New sessions write full scope fields. Old sessions may have null scope fields.

Rules:

- `previous`/`handoff`: primary path is scoped new fields. Legacy fallback is allowed only within the same `source`, excluding current lineage, bounded to recent/ended ordering, and marked in the response. No cross-source/global fallback.
- Ordinary search/browse: gateway sessions default to current scope. CLI remains broad/legacy-friendly unless explicit scope is provided.
- Global search must be explicit: `scope="global"`.

### 5. Reliable ended_at

`/new`, auto reset, session switch, and compression split should mark old sessions ended. This change depends on existing `SessionStore.reset_session()` and `SessionDB.end_session()` behavior; tests must cover `/new`-style ended-session selection.

### 6. Behavior-level regression tests

Required tests:

- QQ user A/B both mention “新增功能”; A scoped search does not see B.
- A `/new` then `mode="handoff"` returns A's just-ended admissions session.
- Adjacent admissions/tutoring sessions: `mode="handoff"` returns admissions and not tutoring/OCR/PDF/学生档案.
- Current lineage is excluded.
- Scoped no-result does not global fallback unless `scope="global"`.
- Legacy null-scope sessions are not lost, but fallback is source-bounded and flagged.
- CLI search is not accidentally constrained by QQ scope rules.

## Implementation Tasks

### Task 1: Add scope columns and write paths

Modify `hermes_state.py`:

- Add nullable columns to `sessions`: `chat_type`, `chat_id`, `thread_id`, `session_key`, `user_id_alt`.
- Keep indexes referencing new columns after `_reconcile_columns()`.
- Extend `_insert_session_row()` / `create_session()` to accept those kwargs.

Modify `gateway/session.py` and `run_agent.py`:

- Pass scope metadata when creating DB sessions.

### Task 2: Add scope helper + scoped filtering

Modify `tools/session_search_tool.py`:

- Add helper to resolve current scope from hidden kwargs.
- Add helper to decide default `scope`:
  - gateway source (`qqbot`, `telegram`, `discord`, `slack`, etc.) + chat scope => current
  - CLI/local with no chat scope => legacy/global-ish
  - explicit `scope="global"` bypasses scope filters
- Add shared session scope matcher.

### Task 3: Add previous/handoff mode

Modify `tools/session_search_tool.py`:

- Add `mode` schema enum: `previous`, `handoff`.
- Implement previous/handoff selection by current scope, excluding current lineage.
- Return recent session metadata + bookend start/end + messages; no FTS keyword search.
- Return empty when scoped none.

### Task 4: Preserve search behavior with scoped default

Modify discovery/browse paths:

- Apply scope filters when default/current scoped.
- Keep explicit global mode.
- Preserve CLI broad search by default.

### Task 5: Verify

Run focused tests:

```bash
python -m pytest tests/tools/test_session_search.py -o addopts='' -q
python -m pytest tests/gateway/test_session*.py tests/test_hermes_state*.py -o addopts='' -q
```

Then inspect:

```bash
git status --short --branch --untracked-files=all
git diff --stat
git diff --check
```
