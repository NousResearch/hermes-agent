# Targeted Session Branch RPC Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Let the Sessions tab branch any historical session row safely, instead of only branching the current live gateway session.

**Architecture:** Add a targeted gateway RPC that clones persisted state.db messages from a requested `session_id`, creates a child with `parent_session_id`, marks the parent `end_reason='branched'` when appropriate, initializes a live TUI session for the branch, and returns the same resume-shaped payload the TUI already knows how to switch into. Keep git worktrees out of scope; this is conversation branching only.

**Tech Stack:** Python Hermes TUI gateway RPC, Bun/OpenTUI React, `src/app/useSession.ts`, `src/tabs/Sessions.tsx`, Bun tests.

---

## Current facts

- Existing `session.branch` in `tui_gateway/server.py` branches only the active gateway session via `_sess(params, rid)` and `session["history"]`.
- Sessions tab currently refuses non-current row forks because there is no safe arbitrary-id branch RPC.
- `session.resume` already validates historical ids and returns `SessionResumeResponse` with `messages` and `info`.
- `sessions-db.ts` classifies branch children when parent `end_reason='branched'` and child starts at/after parent end.
- This plan does not create filesystem worktrees or copy repo directories.

## Task 1: Add gateway test for targeted historical branch

**Objective:** Prove a new RPC can branch a stored, non-current session by id.

**Files:**
- Modify: `/Users/codyclawford/projects/cody-hermes/tests/test_tui_gateway_server.py`
- Modify later: `/Users/codyclawford/projects/cody-hermes/tui_gateway/server.py`

**Step 1: Write failing test**

Add a test near existing `session.resume` / `session.branch` coverage:

```python
def test_session_branch_target_clones_stored_session(monkeypatch, tmp_path):
    # Seed two sessions: active gateway session A, historical stored session B.
    # Call RPC: {"method":"session.branch_target", "params":{"session_id":"B"}}
    # Assert new state row has parent_session_id == "B".
    # Assert messages from B are copied, not A.
    # Assert response has session_id, resumed/new key info, messages, and info.
```

Use existing gateway test fixtures rather than new mocks if available.

**Step 2: Run expected failure**

```bash
cd /Users/codyclawford/projects/cody-hermes
pytest tests/test_tui_gateway_server.py -k branch_target -q
```

Expected: FAIL because RPC method does not exist.

## Task 2: Implement `session.branch_target`

**Objective:** Add the narrow backend primitive without changing current `session.branch` semantics.

**Files:**
- Modify: `/Users/codyclawford/projects/cody-hermes/tui_gateway/server.py:2685-2735`

**Step 1: Extract shared branch creation helper**

Create a small helper near `session.branch`:

```python
def _branch_from_history(db, parent_key: str, history: list[dict], name: str = "") -> tuple[str, str]:
    if not history:
        raise ValueError("nothing to branch — source session has no messages")
    new_key = _new_session_key()
    current = db.get_session_title(parent_key) or "branch"
    title = name or (
        db.get_next_title_in_lineage(current)
        if hasattr(db, "get_next_title_in_lineage")
        else f"{current} (branch)"
    )
    db.create_session(new_key, source="tui", model=_resolve_model(), parent_session_id=parent_key)
    for msg in history:
        db.append_message(session_id=new_key, role=msg.get("role", "user"), content=msg.get("content"))
    db.set_session_title(new_key, title)
    if hasattr(db, "end_session"):
        db.end_session(parent_key, reason="branched")
    return new_key, title
```

Keep the exact db API names aligned with `hermes_state.py`; if `end_session` differs, use the existing method that sets `ended_at`/`end_reason`.

**Step 2: Reuse helper in existing `session.branch`**

Current live branching should still use `session["history"]`, then initialize a live agent exactly as it does now.

**Step 3: Add targeted RPC**

```python
@method("session.branch_target")
def _(rid, params: dict) -> dict:
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5008)
    if not db.get_session(target):
        return _err(rid, 4007, "session not found")
    history = db.get_messages_as_conversation(target)
    try:
        new_key, title = _branch_from_history(db, target, history, params.get("name", ""))
        new_sid = uuid.uuid4().hex[:8]
        tokens = _set_session_context(new_key)
        try:
            agent = _make_agent(new_sid, new_key, session_id=new_key)
        finally:
            _clear_session_context(tokens)
        _init_session(new_sid, new_key, agent, list(history), cols=int(params.get("cols", 80)))
    except Exception as e:
        return _err(rid, 5008, f"branch failed: {e}")
    return _ok(rid, {"session_id": new_sid, "title": title, "parent": target, "resumed": new_key, "messages": _history_to_messages(history), "info": _session_info(agent)})
```

**Step 4: Run backend tests**

```bash
cd /Users/codyclawford/projects/cody-hermes
pytest tests/test_tui_gateway_server.py -k "branch_target or session_branch" -q
```

Expected: PASS.

## Task 3: Wire TUI session client to targeted branch

**Objective:** Give the TUI a typed method that can branch by persisted row id.

**Files:**
- Modify: `src/app/useSession.ts`
- Modify: `src/context/wire.ts`
- Test: `test/app.test.tsx` or `test/sessions.test.tsx`

**Step 1: Add client method**

Extend the `session` object with something like:

```ts
branchTarget: (id: string, name?: string) => Promise<string | null>
```

Call:

```ts
gw.request<SessionResumeResponse>("session.branch_target", name ? { session_id: id, name } : { session_id: id })
```

Then reuse the same state update path as `resume()` so the branch opens immediately.

**Step 2: Add focused test**

Add a failing test that selecting a non-current Sessions row and pressing `f` calls `session.branch_target` with that row id, confirms first, then switches to the returned branch.

**Step 3: Implement minimal pass**

Keep current `branch()` untouched for `/branch` and message-level Fork.

## Task 4: Update Sessions tab fork behavior

**Objective:** Replace the non-current refusal toast with a safe targeted branch confirmation.

**Files:**
- Modify: `src/tabs/Sessions.tsx`
- Modify: `src/tabs/SessionsGroup.tsx` if prop shape changes
- Test: `test/sessions.test.tsx`

**Step 1: Change props**

Replace or supplement `onForkCurrent` with:

```ts
onFork?: (id: string) => void | Promise<void>
```

**Step 2: Preserve safety rail**

Still confirm before branching any row. Body copy should say:

```text
Creates a new conversation branch from this session. It does not create a git worktree.
```

**Step 3: Update tests**

- Current row still branches.
- Historical row branches via targeted RPC.
- Failure shows a toast and does not switch.

## Task 5: Verify and commit

**Objective:** Prove the feature works end-to-end and keep the slice reviewable.

**Commands:**

```bash
cd /Users/codyclawford/projects/herm-pane-right
bun test test/sessions.test.tsx test/app.test.tsx test/hermes-home-sessions.test.ts
bunx tsc --noEmit
bun run build
```

**Backend commit shape:**

```bash
cd /Users/codyclawford/projects/cody-hermes
git add tui_gateway/server.py tests/test_tui_gateway_server.py docs/plans/2026-05-21-session-branch-target-rpc.md
git commit -m "feat: branch historical sessions by target"
```

**TUI follow-up commit shape:**

```bash
cd /Users/codyclawford/projects/herm-pane-right
git add src/app/useSession.ts src/tabs/Sessions.tsx src/tabs/SessionsGroup.tsx test/app.test.tsx test/sessions.test.tsx
git commit -m "feat: fork historical sessions from picker"
```

## Stop conditions

- Backend work lives in `~/projects/cody-hermes`; TUI follow-up work lives in `~/projects/herm-pane-right`. Do not edit `~/.hermes/hermes-agent` unless deliberately promoting a verified runtime change.
- Stop if `hermes_state.py` has no safe public method for setting branch end reason; add backend tests before touching raw SQL.
- Stop if product scope changes to git worktrees. That is a separate workspace-isolation plan, not this RPC.
