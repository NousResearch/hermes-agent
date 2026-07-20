# QQ Project Group Batching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add QQ project-group batching so group messages are observed continuously, main-model turns are rate-limited per group, pending group messages are merged before dispatch, and QQ project groups can use shared group sessions safely.

**Architecture:** Extend the QQ NapCat adapter with a group-level scheduler that buffers observed messages, delays group dispatches until the cooldown/debounce window expires, and emits one merged `MessageEvent` into the existing gateway flow. Unify session-key isolation rules across the adapter, gateway runner, and session store so QQ can opt into shared group sessions without breaking interrupt routing or prompt caching. Update session-context prompts so shared group sessions are treated as multi-user conversations instead of single-user chats.

**Tech Stack:** Python, asyncio, gateway session store, QQ NapCat adapter tests, pytest.

---

### Task 1: Unify session isolation resolution

**Files:**
- Modify: `gateway/config.py`
- Modify: `gateway/session.py`
- Modify: `gateway/platforms/base.py`
- Test: `tests/gateway/test_config.py`
- Test: `tests/gateway/test_session.py`

**Step 1: Write/extend failing tests**

Add tests that prove:
- platform-level `extra.group_sessions_per_user` overrides global gateway config
- session store and adapter build the same key for a QQ group when per-user isolation is disabled
- shared non-thread group sessions do not present a single fixed user in session context

**Step 2: Run tests to verify failure**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_config.py tests/gateway/test_session.py -q
```

**Step 3: Implement minimal support**

Add a single session-isolation resolver used by:
- `GatewayConfig`
- `SessionStore._generate_session_key()`
- `GatewayRunner._session_key_for_source()`
- `BasePlatformAdapter.handle_message()` and related helpers

**Step 4: Run tests to verify pass**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_config.py tests/gateway/test_session.py -q
```

**Step 5: Commit**

```bash
git add gateway/config.py gateway/session.py gateway/platforms/base.py tests/gateway/test_config.py tests/gateway/test_session.py
git commit -m "feat: unify gateway session isolation rules"
```

### Task 2: Add QQ group batch scheduler

**Files:**
- Modify: `gateway/platforms/qq_napcat.py`
- Test: `tests/gateway/test_qq_napcat.py`

**Step 1: Write/extend failing tests**

Add tests that prove:
- non-trigger group messages are observed in project-group mode without immediate dispatch
- a later mention/reply/wake-word flush includes earlier observed messages
- dispatches for the same group respect the configured minimum interval
- messages arriving during cooldown are merged into the delayed batch
- batches wait for an active group session to finish instead of interrupting it

**Step 2: Run tests to verify failure**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_qq_napcat.py -q
```

**Step 3: Implement minimal support**

Add:
- adapter config parsing for project-group mode and batching thresholds
- per-group observed-message history
- per-group pending batch state and flush tasks
- merged `MessageEvent` construction with sender-prefixed lines
- command bypass for slash commands

**Step 4: Run tests to verify pass**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_qq_napcat.py -q
```

**Step 5: Commit**

```bash
git add gateway/platforms/qq_napcat.py tests/gateway/test_qq_napcat.py
git commit -m "feat: batch qq project group messages"
```

### Task 3: Guard prompt/context behavior for shared group sessions

**Files:**
- Modify: `gateway/session.py`
- Test: `tests/gateway/test_session.py`

**Step 1: Write/extend failing tests**

Add tests that shared QQ group sessions:
- render as multi-user group context
- do not pin `**User:** <name>` into the system prompt

**Step 2: Run tests to verify failure**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_session.py -q
```

**Step 3: Implement minimal support**

Teach `build_session_context()` / `build_session_context_prompt()` to distinguish:
- isolated group sessions
- shared group sessions
- shared thread sessions

**Step 4: Run tests to verify pass**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/gateway/test_session.py -q
```

**Step 5: Commit**

```bash
git add gateway/session.py tests/gateway/test_session.py
git commit -m "fix: mark shared group sessions as multi-user"
```

### Task 4: Run end-to-end targeted regression

**Files:**
- Modify as needed based on failures
- Test: `tests/gateway/test_qq_napcat.py`
- Test: `tests/gateway/test_config.py`
- Test: `tests/gateway/test_session.py`
- Test: `tests/gateway/test_approve_deny_commands.py`
- Test: `tests/run_agent/test_run_agent.py`
- Test: `tests/test_model_tools.py`

**Step 1: Run targeted regression**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tests/gateway/test_qq_napcat.py \
  tests/gateway/test_config.py \
  tests/gateway/test_session.py \
  tests/gateway/test_approve_deny_commands.py \
  tests/run_agent/test_run_agent.py \
  tests/test_model_tools.py -q
```

**Step 2: Fix any failures**

Keep changes minimal and scoped to batching/session-resolution behavior.

**Step 3: Commit**

```bash
git add gateway/platforms/qq_napcat.py gateway/config.py gateway/session.py gateway/platforms/base.py tests/gateway/test_qq_napcat.py tests/gateway/test_config.py tests/gateway/test_session.py
git commit -m "feat: enable batched qq project-group routing"
```
