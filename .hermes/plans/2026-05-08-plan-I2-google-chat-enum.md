# Plan I2: Google Chat Platform Enum — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add `GOOGLE_CHAT = "google_chat"` as a built-in enum member to the `Platform` enum in `gateway/config.py`, fixing all 11 Google Chat tests that fail with `AttributeError: GOOGLE_CHAT`.

**Architecture:** The `google_chat` adapter exists as a bundled plugin under `plugins/platforms/google_chat/`. The `Platform` enum supports dynamic pseudo-members via `_missing_()`, but that only works for value-based lookups (`Platform("google_chat")`) — not attribute lookups (`Platform.GOOGLE_CHAT`). All 11 failing tests use attribute access. Adding a built-in member makes Google Chat consistent with every other shipped platform.

**Tech Stack:** Python 3.13, pytest, hermes-agent `gateway/config.py`

---

## Verified State (Do Not Re-verify Root Cause)

The following is already correct and verified (parent task t_f071df92, run 898). Treat as ground truth.

### Defect: `gateway/config.py` lacks `GOOGLE_CHAT` in fork

The **default checkout** (`/home/d/.hermes/hermes-agent/gateway/config.py` lines 111-112):

```python
    YUANBAO = "yuanbao"
    GOOGLE_CHAT = "google_chat"
```

The **fork checkout** (`/home/d/Desktop/agenda/hermes-agent/gateway/config.py` line 110-111):

```python
    YUANBAO = "yuanbao"
    @classmethod
```

`GOOGLE_CHAT` is **absent** in the fork. This is the only delta causing the 11 test failures.

### Plugin Exists (Independent of Enum)

```
plugins/platforms/google_chat/
├── __init__.py
├── adapter.py
├── oauth.py
└── plugin.yaml
```

Commit `44cd79e79` added the plugin. It works regardless of whether `GOOGLE_CHAT` is a built-in enum member or a dynamic pseudo-member.

### Failing Tests (11)

| Test class | Count | Failure pattern |
|---|---|---|
| `TestPlatformRegistration` | 1 | `Platform.GOOGLE_CHAT.value == "google_chat"` |
| `TestEnvConfigLoading` | 8 | `cfg.platforms[Platform.GOOGLE_CHAT]` repeated |
| `TestAuthorizationEmailMatch` | 3 | `Platform.GOOGLE_CHAT` in allowlist logic |

---

## Task 1: Add `GOOGLE_CHAT` Built-in Enum Member

**Objective:** Insert the missing enum member into the fork's `gateway/config.py`.

**Files:**
- Modify: `gateway/config.py` (fork checkout)

**Step 1: Edit `gateway/config.py`**

Locate the `Platform` enum (line 82 in fork). After `YUANBAO = "yuanbao"` (line 110), add:

```python
    GOOGLE_CHAT = "google_chat"
```

The enum block should then read:

```python
    BLUEBUBBLES = "bluebubbles"
    QQBOT = "qqbot"
    YUANBAO = "yuanbao"
    GOOGLE_CHAT = "google_chat"
    @classmethod
    def _missing_(cls, value):
```

**Step 2: Verify the edit**

```bash
cd /home/d/Desktop/agenda/hermes-agent
grep -n "GOOGLE_CHAT" gateway/config.py
```

Expected output:
```
111:    GOOGLE_CHAT = "google_chat"
```

**Step 3: Stage (do not commit)**

```bash
cd /home/d/Desktop/agenda/hermes-agent
git add gateway/config.py
```

---

## Task 2: Run Google Chat Tests

**Objective:** Confirm the one-line fix resolves all 11 failures.

**Files:**
- Read-only: `tests/gateway/test_google_chat.py`

**Step 1: Run the full Google Chat test module**

```bash
cd /home/d/Desktop/agenda/hermes-agent
source .venv/bin/activate
pytest tests/gateway/test_google_chat.py -v --tb=short
```

**Expected output:**
```
tests/gateway/test_google_chat.py::TestPlatformRegistration::test_enum_value PASSED
tests/gateway/test_google_chat.py::TestEnvConfigLoading::test_load_empty_config PASSED
tests/gateway/test_google_chat.py::TestEnvConfigLoading::test_load_basic_config PASSED
... (8 more) ...
tests/gateway/test_google_chat.py::TestAuthorizationEmailMatch::test_email_exact_match PASSED
tests/gateway/test_google_chat.py::TestAuthorizationEmailMatch::test_email_domain_match PASSED
tests/gateway/test_google_chat.py::TestAuthorizationEmailMatch::test_email_no_match PASSED

11 passed in X.XXs
```

**Step 2: If any test fails**

Do not proceed. Run `git diff` to confirm the change is exactly the one line above. If tests still fail, report findings in a `kanban_comment` and `kanban_block` the task.

---

## Notes for Downstream Implementers

### Why only the fork is affected

The default checkout already contains `GOOGLE_CHAT = "google_chat"` (added in a prior task, t_69064a6f). The fork's `gateway/config.py` diverged and lost this line, likely due to an incomplete cherry-pick or merge resolution. The fix is strictly a forward-port of the missing line.

### Why `_missing_()` is not sufficient

Python's `Enum.__getattr__` does **not** delegate to `_missing_()` for attribute access. `_missing_()` is only invoked for value lookups like `Platform("google_chat")`. Because every failing test uses `Platform.GOOGLE_CHAT`, the pseudo-member mechanism is never triggered.

### Why no test changes are needed

The tests correctly assume all shipped platforms are built-in enum members. The defect is in the enum definition, not the tests.

---

## Files Referenced

| File | Role | Lines |
|---|---|---|
| `gateway/config.py` | Fix target | 110-111 (fork) |
| `tests/gateway/test_google_chat.py` | Verification | 2582 lines total |
| `plugins/platforms/google_chat/plugin.yaml` | Plugin metadata | — |
