# Discord Slash Command Sync Stability Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Prevent Discord slash commands from becoming briefly unusable after gateway restart by avoiding unnecessary delete/recreate operations and by making startup sync behavior observable and conservative. During implementation we confirmed `discord.py` 2.7.1 filters `HTTPClient.edit_global_command()` payloads to `name`, `description`, and `options`, so unpatchable metadata-only drift is skipped by default and delete/recreate is explicit opt-in.

**Architecture:** Keep Discord command IDs stable whenever possible. Replace the current metadata-only delete/recreate path with narrower update handling, explicit diff classification, and a degraded-safe fallback that does not churn all commands on startup. Add tests that reproduce the observed 2026-05-14 failure mode: `Safely reconciled 44 slash command(s): unchanged=0 updated=0 recreated=43 created=1 deleted=0`, followed by Discord not delivering a slash interaction to Hermes.

**Tech Stack:** Python, discord.py app-command HTTP routes, pytest/pytest-asyncio, Hermes gateway Discord adapter.

---

## Context / Observed Failure

At 2026-05-14 10:53:20 KST, the live Kamill gateway logged:

```text
[Discord] Safely reconciled 44 slash command(s): unchanged=0 updated=0 recreated=43 created=1 deleted=0
```

At 10:54, the user attempted a Discord slash command. Gateway health was normal (`closed=False running=True ready=True`), but there was no corresponding slash-command invocation log. The likely cause is that Discord's client/server command cache was stale immediately after command IDs were deleted and recreated.

The plan below treats bulk command ID churn as the bug to eliminate.

---

### Task 1: Add a regression test for command-ID churn on metadata-only diffs

**Objective:** Capture the current unsafe behavior with a failing test that expects metadata-only diffs not to delete/recreate every command.

**Files:**
- Modify: `tests/gateway/test_discord_connect.py`
- Read: `gateway/platforms/discord.py:1250-1422`

**Step 1: Add a failing test near `test_safe_sync_slash_commands_recreates_metadata_only_diffs`**

Add a new test named:

```python
@pytest.mark.asyncio
async def test_safe_sync_slash_commands_does_not_recreate_permission_only_diffs_by_default():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    class _DesiredCommand:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self, tree):
            assert tree is not None
            return dict(self._payload)

    class _ExistingCommand:
        def __init__(self, command_id, payload):
            self.id = command_id
            self.name = payload["name"]
            self.type = SimpleNamespace(value=payload["type"])
            self._payload = payload

        def to_dict(self):
            return {
                "id": self.id,
                "application_id": 999,
                **self._payload,
                "name_localizations": {},
                "description_localizations": {},
            }

    desired = {
        "name": "help",
        "description": "Show available commands",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": "8",
    }
    existing = _ExistingCommand(
        12,
        {
            **desired,
            "default_member_permissions": None,
        },
    )

    fake_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(desired)],
        fetch_commands=AsyncMock(return_value=[existing]),
    )
    fake_http = SimpleNamespace(
        upsert_global_command=AsyncMock(),
        edit_global_command=AsyncMock(),
        delete_global_command=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        tree=fake_tree,
        http=fake_http,
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    summary = await adapter._safe_sync_slash_commands()

    assert summary == {
        "total": 1,
        "unchanged": 1,
        "updated": 0,
        "recreated": 0,
        "created": 0,
        "deleted": 0,
    }
    fake_http.delete_global_command.assert_not_awaited()
    fake_http.upsert_global_command.assert_not_awaited()
    fake_http.edit_global_command.assert_not_awaited()
```

**Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest tests/gateway/test_discord_connect.py::test_safe_sync_slash_commands_does_not_recreate_permission_only_diffs_by_default -q -o 'addopts='
```

Expected: FAIL, because current code increments `recreated` and calls delete/upsert for metadata-only diffs; the target behavior is to skip unpatchable metadata drift unless explicit recreate opt-in is set.

---

### Task 2: Replace metadata-only recreate with stable-ID edit when the HTTP route supports it

**Objective:** Prefer `edit_global_command` for patchable command diffs, and skip unpatchable metadata-only diffs by default instead of delete/upsert, preserving Discord command IDs.

**Files:**
- Modify: `gateway/platforms/discord.py:1250-1422`
- Test: `tests/gateway/test_discord_connect.py`

**Step 1: Inspect discord.py HTTP support**

Check whether `http.edit_global_command(app_id, command_id, payload)` accepts the full desired payload or only patchable fields. If it accepts extra fields safely, use the full desired payload. If not, use the patchable subset and add an explicit TODO/comment for unsupported metadata.

**Step 2: Change `_safe_sync_slash_commands` diff handling**

Replace this unsafe branch:

```python
if self._patchable_app_command_payload(current_existing_payload) == self._patchable_app_command_payload(desired):
    await mutate(http.delete_global_command, app_id, current.id)
    await mutate(http.upsert_global_command, app_id, desired)
    recreated += 1
    continue
```

with stable-ID behavior:

```python
if self._patchable_app_command_payload(current_existing_payload) == self._patchable_app_command_payload(desired):
    await mutate(http.edit_global_command, app_id, current.id, self._patchable_app_command_payload(desired))
    updated += 1
    continue
```

Implementation finding: discord.py 2.7.1 filters `edit_global_command()` payloads to `name`, `description`, and `options`. Therefore unpatchable metadata-only drift should be skipped by default and only delete/recreated when `DISCORD_COMMAND_SYNC_ALLOW_RECREATE=true`.

**Step 3: Run the focused test and verify GREEN**

Run:

```bash
python -m pytest tests/gateway/test_discord_connect.py::test_safe_sync_slash_commands_does_not_recreate_permission_only_diffs_by_default -q -o 'addopts='
```

Expected: PASS.

---

### Task 3: Update the existing metadata-only test to encode the new invariant

**Objective:** Remove the old expectation that metadata-only diffs should recreate commands.

**Files:**
- Modify: `tests/gateway/test_discord_connect.py:460-533`

**Step 1: Rename and rewrite the old test**

Rename:

```python
test_safe_sync_slash_commands_recreates_metadata_only_diffs
```

to:

```python
test_safe_sync_slash_commands_updates_metadata_only_diffs_without_recreate
```

Change assertions from:

```python
"updated": 0,
"recreated": 1,
fake_http.edit_global_command.assert_not_awaited()
fake_http.delete_global_command.assert_awaited_once_with(999, 12)
fake_http.upsert_global_command.assert_awaited_once_with(999, desired)
```

to:

```python
"unchanged": 1,
"updated": 0,
"recreated": 0,
fake_http.delete_global_command.assert_not_awaited()
fake_http.upsert_global_command.assert_not_awaited()
fake_http.edit_global_command.assert_not_awaited()
```

**Step 2: Run both sync tests**

Run:

```bash
python -m pytest tests/gateway/test_discord_connect.py::test_safe_sync_slash_commands_only_mutates_diffs tests/gateway/test_discord_connect.py::test_safe_sync_slash_commands_updates_metadata_only_diffs_without_recreate -q -o 'addopts='
```

Expected: PASS.

---

### Task 4: Add sync-summary guardrails for excessive recreate/delete operations

**Objective:** Make future mass churn visible and optionally prevent it unless explicitly allowed.

**Files:**
- Modify: `gateway/platforms/discord.py`
- Modify: `tests/gateway/test_discord_connect.py`
- Modify: `website/docs/user-guide/messaging/discord.md`
- Modify: `website/docs/reference/environment-variables.md`

**Step 1: Add environment toggle**

Add env var:

```text
DISCORD_COMMAND_SYNC_ALLOW_RECREATE=true|false
```

Default should be `false`.

**Step 2: Gate actual delete/recreate operations**

If any branch still must delete/recreate because Discord cannot edit that field, guard it:

```python
if not self._allow_discord_command_recreate():
    logger.warning(
        "[%s] Skipping slash command recreate for %s to avoid command-id churn; set DISCORD_COMMAND_SYNC_ALLOW_RECREATE=true to allow",
        self.name,
        desired_payload["name"],
    )
    unchanged += 1
    continue
```

Use a distinct counter name if needed, e.g. `skipped_recreate`, but preserve existing summary keys unless changing tests/docs together.

**Step 3: Add a focused test for the default gate**

Add a test that creates a truly non-editable diff, leaves `DISCORD_COMMAND_SYNC_ALLOW_RECREATE` unset, and asserts:

```python
fake_http.delete_global_command.assert_not_awaited()
fake_http.upsert_global_command.assert_not_awaited()
```

**Step 4: Add a focused test for explicit allow**

Set:

```python
monkeypatch.setenv("DISCORD_COMMAND_SYNC_ALLOW_RECREATE", "true")
```

Assert delete/upsert happens only under this explicit opt-in.

**Step 5: Document the env var**

Update:

```text
website/docs/reference/environment-variables.md
website/docs/user-guide/messaging/discord.md
```

Document that recreating commands can temporarily invalidate Discord client command caches and should be avoided in normal operation.

---

### Task 5: Add better startup diagnostics around command sync and stale slash interactions

**Objective:** Make future incident analysis faster without exposing secrets.

**Files:**
- Modify: `gateway/platforms/discord.py`
- Modify: `tests/gateway/test_discord_connect.py`

**Step 1: Log command names for changed commands at debug level**

Add debug logs for each changed command:

```python
logger.debug(
    "[%s] Discord slash command diff action=%s name=%s type=%s",
    self.name,
    action,
    desired_payload["name"],
    desired_payload["type"],
)
```

Do not log command args, user text, tokens, or secrets.

**Step 2: Add warning when recreate count is non-zero**

After summary, if `recreated > 0`, log:

```python
logger.warning(
    "[%s] Recreated %d Discord slash command(s); Discord clients may show stale slash commands briefly",
    self.name,
    summary["recreated"],
)
```

**Step 3: Add tests with `caplog`**

Assert warning is emitted only when `recreated > 0`.

---

### Task 6: Run focused verification

**Objective:** Verify the Discord command sync behavior without running the full suite.

**Files:**
- No code changes.

**Step 1: Run gateway Discord tests**

Run:

```bash
python -m pytest tests/gateway/test_discord_connect.py -q -o 'addopts='
```

Expected: PASS.

**Step 2: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

**Step 3: Inspect final diff**

Run:

```bash
git diff -- gateway/platforms/discord.py tests/gateway/test_discord_connect.py website/docs/user-guide/messaging/discord.md website/docs/reference/environment-variables.md
```

Expected: changes are limited to Discord command sync stability, tests, and docs.

---

### Task 7: Runtime rollout checklist

**Objective:** Roll out safely on the user's live Kamill gateway.

**Files:**
- No code changes.

**Step 1: Commit the change**

Run:

```bash
git add gateway/platforms/discord.py tests/gateway/test_discord_connect.py website/docs/user-guide/messaging/discord.md website/docs/reference/environment-variables.md
git commit -m "fix(discord): avoid slash command id churn during sync"
```

**Step 2: Restart the gateway only after tests pass**

Run:

```bash
hermes gateway restart
hermes gateway status
```

Expected: gateway is loaded/running and Discord connected.

**Step 3: Check post-restart logs**

Run:

```bash
grep -i "slash command\|reconciled\|recreated\|rate-limited" ~/.hermes/logs/gateway.log | tail -30
```

Expected: no mass `recreated=43` style churn. Ideally `unchanged` is high and `recreated=0`.

**Step 4: Manual Discord verification**

In Discord:

1. Open the `/` menu fresh.
2. Run a low-risk slash command such as `/status` or `/platforms`.
3. Confirm the gateway logs an invocation/response and Discord does not show an interaction failure.

---

## Out of Scope

- Broad Discord command registry redesign.
- Switching all commands from global to guild-scoped commands.
- Changing slash command names, permissions, or command availability policy.
- Disabling Discord slash commands entirely as the default.

---

## Acceptance Criteria

- Focused tests pass: `python -m pytest tests/gateway/test_discord_connect.py -q -o 'addopts='`.
- `git diff --check` is clean.
- Metadata-only command diffs no longer delete/recreate Discord commands by default.
- Any remaining delete/recreate path requires explicit opt-in via env/config and logs a warning.
- Unpatchable metadata-only drift is counted as unchanged/skipped under the default safe policy rather than churning command IDs.
- Gateway restart no longer produces mass `recreated=N` churn under normal conditions.
