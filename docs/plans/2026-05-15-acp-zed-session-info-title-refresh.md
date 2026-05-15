# ACP Zed Session Info Title Refresh Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make Zed ACP threads update their visible title/last-activity metadata immediately after Hermes auto-generates a session title.

**Architecture:** Hermes already persists session titles via `maybe_auto_title(...)` after each ACP prompt. Add a narrow ACP notification helper that reads the updated session row and emits `SessionInfoUpdate(session_update="session_info_update")` to Zed. Keep persistence unchanged; this PR only adds the live UI update.

**Tech Stack:** Python, ACP Python SDK, `acp_adapter/server.py`, pytest via `scripts/run_tests.sh`.

---

### Task 1: Add a regression test for title update emission

**Objective:** Prove that after a successful prompt and title generation Hermes sends `SessionInfoUpdate` to the ACP client.

**Files:**
- Modify: `tests/acp/test_server.py`

**Step 1: Write failing test**

Add a test near prompt/session-info tests:

```python
from acp.schema import SessionInfoUpdate

@pytest.mark.asyncio
async def test_prompt_sends_session_info_update_after_auto_title(self, agent, monkeypatch):
    mock_conn = MagicMock(spec=acp.Client)
    mock_conn.session_update = AsyncMock()
    agent._conn = mock_conn

    resp = await agent.new_session(cwd="/tmp")
    state = agent.session_manager.get_session(resp.session_id)
    state.agent.run_conversation.return_value = {
        "final_response": "Done.",
        "messages": [{"role": "user", "content": "fix zed titles"}],
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }

    def fake_auto_title(db, session_id, user_text, final_response, history):
        db.update_session_title(session_id, "Fix Zed titles")

    monkeypatch.setattr("agent.title_generator.maybe_auto_title", fake_auto_title)

    mock_conn.session_update.reset_mock()
    await agent.prompt(
        session_id=resp.session_id,
        prompt=[TextContentBlock(type="text", text="fix zed titles")],
    )

    updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
    info_updates = [u for u in updates if isinstance(u, SessionInfoUpdate)]
    assert len(info_updates) == 1
    assert info_updates[0].session_update == "session_info_update"
    assert info_updates[0].title == "Fix Zed titles"
```

**Step 2: Run test to verify failure**

Run:

```bash
scripts/run_tests.sh tests/acp/test_server.py::TestPrompt::test_prompt_sends_session_info_update_after_auto_title -q
```

Expected: FAIL because no `SessionInfoUpdate` is emitted.

### Task 2: Implement a session-info update helper

**Objective:** Centralize title/updated_at reading and notification.

**Files:**
- Modify: `acp_adapter/server.py`

**Step 1: Import schema**

Add `SessionInfoUpdate` to the `acp.schema` import list.

**Step 2: Add helper**

Inside `HermesACPAgent`:

```python
async def _send_session_info_update(self, session_id: str) -> None:
    if not self._conn:
        return
    try:
        row = self.session_manager._get_db().get_session(session_id)
    except Exception:
        logger.debug("Could not read ACP session info for %s", session_id, exc_info=True)
        return
    if not row:
        return
    title = row.get("title")
    updated_at = row.get("updated_at")
    update = SessionInfoUpdate(
        session_update="session_info_update",
        title=title if isinstance(title, str) and title.strip() else None,
        updated_at=str(updated_at) if updated_at is not None else None,
    )
    try:
        await self._conn.session_update(session_id=session_id, update=update)
    except Exception:
        logger.debug("Could not send ACP session info update for %s", session_id, exc_info=True)
```

Adjust the DB accessor if `SessionDB.get_session()` uses a different key shape; inspect `hermes_state.py` before finalizing.

**Step 3: Call helper after auto-title**

In `prompt()`, after `maybe_auto_title(...)` succeeds/fails and before final response/update return, call:

```python
await self._send_session_info_update(session_id)
```

Only call it for turns with a non-empty final response to avoid noisy metadata updates on local slash commands unless intentionally needed later.

### Task 3: Verify focused tests

Run:

```bash
scripts/run_tests.sh tests/acp/test_server.py -q
```

Expected: PASS.

### Task 4: Manual Zed verification

Restart the active ACP server/Zed session. In Zed, start a new Hermes ACP thread and ask a normal question. Confirm the thread title/sidebar changes without reloading history.
