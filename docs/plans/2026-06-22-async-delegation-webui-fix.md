# Fix: Async Delegation Result Delivery for WebUI/TUI Gateway

## Problem

When `delegate_task` is called from the WebUI (hermes-webui), ALL delegations run
in the background automatically (forced by `_model_background_value()` at depth=0).
The completion event is pushed to `process_registry.completion_queue` and the
tui_gateway's `_notification_poller_loop` is supposed to pick it up and inject it
as a new agent turn.

**Three failure modes cause results to be silently lost:**

1. **Session busy → re-queue loop**: If the agent is still generating a response
   when the completion arrives, the event is re-queued (line 6374). If the session
   stays busy long enough (streaming, multiple tool calls), the event can be
   re-queued indefinitely. If the session closes during this window, the event is
   orphaned.

2. **Session closed before completion**: If the user closes the browser tab or
   navigates away before the subagent finishes, the `_notification_poller_loop`
   thread exits. No poller exists to drain the event.

3. **No recovery mechanism**: There is no API to query completed async delegations.
   If the push notification fails for any reason, the result is gone forever. The
   in-memory `_records` dict in `async_delegation.py` tracks completions but
   exposes no REST endpoint.

**Root cause in code:**
- `tui_gateway/server.py:_set_session_context()` does NOT pass `async_delivery=False`
  to `set_session_vars()`, so the default (`True`) is used. This means delegate_task
  takes the async dispatch path for WebUI sessions.
- The gateway's `_async_delegation_watcher` calls `_enrich_async_delegation_routing()`
  before injection, but the tui_gateway's poller does NOT — missing routing metadata.
- `APIServerAdapter.supports_async_delivery = False` is correctly set for the
  gateway's API server, but the tui_gateway is a separate server that bypasses
  the gateway's adapter system entirely.

## Approach

Two changes, both in hermes-agent:

### Change 1: Persist completed async delegation records
Add a `list_async_completions()` API to `async_delegation.py` that returns recent
completed delegations for a given session_key. The in-memory `_records` dict already
stores this data; we just need to expose it.

### Change 2: Recovery sweep in tui_gateway poller
When a `_notification_poller_loop` starts (new session connects), check for any
completed async delegations whose session_key matches the current session AND whose
completion event was never consumed. Inject those as new turns on connect.

### Change 3: REST endpoint for delegation status
Add `GET /api/delegations` to the tui_gateway that returns recent delegation records
for the current session, so the WebUI can poll if it suspects a missed result.

## Files to Modify

1. **`tools/async_delegation.py`** — Add `list_async_completions()` and
   `mark_async_delegation_consumed()` functions
2. **`tui_gateway/server.py`** — Add recovery sweep in `_notification_poller_loop`
   startup, add `_enrich_async_delegation_routing()` call before injection, add
   REST endpoint
3. **`tests/tools/test_async_delegation.py`** — Tests for new functions
4. **`tests/tui_gateway/test_async_delegation_recovery.py`** — Tests for recovery
   sweep and REST endpoint

## Tasks

### T1: Add `list_async_completions()` to async_delegation.py
- Filter `_records` by session_key and status != "running"
- Add a `consumed` flag to records so recovered completions aren't re-injected
- Return list of dicts with delegation_id, goal, summary, status, duration, timestamps

### T2: Add `mark_async_delegation_consumed()` to async_delegation.py
- Set a `consumed` flag on a completed record by delegation_id
- Prevents re-injection on subsequent poller cycles

### T3: Add recovery sweep to tui_gateway poller startup
- On poller start, call `list_async_completions(session_key)`
- For each unconsumed completion, format and inject as a new turn
- Mark as consumed after successful injection

### T4: Add `_enrich_async_delegation_routing` call in tui_gateway poller
- Before calling `format_process_notification(evt)`, ensure routing fields
  (platform, chat_id, thread_id) are populated on the event
- Copy the enrichment logic from gateway/run.py

### T5: Add REST endpoint `GET /api/delegations`
- Returns recent delegation records for the session
- Allows WebUI to poll for missed completions
- Returns JSON: {delegations: [{id, goal, status, summary, duration, ...}]}

### T6: Tests for list_async_completions and mark_consumed
- Test filtering by session_key
- Test consumed flag prevents re-listing
- Test empty results for unknown session

### T7: Tests for recovery sweep
- Test that unconsumed completions are injected on poller start
- Test that consumed completions are skipped
- Test that events from other sessions are not injected

### T8: Tests for REST endpoint
- Test endpoint returns completions for session
- Test endpoint returns empty for session with no completions
- Test endpoint only returns completions for the requesting session

## Verification

```bash
cd ~/.hermes/hermes-agent
python -m pytest tests/tools/test_async_delegation.py -v -o 'addopts='
python -m pytest tests/tui_gateway/test_async_delegation_recovery.py -v -o 'addopts='
python -m pytest tests/gateway/test_async_delivery_capability.py -v -o 'addopts='
```
