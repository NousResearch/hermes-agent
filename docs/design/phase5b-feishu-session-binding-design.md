# Phase 5B Design: Feishu Session Binding Integration

**Date:** 2026-06-02
**Phase:** 5B (design only — do not implement yet)
**Depends on:** Phase 5A (FeishuEntryAdapter)
**Baseline:** v2.10 Phase 5A commit

---

## 1. Current Feishu Flow Summary

```
Feishu WS/Webhook
    ↓
gateway/platforms/feishu.py::FeishuAdapter
    ↓
_on_message_event(data)
    ↓
_process_inbound_message(message, source, event)
    ↓
    chat_id, thread_id extracted
    ↓
    build_session_key(chat_id, thread_id, "feishu", bot_name)
        → session_key: "feishu:oc_xxx:thread:om_yyy:botname"
    ↓
_dispatch_inbound_event(event)
    ↓
    session_key stored in event metadata
    ↓
GatewayRunner._route_message(source, text, context)
    ↓
    SessionStore.get_or_create_session(session_key, source)
        → SessionEntry (session_id, session_key, origin, ...)
    ↓
    agent execution (conversation_loop / delegate_tool)
```

**Key observations:**
- Session identity is derived from `build_session_key()` which concatenates chat_id, thread_id, platform, bot_name
- `SessionStore` manages session lifecycle (create, reset, suspend, resume, prune)
- `SessionEntry` contains: session_key, session_id, origin (SessionSource), metadata
- There is **no** connection to v2.10 `SessionBinding` store or `Workspace`/`Session` models

---

## 2. Design Question 1: Safest Integration Point

**Answer: Sidecar observer inside `_process_inbound_message()`, after session_key is built but before `_dispatch_inbound_event()`.**

Why this point:
- `session_key` is already resolved (chat_id + thread_id + platform + bot_name)
- `source` (SessionSource) is already populated with chat_id, user_id, thread_id
- No transport code is touched
- No dispatch code is touched
- If the sidecar fails, the main flow continues uninterrupted

```
FeishuAdapter._process_inbound_message()
    ↓
    session_key = build_session_key(...)
    ↓
    [NEW] _record_session_binding(source, session_key)
        → normalize to EntryEvent via FeishuEntryAdapter
        → write to SessionBinding store
        → write to v2.10 Session store (optional, Phase 5C)
    ↓
    _dispatch_inbound_event(event)  [UNCHANGED]
```

---

## 3. Design Question 2: Feishu Identifiers → SessionBinding Mapping

| Feishu Field | SessionBinding Field | Notes |
|-------------|----------------------|-------|
| `tenant_id` (optional) | `external_source_id` | Used for workspace resolution |
| `chat_id` (`oc_xxx`) | `external_channel_id` | Primary channel identifier |
| `thread_id` / `root_id` (optional) | `external_thread_id` | Distinguishes thread vs channel |
| `open_id` (`ou_xxx`) | — | Not stored in binding (user identity) |
| `user_id` (`u_xxx`, optional) | — | Not stored in binding (user identity) |
| `message_id` (`om_xxx`) | — | Not stored in binding (event identifier) |
| `session_key` (generated) | — | Maps to `session_id` in v2.10 model |

**Binding key:** `feishu:{chat_id}:{thread_id or ""}`

**Binding value:** `(workspace_id, session_id)`

Where:
- `workspace_id` = `ws-feishu-{tenant_id}` or `ws-feishu-{chat_id}`
- `session_id` = `ses-feishu-{session_key_hash}` or the existing `session_key` itself

---

## 4. Design Question 3: session_key Mapping Strategy

**Answer: `session_key` maps to `Session.session_id` as a compatibility bridge.**

Three options considered:

| Option | Pros | Cons |
|--------|------|------|
| A. session_key → external_thread_id | Preserves full key | external_thread_id is for thread IDs, not session keys |
| B. session_key → binding key | Natural fit | Binding key is lookup key, not session identifier |
| C. session_key → session_id + metadata | Clean v2.10 model | **Chosen. Full backward compat.** |

**Chosen approach (C):**
- `session_id` = the existing `session_key` string (preserves all existing lookups)
- `workspace_id` = derived from tenant/chat (new v2.10 field)
- `entrypoint` = `"feishu"` (new v2.10 field)
- Store in `SessionBinding` as: `("feishu", chat_id, thread_id) → (workspace_id, session_key)`

This means:
- Existing `SessionStore` lookups by `session_key` continue to work
- New v2.10 `SessionBinding` lookups by `(entrypoint, channel_id, thread_id)` also work
- No migration needed

---

## 5. Design Question 4: Avoid Breaking Existing Feishu Behavior

**Rules:**

1. **No modification to `gateway/platforms/feishu.py` transport code.**
2. **Sidecar is optional** — if `FeishuEntryAdapter` or `SessionBinding` import fails, log and skip.
3. **All existing `session_key` strings remain valid** — no rename, no migration.
4. **SessionStore is untouched** — `get_or_create_session(session_key, source)` continues as before.
5. **Dispatch flow is unchanged** — `_dispatch_inbound_event()` receives the same event structure.

**Compatibility layer:**

```python
def _record_session_binding(source: SessionSource, session_key: str) -> None:
    """Sidecar: record Feishu session binding for v2.10 without affecting existing flow.
    
    This function is called after session_key is built but before dispatch.
    If any step fails, the exception is caught and logged; the main flow continues.
    """
    try:
        from agent.managed_agents.feishu_entry_adapter import FeishuEntryAdapter
        from agent.managed_agents.session_binding import put_binding
        
        # Build minimal raw payload from SessionSource
        raw = {
            "chat_id": source.chat_id,
            "message_id": source.message_id or f"auto-{time.time()}",
            "open_id": source.user_id or "unknown",
            "content": "",  # binding record doesn't need content
            "thread_id": source.thread_id,
            "session_key": session_key,
        }
        
        adapter = FeishuEntryAdapter()
        event = adapter.normalize_event(raw)
        
        put_binding(
            entrypoint="feishu",
            external_channel_id=source.chat_id,
            external_thread_id=source.thread_id,
            workspace_id=event.workspace_id,
            session_id=session_key,  # preserves existing session_key
        )
    except Exception:
        logger.debug("Session binding sidecar failed (non-critical): %s", exc_info=True)
```

---

## 6. Design Question 5: Avoid Double-Dispatching

**Answer: The sidecar does NOT dispatch. It only writes to SessionBinding store.**

- Existing flow: `FeishuAdapter` → `GatewayRunner._route_message()` → agent execution
- New sidecar: `FeishuAdapter` → `_record_session_binding()` (write only) → existing flow continues
- No EntryEvent is passed to Hermes Core from the sidecar
- No task is created
- No agent is called

The binding record is written once per session (idempotent — `put_binding` overwrites).

---

## 7. Design Question 6: EntryEvents Without Task Creation

**Answer: Use `FeishuEntryAdapter.normalize_event()` only for normalization, not dispatch.**

The sidecar:
1. Creates a minimal raw payload from `SessionSource`
2. Calls `FeishuEntryAdapter.normalize_event(raw)` to get a canonical `EntryEvent`
3. Extracts `workspace_id` and `session_id` from the `EntryEvent`
4. Writes to `SessionBinding` store
5. **Does NOT** call `EntryAdapterRegistry.ingest()`
6. **Does NOT** create a task
7. **Does NOT** call agents

The `EntryEvent` is used as a data normalization vehicle, not a dispatch trigger.

---

## 8. Design Question 7: Approval/Notification Mode Separation

**Answer: Phase 5B does NOT implement approval/notification mode.**

Approval/notification mode is a **Phase 5C or v3.0** feature:
- It requires config changes (`FEISHU_MODE = "primary" | "notification"`)
- It requires GatewayRunner routing changes
- It requires UI changes in Web Console

Phase 5B is purely about **recording the binding** so that future phases can:
- Query which workspace/session a Feishu chat/thread belongs to
- Filter Feishu sessions in Web Console
- Route approval messages to Feishu without duplicating the mapping logic

---

## 9. Design Question 8: External Martini Bot

**Answer: The sidecar approach is safe regardless of where Martini bot lives.**

If Martini bot is external:
- It uses the same Feishu app credentials (same tenant/chat/thread IDs)
- It uses the same `build_session_key()` logic (or it should)
- The binding record is keyed by Feishu identifiers, not by bot instance
- Both this repo's FeishuAdapter and the external Martini bot can write to the same binding store

If Martini bot is in a separate repo:
- That repo can import `agent.managed_agents.session_binding.put_binding()`
- Or it can read the binding file directly (`data/session_bindings.json`)
- No code coupling required

---

## 10. SessionBinding Strategy

### Data flow

```
FeishuAdapter._process_inbound_message()
    ↓
    session_key = build_session_key(chat_id, thread_id, "feishu", bot_name)
    ↓
    _record_session_binding(source, session_key)
        ↓
        put_binding(
            entrypoint="feishu",
            external_channel_id=chat_id,
            external_thread_id=thread_id,
            workspace_id=workspace_id,
            session_id=session_key,
        )
        ↓
        SessionBinding store (JSON file: data/session_bindings.json)
```

### Lookup flow (future use)

```
Web Console / Operations Dashboard
    ↓
    get_binding("feishu", chat_id, thread_id)
        → (workspace_id, session_id)
    ↓
    Display: "Feishu chat oc_xxx → workspace ws-feishu-xxx, session ses-feishu-..."
```

---

## 11. Compatibility Rules

| Rule | Enforcement |
|------|------------|
| No transport changes | `gateway/platforms/feishu.py` WebSocket/Webhook code untouched |
| No dispatch changes | `_dispatch_inbound_event()` receives same event structure |
| No session store changes | `SessionStore.get_or_create_session()` unchanged |
| No breaking changes | `session_key` strings remain valid |
| Sidecar is optional | Try/except around entire sidecar; failure logs and continues |
| Idempotent writes | `put_binding` overwrites; same binding written N times is safe |
| No new dependencies | Uses existing `FeishuEntryAdapter` and `SessionBinding` from v2.10 |

---

## 12. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Sidecar adds latency to message processing | Low | Write is in-memory + async JSON save; <1ms overhead |
| SessionBinding store file grows unbounded | Low | Existing `session_binding.py` has no eviction; add TTL or prune in Phase 5C |
| Existing session_key format is long and contains colons | Low | Store as-is; colons are safe in JSON strings |
| `source.message_id` may be None | Low | Fallback to auto-generated ID in sidecar |
| Two repos writing to same binding file | Low | JSON file is overwritten on each write; last write wins (acceptable for idempotent mapping) |
| FeishuEntryAdapter.normalize_event() requires content field | Low | Pass empty string for binding records |

---

## 13. Non-Goals (Explicitly Excluded)

- Do NOT implement approval/notification mode
- Do NOT modify GatewayRunner routing
- Do NOT create tasks from Feishu messages
- Do NOT call agents from the sidecar
- Do NOT write ledger events from the sidecar
- Do NOT modify Web Console UI
- Do NOT modify SessionStore schema
- Do NOT migrate existing session records
- Do NOT require external Martini bot changes

---

## 14. Implementation Slices

### Slice B1: Sidecar function (this repo)

Add `_record_session_binding()` to `gateway/platforms/feishu.py` or as a separate module:

```python
# gateway/platforms/feishu_session_binding_bridge.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)


def record_feishu_session_binding(source: "SessionSource", session_key: str) -> None:
    """Sidecar: record Feishu session binding for v2.10 without affecting existing flow.
    
    Called from FeishuAdapter._process_inbound_message() after session_key is built.
    Failures are caught and logged; the main flow continues uninterrupted.
    """
    try:
        from agent.managed_agents.feishu_entry_adapter import FeishuEntryAdapter
        from agent.managed_agents.session_binding import put_binding
        import time

        raw = {
            "chat_id": source.chat_id,
            "message_id": getattr(source, "message_id", None) or f"auto-{int(time.time())}",
            "open_id": source.user_id or "unknown",
            "content": "",
            "thread_id": source.thread_id,
            "session_key": session_key,
        }

        adapter = FeishuEntryAdapter()
        event = adapter.normalize_event(raw)

        put_binding(
            entrypoint="feishu",
            external_channel_id=source.chat_id,
            external_thread_id=source.thread_id,
            workspace_id=event.workspace_id,
            session_id=session_key,
        )
        logger.debug("Recorded session binding for Feishu session_key=%s", session_key)
    except Exception:
        logger.debug("Session binding sidecar failed (non-critical): %s", exc_info=True)
```

### Slice B2: Hook into FeishuAdapter (this repo)

Add one line call in `FeishuAdapter._process_inbound_message()`:

```python
# In gateway/platforms/feishu.py, inside _process_inbound_message()
# After session_key is built, before _dispatch_inbound_event():

from gateway.platforms.feishu_session_binding_bridge import record_feishu_session_binding
record_feishu_session_binding(source, session_key)
```

### Slice B3: Tests (this repo)

- Test: `_record_session_binding` writes to SessionBinding store
- Test: `_record_session_binding` fails gracefully (exception caught)
- Test: Binding is idempotent (same key written twice → same result)
- Test: Existing Feishu flow continues when sidecar fails
- Test: Binding lookup returns correct workspace/session

### Slice B4: Documentation (this repo)

- Update `docs/plans/v2.10-multi-entry-session-binding-plan.md` with Phase 5B completion
- Update ADR if needed

---

## 15. Tests Required

| Test | Scope |
|------|-------|
| `test_record_binding_from_session_source` | Sidecar writes binding correctly |
| `test_record_binding_graceful_failure` | Exception caught, main flow continues |
| `test_record_binding_idempotent` | Same key written twice → same result |
| `test_binding_lookup_returns_workspace_session` | get_binding returns correct values |
| `test_binding_fallback_to_defaults` | get_binding returns defaults for unknown key |
| `test_existing_feishu_flow_unchanged` | _dispatch_inbound_event receives same event |
| `test_session_store_untouched` | SessionStore.get_or_create_session unchanged |

---

## 16. Go / No-Go Recommendation

**Recommendation: GO for Phase 5B.**

Rationale:
- Sidecar approach is the safest integration pattern
- Zero changes to transport, dispatch, or session store
- Existing behavior is fully preserved
- Sidecar failures are non-fatal
- All v2.10 models (EntryEvent, SessionBinding) already exist
- FeishuEntryAdapter from Phase 5A is ready to use
- No external dependencies (Martini bot can be handled separately)

**Conditions for GO:**
1. Slice B1 (bridge module) reviewed and approved
2. Slice B2 (one-line hook) reviewed and approved
3. Slice B3 (tests) written and passing
4. No transport code changes in `gateway/platforms/feishu.py` except the one-line hook

---

## 17. Next Phase After 5B

| Phase | Description | Depends On |
|-------|-------------|-----------|
| Phase 5C | Feishu approval/notification mode config | Phase 5B + config changes |
| Phase 6 | Web Console Session UI | Phase 5B + frontend work |
| Phase 7 | Mac App Shell | Phase 6 + Mac app work |

---

## 18. Files Involved

### Modified
- `gateway/platforms/feishu.py` — add one-line sidecar call (Slice B2)

### New
- `gateway/platforms/feishu_session_binding_bridge.py` — sidecar bridge (Slice B1)
- `tests/gateway/test_feishu_session_binding_bridge.py` — tests (Slice B3)

### Existing (unchanged, used by sidecar)
- `agent/managed_agents/feishu_entry_adapter.py` — Phase 5A
- `agent/managed_agents/session_binding.py` — v2.10 Phase 1
- `gateway/session.py` — existing session store
