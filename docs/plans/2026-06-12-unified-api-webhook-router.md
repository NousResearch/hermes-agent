# Unified API/Webhook Router Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a shared ingress seam for HTTP-driven agent entrypoints so API server runs, generic webhooks, and Graph-style webhooks can converge on one normalized request/event pipeline without changing public behavior.

**Architecture:** Start with a narrow internal seam instead of a big-bang router rewrite. Introduce a small normalized ingress envelope/helper layer used by webhook-style adapters first, then progressively move API server request parsing and run/session orchestration onto the same primitives once the envelope and dispatch contracts are proven stable.

**Tech Stack:** Python 3.11, aiohttp, Hermes gateway adapters, SessionDB/state.db, pytest

---

## Confirmed codebase context

- Generic webhooks currently terminate in `gateway/platforms/webhook.py` and convert an HTTP POST directly into a `MessageEvent`, then call `self.handle_message(event)` in a background task.
- Microsoft Graph webhook ingress in `gateway/platforms/msgraph_webhook.py` independently builds a `MessageEvent` and independently schedules `self.handle_message(event)`.
- API server ingress in `gateway/platforms/api_server.py` does **not** use `MessageEvent` or `BasePlatformAdapter.handle_message()`. It parses HTTP bodies directly, creates an `AIAgent` directly via `_create_agent()`, and separately implements session continuity, SSE streaming, approval wiring, and run status/event plumbing.
- The safest confirmed unification seam today is **before** agent execution but **after** request validation/parsing: a normalized internal ingress envelope plus shared dispatch helpers for HTTP-originated event sources.

## Design constraints

- Do **not** change public HTTP routes, response schemas, or authentication semantics in the first slice.
- Do **not** push API server traffic through the gateway active-session queue until the session/approval/streaming semantics are explicitly mapped.
- Preserve prompt-caching invariants: no mid-conversation toolset or system-prompt mutation.
- Keep the first slice reversible and low-risk: internal-only helpers with immediate concrete consumers.

---

### Task 1: Add a shared ingress envelope module

**Objective:** Create a single internal representation for HTTP-originated gateway events.

**Files:**
- Create: `gateway/ingress.py`
- Test: `tests/gateway/test_ingress.py`

**Step 1: Write failing tests**

Add tests that expect:
- an `IngressEnvelope` dataclass carrying normalized event fields
- a helper that turns an envelope into a `MessageEvent` using a platform adapter
- a helper that schedules `adapter.handle_message(event)` and tracks the background task set

Example assertions to add:

```python
envelope = IngressEnvelope(
    text="hello",
    message_id="evt-1",
    chat_id="webhook:route:evt-1",
    chat_name="webhook/route",
    chat_type="webhook",
    user_id="webhook:route",
    user_name="route",
    raw_payload={"ok": True},
)

event = build_ingress_message_event(adapter, envelope)
assert event.text == "hello"
assert event.message_id == "evt-1"
assert event.source.platform == Platform.WEBHOOK
assert event.source.chat_id == "webhook:route:evt-1"
```

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/gateway/test_ingress.py -q -o 'addopts='`
Expected: FAIL — module/functions do not exist yet

**Step 3: Write minimal implementation**

Create `gateway/ingress.py` with:
- `IngressEnvelope` dataclass
- `build_ingress_message_event(adapter, envelope)`
- `schedule_ingress_envelope(adapter, envelope)`
- `schedule_ingress_event(adapter, event)`

Keep the module internal and dependency-light.

**Step 4: Run test to verify pass**

Run: `python -m pytest tests/gateway/test_ingress.py -q -o 'addopts='`
Expected: PASS

**Step 5: Commit**

```bash
git add gateway/ingress.py tests/gateway/test_ingress.py
git commit -m "feat: add shared ingress envelope helpers"
```

---

### Task 2: Move generic webhook agent-mode dispatch onto the shared seam

**Objective:** Replace webhook-local event construction/dispatch boilerplate with the shared ingress helper.

**Files:**
- Modify: `gateway/platforms/webhook.py`
- Test: `tests/gateway/test_webhook_adapter.py`

**Step 1: Write failing test**

Add or extend a test so it verifies the accepted webhook still:
- returns HTTP 202
- produces the same session-scoped `chat_id`
- sends a `MessageEvent` with the route-derived identity fields
- preserves `delivery_id` as `message_id`

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/gateway/test_webhook_adapter.py -q -o 'addopts='`
Expected: FAIL after the new expectations are added

**Step 3: Write minimal implementation**

In `gateway/platforms/webhook.py`:
- import the new ingress helper(s)
- replace the inline `build_source(...)`, `MessageEvent(...)`, `asyncio.create_task(...)`, and `_background_tasks` plumbing with one `IngressEnvelope` + `schedule_ingress_envelope(...)`
- keep `deliver_only` untouched
- keep response payloads/status codes untouched

**Step 4: Run test to verify pass**

Run: `python -m pytest tests/gateway/test_webhook_adapter.py -q -o 'addopts='`
Expected: PASS

**Step 5: Commit**

```bash
git add gateway/platforms/webhook.py tests/gateway/test_webhook_adapter.py
git commit -m "refactor: route webhook ingress through shared envelope"
```

---

### Task 3: Move MS Graph webhook dispatch onto the same seam

**Objective:** Make the second webhook-like ingress source consume the same envelope helper.

**Files:**
- Modify: `gateway/platforms/msgraph_webhook.py`
- Test: `tests/gateway/test_msgraph_webhook.py`

**Step 1: Write failing test**

Add or extend a test so it verifies accepted notifications still:
- return HTTP 202
- create a `MessageEvent` with `Platform.MSGRAPH_WEBHOOK`
- preserve the receipt key as `message_id`
- schedule through the shared path

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/gateway/test_msgraph_webhook.py -q -o 'addopts='`
Expected: FAIL after the new expectation is added

**Step 3: Write minimal implementation**

In `gateway/platforms/msgraph_webhook.py`:
- import the new ingress helper(s)
- replace `_build_message_event()` and the default scheduler branch with `IngressEnvelope` + `schedule_ingress_envelope(...)`
- keep the custom scheduler extension point intact

**Step 4: Run test to verify pass**

Run: `python -m pytest tests/gateway/test_msgraph_webhook.py -q -o 'addopts='`
Expected: PASS

**Step 5: Commit**

```bash
git add gateway/platforms/msgraph_webhook.py tests/gateway/test_msgraph_webhook.py
git commit -m "refactor: route msgraph webhook ingress through shared envelope"
```

---

### Task 4: Add normalized HTTP request metadata helpers for API/webhook convergence

**Objective:** Introduce shared request-metadata plumbing without changing wire behavior.

**Files:**
- Modify: `gateway/ingress.py`
- Modify: `gateway/platforms/api_server.py`
- Modify: `gateway/platforms/webhook.py`
- Test: `tests/gateway/test_ingress.py`
- Test: `tests/gateway/test_api_server_jobs.py`

**Step 1: Write failing tests**

Add tests for a small request-context helper that extracts sanitized:
- `remote`
- `peer_ip`
- `forwarded_for`
- `real_ip`
- `user_agent`
- `method`
- `path`

and keeps line breaks/control characters out of loggable values.

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/gateway/test_ingress.py tests/gateway/test_api_server_jobs.py -q -o 'addopts='`
Expected: FAIL

**Step 3: Write minimal implementation**

- move the generic request-audit extraction logic out of `api_server.py` into `gateway/ingress.py`
- keep API server response behavior unchanged
- optionally use the same helper for webhook logging/trace context, but do not alter webhook JSON responses

**Step 4: Run test to verify pass**

Run: `python -m pytest tests/gateway/test_ingress.py tests/gateway/test_api_server_jobs.py -q -o 'addopts='`
Expected: PASS

**Step 5: Commit**

```bash
git add gateway/ingress.py gateway/platforms/api_server.py gateway/platforms/webhook.py tests/gateway/test_ingress.py tests/gateway/test_api_server_jobs.py
git commit -m "refactor: share ingress request metadata extraction"
```

---

### Task 5: Document the next API-server migration seam

**Objective:** Leave the next implementer with an exact, safe follow-on slice.

**Files:**
- Modify: `docs/plans/2026-06-12-unified-api-webhook-router.md`

**Step 1: Add follow-on notes**

Document the next migration target precisely:
- extract API-server request parsing into pure helpers
- normalize `chat_completions`, `session_chat`, and `runs` into one internal request model
- only then consider a shared router/executor layer

**Step 2: Verification**

Read the updated plan and confirm it distinguishes:
- current thin slice
- next safe vertical slice
- work explicitly deferred because it is higher risk

**Step 3: Commit**

```bash
git add docs/plans/2026-06-12-unified-api-webhook-router.md
git commit -m "docs: refine unified ingress follow-on slices"
```

---

## Smallest safe vertical slice

1. Add `gateway/ingress.py`.
2. Route `gateway/platforms/webhook.py` through it.
3. Route `gateway/platforms/msgraph_webhook.py` through it.
4. Add tests proving event identity and scheduling semantics are unchanged.

This is the minimum slice that creates a real unification seam with **two live ingress consumers** and no API contract changes.

## Validation commands

Run these from repo root:

```bash
python -m pytest tests/gateway/test_ingress.py -q -o 'addopts='
python -m pytest tests/gateway/test_webhook_adapter.py -q -o 'addopts='
python -m pytest tests/gateway/test_msgraph_webhook.py -q -o 'addopts='
python -m pytest tests/gateway/test_api_server_jobs.py -q -o 'addopts='
python -m pytest tests/gateway/test_webhook_adapter.py tests/gateway/test_msgraph_webhook.py tests/gateway/test_api_server_jobs.py tests/gateway/test_ingress.py -q -o 'addopts='
```

If broader time permits:

```bash
python -m pytest tests/gateway/ -q -o 'addopts='
```

## Risks

- **API server is structurally different today.** Forcing it onto `MessageEvent` too early risks breaking SSE streams, approval waits, and session continuity headers.
- **Webhook route semantics differ from chat semantics.** Route-specific delivery metadata must stay outside the generic envelope or be clearly marked optional.
- **Background task lifecycle matters.** Any shared scheduler helper must preserve `_background_tasks` tracking so disconnect/cleanup behavior remains unchanged.

## Rollback notes

- The thin slice is fully reversible: revert `gateway/ingress.py` plus the adapter call sites in `gateway/platforms/webhook.py` and `gateway/platforms/msgraph_webhook.py`.
- If a regression appears, roll back only the shared-dispatch adoption while keeping the plan document.
- Do **not** partially roll back just one consumer after the helper lands unless tests prove the other consumer still exercises the helper correctly.

## Explicitly deferred

- No public route consolidation yet
- No API server transport rewrite yet
- No shared approval/session executor yet
- No dashboard/control-plane contract changes yet

## Next Migration Seam (Phase 2)

To safely continue the unified router migration without breaking SSE streams, approval waits, or session continuity headers, the next implementer must adhere to this strict sequence:

1. **Extract API-server request parsing:** Move the parsing logic out of `_handle_chat_completions`, `_handle_session_chat`, and `_handle_runs` into pure, testable helpers (similar to the `/v1/responses` slice).
2. **Normalize Request Models:** Converge `NormalizedSessionChatRequest`, `NormalizedRunRequest`, and `NormalizedChatCompletionsRequest` into a single, cohesive internal request model inside `gateway/ingress.py`.
3. **Shared Executor Layer:** Only after parsing and normalization are fully shared and proven green, attempt to converge on a shared router/executor layer that safely handles the gateway's active-session queue.
