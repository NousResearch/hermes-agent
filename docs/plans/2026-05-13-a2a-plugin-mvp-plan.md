# Hermes A2A Plugin MVP Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add an experimental A2A (Agent-to-Agent) plugin to Hermes Agent so one Hermes profile can talk to another over a standardized HTTP protocol with Agent Card discovery and a minimal task lifecycle.

**Architecture:** Implement A2A as a standalone Hermes plugin mounted on the existing gateway/plugin API surface. The MVP will expose an Agent Card at `/.well-known/agent.json`, accept JSON-RPC 2.0 task submissions over HTTP, bridge each incoming A2A task into a real Hermes session running under the target profile, and return a minimal task state machine (`submitted → working → completed/failed`). Multi-turn `input-required`, SSE streaming, and external-agent interoperability validation are deferred until after the first internal Hermes-to-Hermes round-trip works.

**Tech Stack:** Hermes Agent plugin system, FastAPI/Starlette router, existing gateway session/runtime plumbing, JSON-RPC 2.0, optional `a2a-sdk` later (not required for MVP), pytest.

---

## Why this exists

Hermes profiles are currently isolated in the places that matter most for real multi-agent operations:

- per-profile `MEMORY.md` / `USER.md`
- per-profile session transcripts
- no native cross-profile session search
- no native inter-agent request/response protocol

Today the only shared bridge is Hindsight, which is useful but not sufficient. It is recall-oriented, not conversation-oriented. If Switch needs Neo's actual reasoning about a past task, Switch cannot ask Neo directly. It can only hope the right fact was retained and is retrievable.

A2A is the correct missing layer because it gives Hermes:

- standardized agent discovery
- live agent-to-agent requests
- durable task identities
- eventual multi-turn negotiation via `input-required`
- a path to external agent interoperability without inventing a Hermes-only protocol first

This plan intentionally starts with the smallest useful slice: **internal Hermes profile ↔ Hermes profile communication**.

---

## Research summary

### Prior internal findings

From prior Hermes and Construct research:

- A2A is the right protocol boundary for **agent ↔ agent** communication.
- MCP remains the right boundary for **agent ↔ tool** calls.
- Hermes `delegate_task` / ACP handles **internal one-shot subprocess delegation**, not persistent peer-to-peer conversation.
- Hermes Kanban `task_events` can still act as a workflow/event substrate, but they are not a substitute for direct live inter-agent messaging.

### External protocol findings

From the A2A spec and ecosystem research:

- transport: JSON-RPC 2.0 over HTTP(S)
- discovery: Agent Card at `/.well-known/agent.json`
- task lifecycle: `submitted → working → input-required → completed/failed/canceled`
- supports: sync, async, SSE streaming, push notifications
- language SDKs exist, but MVP can be done with direct JSON parsing and plain FastAPI

### Product stance for Hermes MVP

The MVP should **not** try to implement the entire spec.

That would be ceremony without proof.

The MVP should prove four things only:

1. one Hermes profile can discover another via Agent Card
2. one Hermes profile can submit a task to another over HTTP
3. the target profile can execute the request in its own real session context
4. the caller can receive a structured result with task status and final output

If those four work, the rest is iteration.

---

## Scope

## In scope for MVP

- standalone Hermes plugin: `plugins/a2a/`
- plugin HTTP API mounted on gateway
- Agent Card endpoint
- minimal JSON-RPC request handler
- minimal task storage/in-memory registry for active tasks
- bridging incoming task into Hermes runtime under the receiving profile
- synchronous request path first
- minimal status values: `submitted`, `working`, `completed`, `failed`
- explicit config block for enable/disable and profile exposure
- tests for card serving, request validation, task execution, failure handling
- docs + example curl commands

## Out of scope for MVP

- full A2A spec compliance
- `input-required`
- SSE streaming
- push notifications
- external auth schemes beyond local trusted deployment
- agent registry/discovery service
- Kanban integration
- Telegram bot relay
- cross-profile session search
- persistent task DB storage
- external non-Hermes clients beyond basic curl/manual validation

---

## Proposed user-visible behavior

### Receiving side

If profile `neo` runs Hermes gateway with the A2A plugin enabled, it exposes:

- `GET /.well-known/agent.json`
- `POST /api/plugins/a2a/rpc`
- optionally `GET /api/plugins/a2a/tasks/{task_id}` for debugging/task inspection

### Sending side

A caller can send a JSON-RPC request with a task payload such as:

```json
{
  "jsonrpc": "2.0",
  "id": "req-123",
  "method": "tasks/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        { "type": "text", "text": "Summarize what you know about issue X and recommend a fix." }
      ]
    },
    "metadata": {
      "from_agent": "switch",
      "conversation_id": "telegram-7341688567-2026-05-13"
    }
  }
}
```

The receiver translates that into a Hermes session prompt, runs the local profile's agent, and returns:

```json
{
  "jsonrpc": "2.0",
  "id": "req-123",
  "result": {
    "task": {
      "id": "a2a_task_...",
      "status": "completed"
    },
    "artifacts": [
      {
        "type": "text",
        "text": "...final response from neo..."
      }
    ]
  }
}
```

---

## Proposed plugin shape

### Directory

```text
plugins/a2a/
├── plugin.yaml
├── __init__.py
├── server.py
├── models.py
├── task_manager.py
├── agent_card.py
└── README.md
```

### `plugin.yaml`

Use `kind: standalone`.

Expected fields:

```yaml
name: a2a
version: "0.1.0"
description: Experimental Agent-to-Agent protocol plugin for Hermes profiles
author: Hermes
kind: standalone
provides_tools: []
hooks: []
requires_env: []
```

### `__init__.py`

Responsibilities:

- register plugin API router with the dashboard/gateway plugin surface
- load config
- construct task manager singleton
- expose router handlers

### `models.py`

Define minimal typed structures:

- `A2AAgentCard`
- `A2ATask`
- `A2ATaskStatus`
- `JsonRpcRequest`
- `JsonRpcSuccess`
- `JsonRpcError`

Keep this local and small. Do not pull in a heavy SDK until Hermes proves the shape.

### `agent_card.py`

Build a profile-specific Agent Card using:

- current profile name
- optional display name from config
- endpoint URL from gateway host/port if available, otherwise relative path
- skills/capabilities from config, not auto-derived in MVP

### `task_manager.py`

Responsibilities:

- create task ids
- validate supported methods
- hold active task states in memory
- call the Hermes runtime to execute the incoming request
- capture final response / error
- expose lookup by task id

### `server.py`

FastAPI router endpoints:

- `GET /.well-known/agent.json`
- `POST /api/plugins/a2a/rpc`
- `GET /api/plugins/a2a/tasks/{task_id}`

---

## Runtime bridging design

This is the critical part.

The A2A plugin is not its own agent. It is a protocol shim in front of a real Hermes profile.

### Rule

Every inbound A2A task must execute through the receiving profile's normal Hermes runtime so it has access to:

- that profile's `SOUL.md`
- that profile's `MEMORY.md` / `USER.md`
- that profile's enabled tools
- that profile's config, model, provider routing
- that profile's session storage

### MVP execution path

1. Receive JSON-RPC request
2. Validate supported method (`tasks/send` only in MVP)
3. Extract incoming text from `message.parts`
4. Build a normalized internal prompt wrapper such as:

```text
You are responding to an A2A request from another agent.

Caller: switch
Conversation metadata: ...

User request:
<message text>

Respond directly with the result. Do not explain the protocol.
```

5. Start a real Hermes agent run inside the current profile context
6. Wait for completion synchronously
7. Write task status transitions in memory registry
8. Return JSON-RPC result

### Implementation note

Prefer reusing the same code path the API server/gateway already uses for ordinary chat requests, instead of inventing a second execution path. The plugin should be as thin as possible.

If there is no clean reusable session entry point, add a minimal helper in core runtime rather than embedding agent boot logic inside the plugin.

---

## Config design

Add a new config section:

```yaml
a2a:
  enabled: true
  public_base_url: "http://127.0.0.1:8642"
  agent_name: "Neo"
  agent_description: "Hermes specialist profile for coding and technical diagnosis"
  skills:
    - ask_questions
    - summarize_context
    - review_code
  allow_methods:
    - tasks/send
  require_localhost: true
```

### Config rules

- default: disabled
- if enabled and `require_localhost=true`, reject non-local requests unless explicitly relaxed
- do not auto-publish tools as capabilities yet
- do not attempt auth negotiation in MVP

---

## File-by-file plan

### Task 1: Inspect plugin API mounting and choose the cleanest integration point

**Objective:** Confirm exactly how a new plugin exposes HTTP routes in Hermes and identify the smallest valid A2A plugin skeleton.

**Files:**
- Read: `hermes_cli/plugins.py`
- Read: `plugins/kanban/dashboard/plugin_api.py`
- Read: one additional simple plugin with API surface if present
- Create notes in plan only; no code yet

**Step 1: Locate route registration pattern**

Read the plugin manager and the kanban plugin API mounting path.

**Step 2: Identify the exact entry point expected by plugin API modules**

Document whether Hermes expects `dashboard/plugin_api.py`, direct router registration, or another pattern for plugin HTTP routes.

**Step 3: Verify path namespace rules**

Confirm whether the plugin can expose `/.well-known/agent.json` directly or whether Hermes only mounts under `/api/plugins/<name>/...`.

**Step 4: Record decision in code comments or README draft**

If direct root mount is impossible, use a small gateway/core patch to allow a well-known route registration hook.

**Step 5: Commit**

```bash
git add [only if code/comments were created]
git commit -m "docs: note a2a plugin route integration constraints"
```

### Task 2: Create the standalone A2A plugin skeleton

**Objective:** Add the plugin directory, manifest, registration entry point, and a minimal API router that loads without errors.

**Files:**
- Create: `plugins/a2a/plugin.yaml`
- Create: `plugins/a2a/__init__.py`
- Create: `plugins/a2a/server.py`
- Create: `plugins/a2a/README.md`
- Test: plugin-loading test file if appropriate under `tests/`

**Step 1: Write a failing plugin-load test**

Test should assert Hermes can discover the `a2a` plugin and import it successfully when enabled.

**Step 2: Add `plugin.yaml`**

Use minimal standalone manifest.

**Step 3: Add registration stub**

`register(ctx)` should install the router or whatever the plugin API requires.

**Step 4: Add a dummy health endpoint**

Return JSON like `{ "ok": true, "plugin": "a2a" }`.

**Step 5: Run targeted tests**

Use pytest only on the new plugin test.

**Step 6: Commit**

```bash
git add plugins/a2a tests/
git commit -m "feat: add a2a plugin skeleton"
```

### Task 3: Add Agent Card model and endpoint

**Objective:** Expose a valid minimal Agent Card for the current Hermes profile.

**Files:**
- Create: `plugins/a2a/models.py`
- Create: `plugins/a2a/agent_card.py`
- Modify: `plugins/a2a/server.py`
- Test: `tests/plugins/test_a2a_agent_card.py`

**Step 1: Write failing test for `GET /.well-known/agent.json` or fallback mounted path**

Assert response shape contains name, description, capabilities/skills, and endpoint.

**Step 2: Implement Agent Card builder**

Read profile/config values and produce deterministic JSON.

**Step 3: Serve the card endpoint**

If direct `/.well-known/agent.json` is not possible in plugin scope, serve under `/api/plugins/a2a/agent.json` first and add a clearly marked follow-up task to expose the canonical path.

**Step 4: Validate response shape**

Keep the payload minimal, human-readable, and stable.

**Step 5: Commit**

```bash
git add plugins/a2a tests/plugins/test_a2a_agent_card.py
git commit -m "feat: expose minimal a2a agent card"
```

### Task 4: Add JSON-RPC request parsing and method validation

**Objective:** Accept well-formed RPC requests and reject malformed/unsupported ones cleanly.

**Files:**
- Modify: `plugins/a2a/models.py`
- Modify: `plugins/a2a/server.py`
- Create: `tests/plugins/test_a2a_rpc_validation.py`

**Step 1: Write failing tests**

Cases:
- invalid JSON-RPC version
- missing id
- unknown method
- missing message text payload
- valid `tasks/send`

**Step 2: Implement local request/response models**

Return proper JSON-RPC error objects with code/message.

**Step 3: Add `POST /api/plugins/a2a/rpc`**

Only support `tasks/send` in MVP.

**Step 4: Re-run tests**

Make sure invalid payloads fail predictably.

**Step 5: Commit**

```bash
git add plugins/a2a tests/plugins/test_a2a_rpc_validation.py
git commit -m "feat: add a2a json-rpc validation"
```

### Task 5: Build in-memory task manager with lifecycle transitions

**Objective:** Track submitted work with a real internal task object and observable status transitions.

**Files:**
- Create: `plugins/a2a/task_manager.py`
- Modify: `plugins/a2a/models.py`
- Modify: `plugins/a2a/server.py`
- Create: `tests/plugins/test_a2a_task_manager.py`

**Step 1: Write failing tests for task creation and state changes**

Assert:
- task created with `submitted`
- task moves to `working`
- task ends as `completed` or `failed`
- lookup endpoint returns final task state

**Step 2: Implement task registry**

Use in-memory dict keyed by `task_id`.

**Step 3: Add task inspection endpoint**

`GET /api/plugins/a2a/tasks/{task_id}`

**Step 4: Confirm concurrent requests do not overwrite each other**

A simple lock is enough for MVP.

**Step 5: Commit**

```bash
git add plugins/a2a tests/plugins/test_a2a_task_manager.py
git commit -m "feat: add a2a task lifecycle manager"
```

### Task 6: Bridge `tasks/send` into a real Hermes profile run

**Objective:** Execute inbound A2A tasks through the receiving profile's real Hermes runtime and return the final response.

**Files:**
- Modify: `plugins/a2a/task_manager.py`
- Modify: `plugins/a2a/server.py`
- Possibly modify: shared runtime helper in `gateway/` or `run_agent.py` only if needed
- Create: `tests/plugins/test_a2a_execution.py`

**Step 1: Write failing execution test**

Mock the runtime first if needed, but also add one higher-level integration test that proves the bridge function is called.

**Step 2: Implement prompt wrapper**

Wrap inbound A2A request into a normalized Hermes prompt preserving caller metadata.

**Step 3: Call the normal Hermes execution path**

Do not fork a weird new path if existing session execution helpers exist.

**Step 4: Map success/failure to A2A result**

- success → `completed`
- exception/runtime failure → `failed`

**Step 5: Return text artifact**

Use a single text artifact in MVP.

**Step 6: Commit**

```bash
git add plugins/a2a tests/plugins/test_a2a_execution.py [any shared runtime helper]
git commit -m "feat: execute a2a tasks through hermes runtime"
```

### Task 7: Add local-only guardrails and config wiring

**Objective:** Prevent accidental exposure while the feature is experimental.

**Files:**
- Modify: config handling path(s) for plugin config
- Modify: `plugins/a2a/server.py`
- Create: `tests/plugins/test_a2a_security.py`
- Update: docs if config docs live in `website/`

**Step 1: Write failing tests**

Assert remote requests are rejected when `require_localhost=true`.

**Step 2: Read config and apply defaults**

Disabled by default. Local-only by default.

**Step 3: Return explicit error on disallowed origin**

Message should make it obvious that this is intentional, not a crash.

**Step 4: Re-run tests**

**Step 5: Commit**

```bash
git add plugins/a2a tests/plugins/test_a2a_security.py
git commit -m "feat: add local-only guardrails for a2a plugin"
```

### Task 8: Add end-to-end manual validation instructions

**Objective:** Make the experiment runnable by a human without tribal knowledge.

**Files:**
- Update: `plugins/a2a/README.md`
- Create: `docs/plans/` adjacent follow-up notes only if needed
- Optionally update: website docs if worth it

**Step 1: Document two-profile setup**

Example:
- `switch` profile gateway on one port
- `neo` profile gateway on another port

**Step 2: Document discovery test**

```bash
curl http://127.0.0.1:<neo-port>/.well-known/agent.json
```

**Step 3: Document task send test**

Include a full JSON-RPC curl example.

**Step 4: Document expected output and failure modes**

Include notes for plugin not enabled, localhost rejection, runtime exception.

**Step 5: Commit**

```bash
git add plugins/a2a/README.md
git commit -m "docs: add a2a plugin mvp validation guide"
```

---

## Suggested implementation notes

### Prefer direct implementation over SDK for the first cut

The A2A SDK is useful later, but for MVP it may hide too much and slow us down.

For the first pass:

- hand-roll the minimal request/response models
- keep the surface tiny
- adopt the SDK later only if it meaningfully reduces drift from the spec

### Keep task storage in memory first

Do not introduce a DB for MVP.

If the process restarts, tasks disappear. Fine. The point is proving agent-to-agent execution, not persistence.

### Keep artifacts to plain text first

Do not implement file/blob handling in MVP.

Once text round-trips work, file artifacts are easy.

### Do not auto-discover capabilities from tools yet

That sounds elegant and is exactly how you waste a day.

Use explicit config for advertised skills in the Agent Card.

### Reuse existing Hermes session machinery

If the plugin has to instantiate `AIAgent` manually, that is acceptable for the experiment.

If a cleaner internal helper exists or can be added, use it. The plugin should not duplicate session bootstrap logic in three places.

---

## Open questions to resolve during implementation

1. **Can a plugin expose `/.well-known/agent.json` directly?**
   If not, do we accept a temporary non-canonical path or add a small gateway hook for root-level well-known routes?

2. **What is the cleanest entry point for executing a single inbound request inside the current profile?**
   Need to inspect existing gateway/API request handlers.

3. **Should the sender be a generic external HTTP client first, or another Hermes profile immediately?**
   Recommendation: validate with curl first, then Hermes↔Hermes.

4. **Do we want a companion sender tool in Hermes later?**
   Probably yes, but not part of this plugin MVP. For now, curl/manual POST is enough.

5. **Should task ids be spec-shaped or Hermes-local?**
   Hermes-local opaque ids are fine for MVP.

---

## Manual acceptance criteria

The MVP is good enough if all of the following are true:

1. A Hermes profile with the plugin enabled serves an Agent Card.
2. A local POST to the plugin's RPC endpoint with `tasks/send` returns a valid JSON-RPC success or error payload.
3. A valid `tasks/send` request causes the receiving profile to execute a real Hermes run in its own context.
4. The response includes a stable task id and a final `completed` or `failed` status.
5. Two Hermes profiles can be run on different ports, and one can be manually queried as the other's A2A peer.
6. The feature is disabled by default and local-only by default.

If we hit those six, the experiment is successful.

---

## Immediate follow-up after MVP succeeds

In priority order:

1. **Hermes sender tool** — add an `a2a_send` tool or equivalent helper so one Hermes agent can call another without curl.
2. **`input-required`** — enable real multi-turn back-and-forth between agents.
3. **SSE streaming** — for long-running tasks.
4. **Agent Card canonical route support** — if plugin mounting blocked it in MVP.
5. **Kanban bridge** — allow A2A task completion to materialize/advance Kanban tasks.
6. **Telegram/group routing integration** — optional human-visible operations layer.
7. **External interoperability check** — validate against a non-Hermes A2A client/server.

---

## Candidate files likely to inspect while implementing

- `hermes_cli/plugins.py`
- `plugins/kanban/dashboard/plugin_api.py`
- `gateway/run.py`
- `gateway/session.py`
- `run_agent.py`
- API server request handling files under `gateway/` or adjacent runtime modules
- plugin examples with HTTP routes

---

## Suggested branch and commit protocol

Use a clean feature branch in the Hermes Agent repo.

Example:

```bash
cd ~/.hermes/hermes-agent
git checkout -b feat/a2a-plugin-mvp
```

Commit after each task. Keep the plugin experimental and self-contained until the first manual proof works.

---

## Final note

This is the right experiment.

Hermes already has profiles, gateway transport, plugin architecture, and the operational need. What it lacks is the protocol layer that lets one profile ask another profile a question without going through the human every time.

That is exactly what A2A is for.
