# Premium Assistant Platform v0 — Architecture

> Architecture draft for issue #32820.
> 
> Goal: define a premium assistant platform that starts as a single-user product, uses Hermes as the orchestrator/harness, uses LiveKit for realtime transport, and stores memory in Postgres first without overengineering the memory layer.

## 1. Architecture goals

- Make the *core loop* explicit: capture input, assemble context, run Hermes orchestration, execute tools, return response, persist durable state.
- Keep product/platform boundaries clean so the first implementation can ship as a premium single-user experience.
- Make multi-tenant support a data-model property from day one, not a separate architecture rewrite later.
- Prefer simple, auditable storage and retrieval over a complex memory graph.

## 2. Component boundaries

### 2.1 Client app
Owns:
- text chat UI
- voice entry/exit UI
- session history UI
- settings / preferences UI
- feedback controls

Does *not* own:
- memory storage
- orchestration logic
- tool execution
- entitlement / tenant policy

### 2.2 Control plane / platform API
Owns:
- auth and identity bootstrap
- tenant/user/session creation
- entitlement and feature flags
- LiveKit token minting and room/session bootstrap
- API endpoints for conversation/session history
- audit/event ingestion

Does *not* own:
- conversation reasoning
- memory retrieval policy
- STT/TTS implementation details

### 2.3 Hermes brain / orchestration layer
This is the existing Hermes harness/orchestrator layer, centered on the conversation loop.

Owns:
- prompt/context assembly
- conversation turn execution
- tool dispatch
- provider/model routing
- response shaping
- policy enforcement around tools and memory usage
- emitting memory candidates and summaries for async persistence

Suggested reuse points in repo:
- `run_agent.py`
- `model_tools.py`
- `toolsets.py`
- existing gateway/tool adapters where appropriate

### 2.4 Memory service
A thin service or module over Postgres. Keep this intentionally small in v0.

Owns:
- durable memory writes
- retrieval of relevant memory items
- session summaries
- preference/fact storage
- auditability of memory provenance

Does *not* own:
- a complex graph model
- mandatory embeddings-first retrieval
- a separate vector database in v0

### 2.5 Realtime / voice layer
Use LiveKit as the transport layer.

Owns:
- live audio room/session transport
- presence
- audio track lifecycle
- room metadata / session state
- STT/TTS integration boundaries

Does *not* own:
- assistant reasoning
- memory writes
- tool execution

### 2.6 Async workers
Owns:
- post-turn memory extraction
- session summaries
- cleanup / compaction
- analytics/event processing
- optional embeddings later

### 2.7 Storage layer
Owns:
- Postgres persistence
- transaction boundaries
- query indexes
- row-level audit fields

## 3. Request lifecycle

### 3.1 Text turn flow
1. Client authenticates through the control plane.
2. Control plane resolves tenant/user/session, then returns session bootstrap data.
3. Client sends the message to the control plane or orchestration endpoint.
4. Hermes loads the current session state, recent transcript, and relevant memory items from Postgres.
5. Hermes assembles context:
   - system instructions
   - user message
   - recent conversation window
   - curated memory items
   - current tool availability / policies
6. Hermes executes the turn and may call tools.
7. Tool results are folded back into the turn.
8. Hermes returns the answer to the client.
9. The turn transcript, tool events, and memory candidates are written to Postgres.
10. Async workers summarize/extract durable memory from the turn.

### 3.2 Voice turn flow
1. Client joins a LiveKit room using a token minted by the control plane.
2. LiveKit carries the audio stream.
3. STT converts audio to text at the voice boundary.
4. The resulting transcript enters the same Hermes turn pipeline as text.
5. Hermes responds with text and/or TTS-ready output.
6. TTS renders the assistant response back into the LiveKit session.
7. Session transcript, audio metadata, and memory candidates are persisted.

### 3.3 Failure / retry behavior
- If LiveKit fails, text chat must still work.
- If memory retrieval fails, the assistant should degrade to recent conversation + explicit user input.
- If async memory extraction fails, the user-facing turn must still succeed.
- Tool failures must be explicit in the response and logged as events.

## 4. Storage decisions

### 4.1 Postgres is the system of record
Use Postgres first for:
- users
- tenants
- sessions
- turns/messages
- tool executions
- memory items
- summaries
- audit/events

Why:
- simple operational model
- easy to inspect and debug
- strong consistency for v0
- supports single-user-first and multi-tenant later without schema replacement

### 4.2 Memory strategy in v0
Store *small, explicit, auditable* memory items.

Recommended memory retrieval approach for v0:
- recent-turn window
- curated memory item lookup by user/session/importance/tags
- optional text search over memory content and summaries
- embeddings can be added later behind a feature flag

Avoid in v0:
- a full memory graph
- mandatory vector search
- complex agentic memory writing policies

### 4.3 Suggested primitive data model
Keep the primitives stable and tenant-scoped from day one.

- `tenant`
  - `id`
  - `name`
  - `plan`
  - `created_at`

- `user`
  - `id`
  - `tenant_id`
  - `email` or external identity reference
  - `display_name`
  - `created_at`

- `session`
  - `id`
  - `tenant_id`
  - `user_id`
  - `mode` (`text` | `voice`)
  - `status`
  - `started_at`
  - `ended_at`

- `message` / `turn`
  - `id`
  - `session_id`
  - `role`
  - `content`
  - `created_at`

- `tool_execution`
  - `id`
  - `session_id`
  - `turn_id`
  - `tool_name`
  - `input_json`
  - `output_json`
  - `status`
  - `created_at`

- `memory_item`
  - `id`
  - `tenant_id`
  - `user_id`
  - `session_id` nullable
  - `kind` (`preference`, `fact`, `project`, `relationship`, `task`, `summary`)
  - `content`
  - `source_turn_id`
  - `importance`
  - `confidence`
  - `tags_json`
  - `created_at`
  - `updated_at`
  - `deleted_at` nullable

- `event`
  - `id`
  - `tenant_id`
  - `user_id`
  - `session_id`
  - `type`
  - `payload_json`
  - `created_at`

### 4.4 Multi-tenant readiness
Even if only one customer exists in v0:
- every durable row carries `tenant_id`
- every query is tenant-scoped
- no cross-tenant sharing assumptions
- no schema changes required to add more tenants later

## 5. MVP-first constraints

### Build now
- text chat
- voice session entry/exit
- durable memory of useful facts/preferences
- a small set of high-value tool actions
- session history and audit trail
- explicit tenant/user/session scoping

### Do not build yet
- marketplace / plugin ecosystem
- enterprise admin suite
- complex team collaboration
- multi-model routing framework
- full memory graph
- embeddings as the primary retrieval layer
- avatar / branding polish before the core loop works

### Quality bar for v0
- the assistant can remember something useful and use it later
- voice feels live enough to be premium
- tool execution is visible and auditable
- failures degrade gracefully instead of breaking the session

## 6. Implementation order

1. **Define contracts**
   - user/session/tenant IDs
   - turn payloads
   - memory item schema
   - tool execution schema
   - event schema

2. **Implement Postgres persistence**
   - create tables and indexes
   - add tenant scoping
   - add basic audit fields

3. **Wire the text request path**
   - bootstrap session
   - load recent context + memory
   - run Hermes orchestration
   - persist turn and tool events

4. **Add memory extraction and retrieval**
   - write memory candidates after turns
   - retrieve curated memory on new turns
   - keep the policy simple and inspectable

5. **Add LiveKit voice transport**
   - mint tokens
   - join rooms
   - bridge STT/TTS into the same Hermes turn pipeline

6. **Add async workers**
   - summarization
   - cleanup/compaction
   - optional embeddings later

7. **Harden observability and failure handling**
   - turn tracing
   - tool audit logging
   - retries / fallbacks
   - latency measurement

## 7. Non-goals for v0
- no complex memory graph
- no overbuilt multi-tenant admin surface
- no plugin ecosystem
- no custom RTC stack
- no vector-first storage strategy
- no platform abstractions that delay the first usable premium experience

## 8. Architecture doc acceptance criteria

This doc is ready to hand to implementation when it answers:
- what owns each boundary
- how a text or voice request flows end-to-end
- what lives in Postgres
- what remains Hermes responsibility
- what is explicitly not built in v0
- which item to implement first
