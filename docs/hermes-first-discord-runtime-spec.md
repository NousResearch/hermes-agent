# Hermes-First Discord Runtime Specification

Status: Draft
Owner: Daniel
Last Updated: 2026-03-09

## 1. Purpose

This specification defines a Hermes-first Discord runtime that replaces Hermes's current messaging gateway for Discord while preserving Hermes as the agent backend.

The intended outcome is a Discord-native runtime with:

- first-class server channels, threads, and forum-post threads
- surface-scoped sessions with explicit lineage
- real subagent behavior and named agent identities
- middleware-owned authoritative state and observability
- a narrow sidecar API that treats Hermes as a headless backend

This design must support rich Discord interaction patterns without inheriting Hermes gateway coupling or Hermes gateway token-cost behavior.

## 2. Goals

- Use Hermes `AIAgent` as the backend reasoning and tool engine without running the Hermes messaging gateway.
- Make the Discord runtime the only owner of Discord transport, routing, interaction, delivery, and authoritative surface state.
- Treat channels, threads, and forum-post threads as first-class conversation surfaces.
- Key sessions to surfaces by default and attach explicit parent-child lineage when work branches into new surfaces.
- Restrict cross-surface awareness to bounded summaries and server-state metadata rather than raw transcript sharing.
- Support real subagents in the server model and allow identity mode to be configurable per agent or spawn.
- Expose a Hermes sidecar Turn API plus debug and trace endpoints.
- Make context and runtime policy human-tunable and middleware-owned.
- Emit full structured traces for middleware, sidecar, and turn execution.

## 3. Non-Goals

- Reusing Hermes Discord gateway code at runtime.
- Supporting Discord DMs as a first-class surface in the first production-worthy version.
- Achieving transport-agnostic purity in v1. The sidecar contract may be Discord-aware.
- Replacing Hermes internal memory in v1. Hermes memory may be used temporarily as a convenience.
- Supporting all Discord surfaces from day one, such as group DMs and voice-first channels, as primary workflow surfaces.
- Preserving Hermes's current `send_message` and gateway-shaped Discord behavior inside the runtime.

## 4. Assumptions and Boundaries

### 4.1 Assumptions

- The initial deployment target is a single Discord server used by a personal or small trusted group.
- The server is treated as one network of agents rather than a collection of unrelated chats.
- The first visible topology will start with one named agent, but the model must support multiple named agents.
- The middleware will be implemented in TypeScript/Node.
- Hermes will run as a Python sidecar process behind a narrow API.
- `discrawl` may be used as an operator tool in v1 and must be easy to add as an on-demand or background source later.

### 4.2 Owned Scope

The middleware owns:

- Discord gateway connection and REST delivery
- surface discovery and canonical route resolution
- authoritative surface sessions and lineage state
- interaction handling for slash commands, buttons, selects, and modals
- Discord-native reply delivery, retries, chunking, edits, typing, and rate-limit handling
- context budgeting, lineage summaries, and server-state summaries before a turn reaches Hermes
- observability, traces, and operator tooling for Discord runtime behavior

The Hermes sidecar owns:

- invocation of `AIAgent` from [`run_agent.py`](/home/daniel/wsl_code/Hermes-agent/run_agent.py)
- prompt construction for the backend turn
- tool execution
- provider and model invocation
- temporary Hermes memory usage
- Hermes-internal subagent or delegation mechanics, if used

### 4.3 Out of Scope

- Hermes gateway modules such as [`gateway/run.py`](/home/daniel/wsl_code/Hermes-agent/gateway/run.py), [`gateway/platforms/discord.py`](/home/daniel/wsl_code/Hermes-agent/gateway/platforms/discord.py), and [`gateway/session.py`](/home/daniel/wsl_code/Hermes-agent/gateway/session.py) are not part of the runtime.
- Hermes channel directory and `send_message` transport behavior are not authoritative for Discord runtime behavior.
- OpenClaw compatibility is not a v1 implementation requirement, though the architecture should not preclude it.

## 5. System Model

### 5.1 Components and Actors

- `Discord Runtime`: The Node process that receives Discord events, resolves surfaces, manages authoritative state, and delivers responses.
- `Surface Router`: The middleware component that maps Discord objects to canonical surface identities and session keys.
- `State Store`: The authoritative store for surfaces, lineage, summaries, agent bindings, and turn metadata.
- `Context Budgeter`: The middleware component that decides how much local, parent, and server-summary context is passed to Hermes.
- `Lineage Manager`: The middleware component that creates parent-child links when a new surface is created from an existing workflow.
- `Interaction Renderer`: The middleware component that validates and renders declarative reply ops into Discord-native UI and delivery actions.
- `Hermes Sidecar`: A Python service that wraps `AIAgent` and exposes a narrow Turn API plus debug endpoints.
- `Named Agent`: A durable identity in the server model. A named agent may own one or more surfaces over time.
- `Subagent`: A child agent execution context that is real in the server model and may be visible as a distinct actor or may share identity, depending on policy.
- `Operator`: The human owner or maintainer who configures policies, traces, and observability.

### 5.2 Core Entities and Invariants

- `Surface`: A Discord channel, thread, or forum-post thread. A surface must have one canonical `surface_id` and one canonical `surface_kind`.
- `Session`: The authoritative conversation record for a surface. A session must be surface-scoped by default.
- `Lineage Edge`: A parent-child relationship between sessions. A lineage edge must include a bounded handoff summary.
- `Agent Binding`: A mapping between a named agent and a surface session. A surface must have one active primary agent binding at a time.
- `Turn`: One middleware-to-sidecar backend invocation. A turn must have a unique `turn_id`, a `session_key`, and an `agent_id`.
- `Reply Op`: A validated declarative action returned by the Hermes sidecar. Reply ops must be middleware-renderable and must not require raw Discord payload passthrough in v1.

Invariants:

- One surface maps to one authoritative session key at any point in time.
- Cross-surface awareness must be summary-based unless an explicit operator override is configured.
- Child sessions must not inherit full parent transcripts by default.
- Middleware state is authoritative for surface lineage and routing; Hermes is not authoritative for Discord session topology.
- Hermes must not send directly to Discord in the v1 runtime path.

## 6. Lifecycle and State Model

### 6.1 States

- `discovered`: A surface exists in Discord but has not yet been bound to a runtime session.
- `active`: A surface has an authoritative session and may receive turns.
- `branched`: A new child surface has been linked from a parent surface with an explicit lineage edge.
- `idle`: A surface session exists but has no active turn.
- `running`: A turn is in progress for the surface's bound agent.
- `handoff_pending`: A child surface or agent handoff has been created and the summary is being finalized.
- `closed`: The surface session is no longer eligible for new turns without reopening or rebinding.

### 6.2 Transitions

- `discovered -> active`: Triggered when a surface first receives a relevant Discord event.
- `active -> running`: Triggered when middleware dispatches a turn to the Hermes sidecar.
- `running -> idle`: Triggered when the sidecar returns a terminal `TurnResponse`.
- `active -> branched`: Triggered when Discord creates a new child surface from the current surface and middleware records a lineage edge.
- `branched -> handoff_pending`: Triggered when the child surface requires a bounded summary from parent state.
- `handoff_pending -> active`: Triggered when the child summary is durable and the child surface is ready for turns.
- `idle -> closed`: Triggered by operator action, archival policy, or surface deletion policy.

### 6.3 Timeout, Retry, and Cancellation

- Discord delivery retries must be owned by middleware and must follow bounded retry policy with rate-limit awareness.
- A running turn may be cancelled by operator action or newer surface policy, but cancellation semantics must be explicit in traces.
- Sidecar request timeouts must be configurable and traced.
- A failed handoff summary must not corrupt parent or child session state; the child may enter a degraded `active` state with reduced context.

## 7. Interfaces and Contracts

### 7.1 Discord Runtime Ingress

Purpose: Normalize Discord events into surface-scoped runtime events.

Caller / Callee: Discord runtime gateway listener -> Surface Router

Input:

- `discord_event`: Required raw Discord event.
- `received_at`: Required timestamp.
- `server_id`: Required Discord guild id.

Output:

- `NormalizedDiscordEvent`: Canonical event with resolved surface, actor, message, interaction, and reference fields.

Behavior:

- The runtime must resolve one canonical `surface_kind` from `channel`, `thread`, or `forum_post`.
- The runtime must resolve `parent_surface_id` for thread and forum-post thread events.
- The runtime must extract a bounded reply excerpt when the event references another message.
- The runtime may enrich the event with operator-visible server-state metadata.

Errors:

- `UnsupportedSurface`: Event is ignored with trace and operator log.
- `InvalidReference`: Event proceeds without reply excerpt and records degraded trace metadata.

Observability:

- Event normalization trace
- surface resolution metrics
- drop reason metrics

### 7.2 Hermes Turn API

Purpose: Invoke Hermes as a headless backend for one turn.

Caller / Callee: Discord runtime -> Hermes sidecar

Input:

- `TurnRequest`: Required request object containing agent, session, lineage, tunables, summaries, and the normalized user or interaction input.

Output:

- `TurnResponse`: Required response object containing reply ops, turn usage, trace artifact handles, and optional spawn metadata.

Behavior:

- The sidecar must wrap `AIAgent` and must not run Hermes gateway modules.
- The sidecar must treat middleware state as authoritative and must not invent Discord routing.
- The sidecar must return reply ops only; the caller is responsible for Discord rendering.
- The sidecar may return debug or trace artifact references in hybrid mode.
- The sidecar should run with a reduced Hermes toolset that excludes gateway-shaped Discord transport behavior such as direct Discord `send_message` flows in the hot path.

Errors:

- `TurnValidationError`: Caller input invalid; caller may correct and retry.
- `TurnTimeout`: Sidecar exceeded configured limit; caller may retry according to policy.
- `BackendExecutionError`: Hermes backend failed after validation; caller may surface degraded behavior.

Observability:

- one trace span per turn
- request and response size metrics
- token usage metrics
- per-tool timing if available

### 7.3 Reply Ops Contract

Purpose: Express middleware-renderable Discord-aware actions without raw Discord payload passthrough.

Caller / Callee: Hermes sidecar -> Discord runtime

Input:

- `reply_ops`: Ordered list of validated reply operations.

Output:

- Discord-native sends, edits, interaction responses, thread actions, or server-summary mutations.

Behavior:

- Ops must be declarative and deterministic.
- Ops must be validated by the middleware before rendering.
- Middleware must reject invalid or unsupported ops and record a visible trace artifact.
- Middleware may degrade unsupported ops into text notices according to policy.

Errors:

- `UnsupportedOp`: Middleware cannot render the op.
- `InvalidOpPayload`: Required field missing or invalid.

Observability:

- op validation failures
- op render latency
- per-op delivery success rate

### 7.4 Authoritative State Store Contract

Purpose: Persist surface sessions, lineage, summaries, agent bindings, and operator-visible turn state.

Caller / Callee: Discord runtime components -> State Store

Input:

- `surface session upsert`
- `lineage edge upsert`
- `summary write`
- `agent binding write`
- `turn event append`

Output:

- durable record ids or canonical keys

Behavior:

- The store must be authoritative for surface-session routing.
- The store must support parent-child lineage lookup by `session_key` and `surface_id`.
- The store should support summary versioning to avoid stale handoff confusion.

Errors:

- `Conflict`: Caller retries or resolves operator-visible conflict.
- `Unavailable`: Runtime may enter degraded in-memory mode if configured.

Observability:

- store latency
- conflict rates
- degraded mode indicator

### 7.5 Sidecar Admin and Trace API

Purpose: Provide health, debug, and trace access without exposing Hermes gateway concepts.

Caller / Callee: Operator tooling or middleware debug clients -> Hermes sidecar

Endpoints:

- `GET /v1/health`: Returns sidecar liveness and degraded-mode state.
- `GET /v1/traces/{turn_id}`: Returns trace metadata and artifact references for a completed or failed turn.
- `GET /v1/turns/{turn_id}`: Returns minimal turn status, usage, and warning metadata.

Behavior:

- Admin endpoints must be read-only in v1.
- Admin endpoints must not mutate authoritative middleware state.
- Trace endpoints may return artifact handles rather than full payloads when artifacts are large.

Observability:

- endpoint latency
- artifact lookup failures
- degraded-mode markers when trace backends are unavailable

## 8. Operational Model

### 8.1 Runtime Topology

- One Node process runs the Discord runtime.
- One Python sidecar process runs the Hermes bridge service.
- The sidecar is addressed over a local RPC or HTTP boundary.
- The state store is owned by middleware and must be durable.
- Observability must span both processes using shared trace ids.

### 8.2 Capacity and Performance Constraints

- All context budgets must be human-tunable per agent, surface kind, or policy profile.
- A turn must carry only bounded local context, bounded lineage summaries, and bounded server summaries.
- Middleware must prefer summary-based awareness over transcript sharing.
- Backpressure must be visible in traces and operator logs.

### 8.3 Configuration and Secret Handling

- Discord transport config must live in middleware configuration.
- Context, lineage, and routing policy must live primarily in middleware configuration.
- Hermes-side provider and tool configuration may remain Hermes-owned where not transport-specific.
- Secrets for Discord and Hermes providers must remain process-local and must not be mirrored into trace payloads.

## 9. Failure Model and Recovery

### 9.1 Failure Classes

- `Discord delivery failure`: Middleware retries, degrades, or surfaces operator-visible failure.
- `Sidecar unavailable`: Middleware may queue, fail fast, or enter maintenance mode according to policy.
- `Summary generation failure`: Child surface runs with reduced inherited context and an explicit degraded trace marker.
- `State store unavailable`: Middleware may enter degraded in-memory mode if configured, otherwise reject new turns.
- `Invalid reply op`: Middleware blocks render and records validation failure.

### 9.2 Recovery

- Delivery retries must be bounded and rate-limit aware.
- Sidecar failures should preserve enough turn metadata to replay manually.
- Summary failures should not block the entire server runtime.
- State-store recovery may replay pending turn artifacts if the implementation chooses to persist them.

### 9.3 Degraded Modes

- Render text-only when rich Discord ops fail validation.
- Use reduced local-only context if lineage summaries are unavailable.
- Disable subagent spawning when lineage state is unavailable.

## 10. Security, Safety, and Privacy

- The middleware is the trust boundary for Discord transport.
- Middleware must validate all sidecar reply ops before rendering.
- Raw Discord payload passthrough is out of scope in v1.
- Operator tooling such as `discrawl` must not silently expand live context without explicit policy.
- Traces must avoid secret and token leakage.

## 11. Delivery and Migration

### 11.1 Rollout Plan

- Phase 1: Implement middleware ingress, state store, route resolution, and Hermes Turn API with text-only reply ops.
- Phase 2: Add thread and forum lineage, bounded summaries, and human-tunable context policy.
- Phase 3: Add declarative rich reply ops: buttons, selects, modals, thread actions, and summary mutations.
- Phase 4: Add real subagent flow, named identity modes, and child-surface spawning with lineage summaries.
- Phase 5: Add `discrawl` operator workflow and optional on-demand integration path.
- Phase 6: Expand or replace Hermes memory behavior with custom middleware-owned memory.

### 11.2 Migration or Backfill

- No runtime dependency on Hermes gateway transcripts is assumed.
- Existing Hermes memory may be reused temporarily, but authoritative surface state must be middleware-owned from the start.
- Any future decoupling from Hermes should preserve the Turn API and Reply Ops contract.

## 12. Validation and Acceptance

### 12.1 Conformance Checks

- The runtime must function without importing or running Hermes gateway modules in the hot path.
- The sidecar must wrap `AIAgent` directly and return reply ops rather than Discord transport actions.
- New thread or forum surfaces must create explicit lineage edges and bounded handoff summaries.
- Cross-surface context must remain summary-based by default.

### 12.2 Test Matrix

- `channel -> thread branch`: Child thread gets new surface-scoped session plus lineage summary.
- `forum post creation`: Forum-post thread is resolved as first-class surface.
- `reply in thread`: Reply excerpt is bounded and delivered in TurnRequest.
- `rich op render`: Valid button or modal op is rendered successfully.
- `invalid rich op`: Middleware rejects op and emits degraded trace.
- `subagent spawn`: Real child context is created and bound correctly.
- `state-store outage`: Runtime enters configured degraded behavior.
- `sidecar timeout`: Turn fails with explicit trace and retry policy.

## 13. Initial v1 Build Sequence

1. Implement the TypeScript contract package and lock the Turn API and Reply Ops schema.
2. Build the Node Discord runtime with normalized event ingress and canonical surface routing.
3. Build the authoritative middleware store for surfaces, lineage, summaries, and agent bindings.
4. Build the Hermes sidecar wrapper around `AIAgent` only.
5. Implement text-only turn execution end to end.
6. Add explicit lineage summaries and human-tunable context policy.
7. Add declarative Discord-aware reply ops and middleware validation.
8. Add named agent bindings and real subagent spawning.
9. Add full traces and external observability integration.
10. Add `discrawl` operator workflow and future on-demand hook points.

## 14. Open Questions

- Whether Hermes temporary memory should be read-only, write-through, or unrestricted during the transition period.
- Whether the first subagent identity mode default should be distinct actor or shared persona.
- Which external observability stack should be the first-class integration target.
