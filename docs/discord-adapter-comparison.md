# Discord Adapter Comparison: OpenClaw vs Hermes

Related document: [Discord Usage Audit](./discord-usage-audit.md)

## 1. Purpose

This document audits OpenClaw's Discord adapter, compares it against Hermes Agent's current Discord implementation, and extracts design requirements for a reusable agent-agnostic Discord adapter.

The target outcome is not just "make Hermes better." The target outcome is a transport and interaction layer that can sit in front of Hermes, OpenClaw, or any gateway-backed agent without re-inventing Discord semantics each time.

## 2. Scope and Method

This is a code-first static audit. I did not run a live OpenClaw gateway against Discord.

OpenClaw sources reviewed:

- `extensions/discord/index.ts`
- `extensions/discord/src/channel.ts`
- `extensions/discord/src/subagent-hooks.ts`
- `src/discord/monitor/provider.ts`
- `src/discord/monitor/listeners.ts`
- `src/discord/monitor/message-handler.preflight.ts`
- `src/discord/monitor/message-handler.process.ts`
- `src/discord/monitor/inbound-worker.ts`
- `src/discord/monitor/native-command.ts`
- `src/discord/monitor/native-command-context.ts`
- `src/discord/monitor/threading.ts`
- `src/discord/monitor/reply-delivery.ts`
- `src/discord/session-key-normalization.ts`
- recent Discord commits on 2026-03-07 and 2026-03-08

Hermes sources reviewed:

- `gateway/platforms/discord.py`
- `gateway/session.py`
- `gateway/run.py`
- `gateway/channel_directory.py`
- `tools/send_message_tool.py`
- `toolsets.py`
- `run_agent.py`
- prior findings in [Discord Usage Audit](./discord-usage-audit.md)

## 3. Executive Summary

OpenClaw's Discord implementation is materially more mature than Hermes's current adapter.

The main difference is not just feature count. It is boundary discipline:

- OpenClaw treats Discord as a first-class transport with its own routing, threading, command, interaction, and delivery semantics.
- Hermes treats Discord mostly as a thin message ingress/egress shim around a generic gateway session model.

That difference has downstream effects:

- OpenClaw preserves far more Discord-native behavior: thread/forum context, reply context, native commands, components, modals, reactions, preview streaming, thread-bound subagents, voice, and live allowlist resolution.
- Hermes is simpler, but it leaks Discord semantics into generic session code and leaks generic session costs back into Discord usage.

For token usage specifically:

- Hermes's main cost problem is still transcript replay plus the very large default Discord toolset. See [Discord Usage Audit](./discord-usage-audit.md).
- OpenClaw's adapter layer is much more disciplined. It adds bounded, structured context such as reply excerpts, a bounded pending history window, thread-starter text, and forum/thread metadata. It does not appear to shovel raw Discord logs into every request.

For your longer-term goal, OpenClaw is the better reference model for transport design, while Hermes is the clearer example of what happens when the adapter boundary is too thin.

## 4. OpenClaw Adapter Audit

### 4.1 System Model

OpenClaw's Discord support is split across a plugin registration layer and a transport/runtime layer.

- `extensions/discord/index.ts` registers a Discord channel plugin plus Discord-specific subagent hooks.
- `extensions/discord/src/channel.ts` defines the Discord channel capability surface, config schema wiring, security hooks, live directory support, threading behavior, and outbound action hooks.
- `src/discord/monitor/provider.ts` owns the live Discord client, gateway plugin, native command deployment, listener registration, voice manager, thread binding manager, and lifecycle handling.

This is already closer to the architecture you want for a reusable adapter:

- platform transport is isolated
- route/session semantics are explicit
- richer Discord features are adapter-owned rather than agent-owned

### 4.2 Inbound Lifecycle

OpenClaw's inbound path is deliberate and staged:

```mermaid
flowchart LR
    A["Discord Gateway Event"] --> B["Non-blocking Listener"]
    B --> C["Inbound Debouncer"]
    C --> D["Preflight"]
    D --> E["Route + Session Resolution"]
    E --> F["Keyed Inbound Worker"]
    F --> G["Context Builder"]
    G --> H["Agent Dispatch"]
    H --> I["Reply Delivery"]
```

Important properties:

- The message listener is now intentionally non-blocking. Commit `f51cac277` moved per-session ordering out of the raw listener and into the inbound worker queue.
- Debouncing is channel-plus-author keyed. Rapid text bursts can be merged before agent dispatch.
- Preflight is extensive. It handles mention gating, DM policy, pairing flows, allowlists, bot filtering, thread inheritance, PluralKit handling, reply metadata, forum/thread parent resolution, and ACP/thread bindings.
- Actual processing is queued by resolved session key via `createDiscordInboundWorker()` and `KeyedAsyncQueue`, which preserves per-session order without serializing the whole Discord client.

This is a strong pattern. Hermes does not have an equivalent keyed inbound worker for Discord.

### 4.3 Route and Session Handling

This is the strongest part of the OpenClaw adapter.

The adapter resolves:

- DM vs group DM vs channel vs thread
- direct user id vs conversation id
- parent conversation id for threads and forum posts
- configured route bindings
- live thread bindings
- effective session key after bindings

Recent commits show a lot of attention here:

- `bc91ae9ca` fixed native-command session key preservation
- `74e3c071b` extracted Discord-specific session key normalization
- `c1d07b09c` and `6016e22cc` extracted and centralized route resolution
- `189cd9937` tightened outbound session targeting

That commit sequence is significant. It means OpenClaw recently found and fixed exactly the kind of subtle Discord bugs that show up when slash commands, threads, and bound sessions are not resolved through one canonical path.

Hermes does not have this layer. Its session key builder is generic and simple:

- DMs collapse to `agent:main:discord:dm`
- non-DMs become `agent:main:discord:{chat_type}:{chat_id}`

That simplicity is useful, but it is too weak for a best-in-class Discord adapter.

### 4.4 Context Management

OpenClaw injects structured, bounded Discord context rather than dumping raw chat logs.

What the adapter explicitly adds:

- current inbound envelope
- bounded pending channel history from `guildHistories`
- referenced-message excerpt via `ReplyToId`, `ReplyToBody`, `ReplyToSender`
- thread starter text
- forum parent context
- untrusted channel metadata such as channel topic
- group/system prompt overrides from channel config
- thread/session metadata
- media payload descriptors

Important nuance:

- the pending channel history is not a permanent full transcript
- it is cleared after a final response
- it is bounded by `historyLimit`
- it exists to recover missed group context between the bot's replies, not to persist entire Discord backlogs

That is a much better adapter-level cost shape than Hermes.

Hermes, by contrast, does not inject replied-to message content or thread starter content, and its real cost driver is reloading and replaying its entire persisted session transcript on every new Discord turn.

### 4.5 Discord Feature Coverage

OpenClaw supports, or has explicit plumbing for, most of the Discord surface that matters to agent UX:

- DMs
- guild text channels
- threads
- forum/media parent awareness
- native slash commands with argument schemas and autocomplete
- model-picker interactions
- buttons, selects, and modals
- reaction listeners and reaction-based status/acks
- typing heartbeats
- edit-based preview streaming
- auto-thread creation
- thread-bound subagent sessions
- webhook/persona delivery for bound threads
- voice ingress and TTS/voice reply flows
- live directory lookups and allowlist resolution
- Discord-specific exec approval UI

Hermes supports only a narrower slice:

- DMs
- channels
- threads as message sources
- basic slash commands
- typing indicator
- native media attachments
- exec approval buttons

What Hermes does not currently do well:

- reply-aware context construction
- canonical thread/forum parent handling
- auto-threading
- native interactive components beyond approvals
- native-command session routing in threads
- live directory discovery beyond visible text channels and seen DMs
- reaction, presence, thread update, or voice handling

### 4.6 Recent Updates That Matter

The latest OpenClaw Discord work is meaningful, not cosmetic.

- `f51cac277` made the message listener non-blocking and moved ordering concerns into the inbound worker.
- `bc91ae9ca`, `74e3c071b`, `c1d07b09c`, `6016e22cc`, and `8f719e541` tightened native-command routing and session-key behavior.
- `547436bca` extracted shared inbound context helpers, reducing drift between message, native command, and component paths.
- `d902bae55` validated Discord agent-component config, which matters because components are now part of the contract surface.

Taken together, these commits suggest OpenClaw is actively converging on a cleaner adapter boundary.

### 4.7 Risks and Gaps in OpenClaw

OpenClaw is not "done." The main issues are different from Hermes's:

- The implementation is large and operationally complex.
- It is tightly coupled to OpenClaw's routing, session, ACP, and reply-dispatch stack.
- The transient pending-history model is in-memory, so it does not survive restart.
- Auto-threading is best-effort and can fail quietly.
- Live directory discovery and resolution still depend on Discord REST and configured tokens.
- The adapter is reusable in concept, but not yet in packaging. Its Discord logic is still intertwined with OpenClaw runtime contracts.

These are acceptable tradeoffs for a mature first-party adapter, but they are exactly why a standalone agent-agnostic layer would still be valuable.

## 5. Comparison With Hermes

| Dimension | OpenClaw | Hermes |
|---|---|---|
| Adapter shape | Large, explicit transport/runtime subsystem | Thin platform shim around generic gateway |
| Session routing | Canonical route resolution, bound-session overrides, thread parent support | Simple session keys built from `chat_type` and `chat_id` |
| Listener model | Non-blocking listener plus keyed inbound worker | Direct event handler into gateway message pipeline |
| Thread handling | First-class thread and forum-parent semantics | Threads are detected, but parent/forum semantics are mostly absent |
| Reply context | Replied-to message is extracted and injected | Only `reply_to_message_id` is carried |
| Native commands | Rich slash-command path with args, autocomplete, model picker, component fallbacks | Basic slash commands that map to plain text commands |
| Components/modals | Full buttons/selects/modals path | Exec approval buttons only |
| Delivery | Retry-aware, chunk-aware, preview streaming, reply reference planning, webhook personas | Basic chunked sends and simple replies |
| Discord-specific auth | DM policy, allowlists, access groups, pairing, per-command gating | Global allowlists and mention gating |
| Subagent support | Thread-bound subagent sessions | No Discord-specific subagent thread model |
| Voice | Integrated | None |
| Directory | Config plus live discovery and allowlist resolution | Cached text-channel listing plus session-derived DMs |
| Tests | Large Discord-specific test surface: about 78 Discord test files in the cloned repo | No dedicated Discord tests under `tests/` |
| Token cost shape | Adapter adds bounded, structured context | Gateway replays full persisted transcript; very high fixed overhead from tools/context |

## 6. What Hermes Should Steal From OpenClaw

This should be viewed as a priority list, not a request to port OpenClaw wholesale.

### 6.1 Introduce a Real Discord Route Layer

Hermes needs one canonical resolver for:

- DM vs channel vs thread vs forum post
- direct user id vs conversation id
- parent conversation id
- reply target
- delivery target
- session key

Do not let slash commands, message replies, thread messages, and outbound sends each invent their own routing rules.

### 6.2 Split Listener, Preflight, Queue, and Process

Hermes currently handles Discord messages too inline. It should adopt the same broad shape OpenClaw now uses:

- raw listener
- cheap bot/self filter
- preflight
- per-session keyed worker
- process stage
- delivery stage

That will make interrupts, retries, rate limits, and thread behavior much less brittle.

### 6.3 Add Structured Discord Context Instead of Leaning on Transcript Replay

Hermes should add a Discord ingress envelope that includes:

- current message text
- bounded pending group history since last bot reply
- replied-to message excerpt
- thread starter text
- forum parent metadata
- channel topic as untrusted context

That gives the model what Discord users actually expect without relying on huge session transcripts to reconstruct context.

### 6.4 Make Threads and Forum Posts First-Class

Hermes should treat these as distinct conversation surfaces:

- DM
- channel
- thread
- forum post thread

The adapter should know the parent channel and forum parent. The session layer should not have to guess later.

### 6.5 Unify Slash Commands With Normal Session Semantics

Hermes slash commands currently take a simpler path and degrade thread semantics.

OpenClaw's recent fixes are a warning: if native commands do not share the same route/session logic as normal messages, subtle bugs accumulate quickly.

### 6.6 Improve Outbound Delivery

Hermes should add:

- explicit reply reference policy
- retry and rate-limit handling
- message-edit preview streaming
- thread-aware reply planning
- optional auto-thread creation for busy public channels

This is where Discord starts feeling polished rather than merely functional.

### 6.7 Shrink the Default Discord Tool Surface

This comes from the prior Hermes audit, but it matters even more after the OpenClaw comparison.

Hermes's default Discord toolset is too broad for a chat transport. Even a world-class adapter will still be expensive if every Discord turn drags the full core tool schema set behind it.

The highest-value Hermes changes remain:

- smaller Discord-specific toolset
- shorter Discord session reset/compress thresholds
- optional `skip_context_files=True` for messaging sessions
- summarized tool-output persistence instead of replaying raw tool results forever

### 6.8 Build Tests Before Adding More Surface

OpenClaw's Discord maturity is not just architectural. It is backed by a lot of Discord-specific tests.

Hermes should not add auto-threading, richer slash commands, or thread/forum routing without dedicated tests for:

- DM, channel, thread, and forum-post routing
- replies in and out of threads
- slash commands in threads
- session-key normalization
- reply delivery policy
- rate-limit and retry handling

## 7. Design Requirements for an Agent-Agnostic Discord Adapter

This is the useful part for your eventual standalone implementation.

### 7.1 Boundary

The reusable adapter should own:

- Discord gateway and REST integration
- route normalization
- session key normalization
- reply and thread planning
- component and modal handling
- delivery retries and chunking
- typing, reactions, and presence
- feature-specific context packing

The agent bridge should own:

- how to call the downstream agent
- how to persist agent transcripts
- tool policy
- memory/compression policy
- mapping agent outputs into adapter egress requests

Do not bury Discord rules inside each agent implementation.

### 7.2 Ingress Contract

The adapter should emit a stable normalized envelope like:

| Field | Purpose |
|---|---|
| `provider` | `discord` |
| `account_id` | Bot/account identity when multi-account |
| `surface_kind` | `dm`, `channel`, `thread`, `forum_post` |
| `conversation_id` | Current Discord conversation target |
| `parent_conversation_id` | Parent channel/forum when present |
| `session_key` | Canonical adapter-owned session key |
| `sender` | Normalized sender identity |
| `body` | Current message text |
| `reply_excerpt` | Bounded referenced-message excerpt |
| `pending_history` | Bounded history since last bot reply |
| `thread_starter` | Optional starter text |
| `channel_topic` | Untrusted metadata |
| `media` | Typed attachment descriptors |
| `command` | Slash/native command metadata when applicable |
| `capabilities` | What Discord features are available on this surface |

That contract should be agent-neutral.

### 7.3 Egress Contract

The adapter should accept a stable output request like:

| Field | Purpose |
|---|---|
| `target` | canonical Discord target |
| `reply_to` | optional message reference |
| `thread_policy` | stay, create, bind, or none |
| `stream_mode` | off, partial, edit-preview, block |
| `content` | text payload |
| `attachments` | files/media |
| `components` | buttons/selects/containers |
| `modal` | optional form spec |
| `persona` | webhook/display override if supported |
| `delivery_options` | retries, chunking, ephemeral hints, mentions |

Again, agent-neutral.

### 7.4 Context Rules

The adapter should enforce these rules itself:

- never pull raw Discord history by default
- never rely on full agent transcript replay to recover reply context
- always treat threads and forum posts as first-class surfaces
- always distinguish untrusted channel metadata from trusted transport metadata
- bound every adapter-added context field
- isolate transport context from agent memory/context policy

### 7.5 Runtime Rules

The adapter should provide:

- non-blocking listeners
- per-session keyed work queues
- idempotent delivery where practical
- retry/rate-limit handling
- explicit logging of drop reasons
- metrics for queue time, process time, delivery time, retries, and failures

### 7.6 Packaging Rule

The reusable adapter should not depend on Hermes session objects or OpenClaw ACP/session objects.

Instead, it should expose small bridge interfaces:

- `ingest(envelope) -> agent request`
- `egress(agent result) -> delivery request`
- `session_binding_store`
- `interaction_registry`
- `capability_registry`

That is what will make it genuinely pluggable.

## 8. Recommended Build Plan

If you want a world-class reusable adapter rather than another one-off gateway file, the order should be:

1. Define the normalized ingress and egress contracts.
2. Implement canonical route/session normalization for Discord surfaces.
3. Add keyed inbound worker and delivery planner.
4. Add reply excerpts, pending-history windows, and thread/forum context packing.
5. Add native command and component support.
6. Add preview streaming, retries, and rate-limit handling.
7. Add thread bindings and optional agent personas.
8. Add agent bridges for Hermes and OpenClaw.

Do not start by porting features. Start by locking the contracts and ownership boundaries.

## 9. Bottom Line

OpenClaw currently has the better Discord adapter.

Hermes currently has the simpler gateway.

For your goal, the winning direction is:

- take OpenClaw's transport rigor
- avoid OpenClaw's runtime coupling
- fix Hermes's transcript-heavy context model
- define a stable Discord adapter contract that neither agent owns

That is the path to a Discord experience that feels native, performant, and reusable instead of being another per-agent hodgepodge.
