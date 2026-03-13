# honcho-integration-spec

Comparison of Hermes Agent vs. openclaw-honcho — and a porting spec for bringing Hermes patterns into other Honcho integrations.

---

## Overview

Two independent Honcho integrations have been built for two different agent runtimes: **Hermes Agent** (Python, baked into the runner) and **openclaw-honcho** (TypeScript plugin via hook/tool API). Both use the same Honcho peer paradigm — dual peer model, `session.context()`, `peer.chat()` — but they made different tradeoffs at every layer.

This document maps those tradeoffs and defines a porting spec: a set of Hermes-originated patterns, each stated as an integration-agnostic interface, that any Honcho integration can adopt regardless of runtime or language.

> **Scope** Both integrations work correctly today. This spec is about the delta — patterns in Hermes that are worth propagating and patterns in openclaw-honcho that Hermes should eventually adopt. The spec is additive, not prescriptive.

---

## Architecture comparison

### Hermes: baked-in runner

Honcho is initialised directly inside `AIAgent.__init__`. There is no plugin boundary. Session management, context injection, async prefetch, and CLI surface are all first-class concerns of the runner. Context is injected once per session (baked into `_cached_system_prompt`) and never re-fetched mid-session — this maximises prefix cache hits at the LLM provider.

Turn flow:

```
user message
  → _honcho_prefetch()       (reads cache — no HTTP)
  → _build_system_prompt()   (first turn only, cached)
  → LLM call
  → response
  → _honcho_fire_prefetch()  (daemon threads, turn end)
       → prefetch_context() thread  ──┐
       → prefetch_dialectic() thread ─┴→ _context_cache / _dialectic_cache
```

### openclaw-honcho: hook-based plugin

The plugin registers hooks against OpenClaw's event bus. Context is fetched synchronously inside `before_prompt_build` on every turn. Message capture happens in `agent_end`. The multi-agent hierarchy is tracked via `subagent_spawned`. This model is correct but every turn pays a blocking Honcho round-trip before the LLM call can begin.

Turn flow:

```
user message
  → before_prompt_build (BLOCKING HTTP — every turn)
       → session.context()
  → system prompt assembled
  → LLM call
  → response
  → agent_end hook
       → session.addMessages()
       → session.setMetadata()
```

---

## Diff table

| Dimension | Hermes Agent | openclaw-honcho |
|---|---|---|
| **Context injection timing** | Once per session (cached). Zero HTTP on response path after turn 1. | Every turn, blocking. Fresh context per turn but adds latency. |
| **Prefetch strategy** | Daemon threads fire at turn end; consumed next turn from cache. | None. Blocking call at prompt-build time. |
| **Dialectic (peer.chat)** | Prefetched async; result injected into system prompt next turn. | On-demand via `honcho_recall` / `honcho_analyze` tools. |
| **Reasoning level** | Dynamic: scales with message length. Floor = config default. Cap = "high". | Fixed per tool: recall=minimal, analyze=medium. |
| **Memory modes** | `user_memory_mode` / `agent_memory_mode`: hybrid / honcho / local. | None. Always writes to Honcho. |
| **Write frequency** | async (background queue), turn, session, N turns. | After every agent_end (no control). |
| **AI peer identity** | `observe_me=True`, `seed_ai_identity()`, `get_ai_representation()`, SOUL.md → AI peer. | Agent files uploaded to agent peer at setup. No ongoing self-observation. |
| **Context scope** | User peer + AI peer representation, both injected. | User peer (owner) representation + conversation summary. `peerPerspective` on context call. |
| **Session naming** | per-directory / global / manual map / title-based. | Derived from platform session key. |
| **Multi-agent** | Single-agent only. | Parent observer hierarchy via `subagent_spawned`. |
| **Tool surface** | Single `query_user_context` tool (on-demand dialectic). | 6 tools: session, profile, search, context (fast) + recall, analyze (LLM). |
| **Platform metadata** | Not stripped. | Explicitly stripped before Honcho storage. |
| **Message dedup** | None. | `lastSavedIndex` in session metadata prevents re-sending. |
| **CLI surface in prompt** | Management commands injected into system prompt. Agent knows its own CLI. | Not injected. |
| **AI peer name in identity** | Replaces "Hermes Agent" in DEFAULT_AGENT_IDENTITY when configured. | Not implemented. |
| **QMD / local file search** | Not implemented. | Passthrough tools when QMD backend configured. |
| **Workspace metadata** | Not implemented. | `agentPeerMap` in workspace metadata tracks agent→peer ID. |

---

## Patterns

Six patterns from Hermes are worth adopting in any Honcho integration. Each is described as an integration-agnostic interface.

**Hermes contributes:**
- Async prefetch (zero-latency)
- Dynamic reasoning level
- Per-peer memory modes
- AI peer identity formation
- Session naming strategies
- CLI surface injection

**openclaw-honcho contributes back (Hermes should adopt):**
- `lastSavedIndex` dedup
- Platform metadata stripping
- Multi-agent observer hierarchy
- `peerPerspective` on `context()`
- Tiered tool surface (fast/LLM)
- Workspace `agentPeerMap`

---

## Spec: async prefetch

### Problem

Calling `session.context()` and `peer.chat()` synchronously before each LLM call adds 200–800ms of Honcho round-trip latency to every turn.

### Pattern

Fire both calls as non-blocking background work at the **end** of each turn. Store results in a per-session cache keyed by session ID. At the **start** of the next turn, pop from cache — the HTTP is already done. First turn is cold (empty cache); all subsequent turns are zero-latency on the response path.

### Interface contract

```typescript
interface AsyncPrefetch {
  // Fire context + dialectic fetches at turn end. Non-blocking.
  firePrefetch(sessionId: string, userMessage: string): void;

  // Pop cached results at turn start. Returns empty if cache is cold.
  popContextResult(sessionId: string): ContextResult | null;
  popDialecticResult(sessionId: string): string | null;
}

type ContextResult = {
  representation: string;
  card: string[];
  aiRepresentation?: string;  // AI peer context if enabled
  summary?: string;           // conversation summary if fetched
};
```

### Implementation notes

- **Python:** `threading.Thread(daemon=True)`. Write to `dict[session_id, result]` — GIL makes this safe for simple writes.
- **TypeScript:** `Promise` stored in `Map<string, Promise<ContextResult>>`. Await at pop time. If not resolved yet, return null — do not block.
- The pop is destructive: clears the cache entry after reading so stale data never accumulates.
- Prefetch should also fire on first turn (even though it won't be consumed until turn 2).

### openclaw-honcho adoption

Move `session.context()` from `before_prompt_build` to a post-`agent_end` background task. Store result in `state.contextCache`. In `before_prompt_build`, read from cache instead of calling Honcho. If cache is empty (turn 1), inject nothing — the prompt is still valid without Honcho context on the first turn.

---

## Spec: dynamic reasoning level

### Problem

Honcho's dialectic endpoint supports reasoning levels from `minimal` to `max`. A fixed level per tool wastes budget on simple queries and under-serves complex ones.

### Pattern

Select the reasoning level dynamically based on the user's message. Use the configured default as a floor. Bump by message length. Cap auto-selection at `high` — never select `max` automatically.

### Logic

```
< 120 chars  → default (typically "low")
120–400 chars → one level above default (cap at "high")
> 400 chars  → two levels above default (cap at "high")
```

### Config key

Add `dialecticReasoningLevel` (string, default `"low"`). This sets the floor. The dynamic bump always applies on top.

### openclaw-honcho adoption

Apply in `honcho_recall` and `honcho_analyze`: replace fixed `reasoningLevel` with the dynamic selector. `honcho_recall` uses floor `"minimal"`, `honcho_analyze` uses floor `"medium"` — both still bump with message length.

---

## Spec: per-peer memory modes

### Problem

Users want independent control over whether user context and agent context are written locally, to Honcho, or both.

### Modes

| Mode | Effect |
|---|---|
| `hybrid` | Write to both local files and Honcho (default) |
| `honcho` | Honcho only — disable corresponding local file writes |
| `local` | Local files only — skip Honcho sync for this peer |

### Config schema

```json
{
  "memoryMode": "hybrid",
  "userMemoryMode": "honcho",
  "agentMemoryMode": "hybrid"
}
```

Resolution order: per-peer field wins → shorthand `memoryMode` → default `"hybrid"`.

### Effect on Honcho sync

- `userMemoryMode=local`: skip adding user peer messages to Honcho
- `agentMemoryMode=local`: skip adding assistant peer messages to Honcho
- Both local: skip `session.addMessages()` entirely
- `userMemoryMode=honcho`: disable local USER.md writes
- `agentMemoryMode=honcho`: disable local MEMORY.md / SOUL.md writes

---

## Spec: AI peer identity formation

### Problem

Honcho builds the user's representation organically by observing what the user says. The same mechanism exists for the AI peer — but only if `observe_me=True` is set for the agent peer. Without it, the agent peer accumulates nothing.

Additionally, existing persona files (SOUL.md, IDENTITY.md) should seed the AI peer's Honcho representation at first activation.

### Part A: observe_me=True for agent peer

```typescript
await session.addPeers([
  [ownerPeer.id, { observeMe: true,  observeOthers: false }],
  [agentPeer.id, { observeMe: true,  observeOthers: true  }], // was false
]);
```

One-line change. Foundational. Without it, the AI peer representation stays empty regardless of what the agent says.

### Part B: seedAiIdentity()

```typescript
async function seedAiIdentity(
  agentPeer: Peer,
  content: string,
  source: string
): Promise<boolean> {
  const wrapped = [
    `<ai_identity_seed>`,
    `<source>${source}</source>`,
    ``,
    content.trim(),
    `</ai_identity_seed>`,
  ].join("\n");

  await agentPeer.addMessage("assistant", wrapped);
  return true;
}
```

### Part C: migrate agent files at setup

During `honcho setup`, upload agent-self files (SOUL.md, IDENTITY.md, AGENTS.md) to the agent peer via `seedAiIdentity()` instead of `session.uploadFile()`. This routes content through Honcho's observation pipeline.

### Part D: AI peer name in identity

When the agent has a configured name, prepend it to the injected system prompt:

```typescript
const namePrefix = agentName ? `You are ${agentName}.\n\n` : "";
return { systemPrompt: namePrefix + "## User Memory Context\n\n" + sections };
```

### CLI surface

```
honcho identity <file>    # seed from file
honcho identity --show    # show current AI peer representation
```

---

## Spec: session naming strategies

### Problem

A single global session means every project shares the same Honcho context. Per-directory sessions provide isolation without requiring users to name sessions manually.

### Strategies

| Strategy | Session key | When to use |
|---|---|---|
| `per-directory` | basename of CWD | Default. Each project gets its own session. |
| `global` | fixed string `"global"` | Single cross-project session. |
| manual map | user-configured per path | `sessions` config map overrides directory basename. |
| title-based | sanitized session title | When agent supports named sessions set mid-conversation. |

### Config schema

```json
{
  "sessionStrategy": "per-directory",
  "sessionPeerPrefix": false,
  "sessions": {
    "/home/user/projects/foo": "foo-project"
  }
}
```

### CLI surface

```
honcho sessions              # list all mappings
honcho map <name>            # map cwd to session name
honcho map                   # no-arg = list mappings
```

Resolution order: manual map → session title → directory basename → platform key.

---

## Spec: CLI surface injection

### Problem

When a user asks "how do I change my memory settings?" the agent either hallucinates or says it doesn't know. The agent should know its own management interface.

### Pattern

When Honcho is active, append a compact command reference to the system prompt. Keep it under 300 chars.

```
# Honcho memory integration
Active. Session: {sessionKey}. Mode: {mode}.
Management commands:
  honcho status                    — show config + connection
  honcho mode [hybrid|honcho|local] — show or set memory mode
  honcho sessions                  — list session mappings
  honcho map <name>                — map directory to session
  honcho identity [file] [--show]  — seed or show AI identity
  honcho setup                     — full interactive wizard
```

---

## openclaw-honcho checklist

Ordered by impact:

- [ ] **Async prefetch** — move `session.context()` out of `before_prompt_build` into post-`agent_end` background Promise
- [ ] **observe_me=True for agent peer** — one-line change in `session.addPeers()`
- [ ] **Dynamic reasoning level** — add helper; apply in `honcho_recall` and `honcho_analyze`; add `dialecticReasoningLevel` to config
- [ ] **Per-peer memory modes** — add `userMemoryMode` / `agentMemoryMode` to config; gate Honcho sync and local writes
- [ ] **seedAiIdentity()** — add helper; use during setup migration for SOUL.md / IDENTITY.md
- [ ] **Session naming strategies** — add `sessionStrategy`, `sessions` map, `sessionPeerPrefix`
- [ ] **CLI surface injection** — append command reference to `before_prompt_build` return value
- [ ] **honcho identity subcommand** — seed from file or `--show` current representation
- [ ] **AI peer name injection** — if `aiPeer` name configured, prepend to injected system prompt
- [ ] **honcho mode / sessions / map** — CLI parity with Hermes

Already done in openclaw-honcho (do not re-implement): `lastSavedIndex` dedup, platform metadata stripping, multi-agent parent observer, `peerPerspective` on `context()`, tiered tool surface, workspace `agentPeerMap`, QMD passthrough, self-hosted Honcho.

---

## nanobot-honcho checklist

Greenfield integration. Start from openclaw-honcho's architecture and apply all Hermes patterns from day one.

### Phase 1 — core correctness

- [ ] Dual peer model (owner + agent peer), both with `observe_me=True`
- [ ] Message capture at turn end with `lastSavedIndex` dedup
- [ ] Platform metadata stripping before Honcho storage
- [ ] Async prefetch from day one — do not implement blocking context injection
- [ ] Legacy file migration at first activation (USER.md → owner peer, SOUL.md → `seedAiIdentity()`)

### Phase 2 — configuration

- [ ] Config schema: `apiKey`, `workspaceId`, `baseUrl`, `memoryMode`, `userMemoryMode`, `agentMemoryMode`, `dialecticReasoningLevel`, `sessionStrategy`, `sessions`
- [ ] Per-peer memory mode gating
- [ ] Dynamic reasoning level
- [ ] Session naming strategies

### Phase 3 — tools and CLI

- [ ] Tool surface: `honcho_profile`, `honcho_recall`, `honcho_analyze`, `honcho_search`, `honcho_context`
- [ ] CLI: `setup`, `status`, `sessions`, `map`, `mode`, `identity`
- [ ] CLI surface injection into system prompt
- [ ] AI peer name wired into agent identity
