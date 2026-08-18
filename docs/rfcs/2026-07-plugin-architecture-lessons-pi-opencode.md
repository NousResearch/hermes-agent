# Research spike: plugin-architecture lessons from Pi and OpenCode

**Issue:** #64180 · **Informs:** #64164 (event bus), #64161 (streaming hooks), #64162 (pluggable approval), #64165 (manifest v2), #64229 (lifecycle/ledger), #64230 (Plugin Doctor)

**Method.** Both systems were read at source level (shallow clones pinned to a commit), not from docs sites alone: Pi (`badlogic/pi-mono`, now `earendil-works/pi`) at `eb79351` (v0.80.7, 2026-07-14) and OpenCode (`sst/opencode`, now `anomalyco/opencode`) at `c69abee` (v1.18.2, 2026-07-16). Claims below carry file:line references into those commits. Where something could not be found (ADRs, policies, timeouts), that absence was verified by search and is reported as a finding. Hermes ground rules from #64182 (additive-only, prompt-cache sacred, observer-first, fail-closed security) are treated as overriding constraints throughout — this report grades imported patterns against them, per the adapt-don't-copy rubric.

**Headline.** The two systems are near-perfect opposites on the four design axes Hermes currently has open, which makes them a natural controlled experiment:

| Axis | Pi | OpenCode | Hermes proposal on the table |
|---|---|---|---|
| Per-delta streaming hook | Yes — awaited inline, no timeout | Structurally absent (text-end only) | Observer per-delta + never-block contract (#64161) |
| Veto semantics | Typed per-event result vocabulary (`{block}`, `{cancel}`, `"handled"`) | Veto-by-throw (bug and policy denial indistinguishable) | TBD in #64162 |
| Guard-hook failure | Fail closed (`tool_call` only) | No runtime containment at all | Ground rule 4: fail closed on security-adjacent |
| Plugin event bus | Yes — 33 lines, **un-namespaced** channels | None (plugins observe core bus, cannot emit) | Namespaced `ctx.emit`/`ctx.subscribe` (#64164) |

Neither system has hook timeouts. Both have shipped hang-class or drift-class failures because of it. That is the single strongest cross-cutting lesson for Hermes.

---

## 1. Pi (badlogic/pi-mono → earendil-works/pi)

Extensions are in-process TypeScript modules (loaded via jiti) receiving an `ExtensionAPI`; no separate process, no IPC, no manifest permissions. Pi explicitly rejected MCP as its extension mechanism (Zechner, ["What if you don't need MCP at all?"](https://mariozechner.at/posts/2025-11-02-what-if-you-dont-need-mcp/)). Notably, Pi's founding "minimal, no hooks" stance (Nov 2025) reversed into a 33-event extension system within ~7 months — extensibility demand won; the no-MCP stance held.

### 1.1 Hook/event taxonomy

33 event types (`src/core/extensions/types.ts:507-902`). The load-bearing design choice: **every mutating event gets its own typed emitter with its own result vocabulary**, rather than one generic middleware pipe —

- `tool_call` → `{ block: true, reason }`; argument mutation in place, explicitly "no re-validation after your mutation" (`docs/extensions.md:742-765`)
- `session_before_*` → `{ cancel: true }`, first canceller wins, later handlers skipped
- `input` → `"handled"` (short-circuit) vs `"transform"` (chain) (`runner.ts:1148-1188`)
- `tool_result` → partial-patch accumulation across handlers (`runner.ts:835-883`)
- `before_agent_start` → system-prompt chaining with a live `ctx.getSystemPrompt()` reflecting earlier handlers (`runner.ts:1034-1098`)
- Observer events have **no return channel at all** — the observer/mutator split is enforced by which emitter the event flows through, not by convention.

Dispatch is fully sequential async (`runner.ts:759-791`): every handler awaited one at a time; the only ordering rule is extension load order (project → global → CLI, `loader.ts:660-708`) then registration order. No priorities, no phases — and no community demand for them in ~8 months. Name collisions are handled per registry with explicit deterministic policies instead of dependency resolution: first-wins tools, suffixed duplicate commands (`name:2`), last-wins shortcuts with a warning plus an 18-key reserved denylist (`runner.ts:421-604`).

Two ordering guarantees worth copying: extensions see events **before** the UI and **before** session persistence (`agent-session.ts:596-601`), and parallel sibling tool calls are "preflighted sequentially, then executed concurrently" (`docs/extensions.md:750`).

### 1.2 Plugin-to-plugin interaction

A 33-line shared event bus (`src/core/event-bus.ts`): `pi.events.emit/on` on **arbitrary string channels** — no namespacing, no declarations, no collision protection; per-handler try/catch and an unsubscribe closure. Everything richer (capability registries, dependencies) is deliberately absent; a community RFC to build richer coordination as an extension (pi#2715) was left to userland.

### 1.3 Compatibility strategy

No API versioning, no handshake, no deprecation annotations. In its place, three working practices: (a) loader-level alias shims that kept old imports resolving across the `@mariozechner`→`@earendil-works` package rename and a pi-ai API split, with pre-announced removal (`loader.ts:47-71`; `CHANGELOG.md:243`); (b) loud "Breaking Changes" changelog sections with **automatic migrations** (directory renames on startup, session format v2→v3 auto-migrated) for the big hooks+customTools→extensions unification (v0.35-0.37); (c) per-tool `prepareArguments()` shims for old argument shapes.

Real breakage on record: pi#2860 — an internal session-management refactor made `pi.sendUserMessage()` after `ctx.newSession()` silently drop messages. The remediation pattern is distinctive: **stale-context poisoning** — after session replacement, every captured context getter throws a paragraph-long teaching error pointing at the safe `withSession` pattern (`runner.ts:514-527`; `docs/extensions.md:1223-1265` documents the footgun with unsafe-pattern code). Compat by making misuse loud, not by never changing.

### 1.4 Failure isolation

Contain errors, don't contain time: every handler call is individually try/caught and surfaced (red stack trace in chat UI), load failures skip only the broken extension, and a `pi -ne` no-extensions escape hatch exists. The **one deliberate exception**: `tool_call` has no internal catch — a guard-hook crash blocks the tool ("Extension failed, blocking execution"), converted into an error tool result the LLM sees; the loop survives, the guard fails closed (`runner.ts:885-906`; `agent-session.ts:454-467`; `agent-loop.ts:657-665`).

**There are zero timeouts.** All handlers — including per-token `message_update` — are awaited inline in the agent event pipeline (`agent-session.ts:598, 728-734`). A hung extension freezes the agent; the changelog records hang-class fixes (pi#5687 background handles, pi#5115 shutdown drains). Mitigations offered are cooperative only (`ctx.signal` AbortSignal, Esc-abort).

No sandbox, documented as a decision: "a partial in-process sandbox would be easy to misunderstand as a security boundary" (`docs/security.md:33-35`); the only gate is project-trust on load.

### 1.5 Design-history record

**Pi has no ADRs** — the Discord recollection that prompted this spike is not substantiated for Pi. Rationale lives in four places: "footgun" sections inside user docs (effectively inline ADRs), a 5,000-line issue-linked changelog (which records abandoned designs: the hooks/customTools split, slash-commands→prompt-templates rename, a fetch-override proxy abandoned for undici dispatchers), Zechner's blog, and issues themselves.

### 1.6 Prompt/context construction

Pi is the only surveyed system that treats **prompt-cache stability as part of the extension API contract**: extensions receive structured prompt inputs (`systemPromptOptions` — the same decomposed inputs Pi itself uses) rather than a final string; per-request context transforms operate on a `structuredClone` so session history is never corrupted; and cache-friendly dynamic tool loading (v0.80.6, pi#6474) activates tools additively via native provider deferred loading (Anthropic `defer_loading`) explicitly to avoid invalidating the prefix cache — with documented warnings about second-order invalidation through prompt-metadata changes (`docs/extensions.md:2254-2290`).

---

## 2. OpenCode (sst/opencode → anomalyco/opencode)

Server (Bun/Effect) + clients (TUI/desktop/web); plugins are npm packages or local TS files loaded in-process, with **two plugin kinds in one package** — `server` and `tui` entrypoints, hard rule one-or-the-other (`shared.ts:103-114, 293-295`). A v2 plugin API ships side-by-side with v1 (`/v2/effect`, `/v2/promise` subpaths), designed in a candid 516-line `PLAN.md`.

### 2.1 Hook/event taxonomy

One `Hooks` bag returned by an async factory (`packages/plugin/src/index.ts:74, 222-335`): ~16 mutating `(input, output)` hooks (mutate `output` in place; host reads it back) plus declarative registries (tools, auth, providers) and a single firehose `event` observer. The whole dispatch engine is ~13 lines (`plugin/index.ts:280-293`): sequential, awaited, load-order, later hooks see earlier mutations — ordering documented as a spec ("global config → project config → global plugin dir → project plugin dir"; v2: "plugin registration order, then transform registration order").

**Veto is throw.** The documented idiom for denying a tool call is `throw new Error("Do not read .env files")` (`plugins.mdx:247-257`) — policy denial and plugin bug are indistinguishable in every consumer downstream.

**A typed-but-dead hook.** `permission.ask` — the only hook with real decision semantics (`output.status: "ask" | "deny" | "allow"`) — exists in the published types but has **no dispatch site anywhere in the tree**: a permission-subsystem rewrite orphaned it, types kept compiling, plugins silently no-op'd (oc#7006, open since Jan 2026). This is the sharpest single failure mode found in the whole spike.

### 2.2 Plugin-to-plugin interaction

No dependencies, no registry, no plugin-emitted events. Interaction is (a) blind composition through the shared mutable `output` (last-writer-wins per field) and (b) observing the core event bus. The 2026 core bus itself is heavyweight — event-sourced, durable (SQLite, per-aggregate sequences, idempotent replay), versioned wire types, and a **backpressure guard**: `allBounded` wraps a dropping queue and fails slow subscribers with `SubscriberOverflowError` (`packages/core/src/event.ts:152-164`). Plugins get the untyped tail of a three-tier bridge (Effect streams → global emitter → `event` hook), fired **fire-and-forget** — an async observer's rejection is an unhandled promise rejection invisible to host error routing (`plugin/index.ts:251-258`).

### 2.3 Compatibility strategy

Lockstep versioning (`opencode` = `@opencode-ai/plugin` = `@opencode-ai/sdk` = 1.18.2) plus one gate: npm plugins may declare `engines.opencode` semver ranges, checked at load (`shared.ts:194-205`) — but it is plugin-opt-in, and local file plugins skip it entirely. Three generations of module shape are loaded simultaneously; superseded packages are silently ignored via a hardcoded `DEPRECATED_PLUGIN_PACKAGES` list.

The community-experience record is instructive: v1.14.42 — a **patch release** — removed the whole `api.command.*` TUI namespace with no deprecation cycle (oc#26557); the aftermath is visible in-tree as a deprecated shim added back *after* the outcry ("Legacy `api.command` API kept so v1 plugins can initialize. Remove in v2", `tui.ts:87-120`). No written deprecation policy exists anywhere. Shims-after-outcry is the de facto process.

### 2.4 Failure isolation

Strong at the edges, absent in the middle. Load is staged (`install | entry | compatibility | missing | load`) with per-plugin, per-stage containment and user-visible toasts (`loader.ts:82-93`; `plugin/index.ts:215-249`). Runtime hooks have **no catch and no timeout**: a throw in a tool hook fails that tool call (the sanctioned veto), a throw in chat/transform hooks aborts the turn (session errors, server survives), a **hang hangs the turn forever** — the v2 PLAN explicitly lists "Transform timeouts" under *Deferred Decisions* (`PLAN.md:507-510`). Boot re-entrancy burned them: a plugin calling the SDK client during its own init deadlocked startup (oc#7741). The official troubleshooting page's first advice is "start by disabling plugins."

Two mature exceptions worth stealing: the streaming hot path is protected **structurally** — there is no per-delta hook at all; text hooks fire once at `text-end` (`processor.ts:512-524`) — and the TUI runtime has scope-tracked registrations (a `Proxy`-wrapped keymap API auto-records every registration per plugin, enabling clean live deactivate) plus a **hard 5s dispose budget** racing each cleanup against a timer (`runtime.ts:122-226, 388-468`).

### 2.5 Design-history record

No ADR system; the `PLAN.md` for v2 is the exception and is better than most ADR archives — it names v1's mistakes (the returned-hooks bag, finalizer-triggered special cases), specifies ordering as a contract, splits replayable *transforms* from live *hooks*, and — crucially — carries an honest **Deferred Decisions** section (typed error model, transform timeouts) rather than pretending closure.

### 2.6 Prompt/context construction

Plugins can touch every layer, but the deep layers are gated behind an `experimental.` prefix (`experimental.chat.system.transform`, `experimental.chat.messages.transform`, compaction prompt replacement) — a deliberate two-tier stability promise: interception at operation boundaries is stable, rewriting the context itself is not. No cache-stability discipline comparable to Pi's was found.

---

## 3. Adopt / Adapt / Avoid for Hermes

Grading against #64182's ground rules. "Validated" = the proposal already on the Hermes issue is independently confirmed by field evidence.

| # | Lesson | Verdict | Maps to | Evidence |
|---|---|---|---|---|
| 1 | **Typed per-hook result vocabularies, not veto-by-throw.** Pi's `{block, reason}` / `{cancel}` / `"handled"` enums vs OpenCode's throw-idiom (bug ≡ policy denial) and its dead `permission.ask`. Approval/gate hooks need enumerated results dispatched from the policy engine itself. | **Adopt** | #64162 | Pi runner.ts:759-1188; oc plugins.mdx:247-257, oc#7006 |
| 2 | **Guard hooks fail closed; observers fail open.** Pi contains every handler error except `tool_call`, whose crash blocks the tool with an LLM-visible error result. Independently confirms ground rule 4 — and refines it: the failure mode of a *crashed* security hook must also be closed, not just its config default. | **Adopt** (validated) | #64162, #64204 | Pi agent-session.ts:454-467, agent-loop.ts:657-665 |
| 3 | **Hook wire-up drift is the killer bug class: CI-check that every declared hook has a live dispatch site.** OpenCode's `permission.ask` sat typed-but-dead for 6+ months after a subsystem rewrite. Hermes already stores unknown hook names for forward compat (`register_hook`, plugins.py ~L1158) — the same drift is possible. A `VALID_HOOKS` ↔ `invoke_hook(` cross-check is a one-file test. | **Adopt now** (cheap, standalone) | #64230 (Doctor), CI | oc index.ts:261 + absent trigger site, oc#7006 |
| 4 | **Deadline budgets on plugin callbacks — be the first framework to have them.** Neither system times out runtime hooks; both shipped hang-class failures (pi#5687/#5115; oc#7741, "Transform timeouts" deferred twice). OpenCode's own TUI dispose path (hard 5s, per-cleanup timer race) proves the mechanism is practical. Observer hooks: enforce a budget and log-and-drop. Mutating/guard hooks: budget + fail per lesson 2. | **Adopt** (differentiator) | #64161, #64164, #64229 | Pi grep: zero timeout logic; oc PLAN.md:507-510, runtime.ts:122-226 |
| 5 | **Per-delta streaming hooks are viable only if non-blocking is structural, not documentary.** The controlled experiment: Pi offers per-delta and awaits inline → slow observer throttles the visible stream, hangs freeze it; OpenCode offers nothing per-delta → hot path safe, TTS use case unserved. #64161's "never-block contract + buffered-queue helper" is the right middle — but make the bounded queue the *only* consumption path (drop/coalesce policy included, cf. OpenCode's `SubscriberOverflowError` dropping queue), not an optional convenience next to a raw sync callback. | **Adapt** | #64161 | Pi agent-session.ts:728-734; oc processor.ts:512-524, event.ts:152-164 |
| 6 | **The namespaced bus proposal is ahead of both systems — proceed, with their two omissions fixed.** Pi's bus works but has arbitrary un-namespaced string channels and no discoverability; OpenCode has no plugin-emit at all. #64164's `<plugin_key>:` enforcement, reserved `hermes:` prefix, advisory declarations, recursion cap, and deterministic subscription order have no counterexample in the field. Carry over per-callback isolation (both systems do this right) and add lesson-4 budgets. Fire-and-forget with return-values-ignored matches both systems' stable practice. | **Adopt own design** (validated) | #64164 | Pi event-bus.ts (whole file); oc plugin/index.ts:251-258 |
| 7 | **Load order as the only priority system; explicit per-registry collision policies.** Zero configuration, deterministic, and no field demand for priorities in either ecosystem. Document Hermes's ordering as a spec the way OpenCode's PLAN does; pick a collision rule per registry (Pi: first-wins tools / suffixed commands / reserved denylist) instead of building dependency resolution. | **Adopt** | #64164, #64229 | Pi loader.ts:660-708, runner.ts:421-604; oc PLAN.md:144-146 |
| 8 | **Host-enforced compat gate + written deprecation window; migration tooling over semver ceremony.** OpenCode's `engines` gate is the right shape but plugin-opt-in only, and its patch-release API removal (oc#26557) shows lockstep versioning without policy is social, not mechanical. Pi shows the complement: loud breaking changes + automatic migrations + alias shims *before* removal. Manifest v2 should carry a host-checked `api_version` range; the repo should carry a one-paragraph deprecation policy. | **Adapt** | #64165, #64179 | oc shared.ts:194-205, tui.ts:87-120, oc#26557; Pi CHANGELOG:243, 3530-3620 |
| 9 | **Scoped registrations with auto-tracked disposal; poison stale contexts with teaching errors.** OpenCode's Proxy-tracked per-plugin scopes (clean live disable) and Pi's post-replacement context poisoning (silent race pi#2860 → loud self-documenting error) are the two halves of a robust lifecycle story — exactly what the #64229 ownership ledger needs. | **Adopt** | #64229 | oc runtime.ts:143-160; Pi runner.ts:514-527, docs:1223-1265 |
| 10 | **Prompt-cache stability as API contract is real and Pi proves it's implementable.** Structured prompt inputs instead of final strings, `structuredClone` for ephemeral transforms, additive-only tool activation with provider deferred loading, documented second-order invalidation warnings. Strongest possible validation of ground rule 2, with a concrete reference implementation for cache-safe injection. | **Adopt** (validated) | #64167 | Pi docs:2254-2290, CHANGELOG 0.80.6/pi#6474 |
| 11 | **Half-sandboxes: both systems refuse, for the same stated reason.** Pi documents that a partial in-process sandbox "would be easy to misunderstand as a security boundary"; OpenCode runs plugins fully privileged with path-containment only. Viable while plugin authors ≈ users; Hermes's Skills-Hub-style trust/scan pipeline is the nearer-term marketplace answer than in-process isolation. | **Adapt with eyes open** | security posture | Pi docs/security.md:5-37; oc shared.ts:89-97 |
| 12 | **Events to plugins before UI and before persistence; boot-stage the plugin-facing client.** Pi's explicit ordering guarantee removes a whole class of races; OpenCode's plugins-as-API-clients design is elegant but deadlocked startup when a plugin called the API mid-init (oc#7741) — if ctx ever grows client-like powers, stage them ("unavailable until ready"). | **Adapt** | #64178, #64229 | Pi agent-session.ts:596-601; oc plugin/index.ts:142-147, oc#7741 |
| 13 | **Neither system has ADRs — and both paid for it.** Pi's rationale is scattered across changelog/blog/footguns; OpenCode broke APIs in patch releases partly because no decision record said not to. Hermes's per-sub-issue design sketches (#64182 style) are already ahead of both; add an explicit **Deferred Decisions** section per design (OpenCode's PLAN.md's best feature) so open questions stay visible instead of silently unresolved. | **Keep + adapt** | process | verified ADR absence in both repos |

## 4. Verified absences (findings, not gaps in the spike)

- **No ADRs in either repo** (repo-wide searches for `adr`/`decision` artifacts). The Discord recollection of "ADRs describing failure modes" is unsubstantiated for both; the nearest equivalents are Pi's docs footgun sections and OpenCode's single v2 PLAN.md.
- **No runtime hook timeouts in either system** (grep-verified in Pi's `src/core/extensions/`; OpenCode's own plan defers them).
- **No written deprecation or plugin-API stability policy in either repo.**
- Caveats: Pi's #2860→remediation causality is inferred from matching failure/fix, not a maintainer statement; OpenCode pre-rewrite history was not diffed (shallow clones); maintainer replies inside cited issues were not visible in fetched content.

---

*Spike time-boxed per #64180. Primary sources: `earendil-works/pi` @ `eb79351` — `packages/coding-agent/src/core/extensions/{runner,loader,types}.ts`, `src/core/{agent-session,event-bus}.ts`, `packages/agent/src/agent-loop.ts`, `docs/{extensions,security}.md`, `CHANGELOG.md`; `anomalyco/opencode` @ `c69abee` — `packages/plugin/src/{index,tui}.ts`, `packages/plugin/src/v2/effect/PLAN.md`, `packages/opencode/src/plugin/{index,shared,loader}.ts`, `packages/opencode/src/plugin/tui/runtime.ts`, `packages/core/src/event.ts`, `packages/web/src/content/docs/plugins.mdx`; issues pi#2860/#2715/#5080/#5687, oc#7006/#26557/#7741/#4850/#12222; mariozechner.at posts (2025-11-02, 2025-11-30).*
