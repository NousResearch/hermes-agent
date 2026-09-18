---
title: "Intent-routed context and remote performance plan"
date: 2026-09-16
status: proposed
owners:
  - Hermes Agent core
  - Hermes Desktop
scope:
  - session-scoped tool and skill selection
  - prompt-cache-safe fallback capability
  - remote gateway startup and session hydration observability
out_of_scope:
  - reducing model context capacity
  - deleting profiles or skills without evidence
  - opening the gateway publicly
  - changing model quality as the primary fix
---

# Intent-routed context and remote performance plan

## Goal

Make Hermes materially faster without sacrificing long-running task capacity: each session should start with the smallest relevant tool and skill surface, retain access to all authorized capabilities through a stable bridge, keep the model's context window unchanged, and preserve prompt-cache stability for the life of the session.

## Decision summary

1. **Do not reduce context capacity.** Restore the normal compression policy and treat compression as a last-resort safety mechanism, not the performance strategy.
2. **Route capabilities once, at the first user intent.** Build a persisted, session-scoped capability plan before the first model call.
3. **Keep the prompt byte-stable after routing.** Never add/remove tool schemas or skill text in-place during a conversation.
4. **Expose a small direct surface plus a generic fallback bridge.** The model gets relevant direct tools for accuracy and speed; every other authorized tool remains reachable through `tool_search` → `tool_describe` → `tool_call` without changing the tool array.
5. **Select skills automatically.** Inject only high-confidence relevant skill bodies/descriptions; retain a names-only catalog and `skills_list`/`skill_view` fallback.
6. **Preserve long-running continuity in durable state.** Persist the capability plan, project/workspace context, task ledger, and memory references separately from raw transcript tokens.
7. **Fix and measure remote transport independently.** Align Desktop/backend versions, instrument every startup leg, and optimize only legs proven slow.

## Long-term and compounding guarantees

This design is acceptable only if it remains an optimization layer rather than a capability ceiling. The following are architectural invariants, not tuning preferences:

1. **Routing ranks; it never permanently excludes.** Intent selection decides which authorized capabilities are direct, not which capabilities exist. The stable bridge preserves eventual access to every tool and skill permitted by the session.
2. **No closed intent taxonomy.** Intents and capability bundles are declarative manifests contributed by built-ins, plugins, MCP servers, skills, and profile manifests. Adding a capability must not require editing a central switch statement.
3. **No fixed universal tool cap.** Direct-tool count is an adaptive prompt budget derived from provider/model limits and measured schema cost. Configuration may set a guardrail, but `15` is an initial benchmark target—not a permanent product limit.
4. **Authorization remains independent of routing.** Policy determines what may run; routing only determines what is immediately visible. A routing bug can add a search hop, but must never grant or revoke authority.
5. **Stable session, evolvable system.** A session's plan is immutable for prompt caching, while every new session uses the latest versioned manifests and ranking evidence. Long-running work can cross domains through the bridge or spawn a focused child without losing the parent goal.
6. **Open extension contract.** Third-party toolsets and skills publish machine-readable names, descriptions, examples, cost/latency hints, prerequisites, and safety labels. The router consumes this contract without vendor-specific code.
7. **Evidence compounds.** Privacy-safe outcomes record bundle choice, fallback searches, bridge escalations, task success, latency, and schema cost. Offline evaluation turns these traces into better manifests, rankings, and benchmark cases; runtime behavior does not silently self-modify.
8. **Versioned and reversible.** Capability-plan and manifest schemas carry versions and migrations. Feature flags allow an immediate return to legacy static toolsets without losing sessions or state.
9. **Provider-agnostic budgets.** Selection uses measured serialized token cost and cache behavior for the active provider/model rather than assumptions tied to one model family.
10. **Capacity is protected.** Context-window size, memory backends, project state, and session lineage remain independent of capability routing. Future larger windows improve continuity without requiring router redesign.

The compounding loop is:

`new capability publishes manifest → benchmark corpus expands → sessions route with the new option → privacy-safe fallback/success evidence accumulates → offline evaluation improves ranking/manifests → future sessions become faster and more accurate`.

## Evidence and current diagnosis

### Confirmed model-side cost

The live runtime showed that latency rises with prompt size. Fresh requests were about 4.6–7.8 seconds, while calls near 198k–200k input tokens took 12–45 seconds. This is provider inference/context processing, not local CPU or database saturation.

The current fresh tool surface is the largest reducible fixed cost:

- 43 direct tool definitions
- about 58,196 serialized characters, approximately 14.5k tokens
- a practical focused bundle (`file`, `terminal`, `todo`, `clarify`, `skills`, `web`) is about 22,247 characters, approximately 5.6k tokens
- the global skill listing adds about 9,781 characters, approximately 2.4k tokens

A typical focused session can therefore remove roughly 9k–11k fixed input tokens without reducing the model context window.

### Attached remote-gateway claims: disposition

| Claim | Live/source finding | Plan response |
|---|---|---|
| Profile listing takes ~20 seconds | Not reproduced. There are 16 profiles; direct listing measured ~0.596 s cold and ~0.028–0.030 s warm. The current source already has cache/single-flight paths and avoids loading full project trees for basic roster/sidebar calls. | Do not delete profiles. Add per-endpoint telemetry and a regression budget; investigate only if a measured endpoint exceeds it. |
| Session hydration blocks on the full transcript | Partly plausible for version-skewed clients, but current backend source supports `defer_history`, `omit_messages`, asynchronous model-history loading, and paginated REST history. The largest preserved transcript is about 1.19M characters, while its local DB fetch measured ~2.4 ms, so transfer/render—not SQLite—is the likely risk. | Align versions first; verify the Desktop actually negotiates deferred history; instrument payload bytes, latest-page paint, background hydration, and transcript rendering. |
| Tailscale relay causes slowness | Not proven from the container. The server-side path is Tailscale, but direct-vs-DERP status must be measured on the Mac client. | Add a client diagnostic showing RTT, WebSocket handshake, reconnect count, and direct/relay status. Open UDP/firewall paths only if relay is confirmed. |
| A nested local+remote agent loop doubles context | Not the configured topology. The Desktop is acting as a frontend to the remote gateway. | Keep a connection-mode invariant and diagnostic that identifies exactly one authoritative agent backend. |
| Switching tabs aborts hydration | Current backend can decouple UI hydration from agent history loading, but the exact Desktop version behavior is not yet proven. | Add a deterministic Desktop test: switch away during hydration, return, and assert the same request completes/cached state is retained. |

### Existing mechanisms to extend

- `toolsets.py` already defines named toolsets and a narrow core.
- `tools/tool_search.py` already provides progressive disclosure for deferred tools.
- `agent/coding_context.py` already computes a session-start posture from workspace state.
- `agent/prompt_builder.py` already builds skill and context sections.
- `tui_gateway/server.py` already resolves platform-specific session toolsets and defers agent construction.
- `tui_gateway/methods_session.py` already supports lazy resume/history hydration.
- `hermes_state_sessions.py` already persists system-prompt identity and compression lineage.
- Hermes context-engine plugins exist, but are not required for phase 1.

The implementation should extend these paths rather than introduce a parallel agent runtime.

## Target architecture

### 1. Session capability plan

Create a small immutable plan before the first model call:

```json
{
  "version": 1,
  "intent": "software_change",
  "confidence": 0.91,
  "platform": "desktop",
  "toolsets": ["file", "terminal", "todo", "clarify", "skills", "web", "desktop_ui", "project"],
  "direct_tools": ["read_file", "search_files", "patch", "write_file", "terminal", "todo", "clarify", "skill_view", "web_search", "web_extract"],
  "preloaded_skills": ["software-development:application-delivery-engineering"],
  "manifest_hash": "..."
}
```

The stored plan must contain no prompt text, secrets, or tool arguments. Persist it with the session and preserve it through compression lineage. Delegated child agents receive a separately routed plan bounded by the parent's explicit allowed toolsets.

### 2. Deterministic intent router

Route from the first user message plus already-authoritative metadata:

- surface/platform
- project and workspace path
- attachment MIME/types
- explicit verbs and object types
- enabled integrations and configured MCPs
- profile manifest
- caller restrictions such as cron `enabled_toolsets`

Use deterministic scoring first: exact triggers, token/BM25 overlap against toolset and skill manifests, file-extension signals, and project markers. This must be fast, local, explainable, and testable.

Do **not** add a mandatory auxiliary-model call before every turn. For ambiguous intents, choose the conservative general bundle and let the stable search bridges resolve the rest. A future optional classifier can be evaluated offline before becoming a runtime dependency.

### 3. Two-tier tool surface

Split allowed capabilities from directly visible schemas:

- **Always-direct kernel:** only the tools needed to plan, ask, search capabilities, and safely recover.
- **Intent-direct tools:** the small set with high confidence for the session's task.
- **Deferred authorized tools:** all other tools that the session is allowed to use, searchable and callable through the generic bridge.

Extend `tools/tool_search.py` so selected built-ins can be deferred as well as MCP/plugin tools. The bridge must call the existing registry and preserve all policy hooks, approvals, validation, platform gates, and toolset authorization. It must not become an authorization bypass.

The selected tool array and its order must remain byte-identical for every model call in the session. A mid-session task shift uses the bridge; it does not rebuild the prompt. Desktop may offer “Start focused chat” for a major task change, but must never force it.

### 4. Automatic skill selection

Replace the all-descriptions prompt with three layers:

1. names-only compact catalog grouped by category
2. top-ranked descriptions for the current intent
3. full content for only one to three high-confidence skills

Keep `skills_list` and `skill_view` directly available. If confidence is below threshold, preload no full skill and let the agent retrieve it. Skill routing must honor profile/plugin visibility and the existing instruction that matching skills are authoritative.

Cache the skill manifest by `(profile, mtime/hash)` so routing never rescans 1.1 GB of copied skill trees per turn. Invalidate only for a new session after installation/update unless the user explicitly requests cache-breaking `--now` behavior.

### 5. Long-running task continuity

Retain the model's full configured context capacity. Add continuity outside the raw prompt:

- persisted capability plan
- project goal and current task ledger
- verified decisions and receipts
- Honcho/durable memory references
- session lineage and compression summaries
- optional context-engine retrieval for older, low-relevance transcript regions

Compression remains near the model/provider's normal safe boundary. It must preserve goals, unresolved tasks, identifiers, approvals, file/branch state, and capability-plan identity. The system/tool prefix remains unchanged across compression.

### 6. Remote-gateway fast path

At connection/session-open time, trace these independent stages:

1. Desktop backend discovery/version handshake
2. authentication/ticket minting
3. HTTP health and WebSocket handshake
4. profile roster
5. session metadata list
6. latest transcript page
7. first transcript paint
8. background historical hydration
9. agent construction and capability routing
10. provider request and first token

Each span records duration, status, payload byte count, connection/profile/session IDs in hashed or opaque form, and version/capability flags—never prompts, responses, secrets, or tool arguments.

Align Desktop and backend versions before interpreting results. The handshake should advertise support for deferred history, paginated transcript, capability-plan version, and stable bridge features. Compatibility fallbacks must be explicit and observable rather than silent.

## Implementation phases

### Phase 0 — Version alignment and reproducible baseline

**Files/surfaces**

- `apps/desktop/` connection/version capability handshake
- `tui_gateway/server.py`
- `tui_gateway/methods_session.py`
- existing telemetry/logging modules

**Work**

- Update the backend to the exact Desktop-compatible release through the controlled upgrade channel, with rollback.
- Record Desktop and backend build identifiers in one diagnostic view.
- Add stage timing for the ten remote-gateway spans above.
- Capture a fixed benchmark corpus: simple Q&A, web research, software change, document work, scheduling, smart-home/MCP task, long-session continuation, remote session open.
- Record fixed prompt components separately: base instructions, memory, skill catalog/full skills, tool schemas, project context, conversation, cache-read tokens.

**Exit criteria**

- Same compatible build/protocol level confirmed at both ends.
- Every remote startup stage observable.
- Three baseline runs per scenario retained with p50/p95 and payload bytes.
- No production behavior change beyond telemetry.

### Phase 1 — Intent router and immutable session plan

**Files**

- extend `agent/coding_context.py` or extract its generic selection primitives into `agent/session_capabilities.py`
- `tui_gateway/server.py`
- `tui_gateway/methods_session.py`
- `hermes_state_sessions.py` and schema/export/import paths
- `tests/agent/test_coding_context.py`
- new `tests/agent/test_session_capabilities.py`
- session lineage/resume tests under `tests/tui_gateway/`

**Work**

- Define versioned capability-plan dataclass/schema.
- Route exactly once before first agent construction.
- Persist plan and inherit it across compression forks; do not inherit it blindly into delegated children.
- Add config flags under `tools.intent_routing` with safe defaults and a complete rollback switch.
- Log only route label, confidence, selected IDs, manifest hash, and timing.

**Exit criteria**

- Router p95 <100 ms on the benchmark host without network calls.
- Identical intent + environment produces byte-identical plans.
- Resume produces the same system-prompt hash and tool-schema hash.
- Compression lineage retains the plan; delegates respect parent toolset bounds.
- Feature-off mode is behaviorally identical to current Hermes.

### Phase 2 — Deferred built-ins and direct-tool bundles

**Files**

- `toolsets.py`
- `model_tools.py`
- `tools/tool_search.py`
- `tools/registry.py`
- `tests/tools/test_tool_search*.py`
- `tests/tools/test_startup_latency_regressions.py`
- `evals/tool_search/`

**Work**

- Separate `allowed_tools` from `direct_tool_defs`.
- Make nonselected, authorized built-ins discoverable/callable through the existing bridge.
- Keep platform/session gates and approvals enforced at bridge execution.
- Define initial bundles: general, software, research/web, files/documents, communications, scheduling, data/BI, media/creative, smart-home, and operations/admin.
- Keep a conservative bundle for low-confidence routes.

**Exit criteria**

- Typical focused sessions have ≤6k tool-schema tokens and ≤15 direct tools.
- General fallback stays ≤8k tool-schema tokens.
- Nonselected authorized tools remain reachable with zero prompt mutation.
- Unauthorized or platform-incompatible tools are neither described nor callable.
- ≥95% correct first-bundle selection on the benchmark corpus and 100% eventual task capability through fallback.
- System prompt and tool definitions remain byte-stable after turn 1.

### Phase 3 — Automatic skill routing

**Files**

- `agent/prompt_builder.py`
- skill discovery/index modules used by `skills_list`/`skill_view`
- profile/plugin invalidation paths
- `tests/agent/test_prompt_builder.py`
- skill isolation/profile tests

**Work**

- Build a cached compact manifest.
- Rank skills using the same first-intent signals.
- Render names-only catalog + top descriptions + at most three full skill bodies.
- Preserve exact profile/plugin isolation and install/update invalidation semantics.

**Exit criteria**

- Default skill section ≤1k tokens before any selected full bodies.
- Relevant skill is in top 3 for ≥95% of benchmark tasks.
- No cross-profile skill leakage.
- No filesystem rescan on each turn.
- Existing explicit `skill_view` workflow remains functional.

### Phase 4 — Remote hydration and Desktop resilience

**Files**

- `tui_gateway/methods_session.py`
- `hermes_cli/web_routers/sessions.py`
- `hermes_cli/web_routers/profiles.py`
- `apps/shared/` JSON-RPC/transport client
- `apps/desktop/src/store/session*.ts`
- transcript-tail cache and request-router tests

**Work**

- Require/verify deferred-history negotiation for compatible clients.
- Paint the latest page first; hydrate older pages in the background with cancellation-safe request identity.
- Preserve hydration when the user switches tabs; cache completion by connection/profile/session key.
- Add direct/relay and RTT diagnostics where the client can access Tailscale state; otherwise label it “unknown,” never infer.
- Bound retries and expose the failing stage instead of an infinite spinner.

**Exit criteria**

- Session list p95 <500 ms on the measured remote connection.
- Opening a long session paints its latest page p95 <1.5 s, independent of full transcript size.
- Switching tabs during hydration does not restart or lose progress.
- WebSocket handshake p95 and reconnect behavior are visible.
- No full transcript is transferred before first paint.

### Phase 5 — Long-session continuity and optional retrieval

**Files**

- existing compression and context-engine plugin interfaces
- `agent/prompt_builder.py`
- session lineage/state tests
- compaction evals

**Work**

- Define a structured continuity block for goals, tasks, decisions, approvals, identifiers, and verified state.
- Evaluate optional context-engine retrieval against raw long-context baseline; do not enable by default until it improves recall without changing facts.
- Keep raw context capacity unchanged and preserve provider prompt-cache boundaries.

**Exit criteria**

- No reduction in configured context window.
- Long-running benchmark task maintains ≥95% fact/constraint recall after compression or retrieval.
- Zero lost unresolved tasks, approval gates, identifiers, or branch/file state in the benchmark.
- Long-session p95 latency improves without a statistically significant quality regression.

### Phase 6 — Canary, rollout, and rollback

**Work**

- Ship behind config flags to one profile/session class first.
- Compare baseline vs routed mode for at least 100 representative turns.
- Roll out 10% → 50% → 100% only if all quality, security, and latency gates pass.
- Retain one-command rollback to static toolsets/legacy skill prompt.

**Primary success metrics**

- Fresh simple request first token: p50 ≤5 s, p95 ≤10 s on the existing provider.
- Fixed non-conversation prompt: typical ≤10k tokens, down from about 22k.
- Direct tool schema: typical ≤6k tokens, down from about 14.5k.
- Capability-router overhead: p95 <100 ms.
- Tool/skill first-choice accuracy: ≥95%.
- Eventual capability success through bridge: 100% for authorized benchmark tasks.
- Prompt-cache invariant: system prompt/tool hash unchanged after session start.
- Long-context capacity: unchanged.
- Remote long-session latest-page paint: p95 <1.5 s.

**Hard rollback triggers**

- any authorization bypass through deferred tool calls
- prompt/tool hash mutation mid-session
- cross-profile skill/tool leakage
- >2% task-success regression
- loss of durable task/approval state
- session-open regression >20%

## Test strategy

### Unit and contract tests

- deterministic routing and stable ordering
- low-confidence fallback
- platform-specific tool gating
- profile/plugin skill isolation
- authorized deferred built-in discovery and call
- denied tool invisibility and call rejection
- plan persistence, portability, compression inheritance, and delegate bounds
- stable system-prompt/tool hashes across turns and resume

### Integration tests

Run real imports against a temporary `HERMES_HOME`:

- new Desktop session → first intent → persisted plan → agent call
- remote resume with deferred transcript → latest page → background history
- tool absent from direct schemas → search/describe/call → successful execution
- task shift mid-session → bridge use without prompt rebuild
- skill install during a session → current prompt unchanged; next session sees update
- compression fork → same capability plan and stable policy prefix

### End-to-end benchmark

For every corpus task, record:

- route and confidence
- selected direct tools and preloaded skills
- prompt component tokens
- cache read/write tokens
- time to agent-ready, provider request, first token, and completion
- tool search hops
- task success and policy violations

Compare feature-off and feature-on against the same provider/model and warm/cold conditions. Use medians and p95, not a single demonstration.

## Configuration proposal

```yaml
tools:
  intent_routing:
    enabled: false          # canary first
    mode: deterministic
    max_direct_tools: 15
    low_confidence_bundle: general
    expose_deferred_builtins: true
skills:
  intent_routing:
    enabled: false          # canary first
    max_preloaded: 3
    compact_catalog: true
remote:
  diagnostics:
    stage_timings: true
```

These are behavioral settings in `config.yaml`, not environment variables. Secrets remain in `.env`.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Router picks the wrong bundle | Conservative fallback plus generic bridge; evaluate first-choice accuracy before rollout. |
| Dynamic selection breaks prompt cache | Route once, persist, and assert byte-stable hashes on every call. |
| Bridge bypasses policy | Execute through the existing registry and authorization/check/approval chain; add negative security tests. |
| Full skill preload recreates bloat | Hard cap of three; preload only above confidence threshold. |
| Task changes mid-session | Use deferred bridge without schema mutation; optional user-visible focused-session fork. |
| Version skew creates false performance conclusions | Version/capability handshake is phase 0 and a rollout gate. |
| Retrieval loses critical details | Keep raw capacity; retrieval remains optional until recall benchmarks pass. |
| Profile cleanup breaks agents | Do not delete profiles based on current measurements. Optimize only measured endpoints. |

## Recommended order

Implement phases 0–3 first. They address the proven fixed prompt cost while preserving context capacity and prompt caching. Run phase 4 in parallel only after exact Desktop/backend version alignment. Treat phase 5 as an evaluated enhancement, not a prerequisite.

The key principle is: **reduce irrelevant capability surface, not memory capacity**.
