# Hermes Memory Roadmap

> For Hermes: use subagent-driven-development when executing implementation phases.

Goal: build a local-first memory stack for Hermes in the right order: stronger builtin memory, better memory search, a conservative dreaming subsystem, then optional active memory.

Architecture: keep built-in Hermes memory as canonical, keep Obsidian as the human-readable external workspace, and add new memory capabilities as adjunct local services instead of replacing the active provider. Prioritize reliability, dedupe, bounded outputs, and low-regret local storage patterns over speculative memory complexity.

Tech stack: Python, Hermes built-in memory tooling, SessionDB/SQLite, local JSON/Markdown state, Obsidian provider, cron/session-end hooks.

---

## Why this order

1. Builtin memory engine first
   - source of truth must improve before retrieval and consolidation are layered on top
2. Memory search second
   - retrieval quality gives immediate value and becomes the base for active memory later
3. Dreaming third
   - consolidation improves quality only after storage and retrieval are trustworthy
4. Active memory fourth
   - pre-reply recall should only run once the recall stack is good enough not to add noise
5. Advanced memory engines later
   - Honcho/QMD-class systems only after clear evidence the simpler local stack is insufficient

---

## Current baseline

Active configuration today:
- built-in Hermes memory is enabled
- external memory provider is `obsidian`
- Obsidian workspace path is local
- no mem0 provider is configured
- no MCP servers are configured

Relevant current architecture:
- Canonical durable memory:
  - `tools/memory_tool.py`
  - `~/.hermes/memories/MEMORY.md`
  - `~/.hermes/memories/USER.md`
- Session/transcript persistence:
  - `hermes_state.py`
  - `gateway/session.py`
  - `~/.hermes/state.db`
  - `~/.hermes/sessions/*.jsonl`
- External memory provider:
  - `plugins/memory/obsidian/__init__.py`

Core rule going forward:
- `MEMORY.md` and `USER.md` remain canonical
- Obsidian remains a bounded external workspace and review surface
- new systems should read from both, but write conservatively into canonical memory

---

## Phase 1 — Strengthen builtin memory engine

### Objective
Improve promotion quality, dedupe, and memory hygiene without changing the current local-first model.

### Deliverables
- stronger normalization/deduplication for memory writes
- confidence-aware promotion path
- fewer noisy or repetitive entries
- clearer distinction between durable facts and ephemeral task chatter

### Files to inspect/modify
- `tools/memory_tool.py`
- `run_agent.py`
- `agent/builtin_memory_provider.py`
- possibly `agent/memory_manager.py` only if shared helper placement makes sense

### Tasks

#### Task 1: Audit current memory write path
Objective: document exactly how memory writes are parsed, validated, and persisted today.

Files:
- Read: `tools/memory_tool.py`
- Read: `run_agent.py`
- Write: `docs/plans/memory-audit-notes.md` (optional scratch note)

Verification:
- identify where add/replace/remove happen
- identify current limits and duplicate-prevention behavior

#### Task 2: Add normalization helpers
Objective: normalize whitespace, punctuation noise, and repeated near-identical memory strings before persistence.

Files:
- Modify: `tools/memory_tool.py`
- Test: add/update tests for memory normalization

Verification:
- logically equivalent memory strings collapse to one canonical representation

#### Task 3: Add stronger duplicate detection
Objective: prevent trivial rewrites of the same memory from bloating MEMORY.md/USER.md.

Files:
- Modify: `tools/memory_tool.py`
- Test: memory-tool tests

Verification:
- duplicate and near-duplicate writes are rejected or merged predictably

#### Task 4: Add memory quality guardrails
Objective: reject clearly ephemeral content from durable memory writes.

Examples to block:
- temporary task progress
- one-off success logs
- verbose raw tool output

Files:
- Modify: `tools/memory_tool.py`
- Test: memory-tool tests

Verification:
- ephemeral text is blocked with a clear reason
- stable preferences/facts still pass

#### Task 5: Add confidence/reason metadata internally
Objective: allow later systems to understand why a memory was promoted without exposing noisy metadata in the user-facing files.

Files:
- Modify: local structured state alongside memory writes, if needed under `~/.hermes/memories/` or `~/.hermes/state.db`

Verification:
- memory writes can optionally record source/reason metadata locally
- canonical Markdown remains clean and compact

### Success criteria for Phase 1
- built-in memory remains local and simple
- duplicate junk is materially reduced
- durable facts are easier to promote correctly
- no cloud dependencies introduced

---

## Phase 2 — Add local memory search

### Objective
Give Hermes reliable local memory retrieval across durable memory and recent sessions.

### Deliverables
- local search API/helper for durable memory + recent session context
- compact recall blocks suitable for prompts
- query modes such as `recent`, `durable`, `full`

### Files to inspect/modify
- `hermes_state.py`
- `tools/memory_tool.py`
- `run_agent.py`
- possibly new module: `agent/memory_search.py`

### Design guidance
Create an adjunct retrieval module, not a new external provider.

Data sources:
- `MEMORY.md`
- `USER.md`
- `state.db`
- session JSONL fallback
- optional bounded Obsidian notes later

Output shape:
- compact, deduped, prompt-safe snippets
- not raw transcript dumps

### Tasks

#### Task 1: Define search abstraction
Create:
- `agent/memory_search.py`

Methods should include:
- search_durable_memory(query)
- search_recent_sessions(query)
- build_recall_context(query, mode='recent')

#### Task 2: Implement durable memory search
Objective: search canonical memory files first.

Files:
- `agent/memory_search.py`

Verification:
- returns matching durable facts from MEMORY.md/USER.md

#### Task 3: Implement recent session search
Objective: search SessionDB / JSONL locally.

Files:
- `agent/memory_search.py`
- possibly `hermes_state.py` helper use

Verification:
- recent relevant session snippets can be retrieved without loading entire transcripts

#### Task 4: Add dedupe and ranking
Objective: merge overlapping hits and rank by relevance + recency.

Verification:
- output contains concise non-repetitive recall candidates

#### Task 5: Add prompt integration hook
Objective: allow Hermes to inject bounded recall context before response generation when useful.

Files:
- `run_agent.py`

Verification:
- memory search can be called without destabilizing normal responses
- bounded size limits are enforced

### Success criteria for Phase 2
- local search works across durable memory and recent sessions
- recall quality is visibly better than raw keyword search alone
- system remains local-first and privacy-preserving

---

## Phase 3 — Add conservative dreaming MVP

### Objective
Build a local-only consolidation layer that turns repeated/relevant recent signals into reviewable summaries and a tiny number of durable promotions.

### Important rule
Do not implement dreaming as the active external memory provider.
It should be an adjunct local service so it can coexist with Obsidian.

### Deliverables
- local dream state directory
- session-end or scheduled synthesis pass
- bounded `dream-review.md` output in Obsidian
- conservative promotion into `MEMORY.md` / `USER.md`

### Recommended file layout
Create:
- `agent/dreaming.py`
- optional later: `agent/dreaming/`

Local state:
- `~/.hermes/dreams/state.json`
- `~/.hermes/dreams/dreams.jsonl` or `journal.md`

Obsidian output:
- `Hermes/dream-review.md`

### What dreaming should write
Allowed:
- repeated durable preferences
- repeated project/environment facts
- recurring open loops
- compact synthesis of recent themes

Disallowed:
- raw transcripts
- secrets
- speculative psychological summaries
- one-off task logs
- noisy tool chatter

### Promotion policy
Promote only if one of these is true:
- repeated across 2+ sessions
- explicit user durable preference/fact
- stable environment/project convention
- clear long-term relevance

Hard cap:
- 0–3 promotions per dream pass

### Tasks

#### Task 1: Define dream state schema
Create:
- `agent/dreaming.py`

State should track:
- last processed session/message cursor
- dedupe hashes
- last run timestamps
- promoted item ids/hashes

#### Task 2: Implement recent transcript collection
Objective: collect bounded recent material from SessionDB/JSONL.

Verification:
- gathers only recent and relevant slices
- does not load entire history unnecessarily

#### Task 3: Implement candidate extraction
Objective: extract candidate durable facts/open loops from recent sessions.

Verification:
- candidate set is compact and typed

#### Task 4: Implement dream review output
Objective: write a bounded local summary and Obsidian review note.

Outputs:
- local dream artifact
- `Hermes/dream-review.md`

Verification:
- note stays readable and bounded
- no transcript dump behavior

#### Task 5: Implement conservative promotion
Objective: promote only high-confidence items into built-in memory.

Files:
- `agent/dreaming.py`
- `tools/memory_tool.py` integration if needed

Verification:
- promotions are few, explainable, and deduped

#### Task 6: Add trigger points
Objective: run dreaming on session end and/or nightly.

Files:
- `run_agent.py`
- `cron/scheduler.py` if nightly path is chosen

Verification:
- dreaming runs locally and does not block ordinary chats excessively

### Success criteria for Phase 3
- dream outputs are useful and bounded
- durable memory quality improves without memory spam
- Obsidian becomes a better review surface without replacing canonical memory

---

## Phase 4 — Add optional active memory

### Objective
Add a bounded pre-reply recall pass that surfaces relevant local memory before the main response.

### Deliverables
- optional active-memory pass
- strict latency budget
- limited to appropriate session types at first

### Core principle
Active memory should sit on top of the Phase 2 memory search system.
Do not build it before retrieval is trustworthy.

### Suggested behavior
- run one bounded pre-reply recall step
- return a compact memory context block
- no free-form agentic wandering
- no transcript floods

### Suggested initial scope
- direct conversational sessions only
- strict timeout budget
- logging enabled during tuning only
- opt-in config flag

### Tasks

#### Task 1: Create active memory module
Suggested file:
- `agent/active_memory.py`

#### Task 2: Define query modes
- `recent`
- `durable`
- `full`

#### Task 3: Add bounded prompt styles
- balanced
- terse
- maybe analytical later

#### Task 4: Integrate pre-reply hook
Files:
- `run_agent.py`

Verification:
- active memory adds bounded context before the main response only when enabled

#### Task 5: Add config and logging
Files:
- config handling modules

Verification:
- feature is off by default
- logs expose when/why it ran

### Success criteria for Phase 4
- relevant memory appears naturally before reply generation
- latency remains acceptable
- false-positive noisy recalls are rare

---

## Phase 5 — Re-evaluate advanced memory systems

### Objective
Only after the local stack proves itself, evaluate whether more advanced memory engines are actually necessary.

Candidates to consider later
- Honcho-style richer memory/graph workflows
- QMD-style alternate memory engine
- advanced vector/semantic stores
- multi-agent shared memory layers

### Evaluation trigger
Only enter this phase if Phase 1–4 show real bottlenecks such as:
- recall quality plateaus
- cross-session synthesis is still weak
- graph relations become necessary
- team/shared memory becomes a core requirement

### Default decision today
Do not build this yet.

---

## Key architectural rules

1. Local-first by default
- no cloud memory backend required
- session history, memory, and dreaming state stay local

2. Canonical memory stays small
- `MEMORY.md` and `USER.md` should remain high-signal and compact

3. Obsidian is a review/workspace layer
- useful for structured notes and synthesis
- not the canonical durable fact source

4. Dreaming is adjunct, not provider
- do not consume the external memory provider slot with dreaming

5. Retrieval before pre-reply automation
- build memory search before active memory

6. Conservative promotion beats clever overfitting
- avoid memory spam
- prioritize trustworthiness over coverage

---

## Risks and mitigations

Risk: memory spam
- Mitigation: hard promotion caps, dedupe, bounded outputs

Risk: Obsidian becomes a junk drawer
- Mitigation: fixed note set, bounded dream-review, overwrite snapshots instead of append-only logs

Risk: active memory adds latency/noise
- Mitigation: strict timeout, opt-in, direct sessions only initially

Risk: architecture drift from too many memory engines
- Mitigation: keep builtin canonical, one external provider, adjunct services for dreaming/search

Risk: future cloud temptation complicates privacy model
- Mitigation: keep local-only default and evaluate cloud backends only as optional later plugins

---

## Immediate next step

Build target to start with now:
- Phase 1 + the foundation of Phase 2

Recommended first implementation slice:
1. strengthen builtin memory write normalization/dedupe
2. add `agent/memory_search.py`
3. support local search over:
   - MEMORY.md
   - USER.md
   - recent session history

Why this slice first
- highest practical value
- lowest architectural risk
- directly supports both dreaming and active memory later

---

## Ship recommendation

Do now
- Phase 1
- early Phase 2

Do next
- Phase 3 dreaming MVP

Do after that
- Phase 4 active memory

Defer
- Honcho/QMD-class advanced memory systems

---

## Definition of success

Hermes should end up with:
- cleaner canonical memory
- better local recall
- bounded local synthesis
- optional pre-reply memory that actually helps
- no requirement for third-party cloud memory providers
