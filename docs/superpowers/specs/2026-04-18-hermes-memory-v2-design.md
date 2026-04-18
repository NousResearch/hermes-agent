# Hermes Memory V2 Design

Date: 2026-04-18

## Goal

Hermes Memory V2 should make the agent more consistent across sessions, more accurate about durable facts, better at recalling successful prior work, and safer against stale or low-trust memory, without requiring a wholesale rewrite of the existing memory provider system.

The design should unify the way Hermes thinks about memory while keeping the first implementation slice conservative, measurable, and backward-compatible.

## Context

Hermes already has several memory-related capabilities:

- built-in curated prompt memory via `tools/memory_tool.py`
- transcript recall via `tools/session_search_tool.py`
- external provider orchestration via `agent/memory_manager.py`
- one-at-a-time external memory provider plugins via `agent/memory_provider.py` and `plugins/memory/*`

The current system works, but it has four structural gaps:

1. It does not have a single internal memory model shared across built-in memory, transcript recall, and provider integrations.
2. It does not treat freshness, contradiction, trust, capacity, and write discipline as first-class concerns.
3. It does not have a dedicated execution-memory layer for remembering what worked, what failed, and what is reusable for future tasks.
4. It does not specify enough policy behaviour to ensure consistent implementation across phases.

Memory V2 addresses those gaps as a policy-first evolution of the current stack.

## Non-Goals

The following are explicitly out of scope for Memory V2 phase 1:

- replacing all external providers with one new backend
- forcing every provider to adopt a shared physical storage format
- introducing a mandatory graph database
- storing every turn as long-term memory
- treating transcript history as canonical truth automatically
- building a full multi-agent shared-memory system
- removing the current built-in `memory` tool or `session_search` tool
- learning policy weights online from production traffic

## Design Principles

1. Policy before storage.
2. Better memory, not merely more memory.
3. Small prompt-resident memory, larger retrieval-based memory.
4. Explicit freshness, conflict, trust, and capacity handling.
5. Backward-compatible phases with useful intermediate milestones.
6. Evidence before claims: every phase must have baselines, methodology, and rollout gates.
7. The operator must be able to inspect why something was remembered, retrieved, superseded, or ignored.

## High-Level Architecture

Memory V2 is organized as four memory layers governed by one shared policy layer.

### 1. Profile Memory

Purpose: preserve small, high-trust, always-important facts.

Examples:

- user preferences
- operator communication defaults
- stable environment facts
- project conventions
- durable workflow preferences

Properties:

- small and curated
- suitable for prompt injection at session start
- schema-aware internally even if legacy file storage remains human-readable
- difficult to mutate casually
- bounded by strict capacity limits

This is the evolution of the current `MEMORY.md` and `USER.md` model.

### 2. Semantic Memory

Purpose: store durable facts that should be retrievable on demand but are not always worth injecting into the prompt.

Examples:

- project structure facts
- tool quirks and workarounds
- durable conclusions from prior work
- provider-discovered facts with provenance
- mirrored durable records from external providers

Properties:

- metadata-rich
- retrieval-oriented
- can be active, stale, superseded, disputed, expired, or archived
- not injected by default
- compacted and deduplicated over time

### 3. Episodic Memory

Purpose: remember specific prior situations that can serve as reusable task examples.

Examples:

- successful debugging sequences
- deployment recovery procedures
- cutover checklists that worked
- repeated remediation paths with strong outcomes

Properties:

- task-shaped rather than fact-shaped
- outcome-labelled where possible
- retrieved only on strong task similarity
- introduced conservatively because poor episodes can teach bad habits
- subject to tighter validation and shorter lifetime than semantic memory

### 4. Transcript Recall

Purpose: preserve broad searchable historical recall across all sessions.

Examples:

- “What did we do about X last month?”
- “Did we already discuss Y?”
- “What happened during that earlier incident?”

Properties:

- large-capacity
- cheap to append
- slower and more selective to query
- useful for historical lookup, not canonical truth without validation

This is conceptually the current `session_search` layer.

### 5. Memory Policy Layer

The policy layer is the centre of Memory V2.

It decides:

- what gets written
- where it gets written
- what trust level it receives
- when it should be retrieved
- how it is ranked
- when it should be demoted, marked stale, superseded, archived, or ignored
- how capacity pressure is relieved
- how race conditions between foreground writes and background consolidation are resolved

Unlike the layer definitions above, the policy layer is not just a concept. Memory V2 requires default decision mechanisms for write classification, ranking, freshness transitions, contradiction handling, and consolidation triggers. Those defaults are defined in this document and must be implemented before phase 1 can be considered complete.

## Supported Scope Model

Memory V2 phase 1 is designed for Hermes's current single-operator reality.

Supported scopes in phase 1:

- `operator`: durable facts about the operator independent of runtime profile
- `profile`: facts specific to a Hermes profile or persona
- `workspace`: facts specific to a repository, project directory, or operational workspace
- `session`: facts and episodes specific to the active session

Not supported in phase 1:

- multi-tenant user scope
- unconstrained global scope spanning unrelated operators

This avoids designing a multi-user scope model that the current product does not yet need.

### Scope Guidance

Phase 1 scope selection follows these defaults:

- use `operator` for preferences or facts that should remain true across personas and profiles
- use `profile` for persona-specific tone, workflow, or behavioural defaults
- use `workspace` for repository, deployment, or project-specific facts
- use `session` for short-lived task or incident context that should not be promoted automatically

Example:

- “Use British spelling” is normally `operator`
- “J.A.R.V.I.S. should use restrained butler cadence” is `profile`
- “This repo deploys via `make ship`” is `workspace`

## Common Internal Record Model

Memory V2 defines a common internal record shape used by built-in memory and, where practical, by provider adapters.

Suggested fields:

- `record_id`
- `memory_type` (`profile`, `semantic`, `episodic`, `transcript_reference`)
- `scope` (`operator`, `profile`, `workspace`, `session`)
- `topic_key`
- `content`
- `summary`
- `source`
- `source_kind` (`explicit_user_statement`, `tool_observation`, `provider_sync`, `transcript_extraction`, `model_inference`)
- `created_at`
- `last_confirmed_at`
- `last_used_at`
- `review_after`
- `expires_at`
- `trust_tier`
- `salience_tier`
- `status` (`active`, `stale`, `superseded`, `disputed`, `expired`, `archived`)
- `supersedes`
- `conflicts_with`
- `tags`
- `metadata`
- `revision`

### Field Semantics and Constraints

#### `trust_tier`

`trust_tier` is ordinal, not continuous. Phase 1 uses the following fixed tiers:

1. `user_asserted`
2. `observed`
3. `user_approved`
4. `inferred`
5. `unverified`

Internally these may be mapped to monotonic integers for ranking, but the canonical meaning is tier-based. Phase 1 does not expose floating-point trust scores.

#### `salience_tier`

`salience_tier` is also ordinal. Phase 1 uses:

- `critical`
- `high`
- `medium`
- `low`

These tiers are assigned by policy based on write class, explicit user emphasis, recurrence, and demonstrated reuse.

#### `summary`

`summary` is optional. `content` remains the canonical payload.

- Foreground writes may create records without a summary.
- Background consolidation may add or improve summaries later.
- Ranking and retrieval must never require `summary` to exist.
- If `summary` exists and disagrees with `content`, `content` wins and the summary is regenerated or discarded.

#### `topic_key`

`topic_key` is required for contradiction handling and bounded supersession.

Phase 1 definition:

- `topic_key` is a normalized subject identifier for what fact this record is about
- contradiction resolution and supersession are only attempted when `scope` and `topic_key` both match
- `topic_key` should be deterministic and operator-inspectable, not inferred opaquely at retrieval time

Phase 1 construction rules:

- for explicit preferences, use a normalized preference subject such as `preference:response-detail`
- for environment facts, use a normalized subject such as `env:os`, `env:shell`, or `workspace:deploy-command`
- for episodic records, use the episode's `task_signature` rather than a generic fact topic
- if no stable `topic_key` can be formed, the record may still be stored, but it is not eligible for automatic contradiction resolution

This keeps contradiction handling deterministic enough for phase 1 and avoids relying on embedding similarity to decide whether two facts are about the same thing.

#### `supersedes` and `conflicts_with`

These fields create bounded relations, not an arbitrary graph.

Phase 1 rules:

- `supersedes` may reference at most one immediate predecessor record.
- retrieval walks at most one predecessor hop; it does not traverse unbounded chains.
- if a record supersedes a record that itself supersedes another, background consolidation compacts the chain to a single surviving active predecessor relation.
- `conflicts_with` may reference multiple records, but conflict resolution always produces one active preferred record per scope+topic.
- circular `conflicts_with` links are allowed for traceability; circular `supersedes` links are forbidden and must be rejected.

This preserves traceability without turning phase 1 into a graph database project.

## Capacity and Retention Policy

Memory V2 must prevent unbounded growth.

### Phase 1 Capacity Rules

- prompt-resident profile memory remains strictly bounded and compatible with current char-limited behaviour
- semantic memory has per-scope record count limits and compaction thresholds
- episodic memory has the smallest capacity and shortest retention horizon
- transcript recall remains append-heavy, but retrieval payloads stay bounded by query-time limits

### Default Retention Behaviour

- `critical` and `high` profile records are retained unless explicitly superseded or removed
- low-salience semantic records are first candidates for archive under capacity pressure
- episodic records expire or are archived faster than semantic facts unless revalidated by reuse
- transcript recall retains history according to existing session storage behaviour and separate transcript policies

### Eviction and Archive Order

When capacity pressure forces reduction:

1. archive low-salience expired records
2. archive low-salience stale records with no recent use
3. compact superseded chains
4. compress semantically duplicate low-value records
5. never automatically delete active critical profile records

Phase 1 prefers archival over irreversible deletion.

## Policy Mechanisms

Memory V2 requires explicit default mechanisms. These mechanisms are deliberately simple and deterministic enough to implement in phase 1.

### Write Classification Mechanism

Write classification is hybrid:

- first pass: deterministic rule-based classifier
- second pass: optional model-assisted refinement for ambiguous `may_write` candidates only
- final gate: policy validation using trust, scope, and capacity rules

Phase 1 must not allow the model to self-approve writes without policy validation.

#### Default Write Pipeline

```text
candidate event
  -> normalize candidate
  -> rule-based classification
  -> if class in {must_write, do_not_write}: stop classification
  -> if class == should_write or may_write and ambiguity flag is set:
       optional model-assisted justification
       - may confirm current class
       - may demote one class level
       - may promote by at most one class level
       - may never promote to must_write
  -> policy validation
       - trust tier assignment
       - scope assignment
       - topic_key assignment if possible
       - capacity check
       - safety scan
  -> foreground write or background queue
```

#### Rule-Based Classification Defaults

A candidate is `must_write` if any of the following are true:

- user explicitly says to remember it
- user explicitly corrects a stored or recently asserted durable fact
- system discovers a stable preference that immediately affects future interaction and is confirmed by the user

A candidate is `should_write` if all are true:

- expected utility extends beyond the current session
- evidence is `user_asserted`, `observed`, or `user_approved`
- the fact or episode is scoped and not transient

A candidate is `may_write` if:

- utility is plausible but unconfirmed
- evidence is weaker or inferred
- the item is not urgent enough for foreground persistence

A candidate is `do_not_write` if any are true:

- transient planning chatter
- raw reasoning traces
- unverified high-risk claims
- noisy or repetitive conversational filler
- unsafe or adversarial-looking content

### Freshness Transition Mechanism

Freshness transitions are deterministic and event-driven, with optional background review.

#### State Transition Rules

- `active` -> `stale` when `review_after` has passed and the record has not been reconfirmed
- `stale` -> `active` when the record is explicitly reconfirmed or reliably reused without contradiction
- `active` or `stale` -> `superseded` when a higher-trust or newer correction replaces the same scoped fact
- `active` or `stale` -> `disputed` when a conflicting record exists and policy cannot determine a winner automatically
- `stale` -> `expired` when both age threshold and non-use threshold are exceeded
- `expired` -> `archived` during consolidation when capacity pressure exists or when archival policy is configured

#### Freshness Review Timing

Freshness review runs:

- opportunistically during reads of candidate records
- during session end consolidation
- during scheduled or threshold-based background consolidation

Phase 1 does not require a dedicated daemon. Reviews may be piggybacked onto existing session-end and retrieval workflows.

### Contradiction Resolution Mechanism

Contradiction handling is policy-driven and topic-scoped.

#### Resolution Decision Tree

```text
new record conflicts with existing active record
  -> same scope and same topic?
       no -> coexist
       yes -> compare trust tier
            higher trust wins -> supersede lower-trust record
            equal trust -> compare recency and explicitness
                 newer explicit correction wins -> supersede
                 scoped refinement detected -> keep both, narrow scope on newer record
                 unresolved -> mark both disputed and prefer none by default
```

#### Phase 1 Constraints

- automatic supersession is only allowed when the newer record is at least as trustworthy and more specific or more recent
- conditional preference updates are modeled as refinements, not destructive replacement
- disputed records receive heavy ranking penalties and are not surfaced automatically unless the query explicitly asks about the contradiction

### Ranking Mechanism

Phase 1 ranking uses fixed weighted components rather than learned weights.

Default ranking order of importance:

1. scope match
2. trust tier
3. status penalty
4. semantic relevance
5. recency
6. salience tier
7. task-type similarity for episodes

Implementation may compute a numeric score internally, but the default composition must preserve the priority ordering above. In other words, a strongly relevant but disputed low-trust record must not outrank a slightly less relevant active observed record in the correct scope.

## Write Policy

Memory V2 should not store content simply because it appeared in conversation. A write should occur only when the information is likely to remain useful beyond the current turn or session.

### Good Write Candidates

- explicit user preferences
- stable identity facts
- durable environment and project conventions
- reusable successful resolutions
- high-value task episodes
- explicit user corrections to previous assumptions
- tool-discovered facts with clear provenance

### Poor Automatic Write Candidates

- temporary planning chatter
- speculative inferences
- raw reasoning traces
- one-off summaries with no lasting value
- transient errors without reusable lesson value
- unverified claims that may be stale, adversarial, or accidental

### Write Classes

Memory V2 classifies write opportunities as:

- `must_write`: explicit user correction, explicit “remember this”, critical durable preference
- `should_write`: strong-evidence durable fact, reusable successful episode
- `may_write`: potentially useful but uncertain; queue for background consolidation or confirmation
- `do_not_write`: ephemeral, unsafe, or low-value content

### Enforcement and Rate Limits

Write policy must be enforced by code, not intention.

Phase 1 enforcement rules:

- foreground path writes at most a small bounded number of records per turn
- if more than the bound qualify, the system keeps highest-priority `must_write` and highest-salience `should_write` candidates, deferring the rest
- `may_write` candidates are never written synchronously
- repeated writes of materially identical content are deduplicated before persistence

This prevents write storms in long sessions.

## Retrieval Policy

Retrieval should be demand-shaped and budgeted, not maximal.

### Layer Retrieval Order

For normal task execution, retrieval should prefer layers in this order:

1. profile memory
2. semantic memory
3. episodic memory
4. transcript recall only when the task or prompt signals historical lookup

### Retrieval Budgets

Phase 1 budgets are explicit and conservative.

- profile memory: bounded, prompt-resident, always small
- semantic memory: small result set per query
- episodic memory: zero or one example by default, at most two under explicit strong task similarity
- transcript recall: on demand only, bounded by current `session_search` truncation rules

The exact numeric defaults belong in implementation planning, but phase 1 must preserve the ordering principle above and must ship with explicit defaults rather than leaving budgets undefined.

### Ranking Factors

Retrieval ranking combines:

- semantic relevance
- scope match
- recency
- salience tier
- trust tier
- status penalties
- task-type similarity for episodic memory

A slightly less similar but fresh and trustworthy record should outrank an older or disputed record.

## Episodic Memory Specification

Episodic memory is the highest-risk layer in Memory V2 and therefore requires a concrete schema and validation protocol.

### Episode Schema

Each episode record contains:

- `episode_id`
- `task_signature`
- `scope`
- `problem_summary`
- `approach_summary`
- `key_actions`
- `tools_used`
- `outcome`
- `outcome_evidence`
- `failure_notes`
- `created_at`
- `last_used_at`
- `validation_status`
- `reuse_count`
- `source_session_id`

`key_actions` is structured and compact. Phase 1 episodes are not full transcripts and are not free-form chain-of-thought dumps.

### Episode Validation Protocol

An episode may become `validated` only if at least one of the following is true:

- the user explicitly confirmed the approach worked
- a concrete verification command or observable success signal confirmed the outcome
- the same approach succeeded in materially similar contexts more than once

An episode remains `candidate` if it appears promising but lacks outcome evidence.

Only `validated` episodes are eligible for default retrieval.

For phase 1, "materially similar contexts" means the following all hold:

- the `task_signature` matches or normalizes to the same task family
- the tool pattern is the same or equivalent
- the workspace or environment constraints are not known to conflict
- the outcome class is the same kind of success, not merely a superficial resemblance in wording

Phase 1 does not require embedding-only similarity to validate episodes.

### Episode Retrieval Safeguards

- default retrieval cap: one episode
- maximum retrieval cap in phase 1: two episodes only under strong task match
- episodes with high reuse but degraded outcomes are demoted or retired
- failed or ambiguous episodes are never used as default few-shot context

## Freshness Handling

Memory V2 should assume that some facts decay.

Each durable record should carry freshness metadata such as:

- `created_at`
- `last_confirmed_at`
- `last_used_at`
- `review_after`
- `expires_at`

`stale` does not mean false. It means Hermes should apply caution and ranking penalties unless the query strongly indicates that the record is still relevant.

Freshness is especially important for:

- user preferences
- file paths
- deployment state
- credentials and provider configuration
- machine-specific environment facts
- operational instructions that may have changed since the last confirmed use

## Contradiction Handling

Memory V2 should not silently overwrite conflicting information.

Instead, it should support:

- replacement when a record is clearly corrected
- refinement when a preference or rule becomes scoped or conditional
- coexistence when both records are context-dependent and valid
- dispute state when the conflict cannot be resolved automatically

When a newer, higher-trust record supersedes an older one:

- the old record remains traceable
- the relationship is recorded via `supersedes` or `conflicts_with`
- retrieval prefers the newer, active record

This is particularly important for user preferences and environment facts, which often evolve rather than simply disappear.

## Safety and Threat Model

Every memory record should know where it came from.

### Trust Ordering

Recommended trust ordering:

1. explicit user statement
2. confirmed tool or environment observation
3. user-approved summary or conclusion
4. model inference supported by repeated evidence
5. unconfirmed extracted summary

Trust affects both write thresholds and retrieval ranking.

### Adversarial Memory Poisoning

Memory V2 explicitly treats poisoning as a threat.

Poisoning candidates include:

- prompt-injected instructions disguised as facts
- malicious copied text from tools or web pages
- adversarial user instructions framed as durable preferences
- low-trust summaries that attempt to smuggle behavioural override rules into memory

Phase 1 mitigations:

- retain the current memory-content threat scanning
- forbid direct persistence of prompt-like override instructions
- never allow model inference to create high-trust records
- require stronger confirmation for behavioural records than for observational records
- keep retrieval explanations inspectable so suspicious records can be found and removed

## Operational Model

Memory V2 splits writes and maintenance into two operational paths.

### Foreground Writes

Used for:

- explicit user preferences
- explicit corrections
- critical durable facts needed immediately

Properties:

- fast
- low volume
- tightly controlled
- synchronous enough to affect future behaviour predictably

### Background Consolidation

Used for:

- deduplication
- semantic promotion
- episodic extraction
- stale review
- summary generation
- confidence adjustment
- archival and compaction

Properties:

- asynchronous where possible
- conservative by default
- should not slow the core turn loop unnecessarily

### Consolidation Trigger Model

Phase 1 background consolidation runs on simple deterministic triggers:

- end of session
- explicit memory pressure threshold reached
- explicit operator request in the future CLI surface
- optional scheduled maintenance hook if infrastructure already exists

Phase 1 does not require continuous background workers.

### Consolidation Resource Budget

Consolidation is budgeted.

Phase 1 requirements:

- consolidation must have bounded work per run
- if model calls are required, limit them to a single-digit capped number per run; the intended phase 1 order of magnitude is low single digits, not tens
- if budget is exhausted, remaining candidates stay queued for later review
- foreground responsiveness always outranks consolidation completeness

### Concurrency and Race Handling

Foreground writes beat consolidation.

Phase 1 concurrency rules:

- each record carries a `revision`
- consolidation reads a snapshot and writes back only if the target revision has not changed
- if a foreground write changes the record first, consolidation aborts or retries on fresh state
- destructive consolidation operations require archival backup of the prior record version

Concurrent foreground writes are a known phase 1 limitation.

Phase 1 behaviour for concurrent foreground writes:

- if two sessions write to the same `scope + topic_key` concurrently, the second writer must re-read before commit
- if the target revision changed, the second write must not blindly overwrite
- instead it must re-run contradiction resolution or fail closed into a retry path
- phase 1 does not guarantee perfect multi-session merge semantics; it guarantees that silent destructive overwrite is not the default behaviour

### Backup and Rollback

Consolidation and normalization must be reversible enough for operators to recover from bad compaction.

Phase 1 requires:

- archival copy or revision trail before destructive merge/supersession update
- operator-visible inspection path for archived or superseded records
- ability to restore a record from the prior revision path without manual filesystem surgery

## Cold Start Strategy

Memory V2 must acknowledge the cold start problem.

Phase 1 bootstrapping strategy:

- rely on explicit operator facts and built-in memory as the seed layer
- do not invent semantic or episodic memory from a blank slate
- allow transcript recall to remain empty without treating that as failure
- promote only validated durable facts and episodes as experience accumulates

This keeps the system conservative when little evidence exists.

## Inspection and Observability

Memory inspection is a design requirement, not an open question.

Phase 1 must provide an inspectable explanation path for memory behaviour, even if the UX remains minimal.

At minimum, operators and developers must be able to inspect:

- why a record was written
- why a record was retrieved
- the record's status, trust tier, salience tier, scope, and source kind
- whether a record superseded or conflicted with another
- whether a record was archived or expired during consolidation

This can initially be implemented through structured tool output or developer-focused diagnostics. It does not require a polished end-user dashboard in phase 1.

## Fit With The Current Hermes Codebase

Memory V2 should evolve the current architecture rather than replace it.

### Components That Stay

- `tools/memory_tool.py` remains the built-in memory entry point
- `tools/session_search_tool.py` remains the transcript recall system
- `agent/memory_manager.py` remains the orchestration hub
- `agent/memory_provider.py` remains the plugin contract for one active external provider
- existing provider plugins under `plugins/memory/*` remain valid
- the frozen session-start injection pattern for built-in prompt memory remains in place

### Components That Change

- built-in memory becomes typed and metadata-aware internally
- `session_search` is treated as transcript recall, not canonical durable memory
- `MemoryManager` gains policy/routing responsibilities instead of merely aggregating provider hooks
- external providers are mapped into a shared conceptual model where practical
- episodic memory becomes a first-class concern

### Suggested Module Additions

The design warrants a small set of focused modules:

- `agent/memory_policy.py`
  - write classification
  - retrieval budgets
  - freshness rules
  - contradiction rules
  - consolidation triggers
- `agent/memory_records.py`
  - shared record types
  - enums
  - conversion helpers
- `agent/memory_ranking.py`
  - ranking composition and penalties
- `agent/memory_episodes.py`
  - episodic record extraction and retrieval
- `agent/memory_inspection.py`
  - explanation and operator-facing inspection helpers

This is enough structure to support the design without turning the memory stack into a new subsystem for its own sake.

## Migration Strategy

Memory V2 should be delivered in backward-compatible stages.

### Stage 0: Terminology and Internal Model

Introduce the conceptual model of:

- profile memory
- semantic memory
- episodic memory
- transcript recall

No breaking behaviour changes required. This stage is mainly internal framing and documentation alignment.

### Stage 1: Built-In Memory Hardening

Improve the built-in memory layer first:

- typed records
- metadata
- ordinal trust and salience tiers
- status fields
- stronger write policy
- conflict and supersession support
- safer retrieval selection
- memory inspection for built-in records
- capacity and archival behaviour

This yields immediate user value and does not depend on provider cooperation.

### Stage 2: MemoryManager Policy Integration

Teach `MemoryManager` to:

- route writes by class and confidence
- merge built-in memory with provider context more intentionally
- treat transcript recall as a separate path
- apply common retrieval budgets and ranking logic
- expose explanations for retrieved records

This is the first point where Memory V2 should feel coherent at the product level.

### Stage 3: Episodic Memory

Introduce execution memory conservatively:

- store only successful or user-validated episodes
- keep examples compact
- retrieve only on strong task match
- retire or demote contaminated episodes

### Stage 4: Provider Adaptation

Allow providers to opt into richer metadata and ranking contracts over time.

This adaptation must remain backward-compatible. Providers that cannot expose trust or freshness metadata should still work, but with reduced ranking fidelity.

### If Stage 4 Never Ships

Memory V2 is still valid if provider adaptation remains partial or absent.

In that case:

- built-in profile and semantic memory still improve core recall quality
- transcript recall still benefits from clearer conceptual separation
- episodic memory can still exist locally for built-in workflows
- external providers continue to operate as optional enrichment rather than first-class policy peers

Stage 4 is valuable, but not required for Memory V2 to justify itself.

## Provider Adaptation Contract

Provider adaptation is both technical and political. Phase 1 therefore defines a minimal contract.

Providers may optionally expose:

- provider-native trust or confidence metadata
- provider-native timestamps
- provider-native record identifiers
- provider-native relevance hints

Hermes adapter rules:

- Hermes keeps final authority over internal trust tier mapping
- provider-native scores are treated as hints, not canonical truth
- missing provider metadata must degrade gracefully to conservative defaults
- no provider is required to expose every field in the common record model

## Backward Compatibility

Memory V2 should preserve:

- current built-in memory usage patterns
- existing `MEMORY.md` and `USER.md` content
- current `session_search` workflows
- current external-provider activation and setup flow
- current single-external-provider rule

Legacy `MEMORY.md` and `USER.md` entries should be treated as legacy profile records on load, then normalized internally where useful. Users should not need to rewrite their memory stores by hand.

For phase 1 storage, built-in prompt-resident memory remains file-backed for compatibility and operator legibility. Structured metadata may be stored in sidecar or auxiliary local storage, but the human-readable built-in memory surface remains intact.

## Evaluation Plan

Memory V2 should define its evaluation criteria before implementation work begins.

### Baselines

Before phase 1 implementation starts, capture current-system baselines for:

- durable preference recall accuracy
- stale recall rate on environment facts and preferences
- `session_search` success on historical lookup tasks
- prompt-token overhead from built-in memory
- average turn latency impact from current memory operations

If measurement infrastructure is incomplete, phase 1 planning must define the minimum benchmark set required to establish these baselines.

### Core Metrics

- write precision: of stored records, how many are later judged useful?
- retrieval precision: of retrieved records, how many are relevant and safe?
- stale recall rate: how often does Hermes retrieve outdated information?
- contradiction handling accuracy: how often does it prefer the correct active record?
- repeated-task lift: do episodic memories improve recurring tasks?
- token overhead: how much prompt cost does Memory V2 add?
- latency overhead: how much extra response latency comes from retrieval and consolidation?

### Measurement Methodology

Phase 1 planning must define a small labelled evaluation set covering:

- explicit preference updates
- environment fact changes
- contradiction scenarios
- stale path and deployment examples
- recurring task episodes with known successful and unsuccessful outcomes
- transcript recall lookups

Write precision and retrieval precision are measured against this labelled set or a similarly scoped regression suite. They are not inferred from anecdotal usage.

### Initial Rollout Targets

Phase 1 should not proceed to broader rollout unless it demonstrates:

- no regression in current built-in memory correctness on baseline tasks
- measurable improvement in contradiction handling and stale-memory suppression on the labelled set
- bounded overhead within an explicit planning-time budget for prompt cost and turn latency
- inspectable explanations for writes and retrievals on benchmark scenarios

Exact numeric thresholds belong in the implementation plan, but the rollout gate must be explicit before coding begins.

### Rollout Gate Principle

Each implementation phase must be useful on its own:

- phase 1 improves core memory quality and safety
- phase 2 improves coherence across memory surfaces
- phase 3 improves repeated-task performance
- phase 4 improves provider interoperability

If development stops after any phase, Hermes should still be better than before.

## Success Criteria

Memory V2 is successful if it delivers measurable improvements in the following areas:

1. better durable recall of user preferences, project conventions, environment facts, and corrections
2. fewer stale-memory failures and fewer contradictory retrievals
3. better repeated-task performance through validated episodic reuse
4. bounded token and latency overhead
5. backward-compatible usability for current users and provider setups
6. operator-visible explanations for key memory decisions

## Risks and Failure Modes

Primary risks:

- overengineering the architecture before proving value
- retrieval pollution from permissive write policy
- stale or conflicting records being treated as equally valid
- episodic contamination from low-quality remembered workflows
- provider mismatch where metadata support varies widely
- loss of operator trust if retrieval becomes opaque or difficult to inspect
- memory corruption during consolidation or compaction
- runaway growth in semantic or episodic stores
- adversarial memory poisoning
- concurrent session writes producing inconsistent state

The design counters these risks by using a conservative first slice, strict write policy, explicit freshness and contradiction handling, bounded relations instead of arbitrary graphs, archival-before-destruction, and measurable rollout gates.

## Remaining Open Questions For Implementation Planning

The design intentionally resolves several earlier open questions. The following remain for planning:

- whether sidecar metadata storage should be JSON, SQLite, or another local format
- the exact numeric defaults for retrieval budgets and consolidation work caps
- the exact benchmark corpus and regression fixtures for phase 1 evaluation
- the exact CLI or inspection-surface shape for operator-facing diagnostics

These are implementation-planning questions, not architectural gaps.

## Recommended First Implementation Slice

The recommended first implementation slice is intentionally conservative:

1. define the internal record model
2. harden built-in memory with types, status, ordinal trust tiers, salience tiers, and freshness metadata
3. implement rule-based write classification with policy validation
4. implement contradiction and supersession support with bounded relations
5. add inspection output for why a record was written or retrieved
6. keep `session_search` conceptually separate but documented as transcript recall
7. extend `MemoryManager` just enough to understand the new conceptual layers
8. add baseline measurements and a small labelled regression suite before claiming phase 1 success

This first slice is the best balance of strategic value, implementation risk, and compatibility with the current Hermes codebase.

## Final Summary

Hermes Memory V2 should turn the current collection of memory features into one coherent model:

- profile memory for always-important durable facts
- semantic memory for retrievable long-lived knowledge
- episodic memory for validated reusable prior solutions
- transcript recall for broad historical lookup
- one policy layer that decides what to write, what to retrieve, what to distrust, and what to retire

It should be delivered incrementally, measured carefully, and implemented as an evolution of the current Hermes architecture rather than a theatrical rewrite.