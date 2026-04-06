# Capability-Aware Memory Benchmark Expansion Plan

> For Hermes: use this plan to guide a benchmark PR that improves hermes-agent overall, not to privilege any single memory plugin.

Goal: extend the hermes-agent benchmark system so it can evaluate memory backends more fairly across different architectures, while adding missing runtime-relevant behaviors that matter in real agent use.

Architecture:
- Keep a universal core benchmark track that all memory systems can run honestly.
- Add capability-gated tracks for features only some systems support.
- Add a small set of new runtime-relevant benchmark categories justified by Hermes’s actual provider lifecycle hooks.
- Report capability coverage and track-specific scores instead of forcing a misleading single leaderboard.

Tech stack:
- Python benchmark framework under benchmarks/
- Existing BenchmarkableStore adapters and plugin backend discovery
- Hermes MemoryProvider lifecycle hooks
- JSON fixtures + stdlib metrics/judging

---

## 1. PR intent and truth-in-advertising

This PR should explicitly say:

1. Hermes now supports multiple memory provider architectures with different strengths.
2. A fair benchmark must distinguish:
   - universal memory competencies
   - optional architectural capabilities
   - provider-native specialties
3. The benchmark is being upgraded to reduce architecture-shaped bias, not to crown a preferred plugin.

Non-goals:
- Do not claim one universal “best memory backend” score.
- Do not design suites around Mnemoria-only internals.
- Do not require fake shims when a backend lacks native support for a feature.

Recommended PR summary language:

"This PR upgrades Hermes memory benchmarking to be capability-aware. It separates universal memory tasks from optional architecture-specific tracks, adds runtime-relevant benchmark categories derived from Hermes’s provider lifecycle, and enables fairer comparison across local, cloud, structured, conversational, and hybrid memory backends."

---

## 2. Current benchmark strengths to preserve

The existing benchmark branch already covers substantial ground and should not be discarded.

Existing broad coverage:
- semantic recall
- contradictions/currentness
- temporal decay
- cross-reference
- importance filtering
- consolidation and compression
- scopes and scope lifecycle
- adversarial robustness
- scale and capacity stress
- integration lifecycle
- q-learning
- supersession
- typed decay
- notation parsing
- deduplication
- conversation memory
- academic adapters: LongMemEval, LoCoMo, HotpotQA

Keep these. The work is to reorganize and clarify fairness, not to replace everything.

---

## 3. New benchmark categories justified by Hermes runtime

These are the highest-value additions because they map to real Hermes runtime hooks or behaviors rather than Mnemoria-specific theory.

### 3.1 topic_shift_recall

Why it belongs:
- Hermes conversations are multi-topic.
- Memory providers should help the agent recover the right context after a topic pivot.
- This is a real production behavior, not a speculative feature.

Hermes/runtime justification:
- Unified memory branch has topic shift detection and proactive recall warming.
- Hermes provider model already supports per-turn sync/prefetch behavior.

What it tests:
- Store facts around topic A.
- Switch to topic B with different relevant facts.
- Query about B and verify low contamination from A.
- Optionally switch back to A and verify recovery.

What makes it fair:
- Does not require fake time travel.
- Can be implemented by any backend that supports store + recall.

### 3.2 compression_survival

Why it belongs:
- Context compression is one of the most important real-world memory failure points.
- A backend that performs well before compression but loses critical facts afterward is not strong in practice.

Hermes/runtime justification:
- MemoryProvider includes on_pre_compress(messages).
- Hermes supports context compression and provider extraction before old context is discarded.

What it tests:
- Multi-turn conversation containing facts, corrections, decisions.
- Trigger a synthetic compression boundary.
- Measure post-compression recall of critical content.

What makes it fair:
- Can be run as a pure conversation-retention task for all backends.
- Backends with on_pre_compress support can participate in a stronger lifecycle variant.

### 3.3 delegation_memory

Why it belongs:
- Hermes relies heavily on subagents and delegated work.
- Retaining useful outcomes from delegated tasks is central to agent-native memory quality.

Hermes/runtime justification:
- MemoryProvider includes on_delegation(task, result, ...).
- MemoryManager forwards delegation events.

What it tests:
- Parent delegates a task.
- Child returns a result containing decisions/facts/outcomes.
- Provider should preserve the useful result for later parent recall.

What makes it fair:
- Can be simulated without special internal access.
- Lifecycle-capable backends can use the dedicated hook; simpler backends can still be evaluated on the outcome recall form.

### 3.4 decision_tracking (optional for first PR, strong candidate)

Why it belongs:
- Decisions are one of the most important kinds of project memory.
- Many agents fail not because they forget raw facts, but because they forget what was decided and what superseded it.

Hermes/runtime justification:
- Unified memory branch includes explicit decision detection and decision-oriented retrieval.
- Existing conversation tests cover adjacent ground, but decision memory deserves explicit treatment.

What it tests:
- establish a decision
- revise/replace it later
- query for current decision, prior decision, and rationale

What makes it fair:
- This is broadly relevant across all memory systems.
- It should not assume typed-fact support in the universal version.

---

## 4. Capability-aware benchmark model

### 4.1 Problem with the current interface

The current BenchmarkableStore contract includes methods like:
- simulate_time()
- simulate_access()
- consolidate()

These are useful for research backends but are not native operations for many real plugins.
If all backends are forced through them, the comparison becomes adapter-shaped rather than architecture-fair.

### 4.2 Proposed model

Split the benchmark contract into:

1. Universal base interface
2. Declared capabilities
3. Optional capability-specific methods

### 4.3 Proposed backend declaration

Every benchmark backend should declare a capability profile, for example:

```python
@dataclass
class BackendCapabilities:
    universal_store_recall: bool = True
    time_simulation: bool = False
    access_rehearsal: bool = False
    consolidation: bool = False
    scopes: bool = False
    typed_facts: bool = False
    supersession: bool = False
    reward_learning: bool = False
    exploration: bool = False
    turn_sync: bool = False
    precompress_hook: bool = False
    session_end_hook: bool = False
    delegation_hook: bool = False
```

And each adapter exports something like:

```python
BACKEND_CAPABILITIES = BackendCapabilities(
    time_simulation=True,
    scopes=True,
    typed_facts=True,
    supersession=True,
    reward_learning=True,
    exploration=True,
    turn_sync=True,
    precompress_hook=True,
    delegation_hook=True,
)
```

### 4.4 Suite requirements

Each suite/category declares what it requires.
Example:

```python
SUITE_REQUIREMENTS = {
    "semantic_recall": [],
    "topic_shift_recall": ["universal_store_recall"],
    "compression_survival": ["turn_sync"],
    "compression_survival_lifecycle": ["precompress_hook"],
    "temporal_decay": ["time_simulation"],
    "supersession": ["typed_facts", "supersession"],
    "qlearning": ["reward_learning"],
    "exploration_multihop": ["exploration"],
    "delegation_memory": ["delegation_hook"],
}
```

If a backend lacks required capabilities:
- the suite is skipped
- not scored as failure
- coverage report records the omission

This is the heart of fairness.

---

## 5. Scoring model for fair comparison

Do not publish a single global score across incomparable systems.

### 5.1 Report three layers

1. Core score
- Only universal categories
- Comparable across all plugins

2. Capability-track scores
- Temporal track
- Structured track
- Lifecycle track
- Learning track
- Exploration track
- Only compare within eligible backends

3. Coverage profile
- Shows which benchmark capabilities each provider actually supports
- Prevents misleading leaderboard interpretations

### 5.2 Recommended tracks

#### Core track (all backends)
Should include only categories that can be run honestly for every provider:
- semantic_recall
- current_state_recall / correction_handling
- adversarial
- scale
- conversation_memory
- deduplication
- topic_shift_recall
- maybe decision_tracking

Note:
Prefer a correction/currentness category for universal comparison instead of requiring strong time simulation.

#### Temporal track
Requires:
- time_simulation
Optional extras:
- access_rehearsal
Categories:
- temporal_decay
- rehearsal sensitivity
- historical validity if added later

#### Structured track
Requires combinations of:
- scopes
- typed_facts
- supersession
Categories:
- scopes
- scope_lifecycle
- notation_parsing
- supersession
- typed_decay

#### Lifecycle track
Requires some combination of:
- turn_sync
- precompress_hook
- session_end_hook
- delegation_hook
Categories:
- compression_survival
- delegation_memory
- session_summary_retention if added later
- lifecycle topic-shift variant if added later

#### Learning track
Requires:
- reward_learning
Categories:
- qlearning
- reward-driven reranking tests

#### Exploration track
Requires:
- exploration
Categories:
- multi-hop retrieval
- graph bridge traversal
- academic explore mode comparisons

### 5.3 Provider-native track
Allowed, but must be labeled explicitly as provider-native and excluded from universal ranking.
Examples:
- Holographic: trust feedback, entity probing
- Honcho: peer-card/profile synthesis
- Mem0: server-side extraction quality
- Mnemoria: advanced lifecycle extraction or auto-supersession modes

These are informative, but not grounds for a universal winner claim.

---

## 6. Fairness principles to document in the benchmark README

Add a new "Fairness Principles" section to benchmarks/README.md with language like:

1. No fake failures
A backend is not penalized for lacking a capability outside its design scope.

2. No fake support
Adapters should not invent unrealistic behavior solely to satisfy a benchmark hook.

3. Universal vs optional separation
Scores must distinguish core memory competence from optional architectural features.

4. Coverage matters
A backend solving more classes of memory problems may have broader utility, but narrower systems can still excel in their intended lane.

5. Provider-native strengths are real but not universal
Specialized capabilities should be reported, not forced into a universal leaderboard.

6. The benchmark exists to help Hermes make better long-term memory decisions
It is not a marketing surface for any one backend.

---

## 7. Recommended implementation sequence

### Phase 1: capability metadata + skipping behavior

Goal:
Refactor the benchmark runner to support declared capabilities and skip unsupported suites cleanly.

Tasks:
1. Add BackendCapabilities dataclass to benchmarks/interface.py
2. Add per-backend capability declarations
3. Add suite/category requirement metadata
4. Update runner to skip unsupported suites and record coverage
5. Update result JSON schema to include:
   - supported_tracks
   - skipped_categories
   - coverage summary

Deliverable:
Capability-aware benchmark infrastructure without changing most existing fixtures.

### Phase 2: universal/core vs optional track reclassification

Goal:
Reclassify existing suites into core and optional tracks.

Tasks:
1. Define official track mapping in runner or a config module
2. Mark current categories accordingly
3. Ensure reporting shows:
   - core score
   - optional track scores
   - capability coverage

Deliverable:
Truthful scoring model even before adding new categories.

### Phase 3: add new runtime-relevant suites

Recommended order:
1. topic_shift_recall
2. compression_survival
3. delegation_memory
4. optional: decision_tracking

Deliverable:
Benchmark now covers real agent-runtime memory behaviors that were previously missing.

### Phase 4: add external provider adapters to prove fairness

Recommended first adapters:
1. holographic
2. honcho

Why these first:
- Holographic proves local structured comparison is not Mnemoria-shaped.
- Honcho proves cloud/profile-oriented systems can participate honestly in the framework.

Only after these should we introduce Mnemoria adapter comparisons as evidence of fairness.

### Phase 5: document result interpretation

Add documentation and output examples showing:
- how to compare core scores
- how to compare track scores
- how to read coverage
- why missing tracks are not failures

---

## 8. Suggested concrete file changes

Likely files to modify in the benchmark PR:

Core framework:
- benchmarks/interface.py
- benchmarks/runner.py
- benchmarks/README.md
- benchmarks/__main__.py

Possible new helper modules:
- benchmarks/capabilities.py
- benchmarks/tracks.py

New fixtures/suites:
- benchmarks/suite_j/fixtures/topic_shift_recall.json
- benchmarks/suite_k/fixtures/compression_survival.json
- benchmarks/suite_l/fixtures/delegation_memory.json
- optional later: benchmarks/suite_m/fixtures/decision_tracking.json

Possible adapter additions:
- benchmarks/backends/holographic_adapter.py
- benchmarks/backends/honcho_adapter.py

Docs/reporting:
- benchmarks/visualize/report.py
- benchmarks/visualize/charts.py

Tests:
- tests for capability skipping logic
- tests for coverage reporting
- tests for new fixture runners

---

## 9. How to compare Mnemoria fairly once the framework exists

When Mnemoria is eventually added, compare it using this hierarchy:

1. Core score vs all providers
- Does it compete broadly on universal memory tasks?

2. Track scores vs eligible providers only
- Temporal track vs systems with real temporal behavior
- Structured track vs typed/scoped systems
- Lifecycle track vs turn-aware/runtime-aware systems
- Learning track vs systems with reward feedback
- Exploration track vs graph/explore systems

3. Coverage profile
- How much of the memory problem space does it attempt to solve?

This allows an honest result like:
- "Mnemoria has stronger coverage and strong scores in structured/lifecycle tracks"
without saying:
- "Mnemoria beats every provider at everything"

That distinction matters for the long-term health of Hermes.

---

## 10. Reviewer-facing rationale

If challenged on why this benchmark expansion is needed, answer:

1. Hermes has grown from one internal memory stack into a multi-provider ecosystem.
2. The benchmark must evolve to match that ecosystem.
3. The old model risks unfairly favoring backends that happen to expose certain research-oriented hooks.
4. The new model improves benchmark honesty by:
   - preserving strong existing coverage
   - adding real runtime memory tasks
   - separating universal comparison from optional capabilities
5. This benefits all present and future Hermes memory providers, not just Mnemoria.

---

## 11. Recommended acceptance criteria for the PR

Infrastructure:
- Backends can declare capabilities.
- Runner skips unsupported categories without marking them wrong.
- Results include coverage and track-level scoring.

Fairness:
- Core score is computed only from universal categories.
- Optional tracks are only computed for eligible backends.
- README explicitly explains why unsupported tracks are skipped.

New benchmark coverage:
- topic_shift_recall implemented and runnable
- compression_survival implemented and runnable
- delegation_memory implemented and runnable
- optional: decision_tracking implemented and runnable

Proof of ecosystem intent:
- At least one non-Mnemoria external plugin adapter added or specified as immediate follow-up
- Documentation avoids claiming a universal single-score winner

---

## 12. Recommended first-draft PR title and summary

Title:
Capability-aware memory benchmark tracks and runtime lifecycle evaluation

Short summary:
This PR upgrades Hermes memory benchmarking to support fairer comparison across different memory architectures. It adds capability-aware track scoring, coverage reporting, and new runtime-relevant benchmark categories for topic shifts, compression survival, and delegation memory. The goal is to help Hermes evaluate memory systems honestly over the long run, rather than force all providers through one architecture-shaped scoring model.

---

## 13. Final strategic guidance

The benchmark should be a sanctuary for truth.
If we build it right, it will sometimes flatter Mnemoria, sometimes not, and that is exactly why people will trust it.

A fair benchmark for Hermes should answer:
- What problems can this memory backend solve?
- How well does it solve the problems it claims to solve?
- How much of the memory surface area does it cover?

That is better for Hermes than any benchmark that exists to prove our pet system is special.
