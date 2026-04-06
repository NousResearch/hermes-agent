# Benchmark Phase 1 Spec — Capability Schema, Track Mapping, and Result Format

> For Hermes: this is the concrete first-PR spec for making the benchmark capability-aware without yet adding all new suites.

Goal: implement the smallest reviewable benchmark refactor that enables fair comparison across different memory architectures by adding backend capability metadata, track mapping, skip-aware execution, and coverage-aware result reporting.

Scope of this phase:
- Add backend capability declarations
- Add category-to-track mapping and category requirements
- Teach the runner to skip unsupported categories honestly
- Update JSON/report/chart outputs to show core score, optional track scores, and capability coverage
- Reclassify existing categories into universal core vs optional tracks

Out of scope for this phase:
- Adding all new benchmark categories (topic_shift_recall, compression_survival, delegation_memory)
- Writing all external plugin adapters
- Replacing existing suites wholesale
- Changing benchmark judging logic unless needed for skip/report support

---

## 1. Principles for the first PR

This phase should be minimal, truthful, and easy to review.

It should not try to answer:
- which provider is best overall
- whether Mnemoria beats anyone

It should answer:
- how Hermes can compare memory backends fairly despite different architectures

Review heuristic:
If a maintainer can merge this PR while being skeptical of Mnemoria, the PR is scoped correctly.

---

## 2. Exact file targets for Phase 1

Primary files to modify:
- benchmarks/interface.py
- benchmarks/runner.py
- benchmarks/statistical.py
- benchmarks/visualize/report.py
- benchmarks/visualize/charts.py
- benchmarks/README.md
- benchmarks/__main__.py

New files to add:
- benchmarks/capabilities.py
- benchmarks/tracks.py

Optional test files to add in the benchmark branch or repo tests:
- tests/benchmarks/test_capability_filtering.py
- tests/benchmarks/test_track_scoring.py
- tests/benchmarks/test_result_schema.py

If benchmark tests live inside benchmarks/ itself, keep them adjacent and lightweight.

---

## 3. New data model

## 3.1 benchmarks/capabilities.py

Add a single canonical capability dataclass.

Recommended content:

```python
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
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

    def to_dict(self) -> dict:
        return asdict(self)
```

Notes:
- universal_store_recall exists mostly for completeness and future validation.
- turn_sync / precompress_hook / session_end_hook / delegation_hook are runtime-facing even if not yet fully used in Phase 1 categories.
- Keep this intentionally simple. No enums yet.

---

## 3.2 benchmarks/tracks.py

This module should become the single source of truth for:
- category → track mapping
- category → capability requirements
- which categories are part of the universal core score

Recommended structure:

```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class CategorySpec:
    track: str
    required_capabilities: List[str] = field(default_factory=list)
    in_core_score: bool = False
    description: str = ""
```

Then define a registry like:

```python
CATEGORY_SPECS = {
    "semantic_recall": CategorySpec(
        track="core",
        required_capabilities=[],
        in_core_score=True,
        description="Basic semantic retrieval quality",
    ),
    ...
}
```

Helper functions:

```python
def get_category_spec(category: str) -> CategorySpec: ...
def categories_for_track(track: str) -> list[str]: ...
def core_categories() -> list[str]: ...
def backend_supports_category(capabilities, category: str) -> bool: ...
```

---

## 4. Capability declarations per existing backend

In Phase 1, do not try to infer capabilities dynamically. Declare them explicitly when registering backends.

### 4.1 Runner registry change

Current registry:
- backend name → backend class

Phase 1 registry:
- backend name → backend class
- backend name → BackendCapabilities

Recommended shape in benchmarks/runner.py:

```python
BACKENDS: Dict[str, Type[BenchmarkableStore]] = {}
BACKEND_CAPABILITIES: Dict[str, BackendCapabilities] = {}

def register_backend(name: str, cls: Type[BenchmarkableStore], capabilities: BackendCapabilities | None = None):
    BACKENDS[name] = cls
    BACKEND_CAPABILITIES[name] = capabilities or BackendCapabilities()
```

### 4.2 Initial capability declarations

These should be conservative. Better to under-claim than over-claim.

#### baseline-flat
```python
BackendCapabilities(
    universal_store_recall=True,
)
```

#### cognitive
```python
BackendCapabilities(
    universal_store_recall=True,
    time_simulation=True,
    access_rehearsal=True,
    consolidation=True,
    reward_learning=True,
)
```

#### structured
```python
BackendCapabilities(
    universal_store_recall=True,
    scopes=True,
    typed_facts=True,
    supersession=True,
)
```

Do not grant time_simulation or consolidation unless truly implemented in adapter behavior.

#### unified
```python
BackendCapabilities(
    universal_store_recall=True,
    time_simulation=True,
    access_rehearsal=True,
    consolidation=True,
    scopes=True,
    typed_facts=True,
    supersession=True,
    reward_learning=True,
    exploration=True,
)
```

For Phase 1, even if unified has runtime hooks in the product branch, do not declare turn_sync / precompress_hook / session_end_hook / delegation_hook unless the benchmark adapter can exercise them in benchmark mode.

Important rule:
Capabilities should describe benchmark-exercisable support, not aspirational product support.

---

## 5. Category reclassification for Phase 1

This is the most important truth-telling step.

## 5.1 Core track categories

These should stay comparable across all current in-tree backends.

Recommended Phase 1 core categories:
- semantic_recall
- contradictions
- cross_reference
- importance_filtering
- adversarial
- scale
- conversation_memory
- deduplication

Rationale:
- They mainly require store/recall behavior
- They represent broad memory competence
- They avoid overfitting to one architecture

Caution:
The current contradictions runner uses time ordering and repeated stores, but not strict typed/scope assumptions. Keep it in core for now if it can run honestly across current backends. If external providers later need a non-time-dependent currentness suite, that can be a later refinement.

## 5.2 Temporal track

Categories:
- temporal_decay
- consolidation
- compression
- typed_decay

Required capabilities by category:
- temporal_decay → ["time_simulation"]
- consolidation → ["consolidation", "time_simulation", "access_rehearsal"]
- compression → ["consolidation", "time_simulation"]
- typed_decay → ["typed_facts", "time_simulation"]

## 5.3 Structured track

Categories:
- scopes
- supersession
- scope_lifecycle
- notation_parsing

Required capabilities:
- scopes → ["scopes"]
- supersession → ["supersession"]
- scope_lifecycle → ["scopes", "consolidation"]
- notation_parsing → ["typed_facts"]

Note:
scope_lifecycle currently relies on simulate_time + consolidate in the runner. That means it belongs in structured track but also needs temporal-style support in benchmark execution.

## 5.4 Learning track

Categories:
- qlearning

Required capabilities:
- qlearning → ["reward_learning"]

## 5.5 Exploration track

Phase 1 options:
- either add no explicit exploration track yet
- or introduce the track but leave it empty until plugin/adapters and runner coverage improve

Recommendation:
Define the track now, but do not move existing categories into it yet unless there is a true dedicated explore-mode category. Academic explore toggles can remain separate for now.

## 5.6 Integration track

Current category:
- integration

This one is tricky.

It uses:
- store
- recall
- simulate_time
- simulate_access
- consolidate

So it is not universal, despite the intuitive name.

Recommendation for Phase 1:
Move integration to an optional "lifecycle" or "integration" track requiring:
- time_simulation
- access_rehearsal
- consolidation

Do not keep integration in core score.

This change alone improves honesty a lot.

---

## 6. Result schema changes

## 6.1 RunResult additions in benchmarks/interface.py

Add fields to RunResult:

```python
track_scores: Dict[str, float] = field(default_factory=dict)
executed_categories: List[str] = field(default_factory=list)
skipped_categories: Dict[str, str] = field(default_factory=dict)
capability_coverage: Dict[str, bool] = field(default_factory=dict)
```

Where:
- track_scores = score per track for that run
- executed_categories = categories actually run
- skipped_categories = category → reason string
- capability_coverage = backend capabilities snapshot

## 6.2 AggregateResult additions

Add fields:

```python
track_mean: Dict[str, float] = field(default_factory=dict)
track_std: Dict[str, float] = field(default_factory=dict)
core_mean_score: float = 0.0
core_std_score: float = 0.0
executed_categories: List[str] = field(default_factory=list)
skipped_categories: Dict[str, str] = field(default_factory=dict)
capability_coverage: Dict[str, bool] = field(default_factory=dict)
```

Do not remove existing fields. Preserve backward compatibility where possible.

## 6.3 JSON output shape

Current top-level JSON is broadly:
- backend
- mean_score
- std
- per_category_mean
- num_runs
- runs
- retrieval_metrics
- cost_metrics

Phase 1 should extend it to include:

```json
{
  "backend": "unified",
  "mean_score": 0.95,
  "core_mean_score": 0.91,
  "std": 0.01,
  "core_std_score": 0.01,
  "track_mean": {
    "core": 0.91,
    "temporal": 0.97,
    "structured": 0.94,
    "learning": 0.88
  },
  "track_std": {
    "core": 0.01,
    "temporal": 0.00,
    "structured": 0.01,
    "learning": 0.03
  },
  "capability_coverage": {
    "time_simulation": true,
    "access_rehearsal": true,
    "consolidation": true,
    "scopes": true,
    "typed_facts": true,
    "supersession": true,
    "reward_learning": true,
    "exploration": true,
    "turn_sync": false,
    "precompress_hook": false,
    "session_end_hook": false,
    "delegation_hook": false
  },
  "executed_categories": [...],
  "skipped_categories": {
    "scope_lifecycle": "missing capabilities: scopes, consolidation",
    "qlearning": "missing capabilities: reward_learning"
  },
  "per_category_mean": {...},
  "per_category_std": {...},
  "num_runs": 5,
  "runs": [...]
}
```

Important:
- mean_score can remain as “all executed categories” for backward compatibility.
- core_mean_score becomes the key fair-comparison number.
- README must explain the difference.

---

## 7. Runner behavior changes

## 7.1 Category selection flow

Current flow appears to assume selected suite categories all run.

Phase 1 flow should be:

1. Resolve requested suites → categories
2. For each category:
   - load CategorySpec
   - check backend capabilities
   - if supported: execute
   - if unsupported: record skipped reason
3. Compute:
   - overall executed-category score
   - core-only score
   - per-track score

## 7.2 Skip behavior

A skipped category must:
- not count as zero
- not count as correct
- not count in denominator for core or track scores
- be explicitly reported in stdout and JSON

Recommended helper:

```python
def missing_capabilities(capabilities, category: str) -> list[str]: ...
```

Skip reason format:
- "missing capabilities: time_simulation"
- "missing capabilities: scopes, consolidation"

## 7.3 Empty track behavior

If a backend has no executed categories in a track:
- omit the track from track_mean
or
- set to null

Recommendation:
omit from numeric track_mean and list under skipped/coverage instead.
This keeps report formatting simpler.

---

## 8. Statistical aggregation changes

File:
- benchmarks/statistical.py

Add aggregation for:
- track_mean / track_std
- core_mean_score / core_std_score
- union of executed_categories
- merged skipped_categories
- copied capability coverage

Recommended rules:
- track means are averaged across runs using the same approach as per-category means
- core_mean_score is average of per-run core score
- skipped_categories may be taken from the first run if deterministic, or merged if needed

Because skips are backend-static, first-run copy is acceptable for Phase 1.

---

## 9. Reporting changes

## 9.1 CLI stdout in benchmarks/runner.py

Current printed summary should gain:
- capability coverage summary line
- core score line
- track table
- skipped category summary

Recommended stdout structure:

```text
============================================================
  BENCHMARK RESULTS: unified
============================================================
  Runs: 5
  Overall (executed): 0.952 ± 0.004
  Core score:         0.910 ± 0.006

  Tracks:
    core         0.910
    temporal     0.972
    structured   0.944
    learning     0.881

  Coverage:
    supported capabilities: time_simulation, access_rehearsal, consolidation, ...
    skipped categories: none
```

For narrower backends:

```text
  Coverage:
    supported capabilities: scopes, typed_facts
    skipped categories:
      temporal_decay (missing capabilities: time_simulation)
      qlearning (missing capabilities: reward_learning)
```

## 9.2 Markdown report changes

File:
- benchmarks/visualize/report.py

Add sections:
- Core Score
- Track Scores
- Capability Coverage
- Skipped Categories

Keep existing sections.

## 9.3 ASCII dashboard changes

File:
- benchmarks/visualize/charts.py

Add:
- compact track score table
- compact capability coverage display
- skipped category count or list

Do not overdecorate. Terminal clarity first.

---

## 10. README changes

File:
- benchmarks/README.md

Add new sections:

### 10.1 Capability-aware evaluation
Explain:
- some memory systems support more behaviors than others
- unsupported capabilities are skipped, not failed
- fair comparison should focus on core score plus applicable tracks

### 10.2 How to read results
Explain these terms exactly:
- Overall (executed): all categories the backend actually ran
- Core score: universal comparison score
- Track scores: optional capability-specific comparisons
- Coverage: what the backend actually supports

### 10.3 Why this improves fairness
Explain:
- previous single-interface comparison could bias toward certain architectures
- capability-aware reporting reduces that bias
- this helps all current and future memory providers

---

## 11. Backward compatibility strategy

This phase should minimize downstream breakage.

Recommendations:
1. Keep existing fields in JSON
2. Add new fields rather than renaming old ones
3. Keep mean_score semantics as executed-category aggregate
4. Introduce core_mean_score rather than replacing mean_score
5. Keep old visualization paths working even if they ignore new fields initially

If a report consumer only knows the old schema, it should still be able to parse the file.

---

## 12. Concrete implementation order

### Step 1
Create benchmarks/capabilities.py with BackendCapabilities.

### Step 2
Create benchmarks/tracks.py with CategorySpec registry and helpers.

### Step 3
Update benchmarks/runner.py:
- register_backend signature
- capability registry
- category support checks
- skip collection
- per-track/core aggregation per run

### Step 4
Update benchmarks/interface.py dataclasses for new fields.

### Step 5
Update benchmarks/statistical.py to aggregate new track/core fields.

### Step 6
Update JSON writer in runner.py to emit:
- core_mean_score
- track_mean
- track_std
- capability_coverage
- executed_categories
- skipped_categories

### Step 7
Update stdout printing in runner.py.

### Step 8
Update benchmarks/visualize/report.py and charts.py.

### Step 9
Update benchmarks/README.md.

### Step 10
Add tests for:
- category support checks
- skip behavior
- core score computation
- track aggregation
- JSON field presence

---

## 13. Acceptance criteria for Phase 1

A backend with limited support should now produce output like this:
- core score is present
- unsupported categories are skipped, not failed
- skipped reasons are visible
- track scores only appear where relevant
- capability coverage is printed and saved

A strong backend should now produce output like this:
- core score is distinct from overall executed score
- track breakdown is visible
- no misleading single-number claim is implied

Maintainer review should be able to verify:
- no category was silently dropped
- skips are explicit
- fairness model is documented in README
- existing benchmark behavior mostly survives except where honesty required reclassification

---

## 14. What not to do in Phase 1

Do not:
- add Mnemoria adapter yet just to prove a point
- add complex dynamic capability probing
- mix provider-native benchmark ideas into the core model
- rewrite all existing suites
- bikeshed names endlessly

Keep the first PR narrow and principled.

---

## 15. Suggested PR title for Phase 1

Capability-aware benchmark scoring and coverage reporting for memory backends

Suggested PR description opening:

"This PR refactors the Hermes memory benchmark to report core scores, optional track scores, and backend capability coverage. It preserves existing benchmark categories while making cross-backend comparison more honest: unsupported capabilities are now skipped explicitly instead of being treated as failures, and universal comparison is separated from optional architecture-specific evaluation."

---

## 16. Final note

This phase is the scaffolding spell.
It does not yet prove who stands tallest in the forest.
It just makes sure we stop measuring owls by how well they swim.
