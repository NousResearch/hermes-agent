# Memory Benchmark Suite Hardening Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fully harden and extend the Hermes memory benchmark suite before running the provider matrix, so results are fair, reproducible, capability-aware, and useful for comparing heterogeneous memory plugins.

**Architecture:** Keep the benchmark runner's current adapter model, but strengthen result schema, provider preflight, comparison reporting, and missing major test categories. Preserve backward compatibility where practical: existing `mean_score` remains, while official comparisons use `score_views.core`, shared-category views, skipped-category reasons, and provider/runtime metadata.

**Tech Stack:** Python stdlib benchmark harness, pytest, existing `benchmarks/interface.py`, `benchmarks/runner.py`, `benchmarks/tracks.py`, `benchmarks/metrics.py`, JSON fixture suites, provider adapters under `benchmarks/backends/`.

---

## Non-Negotiable Constraints

- Do not commit, tag, push, or publish without explicit user approval.
- Do not print API keys, provider tokens, or credential values.
- Prefer isolated output directories for benchmark runs.
- Provider benchmark claims must report core/shared-suite scores, not only raw mean score.
- One-seed results must never be described as statistically significant.
- Add tests before or alongside every runner/schema change.

---

## Phase 0: Baseline Verification Before New Work

### Task 0.1: Verify current test baseline

**Objective:** Confirm the current branch is green before adding more changes.

**Files:**
- No source changes.

**Steps:**
1. Run:
   ```bash
   cd /workspace/Projects/hermes-agent-benchmark-fairness
   /workspace/Projects/.venvs/hermes-bench/bin/python -B -m pytest tests/benchmarks -q
   ```
2. Expected:
   ```text
   33 passed
   ```
3. If failures appear, fix them before continuing.

---

## Phase 1: Result Schema and Comparison Correctness

### Task 1.1: Extract result serialization into a reusable function

**Objective:** Ensure normal runs and `--compare` runs write identical rich JSON schemas.

**Files:**
- Modify: `benchmarks/runner.py`
- Test: `tests/benchmarks/test_result_serialization.py`

**Implementation:**
Add a helper near existing result-output code:

```python
def build_result_data(config: BenchmarkConfig, agg: AggregateResult, runs: list[RunResult]) -> dict:
    """Build the rich benchmark result JSON schema used for all saved results."""
    import datetime

    requested_categories = requested_categories_for_suites(config.parameters.get("suites", ["a"]))
    executed_categories = sorted(agg.per_category_mean.keys())
    skipped_categories = build_skipped_category_reasons(
        requested_categories,
        executed_categories,
        BACKEND_CAPABILITIES.get(config.backend_name, BackendCapabilities()),
    )
    score_views = build_score_views(runs, requested_categories)

    avg_rm_file = average_run_dicts([r.retrieval_metrics for r in runs if r.retrieval_metrics])
    avg_cm_file = average_run_dicts([r.cost_metrics for r in runs if r.cost_metrics])

    return {
        "schema_version": "2.0",
        "backend": config.backend_name,
        "profile": config.profile,
        "embedding_model": config.embedding_model,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "suites": config.parameters.get("suites", ["a"]),
        "requested_categories": requested_categories,
        "executed_categories": executed_categories,
        "skipped_categories": skipped_categories,
        "score_views": score_views,
        "mean_score": agg.mean_score,
        "std": agg.std_score,
        "ci_95": [agg.ci_95_lower, agg.ci_95_upper],
        "per_category_mean": agg.per_category_mean,
        "per_category_std": agg.per_category_std,
        "num_runs": agg.num_runs,
        "retrieval_metrics": avg_rm_file,
        "cost_metrics": avg_cm_file,
        "runtime": capture_runtime_metadata(config.backend_name),
        "runs": [serialize_run(r) for r in runs],
    }
```

Also add helpers:

```python
def average_run_dicts(items: list[dict]) -> dict:
    if not items:
        return {}
    keys = sorted({k for item in items for k in item})
    return {
        key: sum(float(item[key]) for item in items if key in item) / max(1, sum(1 for item in items if key in item))
        for key in keys
    }


def serialize_run(r: RunResult) -> dict:
    return {
        "seed": r.seed,
        "overall_score": r.overall_score,
        "wall_time_seconds": r.wall_time_seconds,
        "token_usage": r.token_usage,
        "retrieval_metrics": getattr(r, "retrieval_metrics", {}),
        "cost_metrics": getattr(r, "cost_metrics", {}),
        "categories": {
            cat: {
                "score": cr.score,
                "correct": cr.correct,
                "total": cr.total,
                "recall_tokens": cr.recall_tokens,
                "recall_chars": cr.recall_chars,
                "retrieval_metrics": cr.retrieval_metrics,
                "sub_scores": cr.sub_scores,
            }
            for cat, cr in r.results_by_category.items()
        },
    }
```

**Test:**
Create a synthetic aggregate/run and assert `build_result_data()` contains:
- `schema_version`
- `requested_categories`
- `executed_categories`
- `skipped_categories`
- `score_views`
- `runtime`
- `runs[0].categories[...]`

**Verification:**
```bash
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m pytest tests/benchmarks/test_result_serialization.py -q
```

---

### Task 1.2: Make `--compare` use full result serialization

**Objective:** Fix asymmetric result files from the comparison path.

**Files:**
- Modify: `benchmarks/runner.py`
- Test: `tests/benchmarks/test_result_serialization.py`

**Implementation:**
Replace the minimal `json.dump({...})` block inside `if args.compare:` with:

```python
result_data2 = build_result_data(config2, agg2, runs2)
with open(result_file2, "w") as f:
    json.dump(result_data2, f, indent=2)
```

Also save a timestamped history file for the comparison backend using the same pattern as the primary backend.

**Test:**
Use helper-level tests rather than spawning the whole CLI. Assert both primary and comparison paths call the same builder if practical, or test the builder directly.

**Verification:**
```bash
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m pytest tests/benchmarks -q
```

---

### Task 1.3: Add skipped-category reasons

**Objective:** Replace opaque skipped-category names with structured reasons.

**Files:**
- Modify: `benchmarks/runner.py`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Implementation:**
Add:

```python
def build_skipped_category_reasons(
    requested_categories: list[str],
    executed_categories: list[str],
    capabilities: BackendCapabilities,
) -> dict[str, str]:
    from benchmarks.tracks import missing_capabilities

    executed = set(executed_categories)
    skipped: dict[str, str] = {}
    for category in requested_categories:
        if category in executed:
            continue
        missing = missing_capabilities(capabilities, category)
        if missing:
            skipped[category] = "missing capabilities: " + ", ".join(sorted(missing))
        else:
            skipped[category] = "not executed"
    return skipped
```

**Test:**
For default/flat capabilities, request `temporal_decay` and assert reason includes `time_simulation`.

**Compatibility:**
If downstream consumers expect a list, add `skipped_category_names` as a list or document schema version 2.0. Prefer explicit schema version.

---

### Task 1.4: Add shared-category comparison helper

**Objective:** Compute a fair score across only categories executed by every backend in a result set.

**Files:**
- Create: `benchmarks/reporting.py` or add to `benchmarks/runner.py` initially
- Test: `tests/benchmarks/test_reporting.py`

**Implementation:**
Add pure helper:

```python
def shared_category_view(result_payloads: list[dict]) -> dict:
    """Compute per-backend scores over the intersection of executed categories."""
    if not result_payloads:
        return {"categories": [], "backends": {}}

    category_sets = [set(p.get("executed_categories", [])) for p in result_payloads]
    shared = sorted(set.intersection(*category_sets)) if category_sets else []

    backends = {}
    for payload in result_payloads:
        name = payload["backend"]
        correct = 0
        total = 0
        for run in payload.get("runs", []):
            for cat in shared:
                cr = run.get("categories", {}).get(cat)
                if cr:
                    correct += cr.get("correct", 0)
                    total += cr.get("total", 0)
        backends[name] = {
            "score": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        }

    return {"categories": shared, "backends": backends}
```

**Test:**
Two synthetic payloads with overlapping and non-overlapping categories. Assert only intersection is scored.

---

## Phase 2: Runtime Metadata and Provider Preflight

### Task 2.1: Capture runtime and package versions

**Objective:** Make every benchmark result reproducible.

**Files:**
- Modify: `benchmarks/runner.py`
- Test: `tests/benchmarks/test_runtime_metadata.py`

**Implementation:**
Add:

```python
def capture_runtime_metadata(backend_name: str) -> dict:
    import importlib.metadata as md
    import platform
    import subprocess

    package_names = [
        "mnemoria",
        "mem0ai",
        "honcho-ai",
        "hindsight-client",
        "openviking",
        "sentence-transformers",
    ]
    packages = {}
    for name in package_names:
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None

    byterover_version = None
    try:
        proc = subprocess.run(["brv", "--version"], capture_output=True, text=True, timeout=10)
        byterover_version = proc.stdout.strip() or proc.stderr.strip() or None
    except Exception:
        byterover_version = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": backend_name,
        "packages": packages,
        "byterover_cli": byterover_version,
        "git": capture_git_metadata(),
    }
```

Add:

```python
def capture_git_metadata() -> dict:
    import subprocess

    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return proc.stdout.strip() if proc.returncode == 0 else None
        except Exception:
            return None

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--short"])),
    }
```

**Test:**
Mock `importlib.metadata.version` and subprocess calls. Do not require packages to be installed.

---

### Task 2.2: Add credential presence metadata without values

**Objective:** Record whether required env vars were present without leaking secrets.

**Files:**
- Modify: `benchmarks/runner.py`
- Test: `tests/benchmarks/test_runtime_metadata.py`

**Implementation:**
Add constant:

```python
PROVIDER_ENV_VARS = {
    "mem0": ["MEM0_API_KEY"],
    "retaindb": ["RETAINDB_API_KEY"],
    "hindsight": ["HINDSIGHT_API_KEY", "HINDSIGHT_BASE_URL"],
    "honcho": ["HONCHO_API_KEY", "HONCHO_BASE_URL"],
    "openviking": ["OPENVIKING_ENDPOINT", "OPENVIKING_API_KEY"],
    "byterover": ["BYTEROVER_API_KEY"],
}
```

Add to runtime metadata:

```python
"credentials": {
    key: ("set" if os.environ.get(key) else "missing")
    for key in PROVIDER_ENV_VARS.get(backend_name, [])
}
```

**Test:**
Use pytest monkeypatch to set/unset env vars. Assert only `set`/`missing`, never values.

---

### Task 2.3: Add backend preflight helper

**Objective:** Smoke-test every provider before expensive full benchmarks.

**Files:**
- Modify: `benchmarks/runner.py` or create `benchmarks/preflight.py`
- Test: `tests/benchmarks/test_preflight.py`

**Implementation:**
Add a CLI flag:

```python
parser.add_argument("--preflight", action="store_true", help="Run backend setup/store/recall/reset smoke check and exit")
```

Add helper:

```python
def preflight_backend(config: BenchmarkConfig) -> dict:
    backend = get_backend(config.backend_name, config)
    result = {
        "backend": config.backend_name,
        "ok": False,
        "steps": {},
        "error": None,
    }
    try:
        backend.reset()
        result["steps"]["reset_before"] = "ok"
        backend.store("The benchmark preflight code is moon-731.", category="factual")
        result["steps"]["store"] = "ok"
        recalled = backend.recall("What is the benchmark preflight code?", top_k=3)
        result["steps"]["recall"] = "ok" if any("moon-731" in r for r in recalled) else "no_match"
        backend.reset()
        result["steps"]["reset_after"] = "ok"
        result["ok"] = result["steps"]["recall"] == "ok"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return result
```

In `main()`, if `args.preflight`, print JSON and exit 0/1 based on `ok`.

**Test:**
Create fake backend classes: one passing, one failing. Assert structured result and no secret leakage.

---

## Phase 3: CLI Suite Discovery and Reporting Polish

### Task 3.1: Auto-discover `--suite all`

**Objective:** Avoid silently excluding future suites.

**Files:**
- Modify: `benchmarks/runner.py`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Implementation:**
Add:

```python
def discover_suites() -> list[str]:
    return sorted(
        d.name.split("_")[1]
        for d in SUITE_DIR.iterdir()
        if d.is_dir() and d.name.startswith("suite_") and (d / "fixtures").exists()
    )
```

Replace hardcoded `suite all` list with `discover_suites()`.

Update `requested_categories_for_suites()` to call `discover_suites()` too.

**Test:**
Assert current discovery returns `a` through `o` and includes suite `o`.

---

### Task 3.2: Print score views in terminal output

**Objective:** Make fair comparison surfaces visible without opening JSON.

**Files:**
- Modify: `benchmarks/runner.py`
- Test: optional snapshot-style test if existing output tests exist.

**Implementation:**
In `print_results()`, after overall score, print:

```text
Fair comparison views:
  Executed score: 0.xxx over N categories
  Core score:     0.xxx over N categories
  Tracks:
    core:         0.xxx
    temporal:     0.xxx
    structured:   0.xxx
    lifecycle:    0.xxx
```

Use `build_score_views(runs, requested_categories_for_suites(...))`.

---

## Phase 4: Add Missing Major Benchmark Categories

### Task 4.1: Add `abstention` category metadata

**Objective:** Test false positives when no relevant memory exists.

**Files:**
- Modify: `benchmarks/tracks.py`
- Modify: `benchmarks/runner.py`
- Create: `benchmarks/suite_p/fixtures/abstention.json`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Track metadata:**
```python
"abstention": CategorySpec("core", [], True, "Avoiding false-positive recall when no answer is stored"),
```

**Fixture design:**
Create 12 scenarios:
- 4 easy unrelated distractors
- 4 medium semantically close distractors
- 4 hard close distractors with overlapping entities/numbers

Example scenario:
```json
{
  "id": "abs_001",
  "difficulty": "hard",
  "facts": [
    "The deployment region for Project Orion is us-east-1.",
    "The billing owner for Project Lyra is Mina.",
    "Project Orion uses PostgreSQL 16 for analytics."
  ],
  "query": "Who is the billing owner for Project Orion?",
  "gold_answer": "NONE"
}
```

**Runner:**
Correct if recalled results do not answer the query. Use inverted judge logic:
- empty result = correct
- explicit unknown/none = correct
- otherwise judge retrieved text against query/gold and invert verdict

---

### Task 4.2: Add `preference_memory` category

**Objective:** Test stable user preferences, corrected preferences, and identity-like facts.

**Files:**
- Modify: `benchmarks/tracks.py`
- Modify: `benchmarks/runner.py`
- Create: `benchmarks/suite_q/fixtures/preference_memory.json`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Track metadata:**
```python
"preference_memory": CategorySpec("core", [], True, "Remembering stable and corrected user preferences"),
```

**Fixture design:**
Create 15 scenarios across subtypes:
- `stable_preference`
- `corrected_preference`
- `negative_preference`
- `project_convention`
- `identity_fact`

Example:
```json
{
  "id": "pref_004",
  "sub_type": "corrected_preference",
  "turns": [
    "The user prefers concise bullet lists for status reports.",
    "Correction: the user prefers narrative summaries for complex benchmark reports."
  ],
  "query": "How should complex benchmark reports be written for the user?",
  "gold_answer": "narrative summaries"
}
```

**Runner:**
Store turns in order, query, judge top result. Newer correction should win.

---

### Task 4.3: Add `privacy_forgetting` capability-gated category

**Objective:** Test deletion/forgetting/privacy behavior where supported.

**Files:**
- Modify: `benchmarks/capabilities.py`
- Modify: `benchmarks/interface.py`
- Modify: `benchmarks/tracks.py`
- Modify: `benchmarks/runner.py`
- Create: `benchmarks/suite_r/fixtures/privacy_forgetting.json`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Capability:**
Add to `BackendCapabilities`:
```python
forgetting: bool = False
```

**Interface default:**
Add optional method to `BenchmarkableStore`:
```python
def forget(self, content_substring: str | None = None, scope: str | None = None) -> None:
    """Forget matching memories. Optional; required for forgetting capability."""
    raise NotImplementedError("Backend does not support forgetting")
```

**Track metadata:**
```python
"privacy_forgetting": CategorySpec("privacy", ["forgetting"], False, "Deletion and no-recall guarantees after forgetting"),
```

**Fixture design:**
10 scenarios:
- forget one secret among distractors
- forget scope/project
- forget corrected sensitive fact
- verify unrelated retained fact still recalls

**Runner:**
Store facts, call `backend.forget(...)`, query forgotten fact and retained fact. Correct only if forgotten fact does not recall and retained fact does recall.

**Important:**
Do not count unsupported backends as failed. Skip with reason `missing capabilities: forgetting`.

---

### Task 4.4: Add `multi_hop_exploration` category

**Objective:** Test graph-like and multi-hop memory traversal.

**Files:**
- Modify: `benchmarks/capabilities.py`
- Modify: `benchmarks/tracks.py`
- Modify: `benchmarks/runner.py`
- Create: `benchmarks/suite_s/fixtures/multi_hop_exploration.json`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Capability:**
Add if not present:
```python
exploration: bool = False
```

**Track metadata:**
```python
"multi_hop_exploration": CategorySpec("exploration", [], False, "Recovering answers that require linked multi-hop facts"),
```

Start universal, not capability-gated, by using `backend.explore()` which falls back to recall. Later analysis can separate native exploration support by capability.

**Fixture design:**
12 scenarios:
- 4 two-hop bridge
- 4 entity disambiguation
- 4 three-hop chain

Example:
```json
{
  "id": "mh_001",
  "sub_type": "two_hop_bridge",
  "facts": [
    "The Atlas service is owned by Team Cedar.",
    "Team Cedar's on-call channel is #cedar-alerts.",
    "Team Cobalt's on-call channel is #cobalt-alerts."
  ],
  "query": "What on-call channel should Atlas incidents use?",
  "gold_answer": "#cedar-alerts"
}
```

**Runner:**
Store all facts, call `backend.explore(query, top_k=5)`, join top results, judge.

---

### Task 4.5: Add `long_conversation` category

**Objective:** Better approximate real agent cross-session memory.

**Files:**
- Modify: `benchmarks/tracks.py`
- Modify: `benchmarks/runner.py`
- Create: `benchmarks/suite_t/fixtures/long_conversation.json`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Track metadata:**
```python
"long_conversation": CategorySpec("core", [], True, "Long multi-turn and cross-session conversational memory"),
```

**Fixture design:**
10 scenarios, each with 12-30 turns and multiple sessions. Subtypes:
- `preference_evolution`
- `project_decision_history`
- `correction_after_delay`
- `multi_session_synthesis`
- `rejected_option_recall`

**Runner:**
Store selected turns, query at the end. Use scenario-defined `store_turns` if present; otherwise store all user/fact turns.

---

## Phase 5: Provider Adapters and Capability Declarations

### Task 5.1: Audit all backend capability declarations

**Objective:** Ensure all backends honestly declare capabilities after adding new ones.

**Files:**
- Modify as needed:
  - `benchmarks/baseline/flat_store.py`
  - `benchmarks/backends/mnemoria_adapter.py`
  - `benchmarks/backends/holographic_adapter.py`
  - `benchmarks/backends/mem0_adapter.py`
  - `benchmarks/backends/retaindb_adapter.py`
  - `benchmarks/backends/hindsight_adapter.py`
  - `benchmarks/backends/honcho_adapter.py`
  - `benchmarks/backends/openviking_adapter.py`
  - `benchmarks/backends/byterover_adapter.py`
- Test: `tests/benchmarks/test_capability_tracks.py`

**Checklist:**
For each adapter, verify declared values for:
- universal_store_recall
- time_simulation
- access_rehearsal
- consolidation
- scopes
- typed_facts
- supersession
- reward_learning
- exploration
- turn_sync
- precompress_hook
- session_end_hook
- delegation_hook
- forgetting

**Rule:**
Do not lie to make scores look better. Unsupported categories should skip.

---

### Task 5.2: Implement `forget()` where available

**Objective:** Enable privacy_forgetting for backends that can truly delete.

**Files:**
- Adapter-specific.

**Policy:**
Only set `forgetting=True` if the adapter can actually remove or isolate the matching memory so recall no longer returns it.

For local adapters, this may be straightforward.
For cloud providers, if deletion is not reliable or only rotates user/session IDs, do not claim forgetting support unless scenario-level deletion can be implemented.

---

## Phase 6: Tests and Verification Matrix

### Task 6.1: Add metadata coverage tests

**Objective:** Prevent adding runner categories without track metadata.

**Files:**
- Modify: `tests/benchmarks/test_capability_tracks.py`

**Assertions:**
- every `CATEGORY_RUNNERS` key has `CATEGORY_SPECS`
- every fixture file stem has `CATEGORY_SPECS`
- every fixture file stem has a runner
- every required capability exists on `BackendCapabilities`

---

### Task 6.2: Add smoke tests for new categories using baseline-flat

**Objective:** Ensure every new category executes without crashing.

**Files:**
- Create or modify: `tests/benchmarks/test_new_categories_smoke.py`

**Test pattern:**
Use the actual runner functions with a tiny subset or actual fixtures. Assert:
- total > 0, or skipped correctly for capability-gated categories
- score is between 0 and 1
- details include scenario IDs

---

### Task 6.3: Run full benchmark test suite

**Objective:** Verify complete benchmark harness before provider runs.

**Command:**
```bash
cd /workspace/Projects/hermes-agent-benchmark-fairness
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m pytest tests/benchmarks -q
```

**Expected:**
All tests pass. Record exact count.

---

## Phase 7: Dry Runs Before Provider Matrix

### Task 7.1: Run local backend preflights

**Objective:** Confirm preflight works for local backends.

**Commands:**
```bash
cd /workspace/Projects/hermes-agent-benchmark-fairness
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend baseline-flat --preflight
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend holographic --preflight
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend mnemoria --embedding tfidf --preflight
```

---

### Task 7.2: Run one-seed local backend dry runs

**Objective:** Verify rich JSON schema on real benchmark output.

**Commands:**
```bash
export BENCH_OUT=/tmp/memory-benchmark-suite-hardening-dryrun
cd /workspace/Projects/hermes-agent-benchmark-fairness
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend baseline-flat --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend holographic --suite all --runs 1 --seeds 42 --output-dir "$BENCH_OUT"
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend mnemoria --suite all --runs 1 --seeds 42 --embedding tfidf --output-dir "$BENCH_OUT"
```

**Verify JSON contains:**
- `schema_version`
- `runtime`
- `score_views`
- `skipped_categories` as object with reasons
- new categories requested/executed/skipped as expected
- history files exist

---

### Task 7.3: Generate local comparison report

**Objective:** Verify reporting before external provider costs.

**Command concept:**
If reporting CLI exists by then:
```bash
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks.report "$BENCH_OUT"/*.json
```

If no reporting CLI yet, use a small script to load JSON and print:
- raw mean score
- core score
- shared-category score
- per-track scores
- skipped categories
- wall time
- token metrics

---

## Phase 8: External Provider Readiness

### Task 8.1: Verify installed provider versions

**Objective:** Record exact versions before smoke tests.

**Command:**
```bash
cd /workspace/Projects/hermes-agent-benchmark-fairness
/workspace/Projects/.venvs/hermes-bench/bin/python -B -c '
import importlib.metadata as md
for name in ["mem0ai", "honcho-ai", "hindsight-client", "openviking", "sentence-transformers", "mnemoria"]:
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, "not-installed")
'
brv --version || true
```

Do not upgrade anything until we decide whether to use current versions or a fresh isolated provider venv.

---

### Task 8.2: External provider preflight, one by one

**Objective:** Find setup failures before full benchmark runs.

**Commands:**
```bash
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend mem0 --preflight
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend retaindb --preflight
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend hindsight --preflight
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend honcho --preflight
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend openviking --preflight
/workspace/Projects/.venvs/hermes-bench/bin/python -B -m benchmarks --backend byterover --preflight
```

Stop and fix setup for each failed provider. Do not begin full matrix until preflight status is documented.

---

## Acceptance Criteria Before Real Benchmark Matrix

The suite is ready to run provider benchmarks only when all are true:

- `pytest tests/benchmarks -q` passes.
- `--suite all` auto-discovers all suite directories.
- Both primary and `--compare` results use the same rich schema.
- Result JSON includes runtime/package/git metadata.
- Skipped categories include reasons.
- Shared-category comparison helper exists and is tested.
- Provider preflight exists and is tested.
- New major categories are wired with fixtures, runners, track metadata, and tests:
  - abstention
  - preference_memory
  - privacy_forgetting
  - multi_hop_exploration
  - long_conversation
- All local backend preflights pass.
- One-seed local dry runs produce valid rich JSON.
- No secrets are printed or committed.

---

## Suggested Implementation Order

1. Phase 1 result schema fixes.
2. Phase 2 runtime metadata and preflight.
3. Phase 3 suite discovery/reporting polish.
4. Phase 4 new benchmark categories.
5. Phase 5 capability audit.
6. Phase 6 tests.
7. Phase 7 local dry runs.
8. Phase 8 external provider preflight.
9. Only then run provider matrix.

This order gets the fairness/reproducibility foundations in place before expanding scope, then verifies locally before touching external providers.
