# Model Performance Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark every available local, free Nous, and approved GPT/Codex candidate and compile explicit low-latency routes for every Hermes profile and auxiliary surface.

**Architecture:** A reproducible benchmark package discovers candidates, executes deterministic task cases with bounded deadlines, computes correctness-first cost/performance scores, and emits a signed routing artifact. A runtime resolver consumes that artifact with privacy gating, circuit breaking, context admission, and weighted local concurrency.

**Tech Stack:** Python 3.11, asyncio, subprocess/Ollama HTTP, existing Nous/OpenAI provider clients, psutil where already available, JSON/JSONL, YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-llm-egress-and-model-routing-design.md`

## Global Constraints

- Inventory every `ollama list` model and every currently callable free Nous model; unavailable candidates remain in reports with typed reasons.
- Include GPT/Codex 5.4 mini, 5.5, 5.6 Luna, Terra, and Sol; include Spark only if Hermes can invoke it through a supported provider.
- Correctness, tool accuracy, context capacity, and privacy are eligibility gates before latency or price.
- Every profile and invocation surface requires an explicit primary, privacy-compatible fallback, context window, timeout, and benchmark artifact binding.
- No-output stalls get a short hard deadline, at most one transient retry, and a different fallback route.
- Local concurrency protects a detected strict runtime process and never launches it.

---

### Task 1: Candidate discovery and immutable inventory

**Files:**
- Create: `evals/model_router/__init__.py`
- Create: `evals/model_router/inventory.py`
- Create: `evals/model_router/schema.py`
- Create: `tests/evals/test_model_router_inventory.py`

**Interfaces:**
- Produces: `ModelCandidate`, `InventorySnapshot`, `discover_ollama_models()`, `discover_nous_free_models()`, `approved_codex_candidates()`, and `discover_candidates() -> InventorySnapshot`.
- Consumes: `hermes_cli.models.fetch_nous_recommended_models` and the authenticated model-catalog helper without reading token values.

- [ ] **Step 1: Write failing discovery, deduplication, and unavailable-model tests**

```python
def test_inventory_unions_live_portal_and_authenticated_free_models():
    snapshot = discover_candidates(adapters=fakes)
    assert {m.model_id for m in snapshot.candidates} >= {
        "upstage/solar-pro4:free", "poolside/laguna-xs-2.1:free"
    }

def test_catalog_only_model_remains_visible_as_unavailable():
    model = by_id(snapshot, "stale/free:free")
    assert model.availability == "unavailable"
    assert model.unavailable_reason == "inference_catalog_absent"
```

- [ ] **Step 2: Run inventory tests and verify missing module failure**

Run: `pytest -q tests/evals/test_model_router_inventory.py`

- [ ] **Step 3: Implement typed, timestamped discovery without exposing credentials**

```python
@dataclass(frozen=True, slots=True)
class ModelCandidate:
    provider: str
    model_id: str
    destination_class: str
    price_input_per_million: Decimal | None
    price_cached_input_per_million: Decimal | None
    price_output_per_million: Decimal | None
    advertised_context: int | None
    capabilities: frozenset[str]
    availability: str
    unavailable_reason: str | None
```

Parse `ollama list --json` when supported and fall back to its stable tabular output. Fetch the public Nous recommendations, union the authenticated free catalog, and probe callability with a one-token side-effect-free request during operator-run benchmarks.

- [ ] **Step 4: Run inventory tests**

Run: `pytest -q tests/evals/test_model_router_inventory.py`

- [ ] **Step 5: Commit discovery**

```bash
git add evals/model_router/__init__.py evals/model_router/inventory.py evals/model_router/schema.py tests/evals/test_model_router_inventory.py
git commit -m "feat(evals): inventory every model candidate"
```

### Task 2: Deterministic multi-surface benchmark corpus

**Files:**
- Create: `evals/model_router/corpus.py`
- Create: `evals/model_router/fixtures/`
- Create: `tests/evals/test_model_router_corpus.py`

**Interfaces:**
- Consumes: `ModelCandidate`.
- Produces: `BenchmarkCase`, `ExpectedReceipt`, `load_corpus() -> tuple[BenchmarkCase, ...]`, and `score_case_output(case, output, tool_calls) -> CaseVerdict`.

- [ ] **Step 1: Write failing coverage and machine-checkable receipt tests**

```python
REQUIRED_SURFACES = {
    "curator", "review", "review_comment", "reference_aggregator",
    "fallback", "context_admission", "compression", "title", "mcp",
    "approval", "skills_hub", "vision", "coding", "ci_audit",
    "ci_repair", "pr_repair", "merge_maintenance", "orchestration",
    "cron", "kanban", "new_conversation",
}

def test_corpus_has_a_deterministic_case_for_every_required_surface():
    assert REQUIRED_SURFACES <= {case.surface for case in load_corpus()}
    assert all(case.expected_receipt is not None for case in load_corpus())
```

- [ ] **Step 2: Run corpus tests and verify missing fixtures fail**

Run: `pytest -q tests/evals/test_model_router_corpus.py`

- [ ] **Step 3: Add bounded synthetic tasks, exact tool schemas, and secret-canary privacy cases**

```python
@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    surface: str
    prompt_fixture: Path
    tools_fixture: Path | None
    required_capabilities: frozenset[str]
    max_output_tokens: int
    no_output_timeout_seconds: float
    total_timeout_seconds: float
    expected_receipt: ExpectedReceipt
```

Use synthetic repositories and generated credentials only. No benchmark fixture may contain real workspace source or secrets.

- [ ] **Step 4: Run corpus tests**

Run: `pytest -q tests/evals/test_model_router_corpus.py`

- [ ] **Step 5: Commit the corpus**

```bash
git add evals/model_router/corpus.py evals/model_router/fixtures tests/evals/test_model_router_corpus.py
git commit -m "feat(evals): add profile-complete model corpus"
```

### Task 3: Bounded runner and resource measurements

**Files:**
- Create: `evals/model_router/runner.py`
- Create: `evals/model_router/providers.py`
- Create: `tests/evals/test_model_router_runner.py`

**Interfaces:**
- Consumes: `ModelCandidate`, `BenchmarkCase`, and the egress firewall route classifier.
- Produces: `CaseMeasurement`, `run_case(candidate, case)`, and `run_matrix(inventory, corpus, repetitions) -> BenchmarkRun`.

- [ ] **Step 1: Write failing TTFT, timeout, one-retry, capability, cost, and local-memory tests**

```python
async def test_no_output_deadline_aborts_and_retries_only_once_on_a_different_route():
    result = await run_case(primary_stalls, case, fallback=fallback_succeeds)
    assert result.attempt_count == 2
    assert result.attempts[0].outcome == "no_output_timeout"
    assert result.attempts[1].candidate_id != result.attempts[0].candidate_id
```

- [ ] **Step 2: Run runner tests and verify they fail before adapters exist**

Run: `pytest -q tests/evals/test_model_router_runner.py`

- [ ] **Step 3: Implement streamed TTFT, bounded total timeout, token/cost accounting, and local RSS/load measurement**

```python
@dataclass(frozen=True, slots=True)
class CaseMeasurement:
    candidate_id: str
    case_id: str
    outcome: str
    deterministic_score: float
    tool_accuracy: float
    ttft_ms: int | None
    wall_ms: int
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: Decimal | None
    peak_rss_bytes: int | None
    attempt_count: int
```

The runner invokes providers through existing Hermes clients, uses the firewall for source-bearing cases, and stores only synthetic benchmark inputs plus measurements.

- [ ] **Step 4: Run runner tests**

Run: `pytest -q tests/evals/test_model_router_runner.py`

- [ ] **Step 5: Commit the runner**

```bash
git add evals/model_router/runner.py evals/model_router/providers.py tests/evals/test_model_router_runner.py
git commit -m "feat(evals): benchmark bounded model completion"
```

### Task 4: Correctness-first scoring and signed routing artifacts

**Files:**
- Create: `evals/model_router/scoring.py`
- Create: `evals/model_router/report.py`
- Create: `tests/evals/test_model_router_scoring.py`

**Interfaces:**
- Consumes: `BenchmarkRun`.
- Produces: `CandidateScore`, `SurfaceRoute`, `RoutingArtifact`, `score_surface(...)`, and `compile_routes(...)`.

- [ ] **Step 1: Write failing ineligibility, expected-wall-time, cost-per-success, Sol escalation, and deterministic tie-break tests**

```python
def test_faster_incorrect_model_is_ineligible():
    route = compile_routes([fast_wrong, slower_correct], policy)
    assert route.primary.model_id == slower_correct.model_id

def test_expected_wall_time_penalizes_retries():
    score = score_surface(p50_ms=5000, bounded_success_probability=0.5)
    assert score.expected_success_wall_ms == 10000
```

- [ ] **Step 2: Run scoring tests and verify missing implementation**

Run: `pytest -q tests/evals/test_model_router_scoring.py`

- [ ] **Step 3: Implement hard eligibility floors followed by expected successful wall time and cost ceilings**

```python
eligible = (
    score.deterministic_success >= policy.min_success
    and score.tool_accuracy >= policy.min_tool_accuracy
    and score.context_ok
    and score.privacy_ok
)
expected_success_wall_ms = score.mean_wall_ms / max(score.success_probability, 0.001)
```

Emit canonical JSON containing inventory, corpus version, environment fingerprint, measurements, routes, and policy digest; sign it with a SHA-256 content digest stored beside the artifact.

- [ ] **Step 4: Run scoring and report tests**

Run: `pytest -q tests/evals/test_model_router_scoring.py`

- [ ] **Step 5: Commit scoring**

```bash
git add evals/model_router/scoring.py evals/model_router/report.py tests/evals/test_model_router_scoring.py
git commit -m "feat(evals): compile correctness-first model routes"
```

### Task 5: Exhaustive profile and auxiliary route compiler

**Files:**
- Create: `agent/model_performance_router.py`
- Create: `hermes_cli/profile_route_compiler.py`
- Create: `tests/agent/test_model_performance_router.py`
- Create: `tests/hermes_cli/test_profile_route_compiler.py`

**Interfaces:**
- Consumes: `RoutingArtifact` and configured profile inventory.
- Produces: `ResolvedRoute`, `resolve_route(profile: str, surface: str, privacy: str, required_context: int) -> ResolvedRoute`, and `compile_profile_configs(...)`.

- [ ] **Step 1: Write failing exhaustive-matrix and missing-row rejection tests**

```python
def test_every_profile_and_surface_has_explicit_primary_fallback_context_and_timeout():
    compiled = compile_profile_configs(profiles, artifact)
    for profile in profiles:
        for surface in REQUIRED_SURFACES:
            row = compiled[profile][surface]
            assert row.primary and row.privacy_fallback
            assert row.context_window >= 65536
            assert row.no_output_timeout_seconds > 0

def test_missing_auxiliary_row_fails_compilation():
    with pytest.raises(RouteCompilationError, match="skills_hub"):
        compile_profile_configs(profiles, artifact_without("skills_hub"))
```

- [ ] **Step 2: Run compiler tests and verify absence of explicit matrix fails**

Run: `pytest -q tests/agent/test_model_performance_router.py tests/hermes_cli/test_profile_route_compiler.py`

- [ ] **Step 3: Implement explicit rows for curator, review, fallback, context, title, MCP, approval, Skills Hub, compression, vision, and operational lanes**

```python
@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    provider: str
    model: str
    reasoning_effort: str | None
    context_window: int
    max_output_tokens: int
    no_output_timeout_seconds: float
    total_timeout_seconds: float
    concurrency_weight: int
    privacy_class: str
    artifact_digest: str
```

Compilation writes profile configuration atomically only after every row validates. Shared winners are duplicated as explicit justified rows.

- [ ] **Step 4: Run compiler tests**

Run: `pytest -q tests/agent/test_model_performance_router.py tests/hermes_cli/test_profile_route_compiler.py`

- [ ] **Step 5: Commit exhaustive routing**

```bash
git add agent/model_performance_router.py hermes_cli/profile_route_compiler.py tests/agent/test_model_performance_router.py tests/hermes_cli/test_profile_route_compiler.py
git commit -m "feat(routing): compile every profile model lane"
```

### Task 6: Runtime circuit breaker, context admission, and weighted concurrency

**Files:**
- Create: `agent/model_circuit_breaker.py`
- Create: `agent/model_admission.py`
- Modify: `agent/auxiliary_client.py`
- Modify: `agent/conversation_loop.py`
- Modify: `tests/agent/test_auxiliary_transient_retry.py`
- Create: `tests/agent/test_model_admission.py`

**Interfaces:**
- Consumes: `ResolvedRoute`.
- Produces: `ModelCircuitBreaker`, `WeightedModelAdmission`, `strict_runtime_present() -> bool`, and bounded route execution.

- [ ] **Step 1: Write failing stale-loop, half-open, 64K/120K context, and protected-runtime tests**

```python
def test_two_stale_attempts_open_circuit_and_never_reconnect_for_900_seconds():
    first = executor.call(stalling_route)
    second = executor.call(stalling_route)
    assert first.elapsed < route.total_timeout_seconds
    assert second.outcome == "circuit_open"

def test_strict_runtime_reduces_local_weight_without_starting_or_stopping_it(monkeypatch):
    monkeypatch.setattr(admission, "strict_runtime_present", lambda: True)
    assert admission.local_capacity() == protected_capacity
    assert process_control.calls == []
```

- [ ] **Step 2: Run runtime tests and reproduce current long reconnect behavior**

Run: `pytest -q tests/agent/test_auxiliary_transient_retry.py tests/agent/test_model_admission.py`

- [ ] **Step 3: Implement keyed circuits, one-retry maximum, context-floor enforcement, and weighted semaphores**

```python
key = (route.provider, route.model, surface, route.endpoint)
if circuit.is_open(key, now):
    raise ModelCircuitOpen(key)
async with admission.acquire(route.concurrency_weight):
    return await call_with_deadlines(route)
```

Context admission rejects routes below the required window; it never overrides a detected context length merely to pass. The configured 120K cap becomes a benchmarked per-surface budget rather than a universal hard cap.

- [ ] **Step 4: Run runtime and compression regression tests**

Run: `pytest -q tests/agent/test_auxiliary_transient_retry.py tests/agent/test_model_admission.py tests/agent/test_auxiliary_compression_timeout_floor.py tests/agent/test_context_compressor.py`

- [ ] **Step 5: Commit runtime controls**

```bash
git add agent/model_circuit_breaker.py agent/model_admission.py agent/auxiliary_client.py agent/conversation_loop.py tests/agent/test_auxiliary_transient_retry.py tests/agent/test_model_admission.py
git commit -m "perf(routing): bound stalls and model admission"
```

### Task 7: Operator benchmark command, canary deployment, and acceptance report

**Files:**
- Create: `hermes_cli/model_benchmark.py`
- Modify: `hermes_cli/main.py`
- Create: `docs/model-performance-routing.md`
- Create: `tests/hermes_cli/test_model_benchmark.py`

**Interfaces:**
- Consumes: all earlier benchmark and runtime interfaces.
- Produces: `hermes model benchmark --all --repetitions N --output PATH`, `--apply`, and a content-addressed report.

- [ ] **Step 1: Write failing dry-run, exhaustive inventory, apply-gate, and atomic-rollback tests**

```python
def test_apply_refuses_partial_inventory(tmp_path):
    result = cli("model", "benchmark", "--apply", artifact_missing_local_model)
    assert result.exit_code == 1
    assert "inventory_incomplete" in result.stdout

def test_report_has_every_profile_and_required_surface(report):
    assert report.inventory_complete is True
    assert set(report.compiled_routes) == set(all_profiles())
```

- [ ] **Step 2: Run CLI tests and verify command is absent**

Run: `pytest -q tests/hermes_cli/test_model_benchmark.py`

- [ ] **Step 3: Implement read-only benchmark by default and explicit atomic `--apply`**

```text
hermes model benchmark --all --repetitions 3 --output artifacts/model-router/run.json
hermes model benchmark --apply artifacts/model-router/run.json
```

`--apply` requires complete inventory, passing acceptance gates, artifact digest verification, a retained previous route table, and no active protected runtime configuration mutation.

- [ ] **Step 4: Run controlled live benchmarks in waves and compile the route table**

Run local candidates one at a time first, then the highest stable concurrency. Run currently callable free Nous models and approved GPT/Codex candidates with identical cases and repetitions. Record unavailable, rate-limited, or unsupported candidates without dropping them.

Expected: the report includes all installed local models, all live free Nous models, GPT/Codex candidates, and Spark only when callable.

- [ ] **Step 5: Canary title, compression, and read-only review routes; then run acceptance tests**

Run: `pytest -q tests/evals tests/agent/test_model_performance_router.py tests/agent/test_model_admission.py tests/hermes_cli/test_profile_route_compiler.py tests/hermes_cli/test_model_benchmark.py`

Verify from runtime receipts that no-output calls stop within their route deadline, privacy fallbacks stay local, and every profile row binds the applied artifact digest.

- [ ] **Step 6: Commit CLI, documentation, and non-secret routing artifact**

```bash
git add hermes_cli/model_benchmark.py hermes_cli/main.py docs/model-performance-routing.md tests/hermes_cli/test_model_benchmark.py
git commit -m "feat(routing): benchmark and apply profile routes"
```
