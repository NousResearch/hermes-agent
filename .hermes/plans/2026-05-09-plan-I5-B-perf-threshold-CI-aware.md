# Plan I5-B: Performance Threshold CI-Awareness

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make the memory-manager performance threshold tests CI-aware so they do not flake on slow CI runners while still guarding against local regressions.

**Architecture:** `tests/agent/test_memory_performance.py` contains 5 timing-based regression tests. One test (`test_add_ten_providers_under_5ms`) already skips under CI via `@pytest.mark.skipif(os.getenv("CI"), ...)`; the remaining four do not and are at risk of flaking on shared runners. The fix is to make every timing-sensitive test CI-aware by either skipping in CI or relaxing its threshold when `CI` is set.

**Tech Stack:** Python 3.11–3.13, pytest, hermes-agent `tests/agent/test_memory_performance.py`

---

## Verified State (Do Not Re-verify Root Cause)

The following is already correct and verified (parent task `t_51fa25b9`, run 898). Treat as ground truth.

### Current Test Inventory

| Test | Threshold | CI Skip? | Risk on CI |
|---|---|---|---|
| `test_add_provider_single_under_1ms` | `<1ms` | **No** | High — 1ms is tight on shared runners |
| `test_add_ten_providers_under_5ms` | `<5ms` | **Yes** (`skipif CI`) | Low — already skipped |
| `test_get_all_tool_schemas_under_1ms` | `<1ms` | **No** | High — 1ms is tight on shared runners |
| `test_remove_provider_under_1ms` | `<1ms` | **No** | High — 1ms is tight on shared runners |
| `test_concurrent_add_still_fast` | `<50ms` | **No** | Medium — 50ms is usually safe, but lock contention on shared runners can spike |

### Why This Matters

The workflow at `.github/workflows/tests.yml` runs `pytest tests/ -q --ignore=tests/integration --ignore=tests/e2e -n auto` on `ubuntu-latest` with 3 Python versions. GitHub Actions runners have variable CPU scheduling and noisy neighbours; single-millisecond thresholds are not statistically stable there. One or more of these tests can spuriously fail in a PR unrelated to the memory manager.

### Why No Code Changes Were Needed in Spec Phase

Parent spec task `t_51fa25b9` verified that **all 5 tests pass locally** (2.50s total with xdist 8 workers). Local passes do not imply CI passes; the spec task scoped itself to verifying correctness, not CI resilience.

---

## Task 1: Centralise CI-Detection Helper

**Objective:** Introduce a single source-of-truth for CI detection so decorators remain DRY and intention-revealing.

**Files:**
- Modify: `tests/agent/test_memory_performance.py`

**Step 1: Add module-level constant**

At the top of the file, after the imports and before `QuickProvider`, insert:

```python
IS_CI = bool(os.getenv("CI"))
```

The imports already include `import os`, so no new imports are required.

**Step 2: Verify**

```bash
cd /home/d/Desktop/agenda/hermes-agent
grep -n "IS_CI" tests/agent/test_memory_performance.py
```

Expected:
```
13:IS_CI = bool(os.getenv("CI"))
```

---

## Task 2: Add CI Skip Markers to the Fastest Tests

**Objective:** Decorators that skip in CI for tests with sub-millisecond local thresholds.

**Files:**
- Modify: `tests/agent/test_memory_performance.py`

**Step 1: Decorate `test_add_provider_single_under_1ms`**

Change:
```python
    def test_add_provider_single_under_1ms(self):
```

To:
```python
    @pytest.mark.skipif(IS_CI, reason="timing-sensitive, skipped in CI")
    def test_add_provider_single_under_1ms(self):
```

**Step 2: Decorate `test_get_all_tool_schemas_under_1ms`**

Change:
```python
    def test_get_all_tool_schemas_under_1ms(self):
```

To:
```python
    @pytest.mark.skipif(IS_CI, reason="timing-sensitive, skipped in CI")
    def test_get_all_tool_schemas_under_1ms(self):
```

**Step 3: Decorate `test_remove_provider_under_1ms`**

Change:
```python
    def test_remove_provider_under_1ms(self):
```

To:
```python
    @pytest.mark.skipif(IS_CI, reason="timing-sensitive, skipped in CI")
    def test_remove_provider_under_1ms(self):
```

Rationale: These three tests assert completion in under 1ms. That is a local-development performance contract, not a CI contract. Skipping them in CI is the same strategy already used for `test_add_ten_providers_under_5ms`.

---

## Task 3: Add CI-Aware Threshold to the Concurrent Test

**Objective:** Keep `test_concurrent_add_still_fast` running in CI by raising its ceiling when `CI` is set.

**Files:**
- Modify: `tests/agent/test_memory_performance.py`

**Step 1: Adjust the assertion block**

In `test_concurrent_add_still_fast`, locate:

```python
        # 20 providers, should complete in under 50ms even with lock contention
        assert elapsed < 0.05, f"20 concurrent adds took {elapsed*1000:.3f}ms, expected <50ms"
```

Replace with:

```python
        # 20 providers, should complete in under 50ms locally, under 200ms in CI
        threshold = 0.2 if IS_CI else 0.05
        assert elapsed < threshold, (
            f"20 concurrent adds took {elapsed*1000:.3f}ms, "
            f"expected <{threshold*1000:.0f}ms"
        )
```

Rationale: This test still exercises correctness under threading (`len(mgr.providers) == 20`) even if the timing relaxes. Flakes caused by lock contention on shared runners are eliminated by the larger CI ceiling while local regressions remain caught.

---

## Task 4: Verify No Local Regressions

**Objective:** Confirm all 5 tests still pass in a local environment (CI env unset).

**Files:**
- Read-only: `tests/agent/test_memory_performance.py`

**Step 1: Run the module**

```bash
cd /home/d/Desktop/agenda/hermes-agent
source .venv/bin/activate
pytest tests/agent/test_memory_performance.py -v --tb=short
```

**Expected output:**
```
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_add_provider_single_under_1ms PASSED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_add_ten_providers_under_5ms PASSED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_get_all_tool_schemas_under_1ms PASSED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_remove_provider_under_1ms PASSED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_concurrent_add_still_fast PASSED

5 passed in X.XXs
```

**Step 2: Simulate CI locally**

```bash
CI=true pytest tests/agent/test_memory_performance.py -v --tb=short
```

**Expected output:**
```
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_add_provider_single_under_1ms SKIPPED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_add_ten_providers_under_5ms SKIPPED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_get_all_tool_schemas_under_1ms SKIPPED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_remove_provider_under_1ms SKIPPED
tests/agent/test_memory_performance.py::TestMemoryManagerPerformance::test_concurrent_add_still_fast PASSED

1 passed, 4 skipped in X.XXs
```

**Step 3: If test counts differ**

Run `git diff tests/agent/test_memory_performance.py` to confirm the diff exactly matches Tasks 1–3. If counts differ, report findings in a `kanban_comment` and `kanban_block` the task.

---

## Task 5: Stage

**Objective:** Prepare the changes for downstream review/merge.

**Step 1: Stage**

```bash
cd /home/d/Desktop/agenda/hermes-agent
git add tests/agent/test_memory_performance.py
```

Do not commit. The orchestrator or a human will decide when to bundle this into a PR.

---

## Notes for Downstream Implementers

### Why not relax thresholds instead of skipping?

For the `<1ms` tests, the threshold **is** the contract. Relaxing to 5ms or 10ms in CI would make the test meaningless — it would catch nothing while still running. Skipping in CI preserves the tight local contract.

For the concurrent test, the threshold is already loose (50ms) but CI contention can exceed even that. A tiered threshold keeps the test running and meaningful in both environments.

### Why a module-level `IS_CI` instead of repeated `os.getenv("CI")`?

DRY. It also makes future changes (e.g., checking `GITHUB_ACTIONS` as well) a single-line edit.

### Why no changes to the CI workflow YAML?

The tests are discovered automatically by pytest; no test-selection changes are needed in `.github/workflows/tests.yml`. The skip logic lives inside the test module, which is the correct place for environment-dependent test behaviour.

---

## Files Referenced

| File | Role | Lines to change |
|---|---|---|
| `tests/agent/test_memory_performance.py` | Fix target | 13 (insert), 54, 78, 91, 120–121 (modify) |
| `.github/workflows/tests.yml` | Reference only | 55–61 |
