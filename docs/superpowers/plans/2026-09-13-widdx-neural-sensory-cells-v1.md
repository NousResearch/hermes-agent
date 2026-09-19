# WIDDX Neural Sensory Cells v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first explicit WIDDX sensory-cell and sense-to-signal pipeline while keeping neural code advisory and execution-free.

**Architecture:** Sensory cells normalize structured runtime observations into immutable `NeuralEvent` values. The neural runtime routes events through the existing bus to deterministic neurons and optional bounded synapses; no neural component executes tools or changes Hermes authorization.

**Tech Stack:** Python 3.11-3.13, dataclasses, typing protocols, pytest, existing Hermes neural primitives.

**Spec:** `docs/superpowers/specs/2026-09-13-widdx-neural-sensory-cells-v1-design.md`

## Global Constraints

- Keep the neural layer synchronous, in-process, dependency-light, and bounded.
- Sensory cells must be side-effect-free and execution-free.
- Neural signals are advisory and cannot grant permission or alter Hermes policy.
- Preserve existing public neural APIs unless the new API is additive.
- Use TDD and regression tests for all new behavior.
- Do not add network, subprocess, terminal, execute-code, credential, or LLM dependencies.

---

### Task 1: Harden the event payload boundary

**Files:**
- Modify: `neural/events/types.py`
- Test: `tests/neural/test_events.py`

**Interfaces:**
- Consumes: existing `NeuralEvent` constructor payload.
- Produces: an immutable event whose payload cannot be mutated through the caller's original mapping.

- [ ] **Step 1: Write the failing test**

Add a test that creates an event from a nested mutable dictionary, mutates both the original dictionary and a nested list after construction, and asserts the event payload is unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/neural/test_events.py -q`
Expected: the new test fails because the current event retains a mutable payload reference.

- [ ] **Step 3: Write minimal implementation**

Normalize the payload at construction using a deep immutable representation while preserving ordinary dictionary/list input semantics for readers. Do not introduce external dependencies or change event metadata behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/neural/test_events.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neural/events/types.py tests/neural/test_events.py
git commit -m "fix: isolate neural event payloads"
```

### Task 2: Add explicit sensory-cell contracts

**Files:**
- Create: `neural/perception/cells.py`
- Modify: `neural/perception/__init__.py`
- Test: `tests/neural/test_perception.py`

**Interfaces:**
- Consumes: structured observation mappings.
- Produces: `SensoryCell`, `ConversationCell`, `TaskCell`, `TerminalCell`, `ErrorCell`, and `EnvironmentCell` with `observe(...) -> NeuralEvent`.

- [ ] **Step 1: Write the failing tests**

Test each concrete cell emits its fixed event type, preserves source/name identity, accepts structured payloads, and performs no execution. Test invalid empty payload/source inputs raise `ValueError`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/neural/test_perception.py -q`
Expected: FAIL because the concrete cell API does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement a small immutable configuration contract and a shared event factory. Concrete cells should only call the event constructor; they must not import `tools`, `subprocess`, `os.system`, network clients, or gateway/approval modules.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/neural/test_perception.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neural/perception tests/neural/test_perception.py
git commit -m "feat: add explicit neural sensory cells"
```

### Task 3: Build the deterministic sense-to-signal processor

**Files:**
- Create: `neural/processing.py`
- Modify: `neural/runtime.py`
- Modify: `neural/neurons/base.py`
- Test: `tests/neural/test_processing.py`

**Interfaces:**
- Consumes: `NeuralEvent`, registered `Neuron` instances, optional `Synapse` instances.
- Produces: deterministic bounded `NeuralSignal` values and no execution side effects.

- [ ] **Step 1: Write the failing tests**

Test that a processor can register a neuron, process an event, return one bounded signal, and optionally reinforce a matching synapse. Test repeated processing is deterministic when synapse learning is disabled.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/neural/test_processing.py -q`
Expected: FAIL because the processor does not exist.

- [ ] **Step 3: Write minimal implementation**

Create a focused processor with explicit registration and `process(event)` behavior. Clamp activation before signal creation, isolate neuron failures, and never call any Hermes tool or policy API. Keep runtime integration additive.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/neural/test_processing.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neural/processing.py neural/runtime.py neural/neurons/base.py tests/neural/test_processing.py
git commit -m "feat: add deterministic neural processing"
```

### Task 4: Add neural boundary regression tests

**Files:**
- Modify: `tests/neural/test_safety_boundaries.py`
- Modify: `tests/neural/test_runtime.py`
- Modify: `tests/neural/test_bus.py`

**Interfaces:**
- Consumes: completed sensory and processing APIs.
- Produces: regression coverage for execution isolation, bus failure isolation, and safe observation.

- [ ] **Step 1: Write the failing boundary tests**

Add tests proving a sensory cell and processor can complete the full observation-to-signal path without importing or invoking terminal/execute-code facilities, and that one failing neural handler cannot prevent another handler from receiving an event.

- [ ] **Step 2: Run tests to verify they fail where the new behavior is missing**

Run: `pytest tests/neural -q`
Expected: new assertions fail until the integrated path is complete.

- [ ] **Step 3: Implement only the missing boundary wiring**

Keep all execution authority outside `neural/`. Do not add mocks that merely reproduce internal implementation details; assert observable behavior and module-level isolation.

- [ ] **Step 4: Run the full neural suite**

Run: `pytest tests/neural -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/neural
git commit -m "test: enforce neural execution boundaries"
```

### Task 5: Verify repository-wide quality

**Files:**
- No source changes unless a test exposes a direct defect in the touched implementation.

- [ ] **Step 1: Run neural tests**

Run: `pytest tests/neural -q`
Expected: all neural tests pass.

- [ ] **Step 2: Run project tests**

Run: `pytest -q`
Expected: the existing project suite passes.

- [ ] **Step 3: Run available static checks**

Run the repository's documented lint/type/format checks from `pyproject.toml`/CI configuration. Expected: no new failures attributable to the branch.

- [ ] **Step 4: Inspect the final diff**

Run: `git diff origin/main...HEAD -- neural tests/neural docs/superpowers`
Expected: only the designed neural implementation, tests, and design/plan documents are changed.

- [ ] **Step 5: Commit any required documentation update**

If existing neural documentation has a documented index or API summary that must reflect the new public classes, update only those references and commit with `docs: document neural sensory cells`.
