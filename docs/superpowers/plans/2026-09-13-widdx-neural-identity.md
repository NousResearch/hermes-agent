# WIDDX Neural Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce WIDDX branding and a safe, dependency-free neural foundation without breaking existing Hermes behavior.

**Architecture:** Keep the existing Hermes runtime as the stable execution core and add an isolated `neural/` package connected through best-effort in-process events. Branding is changed only where it is safe and clearly owned by this fork; legacy `hermes` commands, paths, and compatibility references remain supported during migration.

**Tech Stack:** Existing Python runtime, dataclasses/stdlib for neural primitives, existing pytest suite, GitHub repository contents API for this implementation session.

**Spec:** `docs/superpowers/specs/2026-09-13-widdx-neural-identity.md`

## Global Constraints

- Preserve current Hermes CLI/config compatibility.
- Do not add external dependencies for the neural foundation.
- Neural failures must not break the existing agent loop.
- Neural learning must never modify security boundaries or approval policy.
- Do not claim or perform the GitHub repository rename from code; repository administration is a separate manual GitHub action.

---

### Task 1: Add neural event primitives

**Files:**
- Create: `neural/__init__.py`
- Create: `neural/events/__init__.py`
- Create: `neural/events/types.py`
- Test: `tests/neural/test_events.py`

**Interfaces:**
- Produces `NeuralEvent` and `NeuralSignal` immutable value objects with IDs, timestamps, source/type, payload, importance, confidence, and correlation metadata.

- [ ] Write failing tests for construction, defaults, and immutability.
- [ ] Run the focused tests and confirm failure.
- [ ] Implement minimal dataclasses using only the standard library.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit the event primitive as an isolated change.

### Task 2: Add the neural bus

**Files:**
- Create: `neural/bus/__init__.py`
- Create: `neural/bus/in_process.py`
- Test: `tests/neural/test_bus.py`

**Interfaces:**
- Produces an in-process `NeuralBus` with `subscribe(event_type, handler)`, `publish(event)`, and safe unsubscribe behavior.

- [ ] Write failing tests for delivery, multiple subscribers, unsubscribe, and handler isolation.
- [ ] Run focused tests and confirm failure.
- [ ] Implement a synchronous, bounded, best-effort bus.
- [ ] Ensure subscriber exceptions do not propagate into the caller.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit the bus implementation.

### Task 3: Add neuron and synapse contracts

**Files:**
- Create: `neural/neurons/__init__.py`
- Create: `neural/neurons/base.py`
- Create: `neural/synapses/__init__.py`
- Create: `neural/synapses/model.py`
- Test: `tests/neural/test_neurons_synapses.py`

**Interfaces:**
- Produces a small `Neuron` protocol/base contract and a bounded `Synapse` model containing weight, confidence, usage, and outcome statistics.

- [ ] Write failing tests for deterministic activation and bounded synapse updates.
- [ ] Run focused tests and confirm failure.
- [ ] Implement minimal contracts and clamped reinforcement updates.
- [ ] Explicitly prevent the model from representing security-policy mutation.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit the contracts.

### Task 4: Add perception/reflex/guardian boundaries

**Files:**
- Create: `neural/perception/__init__.py`
- Create: `neural/reflexes/__init__.py`
- Create: `neural/guardian/__init__.py`
- Test: `tests/neural/test_safety_boundaries.py`

**Interfaces:**
- Produces small adapters/contracts for turning runtime observations into events and for deterministic risk/reflex observations.

- [ ] Write failing tests showing neural observations cannot approve or unblock a forbidden action.
- [ ] Run focused tests and confirm failure.
- [ ] Implement observation-only boundaries.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit the safety boundaries.

### Task 5: Add memory/brain/self-model extension points

**Files:**
- Create: `neural/memory/__init__.py`
- Create: `neural/brain/__init__.py`
- Create: `neural/self_model/__init__.py`
- Test: `tests/neural/test_extension_points.py`

**Interfaces:**
- Produces explicit interfaces only; no persistent storage or LLM orchestration yet.

- [ ] Write failing tests for interface importability and basic lifecycle contracts.
- [ ] Run focused tests and confirm failure.
- [ ] Implement minimal protocols/interfaces with no side effects.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit the extension points.

### Task 6: Introduce WIDDX branding safely

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: the project’s user-facing CLI identity file discovered by repository search
- Test: existing metadata/CLI tests plus targeted new branding test if an established test location exists

**Interfaces:**
- Keeps package import/module compatibility and the `hermes` CLI entry point intact while introducing WIDDX as the product name.

- [ ] Search all user-facing Hermes branding before editing.
- [ ] Write/update tests for the intended WIDDX display name where the codebase already has a test seam.
- [ ] Update safe product metadata and README copy, avoiding upstream-owned installation URLs that this fork does not control.
- [ ] Run targeted tests and repository lint/format checks.
- [ ] Commit branding changes.

### Task 7: Verify the migration boundary

**Files:**
- Modify: only files needed to correct verification findings.

- [ ] Run the neural test suite.
- [ ] Run the project’s documented test command or the narrowest equivalent available in the repository.
- [ ] Search for accidental destructive identity changes to `hermes` compatibility paths.
- [ ] Review the diff for security-boundary changes.
- [ ] Record any environment-limited checks honestly.
- [ ] Commit only after verification passes.
