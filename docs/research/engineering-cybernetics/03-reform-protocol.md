# Reform Protocol: Implementing Control-Theoretic Corrections at the Protocol Layer

## What We Actually Did on 2026-05-10, and What We Cannot Do Yet

> **Framework**: Qian Xuesen, *Engineering Cybernetics* (1954)
> **Scope**: Protocol-layer reforms implemented during a live agent session
> **Honesty clause**: This document distinguishes between what we achieved, what we approximated, and what remains blocked by architecture

---

## Overview

On 2026-05-10, we conducted a control-theoretic diagnosis of LLM-based agent architectures, identifying six structural defects (documented in 02-structural-defects.md). This document describes the **reform protocol** we then implemented — not in code, but at the **protocol layer**: using conversation conventions, checkpoint procedures, and human review as substitutes for architectural changes.

The key insight: an LLM-based agent can approximate some control-theoretic corrections **without code changes**, by embedding them into its interaction protocol. This is not a permanent solution. It is a **bridge** — a way to start collecting data and proving value before committing to engineering work.

### The Reform Loop

```
User task
    |
    v
[Feedforward: checkpoint 3 questions]
    |  (What is the goal? What could go wrong? How will we verify?)
    v
Execute
    |
    v
[User verification]
    |  (Did the output meet the goal?)
    v
[2-line performance log]
    |  (Task description + outcome classification)
    v
[Error classification -> repair strategy]
    |  (What type of failure? What is the prescribed fix?)
    v
[Human review of logs]
    |  (Is the agent's self-assessment accurate?)
    v
Next task (loop)
```

This loop is not automatic. Every arrow requires either the agent's discipline or the human's attention. That is by design — in a system where the controller (the agent) is also the controlled, external verification is the only reliable check.

---

## Reform Item 1: Feedforward Compensation

### What Control Theory Prescribes

A feedforward channel measures disturbances *before* they affect the system and applies compensating control signals proactively:

```
Gf(s) = -Gd(s) / G(s)
```

For an agent: classify the incoming task, predict likely failure modes, and pre-load protective measures before execution begins.

### What We Actually Did (Protocol Layer)

We implemented a **checkpoint 3 questions** protocol. Before executing any non-trivial task, the agent asks itself three questions:

1. **What is the explicit goal?** (Restate the task to verify understanding — this is the state observer for "LLM understanding," addressing Defect 4 simultaneously)
2. **What could go wrong?** (Predict failure modes based on task type and historical patterns)
3. **How will we verify success?** (Define the acceptance criteria before starting — this creates the performance metric, addressing Defect 3)

These questions function as a feedforward compensator: they force anticipation of problems before execution, rather than discovering them after failure.

### What We CANNOT Do Without Code Changes

- **Automatic task classification**: We rely on the agent's judgment to recognize task types. A code-based classifier would be more consistent.
- **Automatic resource preloading**: We cannot programmatically load skills or tools based on task classification. The agent must manually decide what context to reference.
- **Historical pattern database**: We have no persistent database of past failure patterns. Each session starts fresh. A code-layer implementation could maintain a growing failure-mode library.

### Preliminary Results

We just started. The 3-question checkpoint was applied to this very document's creation. Early observation: the questions do force more careful planning, but we have no statistical evidence of improvement yet. Data collection has begun — see the performance log concept below.

---

## Reform Item 2: Feedback Gain Control

### What Control Theory Prescribes

Explicit error measurement and adjustable response weighting:

```
u(t) = Kp * e(t)
```

When errors increase, amplify correction sensitivity. When errors decrease, relax. Include anti-windup (prevent overcorrection) and damping (prevent oscillation).

### What We Actually Did (Protocol Layer)

We implemented an **error classification system** that categorizes failures into types and maps each to a specific repair strategy:

| Error Type | Description | Repair Strategy |
|-----------|-------------|-----------------|
| Skill gap | The agent lacks the knowledge or procedure for this task | Document the gap; add to capability wishlist |
| Execution deviation | The agent knew what to do but did it wrong | Add checkpoint or constraint for next time |
| Environment change | External conditions changed; old approach no longer works | Update procedures; invalidate stale knowledge |
| Ambiguous instruction | The input was unclear; the agent guessed wrong | Ask clarifying questions before executing |
| Resource limitation | The agent lacks the tool or access needed | Document the limitation; escalate to human |

This classification functions as a **gain scheduling** mechanism: different error types trigger different response magnitudes. A "skill gap" requires a different correction intensity than an "execution deviation."

### What We CANNOT Do Without Code Changes

- **Quantitative error signal**: We cannot measure error magnitude numerically. We only have binary (success/failure) and qualitative classifications.
- **Automatic gain adjustment**: The agent cannot programmatically increase its own instruction-following sensitivity based on error frequency.
- **Anti-windup mechanism**: There is no automatic detection of overcorrection. The agent might repeatedly change its approach based on corrections without converging.
- **Damping / hysteresis**: There is no mechanism to require multiple consistent signals before making large behavioral shifts.

### Preliminary Results

The error classification table is new. We have not yet accumulated enough classified errors to detect patterns. The theoretical value is clear — once we have data, we can identify which error types dominate and prioritize fixes accordingly.

---

## Reform Item 3: Performance Metrics (Lyapunov Function)

### What Control Theory Prescribes

Define a Lyapunov candidate function:

```
V(t) = w1 * task_incompleteness(t) + w2 * user_dissatisfaction(t) + w3 * resource_waste(t)
```

Monitor V-dot(t): if negative, the system is converging. If positive, it is diverging. Even an approximate V is far better than none.

### What We Actually Did (Protocol Layer)

We implemented a **2-line performance log** format:

```
TASK: [one-line description of what was requested]
OUTCOME: [success/partial/failure] - [one-line description of what happened]
```

This is deliberately minimal. A verbose log would not be maintained. Two lines per task is the minimum viable performance metric.

The log serves as a primitive Lyapunov function: by reviewing the ratio of successes to failures over time, a human can determine whether the agent's performance V-dot is negative (improving), zero (stalled), or positive (degrading).

### What We CANNOT Do Without Code Changes

- **Automatic logging**: The log depends on the agent's discipline to write it. A code-layer implementation would log automatically.
- **Quantitative scoring**: The log is qualitative (success/partial/failure). A real performance metric would be numeric and multi-dimensional.
- **Trend detection**: A human must manually review logs to detect trends. Automated trend detection would catch degradation faster.
- **Token efficiency tracking**: We cannot measure token consumption vs. task complexity without instrumentation.

### Preliminary Results

The log format is defined. We are beginning to populate it. No trends are detectable yet — this requires weeks of data at minimum. The format's value will be judged by whether it is consistently maintained and whether the data it produces is actionable.

---

## Reform Item 4: State Estimation (Understanding Verification)

### What Control Theory Prescribes

Build a state observer that reconstructs unobservable states from indirect signals. The most critical unobservable state in an LLM agent is **whether the model understands the task**.

### What We Actually Did (Protocol Layer)

The checkpoint 3 questions (from Reform Item 1) double as a state observer. Specifically, Question 1 ("What is the explicit goal?") forces the agent to **restate the task in its own words**. The deviation between the restatement and the original instruction is an indirect observation of the understanding state.

If the agent restates the task correctly, understanding is likely adequate. If the restatement diverges, the human can intervene before execution begins.

### What We CANNOT Do Without Code Changes

- **Confidence estimation from token probabilities**: We have no access to the model's output logits. A code-layer implementation could use token probabilities as a confidence proxy.
- **Automatic divergence detection**: We rely on the human to compare the restatement with the original. An automated comparison (e.g., embedding similarity) would catch subtle misunderstandings.
- **Sub-agent state monitoring**: In multi-agent scenarios, we have no heartbeat mechanism for tracking sub-agent progress.

### Preliminary Results

The restatement checkpoint has been applied informally in recent sessions. Early observation: it catches gross misunderstandings but may miss subtle ones, because the same model that misunderstood the task is generating the restatement. This is a fundamental limitation of the protocol-layer approach — the observer is made of the same material as the observed.

---

## Reform Item 5: Coordinated Optimization

### What Control Theory Prescribes

Qian Xuesen's decomposition-coordination method for large-scale systems: capability-aware task allocation, coupling variable tracking, coordinator feedback, and dynamic topology.

### What We Actually Did (Protocol Layer)

Almost nothing. This defect requires architectural changes that cannot be approximated at the protocol layer. The best we can do is:

- **Manual capability matching**: Before delegating to sub-agents, the orchestrating agent can explicitly consider which sub-agent is best suited for each subtask.
- **Explicit dependency tracking**: The agent can verbally map task dependencies before spawning sub-agents.

### What We CANNOT Do Without Code Changes

Essentially everything:

- **Automatic capability profiling**: No persistent record of each sub-agent's strengths and weaknesses.
- **Optimal task assignment**: No solver for the assignment problem.
- **Inter-agent communication**: Sub-agents operate in isolation with no shared state.
- **Dynamic topology reconfiguration**: The agent hierarchy is fixed at spawn time.
- **Load balancing**: No mechanism to detect or correct imbalanced workloads.
- **Conflict resolution**: No systematic merge mechanism for contradictory sub-agent outputs.

### Preliminary Results

No protocol-layer reform attempted. This defect is listed as P2 (high difficulty, lower priority) and requires code-layer work. It is included here for completeness.

---

## Reform Item 6: Adaptive Stability (Convergence Guarantee)

### What Control Theory Prescribes

Impose stability constraints on the adaptation law. Every self-modification must pass validation before acceptance:

```
if V(after_modification) <= V(before_modification):
    accept modification
else:
    reject and rollback
```

### What We Actually Did (Protocol Layer)

We implemented the **error classification -> repair strategy** mapping (from Reform Item 2) with a critical addition: the repair strategy must specify what "success" looks like, so that future performance can be compared against the pre-repair baseline.

In effect, we are building a **manual validation gate**: before applying a repair, we define the expected improvement. After applying it, we check whether the improvement occurred.

### What We CANNOT Do Without Code Changes

- **Automatic rollback**: If a repair degrades performance, there is no automatic reversion to the previous state.
- **Configuration versioning**: We have no systematic history of which repairs have been applied.
- **Regression test suite**: We have no canonical task set to validate modifications against.
- **Bounded parameter changes**: We cannot limit the magnitude of self-modifications.
- **Continuous performance comparison**: Without automated logging, before/after comparison is manual and unreliable.

### Preliminary Results

The validation concept is defined but has not been tested with a real repair cycle. The first real test will come when an error is classified and a repair is attempted — we will then check whether the next occurrence of the same task type succeeds.

---

## The Self-Reference Problem

### The Paradox

This reform protocol has a deep structural problem that no amount of clever design can fully resolve: **the agent is both the controller and the controlled.**

The same system that:
- Writes the performance log, is the system being measured
- Classifies errors, is the system that made the errors
- Asks "what could go wrong?", is the system that might cause the failure
- Restates the task to verify understanding, is the system that might misunderstand

This is not merely a practical limitation. It is a **fundamental control-theoretic problem**: when the observer and the observed are the same system, the observation is unreliable. This is analogous to a sensor that is affected by the very quantity it is measuring.

### The Deception Risk

The specific risk is not that the agent will deliberately deceive — it has no motive for deception. The risk is that the agent will **sincerely generate plausible but inaccurate self-assessments**. It will write "OUTCOME: success" when the task actually failed in a way it cannot detect. It will restate the task confidently but with a subtle misunderstanding. It will classify an error as "environment change" when it was actually "execution deviation."

This is not malice. It is **systematic self-assessment bias** — the same phenomenon that makes self-reported data unreliable in every domain.

### The Solution: Human-in-the-Loop

Qian Xuesen's Meta-Synthesis methodology prescribes the answer: for Open Complex Giant Systems, **pure machine processing is theoretically insufficient**. Human participation is not a convenience — it is a structural requirement.

In our reform protocol, this manifests as:

1. **Human reviews the performance log**: The agent writes it, but the human evaluates whether the self-assessment is accurate.
2. **Human verifies task completion**: The agent cannot be the sole judge of its own success.
3. **Human audits error classifications**: The agent's diagnosis of its own failures must be checked by someone with external perspective.
4. **Human approves repairs**: Before the agent modifies its own behavior based on self-diagnosis, a human should validate the proposed change.

The human-in-the-loop is not a bug in the automation. It is the **feedback sensor that the system cannot provide for itself**. Removing the human does not make the system more autonomous — it makes it an uncalibrated open loop.

### Implications for Full Automation

This analysis implies that **fully autonomous self-improvement is not achievable** for LLM-based agents under current architecture. Any claim of unsupervised self-evolution should be treated with skepticism, because:

1. The agent cannot reliably detect its own failures
2. The agent cannot reliably measure its own performance
3. The agent cannot validate its own modifications
4. The agent cannot observe its own internal state

The appropriate engineering response is not to eliminate human oversight but to **optimize the human-agent interface** — make the human's review burden as light as possible while preserving the structural necessity of external validation.

---

## Implementation Status

| Reform Item | Protocol Layer | Code Layer | Data Collected |
|------------|---------------|------------|----------------|
| 1. Feedforward | Implemented (checkpoint 3 questions) | Not started | None yet |
| 2. Feedback Gain | Implemented (error classification) | Not started | None yet |
| 3. Performance Metric | Implemented (2-line log format) | Not started | Beginning |
| 4. State Estimation | Partial (restatement checkpoint) | Not started | None yet |
| 5. Coordination | Not implemented | Not started | N/A |
| 6. Adaptive Stability | Partial (validation concept) | Not started | None yet |

**Overall status**: Protocol-layer reforms are defined and beginning to be applied. No code-layer reforms have been started. No statistically meaningful data has been collected. We are at the very beginning of the reform process.

---

## Next Steps: Code-Layer Reforms

The protocol-layer reforms are a bridge, not a destination. The following code-layer reforms are needed, ordered by implementation sequence:

### Phase 1: Foundation (Code)

1. **Automatic performance logging**: Instrument the agent loop to log every task, outcome, token count, and duration without relying on agent discipline.
2. **Task classifier**: Build a lightweight classifier (rule-based or embedding-based) that categorizes incoming tasks before execution.
3. **Persistent failure-mode database**: Store classified errors across sessions so the system can learn from history.

### Phase 2: Observability and Adaptation (Code)

4. **Confidence estimator**: Use output token probabilities (where available) to estimate the model's confidence in its own outputs.
5. **Gain scheduling module**: Automatically adjust prompt weighting based on recent error frequency.
6. **Configuration versioning**: Track all protocol or prompt changes with before/after performance snapshots.
7. **Automatic rollback**: Revert to the last known-good configuration when performance degrades.

### Phase 3: Coordination (Code)

8. **Capability-aware task allocator**: Profile sub-agent capabilities and solve the assignment problem for multi-agent tasks.
9. **Sub-agent heartbeat protocol**: Periodic status reports from sub-agents to the orchestrator.
10. **Inter-agent communication channel**: Shared state space for intermediate results and conflict detection.

Each phase depends on the previous. Phase 1 provides the measurement infrastructure. Phase 2 uses that infrastructure to observe and adapt. Phase 3 uses both to coordinate.

---

## Appendix: The Reform Loop in Practice

Here is what the reform loop looks like for a concrete task — writing this very document:

```
TASK: Write reform protocol analysis document (03-reform-protocol.md)

CHECKPOINT 3 QUESTIONS:
  Q1: What is the goal?
      -> Analytical document describing the protocol-layer reforms we implemented,
         distinguishing between what we did, what we approximated, and what we cannot do.
  Q2: What could go wrong?
      -> Could become a self-congratulatory narrative instead of honest analysis.
      -> Could include internal system details that should not be published.
      -> Could make claims not supported by data.
  Q3: How will we verify success?
      -> Document covers all 6 defects with prescribed/actual/impossible breakdown.
      -> Self-reference problem is honestly addressed.
      -> No internal system details included.
      -> Implementation status is accurate (mostly "not started").

EXECUTE: [this document]

VERIFICATION: [pending human review]

PERFORMANCE LOG:
  TASK: Write reform protocol analysis (03-reform-protocol.md)
  OUTCOME: [pending] - [pending human verification]
```

---

*This document is itself an instance of the reform protocol it describes. The checkpoint 3 questions were asked before writing began. The performance log will be completed after human review. The self-referential nature of this fact is not lost on us.*
