# Team B — AVO-Inspired Recursive Learning Provider

**Project:** Shared-Memory, Self-Evaluating Agent Harness (Hermes fork)  
**Owner:** Team B (two members: B1 and B2)  
**Duration:** 9 build weeks + 1 spare week  
**Status:** Implementation plan; update at weekly syncs

> **Finalized architecture decision:** Team B develops an **independent NVIDIA AVO-inspired learning system**. Team A independently develops a Rust memory engine, Prime-inspired context formation, and a Prime-inspired learning method. Neither team's implementation or evaluation depends on the other. At final integration, the harness may select **Prime-inspired learning** or **AVO-inspired learning** through a configurable provider interface. Both can coexist in the repository; only the selected learning provider controls a given learning run.

---

## 1. Project thesis and scope

Hermes already supplies an agent loop, tools, model access, and task execution. Team B builds a separate learning provider that uses **task outcomes and persistent candidate lineage** to improve a versioned agent artifact over successive runs. An autonomous variation agent proposes an edit, a scorer evaluates it, and a gate accepts or rejects it. Future runs use the accepted artifact.

**Core claim to test:** An AVO-inspired evaluation-and-variation loop can produce persistent, measurable improvements over a fixed baseline, including on held-out tasks, without depending on Team A's memory engine or learning implementation.

### In scope

- Benchmark task suite with objective checks, training and held-out splits
- Task execution adapter and structured trajectory capture
- Deterministic scorer and repeatable baseline measurement
- Autonomous artifact variation and safe sandboxed application of edits
- Append-only lineage of **all** candidates, including rejected ones
- Regression protection, rollback, and learning-cost accounting
- A standalone runner, usable before any Hermes integration
- A thin, optional adapter to the harness's `LearningProvider` selection mechanism

### Out of scope

- Rust memory daemon, embedding model, indexing, retrieval, and session memory
- Prime-inspired context formation or Prime-inspired learning
- Rewriting Hermes's core agent loop or training/fine-tuning model weights
- Population evolution, branching archives, or optimizing several artifact types simultaneously
- Depending on Team A's APIs, data schema, storage, or delivery schedule
- Using an LLM as the primary scorer when deterministic tests are possible

**Important distinction:** AVO is an inspiration, not an implementation claim. NVIDIA AVO concerns iterative optimization of CUDA artifacts with a scorer and lineage. Here the optimized artifact is an **agent behavior artifact** (initial recommendation: a versioned skill/instruction bundle), and the evaluation domain is checkable agent tasks.

---

## 2. Final team boundaries

| Area | Team A | Team B |
|---|---|---|
| Memory and retrieval | Rust memory engine | No ownership or dependency |
| Context formation | Prime-inspired context formation | Uses fixed baseline task context for fair tests |
| Learning | Prime-inspired learning method | **AVO-inspired recursive evaluation and improvement** |
| Learning persistence | Team A's own design | **Own lineage, artifacts, evaluation history, and failure summaries** |
| Benchmarks | Team A's independent measurements | **Own benchmark, scorer, baseline, held-out evaluation** |
| Integration | Optional provider adapter | Optional provider adapter |

### Independence rules

1. Team B can run its entire learning loop on **unmodified/pinned Hermes** or a standalone compatible task runner.
2. Team B stores its own learning state under `/evolve` and `/bench`; it does not write into Team A's Rust memory database.
3. Team B must not call Team A's context builder, retrieval service, or Prime learning method to generate, evaluate, or select candidates.
4. Team B owns its baseline, scorer, tasks, results, and report; it can be graded and demonstrated alone.
5. Optional integration occurs **after** Team B's standalone acceptance tests pass. Integration failure does not invalidate Team B's contribution.
6. A shared `LearningProvider` adapter is a narrow interoperability contract, **not** a shared learning algorithm. The two providers remain replaceable and independent.

---

## 3. Architecture

```text
                   Standalone Team B runtime
              (pinned Hermes + task runner)
                              |
                              v
                  +-----------------------+
                  | Execute benchmark task|
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Trajectory + outcome  |----> immutable run log
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Deterministic scorer  |----> ScoreReport
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Failure/score summary |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | AVO-inspired variation|
                  | agent reads lineage   |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Candidate artifact    |----> version + diff
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Train evaluation +    |
                  | regression gate       |
                  +------+----------+-----+
                         |          |
                      Accept      Reject
                         |          |
                         v          v
                    Promote      Roll back
                         \          /
                          v        v
                  +-----------------------+
                  | Append-only lineage   |
                  +-----------------------+
                              |
                     Next generation
```

### 3.1 What gets learned?

**One versioned artifact type is selected before implementation.** Recommended v1: a bounded **skill/instruction bundle** that Hermes can load for the benchmark tasks. The learning system modifies **files**, not the model's weights. Each accepted version is reused by future task executions. The source README left the choice between skills and policy config open; Team B must confirm the final choice at the first meeting.

The artifact must have an explicit allowed directory, file size limits, a version identifier, and a reproducible way to load it into the runner. The scorer and test suite are never editable by the variation agent.

### 3.2 The learning cycle

1. **Initialize:** snapshot candidate `v0`, pin the runner/model settings, and run baseline tasks.
2. **Observe:** collect task trajectories, test failures, score reports, and previous candidate lineage.
3. **Propose:** the variation agent reads **training-only** feedback and produces a bounded patch to the artifact.
4. **Validate:** check patch scope, syntax, artifact size, and any safety constraints before running it.
5. **Evaluate:** run the candidate on the training suite with identical settings and repeated trials as needed.
6. **Gate:** require correctness and non-regression; accept only if a predeclared primary metric improves sufficiently. Otherwise reject and restore the parent.
7. **Persist:** save candidate ID, parent ID, diff, test results, decision, reason, costs, and timestamp even when rejected.
8. **Repeat:** the next generation starts from the latest **accepted** candidate and its lineage.
9. **Final evaluation:** assess the frozen accepted artifact on the held-out split only after development decisions are finished.

### 3.3 What is stored, and where?

| Record | Example contents | Proposed location |
|---|---|---|
| Artifact version | skill files, content hash, parent version | `/evolve/artifacts/` |
| Lineage record | candidate, parent, diff, score, accepted/rejected, reason, timestamp | `/evolve/state/lineage.jsonl` |
| Run trajectory | steps, tool calls, output references, token use, outcome | `/bench/runs/` |
| Score report | task ID, pass/fail, metrics, failure reason, candidate ID | `/bench/results/` |
| Failure/knowledge summary | recurring failure patterns, previous attempts, useful fixes | `/evolve/state/knowledge/` |
| Experiment configuration | model, seed if supported, temperature, task split, runner revision | `/bench/config/` |

These paths are **proposals**, not claims that files already exist. Use append-only records or transactional writes, stable IDs, and content hashes so every result can be reproduced and traced.

### 3.4 Learning signal and acceptance policy

**Primary metric:** task success rate on the **training** set, with deterministic pass/fail checks.  
**Secondary metrics:** tokens, tool steps, wall-clock time, and learning cost.

Candidate acceptance should require:

- All safety and artifact validation checks pass.
- No unacceptable regression on the previously passing task subset (define tolerance in advance).
- A reproducible improvement in the predeclared primary metric over the accepted parent; handle ties explicitly rather than silently optimizing on held-out results.
- No access to held-out task descriptions, solutions, or scores during variation and selection.

An edit that makes one task pass but breaks other tasks is **not automatically accepted**. A failed candidate remains in lineage so the variation agent can avoid repeating it. Because LLM runs can vary, report repeated-trial mean and spread and avoid treating a single lucky pass as proof of improvement.

### 3.5 AVO concepts mapped to Team B

| AVO-inspired idea | Team B implementation |
|---|---|
| Autonomous improvement agent | LLM-based artifact variation agent |
| Scoring function `f` | Objective task checks + declared aggregate metric |
| Full lineage | Parent/child versions, diffs, scores, rejection reasons |
| Knowledge base | Training failure summaries and previous attempts |
| Iterative optimization | Propose → validate → evaluate → accept/reject |
| Optimized artifact | One agent skill/instruction bundle (pending confirmation) |

---

## 4. Repository layout (proposed)

```text
/bench/
  tasks/                 # tasks and deterministic validators
  config/                # model, runner, and split manifests
  runner/                # Hermes execution adapter
  scorer/                # ScoreReport generation and aggregation
  trajectories/          # trajectory schema and extraction
  results/               # machine-readable scores
  runs/                  # per-task run logs
  tests/                 # scorer and runner tests

/evolve/
  artifacts/             # immutable candidate snapshots
  variation/             # candidate-generation agent and prompt
  lineage/               # append-only history and querying
  guard/                 # scope validation, regression, rollback
  state/                 # lineage records and learning summaries
  provider/              # optional LearningProvider adapter
  tests/                 # mock-scorer and evolution-loop tests

/interfaces/             # optional provider contract; shared by agreement
/docs/team-b/           # protocol, attribution, plots, final report
```

The baseline Hermes checkout should remain untouched. All modifications to Hermes must be optional, minimal integration hooks; do not replace the core task runner or remove its existing functionality.

---

## 5. Contracts: B1 ↔ B2

Agree on these in **days 1–2 of week 1**, then freeze the interface by the end of week 2.

### 5.1 `ScoreReport`

```json
{
  "schema_version": 1,
  "run_id": "run-001",
  "task_id": "task-001",
  "candidate_id": "candidate-000",
  "split": "train",
  "passed": true,
  "score": 1.0,
  "tokens": 2400,
  "steps": 8,
  "time_seconds": 21.5,
  "failure_reason": null,
  "trajectory_ref": "bench/runs/run-001.jsonl"
}
```

The schema above is **illustrative**. The real contract must specify field types, score direction, aggregate rules, missing-value handling, and schema versioning.

### 5.2 Runner contract

```bash
bench run --candidate evolve/artifacts/candidate-000 --split train
bench run --candidate evolve/artifacts/candidate-000 --split heldout
```

These are **target CLI commands to implement**, not existing commands. The runner outputs one `ScoreReport` per task and a run manifest recording the exact settings. Only the **final evaluation process** may invoke the held-out split; the variation agent and its tools must not be able to do so.

### 5.3 Lineage contract

Each candidate records at least `candidate_id`, `parent_id`, `artifact_hash`, `diff_ref`, `score_report_refs`, `accepted`, `decision_reason`, `created_at`, and `learning_tokens`. Preserve rejected candidates and allow restoring any accepted version.

### 5.4 Mock-first rule

B2 starts with `MockScorer` returning fixed fixture `ScoreReport`s. B1 starts with version-0 Hermes and a fixed artifact. Neither member waits for the other's implementation.

---

## 6. Team members and responsibilities

| | **B1 — Evaluation (the judge)** | **B2 — Learning (the improver)** |
|---|---|---|
| Owns | `/bench/` | `/evolve/` |
| Core work | Tasks, validators, runner, scoring, trajectories, split integrity | Variation agent, lineage, artifact versions, regression guard, rollback |
| Measurement | Baseline, repeated runs, held-out performance, statistics | Learning curve, rejected edits, learning tokens, ablation |
| Report | Benchmark protocol, scoring validity, baseline and generalization | Recursive loop, candidate lineage, learning cost and safety |
| Independence | Can run `v0` and score it without B2 | Can evolve against fixture reports without B1 |

Both members contribute code, tests, measurements, documentation, and an individually attributable report section.

---

## 7. Week-by-week implementation plan

| Week | B1 — Evaluation | B2 — Learning | Checkpoint |
|---|---|---|---|
| **1** | Inspect Hermes trajectory format; draft ~20 objectively checkable tasks; define score schema | Read AVO paper; confirm one evolving artifact; define lineage schema; build `MockScorer` | Contract committed by day 2 |
| **2** | Run scorer on untouched Hermes; finalize 15–20 tasks with train/held-out split; capture trajectories | Implement lineage store, immutable artifact versions, patch apply/rollback | **Baseline and interface freeze** |
| **3** | Repeat baseline ≥3 times; measure variability and remove flaky validators | Variation agent proposes a bounded edit using fixture feedback; logs candidate | Candidate generated and traceable |
| **4** | Reliable candidate runner and structured failure summaries | Connect real scorer; unattended propose → test → accept/reject cycle | **Full standalone cycle** |
| **5** | Enforce held-out isolation; improve failure summaries | Multi-generation experiment; first score-vs-generation plot; regression guard | Independent Team B demo |
| **6** | Harden runner and scoring reproducibility; run integration smoke tests **only if available** | Implement optional learning-provider adapter; keep standalone runner working | Optional integration, never a blocker |
| **7** | Final frozen-artifact held-out evaluation, repeated runs and statistics | Account for learning tokens, time, attempts per accepted improvement; compare to baseline | Learning and generalization evidence |
| **8** | Robustness checks; rerun under fixed protocol | Lineage vs no-lineage ablation; longer evolution experiment | Measurements complete |
| **9** | Final benchmark report, plots, evaluation demo | Final learning report, lineage demo, rollback demonstration | Final submission |
| **10** | Spare: fixes or reruns | Spare: fixes or reruns | Buffer |

**If B1 is delayed:** B2 continues against deterministic fixtures; it cannot claim real learning results until the scorer exists.  
**If B2 is delayed:** B1's baseline, task suite, scorer, and repeatability analysis remain independently demonstrable.  
**If Team A is delayed:** Team B's schedule and deliverables **do not change**.

---

## 8. Optional final integration: selectable learning providers

The combined harness may expose a **configuration switch**:

```yaml
learning:
  provider: avo       # alternative: prime
  enabled: true
```

This is a **proposed configuration**, not an existing Hermes setting.

```text
                  Hermes agent core
                         |
                LearningProvider API
                   /           \
                  v             v
       Prime-inspired      AVO-inspired
       provider (Team A)   provider (Team B)
                  \             /
                   Selected by config
```

### Minimal adapter idea

```python
class LearningProvider:
    def observe(self, trajectory, outcome): ...
    def improve(self, evaluation_context): ...
    def get_active_artifact(self): ...
```

This is **pseudocode for an interface to agree on**, not a claim about existing classes. Team B implements a thin adapter around its standalone learning service; it does not move its evaluation, lineage, or variation logic into Team A's code. The adapter may return an artifact path/version for the harness to load. The two providers may maintain different internal data and schedules; the interface should not force identical learning mechanics.

**Selection rule:** use one active learning provider per run, record its name and version in the run manifest, and avoid concurrent writes to the same skill artifact. A `none`/disabled option is useful for clean baselines. Do not silently mix results from the two providers.

---

## 9. Evaluation protocol and success criteria

### Baselines

- **Primary baseline:** unmodified pinned Hermes, same model, task set, machine, tool permissions, context budget, and execution limits.
- **Version-0 baseline:** Team B's runner with learning disabled and the initial artifact; use it to isolate the learning loop's effect from runner changes.
- **Optional comparator:** Hermes's native passive review, measured separately and clearly labeled, when its triggering behavior is verified.

Record the exact Hermes commit, provider/model, temperature, seeds where available, prompts, task versions, and hardware. The project's previously chosen `openai-codex` OAuth route and `gpt-6-luna` model should be recorded **only if confirmed available in the actual environment**; do not substitute a model silently.

### Metrics

| Metric | What it demonstrates |
|---|---|
| Training score vs generation | Whether accepted artifacts improve on the optimization tasks |
| Held-out success rate vs v0 | Whether improvement generalizes beyond tasks seen during variation |
| Previously passing tasks retained | Regression protection |
| Accepted/rejected candidate count | Search behavior and auditability |
| Variation + evaluation tokens per accepted improvement | True cost of learning |
| Wall-clock time and tool steps | Runtime trade-offs |
| With-lineage vs without-lineage ablation | Whether persistent history contributes to improvement |

Run each reported configuration at least **3 times** where feasible; report mean and spread, not only best-case results. Freeze the held-out set before evolution and keep it inaccessible to the variation agent. Never tune acceptance thresholds after inspecting held-out scores.

### Definition of done

1. The benchmark suite and scorer run reproducibly against the pinned baseline.
2. The learner completes multiple unattended propose/evaluate/accept-or-reject cycles.
3. Every candidate, including rejected candidates, has a retrievable lineage record and can be audited.
4. Accepted artifact versions can be reloaded and rolled back safely.
5. There is a measured comparison to v0 on both training and held-out tasks; **report lack of improvement honestly**.
6. The whole system runs without Team A code or services.
7. The optional provider adapter, if built, does not change standalone behavior.
8. Documentation, attribution, and per-member contributions are complete.

---

## 10. Safety and engineering rules

- **Scope changes:** variation may edit only the allowlisted artifact directory; never the scorer, benchmark tests, credentials, harness code, or lineage history.
- **Sandbox executions:** run candidate artifacts with bounded permissions, timeouts, token budgets, and resource limits; do not allow arbitrary privileged operations.
- **Immutable evidence:** record all run manifests, diffs, scores, and accept/reject decisions; preserve the untouched baseline.
- **No data leakage:** training-only context goes to the variation agent; held-out materials and results remain sealed until final evaluation.
- **Fair comparisons:** pin configuration, use the same tasks and limits, and label any unavoidable differences.
- **Minimal integration:** avoid rewriting Hermes internals. The standalone loop is the graded contribution.
- **Licensing:** attribute Hermes and the NVIDIA AVO paper; mark reused code in `/docs/ATTRIBUTION.md`.
- **Claims:** say *AVO-inspired*, not *NVIDIA AVO implemented*; do not claim token savings without measurement.

---

## 11. Risks, mitigations, and cut list

| Risk | Mitigation |
|---|---|
| Scorer is flaky or rewards the wrong behavior | Deterministic validators, repeated baseline, task review |
| Learner overfits training tasks | Held-out isolation, fixed split, final frozen-artifact evaluation |
| Variation edits dangerous files | Allowlist, sandbox, diff validation, rollback |
| Improvements are random noise | Repeated runs, aggregate metrics, documented acceptance thresholds |
| Learning uses more tokens than it saves | Report total learning cost and tokens per accepted improvement |
| Team A integration is late | Keep Team B standalone; optional adapter only |
| No accepted improvements | Preserve lineage, analyze failure modes, report negative results |
| Benchmark execution infrastructure is blocked | Begin with small deterministic local tasks; document blocked external benchmarks separately |

**Cut in this order if behind:** (1) extra ablations, (2) elaborate failure-summary features, (3) optional integration polish, (4) reduce suite to 12 checkable tasks while retaining a sealed held-out split. **Never cut** the scorer, lineage, standalone cycle, or held-out evaluation.

---

## 12. Deliverables

- `bench/`: runnable task suite, deterministic scorer, runner, trajectories, baseline reports
- `evolve/`: variation agent, artifact snapshots, lineage store, regression guard, rollback
- Automated tests for scorer correctness, lineage integrity, edit scope, and provider independence
- Multi-generation learning plot and held-out comparison
- Cost accounting and at least one lineage ablation if time permits
- Standalone demonstration; optional config-selected integration demonstration
- Team B report and per-member contribution log

### Demonstration script

1. Run pinned Hermes with `v0`; show baseline scores.
2. Launch the Team B learner with no Team A service running.
3. Show one proposed candidate, its diff, scorer output, and accept/reject decision.
4. Show persistent lineage across multiple generations and restore a previous version.
5. Show training improvement curve, final held-out results, and learning cost.
6. **Optional:** switch harness learning provider between `prime` and `avo`, verifying that the selected provider alone runs.

---

## 13. Open decisions for Team B's first meeting

| Decision | Owner | Deadline |
|---|---|---|
| Final artifact: skill/instruction bundle or policy config | B1 + B2 | Week 1, day 2 |
| Task domain and deterministic validators | B1 | Week 1 |
| `ScoreReport` and lineage schema | B1 + B2 | Week 1, day 2 |
| Acceptance metric, regression tolerance, tie policy | B1 + B2 | Week 2 |
| Hermes commit, model, temperature, execution limits | B1 + B2 | Before baseline |
| Standalone runner invocation and artifact-loading mechanism | B1 | Week 2 |
| Provider adapter shape (only if integration proceeds) | Both teams | After standalone milestone |

---

## 14. References and verification

- **Hermes Agent (Nous Research):** baseline harness and agent runtime. Verify actual hooks and trajectory formats in the pinned fork before coding against them.
- **NVIDIA AVO:** paper identified in the source project README as *arXiv:2603.24517*. Re-read the primary paper and verify its lineage/scoring claims before citing implementation details.
- **Source project README:** *Shared-Memory, Self-Evaluating Agent Harness*, especially Team B architecture, pair split, evaluation rules, and weekly milestones.

**Verification status:** This README is a proposed Team B implementation plan based on the source README and the newly finalized team separation. File paths, CLI commands, JSON schemas, provider config, and Python interface are design proposals, not verified existing implementations.
