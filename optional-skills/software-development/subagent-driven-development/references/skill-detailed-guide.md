# Detailed guide

## Overview

Execute implementation plans by dispatching fresh subagents per task with systematic two-stage review.

**Core principle:** Fresh subagent per task + two-stage review (spec then quality) = high quality, fast iteration.

## When to Use

Use this skill when:
- You have an implementation plan (from writing-plans skill or user requirements)
- Tasks are mostly independent
- Quality and spec compliance are important
- You want automated review between tasks

**vs. manual execution:**
- Fresh context per task (no confusion from accumulated state)
- Automated review process catches issues early
- Consistent quality checks across all tasks
- Subagents can ask questions before starting work

## Codex-first model routing for implementation plans

Josh prefers Codex wherever practical for non-trivial coding. **Always use Codex for substantive coding work — choose model and reasoning effort deliberately per job.** This is an explicit user preference, not a suggestion.

### Model selection by task risk

| Task class | Model | Reasoning effort | Example |
|---|---|---|---|
| Hostile authority/receipt/permit review | `gpt-5.6-sol` | `high` | P0/P1 defect audit, permit trust boundary review |
| Mechanical multi-file refactors | `gpt-5.6-sol` | `medium` | Cross-crate identity migration, Clippy cleanup |
| Docs/config/scripts/simple tests | `gpt-5.6-sol` | `medium` | Gate scripts, fixture updates, README sync |
| Deep algorithmic/research work | `gpt-5.6-sol` | `high` | Quantization math, kernel design |

Probe live availability, then prefer the newest OAuth-accessible models:

- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-5.6-luna`

Pin the chosen model explicitly per phase. If a model reports that it requires a newer Codex CLI, upgrade Codex and repeat a no-write smoke test before treating the model as unavailable. Vendor catalogs change, so live probes override this list.

Use separate models for independent workstreams only when files do not overlap. The controller still owns exact build, test, benchmark, binary-path, and receipt verification.

### Codex dispatch pattern for hostile-audit remediation

When dispatching Codex for hostile-audit fixes, write a self-contained prompt file to `/tmp/` with:
- Exact file paths and line numbers to inspect
- The specific defect class and required behavior change
- RED/GREEN TDD instructions
- The exact verification command to run

Then dispatch with:
```bash
codex exec --dangerously-bypass-approvals-and-sandbox \
  -m gpt-5.6-sol -c model_reasoning_effort=high \
  -C /path/to/repo "$(cat /tmp/prompt.md)" 2>&1 | tee /tmp/result.log
```

**Never trust Codex self-reports.** After every Codex task returns, re-run the claimed verification commands in the controller session. Codex summaries are self-reports, not receipts. The controller owns `cargo test`, `cargo clippy`, `cargo fmt --check`, and `git diff --check`.

## Hybrid Codex / Claude-Fable routing for implementation plans

When Josh explicitly asks for Codex for most work but Claude Code Fable for the difficult parts, route by task difficulty instead of using one agent family everywhere:

- Codex `gpt-5.3-codex-spark`: mechanical docs/config/scripts/simple tests.
- Codex `gpt-5.5`: normal hard implementation, multi-file firmware/protocol/model-export work.
- Claude Code `--model fable`: genuinely difficult reasoning-heavy sections such as tensor-parallel correctness, custom SIMD math, transformer runtime, model-sharding/export feasibility, and 33M-memory analysis.

Do not treat a Codex `-m fable` rejection under ChatGPT auth as proof Fable is unavailable globally. If Claude Code auth works and `claude -p ... --model fable` smoke-tests OK, use Claude Code for Fable tasks and Codex for the rest.

Controller verification remains mandatory: after any Codex or Claude-Fable agent, re-run the actual build/tests/hardware receipts in the controller session. Agent summaries are not receipts.

## The Process

### 1. Read and Parse Plan

Read the plan file. Extract ALL tasks with their full text and context upfront. Create a todo list:

```python
# Read the plan
read_file("docs/plans/feature-plan.md")

# Create todo list with all tasks
todo([
    {"id": "task-1", "content": "Create User model with email field", "status": "pending"},
    {"id": "task-2", "content": "Add password hashing utility", "status": "pending"},
    {"id": "task-3", "content": "Create login endpoint", "status": "pending"},
])
```

**Key:** Read the plan ONCE. Extract everything. Don't make subagents read the plan file — provide the full task text directly in context.

### 2. Per-Task Workflow

For EACH task in the plan:

#### Step 1: Dispatch Implementer Subagent

Use `delegate_task` with complete context:


> 📄 See [references/code-0.python.md](references/code-0.python.md) for the complete code.


#### Step 2: Dispatch Spec Compliance Reviewer

After the implementer completes, verify against the original spec:

```python
delegate_task(
    goal="Review if implementation matches the spec from the plan",
    context="""
    ORIGINAL TASK SPEC:
    - Create src/models/user.py with User class
    - Fields: email (str), password_hash (str)
    - Use bcrypt for password hashing
    - Include __repr__

    CHECK:
    - [ ] All requirements from spec implemented?
    - [ ] File paths match spec?
    - [ ] Function signatures match spec?
    - [ ] Behavior matches expected?
    - [ ] Nothing extra added (no scope creep)?

    OUTPUT: PASS or list of specific spec gaps to fix.
    """,
    toolsets=['file']
)
```

**If spec issues found:** Fix gaps, then re-run spec review. Continue only when spec-compliant.

#### Step 3: Dispatch Code Quality Reviewer

After spec compliance passes:

```python
delegate_task(
    goal="Review code quality for Task 1 implementation",
    context="""
    FILES TO REVIEW:
    - src/models/user.py
    - tests/models/test_user.py

    CHECK:
    - [ ] Follows project conventions and style?
    - [ ] Proper error handling?
    - [ ] Clear variable/function names?
    - [ ] Adequate test coverage?
    - [ ] No obvious bugs or missed edge cases?
    - [ ] No security issues?

    OUTPUT FORMAT:
    - Critical Issues: [must fix before proceeding]
    - Important Issues: [should fix]
    - Minor Issues: [optional]
    - Verdict: APPROVED or REQUEST_CHANGES
    """,
    toolsets=['file']
)
```

**If quality issues found:** Fix issues, re-review. Continue only when approved.

#### Step 4: Mark Complete

```python
todo([{"id": "task-1", "content": "Create User model with email field", "status": "completed"}], merge=True)
```

### 3. Final Review

After ALL tasks are complete, dispatch a final integration reviewer:

```python
delegate_task(
    goal="Review the entire implementation for consistency and integration issues",
    context="""
    All tasks from the plan are complete. Review the full implementation:
    - Do all components work together?
    - Any inconsistencies between tasks?
    - All tests passing?
    - Ready for merge?
    """,
    toolsets=['terminal', 'file']
)
```

### 4. Verify and Commit

```bash
# Run full test suite
pytest tests/ -q

# Review all changes
git diff --stat

# Final commit if needed
git add -A && git commit -m "feat: complete [feature name] implementation"
```

## Task Granularity

**Each task = 2-5 minutes of focused work.**

**Too big:**
- "Implement user authentication system"

**Right size:**
- "Create User model with email and password fields"
- "Add password hashing function"
- "Create login endpoint"
- "Add JWT token generation"
- "Create registration endpoint"

**Red Flags — Never Do These**

- Start implementation without a plan
- Skip reviews (spec compliance OR code quality)
- Proceed with unfixed critical/important issues
- Dispatch multiple implementation subagents for tasks that touch the same files
- Make subagent read the plan file (provide full text in context instead)
- Skip scene-setting context (subagent needs to understand where the task fits)
- Ignore subagent questions (answer before letting them proceed)
- Accept "close enough" on spec compliance
- Skip review loops (reviewer found issues → implementer fixes → review again)
- Let implementer self-review replace actual review (both are needed)
- **Start code quality review before spec compliance is PASS** (wrong order)
- Move to next task while either review has open issues

## Multi-Batch Hostile-Audit Fix Pass — Ship, Verify, Iterate

When a hostile audit produces 30+ fix items and the user says "fix everything," don't try to ship all of them in one agent or one context. The pattern that worked on the Gloss 2026-06-10 pass (44 items, 5 batches, 6 commits, full green gauntlet):

### The shape

```
Hostile audit produces prioritized list of N fix items (CRITICAL/HIGH/MEDIUM/LOW)
       ↓
Group items into batches by:
  - Coupling: items that touch the same files go in the same batch
  - Risk: high-risk mechanical refactors (Arc<>, type changes, new deps) get their own batch
  - Verification cost: batches that need npm install / new build deps run alone
       ↓
Ship Batch A first (the 3-5 items that stop the bleeding — usually
  "the default is the dangerous thing" + "the unsafe band-aid" + "the
  silent timeout"). Run full verification gauntlet. Commit.
       ↓
Iterate: for each subsequent batch, decide:
  - Is this one big enough to warrant a codex exec task? (>= 3 items OR
    invasive refactor spanning 3+ files) → dispatch
  - Is it 1-2 small mechanical changes? → do it in the controller session
  - Is it 1+ items that need a human design call (e.g. "design a command
    palette")? → split into a focused codex spec with explicit UX
    constraints
       ↓
After every batch: cargo fmt + cargo check + cargo test + npm build +
  npm test + 5 AGENTS.md gates. Commit on green. If red, fix in
  controller (1-2 patches per error) before moving on.
       ↓
After the last batch: write a HOSTILE_AUDIT_FINDINGS_<date>.md with:
  - What shipped (file:line for every claim)
  - What was deferred (item + honest reason, not "ran out of time")
  - Risk assessment of deferred work (correctness? edge case? polish?)
  - Receipts (every verification command + output)
  - Hostile-auditor handoff paragraph
```

### Shared-worktree implementation and council isolation

When executing a broad plan in a shared canonical worktree, do not run multiple writers or read-only auditors against the same tree concurrently unless file ownership is disjoint and explicitly recorded. A source edit can invalidate generated evidence projections (for example, source digests in a completion ledger) while another lane is testing them, producing mixed-generation results.

Use this sequence:
1. Freeze and record `git status`, HEAD, and intended file ownership before dispatch.
2. Dispatch one cohesive implementation batch at a time for overlapping files.
3. After the batch returns, controller-owned tests and digest/provenance checks decide whether it is accepted.
4. Regenerate projection artifacts only after source-affecting edits stabilize; never hand-edit a stale digest to make a test green.
4. Commit only after the projection, source, and tests are from the same generation.
5. Treat a passing subagent report as provisional until the controller reruns the exact tests. If a later batch changes any file included in a digest/provenance projection, the prior projection is invalidated even when the code tests still pass; regenerate it and rerun its strict freshness test before marking the phase complete.
6. Keep phase acceptance atomic: source edits, generated projections, tests, and recorded receipts must describe one HEAD/generation. Do not advance the todo phase or start a dependent owner-boundary change while that set is mixed-generation.

If the user explicitly requests a council of a particular model family, use that model family only for a narrowly scoped decision. For authority, lifecycle, promotion, security, or evidence-boundary ambiguity, stop implementation at the decision gate and dispatch a small council (preferably 3 independent lanes plus a tie-breaker only if needed). Give each lane the same bounded evidence summary and require a forced choice, citations/decision criterion, RED/GREEN tests, and migration/rollback. Do not resolve a split by personal preference. If the requested model family cannot authenticate or returns only repository-noise before a bounded memo, report the failed council as non-evidence, narrow the prompt to source-bounded evidence, retry once, then preserve the blocker rather than silently substituting another model family.

### Process overhead vs direct implementation (2026-07-20)

When a user says "this is taking forever" or "I thought I had all the libraries necessary", the councils, agents, and verification cycles are the bottleneck, not the implementation. The pattern that frustrated Josh:

1. SOL council produces an 18-task plan → each task is a 5-10 minute Spark agent → agents conflict on shared files → controller fixes compile errors → re-verify → re-commit → next batch. Total wall time: hours for what should be 30 minutes of direct editing.

2. The fix Josh explicitly requested: "Stop the council/agent ceremony — directly implement tasks in one shot, verify, commit." This means: when the plan is clear and the code is mechanical composition of existing APIs, **do it directly in the controller session**. Use `patch` and `terminal` tools, not `delegate_task` or `codex exec`.

3. When to still use agents: genuinely difficult reasoning, hostile audits, or when the user explicitly requests a council. When the user says "use a council to plan it out and then implement with spark agents", that's the explicit request — follow it. But when the user says "finish everything up" or "this is taking forever", switch to direct implementation.

4. **Parallel Spark agents editing the same crate produce broken compile states.** Two agents dispatched simultaneously that both touch `aidens-runner/src/lib.rs` or `aidens-cli/src/lib.rs` will conflict. The controller must then `git checkout -- . && git clean -fd` to recover. Rule: only dispatch parallel agents when file scopes are provably disjoint. When both need to modify the same crate's `lib.rs` for module wiring, serialize them.

### Learning-agent closure: owner gaps and mixed-generation projections

For closed-loop learning, treat “strong primitives exist” as different from “the operator-runnable loop is complete.” Before adding promotion, replay, recovery, or terminal orchestration, inventory the canonical owner contracts for candidate identity, patch/source/verifier/environment/store identity, retained inputs, adjudication, permits, and lifecycle receipts. If a required owner-issued identity or receipt is absent, stop at the gate and escalate; never derive canonical truth from local path hashes, caller-supplied metadata, fixture/oracle scores, or a projection-only adapter.

Use Codex `gpt-5.6-luna` with low effort for clearly specified coding tasks when requested; use `gpt-5.6-sol` council lanes for authority-boundary decisions. Keep writers sequential when they touch overlapping files. After each writer, rerun the exact tests in the controller and refresh any completion ledger, digest, manifest, or claim projection that references changed source. A projection generated before a later source edit is stale even if its own test previously passed. Commit only one source/projection/test generation together, and report skipped live evidence as `SKIP`/`INDETERMINATE`, never as success.

When a public command is a deliberate fail-closed stub because canonical owner APIs are insufficient, preserve the stub and document the exact missing owner contract. Do not replace it with a metadata comparison or locally computed replay. If a subagent reports “complete” but the requested verification is missing, rerun it yourself; if it reports a blocker, verify the cited source and preserve the blocker unless an owner-backed fix is available.

### Generation and projection freshness

Generated ledgers, receipt indexes, manifests, digests, and claim projections are projections of a specific source generation. If any source-affecting file changes after a projection is produced, invalidate and regenerate the projection after the writer batch stabilizes; never hand-edit a stale digest or weaken its freshness test. Controller acceptance is atomic: source edits, projection artifacts, tests, and recorded receipts must all describe the same HEAD/generation. Do not run read-only council/audit sessions over a mutating shared worktree unless ownership is disjoint; a council can otherwise observe a mixed generation and produce invalid findings. Freeze writers before final council collection, then revalidate every decisive citation against live HEAD.

### When to dispatch codex exec vs do in controller

| Pattern | Dispatch codex | Do in controller |
|---|---|---|
| Scope | 3+ files, 1 cohesive refactor | 1-2 lines in 1 file |
| Risk | High (lock changes, type changes, new dep) | Low (rename, add test, tweak const) |
| Verifiability | Agent's `cargo check` is easy to verify | Trivial to verify by eye |
| Time | 5-15 min in agent < 5 min in controller | Either is fine; controller faster |
| Knowledge of types | Domain knowledge needed (already in spec) | Trivial fix, no context needed |

### The 5-priority batch split (worked example)

For a typical 44-item "fix everything" pass on a Tauri/React/Rust app:

1. **Batch A — stop the bleeding** (5-7 items)
   - Default-flips (e.g. dangerous default → safe default)
   - Remove periodic-reset / sleep band-aids
   - Wrap blocking calls in timeouts
   - The 1-2 things the user said "always do this" or "never do that"
   - This batch must ship + verify + commit before any other batch.

2. **Batch B — unblock the hot path** (5-7 items)
   - Pool routing, batched DB methods, Arc<service>
   - Shared reqwest::Client across providers
   - O(N²) → O(N) parser refactor
   - These touch shared files (AppState, providers/mod.rs, hybrid_search.rs); expect 5-10 cross-cutting errors, fix in controller.

3. **Batch C — UX polish** (5-10 items, but smaller scope)
   - Real command palette (cmdk or similar new dep)
   - Onboarding empty state
   - Splash screen
   - Light theme
   - SettingsDialog split
   - These are design-heavy; the spec must include design constraints
     ("match the existing design tokens at globals.css:X-Y").

4. **Batch D — reliability** (3-5 items)
   - Eager warmup
   - ErrorBoundary remount
   - Fatal-error dialog
   - These are typically 1-2 files each; can be done in controller
     without dispatching an agent.

5. **Batch E — bonus from late agents** (1-3 items)
   - If a codex task's output arrived after the session had already
     reported "done" and the diff actually compiles cleanly, take it.
     Don't refuse a working late result just because you reported done.
   - Update the handoff doc to record what was added.

### Cross-batch verification (run between every batch)

```bash
# Must all pass before committing the batch
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check    # 0 diff
cargo test --manifest-path src-tauri/Cargo.toml --features <X> --lib --no-fail-fast  # 0 fail
cargo test --manifest-path src-tauri/Cargo.toml --features <X> --lib commands::chat  # 0 fail
cargo test --manifest-path src-tauri/Cargo.toml --features <X> --lib providers::  # 0 fail
npm run build                                                       # 0 TS error
npm test                                                            # 0 fail
for gate in validation/validate_*.py; do python3 "$gate" .; done   # 5+ PASS lines
```

The total wall time is 60-90 seconds when tests are cached. **Do not skip**
the cross-batch check. A 5-minute codex task that breaks 2 test files is
faster to fix at the batch boundary than 5 hours later when the working
tree has 30 modified files.

### The "I can't finish it all, be honest" decision

When you estimate "I have time for N more batches and the user said
'finish everything'":

- **Right move:** Ship the prioritized batches you can verify, write
  the handoff doc with the deferred list, let the user choose what
  to do next.
- **Wrong move:** Pretend the deferred items don't exist, claim
  "everything is done," or burn the remaining context trying to
  ship unverified work.

The user can re-prioritize the deferred list with full information.
The user CANNOT make that decision from "everything is done" because
it's a lie.

The `codex` skill's "Time-Boxing Thrashing Codex Agents" section covers
the kill protocol for agents that are stuck in compile-error retry loops.
The "Late-Arriving Codex Agent Re-Modification" section covers agents
that complete after you've already reported done — they may re-modify
files, and a verification gauntlet is mandatory before treating them as
shipped.

The `codex` skill's "Honest Scope-Tracking" section covers the handoff
doc format. The pattern above is the multi-batch shipping discipline
that pairs with it.

### The late-arriving agent win

If a `codex exec` task was launched hours ago and shows up after you've
already reported "done":

- **Do** check whether the late diff actually compiles and passes tests
- **Do** commit it as a bonus batch if it's clean
- **Do** update the handoff doc to reflect the new state
- **Don't** refuse a working late result because you've already
  reported; the user values receipts over punctuality
- **Don't** accept a late result blindly — re-run the verification
  gauntlet on the new state; sometimes the late diff is 41 errors

The `codex` skill's "Cross-File Conflict Pattern" section covers
detecting and handling agent-on-agent collisions. The late-arrival case
is the same pattern with a different timing.

### Incremental Handoff Doc Drafting

Don't wait until the end of the session to start the
`HOSTILE_AUDIT_FINDINGS_<project>_<date>.md` doc. The shape of the doc
("What shipped" + "What was deferred" + "Receipts" + "Risk assessment")
is most useful when it's a **live document** that grows with each batch:

- After Batch A: open the doc with a "What this pass did so far"
  section listing the items shipped in Batch A. List the deferred items
  honestly.
- After Batch B: append the Batch B items to "What shipped." Move
  anything you now realize is deferred to the deferred list.
- After each batch: re-run the verification gauntlet, paste the actual
  output into the "Receipts" section.

This discipline produces three benefits:
1. **The doc is always a truthful snapshot of state**, even if the
   session ends abruptly. A controller iteration limit or a tool
   failure mid-batch doesn't lose the doc.
2. **You catch deferrals early.** Items you thought were easy often
   turn out to be hard; discovering that in batch 2 of 5 is much
   better than in batch 5 of 5 when there's no time to re-prioritize.
3. **The final handoff is a 2-minute "update receipts" pass**, not a
   20-minute "remember everything I did and write it up" pass that
   inevitably misses things.

When the final batch lands, the doc is already 90% written. The
controller's last action is: update receipts, add the late-arrival
section if any, commit. The user gets an accurate handoff without
waiting for a writeup.

## Cross-File Conflict Resolution After Parallel Mechanical Refactors

The `codex exec` / `claude -p` parallel-dispatch pattern produces predictable cross-file conflicts when two agents edit a shared test initializer, struct constructor, or downstream caller. Do NOT re-dispatch a "fix it" agent — fix in the controller:

**Why this happens:** Each agent's "Out of scope" list is honored at the file level, but the *effects* of their changes propagate (signature changes, type renames, lock-method swaps). Two agents who each change a private field type but test it via a shared initializer will both edit the initializer differently.

**Pattern that worked:**
1. After all parallel agents return, run `cargo check` / `tsc --noEmit` in the controller. Expect 3-10 errors.
2. Group errors by file. Most files have 1-2 errors that are mechanical (add an argument, swap `.lock()` for `.read()`, fix a borrow conflict).
3. Fix all errors in one focused `patch` session per file. Re-run `cargo check` between files.
4. Commit each workstream (or the whole batch) once green.

**Common conflict shapes (Rust):**
- `E0061` "takes N arguments but M supplied" — a constructor signature changed, downstream caller didn't update
- `E0599` "no method named `lock` found for `RwLock`" — a field was changed from `Mutex<T>` to `RwLock<T>` and an old `.lock()` call survived
- `E0502` borrow conflict — usually from a refactor that changed buffer/collection types
- `E0063` missing struct fields — a field was added to `AppState` and a test initializer wasn't updated

**Tooling note:** The `patch` tool's auto-lint (when editing Rust files standalone) emits false positives like `async fn not permitted in Rust 2015` because it runs `rustc` on a single file without the package context (Cargo workspace `edition = "2021"`). **Always use `cargo check --manifest-path <Cargo.toml> --features <feature>` as the source of truth** — the patch tool's lint output is not authoritative for Rust packages.

## Two-Model Stack: When the Subagent *Is* a Frontier Model

When the bottleneck isn't headcount but model capability — Rust+quantization math, long-context reasoning, deep algorithmic research — subagents that delegate to `codex` or `claude-code` are not "extra hands," they are *access to a different model family* than the one running the controller. Pattern:

1. **Scope the question tightly** in the controller. Frontier models don't need more autonomy; they need better prompts. Restrict to one specific decision: "Why does 11× hold at 4K and what breaks at 32K," not "improve poly-kv."
2. **Force a strict output format** in the subagent prompt: findings + cited code paths (file:line) + suggested experiments. Not a free-form essay.
3. **Always pass `--max-turns` and `--max-budget-usd`.** Print mode (`-p`) preferred for one-shots. Default budget discipline per the codex / claude-code skills.
4. **Audit the subagent's output against the doctrine before acting on it.** Subagent summaries are self-reports. Re-derive the key claim from the cited code paths yourself.
5. **Report what you spent** to the user. Cost transparency is part of the contract.

**When NOT to use a two-model stack:**
- Doctrine design or enforcement work (OpenClaw hooks, audit-crate shape, agent-loop gates) — the design encodes user values, not raw model capability.
- Investigative work where the *question itself* is the value.
- Anything where a subagent will default to the wrong answer because the wrong answer is the path-of-least-resistance for the model.

**Verification after every codex/claude-code subagent:** Re-run the most expensive claim in the controller session. Subagents optimize for plausible answers; the doctrine is the controller's job.

### Parallel Library Audits via codex worktrees

Mechanical multi-crate audits are the textbook case for codex parallel dispatch: each crate is independent, the work is well-scoped, and the audit is the safety net (not the agent loop). Pattern: `git worktree add -b audit/<crate> /tmp/audit-<crate> main` per crate, then `codex --yolo exec "Apply the hostile-audit finish pack to this crate, commit results."` per worktree in background. `--yolo` is appropriate because the audit IS the safety net.

## Commit and Publication Boundaries

Treat local checkpoints and remote publication as separate operations:

- `commit` means verified local commit(s) only.
- `push`, `publish`, `open a PR`, or `release` requires an explicit request for that remote side effect.
- If agents still edit a repository, do not snapshot half-written work. Commit completed independent repositories first; then wait for or deliberately stop remaining editors, run controller-owned gates, and commit their final state.
- Scope staging in dirty repositories. Exclude downloaded datasets, generated OpenAPI dumps, temporary environments, caches, and disposable benchmark inputs unless explicitly requested.
- Report local commit SHAs and separately state whether remote branches changed.

For temporal benchmarks, verify persistence granularity before accepting results. A millisecond sleep is not a valid as-of separator for second-granularity timestamps; add a regression and cross a verified persisted boundary. Exclude that harness wait from serving-latency claims.

## Handling Issues

### If Subagent Asks Questions

- Answer clearly and completely
- Provide additional context if needed
- Don't rush them into implementation

### If Reviewer Finds Issues

- Implementer subagent (or a new one) fixes them
- Reviewer reviews again
- Repeat until approved
- Don't skip the re-review

### If Subagent Fails a Task

- Dispatch a new fix subagent with specific instructions about what went wrong
- Don't try to fix manually in the controller session (context pollution)

## Efficiency Notes

**Why fresh subagent per task:**
- Prevents context pollution from accumulated state
- Each subagent gets clean, focused context
- No confusion from prior tasks' code or reasoning

**Why two-stage review:**
- Spec review catches under/over-building early
- Quality review ensures the implementation is well-built
- Catches issues before they compound across tasks

**Cost trade-off:**
- More subagent invocations (implementer + 2 reviewers per task)
- But catches issues early (cheaper than debugging compounded problems later)

### Parallel Crate Creation — When to Batch, When to Verify

**Pattern — creating multiple new crates:** dispatch in batches of 2-3, NOT all at once. Each crate creation is 40-60 tool calls, 5-7 source files, 5-10 minutes. With 7 crates in one `delegate_task`, 3 timed out and 2 produced incomplete scanners.

**Pattern — executing a pre-existing multi-phase plan:** When the plan is already decomposed and phases are in different code areas, dispatch ONE subagent per phase (not per task). Run 3-4 phases in parallel. The controller verifies the final state after all return. Two-stage review is overkill for plan execution — the plan IS the spec. Verification steps gate the next phase, not a separate reviewer subagent.

**Iteration limit truncation is expected:** Subagents hitting `max_iterations` before reporting test results is the normal failure mode. The controller must:
1. Run `cargo check` and `cargo test` for the affected crate(s) immediately on return
2. Fix any compile errors or test failures directly in the controller session
3. Do NOT re-dispatch to fix — the truncation is a budget limit, not a capability limit; fixing in controller is faster and cleaner

**Batch sizing:**
- Batch 1: 3 crates (budget ~150 tool calls total)
- Verify each individually: `cargo check -p <crate>` + `cargo test -p <crate>`
- Fix any issues before Batch 2
- Batch 2: next 3 crates
- Final crate: individual dispatch for highest-risk item

**Timeout awareness — match timeout to task complexity:**
- A "rewrite this module" task for a 1000-line file is 15+ minutes, not 10 minutes.
- Estimate: 5 min per file touched + 5 min per 100 lines of new/changed code + 5 min for test writing.
- If the estimate exceeds the default subagent timeout, either: (a) split the task smaller, (b) increase the timeout explicitly, or (c) do it in the controller session.
- Subagents that time out mid-implementation leave the workspace in a broken state. Always run `cargo check` after a timed-out subagent returns.

**Why not all-at-once:**
- Shared context budget is finite — 7 subagents × 50 tool calls = 350 calls, which exceeds the 50-call-per-script limit
- Subagent timeouts are per-subagent, but controller session timeout is global
- Fix effort compounds: fixing 1 crate's bug is fast, fixing 4 simultaneously is thrash

**Verification after parallel dispatch:**
```bash
# Always verify individually, never trust "I created it" self-report
cargo check -p boundary-compiler -p bitemporal-runtime -p quant-governor
# Run tests and grep for FAIL — subagents don't always catch their own test failures
cargo test -p <crate> 2>&1 | grep -E "FAILED|test result"
```

### Rust Workspace — Controller Verification Checklist

Subagents commonly return "complete" but leave the workspace in a broken state. Per-crate `cargo check` is **not sufficient** — always run the full workspace command after subagent batches return.

**After every subagent (or batch) returns, run ALL of:**
```bash
# 1. Compile check — must be zero errors across entire workspace
cargo check --workspace 2>&1 | grep "^error" | wc -l
# Expected: 0

# 2. Test run — must be zero failures across entire workspace
cargo test --workspace 2>&1 | grep "FAILED" | wc -l
# Expected: 0

# 3. Specific failure patterns to grep for:
cargo check --workspace 2>&1 | grep -E "E0433|cannot find type"   # missing re-exports
cargo test --workspace 2>&1 | grep "E0063"                        # missing struct fields in tests
```

**Common subagent failures that per-crate checks miss:**
1. **Missing re-export** — added type to `types.rs` but didn't add to `lib.rs` pub use list. Downstream crates fail with `error[E0433]: cannot find type X`. Per-crate `cargo check -p <crate>` passes because the crate itself exports the type; downstream consumer is what breaks. Fix: add to `pub use types::{...}` in lib.rs.
2. **Optional-dep stub uses the optional dep** — stub module's public struct references `CodecProfile` from `quant_governor` (optional dep). Per-crate passes (feature off, stub isn't checked deeply); workspace fails because other crates transitively pull in the stub. Fix: make stub self-contained (§4.12 of rust-workspace-stabilization).
3. **Deprecated struct fields in tests** — subagent added new struct fields but tests weren't updated. `cargo test -p <crate>` fails with E0063. Fix: add `#![allow(deprecated)]` to test file, populate all deprecated fields.
4. **Subagent hit iteration limit before running tests** — summary says "tests pass" but test binary was never compiled. Always re-run `cargo test -p <crate>` independently.

### Parallel New-Module Creation (Independent Files)

When adding N new modules that don't depend on each other (e.g., scoring.rs, residual.rs, compat.rs, wire.rs, sidecar.rs, eval.rs), dispatch N subagents in parallel. Each creates exactly one file with complete context. This is 3-4x faster than serial implementation.

**Pattern that worked (fib-quant maturation, 2026-06-25):**
1. Controller reads all relevant existing files and writes the plan/gap list
2. Dispatch 3-4 subagents in parallel, each with:
   - "Read these files first:" section with exact paths
   - "Your task:" section with specific requirements
   - "Only create X.rs and modify lib.rs. Do NOT touch other files."
3. After ALL return: `cargo fmt -p <crate> && cargo check -p <crate> --all-features`
4. Fix compile errors in the controller (unused imports, wrong types, missing re-exports)
5. Re-run tests: `cargo test -p <crate> --all-features`
6. If a subagent's file didn't compile, fix in controller — do NOT re-dispatch

**Key constraints:**
- Each subagent MUST receive complete context — don't make them discover the codebase
- The controller MUST run full `cargo check --all-features` after all subagents return
- Subagents hitting iteration limits is normal for large files (wire.rs, persistence.rs) — the controller finishes the last 10%
- When a subagent creates a file but doesn't add it to lib.rs, the controller patches lib.rs centrally
- If multiple subagents modified lib.rs, re-read it before each patch (stale reads cause failed patches)

**Pitfall:** When multiple subagents write to the same log file (e.g. `AGENT_LOG.md`), they use `write_file` which **overwrites**, not appends. Four subagents writing to the same file produces only the LAST subagent's content.

**Prevention:**
1. **Never let subagents write to shared log files.** The controller session writes the log. Subagents report via return value only.
2. If a subagent MUST log, give each a **dedicated temp file** (`/tmp/agent-<name>-<timestamp>.md`), then the controller merges them into the canonical log.
3. After parallel dispatch, **re-read the log file** before appending — it may have been overwritten by a sibling subagent that finished during your own work.

**Example:**
```python
# WRONG — subagents overwrite each other
for crate in crates:
    delegate_task(context="append to AGENT_LOG.md ...")

# RIGHT — controller owns the log
results = delegate_task(tasks=[...])  # subagents return data only
for r in results:
    append_to_agent_log(r.summary)  # controller writes once
```

### Cross-Language Integration Pitfalls

These patterns emerged from multi-repo hostile-audit remediation (Python hooks + Rust MCP server, 2026-07-13). They apply any time you're fixing issues across language boundaries.

**Pitfall — test mock patching after import restructuring:** When you move a function import from one module to another (e.g. `http_post` from `memory_recall` to `common`), any test that mocks `memory_recall.http_post` breaks with `AttributeError: does not have the attribute`. The fix is to mock the NEW module's attribute (`common.http_post`) instead. After any import restructuring, grep tests for `mock.patch.object(old_module, "moved_name")` and update to the new module.

**Pitfall — `frame_hits()` requires provenance fields:** The `frame_hits()` function from `injection_framing.py` requires `state` and `retrieval_receipt_ref` fields on every hit. Raw search results from unwitnessed endpoints don't have these. You must call `propagate_retrieval_context(response)` first to hydrate response-level state/receipt onto individual hits before calling `frame_hits()`. Otherwise `frame_hits()` returns an empty string (all hits rejected by `admit_provenanced_hits`).

**Pitfall — `SearchResult` namespace is inside the enum variant:** In the semantic-memory Rust crate, `SearchResult` doesn't have a direct `namespace` field. It's inside the `SearchSource` enum (`SearchSource::Fact { namespace, .. }`). You must pattern-match to extract it: `let namespace = match &r.source { SearchSource::Fact { namespace, .. } => namespace.clone(), _ => String::new() };`. Don't assume struct fields exist on the outer type — check the enum variants.

**Pitfall — Codex audit findings need controller verification:** Codex dispatched as a hostile auditor can produce false positives. In the 2026-07-13 pass, Codex claimed `doctor.py` used `~/.hermes/semantic-memory.db` while the actual code used `~/.local/share/semantic-memory` — the finding was wrong. Always verify Codex audit findings against live source before acting on them. The `codex-hostile-audit-pattern.md` reference covers the full dispatch pattern.

## Integration with Other Skills

### With writing-plans

This skill EXECUTES plans created by the writing-plans skill:
1. User requirements → writing-plans → implementation plan
2. Implementation plan → subagent-driven-development → working code

### With test-driven-development

Implementer subagents should follow TDD:
1. Write failing test first
2. Implement minimal code
3. Verify test passes
4. Commit

Include TDD instructions in every implementer context.

### With requesting-code-review

The two-stage review process IS the code review. For final integration review, use the requesting-code-review skill's review dimensions.

### With systematic-debugging

If a subagent encounters bugs during implementation:
1. Follow systematic-debugging process
2. Find root cause before fixing
3. Write regression test
4. Resume implementation

## Example Workflow

```
[Read plan: docs/plans/auth-feature.md]
[Create todo list with 5 tasks]

--- Task 1: Create User model ---
[Dispatch implementer subagent]
  Implementer: "Should email be unique?"
  You: "Yes, email must be unique"
  Implementer: Implemented, 3/3 tests passing, committed.

[Dispatch spec reviewer]
  Spec reviewer: ✅ PASS — all requirements met

[Dispatch quality reviewer]
  Quality reviewer: ✅ APPROVED — clean code, good tests

[Mark Task 1 complete]

--- Task 2: Password hashing ---
[Dispatch implementer subagent]
  Implementer: No questions, implemented, 5/5 tests passing.

[Dispatch spec reviewer]
  Spec reviewer: ❌ Missing: password strength validation (spec says "min 8 chars")

[Implementer fixes]
  Implementer: Added validation, 7/7 tests passing.

[Dispatch spec reviewer again]
  Spec reviewer: ✅ PASS

[Dispatch quality reviewer]
  Quality reviewer: Important: Magic number 8, extract to constant
  Implementer: Extracted MIN_PASSWORD_LENGTH constant
  Quality reviewer: ✅ APPROVED

[Mark Task 2 complete]

... (continue for all tasks)

[After all tasks: dispatch final integration reviewer]
[Run full test suite: all passing]
[Done!]
```

## Remember

```
Fresh subagent per task
Two-stage review every time
Spec compliance FIRST
Code quality SECOND
Never skip reviews
Catch issues early
```

**Quality is not an accident. It's the result of systematic process.**

## Further reading (load when relevant)

- **`references/claude-code-model-routing-recursiveintell-embedded.md`** — Josh-specific mixed Claude Code model-routing discipline for ESP32-S3 / quantization / tensor-parallel cluster work: Sonnet for routine tasks, Opus for architecture/review, Fable for hard SIMD/transformer/TinyStories sections if available, with controller-owned receipts.
- **`references/hybrid-codex-fable-routing.md`** — Codex-first implementation pattern with Claude Code/Fable reserved for genuinely difficult reasoning sections. Load when the user asks to use Codex for most work but Fable for hard parts; includes live model smoke-test discipline, routing-doc requirements, and controller-owned verification rules.

When the orchestration involves significant context usage, long review loops, or complex validation checkpoints, load these references for the specific discipline:

- **`references/context-budget-discipline.md`** — Four-tier context degradation model (PEAK / GOOD / DEGRADING / POOR), read-depth rules that scale with context window size, and early warning signs of silent degradation. Load when a run will clearly consume significant context (multi-phase plans, many subagents, large artifacts).
- **`references/long-plan-iteration-budget-and-resumable-checkpoints.md`** — Controller-call budgeting, durable continuation ledgers, precise phase-state vocabulary, pending-review discipline, closure-capacity reserves, and forced-stop handoffs. Load before a multi-phase or multi-repository plan likely to approach the session/tool iteration limit.
- **`references/gates-taxonomy.md`** — The four canonical gate types (Pre-flight, Revision, Escalation, Abort) with behavior, recovery, and examples. Load when designing or reviewing any workflow that has validation checkpoints — use the vocabulary explicitly so each gate has defined entry, failure behavior, and resumption rules.
- **`references/handoff-doc-template.md`** — The `HOSTILE_AUDIT_FINDINGS_<project>_<date>.md` template for honest scope-tracking when a multi-batch workstream lands partially. Load when 3+ batches were dispatched and not everything shipped. Shape: "What this pass did" + "What was NOT done (and why)" + "Receipts" + "Risk assessment" + "Hostile-auditor handoff."
- **`references/mixed-frontier-agent-research-pass-2026-06-26.md`** — Worked pattern for executing a broad Rust research-implementation plan with Claude Code and Codex in parallel while the controller owns scope, cargo verification, security scan, semantic-memory capture, and final receipts. Load when the user asks to "do it" from a research plan and explicitly wants Claude/Codex used where possible.
- **`references/live-system-remediation-controller-owned-runtime.md`** — Pattern for broad live-system remediation where subagents can edit/test source, but the controller must own process/config mutations and distinguish on-disk config proof from fresh-process proof from current-process proof. Load when fixing Hermes/MCP/gateway/hooks in a running system without restarts.
- **`references/live-trust-kernel-upgrade-activation.md`** — Required when a core memory/storage governance invariant changes across the library, MCP/HTTP adapters, hooks, installed binaries, and running services. Covers adapter contract audits, hard-delete→governed-forgetting migration, cross-adapter principals, installed hash verification, stale MCP handles, and live write/retrieve/forget certification.
- **`references/controller-owned-benchmark-verification.md`** — Required when agents implement or run benchmarks. Covers pinning the exact freshly built executable, resolving agent/controller receipt disagreement, deterministic bitemporal fixtures, and continuing deferred work after a “finish everything” directive.
- **`references/codex-hostile-audit-pattern.md`** — Dispatch Codex CLI as a read-only hostile reviewer against a specific codebase, collect findings, then fix in the controller. Distinct from the multi-batch fix pass — the auditor only reads and reports. Use when the user says "hostile audit X" or you need a fresh pair of eyes from a different model family.
- **`references/codex-model-effort-and-commit-proof.md`** — Deliberate Codex model/effort routing, supported effort configuration, writer isolation, controller-owned verification, and pre/post-HEAD commit proof. Load before substantial Codex-backed remediation or final closure.
- **`references/parallel-worktree-receipt-integration.md`** — Controller-owned integration for asynchronous Rust worktrees: inventory canonical owner symbols before implementation, reconcile live branch topology, quarantine delayed reports from stale source generations, keep a delegation-ID intake ledger for out-of-order results, interpret `git cherry`/ancestry precisely, mine only current-contract hunks from superseded commits, prove rewritten cherry-picks with stable patch IDs, maintain a dirty-tree ownership ledger, verify source lanes before dependency-ordered integration, avoid starting conflict-prone Git transactions without resolution capacity, account for nested-workspace lockfiles, test receipt recovery as an exact semantic matrix, run downstream writer smoke tests after persistence hardening, never launder unstable IDs, revalidate every corpus consumer, prove staged-workspace rollback by digest, budget disk for parallel Rust builds, distinguish deterministic container tests from live runtime proof, avoid the background-writer pre-kill transition race, and reserve closure capacity for hostile-review collection and final gates. Load for multi-lane plans with dedicated worktrees, delayed asynchronous results, dirty integration trees, overlapping source commits, or context-compressed handoffs.
- **`references/live-execution-evidence-and-owner-reinventory.md`** — Reinventory canonical owner APIs against the current integration HEAD, reject stale “API missing” reports, distinguish receipt/configuration shape from current-execution proof, keep fake backends fail-closed, and close the full receipt/verification/CEA/export/lifecycle/replay chain before claiming an effectful learning lane is complete.

Both references adapted from gsd-build/get-shit-done (MIT © 2025 Lex Christopherson), except the mixed-frontier reference which is from the 2026-06-26 Libraries research pass.


## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Skill not triggering | Trigger phrases not matched | Check that your request contains keywords from the skill's description |
| Unexpected output | Model or agent differences | Review the skill's guidance and adjust for your specific context |
| Tool not found | Missing dependency or path | Verify that required tools are installed and accessible |

## Limitations

- This skill provides guidance but does not replace judgment for edge cases.
- Results may vary depending on the specific agent, model, and environment used.
- Review outputs critically; the skill is a starting point, not a substitute for verification.


## Prerequisites

- The target repository or artifact is identified and readable.
- Required project-specific tools and dependencies are available; verify them before editing.

## Purpose

Use this skill to apply the workflow below with explicit scope, evidence, and verification gates.


## Examples

- Start with a narrow, observable change, then run the documented gate before expanding scope.
- If a prerequisite or verification command fails, preserve the failure evidence and stop at the defined boundary.


## Reliability

For external or MCP-backed steps, verify the connection and required capability before use. On transient failure, retry only within the documented limit; record the error and stop rather than substituting an unverified result.
