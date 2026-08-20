# Parallel Worktree Integration for Receipt-Grounded Rust Plans

Use this pattern when a decomposed Rust plan is implemented by several asynchronous agents in dedicated worktrees and the controller owns the integration branch.

## 1. Isolate writers

- Create one branch/worktree per non-overlapping lane.
- Give every agent exact files, required RED/GREEN commands, commit scope, and a no-push boundary.
- Do not let two agents edit the same integration worktree.

## 2. Reconcile live branch state, not stale summaries

After an asynchronous batch completes—or before continuing after context compression—inspect every worktree directly:

```bash
git status --short --branch
git log -5 --oneline --decorate
git diff --stat
git diff --check
```

A branch may contain additional valid commits beyond the last subagent summary. Conversely, a reported commit may be incomplete. Treat summaries as leads; branch state and tests are evidence.

## 3. Verify source lanes before integration

Run each lane's focused command in its own worktree before cherry-picking. For feature-gated Rust security paths, run both default and feature-complete suites; a default test count can omit the actual hostile/container tests.

Examples:

```bash
cargo test --manifest-path AiDENs/Cargo.toml -p aidens-contracts
cargo test --manifest-path AiDENs/Cargo.toml -p aidens-receipts
cargo test --manifest-path AiDENs/Cargo.toml -p aidens-runner learning
cargo test --manifest-path Primitives/Cargo.toml -p check-runner
cargo test --manifest-path Primitives/Cargo.toml -p check-runner --all-features
```

Record executed, passed, failed, ignored, and filtered counts. A matching filter that executes zero tests is not a gate.

## 4. Integrate commits in dependency order

Prefer path-scoped cherry-picks of task commits over merging whole branches that also contain copied prerequisite commits. Typical order:

1. canonical contracts and material-bound identity;
2. durable receipt publication/recovery;
3. immutable fixtures/corpus;
4. runner composition;
5. effect backend and adapters;
6. operator CLI;
7. end-to-end wiring and review fixes.

After each batch, inspect the resulting log and status. Resolve cross-cutting compile errors in the controller.

## 5. Account for nested workspace lockfiles

A dependency change can require lockfile updates in more than one workspace. Run the affected nested Cargo command, inspect the generated lock diff, and commit only attributable dependency entries. Do not assume updating the root or Primitives lock also updates an AiDENs lock.

## 6. Re-run integrated gates

Source-worktree green is not integration proof. On the controller branch rerun:

- focused affected packages;
- nested workspace `cargo check --workspace --all-targets`;
- feature-complete hostile suites;
- deterministic schema/corpus generation twice and compare trees/digests;
- project guards and negative fixtures;
- `git diff --check` and final clean status.

If `cargo fmt --all -- --check` traverses path dependencies and reports unrelated baseline drift, record the exact file as a global blocker and run scoped formatting checks for changed crates. Do not reformat unrelated user code merely to turn the broad receipt green.

## 7. Verify operator process semantics, not just JSON projections

A fail-closed JSON body is insufficient if the executable exits successfully. Smoke-test the built CLI and capture both output and exit status for every terminal class:

```bash
set +e
cargo run --manifest-path AiDENs/Cargo.toml -p aidens-cli -- learn run --mode fixture >report.json
rc=$?
set -e
```

Require nonzero status for mock, fixture, degraded, missing-evidence, replay-unavailable, and blocked lifecycle operations when the command semantics request success. Parse the report separately to prove the state is closed. Also test lifecycle commands with syntactically valid but semantically unauthoritative permit files: an arbitrary readable JSON file must not produce a `promoted`, `revoked`, or `stopped` claim. If the canonical lifecycle owner is not wired, emit an explicit blocked projection and return nonzero.

## 8. Reconcile path dependency versions before broad workspace gates

Cargo requires a path package's declared version to satisfy the consumer's version requirement even when the dependency is optional. If a broad workspace check fails before compilation with a path-version mismatch:

1. inspect both manifests and `git blame` to determine which side drifted;
2. align the consumer requirement to the live local package version unless an intentional package release bump is separately specified;
3. rerun the broad command so Cargo updates the correct root lockfile;
4. inspect the lock diff for stale package versions and newly required dependencies;
5. commit manifest and attributable lock changes together.

Do not label a version-selection error as an unavailable registry dependency when Cargo reports a local `path` location.

## 9. Close asynchronous work honestly

- A background command is not a passing receipt until its exit status/output is collected.
- Do not issue the final completion report while implementation agents still own unintegrated worktrees.
- If a controller/tool limit interrupts closure, report integrated SHAs, verified command counts, uncollected processes, unintegrated branches, and remaining gates separately.
- Treat broad Clippy failures outside the changed scope as exact named blockers. Do not accumulate partial drive-by lint edits merely to move the first failure; either own and finish that cleanup as an explicit lane or revert it and preserve the blocker receipt.
- Never convert partial POC evidence into MVP, production-safety, or containment claims.

## 10. Prove source-lane equivalence after rewritten integration commits

Cherry-picks and conflict resolution can rewrite commit IDs. Do not infer equivalence from similar subjects. Prove representation mechanically:

```bash
git show <source-sha> --pretty=format: --binary | git patch-id --stable
git show <integration-sha> --pretty=format: --binary | git patch-id --stable
git cherry <integration-branch> <source-branch>
git show --name-status --format= <source-sha>
git show --name-status --format= <integration-sha>
```

Matching stable patch IDs plus a `-` result from `git cherry` is strong evidence that the source patch is represented. Record follow-up commits separately; patch equivalence proves the imported patch, not later behavior.

## 11. Preserve unrelated dirty work with an ownership ledger

Before editing a dirty integration tree:

1. Capture `git status --short`, `git diff --stat`, `git diff --check`, and path-scoped diffs.
2. Classify every path as `task-owned`, `pre-existing/user-owned`, `generated`, or `unknown`.
3. Never stage, restore, reformat, or include user-owned/unknown paths in a task commit.
4. If the controller accidentally edits a dirty user-owned file, revert only the controller's exact hunk. Never use whole-file checkout/reset merely to recover.
5. Test the preserved working tree when those changes can affect behavior, but state that the result includes uncommitted material.
6. For committed-source certification, create a separate clean worktree at the exact integration HEAD and run release gates there.

A clean auxiliary worktree proves the committed source generation; it does not make the original working tree clean. Report both states explicitly.

## 12. Test receipt recovery as a semantic matrix

Do not collapse recovery into one generic failure state. Inject and assert exact state/reason pairs for:

| Injected state | Expected handling |
|---|---|
| bundle write failure | no index publication; explicit persistence failure |
| index append failure after bundle rename | retain bundle; recover as pending index |
| duplicate index record | explicit duplicate classification |
| index references missing bundle | indeterminate / missing bundle |
| unreadable bundle | indeterminate / read failure |
| malformed JSON bundle | quarantined / malformed bundle |
| unsupported schema | quarantined / unsupported schema |
| content-digest mismatch | indeterminate / digest mismatch |
| valid orphan bundle | pending index, never published success |
| effect without closed outcome | quarantined |
| corrupt trailing index record | quarantine tail without poisoning valid history |

For child receipts, compare required and observed sets exactly as `(owner_id, digest)` pairs. Reject missing, duplicate, unexpected, wrong-owner, wrong-digest, and `closed=false` children. Count-only or digest-only comparisons are insufficient.

Use RED/GREEN evidence for the semantic distinctions: first assert the exact state/reason and observe the collapsed classification fail, then implement the minimal classifier change and rerun the full recovery suite.

## 13. Separate deterministic backend tests from live capability proof

Keep three evidence levels distinct:

1. **Deterministic tests:** injected runtime, hostile argv, receipt verification.
2. **Host preflight:** executable present, rootless mode and host mechanisms observed.
3. **Live sealed execution:** the digest-pinned image actually runs under required network, capability, mount, environment, resource, timeout, process-tree, and cleanup controls; denial fixtures fail as expected.

Only level 3 supports a claim that the specified live sandbox profile executed. If unavailable, record `UNAVAILABLE` and preserve fail-closed behavior; never upgrade level 1 or 2 evidence into live containment proof.

## 14. Reserve closure capacity and dispatch final review early

Long integrations often fail at closure even when implementation is sound:

- Dispatch read-only hostile reviewers immediately after the final source commit, before broad gates.
- Keep a durable gate ledger with command, cwd, revision, exit, passed/failed/ignored/filtered counts, and output digest.
- Run focused gates while reviewers work, then remediate findings before the clean-worktree release run.
- Reserve explicit tool calls for reviewer collection, remediation, repeat gates, closure docs, skill maintenance, final status, and any local commit.
- If the session limit arrives first, do not label pending reviews or uncollected commands as passed. Report the exact last verified revision and outstanding gates.

An uncollected hostile review or background command cannot authorize a release claim.

## 15. Do not start a conflict-prone Git operation without resolution capacity

A cherry-pick, rebase, or merge is not a single tool call; it is a transaction that may require source inspection, conflict resolution, focused tests, continuation, and status verification. Before starting one:

1. Confirm the current dirty paths have an ownership classification and are either preserved or committed as a scoped checkpoint.
2. Inspect the candidate commit with `git show --name-status` and compare it to live integrated changes with `git cherry`, stable patch IDs, and path-scoped diffs.
3. If the candidate overlaps files already changed by later integration fixes, prefer manually porting only missing behavior under RED/GREEN tests instead of blindly cherry-picking the whole commit.
4. Reserve enough controller/tool iterations to resolve conflicts, run focused gates, and prove `git status` has no active operation.
5. If capacity becomes insufficient after a conflict, abort the operation before ending when aborting will not discard owned work. Otherwise report the exact conflicted paths and active operation prominently; never describe that tree as usable or ready.

A verified source-branch commit is evidence of useful work, not proof that whole-commit replay is the safest integration mechanism.

## 16. Canonical-owner reuse gate before implementation

For plans spanning receipt, memory, verification, causal, patch, sandbox, or lifecycle crates, create a symbol-level ownership map before writing adapters:

- search live source for concrete public types/functions and all existing facade reexports;
- identify the canonical owner of storage, identity, policy, execution, verification, replay, and lifecycle separately;
- inspect neighbouring adapters and workspace manifests to find the intended dependency seam;
- mark each planned component as `reuse directly`, `thin adapter`, `owner-native extension`, or `genuine gap`;
- prohibit a new store, runtime, receipt family, verifier, causal graph, replay database, or lifecycle state machine unless the inventory proves no owner surface exists;
- re-run the inventory after integrating asynchronous branches, because a supposedly missing seam may already have landed elsewhere.

The controller should pass exact owner symbols to implementation agents. “Use the existing library” without symbol references invites duplicate abstractions.

## 17. Run downstream writer smoke tests immediately after persistence hardening

A receipt-store package can be fully green while a real producer is broken. Boundary hardening often exposes legacy writers that still emit process-local IDs, incomplete owner references, or projection-only bundles.

After changing durable identity or publication validation:

1. Run the receipt package tests.
2. Find every production caller of the write API.
3. Run at least one end-to-end writer → store → reopen → inspect test for each caller class.
4. Treat a rejected legacy writer as a writer-migration RED, not permission to weaken the store.
5. Preserve the failing downstream test until the producer emits valid material identity at the original construction sites.

Package-local green proves the validator. Consumer smoke tests prove the ecosystem still has a valid writer.

## 18. Never launder unstable IDs at the persistence boundary

Do not make `local-process-seq`, random, wall-clock-only, or branch-order IDs appear durable by hashing or wrapping them after construction. A digest of unstable material is still unstable material and breaks replay identity even if the resulting string looks content-addressed.

For a durable-writer migration:

- trace every persisted reference to its original owner receipt constructor;
- derive identity from canonical semantic material there;
- bind repeated but distinct calls with material attempt/step/invocation identity, not incidental process order;
- update cross-references and persisted owner receipts together;
- keep generic durable identity checks separate from stricter domain-specific lineage checks when a shared bundle serves multiple product profiles;
- require domain-specific constructors to enforce their complete owner-role set;
- keep legacy **read** compatibility explicit, but do not silently retain legacy writes when the migration contract says new writes are material-bound.

If a broad migration cannot be completed in the current batch, leave the writer fail-closed and report the exact downstream RED. Do not invent owner backpointers or substitute content digests for receipt IDs unless the contract explicitly defines that field as a digest reference.

## 19. Revalidate every implementation of a shared corpus invariant

Immutable corpus rules are often implemented independently in Python tooling, a Rust runner, and an operator CLI. Changing the manifest and only one validator creates cross-lane regressions.

For family-aware splits, distinguish these invariants precisely:

- multiple cases from one family in the **same** split are valid;
- one family appearing in different splits is leakage and invalid;
- task IDs and fixture paths remain globally unique;
- split denominators must satisfy the declared ratio exactly;
- holdout oracle content stays outside runner-safe task projections.

After any corpus or manifest update, run every validator and at least one real consumer. A deterministic validator run twice is necessary but not sufficient; the runner must also consume the final manifest successfully.

## 20. Prove rollback on the staged workspace, not by declaration

For an effectful sandbox evaluation, a rollback string or declared rollback step is not evidence. Reuse the canonical staging owner to snapshot the prepared ephemeral workspace before mutation, then after checks and independent verification:

1. restore the staged workspace from that snapshot;
2. recompute the complete tree digest;
3. require equality with the pre-patch tree digest;
4. persist both the rollback digest and boolean result;
5. prevent lifecycle registration/promotion when rollback is missing or mismatched.

Keep this distinct from host-repository rollback: the host source must never be mutated by the bounded lane. A rollback failure must remain an explicit failed/quarantined terminal condition rather than being ignored after otherwise-green checks.

## 21. Budget disk before parallel Rust worktree verification

Parallel Rust worktrees can independently create very large `target/` trees and exhaust the filesystem mid-compile. Before launching several writer lanes:

- measure free space and existing workspace target sizes;
- remove only disposable Cargo build artifacts when space is tight (`cargo clean` in the exact affected workspace);
- preserve source, fixtures, receipts, and evidence directories;
- recheck free space while several agents are compiling;
- avoid citing a compile interrupted by disk exhaustion as source evidence; rerun after cleanup.

This is execution hygiene, not a claim that any particular host always lacks space.

## 22. Quarantine delayed reports from stale source generations

Asynchronous audit and implementation reports can arrive after the integration branch has advanced through several commits. Treat each report as evidence about the exact revision it inspected—not as a current verdict.

On every delayed result:

1. Extract its reported branch, HEAD, and dirty-state declaration.
2. Compare that source generation with the current integration HEAD.
3. If stale, retain findings only as hypotheses and re-check each claimed missing symbol or defect in live source.
4. Do not replay stale patches, revert newer verified behavior, downgrade current evidence, or launch duplicate work because an older audit says a component was absent.
5. Separate stale read-only inventories from the final hostile review in the gate ledger.

When several queued audits repeat the same gap against obsolete revisions, acknowledge them once and continue the already-active current-generation lane. Source generation outranks arrival order.

### Delayed-result intake ledger

Queued batches may arrive out of dispatch order, especially after context compression. Keep one compact controller ledger keyed by delegation ID with: dispatched task class, source branch/HEAD, read-only vs writer, reported commit, patch-equivalence status, and disposition (`represented`, `missing`, `superseded`, `hypothesis-only`). Do not let each stale delivery restart the plan or trigger a new user-facing progress summary.

For reported commits, interpret Git evidence precisely:

- `git cherry <integration> <source>` returning `- <sha>` means an equivalent patch is represented;
- `+ <sha>` means it is not patch-equivalent, not that the whole old commit should be replayed;
- no line can mean the source commit is already an ancestor, so also inspect ancestry/log state;
- a non-identical old patch may contain one still-useful regression test while its production logic, corpus digest, schema, or error enum is superseded.

When only a narrow test or guard is missing, port that hunk under the **current** contract and run it against current source. Do not cherry-pick an obsolete corpus generation, restore an old canonical digest, or reintroduce a retired error variant merely to preserve patch identity. Record the old commit as `partially mined / superseded`, not `integrated`.

## 23. Avoid the background-agent pre-kill transition race

A quiet log and a momentary absence of compiler children do not prove a writer is idle. The agent may be reasoning between tool calls or about to launch its final gate, and notifications can contain output newer than the last process sample.

Before killing a substantive background writer that has already produced useful code:

1. preserve and classify the worktree diff;
2. compare log line-count growth across a short bounded grace interval;
3. inspect the full descendant process tree, not only direct children;
4. if its latest message names a specific next test, allow one grace interval for that command to appear;
5. if the time box still expires, kill it, then immediately run that named test in the controller.

Treat termination in this state as partial work requiring controller closure, not proof the agent failed. Reconcile the kill notification—which may include late test output—with the final worktree before deciding what remains.
