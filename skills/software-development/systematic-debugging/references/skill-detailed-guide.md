# Detailed guide

## Overview

Random fixes waste time and create new bugs. Quick patches mask underlying issues.

**Core principle:** ALWAYS find root cause before attempting fixes. Symptom fixes are failure.

**Violating the letter of this process is violating the spirit of debugging.**

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose fixes.

## When to Use

Use for ANY technical issue:
- Test failures
- Bugs in production
- Unexpected behavior
- Performance problems
- Build failures
- Integration issues

**Use this ESPECIALLY when:**
- Under time pressure (emergencies make guessing tempting)
- "Just one quick fix" seems obvious
- You've already tried multiple fixes
- Previous fix didn't work
- You don't fully understand the issue

**Don't skip when:**
- Issue seems simple (simple bugs have root causes too)
- You're in a hurry (rushing guarantees rework)
- Someone wants it fixed NOW (systematic is faster than thrashing)

## The Four Phases

You MUST complete each phase before proceeding to the next.

---

## Phase 1: Root Cause Investigation

**BEFORE attempting ANY fix:**

### 1. Read Error Messages Carefully

- Don't skip past errors or warnings
- They often contain the exact solution
- Read stack traces completely
- Note line numbers, file paths, error codes

**Action:** Use `read_file` on the relevant source files. Use `search_files` to find the error string in the codebase.

### 2. Reproduce Consistently

- Can you trigger it reliably?
- What are the exact steps?
- Does it happen every time?
- If not reproducible → gather more data, don't guess

**Action:** Use the `terminal` tool to run the failing test or trigger the bug:

```bash
# Run specific failing test
pytest tests/test_module.py::test_name -v

# Run with verbose output
pytest tests/test_module.py -v --tb=long
```

### 3. Check Recent Changes

- What changed that could cause this?
- Git diff, recent commits
- New dependencies, config changes

**Action:**

```bash
# Recent commits
git log --oneline -10

# Uncommitted changes
git diff

# Changes in specific file
git log -p --follow src/problematic_file.py | head -100
```

### 4. Gather Evidence in Multi-Component Systems

**WHEN system has multiple components (API → service → database, CI → build → deploy):**

**BEFORE proposing fixes, add diagnostic instrumentation:**

For EACH component boundary:
- Log what data enters the component
- Log what data exits the component
- Verify environment/config propagation
- Check state at each layer

Run once to gather evidence showing WHERE it breaks.
THEN analyze evidence to identify the failing component.
THEN investigate that specific component.

### 5. Trace Data Flow

**WHEN error is deep in the call stack:**

- Where does the bad value originate?
- What called this function with the bad value?
- Keep tracing upstream until you find the source
- Fix at the source, not at the symptom

**Action:** Use `search_files` to trace references:

```python
# Find where the function is called
search_files("function_name(", path="src/", file_glob="*.py")

# Find where the variable is set
search_files("variable_name\\s*=", path="src/", file_glob="*.py")
```

### Phase 1 Completion Checklist

- [ ] Error messages fully read and understood
- [ ] Issue reproduced consistently
- [ ] Recent changes identified and reviewed
- [ ] Evidence gathered (logs, state, data flow)
- [ ] Problem isolated to specific component/code
- [ ] Root cause hypothesis formed

**STOP:** Do not proceed to Phase 2 until you understand WHY it's happening.

---

## Phase 2: Pattern Analysis

**Find the pattern before fixing:**

### 1. Find Working Examples

- Locate similar working code in the same codebase
- What works that's similar to what's broken?

**Action:** Use `search_files` to find comparable patterns:

```python
search_files("similar_pattern", path="src/", file_glob="*.py")
```

### 2. Compare Against References

- If implementing a pattern, read the reference implementation COMPLETELY
- Don't skim — read every line
- Understand the pattern fully before applying

### 3. Identify Differences

- What's different between working and broken?
- List every difference, however small
- Don't assume "that can't matter"

### 4. Understand Dependencies

- What other components does this need?
- What settings, config, environment?
- What assumptions does it make?

---

## Phase 3: Hypothesis and Testing

**Scientific method:**

### 1. Form a Single Hypothesis

- State clearly: "I think X is the root cause because Y"
- Write it down
- Be specific, not vague

### 2. Test Minimally

- Make the SMALLEST possible change to test the hypothesis
- One variable at a time
- Don't fix multiple things at once

### 3. Verify Before Continuing

- Did it work? → Phase 4
- Didn't work? → Form NEW hypothesis
- DON'T add more fixes on top
- DON'T rerun the exact same failing command without changing either the code, the environment, or the diagnostic question. A repeated identical failure is evidence; switch to log slicing, raw artifact retrieval, or a narrower command.

### 4. When You Don't Know

- Say "I don't understand X"
- Don't pretend to know
- Ask the user for help
- Research more

---

## Phase 4: Implementation

**Fix the root cause, not the symptom:**

### 1. Create Failing Test Case

- Simplest possible reproduction
- Automated test if possible
- MUST have before fixing
- Use the `test-driven-development` skill

### 2. Implement Single Fix

- Address the root cause identified
- ONE change at a time
- No "while I'm here" improvements
- No bundled refactoring

### 3. Verify Fix

```bash
# Run the specific regression test
pytest tests/test_module.py::test_regression -v

# Run full suite — no regressions
pytest tests/ -q
```

### 4. If Fix Doesn't Work — The Rule of Three

- **STOP.**
- Count: How many fixes have you tried?
- If < 3: Return to Phase 1, re-analyze with new information
- **If ≥ 3: STOP and question the architecture (step 5 below)**
- DON'T attempt Fix #4 without architectural discussion

### 5. If 3+ Fixes Failed: Question Architecture

**Pattern indicating an architectural problem:**
- Each fix reveals new shared state/coupling in a different place
- Fixes require "massive refactoring" to implement
- Each fix creates new symptoms elsewhere

**STOP and question fundamentals:**
- Is this pattern fundamentally sound?
- Are we "sticking with it through sheer inertia"?
- Should we refactor the architecture vs. continue fixing symptoms?

**Discuss with the user before attempting more fixes.**

This is NOT a failed hypothesis — this is a wrong architecture.

---

- **S0 issue first**: When auditing a system with a known issue matrix (like AICC's 65 S0-S3 issues), prioritize S0 issues first — they block release and the test suite is ground truth.
- **Double-check tool behavior before flagging test failures**: When a test seems to read an obviously-wrong path, run the actual tool first to establish ground truth. Tests are authored against the expected output structure, but the tool's output structure is the authoritative source. A test that fails because the tool's internal subdirectory doesn't match the test's assumption = test is not buggy, it's correct against the actual behavior.
- **False-positive audit flags**: When delegating audit to subagents, treat the findings as preliminary. Re-read the flagged code directly and re-run the verification before accepting. Subagents achieve ~95% accuracy on first-pass audits, so the ~5% false-positive rate requires a re-read check on anything that seems surprising.

If you catch yourself thinking:
- "Quick fix for now, investigate later"
- "Just try changing X and see if it works"
- "Add multiple changes, run tests"
- "Skip the test, I'll manually verify"
- "It's probably X, let me fix that"
- "I don't fully understand but this might work"
- "Pattern says X but I'll adapt it differently"
- "Here are the main problems: [lists fixes without investigation]"
- Proposing solutions before tracing data flow
- **"One more fix attempt" (when already tried 2+)**
- **Each fix reveals a new problem in a different place**

**ALL of these mean: STOP. Return to Phase 1.**

**If 3+ fixes failed:** Question the architecture (Phase 4 step 5).

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Issue is simple, don't need process" | Simple issues have root causes too. Process is fast for simple bugs. |
| "Emergency, no time for process" | Systematic debugging is FASTER than guess-and-check thrashing. |
| "Just try this first, then investigate" | First fix sets the pattern. Do it right from the start. |
| "I'll write test after confirming fix works" | Untested fixes don't stick. Test first proves it. |
| "Multiple fixes at once saves time" | Can't isolate what worked. Causes new bugs. |
| "Reference too long, | Partial understanding guarantees bugs. Read it completely. |
| "I see the problem, | Seeing symptoms ≠ understanding root cause. |
| "One more fix | 3+ failures = architectural problem. Question the pattern, don't fix again. |

## Audit / Status Claim Boundary

**When the user asks whether a project "has all intended features" or whether a known-bug area is "working": do NOT collapse automated proof into live/product proof.** Treat this as a status audit, not a reassurance task.

Required answer shape:
1. Separate **implemented/present in source** from **covered by automated gates/tests** from **proven in a live headed/runtime workflow**.
2. For known fragile areas (chat streaming, persistence, retrieval, provider/model selection), run or cite the narrow regression gates/tests first, then state exactly what those tests prove.
3. Explicitly name what is still unproven: headed desktop smoke, real provider/model conversation, import→retrieval→chat, persistence after reload, packaging launch, performance timing, etc.
4. Use a blunt verdict: "RC-green" / "release-candidate" / "release-proven" are different states.
5. If a receipt claims an artifact exists, verify the artifact exists before repeating the claim as current filesystem truth.

Session-specific reference: `references/gloss-chat-rc-proof-boundary.md` captures the Gloss example where chat gates were green but live chat still needed headed-provider smoke.

**Multi-channel observability is not a single channel:** When a system has independent observability channels (e.g. an on-disk trace file written by the backend AND a Tauri event bus delivered to the frontend), `trace.first_token_seen = true` does NOT prove the JS-side store received the token. The trace is a side-channel; the event bus is a separate delivery path that can be lost independently (Vite HMR reload, panel unmount, webview backgrounding). For any "is the chat working?" audit, both channels must be verified: the trace file shows the backend completed, AND the user-visible UI shows the streamed content. When the trace says done and the UI says nothing, the issue is in the JS event delivery, not the Rust pipeline. The recovery path is to re-fetch from the authoritative DB on the `chat:done` handler so a lost streaming event doesn't lose the persisted response. See `tauri-desktop-development` "Pitfall — slow LLM first-token latency can lose the streaming UI entirely" for the full pattern.

## Audit False-Positive Prevention

**When auditing a codebase and delegating findings to subagents:** Subagent audits achieve ~95% accuracy — the ~5% false-positive rate is real. Before adding any flagged issue to an audit report:

1. **Re-read the exact file and line** the subagent cited
2. **Run the actual tool** (or write a one-liner probe) to establish ground truth
3. **Only then accept or reject** the finding

**Example from this session:** A subagent flagged `tests/test_codex_phase_family_and_receipts.py:9` as reading the wrong path (`out/codex/PHASE_MANIFEST.json` should be `out/PHASE_MANIFEST.json`). Manual re-read + `aicc build` execution confirmed: `aicc build --profile codex-implementation` creates an internal `codex/` subdirectory, so the double `codex/` IS correct. The test was right, the audit flag was wrong. **Ruled-out findings must be explicitly marked** in the audit report with the verification evidence — not silently dropped.

**The pattern that protects against false positives:** Read file → run tool → verify output → then accept. Skipping the "run tool" step is how false positives enter audit reports and waste fix effort.

### Pre-existing Compile Errors vs Introduced Errors

**When fixing a bug and `cargo check` shows errors:** Always distinguish pre-existing errors from introduced ones before concluding your patch broke something.

**The problem:** A multi-error output might contain:
- Errors introduced BY your patch (fix these)
- Errors that existed BEFORE your patch (ignore these)

**Example from this session:** Patching `libc::killpg(child_pid, ...)` in `check-runner/src/lib.rs` to fix an E0308 type mismatch. After the patch, `cargo check` still showed E0670 errors ("`async fn` is not permitted in Rust 2015") — but these were pre-existing across the entire crate, not introduced by the `.expect()` fix.

**Filter pattern:**
```bash
cargo check -p <crate> 2>&1 | grep -E "^error" | grep -v E0670  # only introduced errors
cargo check -p <crate> 2>&1 | grep -E "^error"                 # all errors including pre-existing
```

**Rule:** If pre-existing errors exist, note them explicitly in your patch record and verify your specific change doesn't add new error codes. Never let a pre-existing error masquerade as evidence that your fix broke something.

### Panic in Test Code vs Library Code

**When a task says "replace panic! in X" — first identify where the panics are:**

| Location | Doctrine Status | Correct Fix |
|----------|----------------|-------------|
| `src/` (library code) | **Forbidden** — panic corrupts callers | Replace with `Result` / `Error` |
| `tests/*.rs` (test code) | **Acceptable** — signals test contract violation, aborts test correctly | Leave unchanged |
| `#[cfg(test)]` blocks in lib | **Borderline** — test utilities in lib can be necessary | Evaluate case-by-case |

**The distinction matters:** A `panic!("expected {Variant}, got {other:?}")` in an exhaustive match arm inside a test file is CORRECT behavior — it fires only when the test's assumptions about enum variant coverage are violated. Removing it would hide real bugs. Replacing it with `Result` would complicate the test for no benefit.

**Before prescribing a panic! replacement:** Read the file. If the panics are in test functions (`#[test]`), leave them. If they're in `src/` library code, replace them with proper error handling.

## Rust Workspace Package Replay Pitfalls

When debugging Rust workspace source packages, distinguish package validation from extracted self-replay:

- `assert_package_validation` passing means the archive/sidecars are structurally coherent; it does NOT prove the archive can build/test in isolation.
- A replay failure in a temp directory with `failed to read /tmp/.../Cargo.toml`, `failed to find a workspace root`, or path deps escaping the extracted tree is an external-path-dependency replay blocker — not a missing-cargo/toolchain blocker just because `cargo fmt` printed help text.
- If the archive root is above the project root, normalize classification paths (`scripts/foo.py` and `Project/scripts/foo.py`) before declaring files stale.
- For synthetic root Cargo manifests, rewrite both `members` and `default-members`, while preserving `[workspace.dependencies]`, workspace lints, and other tables.
- Protect release-critical root docs (for example audit/source-truth files) before broad artifact archivers move/delete `*_AUDIT.md` or similar residue patterns.

See `references/rust-workspace-package-replay.md` for the AiDENs P32 package/replay debugging transcript distilled into reusable checks.

## Retrieval / RAG Ranking: First-Source Bias

**When a RAG/local-memory system keeps pulling from the first documents instead of the most actionable evidence:** treat it as a retrieval-ranking bug, not a prompt issue.

Debug checklist:
1. Inspect whether FTS/BM25 does one global `LIMIT k` across all selected sources. That can let early/repeated weak matches fill the pool before later sources are represented.
2. Check every retrieval path: hybrid retrieval, local-memory fallback/backend, degraded source-order fallback, and context assembly can each have their own sanitizer, scorer, or truncation.
3. Write a RED regression with many weak chunks in an `aaa-first` source and one strong actionable chunk in a later source. Assert the later actionable chunk ranks first for an action/improvement query.
4. Fix with fair per-source candidate pooling before global rerank, action-intent query expansion/rerank, deterministic score tie-breakers, and post-rerank truncation.
5. Re-verify scope preservation: explicit/none scopes must never widen, and SQL source filtering must stay parameterized.

See `references/retrieval-ranking-first-source-bias.md` for the fixture shape, failure mode, and durable fix patterns.

## Quick Reference

| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Root Cause** | Read errors, reproduce, check changes, gather evidence, trace data flow | Understand WHAT and WHY |
| **2. Pattern** | Find working examples, compare, identify differences | Know what's different |
| **3. Hypothesis** | Form theory, test minimally, one variable at a time | Confirmed or new hypothesis |
| **4. Implementation** | Create regression test, fix root cause, verify | Bug resolved, all tests pass |

## Rust Optional-Feature Compile Break: Gate Feature-Specific Helpers and Provide Stubs

**When a crate supports multiple optional backends/features and one feature-specific path references modules that are cfg'd out under another feature set:**

**Symptom:**
- `cargo test --features A` fails, even though the broken code path is only meant for feature `B`
- Typical pattern: a dispatcher or helper function compiles unconditionally and references `crate::b_backend::*`, but module `b_backend` only exists under `#[cfg(feature = "B")]`

**Example from this session:** `semantic-memory/src/search.rs` compiled `provekv_pool_vector_outcome()` even in `--features turbo-quant-codec` builds. That function referenced `crate::provekv_pool::load_or_decode_compact_pool_payload(...)`, but `provekv_pool` is only present under `feature = "poly-kv-pool"`. Result: turbo-only builds broke even though runtime config would have rejected the proveKV backend.

**Root cause:** Runtime validation is not enough. Rust still type-checks reachable items in the active cfg graph. A "this config would error at runtime anyway" argument does NOT prevent compile-time references from failing.

**Correct fix pattern:**
1. Gate the real feature-specific implementation with `#[cfg(feature = "B")]`.
2. Add a matching `#[cfg(not(feature = "B"))]` stub with the SAME signature.
3. The stub should return the same `InvalidConfig` / feature-disabled error the runtime would have produced.
4. Verify all relevant feature sets explicitly:
   - feature A only
   - feature B only
   - A + B together

**Template:**
```rust
#[cfg(feature = "poly-kv-pool")]
fn provekv_pool_vector_outcome(...) -> Result<_, _> {
    // real implementation
}

#[cfg(not(feature = "poly-kv-pool"))]
fn provekv_pool_vector_outcome(...) -> Result<_, _> {
    Err(MemoryError::InvalidConfig {
        field: "search.derived_vector_backend",
        reason: "provekv_pool_candidate_only requires the poly-kv-pool feature".to_string(),
    })
}
```

**Why the stub matters:** It preserves dispatch shape and keeps callers simple. You avoid scattering cfgs at each callsite while still keeping isolated feature builds healthy.

**Verification commands:**
```bash
cargo test --features turbo-quant-codec
cargo test --features poly-kv-pool
cargo check --features 'turbo-quant-codec,poly-kv-pool'
```

**Class-level lesson:** In Rust multi-backend code, every optional backend needs both (a) runtime validation and (b) cfg-correct compile boundaries. If one feature can be built alone, test it alone.

## Rust `cargo test` Cached Binary Stale After Source Patch

**When a test passes with `cargo test -p <crate>` but fails with `cargo test --workspace` after a source fix:**

**Root cause:** `cargo test --workspace` and `cargo test -p <crate>` produce DIFFERENT compiled binaries. The workspace binary is cached in `target/debug/deps/<crate>-<hash1>` while the per-package binary is in `target/debug/deps/<crate>-<hash2>`. Modifying source and running `-p` rebuilds hash2 but NOT hash1. The workspace run still uses the stale hash1 binary.

**Symptom:** `cargo test -p check-runner` → 12/12 pass. `cargo test --workspace` → `wave1_tests::select_backend_falls_back` FAILS with old assertion message.

**Detection:**
```bash
# Two different binaries exist
ls target/debug/deps/check_runner-*
# check_runner-03d03dfe95cb83f1  (stale, from workspace build)
# check_runner-b8c13f8add68fc10  (fresh, from -p build)
```

**Fix (minimal):**
```bash
# Remove only the stale binary, not the entire target/ directory
rm -f target/debug/deps/check_runner-03d03dfe95cb83f1*
# Then workspace test uses the fresh one
cargo test --workspace
```

**Fix (nuclear, if stale binaries are widespread):**
```bash
cargo clean -p <crate>  # removes ALL binaries for this crate, both hashes
cargo test --workspace
```

**Avoid:** `cargo clean` without `-p <crate>` — this deletes 70GB of artifacts for a 50-crate workspace and forces a full rebuild.

**Prevention pattern:** After patching a test that was failing in workspace runs, ALWAYS run:
```bash
cargo test -p <crate>      # verify the fix
cargo test --workspace    # verify no stale binary regression
```
If the second run fails, check for stale cached binaries before concluding the fix didn't work.

## Rust `cargo test --workspace` Environment-Dependent Flaky Test

**When a test behavior depends on whether Docker / a specific runtime / a specific file exists on the host machine:**

**Example from this session:** `select_backend_falls_back_to_host_when_allowed` tests that "auto" backend preference falls back to Host when Container is unavailable. But on a machine WITH Docker, ContainerBackend::new() SUCCEEDS, so the test gets Container instead of Host and fails.

**Root cause:** The test assumes container runtime is unavailable. This assumption is environment-dependent.

**Fix pattern — force the unavailable condition explicitly:**
```rust
#[test]
fn select_backend_falls_back_to_host_when_allowed() {
    let mut cfg = config();
    cfg.execution_backend_preference = "auto".into();
    // Use a runtime that definitely won't exist — forces container path to fail
    cfg.container_runtime_preference = "nonexistent_runtime".into();

    let backend = select_backend(&cfg).unwrap();
    assert_eq!(backend.kind(), ExecutionBackendKind::Host);
}
```

**Alternative patterns:**
- Mock the runtime probe function (inject a test-only probe that always returns `Err`)
- Use a test-only config flag that bypasses real runtime detection
- Guard the test with `#[cfg(not(feature = "container"))]` (simpler but loses coverage on container-enabled builds)

**Key principle:** Tests that probe the host environment are inherently flaky. Either mock the probe, or explicitly configure the test environment so the condition under test is deterministic.

## Rust Workspace Test Cache Stale Binary Pattern

**When `cargo test -p <crate>` passes but `cargo test --workspace` fails after a source fix:**

**Root cause:** `cargo test -p <crate>` and `cargo test --workspace` compile DIFFERENT binaries with different hashes in `target/debug/deps/`. The per-package binary rebuilds when you patch source, but the workspace binary may stay stale.

**Symptom:** Test passes in isolation (`-p`) but fails in workspace run with old assertion message or behavior.

**Detection:**
```bash
# Check for multiple binary hashes
ls target/debug/deps/<crate_name>-*
# If you see two hashes like:
#   check_runner-03d03dfe95cb83f1  (stale workspace binary)
#   check_runner-b8c13f8add68fc10  (fresh per-package binary)
# → workspace is using the stale one
```

**Fix (minimal — recommended):**
```bash
# Remove only the stale binary hash
rm -f target/debug/deps/<crate_name>-<stale_hash>*
# Then workspace test picks up the fresh binary
cargo test --workspace
```

**Fix (clean — nuclear but safe):**
```bash
cargo clean -p <crate>  # removes ALL binaries for this crate
cargo test --workspace
```

**Avoid:** `cargo clean` without `-p` — deletes entire `target/` (70GB+ for large workspaces) and forces full rebuild.

**Prevention:** After patching a test that was failing in workspace runs:
```bash
cargo test -p <crate>      # verify fix in isolation
cargo test --workspace    # verify no stale binary regression
```
If workspace fails but `-p` passes, check for stale cached binaries before concluding the fix didn't work.

**Session evidence:** V30 hardening session (2026-05-27/28) — `check-runner` wave1_tests passed 12/12 with `-p` but failed workspace run. Removed stale binary `check_runner-03d03dfe95cb83f1` directly, workspace passed immediately.

## Rust/String UTF-8 Pitfall

**When debugging string slicing panics or encoding errors in Rust code:**

Byte-slice truncation `&s[..n]` **panics** on strings containing multi-byte UTF-8 characters (em-dashes, unicode quotes, CJK, etc.) if `n` falls in the middle of a character. The panic message is typically "byte index N is not a char boundary."

**Correct pattern — use char iterator:**
```rust
// WRONG (byte-slice, can panic):
format!("{}...", &s[..max_len])

// CORRECT (char-wise, safe):
format!("{}...", s.chars().take(max_len).collect::<String>())
```

**Detection during investigation:**
- Search for `&s\[.*\.\.]` or `&str\[` patterns in Rust source
- Check `s.len()` vs `s.chars().count()` — len() is bytes, not chars
- Look for truncation utilities: search `truncate`, `safe_truncate`, `truncate_chars` across the codebase — implementations are often inconsistent across crates

**Root cause:** `&s[..n]` slices by byte offset, not character index. Rust `&str` is UTF-8 encoded. Characters can be 1-4 bytes. A 3-character string like "a—b" is 5 bytes; slicing `[0..3]` gives "a\—" (2 chars but 3 bytes), but if the cut point lands mid-codepoint, it panics.

**For full RecursiveIntell portfolio context**, see `high-assurance-engineering` skill references: `portfolio-critical-findings.md` documents the known buggy `truncate()` in `llm-output-parser/src/error.rs` vs the correct `truncate_chars()` in `semantic-memory/src/store_support.rs`.

## Rust `unsafe_code = "deny"` Workspace Conflict

**When a crate has legitimate architectural unsafe code (e.g. `libc::killpg`, raw syscall wrappers) and the workspace enforces `unsafe_code = "deny"` in `[workspace.lints.rust]`:**

1. **Never suppress globally** — `#[allow(unsafe_code)]` on the crate disables the check for the entire crate, not just the block.
2. **Never remove `unsafe_code = "deny"`** from the workspace — that's the compile-time law.
3. **The correct pattern:** Place `#[allow(clippy::arc_with_non_send_sync)]` (or whatever the specific deny is) directly on the `unsafe {}` block, with a comment explaining WHY this is the minimal correct primitive. Example:


> 📄 See [references/code-1.rust.md](references/code-1.rust.md) for the complete code.


4. **Verify:** `RUSTFLAGS="-D unsafe_code" cargo check -p <crate>` passes without globally disabling the deny.

5. **Test-only unsafe (lines 828/840 `std::env::set_var/remove_var`):** These appear only in `#[cfg(test)]` blocks. They don't appear in release builds. For cleanliness, isolate test utilities in a dedicated `#[cfg(test)]` module rather than keeping them in the main lib's test harness.

### `Option<T>` Unwrap in Post-Spawn Guards

**When fixing a type error involving `Option<T>` returned by a function that "can't fail" at a specific call site:** Use `.expect()` with an explicit safety comment that names WHY the `None` variant is structurally impossible at that point.

**Example from check-runner (`lib.rs:250`):**
```rust
// child_pid is Option<i32> — unwrap is safe here because:
// 1. The child was just spawned (Ok) — PID is always Some immediately after spawn
// 2. The Option exists because id() can technically fail pre-spawn (not our case)
// 3. If None, killpg(0, SIGKILL) would signal our own group — explicitly avoid that
let pgid = child_pid.expect("child PID must be Some immediately after spawn");
libc::killpg(pgid, libc::SIGKILL);
```

**The key pattern:** Explain WHY `None` is impossible — not just that it "can't happen." The comment must cover:
- Which specific precondition makes `None` structurally impossible
- What the fallback behavior (passing `None`-derived value) would cause, to show the stakes

**Always verify after patching:**
```bash
cargo check -p <crate> --lib  # only library, not tests (ignore pre-existing edition errors)
```
Then filter out pre-existing errors (e.g. E0670 async fn / Rust 2015 edition errors) to confirm zero new errors introduced.

## Embedded Git Repo Warning in Monorepos

**When `git add -A` produces the warning:** `warning: adding embedded git repository: <subdir>`

This means a subdirectory inside the repo contains its own `.git/` directory. Git treats it as a nested repo, not as files.

**Two legitimate fixes:**
- **Convert to submodule** (correct for external dependencies): `git submodule add <url> <subdir>`
- **Un-embed** (correct for crates that were copied in as snapshots): `git rm --cached <subdir> && rm -rf <subdir>/.git` then re-add as plain files

**Danger:** If you leave it embedded, a fresh `git clone` of the outer repo will populate the nested repo directory as an empty/inactive git worktree. The contents won't be cloned.

**Session case:** `fib-quant`, `poly-kv`, `turbo-quant` triggered this warning. Needs resolution before V30 closeout.

See `references/v30_hardening_session.md` for full session log including V30 audit findings and fix state.

## Subprocess Wrapping Pitfalls

**When wrapping a CLI tool (e.g. z.py, snap.py) as a subprocess in Python:**

1. **Path arguments that are directories vs filenames — read the tool's default behavior first.** Running `z.py package --out /tmp/mydir` where `/tmp/mydir` is an existing directory fails with `Is a directory` because z.py interprets `--out` as a **filename prefix**, not a directory. The fix: use a file-prefix like `/tmp/mydir/package` or a prefix in a fresh temp dir.

2. **Sentinel files without extensions.** Some tools write their primary output as a bare filename (no extension) alongside sidecars with extensions. In z.py, `bundle --out /tmp/bundle` writes BOTH `bundle` (the .zip, no extension) AND `bundle.manifest.json`, `bundle.report.md`, etc. Always check for and handle the bare-prefix file separately from the glob of `prefix.*` sidecars.

3. **Globbing in the wrong directory.** If the tool is passed `--out /tmp/mydir/bundle` and creates outputs in `/tmp/mydir/` (sibling to `bundle/`), glob in `zpy_prefix.parent`, not inside `zpy_prefix` itself. Verify the directory structure by running a print-once probe with `is_dir()` / `is_file()` checks before writing the collection loop.

4. **`capture_output=True` + `text=True`** — always use both together. Use `timeout=` explicitly. Handle `FileNotFoundError` (python not found) and `subprocess.TimeoutExpired` explicitly.

## Hermes Agent Integration

### Investigation Tools

Use these Hermes tools during Phase 1:

- **`search_files`** — Find error strings, trace function calls, locate patterns
- **`read_file`** — Read source code with line numbers for precise analysis
- **`terminal`** — Run tests, check git history, reproduce bugs
- **`web_search`/`web_extract`** — Research error messages, library docs

### With delegate_task

For complex multi-component debugging, dispatch investigation subagents:

```python
delegate_task(
    goal="Investigate why [specific test/behavior] fails",
    context="""
    Follow systematic-debugging skill:
    1. Read the error message carefully
    2. Reproduce the issue
    3. Trace the data flow to find root cause
    4. Report findings — do NOT fix yet

    Error: [paste full error]
    File: [path to failing code]
    Test command: [exact command]
    """,
    toolsets=['terminal', 'file']
)
```

### With test-driven-development

When fixing bugs:
1. Write a test that reproduces the bug (RED)
2. Debug systematically to find root cause
3. Fix the root cause (GREEN)
4. The test proves the fix and prevents regression

## Real-World Impact

From debugging sessions:
- Systematic approach: 15-30 minutes to fix
- Random fixes approach: 2-3 hours of thrashing
- First-time fix rate: 95% vs 40%
- New bugs introduced: Near zero vs common

**No shortcuts. No guessing. Systematic always wins.**



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
