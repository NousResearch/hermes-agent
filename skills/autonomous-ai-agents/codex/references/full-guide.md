<!-- Full original SKILL.md preserved verbatim below. Detailed domain and safety guidance lives here. -->

---
name: codex
description: "Delegate coding to OpenAI Codex CLI (features, PRs)." Use when working with codex.
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Codex, OpenAI, Code-Review, Refactoring]
    related_skills: [claude-code, hermes-agent]
---

## When to use

**WHEN to use:** Use when delegating a bounded implementation, refactor, review, or batch-fix task to the OpenAI Codex CLI in a repository with a verifiable working tree.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `codex` CLI | Delegate an implementation or review task | Prompt and approved working directory |

When a repository helper is required, use `run_script()` with a bounded timeout and inspect its exit status and stderr; do not treat partial output as success.

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

## Purpose

Codex provides structured guidance for the tasks it covers. Use the patterns and workflows below to complete your work efficiently.



- Building features
- Refactoring
- PR reviews
- Batch issue fixing

Requires the codex CLI and a git repository.

## Prerequisites

- Codex installed: `npm install -g @openai/codex`
- OpenAI auth configured: either `OPENAI_API_KEY` or Codex OAuth credentials
  from the Codex CLI login flow
- **Must run inside a git repository** — Codex refuses to run outside one
- Use `pty=true` in terminal calls — Codex is an interactive terminal app

For Hermes itself, `model.provider: openai-codex` uses Hermes-managed Codex
OAuth from `~/.hermes/auth.json` after `hermes auth add openai-codex`. For the
standalone Codex CLI, a valid CLI OAuth session may live under
`~/.codex/auth.json`; do not treat a missing `OPENAI_API_KEY` alone as proof
that Codex auth is missing.

## One-Shot Tasks

```
terminal(command="codex exec 'Add dark mode toggle to settings'", workdir="~/project", pty=true)
```

For scratch work (Codex needs a git repo):
```
terminal(command="cd $(mktemp -d) && git init && codex exec 'Build a snake game in Python'", pty=true)
```

## Background Mode (Long Tasks)

```
# Start in background with PTY
terminal(command="codex exec --full-auto 'Refactor the auth module'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Codex asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Key Flags

| Flag | Effect |
|------|--------|
| `exec "prompt"` | One-shot execution, exits when done |
| `--full-auto` | Sandboxed but auto-approves file changes in workspace |
| `--yolo` | No sandbox, no approvals (fastest, most dangerous) |

## PR Reviews

Clone to a temp directory for safe review:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main", pty=true)
```

## Parallel Issue Fixing with Worktrees


> 📄 See [references/code-1.sh.md](references/code-1.sh.md) for the complete code.


## Batch PR Reviews

```
# Fetch all PR refs
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# Review multiple PRs in parallel
terminal(command="codex exec 'Review PR #86. git diff origin/main...origin/pr/86'", workdir="~/project", background=true, pty=true)
terminal(command="codex exec 'Review PR #87. git diff origin/main...origin/pr/87'", workdir="~/project", background=true, pty=true)

# Post results
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## Rules

1. **Always use `pty=true`** — Codex is an interactive terminal app and hangs without a PTY
2. **Git repo required** — Codex won't run outside a git directory. Use `mktemp -d && git init` for scratch
3. **Use `exec` for one-shots** — `codex exec "prompt"` runs and exits cleanly
4. **`--full-auto` for building** — auto-approves changes within the sandbox
5. **Background for long tasks** — use `background=true` and monitor with `process` tool
6. **Don't interfere** — monitor with `poll`/`log`, be patient with long-running tasks
7. **Parallel is fine** — run multiple Codex processes at once for batch work

## Claude Code as Fallback When Codex Hits Usage Limits

When Codex hits "You've hit your usage limit" (ChatGPT account), Claude Code
is the immediate fallback:

```bash
cat /tmp/task-spec.md | claude -p "$(cat /tmp/task-spec.md)" \
  --dangerously-skip-permissions --max-turns 30 --model sonnet \
  --add-dir /path/to/workspace 2>&1 | tee /tmp/claude-task.log
```

Key differences from Codex:
- Use `claude -p` (print mode), NOT `--bare` (skips OAuth, fails with
  "Not logged in")
- `--dangerously-skip-permissions` auto-approves all tool use (like
  Codex `--dangerously-bypass-approvals-and-sandbox`)
- `--max-turns 30` prevents runaway (like Codex timeout)
- `--model sonnet` is the cost-efficient default for well-specified tasks
- `--add-dir` grants access to additional directories (for monorepo work)
- Claude Code can run `cargo check` / `cargo test` in the main workspace
  (same as Codex). Neither can in git worktrees (path deps fail).
- Claude Code had NO usage limit issues when Codex was blocked (2026-06-24).

When both Codex and Claude Code are available, Claude Code print mode is
preferred for Rust coding tasks because:
1. No usage limit (Codex ChatGPT auth has monthly caps)
2. Compiles correctly on first try (both Task 1 and Task 2 produced clean code)
3. Faster (5 minutes per batch vs Codex's 5-10 minutes + possible timeout)
4. Can verify compilation in the main workspace

## Auth — When OAuth Is ChatGPT-Account-Limited

`codex login status` reports one of two modes:
- `Logged in using OPENAI_API_KEY` — full model catalog available
- `Logged in using ChatGPT` — **only ChatGPT-allowed models work**; explicit model names like `gpt-5.1-codex-mini` or `gpt-5.5-codex` will 400 with "model is not supported when using Codex with a ChatGPT account"

When the user has ChatGPT-account auth (typical Hermes-on-OAuth setup), probe with the cheapest working model first instead of guessing:
```bash
codex exec --dangerously-bypass-approvals-and-sandbox -m gpt-5.3-codex-spark 'say OK' 2>&1 | tail -3
```

Confirmed-working models under ChatGPT auth (verify before relying on it — vendors reshuffle these often):
- `gpt-5.3-codex-spark`
- `gpt-5.5` (codex 0.139.0 default; confirmed working under ChatGPT
  account auth on 2026-06-12 with full multi-phase orchestration. Pick
  this unless you have a specific reason to pin a smaller/cheaper model.)
- `o3` (when explicitly allowed)

Avoid explicit `gpt-5.x-codex-mini` variants and `gpt-5.5-codex` on ChatGPT auth — they 400.

## Using `codex exec` as a Claude Code Substitute

When `claude` reports "Not logged in · Please run /login" and no `ANTHROPIC_API_KEY` is in env, `codex exec` is a clean fallback for the same work — it just needs the task spec as a string and a git repo:

```bash
cat /tmp/task-spec.md | codex exec --dangerously-bypass-approvals-and-sandbox \
    -C /path/to/repo -m gpt-5.3-codex-spark -s danger-full-access 2>&1 | tail -150
```

Key flags for parallel mechanical refactors (the same pattern that `claude --bare --print` would handle if auth worked):
- `-C <dir>` — workdir (no need to `cd` first)
- `-m <model>` — pin a known-working model
- `-s danger-full-access` — skip sandbox (required for agents that edit many files; `--full-auto` is the gentler variant)
- `--dangerously-bypass-approvals-and-sandbox` — required for non-interactive headless runs that don't pass through the approval prompt
- `cat spec.md | codex exec "..."` — pipe spec as the prompt (avoids shell-quoting headaches for long specs)

Always `tail -150` (or similar) on the output — Codex writes progress to stdout and the final summary is the only thing that matters; intermediate `tool_call` JSON pollutes the output buffer.

Output buffering quirk: `codex exec` only flushes stdout on exit. With `tail -150`, you see the summary at the end but the body is buffered. To monitor in-flight progress, use `background=true` with `process(poll)` / `process(log)` instead of `tail` on the foreground call.

## Parallel `codex exec` in Git Worktrees — Monorepo Path Dependency Trap

When dispatching parallel `codex exec` agents using `git worktree add`,
agents CAN write code but CANNOT run `cargo test` or `cargo check` when
the crate has path dependencies on sibling crates in the monorepo
workspace. The worktree checkout doesn't include the sibling crate
directories, so `cargo` fails with:
```
error: failed to get `fib-quant` as a dependency of package `poly-kv`
  No such file or directory (os error 2)
```

**Pattern that worked (2026-06-20, 6 parallel agents on semantic-memory):**
1. `git worktree add /tmp/codex-comboN HEAD` for each agent
2. Write per-agent task specs to `/tmp/codex-task-N.md` with explicit
   "Out of scope (do NOT touch)" sections listing files other agents own
3. Launch with `cat /tmp/codex-task-N.md | codex exec --dangerously-bypass-approvals-and-sandbox -C /tmp/codex-comboN/semantic-memory -m gpt-5.3-codex-spark -s danger-full-access`
4. Agents produce module files but can't verify with cargo test
5. Controller copies files back: `cp /tmp/codex-comboN/semantic-memory/src/module.rs /path/to/semantic-memory/src/`
6. Controller merges Cargo.toml and lib.rs changes (feature flags + module registrations) in the main workspace
7. Controller runs `cargo test --features "..."` once in the main workspace
8. Fix any compile errors (usually GraphEdgeType variant patterns, missing Serialize/Deserialize derives, or moved-value borrow errors)

**Key lesson:** The task spec should tell the agent NOT to run cargo test
(it will fail on path deps). Instead, tell the agent to focus on writing
correct code and the controller will verify. This saves 3-5 minutes per
agent of thrashing on unfixable build errors.

**When an agent doesn't produce the file:** Kill it after 5 minutes of
no output change, write the module yourself in the controller. This is
faster than re-dispatching.

## Parallel `codex exec` for Mechanical Refactors — Cross-File Conflict Pattern

When dispatching 3+ `codex exec` agents in parallel for invasive refactors (e.g. "fix everything in the slowdown audit"), they WILL conflict on shared files. The pattern that worked:

1. **Write each task spec to `/tmp/<task>.md`** (don't inline in the shell command).
2. **In each spec, write an explicit "Out of scope (do NOT touch)" section** listing every file another agent owns. Without this, the agents stomp on each other in shared files (test initializers, the chat caller, the AppState struct).
3. **After all 3 return, run `cargo check`/`cargo test` ONCE in the controller session.** Expect 5-10 cross-cutting compile errors (E0061 wrong-arity, E0599 wrong-lock-method, E0502 borrow conflicts). These are mechanical to fix in 1-2 minutes per file with `patch` — faster than re-dispatching a fix-up agent.
4. **Commit each workstream independently** so partial wins are preserved.

The agents won't tell you the conflicts cleanly — they each report "pre-existing compile errors" in their summary. The actual conflicts are usually visible as 2-3 instances of the same `E0061` / `E0599` error pointing at the same line in a file both agents edited.

Example concurrent refactor that worked: split a 7-item slowdown audit into 3 workstreams (DB-side, embedder-side, provider-side), launched them in parallel, fixed 6 cross-cutting errors in 4 minutes, committed.

## semantic-memory Optimization Parallel Dispatch (2026-06-23)

When implementing a 15+ phase optimization plan on semantic-memory, group phases by file overlap and dispatch parallel codex agents with file-isolation.

### Task spec requirements
- Write to `/tmp/codex-task-{ID}.md` (not inline)
- Include "Do NOT run cargo test -- workspace path deps won't resolve"
- Include "Do NOT modify Cargo.toml"
- Include explicit "Out of scope (do NOT touch)" listing files other agents own
- Provide exact code snippets, not descriptions

### File-isolation grouping
Group by which `src/*.rs` files each phase touches. Agents touching the same file MUST run sequentially. Agents touching different files CAN run in parallel.

### Model choice and quota-efficiency routing

Hermes-managed OpenAI Codex OAuth and the standalone `codex` CLI are distinct model surfaces. Do not infer standalone CLI support from Hermes support. As of 2026-07-11, Hermes/OpenAI-Codex exposes the GPT-5.6 family:

- `gpt-5.6-luna`, `gpt-5.6-luna-pro`
- `gpt-5.6-terra`, `gpt-5.6-terra-pro`
- `gpt-5.6-sol`, `gpt-5.6-sol-pro`

Route by the smallest model that can likely finish in one pass:

- Luna: bounded lookup, mechanical edits, test execution, formatting, receipt extraction.
- Terra: focused debugging, normal multi-file implementation, code review.
- Sol: architecture, ambiguous root-cause work, cross-repository synthesis, hard Rust changes.
- `-pro`: escalation only after the non-pro peer fails or the task is demonstrably high-risk/high-ambiguity.

Use an empirical Verified Work per Quota metric rather than model prestige:

`VWQ = verified_acceptance_points / (quota_debit * (1 + retry_count) * (1 + rework_fraction))`

Acceptance points are binary and receipt-backed: compile/test gate, task-specific regression, and requested artifact each count 1. `rework_fraction = controller repair time / total task time`. When the provider does not expose exact quota debit, use model calls as the temporary denominator and record the limitation; do not invent tier multipliers. Prefer the model with the highest observed VWQ for that task class. A cheap model that needs retries or controller repair can be less efficient than a stronger one-pass model.

Before a delegated run, record: task class, chosen model, expected duration, acceptance gates. Afterward record: pass/fail, retries, wall time, controller rework, and quota/reset evidence if exposed. Promote/demote routing only from repeated verified runs, not one anecdote.

Standalone CLI rule: probe support before a substantive run, and preserve the exact model suffix. On ChatGPT OAuth, bare `gpt-5.6` returned HTTP 400 unsupported on 2026-07-11, while `gpt-5.6-sol` succeeded in the standalone CLI the same day. Prior benchmark sessions also successfully ran standalone `gpt-5.6-sol` and `gpt-5.6-terra`. Do not collapse a failed bare-family probe into a claim that suffixed models are unavailable.

### Post-agent merge
After ALL agents complete, run `cargo check --all-features` ONCE in the main workspace. Expect 5-10 cross-cutting errors (missing struct fields, wrong error variant names, borrow conflicts from `.collect()` not bound before `.retain()`). Fix with targeted `patch` calls, not re-dispatching agents.

### Struct field addition pitfall
When adding a field to a struct used across many files (e.g. `Bm25Hit` in search.rs):
- ALL construction sites must be updated, including test files
- Different modules may have same-named structs (`search::VectorHit` vs `vector_backend::VectorHit`) -- only add to the right one
- Automated scripts that insert after pattern matches WILL corrupt struct definitions and match expressions. Use targeted `patch` per file instead.

The "Cross-File Conflict Pattern" above works for Rust where compile errors are mechanical (1-2 patches per error). It does **not** work for TypeScript when a refactor agent touches state-management libraries it doesn't fully understand. Specific failure shapes seen on the Gloss slowdown pass:

- `useStore(useShallow(s => ({ a, b, c })))` — zustand v5's generic inference returns `unknown` when the selector shape is large, breaking the destructure with `TS2339: Property 'X' does not exist on type 'unknown'` across 20+ destructure lines
- The agent **removes an import while adding a new one** (e.g. drops `useNoteStore` import while adding `Virtuoso` import) — leaves broken references like `Cannot find name 'useShallow'`
- Implicit-any errors: `Parameter 'm' implicitly has an 'any' type` on callback params inside the refactored component
- Total: **41 TypeScript errors** from one React agent's diff vs ~5-10 from a Rust agent

**Do NOT re-dispatch a fix-up agent.** Re-dispatching with "please fix the 41 TypeScript errors" produces a 3rd diff that breaks 10 more things. The agent's mental model of the store shape is gone; it will guess at types and break the file worse.

**Pattern that worked — revert in controller:**
```bash
# Identify the agent's touched files from its report
git status --short | grep "\.tsx\?$"

# For each TS-broken file, revert to HEAD. NOTE: if there are PRE-EXISTING
# uncommitted changes in those files (not from this agent), capture them
# first with `git diff <file>` and re-apply after the revert.
git diff src/components/chat/ChatPanel.tsx > /tmp/agent-changes.patch
git checkout HEAD -- src/components/chat/ChatPanel.tsx \
                     src/components/sources/SourcesPanel.tsx \
                     src/components/notebooks/NotebookSidebar.tsx
```

Then verify:
```bash
npm run build 2>&1 | grep -E "error TS" | head -5
# Should show 0 errors, OR only pre-existing errors from other
# uncommitted work in the working tree.
```

Capture the perf win the agent tried to land (e.g. `useShallow` selectors, Virtuoso list, React.memo wraps) as a follow-up spec. Run it as a FOCUSED single-file task with explicit "do not remove existing imports" and "match the pattern at `src/components/layout/StatusBar.tsx:21-34`" — pointing at a working example in the same repo is the difference between success and 41 errors.

**The zustand v5 individual-selector alternative (preferred for v5):**

The `useShallow` destructure pattern fails because zustand v5's generic
inference returns `unknown` on large object shapes. The pattern that
works in v5 (verified 2026-06-10 on Gloss):

```tsx
// DON'T — useShallow with large object breaks inference
const { a, b, c, ... } = useStore(useShallow(s => ({ a: s.a, b: s.b, c: s.c })));

// DO — individual selectors
const a = useStore(s => s.a);
const b = useStore(s => s.b);
const c = useStore(s => s.c);
```

This produces more lines but compiles reliably and re-renders on
exactly the slice that changed. Use this in the agent spec.

**Why this beats "fix it yourself in 41 patches":** The patch tool doesn't have the type inference the agent had when it wrote the broken code. You'll spend 20+ minutes guessing at types before you can fix the first compile error. Reverting and re-spec'ing is faster.

**Red flag to revert vs fix:** If `tsc --noEmit` shows > 20 errors and the agent's diff touched > 3 files, revert. The per-file patch cost grows superlinearly with type-inference breakage — there's no incremental win in fixing one error at a time when the rest of the file is structurally wrong.

## Honest Scope-Tracking: The "Shipped N / Deferred M" Handoff Doc

When a multi-batch workstream lands partially (cross-cutting agent conflicts force deferrals), write a single handoff doc at the end: `HOSTILE_AUDIT_FINDINGS_<project>_<date>.md`. Shape:

1. **What this pass did** — committed work grouped by workstream, with file:line for every claim
2. **What was NOT done (and why)** — explicit list of deferred items grouped by reason:
   - "Agent's selector refactor was 41 TS errors" → concrete reason
   - "Capability gap, not effort gap" → honest capability note
   - "Out of session time" → be honest
3. **Risk assessment of the deferred work** — does it affect correctness? edge cases only? common path? Prized-tool polish?
4. **Receipts** — verification command + output, one per line
5. **Hostile-auditor handoff** — one paragraph: "next session should pick up from <list>"

This doc is the user's "did it actually ship" answer. The user's mental model after the session is "X was done, Y was deferred, here's why" — a flat list of "fixed everything" without the deferral rationale is a lie.

**Pattern that worked:** write the doc BEFORE running the final verification, then update the receipts section with actual command output. This prevents you from claiming "170 tests pass" before you've actually run the tests.

## Multi-Phase Codex Orchestration: Receipts per Phase

When a repair / migration / refactor spans 5-10 phases and each phase
gets its own `codex exec` session, you need **per-phase receipts** to
(a) prove each phase did what it claims and (b) recover when a late
phase regresses an earlier one. The pattern that worked on a 10-phase
Gloss repair run:

1. **One `codex exec` per phase, in dependency order.** Don't try to
   make one agent do all phases — context window exhaustion, scope
   drift, and the inability to bisect later regressions are all
   reasons to keep phases isolated.

2. **Wrap each invocation with a small bash launcher** that:
   - Writes the phase spec + orchestrator preamble to `PHASE_<id>_PROMPT.md`
   - Captures the full JSONL session stream to `PHASE_<id>.codex.jsonl`
   - Extracts the last `agent_message` text to `PHASE_<id>.codex.txt`
   - Records the exit code to `PHASE_<id>.codex.exit`
   See `scripts/run_codex_phase.sh` in this skill for a reusable
   implementation. The key CLI flag set:
   `codex exec --sandbox workspace-write -C <root> --ephemeral --json`.

3. **Each phase's prompt names the receipt files the agent must
   write.** Without explicit output paths, the agent will invent
   its own (or skip writing entirely) and you'll have nothing to
   audit later.

4. **Launch with `terminal(background=true, notify_on_complete=true)`**
   from the controller. Do NOT block the controller waiting on a
   long codex run — Hermes `process wait` is clamped to 60s and
   `terminal` foreground is 600s. Background + notify is the only
   sane pattern for phases that take 5-20 minutes.

5. **Between phases, re-run the static gates yourself** in the
   controller. The codex agent may claim "all green" but the static
   gates are cheap (sub-second) and catch scope drift, missed file
   paths, and structural regressions.

6. **When a codex phase subsumes later phase work** (e.g. Phase 02
   recognized that the static gates for Phase 03/04 couldn't pass
   without it and applied those changes in the same diff), write
   back-pointer receipts for the subsumed phases:
   - `PHASE_03.md` / `PHASE_03.SUMMARY.md` that say "substantive
     work is in PHASE_02.md" plus the verification you just ran
   - This keeps the per-phase receipt chain intact for hostile-
     auditor handoff even though the actual codex sessions are
     fewer than the phase count

7. **Pitfall — `--output-schema /dev/null`:** if you pass an empty
   or non-JSON file to `--output-schema`, codex exits immediately
   with "Output schema file /dev/null is not valid JSON" and
   consumes the prompt before running. If you don't need a schema,
   omit the flag entirely. The launcher script omits it by default.

8. **Pitfall — `process wait` is not a blocking wait.** It returns
   after 60s with a "still running" status. The notification fires
   when the process actually exits. Polling with `process(poll)` is
   fine for status checks; don't chain `wait` calls together as if
   they would block longer.

## Worktree Workspace Dependency Resolution Pitfall

When creating git worktrees from a Cargo workspace with path dependencies
(e.g. `poly-kv = { path = "../poly-kv" }`), the worktree only contains the
target crate — sibling crates are NOT present. This means:

- `cargo check` and `cargo test` fail in the worktree with errors like
  `failed to get 'fib-quant' as a dependency of package 'poly-kv'`
- Codex agents in worktrees CAN write code (they can read the source files
  and create new modules) but CANNOT verify compilation
- The agents get stuck in retry loops trying to run cargo commands

**Pattern that works:**
1. Create worktrees for code isolation (prevents file conflicts)
2. Let Codex agents write the code — they produce correct files even without
   compilation verification
3. Kill agents after ~5 min if they're stuck on cargo errors (they've usually
   already written the files by then)
4. Copy produced files back to the main workspace
5. Verify compilation + run tests in the main workspace (controller)
6. Fix cross-cutting errors (Cargo.toml feature flags, lib.rs module
   registrations, missing derives) in the controller

**When to NOT use worktrees:** If the task only creates new files (no shared
file modifications), worktrees are overkill. Use the main workspace directly
and let Codex agents write files in parallel. The only risk is
Cargo.toml/lib.rs conflicts, which are mechanical to fix.

**Alternative for workspace crates:** Point Codex at the main workspace
directory (`-C /path/to/workspace/`) instead of a worktree. The agents can
run cargo commands but risk file conflicts if run in parallel. Serialize
agents that touch the same files.

For long specs, the spec is piped via stdin but `codex exec`'s exit code is what matters — the final agent summary may be truncated to 50-100 lines by `tail -N` even if the agent ran for 5 minutes. To preserve the full output:
```bash
codex exec "..." 2>&1 | tee /tmp/codex-task-N.log | tail -150
cat /tmp/codex-task-N.log | tail -200
```

`tee` before `tail` keeps the full log; `tail` is for quick inspection during orchestration.

## Worktree Dependency Resolution Failure (Monorepo Path Deps)

When using `git worktree add` to isolate parallel Codex agents, the worktree
only contains the subdirectory you're working in — NOT the full monorepo
workspace. If the crate has `path = "../sibling-crate"` dependencies in
Cargo.toml, `cargo test` and `cargo check` will fail in the worktree with:

```
error: failed to get 'fib-quant' as a dependency of package 'poly-kv v0.1.0-alpha.1 (/tmp/codex-comboN/poly-kv)'
  No such file or directory (os error 2)
```

The Codex agent will get stuck trying to run verification commands that
can't work, repeatedly hitting the same dependency resolution failure.

**Fix**: Add to the task spec explicitly:
> "Do NOT attempt to run cargo test or cargo check — the worktree doesn't
> have workspace path dependencies. Write code by inspection and ensure
> it compiles conceptually. The controller will run verification in the
> main workspace after copying files back."

Then run `cargo test --lib --features "<feat>"` in the main workspace
after copying the produced files back.

**Alternative**: Use `git worktree add` from the workspace root (not the
crate subdirectory) so the full workspace is available. But this means
all agents share the same working tree and conflict on shared files
(Cargo.toml, lib.rs). The worktree-per-crate approach avoids file
conflicts but can't verify compilation.

**Pattern that worked** (semantic-memory 7-combination pass, 2026-06-20):
1. Worktree per agent at `/tmp/codex-comboN/` containing just semantic-memory/
2. Agents write module files + modify Cargo.toml + lib.rs in their worktree
3. Controller copies module files back to main workspace
4. Controller merges Cargo.toml and lib.rs changes manually (these are the
   shared files that would conflict if agents worked in the same tree)
5. Controller runs `cargo test --lib --features "<feat>"` for each module
   in the main workspace

4 of 6 agents completed successfully. 2 were killed after 7+ minutes stuck
on cargo test failures in their worktrees. The controller wrote the missing
module (rl_routing.rs) directly using the same task spec.

## Missing-Input File After a Foreground `tee`/`>` Timeout

A `codex exec` task that runs from a background `terminal` will fail in ~28s with `bash: <file>: No such file or directory` if the **stdin file or output redirect target was supposed to be created by a prior foreground call that timed out**.

**What happened (MiniRecall Phase 01, 2026-06-14):**
- Controller composed a 12KB stdin file via `cat > /tmp/stdin.md` inside a foreground `terminal(command=..., timeout=900)`. The call hit the 600s cap (foreground max) before the heredoc completed. The `/tmp/stdin.md` file was **never created** because the heredoc was inside the timed-out shell, not a separate step.
- Controller then re-dispatched from a background `terminal(background=true, ...)` reading the same `/tmp/stdin.md` path. The background process exited 28s later with "No such file or directory" and a 0-byte log.
- The bg failure looked like Codex hung — but Codex never started; the redirect target was missing.

**Recipe to prevent:**
1. **Verify the input file exists immediately before `codex exec`.** A bare `ls -la /tmp/stdin.md` (not part of the timed-out heredoc) tells you whether the file landed.
2. **Build the stdin file in a separate foreground call** that finishes in <5s, *then* dispatch codex. Don't pipeline composition into the same shell call that runs codex.
3. **If a file is missing, do not retry from background.** Compose the file in a fresh foreground call first.
4. **Add `set -euo pipefail` to the launcher script** so a missing redirect aborts before codex starts and the exit code reflects the setup failure, not "codex completed normally in 0s."

**Self-check that would have caught this:** `ls -la /tmp/stdin.md` at the top of the bg launcher returned "No such file or directory" — the controller didn't run that check before dispatch.

## Stuck-`pytest` Inside a Codex Sandbox

When a `codex exec` task's spec says "run `pytest` and report," the agent may end up running a `pytest` process it cannot kill from a separate shell. The agent then waits for the process to drain, polling tool sessions that are also stuck. This is the same thrash pattern, but the *stuck process is inside Codex's tool sandbox*, not the controller's bash.

**Symptoms (MiniRecall Phase 01, 2026-06-14):**
- Codex log shows `codex: The stuck pytest session appears isolated from the process listing exposed to new shell calls, so I can't terminate it directly from another command.`
- `pgrep -af pytest` from the controller returns nothing — the test is in Codex's tool process, not a controller-visible process.
- Killing the `codex exec` parent does not kill the pytest (it can survive the parent in the tool sandbox's child process tree).

**Recipe:**
1. **Kill the codex process from the controller** with `pkill -9 -f 'codex exec'` after 8-10 min of no progress. The sandbox-orphaned pytest typically dies with the parent because the tool sandbox tears down child processes on parent termination.
2. **Verify with `pgrep -af pytest` after the kill** — if a real orphan survives, kill it directly; if the controller's pgrep returns nothing, the test was sandbox-contained and the parent-kill was sufficient.
3. **The right answer in the spec is "validate this in a narrower way":** for a gateway, `python3 -c "from fastapi.testclient import TestClient; ..."` is faster and more killable than a full `pytest` session. Tell the agent to use the narrow check first and only escalate to `pytest` if needed.
4. **Sanity-check your spec** before sending it: any test command the agent can't interrupt (`pytest` with no `--timeout`, `npm test` that hangs on a watcher) is a stuck-tooling trap.

## Time-Boxing Thrashing Codex Agents

A `codex exec` task that hasn't reported within **10-15 minutes** is likely stuck in a thrash loop: re-running `cargo check` or `tsc --noEmit` after every micro-edit, hitting the same compile error repeatedly, or making validation-only changes that the user doesn't want. Don't wait it out — kill it and do the work in the controller session.

**How to tell it's thrashing (vs. just slow on a real long task):**
- `git status --short` shows no new files and no new edits to listed files
- The agent's `output_preview` (in `process(poll)`) is empty or shows the same retry line
- `cargo test` / `npm run build` wall-time is < 30s per pass; if the agent is 50 min in, it has run dozens of cycles with no progress
- The agent's reported "pre-existing compile errors" line hasn't changed in 10+ min

**Kill protocol:**
```python
process(action="kill", session_id="<id>")
# Then re-verify state
git status --short  # confirm no new files from the agent
cargo check --manifest-path src-tauri/Cargo.toml --features <X> 2>&1 | tail -5
```

**Then do the work in the controller.** Mechanical refactors (1-10 line patches per item) are faster with `patch` + `cargo check` than re-dispatching an agent. Re-dispatching a "fix the 41 errors" agent produces a 3rd diff that breaks 10 more things; the agent's mental model of the broken state is the problem.

**Time-box the controller's fallback too.** If the controller can't ship the items in 5-10 focused patches, write the handoff doc with the deferral rationale and stop. The "I tried for 50 min and it didn't work" failure mode is worse than "I shipped what I could and documented the rest" — the latter gives the user a decision.

## Late-Arriving Completion Notifications — THE Critical Pattern

When you launch multiple `codex exec` agents in parallel via `terminal(background=true, notify_on_complete=true)`, their completion notifications can arrive **minutes to hours after the agent actually finished writing code**. The agent writes files, exits, and the notification is queued — but the controller session may have already moved on.

**This is the #1 source of duplicate work and wasted effort.** The pattern that happened on the AiDENs integration session (2026-06-20):

1. Launched 4 codex agents in parallel (phases 2-5, each touching a different crate)
2. Checked `git diff --stat HEAD` 10 minutes later — saw no changes
3. Concluded "codex didn't write anything" and started implementing directly
4. The agents HAD written code — I was checking the wrong path / the diff was misleading
5. Completion notifications arrived 5-15 minutes later, confirming the agents had written real code with passing tests
6. Some of my direct implementation duplicated what codex already did

**The fix — WAIT for the notification before implementing directly:**

```python
# After launching background codex agents:
# 1. Do NOT check git diff and conclude "nothing happened"
# 2. Do NOT start implementing the same task directly
# 3. WAIT for the notify_on_complete notification
# 4. When it arrives, THEN check what the agent wrote:
#    git diff --stat HEAD -- crates/<specific-crate>/
#    cargo test -p <crate-name>
# 5. Only implement directly if the notification shows the agent failed
```

**Time-box for waiting:** If an agent hasn't reported within 10-15 minutes, it may be stuck (see "Time-Boxing Thrashing Codex Agents" below). But if it's a small crate modification (< 500 lines target), the agent likely finished in 2-5 minutes and the notification is just delayed. Poll the tee log file directly (`tail -20 /tmp/aidens-phaseN.log`) to see if the agent has exited, rather than relying on the notification.

**Verification before assuming "nothing happened":**
```bash
# WRONG: git diff --stat HEAD  (can miss changes if there are unrelated dirty files)
# RIGHT: git diff --stat HEAD -- crates/<specific-crate>/
# RIGHT: wc -l /tmp/<task>.log  (grows while codex is running, stops when it exits)
# RIGHT: pgrep -c "codex exec"  (0 = all agents have exited)
```

**The tee log file is the real-time signal.** `codex exec` buffers stdout and only flushes on exit. But `tee /tmp/<task>.log` writes to disk continuously. When the log file stops growing and `pgrep -c "codex exec"` returns 0, the agent has finished — even if the Hermes notification hasn't fired yet.

## Late-Arriving Codex Agent Re-Modification

A `codex exec` task that you killed (or that "completed" hours ago) can re-surface in the system: a delayed completion notification fires after you've already committed downstream work. The late arrival's diff may or may not compile — it depends on whether the agent's working tree state was still valid when it wrote the result.

**The most common variant: codex DID write code, you just didn't see it yet.**
When you dispatch 4-5 codex agents in parallel and check `git diff --stat HEAD`
10 minutes later, some crates may show zero changes — leading you to believe
the agent failed. In reality, the agent may have written code to the files but
the completion notification hasn't fired yet, or the `git diff --stat HEAD`
output is misleading because of unrelated working-tree changes.

**Verification protocol before re-implementing:**
1. Check `pgrep -c "codex exec"` — if >0, agents are still running. Wait.
2. Check `git diff --stat HEAD -- crates/SPECIFIC_CRATE/` — scope the diff
   to the specific crate, not the whole workspace. Unrelated changes in
   sibling crates can mask the agent's work in the aggregate stat.
3. If the diff is truly empty AND the process has exited, THEN implement
   directly with `patch`.
4. If a late completion notification arrives after you've already
   implemented, compare the two versions. The codex version may be more
   thorough (it read more source files). If the codex version is better,
   revert your patch and use the codex version.

**Pattern that happened (AiDENs integration, 2026-06-20):**
- Dispatched 4 codex agents (phases 2-5) in parallel
- Checked `git diff --stat HEAD` — showed only Cargo.lock changes, no crate changes
- Re-implemented phases 2 and 6 directly with `patch`
- Late notifications arrived: codex HAD written code for all 4 phases
- Both versions converged (same method signatures, same delegation pattern)
- The codex version for phase 6 was MORE thorough (added governance
  integration in the turn executor, control records, memory grounding
  receipts — things the spec mentioned but I didn't implement in my patch)

**Lesson: the codex agent is more capable than its real-time output suggests.
It buffers output and only flushes on exit. The 5,500-line log you see in
`process(poll)` is the agent reading source files — it's doing research
before writing. Give it time.** The late arrival's diff may or may not compile — it depends on whether the agent's working tree state was still valid when it wrote the result.

**Pattern that happened (Gloss slowdown fix, 2026-06-10):**
- I dispatched 3 codex agents in parallel for React-perf, command-palette, reliability
- The React-perf agent's first diff produced 41 TS errors; I reverted its 4 files to HEAD and reported done
- ~10 min later, a late notification fired: the React-perf agent had re-completed with a *different* diff that DID compile
- I had already shipped a handoff doc claiming "React perf deferred"
- A final `cargo check` + `npm run build` was needed before committing the late result as a bonus batch

**Re-check protocol when a late completion fires:**
```bash
# 1. Did the late agent actually change anything?
git status --short | head -20
git diff --stat HEAD 2>&1 | tail -3

# 2. Run the full verification gauntlet
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo test --manifest-path src-tauri/Cargo.toml --features <X> --lib --no-fail-fast 2>&1 | tail -3
npm run build 2>&1 | tail -3
npm test 2>&1 | grep '"status"'
# 5 AGENTS.md gates

# 3. If green, commit as a bonus batch and update the handoff doc
# If red, revert those specific files and document as "agent thrashed, no usable result"
```

**Don't** trust a late-arrival's "I built it" self-report blindly. The agent may have built against a state that's no longer in the working tree. Re-verify against `HEAD` (not the agent's starting state).

## "Reads But Doesn't Write" — Large File Context Exhaustion

A codex agent (especially `gpt-5.3-codex-spark`) dispatched to modify a
large file (1,500+ lines) will often **read the entire file, search for
API signatures across the workspace, then exit without writing any
changes**. The agent's context budget is consumed by reading and
searching, leaving nothing for the actual edit.

**Symptoms (AiDENs runner integration, 2026-06-20):**
- Agent ran for ~90 seconds, produced 5,500+ lines of log
- Log shows `rg` searches, `cat Cargo.toml`, reading source files
- `git diff --stat HEAD` shows 0 changes on the target crate
- Agent may modify `Cargo.toml` or `Cargo.lock` (trivial edits) but
  not the actual source file
- No error message — the agent simply finished its turn

**This is NOT thrashing** (no repeated failed attempts) and NOT a
timeout (the agent exited normally). It's context budget exhaustion
from reading too much before writing.

**When to expect it:**
- Target file is 1,500+ lines
- Task requires understanding existing struct/function layout before
  making surgical additions (e.g. "add fields to a struct in the
  middle of a 1,929-line file")
- Agent needs to read sibling crate APIs to know correct types

**Fix — do it in the controller:**
When the target file is large and the change is surgical (add fields,
add methods, add imports), use `patch` directly instead of dispatching
codex. You already know the exact location from `read_file` with
offset/limit — a `patch` with the old/new string is faster and more
reliable than hoping the agent has enough budget left after reading.

**When codex IS still worth trying on large files:**
- The change is scattered across many files (the agent's file-reading
  is productive — it's finding all the call sites)
- The change requires understanding patterns you don't know yet
  (the agent's API research is valuable)
- Use `gpt-5.5` instead of `gpt-5.3-codex-spark` — the larger context
  window handles bigger files

**Pattern that worked (AiDENs phases 1-7, 2026-06-20):**
- Phases 1-5 targeted small files (112-475 lines each) → codex
  succeeded on 4 of 5 (phase 1 wrote 325 lines, phase 3 wrote 203
  lines, phases 4-5 wrote 179+52 lines)
- Phase 2 (112 lines) codex read but didn't write → controller
  patched in 2 minutes
- Phase 6 (1,929-line runner) dispatched with gpt-5.5 → codex DID
  write 298 lines (struct fields, builder methods, governance
  integration in turn executor, memory grounding receipts) but the
  controller initially thought it wrote nothing because `git diff
  --stat HEAD` was checked against the wrong path. The lesson: check
  `git diff` on the SPECIFIC crate directory, not just the workspace
  root, and WAIT for the completion notification before concluding
  the agent didn't produce output.

## Late Completion Notifications — Wait Before Re-Implementing

Codex agents launched with `background=true, notify_on_complete=true`
will send a completion notification AFTER the process exits. The
notification can arrive minutes after the process actually finished,
especially when multiple agents run in parallel and the notification
queue backs up.

**Critical lesson (AiDENs full integration, 2026-06-20):**
The controller checked `git diff --stat HEAD` too early, saw no changes,
and re-implemented phases 2-5 directly — even though the codex agents
HAD written the code. The late-arriving notifications confirmed all 4
agents completed successfully with real code changes. The controller's
re-implementation was redundant work that converged on the same result.

**Rules to avoid redundant re-implementation:**
1. **Wait for the `notify_on_complete` notification** before concluding
   a codex agent didn't produce output. The notification is the
   authoritative signal — `git diff` checked before it fires may not
   reflect the agent's writes if the process is still flushing.
2. **Check `git diff` on the specific crate directory**, not just
   `git diff --stat HEAD` at the workspace root. The root-level diff
   can be polluted by Cargo.lock changes, sibling crate modifications
   from build artifacts, or unrelated untracked files.
   ```bash
   git diff --stat HEAD -- crates/aidens-kernel-kit/
   # NOT just:
   git diff --stat HEAD
   ```
3. **When multiple agents run in parallel**, wait for ALL
   notifications before starting controller-side implementation.
   A `pgrep -c "codex exec"` returning 0 means processes have exited,
   but notifications may still be in-flight.
4. **If you must check before notifications arrive**, use `pgrep` to
   verify the process has exited, then check `wc -l` on the tee log
   to see if the agent produced substantive output, then check
   `git diff` on the specific target directory.
5. **The user's instruction "try to learn from how the codex agents
   responded (later than you thought)" is a direct correction** —
   encode it: do not re-implement what a codex agent already wrote
   just because the notification hasn't arrived yet.

## Git Worktree Failure on Massive Monorepos

`git worktree add` on a monorepo with 7,800+ files (including large
docs/archive trees) can fail with `error: unable to write file` on
hundreds of files, followed by `fatal: Could not reset index file to
revision 'HEAD'`. This happens when the worktree checkout runs out
of file descriptors, disk I/O bandwidth, or hits filesystem limits
during the mass file creation.

**What happened (AiDENs parallel phase launch, 2026-06-20):**
- Attempted `git worktree add -b feat/phase1 /tmp/aidens-phase1 HEAD`
  on the Libraries monorepo (7,825 files, many large JSON manifests)
- Checkout hit hundreds of "unable to write file" errors on
  docs/post-salvage-validation/sidecars/*.json and similar large files
- Final error: `fatal: Could not reset index file to revision 'HEAD'`
- Worktree was partially created (corrupt state)

**Fix for sub-workspaces:**
If the target is a **sub-workspace** (has its own `Cargo.toml` with
`[workspace]`), run codex directly in that directory instead of
creating a worktree. The sub-workspace has its own `target/` directory
and its own dependency resolution. Multiple codex agents can run
sequentially in the same directory without worktree isolation, as
long as they touch different files.

**When worktrees ARE needed:**
- When agents will modify the SAME files (Cargo.toml, lib.rs)
- When the monorepo is small enough for the checkout to succeed
- When you need parallel execution AND file isolation

**When worktrees are NOT needed:**
- Sub-workspace with its own Cargo.toml (run codex in-place)
- Sequential phases touching different crates (no parallel conflict)
- Small surgical patches (controller does it directly)

## Same-Workspace Parallel Codex (No Worktrees Needed)

When Codex agents touch **different files** in the same crate, worktrees
are unnecessary. Launch them directly in the main workspace with
`-C /path/to/crate`. Each agent writes a different file with zero
conflicts.

**Pattern that worked (2026-06-20, semantic-memory audit remediation):**
1. Write task specs to `/tmp/codex-task-N.md` — each spec names exactly
   one file to modify and says "Do NOT run cargo test — write code by
   inspection, the controller will verify"
2. Launch 3 agents in parallel:
   ```bash
   cat /tmp/codex-task-N.md | codex exec --dangerously-bypass-approvals-and-sandbox \
     -C /path/to/semantic-memory \
     -m gpt-5.3-codex-spark -s danger-full-access \
     2>&1 | tee /tmp/codex-task-N.log | tail -150 &
   ```
3. All 3 completed in ~2-3 minutes, each wrote a different file
4. Controller ran `cargo check` and `cargo test` ONCE after all finished
5. One fixup: test functions calling gated APIs needed `#[cfg(feature)]`
   — mechanical 4-line patch per test function

**When this works:** Each agent touches a different file. No shared file
modifications (Cargo.toml, lib.rs) from multiple agents.

**When worktrees ARE still needed:** Multiple agents modify the SAME
file (Cargo.toml, lib.rs). Use the worktree pattern from the section
above to isolate, then merge in the controller.

**Key lesson:** The "Do NOT run cargo test" instruction is critical for
same-workspace parallel agents. Without it, agents thrash trying to
build while another agent's file is mid-edit, producing spurious compile
errors that don't reflect the final state.

## Claude Code as Fallback When Codex Hits Usage Limits

Codex ChatGPT-account auth has a usage limit. When `codex exec` returns
`ERROR: You've hit your usage limit`, switch to Claude Code (`claude -p`).

### Setup verification (do this BEFORE you need it)
```bash
claude --version          # needs v2.x+
claude auth status        # should show loggedIn: true, authMethod: claude.ai
```

### Claude Code print mode (non-interactive, like codex exec)
```bash
claude -p "task description" \
  --dangerously-skip-permissions \
  --max-turns 30 \
  --model sonnet \
  --add-dir /path/to/other/workspace/dirs \
  2>&1 | tee /tmp/claude-task.log
```

### Key differences from codex exec
- `--bare` skips OAuth and does NOT work with claude.ai auth. Omit it.
- `--dangerously-skip-permissions` auto-approves all tool use (like `--yolo`).
- `--max-turns N` caps agentic loops (prevents runaway).
- `--model sonnet` is cost-efficient for well-specified tasks. Use `opus` for complex reasoning.
- `--add-dir` grants access to additional directories (useful for monorepos).
- Claude Code can run `cargo test` and `cargo check` in the main workspace (not worktrees).
- Task specs can be piped via stdin or passed as the `-p` argument.

### Parallel Claude Code agents (same pattern as codex)
```bash
# Write specs to files
cat /tmp/claude-task-1.md | claude -p "$(cat /tmp/claude-task-1.md)" \
  --dangerously-skip-permissions --max-turns 30 --model sonnet &
cat /tmp/claude-task-2.md | claude -p "$(cat /tmp/claude-task-2.md)" \
  --dangerously-skip-permissions --max-turns 30 --model sonnet &
```

Use `terminal(background=true, notify_on_complete=true)` for each agent.

### Time-awareness for delegation (USER CORRECTION 2026-06-23)

> "you need to be more time aware before you delegate to sub agents. time aware for how long the task should take."

**Lesson:** Before dispatching any subagent (codex or claude), estimate how long the task should take:
- Single file, < 200 lines: 2-3 minutes. Set `--max-turns 5`.
- Multi-file, surgical changes: 5-10 minutes. Set `--max-turns 15`.
- Complex feature, multiple modules: 10-20 minutes. Set `--max-turns 30`.

If the agent hasn't produced output within 2x the estimated time, it's stuck. Kill it and do the work in the controller. The controller's `patch` + `cargo check` cycle is often faster than re-dispatching for surgical Rust changes.

**Do NOT give open-ended "implement X" goals without time bounds.** Always set `--max-turns` to cap the agent's effort. An agent with 50 turns on a 2-minute task is wasting tokens and context.

### Which agent to choose
| Situation | Agent | Why |
|-----------|-------|-----|
| Well-specified Rust task, fresh session | Claude Code (`claude -p`) | No usage limit, can run cargo |
| Quick mechanical refactor | Codex (`codex exec`) | Faster startup, less overhead |
| Codex usage limit hit | Claude Code | Fallback -- always works |
| Need to read many files first | Codex with `gpt-5.5` | Larger context for research |
| Parallel agents touching different files | Either | Both work, same pattern |
| Parallel agents touching same files | Neither | Serialize in controller |

### Claude Code 50-turn limit on multi-task specs (2026-06-24)

When a task spec contains 9+ sub-tasks that each require reading multiple files, Claude Code hits the `--max-turns 50` limit before completing all tasks. Symptoms: `Error: Reached max turns (50)` in the log, partial work committed.

**Fix -- split into focused single-task specs:**
- 1 task per `claude -p` invocation, `--max-turns 10-15` each
- Launch in parallel when tasks touch different files
- Serialize when tasks share files (e.g. both modify search.rs)
- The controller verifies compilation after each batch

**For interdependent Rust changes** (struct field additions that propagate across many files), direct `patch` in the controller is more reliable than agent delegation. The agent will spend turns reading files, and the struct field pitfall (documented above) means automated insertions corrupt struct definitions.

**Session 2026-06-24 results:** Claude Code (sonnet, 50 turns) completed ~2-3 medium tasks before hitting the limit. A `delegate_task` subagent (600s timeout) completed ~5 tasks but made compilation errors requiring controller fixes (mut binding, PreferExact guard, receipt_metadata clone). The pattern: agents write ~80% of the code, controller fixes the last 20% of compile errors.

### Delegation timeout pattern (subagent + execute_code)

When `delegate_task` times out (600s default), check what the subagent wrote before re-implementing:
```bash
git diff --stat HEAD -- src/  # scope to the specific crate
cargo check --all-features 2>&1 | grep "error\[" | head -5
```
The subagent may have completed 80% of the work before timing out. Fix the remaining compile errors directly rather than re-dispatching.

## References

- `references/aidens-integration-patterns.md` — Facade wiring pattern for adapter frameworks, codex size limits on large files, graceful test degradation, Non-Copy field fix, struct construction pitfalls. From the AiDENs full integration session (2026-06-20).
- `references/rust-monolith-split-technique.md` — Mechanical sed-based technique for splitting large Rust lib.rs files into submodules. Covers the `pub(crate) use` pattern, cross-module visibility, module name collision avoidance, and when codex can vs can't help. Tested on 3,396-line and 4,996-line files.
- `scripts/run_codex_phase.sh` — Reusable bash launcher for multi-phase codex orchestration with JSONL capture and exit code recording.

## Background Process Notification Race

`terminal(background=true, notify_on_complete=true)` on a `codex exec` is fire-and-forget. The system delivers the completion notification when the process exits. If you commit work in the controller before the notification fires, the late completion lands on top of a moved HEAD — and the diff may re-modify files you just committed (because the agent's working tree was at the older HEAD).

**Mitigation:**
- `git status --short` after every background-process notification
- Re-run the verification gauntlet before declaring a batch done
- Tag the working tree (`git tag pre-late-arrival`) if you're about to commit and a background agent is still running — easy rollback if the late diff breaks things

The "shipped N / deferred M" doc is correct as of WHEN YOU WROTE IT. Late arrivals are how it stops being correct. Re-check before final handoff.
