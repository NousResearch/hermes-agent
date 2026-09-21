---
name: simplify-swarm
description: Use when code has been written or modified and needs multi-pass simplification. Dispatches 3 parallel sub-agents (Hygiene, Clarity, Correctness) then consolidates findings and applies in SAFE→CAREFUL→RISKY tier order.
version: 1.1.0
author: KENSEI (Sahil 1000x)
license: MIT
metadata:
  hermes:
    tags: [simplify, refactor, code-quality, multi-agent, parallel, cleanup, dead-code, correctness]
    related_skills: [requesting-code-review, subagent-driven-development, octacon-grill, octacon-arch-review, simplify-code]
---

# Simplify Swarm

## Overview

Multi-agent code simplification. Three specialist sub-agents run in parallel over the same diff, each focused on a distinct concern. A consolidator merges their findings, de-duplicates, and applies changes in risk-tier order: SAFE → CAREFUL → RISKY.

**Why three agents instead of one:** No single agent can reliably spot dead code, naming issues, AND N+1 queries in one pass. Specialization + parallelization catches more with less context-burn per agent.

**Core principle:** Fresh eyes per concern. Analyze first (read-only), consolidate, then apply. Never edit during the analysis phase.

## When to Use

- After implementing a feature or bug fix ("simplify this", "clean up this code")
- Pre-commit hygiene pass before `git commit`
- When code works but feels heavy, repetitive, or fragile
- Post-AI-generation cleanup (removes slop patterns)
- User says "simplify", "clean up", "deslop", "tighten up", or "/simplify"

**Focus modifiers** — narrow the swarm to one or two agents:
- "simplify focus on safety" → Hygiene only
- "simplify focus on readability" → Clarity only
- "simplify focus on performance" → Correctness only
- "simplify focus on hygiene,clarity" → Hygiene + Clarity (skip Correctness)

**Dry run:** "simplify but don't change anything" / "just report" → run all three, present findings, apply nothing.

**Skip for:** config-only changes, docs-only changes, code already processed by the swarm in current session, code with no git history (can't verify what's modified).

## Architecture

```
┌──────────────────────────────────────────┐
│         Simplify Swarm Orchestrator       │
│         (this skill)                      │
│                                            │
│  1. Scope detection (git diff)            │
│  2. Parallel dispatch (3 agents)          │
│  3. Consolidation (merge + dedupe)        │
│  4. Tiered application (SAFE→CAREFUL→RISKY)│
└──────────┬───────────┬───────────┬───────┘
           ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │ Hygiene  │ │ Clarity  │ │ Correctness  │
    │ (SAFE)   │ │(CAREFUL) │ │ (RISKY)       │
    └────┬─────┘ └────┬─────┘ └──────┬───────┘
         │            │              │
         └────────────┴──────────────┘
                      │
                      ▼
              ┌─────────────┐
              │ Consolidator│
              │ (merge +    │
              │  apply in   │
              │  tier order)│
              └─────────────┘
```

## Step 1 — Scope Detection

Identify what code to analyze. Default: recently modified code in current session.

```bash
# Primary: staged changes
git diff --cached --name-only 2>/dev/null

# Fallback: unstaged changes
git diff --name-only 2>/dev/null

# Fallback: recently committed (last 3 commits)
git diff --name-only HEAD~3 HEAD 2>/dev/null
```

Filter to source files only. Exclude:
- `*.json`, `*.yaml`, `*.yml`, `*.toml` (config)
- `*.md`, `*.txt` (docs)
- `*.lock`, `package-lock.json` (lockfiles)
- `*.test.*`, `*.spec.*` (tests — simplify separately if requested)
- `*.min.*`, `*.generated.*` (generated)

If the filtered list is empty: report "No source files to simplify" and exit.

If user specified explicit files or directories, use those instead.

## Step 2 — Parallel Dispatch

Dispatch all three agents simultaneously using `delegate_task` with the `tasks` array. Each agent receives the file list, git diff for those files, and its specialized prompt. Agents are READ-ONLY — they analyze and report, they do NOT edit.

```python
delegate_task(
    tasks=[
        {
            "goal": "Hygiene agent: Analyze code for dead code, AI slop (any casts, redundant guards, comments that restate code), redundant abstractions (pass-through wrappers, single-use helpers), stale state (feature flags always true/false, unreachable branches), and utility discovery opportunities.",
            "context": """<file_list>{newline-separated file paths}</file_list>
<git_diff>{git diff output for these files}</git_diff>
<instructions>{Load from references/hygiene-agent.md}</instructions>""",
            "toolsets": ["terminal", "file"],
            "role": "leaf"
        },
        {
            "goal": "Clarity agent: Analyze code for duplication, naming issues, structural complexity, and consistency problems.",
            "context": """<file_list>{newline-separated file paths}</file_list>
<git_diff>{git diff output for these files}</git_diff>
<instructions>{Load from references/clarity-agent.md}</instructions>""",
            "toolsets": ["terminal", "file"],
            "role": "leaf"
        },
        {
            "goal": "Correctness agent: Analyze code for N+1 queries, memory leaks, concurrency issues, leaky abstractions, silent failures, and performance problems.",
            "context": """<file_list>{newline-separated file paths}</file_list>
<git_diff>{git diff output for these files}</git_diff>
<instructions>{Load from references/correctness-agent.md}</instructions>""",
            "toolsets": ["terminal", "file"],
            "role": "leaf"
        }
    ]
)
```

**Before dispatching:** Read the three reference files so their content can be injected into each agent's context. Use `skill_view(name='simplify-swarm', file_path='references/hygiene-agent.md')` etc.

**Large diffs:** If the git diff exceeds 12,000 characters, split by file and dispatch per-file agent batches. Each agent still analyzes all files, but the diff in context is scoped to one file at a time to avoid context overflow.

## Step 3 — Consolidation

When all three agents return, merge their findings into a single change plan. Each agent returns structured JSON (see reference files for output format).

### Merge Rules

1. **De-duplicate:** If two agents flag the same line for the same reason, keep one.
2. **Resolve conflicts:** If agents disagree (e.g., Clarity says rename, Correctness says the name is part of a public contract), the more conservative agent wins. Public contracts are never renamed.
3. **Assign risk tier:** Each finding inherits its agent's risk tier unless the agent explicitly tagged it higher/lower.
4. **Sort by tier then by file:** SAFE first, then CAREFUL, then RISKY. Within each tier, group by file to minimize edit churn.
5. **Drop no-ops:** If a finding would require a change that doesn't improve anything concrete, drop it.

### Consolidation Output

Produce a single structured plan:

```
## Simplify Swarm — Consolidation Report

### Scope: {N} files analyzed

### Hygiene Findings (SAFE — {count} items)
- [SAFE] path/to/file.ts:42 — Unused import `lodash`
- [SAFE] path/to/file.ts:78-82 — Unreachable branch after early return
- [SAFE] path/to/file.ts:105 — Commented-out code block
- [SAFE] path/to/file.ts:130 — Pass-through wrapper, inline the call
...

### Clarity Findings (CAREFUL — {count} items)
- [CAREFUL] path/to/file.ts:15 — Nested ternary → if/else
- [CAREFUL] path/to/file.ts:45 — Generic name `data` → `userProfile`
- [CAREFUL] path/to/file.ts:60-95 — Function too long (52 lines), extract helper
...

### Correctness Findings (RISKY — {count} items)
- [RISKY] path/to/file.ts:120 — N+1: query inside loop, missing .include()
- [SAFE] path/to/file.ts:200 — Unused event listener (Correctness-tagged SAFE)
- [CAREFUL] path/to/file.ts:250 — Empty catch block
...
```

Report this to the user before applying. For autonomous mode (config `auto_apply: true`), skip the report display and proceed directly to Step 4.

## Step 4 — Tiered Application

Apply changes in risk-tier order. After each tier: run tests. If tests fail, revert that tier's changes and escalate.

### Tier 1: SAFE (auto-apply)

Apply all SAFE findings directly. These are changes proven not to affect behavior:
- Remove unused imports, variables, exports (verified by grep)
- Delete unreachable branches (verified by control flow analysis)
- Remove commented-out code blocks
- Inline pass-through wrappers
- Remove redundant type assertions
- Delete stale feature flags (check: is the flag always true or always false? If always false, flag the UNREACHABLE BRANCHES, not the import itself — the import IS used in the dead branches)
- Remove AI slop comments (comments that restate WHAT the code does. KEEP comments that explain WHY — intent, workaround rationale, domain context. Check 2 lines of surrounding code: if the comment adds no information beyond the code itself, flag it for removal)

```bash
# After applying all SAFE changes
{project_test_command} 2>&1 | tail -5
```

If tests fail: `git checkout -- .` (revert all), report which change caused the failure, escalate to user.

### Tier 2: CAREFUL (apply with verification)

Apply CAREFUL findings one file at a time, running tests after each file:

```
for each file with CAREFUL findings:
    1. Apply all CAREFUL changes for that file
    2. Run tests
    3. If pass → commit with message "simplify(clarity): {description}"
    4. If fail → revert file, skip that file's changes, continue
```

CAREFUL changes include:
- Rename variables (check it's not an exported symbol first)
- Flatten nested ternaries to if/else
- Extract repeated logic to helper
- Improve function decomposition
- Consolidate duplicate blocks
- Replace magic numbers with named constants

**Critical:** Before renaming anything, grep the full codebase to confirm it's not an export or public API. If it is, escalate to RISKY tier.

### Tier 3: RISKY (flag for review)

Do NOT auto-apply RISKY findings. Present them to the user with:
- The specific finding and file location
- The risk (what could break)
- A recommended fix
- Whether tests exist that cover this code path

User decides: apply, modify recommendation, or ignore.

RISKY findings include:
- N+1 query restructuring
- Memory leak fixes (changing resource lifecycle)
- Concurrency fixes (changing execution order)
- Public API renames
- Error handling changes (adding/removing try-catch)
- Leaky abstraction repairs

## Step 5 — Final Verification

After all applied tiers:

```bash
# Full test suite
{project_test_command}

# Lint
{project_lint_command}

# Type check (if applicable)
{project_typecheck_command}

# Build (if applicable)
{project_build_command}
```

All must pass. If any fail, revert the last tier and report.

## Configuration

Optional `.kensei/simplify.yaml` in project root:

```yaml
simplify:
  enabled: true
  auto_apply: false         # Skip approval for SAFE+CAREFUL tiers
  scope: modified            # modified | staged | all | <glob>
  skip_patterns:             # Files to skip
    - "*.test.*"
    - "*.spec.*"
    - "*.generated.*"
  max_file_lines: 800        # Skip files larger than this
  languages:                 # Language-specific tooling
    typescript:
      dead_code_tools: [knip, depcheck, ts-prune]
      lint: "npx eslint"
      typecheck: "npx tsc --noEmit"
    python:
      dead_code_tools: [vulture, autoflake]
      lint: "ruff check"
      typecheck: "mypy"
  correctness:               # Correctness agent sensitivity
    n_plus_one: true
    memory_leaks: true
    concurrency: true
    leaky_abstractions: true
    silent_failures: true
```

## Language-Specific Detectors

The Correctness agent uses language-specific patterns. Key detectors:

### TypeScript / JavaScript

| Concern | Detection Pattern |
|---------|------------------|
| N+1 queries | `.find()` / `.map()` inside `for`/`.forEach`, missing `.include()` / `.preload()` / `.populate()` |
| Memory leaks | `addEventListener` without `removeEventListener`, `setInterval` without `clearInterval`, closures capturing large scope, unmounted React state updates |
| Concurrency | `Promise.all` on dynamic-length arrays, `.forEach(async` antipattern, shared mutable state across async boundaries |
| Silent failures | `.catch()` with empty body, `.catch(() => {})`, `try {} catch {}` with no handling |

### Python

| Concern | Detection Pattern |
|---------|------------------|
| N+1 queries | ORM `.get()` / `.filter()` inside `for` loop, missing `.select_related()` / `.prefetch_related()` |
| Memory leaks | Circular references with `__del__`, unclosed file handles, growing global lists/dicts, signal handlers without disconnect |
| Concurrency | `asyncio.gather` with mutable shared state, missing `asyncio.Lock`, bare `threading.Thread` without join |
| Silent failures | `except: pass`, bare `except:`, `except Exception:` that swallows, logging.error without re-raise |

### Go

| Concern | Detection Pattern |
|---------|------------------|
| N+1 queries | `db.Query()` / `db.QueryRow()` inside `for` loop |
| Memory leaks | Goroutine without cancellation, unbuffered channel with no reader, `resp.Body` not closed |
| Concurrency | Mutex not unlocked (missing `defer mu.Unlock()`), channel deadlock potential, data races on shared slices |
| Silent failures | `_ = err`, `_ = result`, error not checked before use |

## Integration with Other Skills

### With requesting-code-review

Run simplify-swarm BEFORE requesting-code-review. Simplification reduces the diff surface the reviewer needs to analyze. The reviewer sees cleaner code.

### With subagent-driven-development

Add simplify-swarm as a post-task step in the per-task workflow:
```
Implementer → Spec Reviewer → Quality Reviewer → Simplify Swarm → Mark Complete
```

### With octacon-grill

When octacon-grill challenges a coding plan, run simplify-swarm on the implemented code to validate that the plan didn't produce unnecessarily complex output.

### With octacon-arch-review

Correctness agent's leaky-abstraction and architectural findings feed into octacon-arch-review. If Correctness flags 3+ leaky-abstraction issues, trigger an architecture review.

## Common Pitfalls

1. **Editing during analysis phase.** Agents must be read-only. They return JSON findings — the orchestrator applies changes. An agent that edits files directly can conflict with other agents.

2. **Over-trusting dead code tools.** `knip` and `ts-prune` flag exports that ARE used dynamically (string-based imports, reflection). Always grep for the symbol name before removing.

3. **Renaming without checking public contracts.** Export names, API route paths, DB column names, and config keys are contracts. Even if the name is bad, renaming breaks consumers. Flag as RISKY, don't auto-rename.

4. **Simplifying code you don't understand.** Chesterton's Fence: if you don't know why code exists, don't touch it. The agents must run `git blame` on suspicious patterns before flagging them for removal.

5. **Batching too many changes.** Apply one file's worth of changes, then test. If you batch 10 files and something breaks, you don't know which change caused it.

6. **Removing "unnecessary" error handling.** That empty catch block might be intentional — the error is expected and benign. Flag it, don't remove it. Let the user decide.

7. **Context overflow on large diffs.** If the git diff is >12k chars, don't stuff it all into agent context. Split by file. Each agent gets one file's diff at a time.

8. **Skipping the consolidation step.** If you apply Hygiene findings, then Clarity findings, then Correctness findings directly, you might apply contradictory changes. Always consolidate first.

## Verification Checklist

- [ ] Scope correctly identified (git diff, not full repo)
- [ ] All three agents dispatched in parallel (not sequentially)
- [ ] Each agent returned structured JSON (not free text)
- [ ] Consolidation merged + de-duplicated correctly
- [ ] Changes applied in SAFE → CAREFUL → RISKY order
- [ ] Tests pass after each tier
- [ ] No public API contracts broken
- [ ] RISKY findings presented to user, not auto-applied
- [ ] Final lint/typecheck/build all pass
- [ ] Git diff after application is clean and reviewable
