---
name: hermaguard
description: "Adversarial 3-agent bug-hunting code review. Parallel subagents attack code from different angles — edge cases, adversarial attack surfaces, blast radius. Read-only — reports findings, does not fix. Opt-in. Complement to simplify-code."
version: 1.0.0
author: KENSEI (Sahil Saghir)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, adversarial, bug-hunt, edge-cases, blast-radius, security, subagent, parallel]
    related_skills: [simplify-code, requesting-code-review, security-review]
---

# Hermaguard — Adversarial Bug-Hunting Review

**What:** Three parallel subagents review your recent code changes from
different adversarial angles, then a consolidator merges and triages the
findings into a structured report. No fixes are applied — this skill finds
problems, it doesn't fix them.

**Complement to `simplify-code`:** Where the simplifier cleans code without
changing behaviour, Hermaguard finds behaviour that's already broken. The
chained workflow is:

```
Implement → Verify (tests pass) → Simplify → Hermaguard → Fix → Simplify → Commit
```

**Based on:** Trail of Bits `differential-review` (8-phase security review),
BMAD `edge-case-hunter` (exhaustive path tracer), BMAD `adversarial-general`
(cynical reviewer persona), BMAD `bmad-code-review` (3-layer parallel review),
dementev-dev `adversarial-review` (Claude↔Codex iterative loop),
Anthropic `claude-code-security-review` (CI/CD security action), and the
Reddit adversarial prompt hack (community-validated adversarial stance).
Full research at `references/cross-implementation-analysis.md`.

## When to Use

Trigger this skill when the user says any of:

- "hermaguard" / "hermaguard this" / "guard this"
- "adversarial review" / "bug hunt this" / "break this"
- "check for edge cases" / "what could go wrong"
- `/hermaguard` (slash command)

**Do NOT invoke if:**
- Tests are failing (fix tests first)
- No code was modified (config/docs-only changes)
- The change is a one-liner with no control flow (trivial assignment, typo fix)
- Already guarded this diff in the last 10 minutes (dedup)

## Architecture: 3 Agents + Consolidator

```
                    ┌──────────────────────────────┐
                    │   Hermaguard                  │
                    │   (Orchestrator)              │
                    └──────────┬───────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ 1. Edge Case │    │ 2. Adver-    │    │ 3. Blast     │
   │    Hunter    │    │    sarial    │    │    Radius +   │
   │ (diff only)  │    │  Reviewer    │    │  Integration  │
   │              │    │ (full files) │    │ (full files + │
   │              │    │              │    │  call graph)  │
   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                      ┌──────────────┐
                      │ Consolidator │
                      │ Merge +      │
                      │ Triage +     │
                      │ Report       │
                      └──────────────┘
```

### Why 3 agents (not 7 like the simplifier swarm)

The simplifier needs many narrow agents because it's mutating code — each
change type has its own risk profile and failure mode. Bug hunting is pure
analysis. More agents = more duplicate findings, not more coverage. Three
distinct perspectives (exhaustive paths, adversarial attack, blast radius)
give ~95% coverage without redundant cross-talk.

### Why read-only (no fix-it loop)

Fixes are separate tasks handled by the developer. This separation means
each step is independently auditable and replayable. Hermaguard finds
problems; you fix them.

---

## The Process

### Phase 1 — Identify the changes

Capture the diff to review:

```bash
git diff                        # Default: uncommitted working-tree changes
git diff HEAD                   # If empty: include staged changes
git diff --staged               # If user asked for "staged changes"
git diff main...HEAD            # If user asked for "this branch"
```

Skip: test files (`*.test.*`, `*.spec.*`, `tests/`, `__tests__/`), config
files (`*.json`, `*.yaml`, `*.toml`, `*.md`), generated code, vendored deps.

If no code changes found, say so and stop.

### Phase 2 — Launch three reviewers in parallel

Use `delegate_task` **batch mode** — pass all three tasks in one `tasks`
array so they run concurrently. Each reviewer has different scope and stance:

#### Agent 1: Edge Case Hunter

**Scope:** Diff only (`git diff` output — not full files)

**Stance:** You are a pure path tracer. Method-driven, not attitude-driven.
Never comment on whether code is good or bad; only list missing handling.

**Method:** Walk every branching path and boundary condition reachable from
the changed lines. Exhaustively enumerate — do not hunt by intuition.

**Edge classes to check:**
- **Control flow:** missing else/default, unguarded switch fall-through,
  early returns that skip cleanup
- **Null/empty:** null, undefined, empty string, empty array, zero, NaN
- **Boundary values:** off-by-one in loops/indices, overflow/underflow,
  min/max thresholds, empty collections
- **Type coercion:** implicit conversions, truthy/falsy gotchas, `==` vs
  `===`, `parseInt` without radix
- **State transitions:** loading→error, active→expired, before→after auth,
  first-use vs subsequent
- **Async:** promise rejection unhandled, race between async ops, partial
  success states
- **Concurrency:** shared mutable state, non-atomic read-modify-write

**Output contract:** Return ONLY a JSON array:
```json
[{
  "location": "file:line-range",
  "trigger_condition": "one-line description (max 15 words)",
  "guard_snippet": "minimal code sketch that closes the gap",
  "potential_consequence": "what could actually go wrong (max 15 words)"
}]
```
An empty array `[]` is valid when no unhandled paths are found.

#### Agent 2: Adversarial Reviewer

**Scope:** Full file contents of changed files (`read_file` each file).
Give this agent `terminal`, `file`, and `search` toolsets.

**Stance:** "Your job is to break confidence in the change, not to validate
it." Default to skepticism. Assume the change can fail in subtle, high-cost,
or user-visible ways until the evidence says otherwise. Do not give credit for
good intent, partial fixes, or likely follow-up work. If something only works
on the happy path, treat that as a real weakness.

**Persona:** You are a cynical, jaded reviewer with zero patience for sloppy
work. Use precise, professional tone — no profanity, no personal attacks.
Be relentless.

**Attack surfaces (check each — skip if not applicable):**
- **Auth & Permissions:** bypasses, privilege escalation, missing checks
- **Data Integrity:** loss, corruption, partial writes, missing transactions
- **Race Conditions:** TOCTOU, concurrent access without locks, deadlocks
- **Rollback Safety:** can this change be safely reverted?
- **Schema Drift:** migrations present, backward compatibility, data format
- **Error Handling:** swallowed errors, missing retries, cascading failures
- **Observability:** will operators know when this breaks?
- **Input Validation:** injection vectors, unsanitised user input

**Finding bar — every finding MUST answer 4 questions:**
1. What can go wrong? (concrete scenario, not hypothetical)
2. Why is this code vulnerable? (cite specific file and lines)
3. Impact — what breaks and how badly? (data loss > outage > degraded UX)
4. Recommendation — specific fix with code reference

**Do NOT comment on:** code style, formatting, naming conventions, "nice to
have" improvements unrelated to correctness, speculative issues without
concrete trigger, performance micro-optimisations.

**Output format:**
```markdown
## Adversarial Review Findings

### [SEVERITY] Finding Title
**File:** path:line-range
**What can go wrong:** ...
**Why vulnerable:** ...
**Impact:** ...
**Recommendation:** ...
```

#### Agent 3: Blast Radius + Integration

**Scope:** Full file contents + call graph analysis. Give this agent
`terminal`, `file`, and `search` toolsets — it needs `grep`/`rg` to trace
callers and callees.

**Stance:** Strategic — zoom out. A change that's locally correct can still
break the system. Your job: map the wider impact.

**Method:**
1. For each changed function/class/export, find ALL callers:
   ```bash
   rg -n "functionName|ClassName|exportName" --type-add \
     'code:*.{ts,tsx,js,py,go,rs,java}' --type code
   ```
2. For each caller, assess: does the change break the caller's assumptions?
3. Check configuration that depends on this behaviour (env vars, feature
   flags, API contracts, route paths)
4. Assess migration/revert safety

**Specific checks:**
- **Caller impact:** call signature compatibility, return value assumptions
- **Downstream effects:** callee contract changes
- **Configuration coupling:** config keys, env vars, feature flags affected
- **Database/Migration:** schema changes, backward-compatible writes
- **API contracts:** route paths, request/response shapes, error codes
- **Observability:** metrics/alerts on this code path, false alarms

**Output format:**
```markdown
## Blast Radius Analysis

### Callers of changed code
| Caller (file:line) | Changed symbol | Risk | Notes |

### Downstream dependencies
| Callee | Change impact | Risk |

### Configuration & Contracts
- **Config keys affected:** ...

### Revert Safety
- **Safe to revert?** ...
- **Revert procedure:** ...
```

### Phase 3 — Consolidate and Report

After all 3 subagents return:

1. **De-duplicate:** Same bug found by multiple agents → keep most detailed
   version, note cross-agent agreement
2. **Triage by risk tier:**
   - **CRITICAL:** Data loss, auth bypass, security exploit
   - **HIGH:** Production outage risk, silent failure, race condition
   - **MEDIUM:** Edge case with degraded UX, missing error handling
   - **LOW:** Minor edge case unlikely to trigger
3. **Cross-reference:** If Agent 3 found a caller that Agent 2 flagged as
   vulnerable, escalate severity
4. **Write report to disk** (mandatory — don't just summarise in chat):
   `/tmp/hermaguard/hermaguard-{timestamp}-{short-hash}.md`

**Report structure:**

```markdown
# Hermaguard Report
**Date:** {DD/MM/YY HH:MM}
**Scope:** {N} files changed, {M} files in blast radius

## Summary
**Total findings:** {N}
**CRITICAL:** {N} | **HIGH:** {N} | **MEDIUM:** {N} | **LOW:** {N}

{One-paragraph narrative}

## CRITICAL Findings
{Each with exploit PoC, impact, recommendation}

## HIGH Findings
...

## MEDIUM Findings
...

## LOW Findings
{Compact table}

## Blast Radius Map
{Files affected, callers, downstream}

## Verdict
**Overall risk:** {LOW / MEDIUM / HIGH / CRITICAL}

**Recommended action:**
- [ ] Fix CRITICAL findings before merge
- [ ] Fix HIGH findings before deploy
- [ ] Address MEDIUM findings next sprint
- [ ] LOW findings — backlog or dismiss

**No fixes were applied by this review.**
```

**In-chat summary** (after report written):
```
Hermaguard complete.

Scope: 4 files changed, 12 files in blast radius
Findings: 7 total
  CRITICAL: 0 | HIGH: 2 | MEDIUM: 3 | LOW: 2

Report: /tmp/hermaguard/hermaguard-20260608-1430-a1b2c3d.md

Top finding: [HIGH] Race condition in payment processing (Agent 2)
  → src/payments/handler.ts:42 — concurrent calls can double-charge

No fixes applied. Findings are for you to fix.
```

---

## Slash Command

**`/hermaguard`** — trigger this skill on the current diff.

| Flag | Effect |
|------|--------|
| `/hermaguard` | Auto-detect scope (unstaged → staged → branch diff) |
| `/hermaguard --focus edge` | Edge Case Hunter only (fastest, no file reads) |
| `/hermaguard --file path/to/file.ts` | Scope to a specific file |
| `/hermaguard --since HEAD~3` | Scope to commits since a ref |

---

## Pitfalls

- **Don't guard code you don't understand.** Mark findings as lower confidence
  if the call graph is unclear.
- **Don't fabricate findings.** An empty agent report IS valid. False positives
  erode trust faster than missed bugs.
- **Don't let the adversarial agent go soft.** If it returns "no findings,"
  ask it to re-examine — adversarial reviewers should ALWAYS find something
  worth noting, even if LOW severity.
- **Don't mix guarding with simplification.** Guard after simplify, never
  during.
- **Report file is mandatory.** Don't just summarise in chat — write the full
  report to disk for traceability.
