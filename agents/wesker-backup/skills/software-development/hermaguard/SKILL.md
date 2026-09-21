---
name: hermaguard
description: "Adversarial bug-hunting code review with 3 parallel specialist subagents (Edge Case Hunter, Adversarial Reviewer, Blast Radius + Integration) and a consolidator. Finds what's broken, not what looks nice. Read-only — no fixes applied. Reports only."
version: 1.1.0
category: software-development
---

# Hermaguard

## What This Is

A read-only adversarial review skill that hunts bugs, edge cases, and integration risks in code changes. 3 parallel subagents each attack the code from a different angle, then a consolidator merges and triages the findings into a structured report.

**This skill does NOT apply fixes.** It finds problems. Octacon or the feature pipeline picks up findings as separate tasks.

Based on synthesis of 8 implementations: Trail of Bits `differential-review` (8-phase security review), BMAD `edge-case-hunter` (exhaustive path tracer), BMAD `adversarial-general` (cynical reviewer persona), BMAD `bmad-code-review` (3-layer parallel review), dementev-dev `adversarial-review` (Claude↔Codex iterative loop), Anthropic `claude-code-security-review` (CI/CD security action), Anthropic Claude Code Code Review Plugin (4 parallel agents), and the Reddit adversarial prompt hack (community-validated adversarial stance).

Full research at `references/cross-implementation-analysis.md`. SkillOpt readiness evaluation at `references/skillopt-evaluation.md` — recommended before production gating.

## When to Invoke

**Auto-invoke after:**
- Any implementation task completes and passes `code-simplifier`
- Pre-commit when `--hermaguard` flag is passed
- New feature or refactor that touches >3 files (blast radius concern)

**Manual invoke when Sahil says:**
- "hermaguard this" / "guard this" / "adversarial review"
- "bug hunt this" / "break this" / "what could go wrong"
- "check for edge cases" / "security review this change"

**Do NOT invoke if:**
- Tests are failing (fix tests first — you can't hunt bugs in broken code)
- No code was modified (config/docs-only changes)
- The change is a one-liner with no control flow (trivial assignment, typo fix)
- Already guarded this diff in the last 10 minutes (dedup)

## Slash Command

**`/hermaguard`** — triggers this skill on the current diff. On Hermes Agent, the skill loader matches keywords ("hermaguard this", "guard this", "adversarial review") — the slash syntax here is for Claude Code compatibility. On Hermes, any of the trigger phrases in "When to Invoke" will load this skill.

Optional flags:

| Flag | Effect |
|------|--------|
| `/hermaguard` | Auto-detect scope (unstaged → staged → branch diff) |
| `/hermaguard --full` | Run all 3 agents even if <3 files changed |
| `/hermaguard --quick` | Edge Case Hunter only (fastest) |
| `/hermaguard --file path/to/file.ts` | Scope to a specific file |
| `/hermaguard --since HEAD~3` | Scope to commits since a ref |

The `/hermaguard` command is **opt-in only** — no auto-trigger on commits. Sahil runs it explicitly when he wants adversarial review.

## Relationship to Code Simplifier

| Aspect | Code Simplifier | Hermaguard |
|--------|----------------|------------|
| Goal | Cleaner code, same behaviour | Find behaviour problems |
| Stance | Constructive refinement | Adversarial, destructive |
| Scope | Recently modified code | Modified + blast radius |
| Applies changes | Yes (auto-apply or diffs) | No (read-only report) |
| Output | Modified files + summary | Risk-triaged findings report |

**Chained workflow:**
```
Implement → Verify (tests pass) → Simplify → Hermaguard → Fix (Octacon) → Simplify → Commit
```

## Architecture: 3 Agents + Consolidator

```
                    ┌──────────────────────────────┐
                    │   Hermaguard                  │
                    │   (Orchestrator — this skill) │
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

### Rationale for 3 agents (not 7 like the simplifier swarm)

The simplifier needs many narrow agents because it's mutating code — each change type has its own risk profile and failure mode. Bug hunting is different: it's a pure analysis task. More agents = more duplicate findings, not more coverage. Three distinct perspectives (exhaustive paths, adversarial attack, blast radius) give ~95% coverage without redundant cross-talk.

---

## Execution Flow

### Phase 0: Scope Detection

```bash
# Detect changes — try unstaged first, then staged, then branch diff
git diff --name-only -- '*.ts' '*.tsx' '*.js' '*.py' '*.go' '*.rs' '*.java' 2>/dev/null
# If empty, try staged:
git diff --cached --name-only -- '*.ts' '*.tsx' '*.js' '*.py' '*.go' '*.rs' '*.java'
# If still empty, try branch diff:
git diff --name-only $(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's|refs/remotes/origin/||' || echo main)...HEAD -- '*.ts' '*.tsx' '*.js' '*.py' '*.go' '*.rs' '*.java'
```

Skip: test files (`*.test.*`, `*.spec.*`, `tests/`, `__tests__/`), config files (`*.json`, `*.yaml`, `*.toml`, `*.md`), generated code, vendored deps.

If no files found: "No code changes to guard." Stop.

### Phase 1: Parallel Dispatch

Spawn all 3 subagents simultaneously as `delegate_task` calls:

#### Agent 1: Edge Case Hunter

```
Scope: Diff only (git diff output)
Context: Only the actual diff hunks — not full files
```

**Stance:** You are a pure path tracer. Method-driven, not attitude-driven. Never comment on whether code is good or bad; only list missing handling.

**Method:** Walk every branching path and boundary condition reachable from the changed lines. Exhaustively enumerate — do not hunt by intuition.

**Edge classes to check:**
- **Control flow**: missing else/default, unguarded switch fall-through, early returns that skip cleanup
- **Null/empty**: null, undefined, empty string, empty array, zero, NaN
- **Boundary values**: off-by-one in loops/indices, overflow/underflow, min/max thresholds, empty collections
- **Type coercion**: implicit conversions, truthy/falsy gotchas, `==` vs `===`, `parseInt` without radix
- **State transitions**: loading→error, active→expired, before→after auth, first-use vs subsequent
- **Async**: promise rejection unhandled, race between async operations, partial success states
- **Concurrency**: shared mutable state, non-atomic read-modify-write
- **Degenerate handlers**: empty catch/then blocks, no-op error paths, fallthrough switch cases with identical bodies, placeholder return values that mask failures

**Output contract:** Return ONLY a JSON array. Each object has exactly these 4 fields:
```json
[{
  "location": "file:line-range",
  "trigger_condition": "one-line description (max 15 words)",
  "guard_snippet": "minimal code sketch that closes the gap",
  "potential_consequence": "what could actually go wrong (max 15 words)"
}]
```
An empty array `[]` is valid when no unhandled paths are found.

**Halt conditions:**
- Diff is empty or undecodable → return `[{"location":"N/A","trigger_condition":"Input empty","guard_snippet":"Provide valid diff","potential_consequence":"Review skipped"}]`
- Zero findings → return `[]` — this IS valid; do not fabricate findings

---

#### Agent 2: Adversarial Reviewer

```
Scope: Full file contents of changed files (read_file on each changed file)
Context: Entire file — you need to understand what the change sits within
```

**Stance:** "Your job is to break confidence in the change, not to validate it." Default to skepticism. Assume the change can fail in subtle, high-cost, or user-visible ways until the evidence says otherwise. Do not give credit for good intent, partial fixes, or likely follow-up work. If something only works on the happy path, treat that as a real weakness.

**Persona:** You are a cynical, jaded reviewer with zero patience for sloppy work. Use a precise, professional tone — no profanity, no personal attacks. But be relentless.

**Attack surfaces (check each — skip if not applicable):**
- **Auth & Permissions:** bypasses, privilege escalation, missing checks on new endpoints
- **Data Integrity:** loss, corruption, partial writes, constraint violations, missing transactions
- **Race Conditions:** TOCTOU, concurrent access without locks, deadlocks, non-atomic operations
- **Rollback Safety:** can this change be safely reverted? Does it need a migration rollback plan?
- **Schema Drift:** migrations present and correct, backward compatibility maintained, data format changes handled
- **Error Handling:** swallowed errors (empty catch), missing retries, cascading failure chains, `except: pass`
- **Observability:** will operators know when this breaks? Are error logs meaningful? Are internal error details (stack traces, DB queries, file paths) exposed to untrusted clients through error responses?
- **Input Validation:** injection vectors (SQL, command, path traversal), unsanitised user input reaching dangerous sinks
- **Return Value Integrity:** does the function return a semantically correct value in all paths? Are there placeholder/fallback returns that mask failures (e.g., returning a fake success after an empty catch)?

**Finding bar (every finding MUST answer 4 questions):**
1. What can go wrong? (concrete scenario, not hypothetical)
2. Why is this code vulnerable? (cite specific file and lines)
3. Impact — what breaks and how badly? (data loss > outage > degraded UX)
4. Recommendation — specific fix with code reference

**Scope exclusions — DO NOT comment on:**
- Code style, formatting, naming conventions
- "Nice to have" improvements unrelated to correctness or safety
- Speculative issues without concrete trigger scenario
- Performance micro-optimisations

**Calibration:** Prefer one strong finding over several weak ones. If the change is genuinely solid, say so clearly — false positives erode trust.

**Output format:**
```markdown
## Adversarial Review Findings

### [SEVERITY] Finding Title

**File:** path:line-range

**What can go wrong:** ...
**Why vulnerable:** ...
**Impact:** ...
**Recommendation:** ...

(Repeat per finding. Zero findings → "No adversarial findings — change appears robust against attack.")
```

---

#### Agent 3: Blast Radius + Integration

```
Scope: Full file contents + call graph analysis
Context: Changed files PLUS grep/rg for all callers and callees
```

**Stance:** Strategic — zoom out. A change that's locally correct can still break the system. Your job: map the wider impact.

**Method:**
1. For each changed function/class/export, find ALL callers:
   ```bash
   rg -n "functionName|ClassName|exportName" --type-add 'code:*.{ts,tsx,js,py,go,rs,java}' --type code
   ```
2. For each caller, assess: does the change break the caller's assumptions?
3. Check configuration that depends on this behaviour (env vars, feature flags, API contracts, route paths)
4. Assess migration/revert safety

**Specific checks:**
- **Caller impact:** List every caller. For each: is the call signature compatible? Are return value assumptions still valid?
- **Downstream effects:** What does THIS code call? Have those callees' contracts changed?
- **Configuration coupling:** Does a config key, env var, or feature flag control this behaviour? Has the default changed?
- **Database/Migration:** Schema changes? New queries that need indexes? Backward-compatible writes?
- **API contracts:** Do route paths, request/response shapes, or error codes change?
- **Observability:** Are there metrics/alerts on this code path? Will this change trigger false alarms or silence real ones?
- **Performance anti-patterns:** N+1 queries (DB calls inside loops), unbounded loops without pagination/LIMIT, missing batch operations where available, caching opportunities missed on hot paths

**Output format:**
```markdown
## Blast Radius Analysis

### Callers of changed code

| Caller (file:line) | Changed symbol | Risk | Notes |
|---|---|---|---|
| src/handler.ts:42 | `processPayment()` | HIGH | Assumes sync return, change makes it async |

### Downstream dependencies

| Callee | Change impact | Risk |
|---|---|---|
| payments/gateway.ts | Now called without retry wrapper | MEDIUM |

### Configuration & Contracts

- **Config keys affected:** `PAYMENT_TIMEOUT_MS` — default changed from 30s to 10s
- **API contract changes:** Response shape adds `correlationId` field (backward-compatible)
- **Migration notes:** No schema change required

### Revert Safety

- **Safe to revert?** Yes — old behaviour is additive
- **Revert procedure:** Roll back this commit, no data migration needed
```

---

### Phase 2: Consolidation

After all 3 subagents return, merge their findings:

1. **De-duplicate:** Same bug found by multiple agents → keep the most detailed version, note cross-agent agreement
2. **Triage by risk tier:**
   - **CRITICAL:** Data loss, auth bypass, security exploit, unrecoverable state corruption
   - **HIGH:** Production outage risk, silent failure, race condition with user-visible impact
   - **MEDIUM:** Edge case with degraded UX, missing error handling in non-critical path, observability gap
   - **LOW:** Minor edge case unlikely to trigger, missing guard on validated input, cosmetic in logs

3. **Cross-reference:** If Agent 3 found a caller that Agent 2 flagged as vulnerable, escalate the finding severity. Also escalate when Agent 2 flags an auth/security vulnerability and Agent 3 confirms the affected surface is externally accessible (endpoint without auth gate, public API, config exposed to untrusted callers) — the blast radius finding provides independent confirmation of exploitability.

4. **Generate the report**

---

### Phase 3: Report

Output a single structured markdown report. The report is written to disk AND summarised in chat.

**Report location:** `/tmp/hermaguard/hermaguard-{timestamp}-{short-hash}.md`

**Report structure:**

```markdown
# Hermaguard Report

**Date:** {DD/MM/YY HH:MM}
**Scope:** {N} files changed, {M} files in blast radius
**Diff reference:** `git diff {from}..{to}`

---

## Summary

**Total findings:** {N}
**CRITICAL:** {N} | **HIGH:** {N} | **MEDIUM:** {N} | **LOW:** {N}

{One-paragraph narrative: the change, the key risks, what needs attention first}

---

## CRITICAL Findings

{Only CRITICAL findings here — each a full section with exploit PoC, impact, recommendation}

## HIGH Findings

{HIGH findings}

## MEDIUM Findings

{MEDIUM findings}

## LOW Findings

{LOW findings — can be a compact table}

---

## Blast Radius Map

{Files affected, callers, downstream — links to the Integration section}

---

## Verdict

**Overall risk:** {LOW / MEDIUM / HIGH / CRITICAL}

**Recommended action:**
- [ ] Fix CRITICAL findings before merge
- [ ] Fix HIGH findings before deploy
- [ ] Address MEDIUM findings next sprint
- [ ] LOW findings — backlog or dismiss

**No fixes were applied by this review.** Findings to be picked up by Octacon as separate tasks.
```

---

## Integration Points

### With Octacon (Coding Lead)
Octacon reads the Hermaguard report after any implementation task. Flow:
```
Octacon implements → Simplify → Hermaguard → Octacon fixes → Simplify → Commit
```

### With the Feature Pipeline
The feature pipeline's review step (`sdlc-review`) checks that Hermaguard was run. If a task reaches review without Hermaguard evidence for tier=full tasks, flag it. The pipeline integration is configured in `governance/multi-gate-qa.md` under a new **Gate 0 (Adversarial Review)** that gates entry to the review column.

**Gate 0 wiring** (amended in `multi-gate-qa.md`):
- Gate 0: Hermaguard — adversarial review before SDLC review
- Applies to: `backend`, `frontend`, `security`, `new-feature` tiers
- Skips for: `config`, `content`, `infra` tiers
- Evidence required: Hermaguard report path in task metadata

### With KENSEI Governance
CRITICAL findings should be logged to `#governance` — they're system-level risks.

---

## Configuration

Project-level config (optional, create `.kensei/hermaguard.yaml` in the project root):

```yaml
hermaguard:
  enabled: true
  auto_trigger: false           # Hermaguard is opt-in — no auto-trigger on commit
  scope: modified               # modified | staged | all
  skip_patterns:
    - "*.test.*"
    - "*.spec.*"
    - "*.md"
    - "*.json"
    - "*.yaml"
  max_files: 50                 # skip if more than this many files changed
  blast_radius:
    max_depth: 2                # max caller/callee hop depth
    skip_patterns:
      - "node_modules/"
      - "vendor/"
      - "dist/"
  severity_threshold: LOW       # minimum severity to report (LOW reports everything)
  agents:
    parallel: 3                 # always 3 — non-configurable
```

---

## Output Format Summary

**In chat** (after report is written):
```
Hermaguard complete.

Scope: 4 files changed, 12 files in blast radius
Findings: 7 total
  CRITICAL: 0
  HIGH: 2
  MEDIUM: 3
  LOW: 2

Report: /tmp/hermaguard/hermaguard-20260608-1430-a1b2c3d.md

Top finding: [HIGH] Race condition in payment processing (Agent 2)
  → src/payments/handler.ts:42 — concurrent calls can double-charge

No fixes applied. Handing findings to Octacon.
```

---

## Pitfalls

- **Don't guard code you don't understand.** If you can't trace the full call graph, mark findings as lower confidence.
- **Don't fabricate findings.** An empty agent report IS valid. False positives erode trust faster than missed bugs.
- **Don't fight the blast radius agent.** If it says "safe to revert," believe it unless you have concrete contrary evidence.
- **Don't let the adversarial agent go soft.** If it returns "no findings," ask it to re-examine — adversarial reviewers should ALWAYS find something worth noting, even if LOW severity.
- **Don't mix guarding with simplification.** They're separate skills with separate triggering conditions. Guard after simplify, never during.
- **Report file is mandatory.** Don't just summarise in chat — write the full report to disk for governance and traceability.

---

## Platforms

This skill is designed to run on any platform that supports Hermes Agent or Claude Code:
- **Hermes Agent** (native) — via `delegate_task` subagent dispatch
- **Claude Code** — via subagent system or as a standalone skill
- **Codex CLI** — via subprocess delegation
- **Any agent framework** with subagent capabilities
