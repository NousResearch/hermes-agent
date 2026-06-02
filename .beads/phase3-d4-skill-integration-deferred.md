---
id: phase3-d4-skill-integration-deferred
title: Phase 3 D4 — code-scan SKILL.md: surface --incremental/--full flags (DEFERRED)
status: deferred
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase3-plan
allowed_files:
  - skills/code-analysis/code-scan/SKILL.md
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/extract_imports.py
  - scripts/code-scan/fingerprints.py
  - scripts/code-scan/assemble_graph.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - .hermesignore
depends_on:
  - phase3-d2-incremental-scan
verification:
  - wc -l skills/code-analysis/code-scan/SKILL.md
  - grep -c "incremental" skills/code-analysis/code-scan/SKILL.md
risk: low
---

# Phase 3 D4 — code-scan SKILL.md: Surface --incremental/--full Flags (DEFERRED)

## Context & Intent

**Why this bead exists (if approved).** Phase 3 D2 adds `--incremental` and `--full` flags to `scan_project.py`. D4 makes the code-scan skill aware of these flags so that when an agent re-scans a project that already has fingerprints, it prefers `--incremental` by default and offers `--full` as an explicit override.

**⚠️ DEFAULT STATUS: DEFERRED.** This deliverable is **not included** in the standard Phase 3 approval scope. It will only execute if JC **explicitly** includes D4 in the Phase 3 approval.

**Authority docs.** `.plans/phase-3-incremental-analysis.md` (D4 section) defines the scope. Phase 2 D2 bead (`.beads/phase2-d2-code-scan-skill.md`) defines the original SKILL.md structure.

**Intent.** Update `skills/code-analysis/code-scan/SKILL.md` to:
1. Mention the `--incremental` flag in the orchestration steps.
2. Add conditional logic: if `.hermes/code-state/fingerprints.json` exists in the target project, recommend `--incremental` to the agent; offer `--full` as an override.
3. Keep the SKILL.md under 80 lines total (current: 39 lines; ~40 lines of headroom available).

**Non-goals.** No changes to scan_project.py (already handled by D2). No new files. No always-on behavior changes.

## Implementation Details

### Target files

| File | Purpose | Max LOC |
|---|---|---|
| `skills/code-analysis/code-scan/SKILL.md` | Add incremental mode references | ≤80 lines total |

### Required changes to SKILL.md

The skill currently has 39 lines. D4 would add approximately 15-20 lines of conditional logic and references to `--incremental`/`--full`. Total must stay ≤80 lines.

The recommended additions are woven into the existing orchestration steps rather than appended:

```markdown
<!-- Current step 2 (modified): -->
2. Run scan_project.py against the project:
   - If .hermes/code-state/fingerprints.json exists, use: --incremental
   - To force a full re-scan (ignoring fingerprints), use: --full
   - First scan (no fingerprints): --incremental behaves as full scan automatically
   - Example: python scripts/code-scan/scan_project.py <target> --incremental --output scan.json
```

### Exact proposed additions

```markdown
<!-- In the skill body, after confirming project directory: -->
Check for existing fingerprints: .hermes/code-state/fingerprints.json in the target project.
- If present: append --incremental to the scan_project.py command
- If absent: --incremental is still safe; it auto-falls to full scan
- If user requests full re-scan: use --full (overrides --incremental)
```

## Complexity Tier

**T1** — Minor edit to an existing SKILL.md file. Line-count constraint requires careful wording. No code changes. Estimated 3-4 subagent iterations. Requires coder subagent + Hermes verification + reviewer signoff.

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent via `delegate_task` with full plan context.

**Process:**
1. Coder subagent reads current SKILL.md, drafts additions that fit within 80-line budget.
2. Update SKILL.md with incremental references.
3. Hermes verifies `wc -l skills/code-analysis/code-scan/SKILL.md` ≤ 80.
4. Reviewer validates: line budget, clarity of incremental instructions, no regression to existing steps.
5. Hermes presents evidence bundle to JC.

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 3 D2 (--incremental flag) | prerequisite | Must be complete before D4 |
| skills/code-analysis/code-scan/SKILL.md | target file | Exists (39 lines, from Phase 2 D2) |
| Python 3.10+ | not applicable | No code changes |

## Test Obligations

### Contract tests

| Test | Command | Pass Criteria |
|---|---|---|
| Line budget | `wc -l skills/code-analysis/code-scan/SKILL.md` | ≤80 lines |
| Keyword presence | `grep -c "incremental" skills/code-analysis/code-scan/SKILL.md` | ≥1 |
| Keyword presence | `grep -c "\-\-full" skills/code-analysis/code-scan/SKILL.md` | ≥1 |
| Fingerprint reference | `grep -c "fingerprints" skills/code-analysis/code-scan/SKILL.md` | ≥1 |
| No skill regression | Read full SKILL.md | All 7 original orchestration steps still present |

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent

# Line budget check
wc -l skills/code-analysis/code-scan/SKILL.md | awk '{if ($1 > 80) { print "FAIL: " $1 " lines"; exit 1 } else { print "PASS: " $1 " lines" }}'

# Keyword presence
grep -q "incremental" skills/code-analysis/code-scan/SKILL.md && echo 'INCREMENTAL REF PASS' || echo 'INCREMENTAL REF FAIL'
grep -q "\-\-full" skills/code-analysis/code-scan/SKILL.md && echo 'FULL REF PASS' || echo 'FULL REF FAIL'
grep -q "fingerprints" skills/code-analysis/code-scan/SKILL.md && echo 'FP REF PASS' || echo 'FP REF FAIL'

# Scope guardrail
git diff --name-only | grep -vE '^(skills/code-analysis/code-scan/SKILL\.md)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'

# Forbidden file check
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py | grep -q . && echo 'FORBIDDEN FAIL' || echo 'FORBIDDEN PASS'
```

## Approval Evidence

**1. Line budget:**
```bash
wc -l skills/code-analysis/code-scan/SKILL.md  # Must be ≤80
```

**2. Keyword check:**
```bash
grep -c "incremental\|\-\-full\|fingerprints" skills/code-analysis/code-scan/SKILL.md
```

**3. Diff:**
```bash
git diff skills/code-analysis/code-scan/SKILL.md
```

**4. Reviewer verdict:**
- [ ] Line budget ≤80
- [ ] Incremental references present and clear
- [ ] Original orchestration steps preserved
- [ ] Scope preservation (only SKILL.md modified)
- [ ] Workspace remains clean outside allowed files

**5. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
```

---

> **DEFAULT STATUS: DEFERRED.** Not part of Phase 3 approval scope.
> **Execute only if JC explicitly includes D4 in the Phase 3 approval.**
> Coder subagent has NO commit/push authority.
