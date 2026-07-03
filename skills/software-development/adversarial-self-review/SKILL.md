---
name: adversarial-self-review
description: Adversarially review your own output before delivering, using the C/C/G/H four-dimension audit framework.
version: 1.0.0
author: Yuhao Lin (YuhaoLin2005)
license: MIT
metadata:
  hermes:
    tags: [Software Development, Quality, Review, Self-Improvement]
    related_skills: [requesting-code-review]
---

# Adversarial Self-Review Skill

Adversarially review your own code, documents, or configuration before delivering to the user. Instead of checking your own work (confirmation bias), spawn independent subagents via `delegate_task` instructed to find flaws. Each reviewer checks four dimensions — Completeness, Consistency, Groundedness, and Honesty (C/C/G/H) — the same framework used across quality gates to ensure no dimension is overlooked.

This skill does NOT replace human code review. It catches the class of errors that self-review is blind to: untested assumptions, missing edge cases, format inconsistencies, and silent failures.

## When to Use

- After writing code, scripts, documents, or configuration where correctness matters
- When the user says "review this", "check my work", "is this correct?"
- Before pushing a PR, publishing content, or delivering a complex output
- After any multi-file edit session where errors could cascade

Do NOT use for:
- Single-line typo fixes or trivial formatting changes
- Tasks the user explicitly says don't need review
- Mid-task — complete the work first, then review

## Prerequisites

- `delegate_task` tool must be available (bundled with Hermes)
- No external API keys or MCP servers required
- Works on all platforms

## How to Run

Invoke adversarial review by calling `delegate_task` with a deliberately adversarial prompt structured around the four C/C/G/H dimensions:

```
delegate_task(
  task="You are an adversarial reviewer. You did NOT write this output.
Your job is to find every problem before it reaches production.
Be harsh. Assume nothing works. Check all four dimensions:

1. COMPLETENESS — Are all requirements covered? Edge cases handled?
   Error paths accounted for? Security concerns addressed?

2. CONSISTENCY — Does the output contradict itself? Are naming
   conventions, formats, and patterns consistent throughout?

3. GROUNDEDNESS — Are claims backed by evidence? Are assumptions
   tested? Or does it just assert things work without proof?

4. HONESTY — Are limitations acknowledged? Are error messages
   accurate? Is the confidence level proportional to the evidence?

Output format: [C/H/G/C — CRITICAL/HIGH/MEDIUM/LOW] <finding> — <why it matters>",
  context="<paste the code or document to review here>"
)
```

For higher confidence, spawn 3 independent `delegate_task` calls with varied expertise angles and require ≥2/3 agreement before dismissing a finding.

## C/C/G/H Review Dimensions

| Dimension | What to Check | Example Finding |
|-----------|--------------|-----------------|
| **Completeness** | Edge cases, missing error handling, uncovered requirements | "No error handling for empty input — returns undefined" |
| **Consistency** | Format drift, naming contradictions, pattern breaks | "Uses camelCase in auth.js but snake_case in db.js" |
| **Groundedness** | Untested assumptions, unsupported claims, missing validation | "Claims 'fast enough' without benchmark or threshold" |
| **Honesty** | Missing limitations, overconfident assertions, misleading errors | "Returns 'success' on partial failure — masks data loss" |

## Quick Reference

| Scenario | Subagents | Review standard |
|----------|-----------|-----------------|
| Quick check (simple fix) | 1 | One adversarial pass, all four C/C/G/H dimensions |
| Standard review (PR, config) | 2 | Both flag same issue = confirmed |
| Critical (production, exam) | 3 | ≥2/3 majority = confirmed finding |

## Procedure

### 1. Identify what to review

After completing a task, identify the output artifacts. If the task involved ≥3 `write_file` or `patch` calls, adversarial review is strongly recommended.

### 2. Frame the review prompt

Do NOT ask "is this correct?" — that triggers confirmation bias. Use adversarial framing structured around C/C/G/H:

- "You did NOT write this" — breaks ownership bias
- "Check all four dimensions: Completeness, Consistency, Groundedness, Honesty" — prevents blind spots
- "Assume nothing works" — forces verification
- "Be harsh" — sets adversarial tone
- "Zero findings = invalid review round" — prevents rubber-stamping

### 3. Spawn subagents via `delegate_task`

For each reviewer, call `delegate_task` with the C/C/G/H-structured adversarial prompt. If using multiple reviewers, vary their expertise angles while keeping the four-dimension structure:
- Reviewer A: logic correctness and edge cases (C-focused)
- Reviewer B: security and error handling (C+G-focused)
- Reviewer C (optional): format consistency and documentation accuracy (C+H-focused)

### 4. Triage findings

Tag each finding with its C/C/G/H dimension for cross-referencing:

- **CRITICAL**: Incorrect output, data loss, or security vulnerability — must fix
- **HIGH**: Failure in specific conditions — must fix
- **MEDIUM**: Quality reduction without breaking functionality — fix if time permits
- **LOW**: Style, naming, minor improvements — note but don't block

### 5. Fix and verify

Fix all CRITICAL and HIGH findings. After fixing, re-run review on changed sections only.

### 6. Report to user

Summarize: how many subagents spawned, how many findings by severity and by C/C/G/H dimension, what was fixed. Always include at least one concrete example of what was found.

## Pitfalls

- **Standard prompts produce generic results.** "Please review this" → "looks good." Adversarial framing with C/C/G/H structure → real findings across all four dimensions.
- **Single reviewer = single blind spot.** Critical work needs 2-3 independent reviewers with different angles, all using the same four-dimension framework.
- **Don't argue with findings.** If a reviewer flags something, verify objectively before dismissing.
- **Zero findings usually means weak framing.** Re-run with harsher instructions before accepting a clean pass.
- **Don't review mid-task.** Complete the work first. Context-switching between creating and reviewing degrades both.
- **Dimension imbalance.** If all findings are in one C/C/G/H dimension, the review framing may be skewed — re-run with explicit prompts for the missing dimensions.

## Verification

After running an adversarial review, confirm:
1. All CRITICAL and HIGH findings are addressed in the output
2. Each subagent returned at least one specific, actionable finding (not generic approval)
3. Findings span multiple C/C/G/H dimensions (all-Completeness = framing too narrow)
4. The user received a summary of what was found and what was fixed
