---
name: adversarial-self-review
description: Adversarially review your own output before delivering.
version: 1.0.0
author: Yuhao Lin (YuhaoLin2005)
license: MIT
metadata:
  hermes:
    tags: [Software Development, Quality, Review, Self-Improvement]
    related_skills: [requesting-code-review]
---

# Adversarial Self-Review Skill

Adversarially review your own code, documents, or configuration before delivering to the user. Instead of checking your own work (confirmation bias), spawn independent subagents via `delegate_task` instructed to find flaws. Proven in practice: a 200-line Python script had 9 bugs caught by 4 rounds of automated adversarial review — the same script had zero bugs found by standard self-review.

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

Invoke adversarial review by calling `delegate_task` with a deliberately adversarial prompt. The exact tool call follows Hermes' standard subagent invocation pattern:

```
delegate_task(
  task="You are an adversarial reviewer. You did NOT write this output.
Your job is to find every problem before it reaches production.
Be harsh. Assume nothing works. Check for:
1. Logic errors and edge cases
2. Untested assumptions
3. Missing error handling
4. Format inconsistencies
5. Security concerns

Output format: [CRITICAL/HIGH/MEDIUM/LOW] <finding> — <why it matters>",
  context="<paste the code or document to review here>"
)
```

For higher confidence, spawn 3 independent `delegate_task` calls with varied expertise angles and require ≥2/3 agreement before dismissing a finding.

## Quick Reference

| Scenario | Subagents | Review standard |
|----------|-----------|-----------------|
| Quick check (simple fix) | 1 | One adversarial pass |
| Standard review (PR, config) | 2 | Both flag same issue = confirmed |
| Critical (production, exam) | 3 | ≥2/3 majority = confirmed finding |

## Procedure

### 1. Identify what to review

After completing a task, identify the output artifacts. If the task involved ≥3 `write_file` or `patch` calls, adversarial review is strongly recommended.

### 2. Frame the review prompt

Do NOT ask "is this correct?" — that triggers confirmation bias. Use adversarial framing:

- "You did NOT write this" — breaks ownership bias
- "Assume nothing works" — forces verification
- "Be harsh" — sets adversarial tone
- "Zero findings = invalid review round" — prevents rubber-stamping

### 3. Spawn subagents via `delegate_task`

For each reviewer, call `delegate_task` with the adversarial prompt. If using multiple reviewers, vary their expertise angles:
- Reviewer A: logic correctness and edge cases
- Reviewer B: security and error handling
- Reviewer C (optional): format consistency and documentation accuracy

### 4. Triage findings

- **CRITICAL**: Incorrect output, data loss, or security vulnerability — must fix
- **HIGH**: Failure in specific conditions — must fix
- **MEDIUM**: Quality reduction without breaking functionality — fix if time permits
- **LOW**: Style, naming, minor improvements — note but don't block

### 5. Fix and verify

Fix all CRITICAL and HIGH findings. After fixing, re-run review on changed sections only.

### 6. Report to user

Summarize: how many subagents spawned, how many findings by severity, what was fixed. Always include at least one concrete example of what was found.

## Pitfalls

- **Standard prompts produce generic results.** "Please review this" → "looks good." Adversarial framing → real findings.
- **Single reviewer = single blind spot.** Critical work needs 2-3 independent reviewers with different angles.
- **Don't argue with findings.** If a reviewer flags something, verify objectively before dismissing.
- **Zero findings usually means weak framing.** Re-run with harsher instructions before accepting a clean pass.
- **Don't review mid-task.** Complete the work first. Context-switching between creating and reviewing degrades both.

## Verification

After running an adversarial review, confirm:
1. All CRITICAL and HIGH findings are addressed in the output
2. Each subagent returned at least one specific, actionable finding (not generic approval)
3. The user received a summary of what was found and what was fixed
