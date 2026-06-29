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

Adversarially review your own code, documents, or configuration before delivering to the user. Instead of checking your own work (confirmation bias), spawn independent subagents instructed to find flaws. Proven to catch bugs that self-review misses — one 200-line script had 9 bugs found, 8 by adversarial review alone.

This skill does NOT replace human code review. It catches the class of errors that self-review is blind to: untested assumptions, missing edge cases, format inconsistencies, and silent failures.

## When to Use

- After writing code/scripts/documents where correctness matters (满分 scenarios, production configs, security-sensitive code)
- When the user says "review this", "check my work", "is this correct?", or "自审"
- Before pushing a PR, publishing content, or delivering a complex output
- After any multi-file edit session where errors could cascade

Do NOT use for:
- Single-line typo fixes
- Trivial formatting changes
- Tasks the user explicitly says don't need review

## Prerequisites

- `delegate_task` tool must be available (bundled with Hermes)
- No external API keys or MCP servers required
- Works on all platforms

## How to Run

Invoke through a `delegate_task` subagent with adversarial framing. The key is the prompt — standard review prompts trigger confirmation bias. Adversarial prompts force the reviewer to assume bugs exist:

```
Spawn via delegate_task:
"You are an adversarial reviewer. You did NOT write this output.
Your job is to find every problem before it reaches production.
Be harsh. Assume nothing works. Check for:
1. Logic errors and edge cases
2. Untested assumptions
3. Missing error handling
4. Format inconsistencies
5. Security concerns

Output: [CRITICAL/HIGH/MEDIUM/LOW] <finding> — <why it matters>"
```

For higher confidence, spawn 3 independent subagents and require ≥2/3 agreement before dismissing a finding.

## Quick Reference

| Scenario | Subagent count | Review standard |
|----------|---------------|-----------------|
| Quick check (simple fix) | 1 | One adversarial pass |
| Standard review (PR, config) | 2 | Both must agree on pass/fail |
| 满分 (exam/critical) | 3 | 2/3 majority = confirmed finding |

## Procedure

### 1. Identify what to review

After completing a task, identify the output artifacts: code files, documents, config changes, PR descriptions. If the task involved ≥3 file edits, adversarial review is mandatory.

### 2. Frame the review prompt

Do NOT ask "is this correct?" — that triggers confirmation bias. Instead frame as:

```
"You did NOT write this code. Someone else did. Your job is to find every
problem with it before it reaches production. Be harsh. Assume nothing works."
```

Key adversarial framing techniques:
- "You did NOT write this" — breaks ownership bias
- "Assume nothing works" — forces verification
- "Be harsh" — sets adversarial tone
- Require specific findings — "zero findings = invalid review round"

### 3. Spawn subagents via `delegate_task`

For each reviewer, use `delegate_task` with the adversarial prompt. If using multiple reviewers, vary their expertise angles:
- Reviewer A: logic correctness and edge cases
- Reviewer B: security and error handling
- Reviewer C (optional): format consistency and documentation accuracy

### 4. Triage findings

Classify each finding:
- **CRITICAL**: Would cause incorrect output, data loss, or security issue
- **HIGH**: Would cause failure in specific conditions
- **MEDIUM**: Reduces quality but doesn't break functionality
- **LOW**: Style, naming, minor improvements

### 5. Fix and verify

Fix all CRITICAL and HIGH findings. For MEDIUM, fix if time permits. For LOW, note but don't block delivery. After fixing, re-run review on changed sections only.

### 6. Report to user

```
Adversarial review complete:
- 3 subagents spawned
- 2 CRITICAL, 3 HIGH, 1 MEDIUM found
- All CRITICAL/HIGH fixed
- MEDIUM: <brief note if unfixed>
```

## Pitfalls

- **Don't skip the adversarial framing.** "Please review this code" produces generic suggestions. "You didn't write this, find every bug" produces real findings.
- **One reviewer can miss things.** For critical work, use 2-3 independent reviewers with different angles.
- **Don't argue with findings.** If a reviewer flags something you disagree with, verify objectively before dismissing. The whole point is they see what you can't.
- **Zero findings is suspicious.** If all reviewers return nothing, the adversarial framing probably wasn't strong enough. Re-run with harsher instructions.
- **Don't review mid-task.** Complete the work first, then review. Context-switching between creating and reviewing degrades both.

## Verification

After running an adversarial review, confirm:
1. All CRITICAL/HIGH findings are addressed in the output
2. The subagent response contains at least one specific, actionable finding (not "looks good")
3. The user sees a summary of what was found and fixed
