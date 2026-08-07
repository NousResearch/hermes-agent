---
name: cognitive-review-loop
description: Evidence-first reflection before risky engineering moves.
version: 1.0.0
author: SmokeDev (TheSmokeDev)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [software-development, debugging, code-review, reflection, self-improvement, planning]
    category: software-development
    related_skills: [systematic-debugging, plan, test-driven-development, requesting-code-review]
---

# Cognitive Review Loop Skill

A six-step reflection loop for engineering work where a confident-but-wrong move
is expensive: state the objective, gather source evidence, name contradictions,
make one bounded move, verify it, and capture a reusable lesson. It structures
reasoning around existing tools (`read_file`, `search_files`, `terminal`); it
adds no automation and is not for tasks that already have an exact plan.

## When to Use

- Debugging a failure whose cause is not obvious.
- Reviewing a PR or implementation plan against a stated goal.
- Resuming work after context loss, branch drift, or failed validation.
- Hardening an agent workflow after it acted on a bad assumption.
- Deciding whether to patch, split, close, or rewrite an existing contribution.
- Writing a short postmortem or reusable lesson after a fix.

Skip it for simple questions, single-command lookups, or tasks where the user
already supplied an exact implementation plan.

## Prerequisites

None. The skill is prose-only: no env vars, no scripts, no network access.
Evidence gathering uses the native `read_file`, `search_files`, and `terminal`
tools.

## How to Run

Work through the six Procedure steps in order and finish with the compact
output block from the Quick Reference. Gather evidence with `read_file`
(file contents), `search_files` (symbols and paths), and `terminal`
(git state, test runs).

## Quick Reference

| Step | Action |
|---|---|
| 1 | State the objective in one sentence, with the evidence needed to call it done |
| 2 | Gather source evidence; record facts separately from inferences |
| 3 | Name contradictions and stale assumptions, or state "None found" |
| 4 | Choose one bounded move |
| 5 | Execute, then verify with the most specific check first |
| 6 | Capture a reusable lesson in 1-3 bullets |

Output shape:

```markdown
**Objective** [one sentence]
**Evidence** - [fact with source]
**Contradictions** - [mismatch, stale assumption, or "None found"]
**Move** [one bounded action and why]
**Verification** - `[command]` -> [result]
**Lesson** - [reusable rule]
```

## Procedure

### 1. State the objective

One sentence naming the desired outcome and the evidence needed to call it
done. "Make the Windows MCP env-var PR mergeable by rebasing it and proving
the focused tests pass" is an objective; "improve the PR" is not.

### 2. Gather source evidence

Inspect the smallest set of sources that can prove the current state: the
current branch and diff (`terminal`), the failing test or linked issue, the
owning module and its nearby tests (`read_file`, `search_files`), and repo
instructions such as `AGENTS.md` or `CONTRIBUTING.md`. Record facts separately
from interpretations:

```text
Fact: PR #123 changes one file and is labeled P2.
Fact: The branch is 400 commits behind main.
Inference: A clean rebase is a higher-leverage move than adding scope.
```

### 3. Name contradictions and stale assumptions

Look for mismatches before editing: claimed behavior vs actual diff, test
proof vs runtime proof, old branch base vs current main, user request vs
implementation scope, documentation vs code path, local success vs CI or
platform constraints. If no contradiction is found, say so and continue.

### 4. Choose one bounded move

Pick the smallest action that moves the task toward done: rebase before
redesign, focused fix before broad cleanup, one-file skill before core runtime
changes, existing test target before new validation machinery, a comment with
exact evidence before a vague maintainer ping. Reject moves that make review
harder unless the evidence requires them.

### 5. Execute and verify

Run the most specific verification first, then broaden only if the change
justifies it. In this repository:

```bash
scripts/run_tests.sh tests/tools/test_mcp_tool.py -q
ruff check tools/mcp_tool.py
```

When a preferred command cannot run, report why and use the closest valid
alternative. Do not present fallback verification as identical to the
preferred path.

### 6. Capture the lesson

End with 1-3 concrete, reusable bullets. "For a first upstream contribution,
ship one optional skill or one P2 bug fix before proposing core memory
changes" is reusable; "be careful next time" is not.

## Pitfalls

- A vague objective ("improve X") makes every later step unverifiable.
- Evidence recalled from memory instead of re-read from source files.
- Inferences presented as facts instead of being labeled.
- Scope creep inside the "one bounded move" step.
- Fallback verification reported as if it were the preferred check.
- Lessons too generic to guide the next similar task.

## Verification

The skill ships a standards test covering frontmatter shape, description
budget, section order, and wrapper-only test invocations:

```bash
scripts/run_tests.sh tests/skills/test_cognitive_review_loop_skill.py -q
```

A completed loop is verifiable when the output block's Verification line
contains a command actually run this session and the Lesson stands on its own
without the surrounding conversation.
