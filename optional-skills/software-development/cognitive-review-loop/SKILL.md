---
name: cognitive-review-loop
description: >
  Structured cognitive review workflow for complex engineering work, bug triage,
  PR review, agent self-improvement, and post-failure analysis. Use when a task
  needs deliberate reflection before action: clarify the objective, inspect
  source evidence, identify contradictions or stale assumptions, choose the
  smallest useful intervention, verify the result, and capture a reusable lesson.
version: 1.0.0
author: TheSmokeDev
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [software-development, debugging, code-review, reflection, self-improvement, planning]
    category: software-development
    related_skills: [systematic-debugging, writing-plans, test-driven-development, requesting-code-review]
---

# Cognitive Review Loop

Use this skill to slow down just enough to avoid confident-but-wrong engineering work. The loop is evidence-first: inspect what is true, name contradictions, make one bounded move, verify it, then capture the lesson so the next similar task is cheaper.

## Use Cases

- Debugging a failure whose cause is not obvious.
- Reviewing a PR or implementation plan against a stated goal.
- Resuming work after context loss, branch drift, or failed validation.
- Hardening an agent workflow after it made a bad assumption.
- Deciding whether to patch, split, close, or rewrite an existing contribution.
- Writing a short postmortem or reusable lesson after a fix.

Do not use this for simple questions, single-command lookups, or tasks where the user already gave an exact implementation plan.

## Loop

### 1. State The Objective

Write one sentence that names the desired outcome and the evidence needed to call it done.

Good:

```text
Make the Windows MCP env-var PR mergeable by rebasing it, proving the focused tests pass, and keeping the diff to one file.
```

Avoid:

```text
Improve the PR.
```

### 2. Gather Source Evidence

Inspect the smallest set of sources that can prove the current state.

For code work, prefer:

- Current branch and diff.
- Relevant issue, PR, or failing test.
- Owning module and nearby tests.
- Repo instructions such as `AGENTS.md`, `CONTRIBUTING.md`, or package scripts.

Record facts separately from interpretations.

```text
Fact: PR #123 changes one file and is labeled P2.
Fact: The branch is 400 commits behind main.
Inference: A clean rebase is likely a higher-leverage move than adding scope.
```

### 3. Name Contradictions And Stale Assumptions

Look for mismatches before editing:

- Claimed behavior vs actual diff.
- Test proof vs runtime proof.
- Old branch base vs current main.
- User request vs implementation scope.
- Documentation vs code path.
- Local success vs CI or platform constraints.

If no contradiction is found, say so and continue.

### 4. Choose One Bounded Move

Pick the smallest action that moves the task toward done.

Prefer:

- Rebase before redesign.
- Focused fix before broad cleanup.
- One-file skill before core runtime changes.
- Existing test target before new validation machinery.
- Comment with exact evidence before vague maintainer pings.

Reject moves that make review harder unless the evidence requires them.

### 5. Execute And Verify

Run the most specific verification first, then broaden only if the change justifies it.

Examples:

```bash
scripts/run_tests.sh tests/tools/test_mcp_tool.py
python -m pytest tests/tools/test_mcp_tool.py::TestBuildSafeEnv -q
ruff check tools/mcp_tool.py
```

When a preferred command cannot run, report why and use the closest valid alternative. Do not present fallback verification as identical to the preferred path.

### 6. Capture The Lesson

End with a reusable lesson in one to three bullets. Keep it concrete enough to guide future work.

Useful lesson:

```text
- For first-time upstream contribution, ship one optional skill or one P2 bug fix before proposing core memory changes.
```

Weak lesson:

```text
- Be careful next time.
```

## Output Format

Use this compact shape unless the user asks for a different artifact:

```markdown
**Objective**
[one sentence]

**Evidence**
- [fact with source]
- [fact with source]

**Contradictions**
- [mismatch, stale assumption, or "None found"]

**Move**
[one bounded action and why]

**Verification**
- `[command]` -> [result]

**Lesson**
- [reusable rule]
```

## Review Checklist

Before finishing, confirm:

- The objective is concrete.
- Evidence comes from source files, tests, issues, PRs, or runtime output.
- Inferences are labeled as inferences.
- The action is smaller than the problem.
- Verification matches the actual behavior changed.
- The final lesson is reusable without needing this conversation.
