---
name: bug-pipeline
description: Six-stage quality-gate pipeline for fixing bugs — research, analysis, implementation, review, test, document. Each stage gates on evidence before the next proceeds. Uses delegate_task with MPM profiles for execution.
version: 2.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, pipeline, bug-fix, quality]
    related_skills: [task-router, orchestration-patterns]
---

# Bug Pipeline — 6-Stage Quality Gate

> One bug, one pipeline, one verified fix. Never skip stages. Never claim done without evidence.

## Trigger

Apply this skill when fixing a **bug** from a GitHub issue, kanban card, user report, or test failure.

Do NOT trigger for:
- Feature work (use `orchestration-patterns`)
- Small fixes ≤10 lines, ≤2 files, no security impact, no API changes → do it inline
- Pure research

## How the pipeline works

The pipeline is something **you** (the main agent) execute sequentially. You do NOT delegate the whole pipeline — you walk each stage yourself, delegating sub-tasks to MPM profiles where specialized work is needed.

For each stage you:
1. Read the stage procedure
2. Do the work (delegate, search, read code, run tests as needed)
3. Record evidence in the KB (`lore-axi kb-add key=<stage>/<evidence> value="..."`)
4. Move to the next stage

---

## Stage 1 — Research

**You do this yourself.** Use your installed tools (tavily, exa, gh-axi, lore-axi, terminal, file_read).

1. **Duplicate check:** `gh-axi pr search "is:open <issue keywords>"`. If a PR exists → review it instead. If a PR was rejected → read why. Record result.
2. **Understand the bug:** Read the issue body. Note exact environment, expected vs observed behavior, reproduction steps.
3. **Find relevant files:** Use `terminal` + grep/find or code search tools to locate the components mentioned.
4. **Identify constraints:** Platform compat (Windows?), security boundaries, no-regression requirements.
5. **Record:** `lore-axi kb-add key="research/duplicate-check" value="pass|fail — details"` and `lore-axi kb-add key="research/files-found" value="file1, file2, ..."`

**Gate:** All evidence recorded → move to Analysis.

---

## Stage 2 — Code Analysis

**Delegate this to the `debugger` or `think` profile.** They have the toolset for deep code tracing.

```
delegate_task(
  profile="debugger",
  goal="Trace the root cause of bug <N>: <title>",
  context="Files to examine: <paths>. Bug reproduction: <steps>"
)
```

The subagent should return:
- Root cause identification (which code path, what condition fails)
- Fix approach (numbered list of changes)
- Scope assessment (what files to touch, minimal change)

**Record:**
- `lore-axi kb-add key="analysis/root-cause" value="..."` 
- `lore-axi kb-add key="analysis/fix-plan" value="..."`

**Gate:** Root cause + fix plan documented → move to Implementation.

---

## Stage 3 — Implementation

**Delegate this to the `engineer` profile.** They have terminal + file + code_execution toolsets.

```
delegate_task(
  profile="engineer",
  goal="Apply the fix for bug <N>",
  context="Fix plan: <numbered changes>. Files to modify: <paths>."
)
```

After the subagent returns, **verify yourself:**
- Syntax check: `python3 -c "ast.parse(open('file.py').read())"` (or equivalent for your language)
- Test the affected module: run the existing test suite

**Record:**
- `lore-axi kb-add key="implementation/changes" value="<diff summary>"`
- `lore-axi kb-add key="implementation/tests-pass" value="X tests passed, 0 regressions"`
- `lore-axi kb-add key="implementation/lint-pass" value="syntax OK"`

**Gate:** Tests pass + lint OK → move to Review.

---

## Stage 4 — Code Review

**You do the review yourself** using the severity-tagged checklist below. Read the actual diff.

| Severity | What to check |
|----------|---------------|
| CRITICAL | Secrets hardcoded, SQL injection, code execution, auth bypass, infinite loops |
| HIGH | No bare except, no mutable defaults, tests pass, error cases handled |
| MEDIUM | Functions ≤20 lines, docstrings on public methods, no `Any` types |
| LOW | Naming, imports, no commented-out code |

**Verdict:**
- **APPROVE** — no CRITICAL/HIGH → proceed
- **WARN** — no CRITICAL, some HIGH → log findings, proceed
- **BLOCK** — any CRITICAL → stop, surface to user

If MPM gate is enabled (`hermes mpm gate-status`), you can route the diff through it for an independent reviewer:
```
delegate_task(profile="think", goal="Review this diff for bugs...", context="<diff>")
```

**Record:**
- `lore-axi kb-add key="review/verdict" value="APPROVE|WARN|BLOCK"`
- `lore-axi kb-add key="review/findings" value="<finding table>"`

**Gate:** APPROVE or WARN → move to QA.

---

## Stage 5 — QA

**You run tests yourself** via the terminal tool.

1. Run the full test suite for the affected module(s)
2. Check for regressions (all pre-existing failures should be noted)
3. Run lint/type check if available

**Record:**
- `lore-axi kb-add key="qa/test-results" value="X passed, Y failed — 0 regressions"`

**Gate:** Tests pass → move to Documentation.

---

## Stage 6 — Documentation

**You do this yourself.**

1. Update inline comments for any non-obvious design decisions
2. Write a changelog fragment if the project uses them (changelog.d/ or similar)
3. Create the pull request via `gh-axi`

```
gh-axi pr create \
  --title "fix(<scope>): <short description> (#<N>)" \
  --body "<summary of the fix>" \
  --base main
```

**Record:**
- `lore-axi kb-add key="documentation/comments-updated" value="<what was documented>"`
- `lore-axi kb-add key="documentation/pr-link" value="<PR URL>"`

---

## Circuit Breakers

| # | Rule | What to do instead |
|---|------|-------------------|
| 1 | NEVER edit files yourself | Delegate to `engineer` profile |
| 2 | NEVER use bash for modifications | Bash only for: git, ls, running tests |
| 3 | NEVER claim "done" without evidence | Show subagent output or test output |
| 4 | NEVER skip stages | Each stage must complete sequentially |
| 5 | Parallelize independent tasks | Use `tasks=[...]` in delegate_task |

## Fast-Path (skip heavy pipeline)

**ALL must be true:** ≤10 source lines, ≤2 files, no new deps, no API changes, no security impact.

Route: `Research (5-min scan) → Implementation → Docs`. Skip Analysis, Review, full QA. Run `pytest <module>` as sanity check only.

## KB evidence cheatsheet

Use lore-axi to record gates:

```bash
lore-axi kb-add key="research/duplicate-check" value="no existing PRs — proceeding"
lore-axi kb-add key="analysis/root-cause" value="<verdict>"
lore-axi kb-add key="implementation/tests-pass" value="57 passed"
lore-axi kb-add key="review/verdict" value="APPROVE"
lore-axi kb-add key="qa/test-results" value="57 passed, 0 regressions"
lore-axi kb-add key="documentation/pr-link" value="https://github.com/.../pull/N"
```

To check earlier stages:
```bash
lore-axi kb-search "research/" | head
```
