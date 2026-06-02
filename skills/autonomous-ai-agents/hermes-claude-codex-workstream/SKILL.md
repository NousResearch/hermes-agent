---
name: hermes-claude-codex-workstream
description: Use when setting up or running a Hermes Agent controlled build workflow where Hermes sharpens the brief and routes, Claude Code plans/designs/reviews and gap-checks, Codex CLI implements, Claude plus Codex independently review, and Hermes reconciles, verifies, and signs off.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes-agent, claude-code, codex, orchestration, build-workflow, verification]
    related_skills: [hermes-agent, claude-code, codex]
---

# Hermes Claude Code and Codex Workstream

## Overview

This skill packages a Hermes-specific external engine workflow:

```text
Human intent
→ Hermes Agent sharpens the brief and routes
→ Claude Code plans, designs, reviews, and pushes back when the brief has gaps
→ Codex CLI implements
→ Claude Code and Codex CLI independently review, using their own subagents when useful
→ Hermes Agent reconciles, verifies, and signs off
```

Hermes Agent stays the controller, memory layer, brief writer, run logger, and final verifier. Claude Code is the planning, design, critique, and review engine. Codex CLI is the implementation and test-loop engine. Human signoff stays explicit when product direction, design, or risky scope changes are involved.

The point is not to run two coding agents and hope. The edge is the artefact loop: Hermes sharpens the human intent into a verifiable brief, Claude Code refuses vague or missing requirements, Codex builds only after the slice is sharp, both engines review independently, and Hermes reconciles disagreements with real verification commands.

## When to Use

Use this skill when the user wants to:

- Set up a Hermes Agent workflow using Claude Code plus Codex CLI.
- Turn a rough product or engineering intent into a buildable external-agent workstream.
- Run a feature, bug fix, PR review, prototype, or cleanup through a controlled Claude and Codex loop.
- Teach another Hermes user how to reproduce the workflow.
- Create a public or team-safe version of an internal Hermes build workflow.

Do not use this for:

- Tiny one-command edits where Hermes can patch and verify faster directly.
- Tasks requiring secrets to be shared with external CLIs.
- Unbounded multi-agent chats where Claude and Codex talk freely without Hermes control.
- Production deploys unless the deploy runbook and rollback path are already defined.

## Prerequisites

The Hermes controller needs terminal access and process management.

Install and verify the CLIs:

```bash
# Claude Code
npm install -g @anthropic-ai/claude-code
claude auth status --text || claude auth login
claude doctor
claude --version

# Codex CLI
npm install -g @openai/codex
codex --version

# Repo helpers
git --version
gh --version || true
```

Authentication notes:

- Claude Code can use browser OAuth, console auth, SSO, or `ANTHROPIC_API_KEY` depending on the user's account.
- Codex CLI can use Codex OAuth or `OPENAI_API_KEY` depending on the user's setup.
- Hermes itself may use a different provider configuration from the standalone CLIs. Do not assume Hermes auth proves CLI auth, or the reverse.
- Never print API keys, OAuth tokens, auth JSON, or credential file contents.

Before running real work, load the companion skills if available:

- `claude-code` for exact Claude Code CLI flags and PTY handling.
- `codex` for exact Codex CLI flags and PTY handling.
- `hermes-agent` when configuring Hermes profiles, tools, gateways, or skills.

## Roles

### Hermes Agent controller

Hermes owns:

- Intent grilling and scope control.
- The brief, constraints, acceptance criteria, and verification plan.
- Worktree or workspace setup.
- Prompts sent to Claude Code and Codex CLI.
- Run directory and artefact capture.
- Review reconciliation.
- Final verification using real commands.
- User-facing report with action, target, verification command, result, and remaining risks.

Hermes does not treat external-agent output as proof. Agent claims are input. Command output is proof.

### Claude Code

Use Claude Code for:

- Product and UX direction critique.
- Architecture and implementation planning.
- Design tradeoff analysis.
- Brief gap detection and pushback before build starts.
- Maintainability review.
- Security and risk review.
- Second-opinion review of Codex changes.

Claude Code should not passively accept a weak brief. If acceptance criteria, constraints, design direction, verification commands, or scope boundaries are missing, Claude should return the gaps and ask Hermes to resolve them before Codex starts implementation.

Prefer Claude print mode for bounded tasks:

```bash
claude -p "Read the brief and produce a concise implementation plan with risks and acceptance criteria." \
  --output-format json \
  --max-turns 8
```

Use interactive Claude sessions only when the workflow needs multi-turn exploration, slash commands, or human-in-the-loop decisions.

### Codex CLI

Use Codex CLI for:

- Implementing the signed-off slice.
- Running tests and fixing failures.
- Producing diffs.
- Narrow bug review.
- Scoped fix passes after review.

Codex should implement the signed-off brief, not reinterpret the product direction. If Codex finds the brief impossible, contradictory, or under-specified, it should stop and report the blocker rather than inventing scope.

Codex generally needs a git repository and a PTY:

```bash
codex exec --full-auto "Implement the attached brief. Run the listed verification commands. Report changed files and remaining failures."
```

Use isolated worktrees for implementation by default.

## Core Workflow

### 1. Capture human intent and sharpen the brief

A brief must include:

- Goal in one sentence.
- Non-goals.
- Constraints.
- Scope boundaries.
- Acceptance criteria.
- Verification commands.
- Expected artefacts.
- Signoff trigger, if product or design direction matters.

If success cannot be verified in one or more commands or observable artefacts, the brief is not ready.

Hermes should also decide the route:

- Claude Code first when design, architecture, requirements, or risk need pressure-testing.
- Codex first only for narrow, already-specified implementation tasks.
- Human signoff before Codex when Claude identifies material gaps or product/design choices.

### 2. Create a run directory

Keep every external-agent workstream auditable.

```bash
RUN_ID="run-$(date +%Y%m%d-%H%M%S)-topic"
mkdir -p ".hermes/runs/$RUN_ID"
```

Recommended layout:

```text
.hermes/runs/<run-id>/
  brief.md
  claude-plan.json
  claude-plan.md
  codex-build.log
  changed-files.txt
  diff.patch
  claude-review.json
  claude-review.md
  codex-review.log
  reconciled-findings.md
  fix-pass.log
  verification.txt
  signoff.md
```

If `.hermes/runs/` is not appropriate for the repo, use `.agent-runs/` or another ignored run-artifact directory.

### 3. Ask Claude for plan, design, review, and gap detection

Use Claude before build when the product shape, design, architecture, or risk profile is uncertain.

Example:

```bash
claude -p "$(cat .hermes/runs/$RUN_ID/brief.md)

Return:
1. recommended approach
2. risks
3. missing decisions
4. implementation plan
5. verification plan
6. whether this is ready for Codex implementation
Do not write files." \
  --output-format json \
  --max-turns 8 \
  > ".hermes/runs/$RUN_ID/claude-plan.json"
```

Hermes then reads and critiques the plan. If Claude says the brief has gaps, Hermes resolves the gaps or asks the human for a decision. If direction needs human signoff, stop and get signoff before Codex builds.

### 4. Create an isolated implementation worktree

```bash
git fetch --all --prune
BASE_BRANCH="main"
BRANCH="agent/$RUN_ID"
WORKTREE="/tmp/$RUN_ID"
git worktree add -b "$BRANCH" "$WORKTREE" "$BASE_BRANCH"
```

Rules:

- One implementation owner per worktree.
- Do not run broad unrelated cleanup in the same slice.
- Do not let Codex edit the controller's run notes except where explicitly asked.
- If the worktree is dirty before Codex starts, stop and resolve it.

### 5. Send Codex the signed-off build prompt

The Codex prompt should contain:

- Brief path or brief text.
- Exact scope.
- Files or directories likely in scope.
- Verification commands.
- Instruction to report failures honestly.
- Instruction not to commit unless requested.

Example:

```bash
codex exec --full-auto "
You are implementing one signed-off slice.

Read this brief:
$(cat .hermes/runs/$RUN_ID/brief.md)

Rules:
- Stay inside scope.
- Do not commit.
- Run the verification commands from the brief.
- If a command fails, keep the raw failure and explain the blocker.
- End with changed files, commands run, failures, and remaining risks.
" 2>&1 | tee ".hermes/runs/$RUN_ID/codex-build.log"
```

When running through Hermes tools, use `terminal(..., pty=true)` for Codex and `background=true` with `notify_on_complete=true` for long bounded work.

### 6. Capture diffs and artefacts

After Codex exits, Hermes must inspect the worktree.

```bash
git status --short > ".hermes/runs/$RUN_ID/changed-files.txt"
git diff --stat > ".hermes/runs/$RUN_ID/diff-stat.txt"
git diff > ".hermes/runs/$RUN_ID/diff.patch"
```

If no files changed but Codex claimed completion, treat that as a failed run until explained.

### 7. Run independent Claude and Codex review passes

Claude Code and Codex CLI review independently. They can use their own subagents or internal review modes when available, but Hermes only needs the final artefacts: findings, evidence, uncertainty, and whether each finding is actionable.

Claude review prompt:

```bash
claude -p "
Review this implementation against the brief.

Brief:
$(cat .hermes/runs/$RUN_ID/brief.md)

Diff:
$(cat .hermes/runs/$RUN_ID/diff.patch)

Return:
- must-fix issues
- should-fix issues
- product or UX regressions
- security or maintainability risks
- false positives or uncertainties
Do not modify files.
" --output-format json --max-turns 8 > ".hermes/runs/$RUN_ID/claude-review.json"
```

Codex review prompt:

```bash
codex exec "Review the current diff for bugs, regressions, and missing tests. Do not modify files. Report only actionable findings."
```

Hermes reconciles both reviews into:

- Must fix.
- Should fix.
- False positive.
- Out of scope.
- Needs human decision.

### 8. Fix only scoped findings

If fixes are needed, Codex gets a narrow fix-pass prompt with the reconciled findings. It must not open new scope.

```bash
codex exec --full-auto "
Fix only the must-fix findings in .hermes/runs/$RUN_ID/reconciled-findings.md.
Do not introduce unrelated refactors.
Run the verification commands again.
Report changed files, commands run, and failures.
" 2>&1 | tee ".hermes/runs/$RUN_ID/fix-pass.log"
```

### 9. Final verification by Hermes

Hermes runs the real commands itself. The controller does not rely on Claude or Codex claims.

Examples:

```bash
git status --short
pytest -q
npm test
npm run build
gh pr view --json statusCheckRollup
```

The exact commands come from the brief and repo conventions.

### 10. Sign off or reject

Final response should include:

- Action taken.
- Exact target changed.
- Verification command used.
- Verification result.
- Review findings and disposition.
- Anything still broken or not verified.
- Artefact paths.

If verification is unavailable, say `NOT VERIFIED` and explain why.

## Hermes Tool Patterns

Use normal Hermes tools for small direct steps and `terminal` or `process` for CLI agents.

Claude print mode example:

```python
terminal(
  command="claude -p \"Review the brief and return risks only\" --output-format json --max-turns 5",
  workdir="/path/to/repo",
  timeout=180,
)
```

Codex background example:

```python
terminal(
  command="codex exec --full-auto \"Implement the signed-off brief in .hermes/runs/run-id/brief.md\"",
  workdir="/tmp/run-id",
  pty=True,
  background=True,
  notify_on_complete=True,
)
```

Poll bounded long work:

```python
process(action="poll", session_id="<session-id>")
process(action="log", session_id="<session-id>", limit=200)
process(action="wait", session_id="<session-id>", timeout=300)
```

Do not use background mode silently for bounded build work. Use `notify_on_complete=true` so the controller gets completion evidence.

## Public Setup Recipe

For a user who wants to reproduce the workflow:

1. Install Hermes Agent and enable terminal/process tools.
2. Install and authenticate Claude Code.
3. Install and authenticate Codex CLI.
4. Put this skill under `~/.hermes/skills/autonomous-ai-agents/hermes-claude-codex-workstream/SKILL.md`, or install it from the Hermes skill library if available.
5. Ask Hermes to load this skill before build orchestration.
6. Start with one low-risk repo and one small slice.
7. Keep external-agent artefacts in a run directory.
8. Require local verification before reporting completion.

Example user prompt:

```text
Use the hermes-claude-codex-workstream skill. I want to fix the settings page loading bug in this repo. Claude should plan/review, Codex should implement in an isolated worktree, and Hermes should run final verification before signoff.
```

## Guardrails

- Do not share secrets with Claude Code or Codex CLI unless the user explicitly approves and the tool genuinely needs them.
- Do not paste `.env`, auth files, token files, private keys, cookies, or OAuth JSON into prompts.
- Do not let Claude and Codex recursively delegate to each other.
- Do not run multiple implementation agents in the same worktree.
- Do not accept `looks good` as review evidence.
- Do not accept `tests passed` without the raw command output or an independent rerun.
- Do not commit, push, deploy, or publish unless the user asked for that side effect.
- Do not turn a review pass into a broad refactor pass.
- Do not hide failed commands. Failed commands are evidence.

## Common Pitfalls

1. **Starting with Codex before the brief is sharp.** Codex is fast enough to build the wrong thing quickly. Claude planning or human signoff comes first when direction is uncertain.

2. **No run directory.** Without artefacts, the controller cannot reconcile claims, reviews, diffs, and verification later.

3. **Using one giant prompt instead of a workflow.** The value is in staged delegation and independent verification, not prompt stuffing.

4. **Letting external agents own final judgement.** Hermes is the judge. Claude and Codex are engines.

5. **Skipping worktrees.** Isolated worktrees make rollback, review, and cleanup simple.

6. **Over-parallelising.** Parallel review is fine. Parallel implementation of the same slice is usually noise.

7. **Publishing internal examples.** For public skills, scrub private repo names, client data, user names, tokens, URLs, and operational transcript excerpts.

8. **Mistaking CLI auth for Hermes provider auth.** Verify each surface independently.

## Verification Checklist

Before saying the workstream is set up or a slice is complete:

- [ ] `claude --version` and Claude auth or `claude doctor` verified.
- [ ] `codex --version` and Codex auth path verified.
- [ ] Git repo and clean starting state verified.
- [ ] Run directory created.
- [ ] Brief written with acceptance criteria and verification commands.
- [ ] Claude plan or critique captured when needed.
- [ ] Human signoff captured when product/design direction changed.
- [ ] Codex ran in the intended worktree.
- [ ] Changed files and diff captured.
- [ ] Independent review pass completed or explicitly skipped with reason.
- [ ] Must-fix findings resolved or rejected with evidence.
- [ ] Hermes reran final verification commands directly.
- [ ] Final report includes artefact paths and remaining risks.
