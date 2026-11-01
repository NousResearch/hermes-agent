---
name: hermes-claude-codex-workstream
description: Drive briefs through Claude Code planning and Codex builds.
version: 1.1.0
author: Joel Brilliant (@joelbrilliant) with Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [hermes-agent, claude-code, codex, orchestration, build-workflow, verification]
    related_skills: [hermes-agent, claude-code, codex]
---

# Hermes Claude Codex Workstream Skill

Run a controlled build workstream where Hermes stays the controller: it
sharpens the brief, routes work, and verifies results; Claude Code plans,
designs, and reviews; Codex CLI implements. This skill defines the staged
loop and its artefact trail — it does not replace direct Hermes edits for
small tasks, and it never lets the external engines own final judgement.

## When to Use

- The user wants a Hermes-controlled workflow across Claude Code and Codex CLI.
- A rough product or engineering intent needs to become a buildable, verifiable slice.
- A feature, bug fix, PR review, or prototype should run through independent plan/build/review passes.

Do not use for: one-command edits Hermes can patch directly, tasks that
would require sharing secrets with external CLIs, or free-running
agent-to-agent chats without Hermes control.

## Prerequisites

- `terminal` and `process` tools enabled for the Hermes controller.
- Claude Code installed and authenticated: `claude --version`, `claude doctor`.
- Codex CLI installed and authenticated: `codex --version`.
- A git repository with a clean starting state.
- Load the `claude-code` and `codex` companion skills for exact CLI flags and PTY handling.

Auth is per-surface: Hermes provider auth, Claude auth, and Codex auth are
independent — verify each. Never print keys, tokens, or auth file contents.

## How to Run

```text
Human intent
→ Hermes sharpens the brief and routes
→ Claude Code plans, reviews the brief, pushes back on gaps
→ Codex CLI implements in an isolated worktree
→ Claude and Codex review independently
→ Hermes reconciles findings, reruns verification, signs off
```

Route Claude first when design, architecture, or risk needs pressure-testing.
Route Codex first only for narrow, fully specified implementation. Get human
signoff before build when product or design direction changes.

## Quick Reference

| Role | Owns |
|------|------|
| Hermes | Brief, scope, routing, run artefacts, reconciliation, final verification, report |
| Claude Code | Planning, design critique, gap detection, independent review |
| Codex CLI | Implementation, test loop, diffs, scoped fix passes |

Artefact layout (all paths absolute, under the repo the slice targets):

```text
$RUN_DIR/
  brief.md            claude-plan.json     codex-build.log
  changed-files.txt   diff.patch           claude-review.json
  codex-review.log    reconciled-findings.md
  fix-pass.log        verification.txt     signoff.md
```

Full prompt templates: [references/prompt-templates.md](references/prompt-templates.md).
Public setup recipe and `terminal`/`process` invocation patterns:
[references/public-setup.md](references/public-setup.md).

## Procedure

1. **Sharpen the brief.** It must state goal, non-goals, constraints, scope
   boundaries, acceptance criteria, verification commands, expected
   artefacts, and the signoff trigger. If success cannot be verified by
   commands or observable artefacts, the brief is not ready.

2. **Create the run directory** with absolute paths, before any worktree:

   ```bash
   REPO_ROOT="$(git rev-parse --show-toplevel)"
   RUN_ID="run-$(date +%Y%m%d-%H%M%S)-topic"
   RUN_DIR="$REPO_ROOT/.hermes/runs/$RUN_ID"
   mkdir -p "$RUN_DIR"
   ```

   If `.hermes/runs/` does not suit the repo, use another gitignored
   directory — but keep `RUN_DIR` absolute so every later step and every
   worktree can reference it unambiguously.

3. **Ask Claude for plan and gap check** (template in references). Save to
   `"$RUN_DIR/claude-plan.json"`. If Claude reports gaps, resolve them or
   get a human decision before any build starts.

4. **Create an isolated worktree** as a sibling of the repo:

   ```bash
   BASE_BRANCH="main"
   BRANCH="agent/$RUN_ID"
   WORKTREE="$(dirname "$REPO_ROOT")/$(basename "$REPO_ROOT")-wt-$RUN_ID"
   git worktree add -b "$BRANCH" "$WORKTREE" "$BASE_BRANCH"
   ```

   One implementation owner per worktree. If it is dirty before Codex
   starts, stop and resolve first.

5. **Send Codex the signed-off build prompt** (template in references), run
   from `"$WORKTREE"`, always referencing `"$RUN_DIR/brief.md"` by absolute
   path. Capture output to `"$RUN_DIR/codex-build.log"`. Codex implements
   the brief; if it finds the brief contradictory it stops and reports
   rather than inventing scope.

6. **Capture the evidence** from `"$WORKTREE"`:

   ```bash
   git status --short > "$RUN_DIR/changed-files.txt"
   git diff > "$RUN_DIR/diff.patch"
   ```

   No diff but a completion claim = failed run until explained.

7. **Run independent reviews.** Claude reviews the diff against the brief;
   Codex runs its own review pass (templates in references). Hermes
   reconciles into must-fix / should-fix / false positive / out of scope /
   needs human decision, saved to `"$RUN_DIR/reconciled-findings.md"`.

8. **Fix only scoped findings.** Codex gets a narrow fix-pass prompt
   pointing at the reconciled findings file; no new scope, no unrelated
   refactors.

9. **Verify directly.** Hermes reruns the brief's verification commands
   itself in `"$WORKTREE"` — agent claims are input, command output is
   proof. Save output to `"$RUN_DIR/verification.txt"`.

10. **Sign off or reject.** The final report states action, target,
    verification command and result, finding dispositions, artefact paths,
    and anything not verified (say `NOT VERIFIED` and why).

## Pitfalls

- Building before the brief is sharp — Codex builds the wrong thing fast.
- Relative artefact paths: a prompt that runs in the worktree cannot see a
  repo-relative `.hermes/runs/...`; always pass `"$RUN_DIR"` absolute.
- Letting external agents own final judgement; Hermes is the judge.
- Accepting "tests passed" without raw output or an independent rerun.
- Sharing `.env`, auth files, or tokens with either CLI.
- Letting Claude and Codex delegate to each other recursively.
- Turning a review pass into a broad refactor pass.
- Publishing internal examples without scrubbing repo names, client data,
  and transcript excerpts.

## Verification

- [ ] Both CLIs verified (`claude doctor`, `codex --version`), auth confirmed per surface.
- [ ] `RUN_DIR` created absolute; brief includes acceptance criteria and verification commands.
- [ ] Claude plan captured; human signoff captured when direction changed.
- [ ] Codex ran only in its worktree; diff and changed files captured.
- [ ] Independent reviews reconciled; must-fix findings resolved or rejected with evidence.
- [ ] Hermes reran verification commands directly; report includes artefact paths and remaining risks.
