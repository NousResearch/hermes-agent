# Prompt templates

All templates assume the Procedure's variables are set: `$REPO_ROOT`,
`$RUN_ID`, `$RUN_DIR` (absolute), `$WORKTREE` (absolute). Commands that act
on the implementation run from `"$WORKTREE"`; artefacts always go to
`"$RUN_DIR"` by absolute path.

## Claude plan and gap check (step 3)

```bash
claude -p "$(cat "$RUN_DIR/brief.md")

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
  > "$RUN_DIR/claude-plan.json"
```

Claude should not passively accept a weak brief. If acceptance criteria,
constraints, design direction, verification commands, or scope boundaries
are missing, it returns the gaps and Hermes resolves them before build.

## Codex build (step 5)

Run from `"$WORKTREE"`:

```bash
codex exec --full-auto "
You are implementing one signed-off slice.

Read this brief: $RUN_DIR/brief.md

Rules:
- Stay inside scope.
- Do not commit.
- Run the verification commands from the brief.
- If a command fails, keep the raw failure and explain the blocker.
- End with changed files, commands run, failures, and remaining risks.
" 2>&1 | tee "$RUN_DIR/codex-build.log"
```

## Claude review (step 7)

```bash
claude -p "
Review this implementation against the brief.

Brief:
$(cat "$RUN_DIR/brief.md")

Diff:
$(cat "$RUN_DIR/diff.patch")

Return:
- must-fix issues
- should-fix issues
- product or UX regressions
- security or maintainability risks
- false positives or uncertainties
Do not modify files.
" --output-format json --max-turns 8 > "$RUN_DIR/claude-review.json"
```

## Codex review (step 7)

Run from `"$WORKTREE"`:

```bash
codex exec "Review the current diff for bugs, regressions, and missing tests. Do not modify files. Report only actionable findings." \
  2>&1 | tee "$RUN_DIR/codex-review.log"
```

## Codex fix pass (step 8)

Run from `"$WORKTREE"`:

```bash
codex exec --full-auto "
Fix only the must-fix findings in $RUN_DIR/reconciled-findings.md.
Do not introduce unrelated refactors.
Run the verification commands again.
Report changed files, commands run, and failures.
" 2>&1 | tee "$RUN_DIR/fix-pass.log"
```
