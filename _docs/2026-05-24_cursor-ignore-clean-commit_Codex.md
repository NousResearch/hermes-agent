# 2026-05-24 Cursor Ignore Clean Commit

## Overview

Cleaned the Hermes Agent main worktree by moving local Cursor state out of git tracking, adding `.cursor/` to `.gitignore`, and removing temporary local-only files from the workspace.

## Background / Requirements

- User requested `.cursor` be added to `.gitignore`.
- User requested the worktree be cleaned, committed, and pushed.
- The worktree contained tracked Cursor hook state, an `AGENTS.md` operational-note update, and several untracked temporary helper files.

## Assumptions / Decisions

- `.cursor/` is local editor/runtime state and should not remain tracked.
- `AGENTS.md` contains durable local project guidance, so it belongs in the commit.
- `_docs` is ignored by default, so this implementation log must be force-added if it is part of the committed evidence.
- Untracked root files named `_tmp_*`, `_patch_*`, `_fix_env_bom.py`, `heartbeat_message.txt`, and `heartbeats.py` were treated as local temporary artifacts after inspecting their contents.

## Changed Files

- `.gitignore`
- `AGENTS.md`
- `.cursor/hooks/state/continual-learning.json` removed from git tracking
- `_docs/2026-05-24_cursor-ignore-clean-commit_Codex.md`

## Implementation Details

- Added `.cursor/` to `.gitignore`.
- Ran `git rm --cached -r -- .cursor` so the existing Cursor hook state remains local but is removed from the repository index.
- Removed local temporary files after verifying their resolved paths stayed inside the workspace.
- Kept unrelated ignored local runtime state untracked.

## Commands Run

```text
git status --short --branch
git diff -- AGENTS.md
git diff -- .cursor/hooks/state/continual-learning.json
git ls-files .cursor
git check-ignore -v .cursor/hooks/state/continual-learning-index.json .cursor/hooks/state/continual-learning.json
git rm --cached -r -- .cursor
git diff --check
```

## Test / Verification Results

- `git check-ignore -v` confirms both `.cursor/hooks/state/continual-learning-index.json` and `.cursor/hooks/state/continual-learning.json` match `.gitignore:8:.cursor/`.
- Temporary untracked root helper files were removed.
- `git diff --check` completed without whitespace errors.
- Final post-push clean status is recorded in the chat closeout.

## Residual Risks

- This change intentionally removes `.cursor` state from repository tracking. Any future project-level Cursor configuration that should be shared must be placed outside `.cursor/` or unignored explicitly.

## Recommended Next Actions

- Keep editor/runtime state local by default.
- If a shared Cursor rule file is needed later, add a narrow negated `.gitignore` rule for that exact file rather than tracking the whole `.cursor` tree.
