# Proving Test Failures Pre-Existing (Stash Technique)

Observed 2026-08-05 on `CLOE_RETRIEVAL_ANSWER_SUFFICIENT_NATIVE_TOOLS_PRESERVATION_REPAIR_V1`
(PR #701). A regression batch showed 2 failures in `gateway-readonly-conversation-router.test.js`
— a file that did NOT import the modified service. Needed to prove they were
pre-existing on the base SHA, not caused by the patch.

## Technique

```bash
cd <worktree>
# 1. Stash ONLY the tracked source change (untracked new test files stay put)
git stash push -m "patch-temp" -- <changed-source-file>

# 2. Confirm working tree is back at baseline for tracked files
git status --porcelain   # should show only untracked files

# 3. Run the failing test(s) on baseline — compare fail counts
node --test --test-concurrency=1 test/<failing-file>.test.js 2>&1 | grep -E "^# (tests|pass|fail)"
# e.g. # tests 11 / pass 9 / fail 2  → IDENTICAL to post-patch run

# 4. Restore the patch
git stash pop
git status --porcelain   # patch restored
```

## Deciding factors (when the stash is even needed)

- Check FIRST whether the failing test file imports the modified module:
  `grep -n "chat-completions-service\|buildNativeChatCompletions" test/<file>.test.js`.
  If it does not, the patch cannot causally affect it — but still run the stash
  comparison because the reviewer/operator will ask for proof.
- If the failing file DOES import the modified module, the stash comparison is
  mandatory to distinguish regression from pre-existing.
- Record BOTH numbers in the operator packet / PR body: post-patch batch (e.g.
  `200 pass / 2 fail`) AND baseline comparison (`9 pass / 2 fail` on base SHA).

## Pitfalls

- Stash only the tracked files you changed: `git stash push -- <paths>` scoped.
  A bare `git stash` also stashes nothing untracked by default but may grab
  unrelated dirty files from the shared canonical `.git` if this is a worktree.
- `git stash pop` must succeed before continuing — verify with `git status` that
  the patch is back and `git diff --stat` matches the pre-stash state.
- Do not `git stash --include-untracked` unless you also want the new test file
  out of the tree (usually you do NOT — you want the failing test to still exist
  so you can run it, but with the source restored; if the new test file itself is
  the one failing, you need it present).
- For worktrees sharing the canonical `.git`: `git status --porcelain` after stash
  will still show the canonical checkout's dirty files — scope to your directory.
- **Huge integration test files can OOM node's default heap on the 8 GB VPS** (observed
  2026-08-08: `readonly-operator-cli-client.test.js`, a 1300+ line file that imports the whole
  pipeline, dies with `FATAL ERROR: Ineffective mark-compacts near heap limit` even with
  `NODE_OPTIONS="--max-old-space-size=4096"`; it also takes 2+ minutes per attempt). Do NOT
  fight the big file: (1) prove the failure pre-existing by running the same file against the
  base worktree / control checkout — if it OOMs there too, the file itself is the problem, not
  your patch; (2) validate your acceptance-path change with an ISOLATED minimal test file that
  imports only the function you changed (e.g. `buildBrainAskResponse`) and mocks `fetchImpl` +
  the env vars the module reads from `process.env` (set `process.env.X` in the test, restore in
  `finally`). A tiny new test file runs in seconds and gives the RED→GREEN proof the big file
  cannot. The full-suite run still reports the big file as a pre-existing `not ok` — cite the
  isolated file + baseline comparison instead of blocking on the OOM file.
