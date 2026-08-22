# Git Verification & Workspace Pitfalls (CLOE persona parity mission, 2026-08-02)

Class-level git traps discovered while executing
CLOE_TELEGRAM_CANONICAL_PERSONA_CONTEXT_PARITY_REPAIR_V1. Apply to any LAH
mission that branches from a workspace clone, verifies file existence in a
tree, or checks production SHAs.

## 1. `git ls-tree <ref> -- <path>` is NOT an existence check

Symptom: `git ls-tree origin/main -- lah-openclaw-mvp/src/services/foo.js && echo PRESENT || echo ABSENT`
printed `PRESENT` for a file that was NOT in the tree — `git cat-file -e
origin/main:path` said ABSENT. `git ls-tree` exits 0 whenever the ref resolves,
even if the path filter matched nothing. It reports paths, it does not test
existence.

Fix: use `git cat-file -e <ref>:<path> && echo PRESENT || echo ABSENT` for
existence. Re-verify with `git show <ref>:<path> | head -30` when the finding
changes mission scope.

## 2. Re-run file inventory AFTER a rebase

Symptom: an initial `find src -name "cloe-persona*"` reported a file absent;
after rebasing the work branch onto the real origin/main, the same file was
present. The inventory ran against a stale base (branch created from a stale
`origin/main`).

Fix: after `git rebase origin-https/main`, re-check `git ls-files` /
`cat-file -e` for any file whose existence drives a decision. Never finalize
"file X does not exist" from a pre-rebase scan.

## 3. Workspace clone `origin` (plain SSH) can be stale vs `origin-https`

Symptom: in `/home/deploy/openclaw-runtime` (workspace clone), `origin` pointed
at main 53641a8 while `origin-https` (token remote) had the real main 30c7da3
containing production SHA 702c61e. `git checkout -b feat/... origin/main`
created a branch missing the canary commits and their files. Both remotes
appeared consistent locally because the refs were both stale in different ways.

Fix:
```bash
git fetch origin main        # plain SSH remote — may be stale
git fetch origin-https main  # token remote — authoritative for this workspace
git merge-base --is-ancestor <PROD_SHA> origin-https/main && echo "prod present"
git checkout -b feat/<mission> origin-https/main
```
Always branch from the remote that contains the production SHA. Verify with
`git merge-base --is-ancestor`.

## 4. Full-suite test globs in lah-openclaw-mvp

Use explicit globs (`test/*.test.js test/*.test.mjs test/cloe-persona-context-parity/*.test.js`).
Bare `node --test` scans `releases/` + `test/fixtures/runner-stdin-echo.mjs`
(hangs) and test runs regenerate tracked `test/reports/*.json` (restore before
commit).

## 5. Parity proof: canonical file hash, not the runtime system-prompt hash

The Center runtime's `systemPromptReport.hash` VARIES per session (bootstrap +
daily memory included) — it cannot prove Center↔Telegram persona parity. Use a
canonical hash over the persona file contents only:
`sha256(concat SOUL.md + IDENTITY.md + USER.md)` (e.g. aae8f175… for cloe-poc
2026-08-01). Both channels read the SAME workspace files → same canonical hash.
Expose `persona_hash` / `persona_version` / `persona_context_injected` in the
brain-ask trace (`buildBrainAskResponse` data) so parity is observable per
request (LOT 11-style alert: same user + same version + different hash = alert).

## 6. Canonical checkout `main` can be locally diverged from true origin/main
Symptom (CLOE_DASHBOARD_REPLY_SESSION_CONFLICT_REPAIR_V1 phase-2): the canonical
checkout at `/home/deploy/lah-stack-repos/openclaw-runtime` had local `main` at
f76c845 while the REAL remote `origin/main` was 101526b8 (the actual merged PR
#667). Committing the phase-2 work onto the local f76c845 produced a PR whose
diff was 109 files / 15,902 insertions — mostly unrelated pre-existing work.
The local `main` had its own history that was NOT an ancestor of origin/main.

Fix — always verify before committing when the mission references a known main SHA:
```bash
git fetch origin main
git rev-parse <local-main>^{tree} 2>/dev/null || echo "missing"
git rev-parse origin/main^{tree}
# tree SHAs differ → local main diverged from remote
git diff f76c845 origin/main -- lah-openclaw-mvp/scripts/session-accessor-patch.mjs | wc -l
# 0 lines → phase-1 file is content-identical, only history diverged
git checkout -B fix/<mission-branch> origin/main   # rebuild branch on TRUE remote main
git checkout <wip-sha> -- <mission-files>           # re-apply ONLY mission files
git commit -m "<TAG> ..." && git rev-parse HEAD
# verify: git log --oneline origin/main..HEAD | wc -l == 1, git diff origin/main..HEAD --stat shows ONLY mission files
```
A mission file can be content-identical between the stale local main and the
true remote main (0 diff lines) even though the branches are unrelated — check
the tree hashes, not just the file, and always base the branch on `origin/main`
after `git fetch`.

## 7. Fresh worktrees have NO node_modules — symlink the canonical checkout's

Symptom: after `git worktree add`, express-dependent tests fail with
`ERR_MODULE_NOT_FOUND: Cannot find package 'express' imported from
test/zone-monitor-orchestrator-routes.test.js` — looks like a test regression
but is purely an environment gap. Pure-ESM tests (no imports of installed
packages) pass fine, so the failure set is exactly the express/route tests.

Fix:
```bash
# The worktree shares .git with the canonical checkout but not node_modules.
ln -s /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/node_modules \
      /home/deploy/lah-stack-worktrees/<worktree>/lah-openclaw-mvp/node_modules
ls <worktree>/lah-openclaw-mvp/node_modules/express/package.json && echo LINK_OK
```
The symlink is not tracked by git and does not dirty the worktree. Detect
quickly: `ls -d <worktree>/lah-openclaw-mvp/node_modules` returns nothing on a
fresh worktree; `ls -d <canonical>/lah-openclaw-mvp/node_modules` confirms the
canonical has one to link. Always re-run the previously "failing" suite after
linking before classifying anything as a pre-existing failure (this exact
pattern hit on CLOE_PROFIT_TRUTH_AND_BOUNDED_AUTO_OPTIMIZER_V1: 2 zone tests
failed with MODULE_NOT_FOUND, then passed 66/67 after the link — the residual
1 failure was a genuine pre-existing cadence flake).
