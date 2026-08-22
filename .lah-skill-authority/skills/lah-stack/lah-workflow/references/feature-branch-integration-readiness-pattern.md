# Feature Branch → Main Integration Readiness Audit

**Created during:** `CLOE_FEATURE_BRANCH_TO_MAIN_INTEGRATION_READINESS_V1`

Use when a mission requires determining whether a feature branch can be merged into `origin/main`. This is a **READ_ONLY_AUDIT** — no merge, no deploy, no mutation outside temporary test worktrees.

## Trigger

- User asks: "is feature X ready to merge?"
- A feature branch has accumulated multiple commits since fork
- You need to prove: what's on the branch, what tests pass, what would break, what would activate

## Prerequisites

- Canonical repo path
- Feature worktree (or clone) path
- Known feature branch name and expected HEAD SHA
- Known merge base

## Required data per phase

| Phase | What to collect | Key commands |
|-------|----------------|--------------|
| Phase 0 — Baseline | origin/main SHA, feature HEAD, merge base, ahead/behind, worktree cleanliness | `git fetch --prune origin`, `git rev-parse origin/main`, `git merge-base`, `git rev-list --left-right --count`, `git status --short` |
| Phase 1 — Commit lineage | Commit graph, parentage, files-per-commit, classification table | `git log --oneline --parents origin/main..HEAD`, `git diff-tree --no-commit-id --name-status -r <sha>` |
| Phase 2 — File inventory | Full diff grouped by component category | `git diff --stat origin/main..HEAD`, `git diff --name-status` |
| Phase 3 — Wiring proof | Production call graph from entry point to new feature | Read source: entry point → route → service → orchestrator; trace imports; identify fail-closed paths |
| Phase 4 — Classifier/intent comparison | Diff of classifier patterns, verify each required pattern | `git diff origin/main..HEAD -- <classifier>` |
| Phase 5 — Merge simulation | Merge conflicts or clean auto-merge | `git worktree add --detach /tmp/sim origin/main; cd /tmp/sim; git merge --no-commit --no-ff <feature>` |
| Phase 6 — Dependencies | package.json/lock diff | `git diff origin/main..HEAD -- package.json` |
| Phases 7-9 — Regression tests | Test-by-test pass/fail for campaign, DT, combined | `node --test --test-concurrency=1 <test-files>` |
| Phase 10 — Activation audit | What activates on merge vs deploy | Read source for auto-start, import chains, env guards |
| Phase 11 — Duplicate authority | Compare new worker/tool against existing lah-stack-tools equivalents | `find` both directories, compare file count and capabilities |
| Phase 12 — Gap preservation | Verify incomplete pipelines remain explicit | Trace: Creative Factory → DT job → Mac → VPS asset → import |
| Phase 13 — Memory safety | Check writeCampaignMemory calls, idempotency | `grep -r "writeCampaignMemory" src/` |
| Phase 14 — Security review | Path traversal, exec, auth, env exposure | `git diff origin/main..HEAD -- src/middleware/auth.js`, grep for `child_process` in new files |
| Phase 15 — Integration strategy | Recommend full merge, cherry-pick, or block | Based on conflict analysis + test results + activation audit |
| Phase 16 — PR plan | Base, head, title, risk statement, deployment prohibition | Write as text, do not open PR |

## Temporary worktree merge simulation

Use this exact pattern to avoid contaminating real worktrees:

```bash
TMP_DIR="/tmp/<unique-mission-name>"
git -C <canonical-repo> worktree add --detach "$TMP_DIR" origin/main
cd "$TMP_DIR"
git merge --no-commit --no-ff <feature-branch>
# Inspect conflicts
git diff --name-only --diff-filter=U 2>/dev/null || echo "No conflicts"
# Cleanup
git -C <canonical-repo> worktree remove --force "$TMP_DIR"
```

## Known pitfalls

| Trap | Fix |
|------|-----|
| **Temp worktree not deletable after merge** | Merge leaves modified files. Use `--force` on `worktree remove`. |
| **origin/main not refetched** | Always `git fetch --prune origin` first — main may have moved since last session. |
| **Test concurrency collisions** | Shared `data/` directory. Always use `--test-concurrency=1`. |
| **HTTP route proof tests timeout in group** | Large route-proof tests (1040+ lines) create real Express servers. Run them individually or with sufficient timeout (120s+). |
| **Counting feature branch as ahead of stale main** | The ahead/behind count is only meaningful if origin/main was just fetched. Fetch first. |
| **Worktree status contaminated by canonical checkout dirt** | Worktrees share `.git`. Use `git diff <base> HEAD --name-only` to see only the worktree's own changes, not `git status`. |
