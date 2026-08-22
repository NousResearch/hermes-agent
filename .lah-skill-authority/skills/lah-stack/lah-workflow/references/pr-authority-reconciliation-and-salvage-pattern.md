# PR Authority Reconciliation & Unique-Delta Salvage Pattern

Mission class: audit open PRs across LAH repos, classify authority deterministically,
close provably obsolete PRs WITHOUT merge, and for REVIEW_REQUIRED PRs decide whether
any unique semantic delta deserves minimal salvage into current canonical code.

Established by:
- LAH_OPEN_PR_AUTHORITY_RECONCILIATION_AND_SUPERSEDED_CLEANUP_V1 (classifier merged via openclaw-runtime PR #712, continuity #713)
- LAH_PR_UNIQUE_DELTA_SALVAGE_631_92_30_V1 (salvage via lah-stack-tools PR #105, continuity openclaw-runtime #714)

## Reusable capability

`npm run pr-factory:authority-audit` (lah-openclaw-mvp):
- `classifyOpenPrAuthority(pr, evidence)` — deterministic pure classifier
- `assertMutationSafe(pr, audit, current)` — pre-mutation gate with typed block reasons
  (`PR_NOT_OPEN`, `HEAD_SHA_DRIFT`, `PR_MERGED_CONCURRENTLY`, `CLASSIFICATION_NOT_CLOSE`)
- CLI is strictly read-only (`mutation_performed: false` always; no close/comment/merge surface)

## Authority classification model

Classifications (bounded set): `KEEP_ACTIVE / CLOSE_SUPERSEDED / CLOSE_DUPLICATE /
CLOSE_HISTORICAL_FIXTURE / REVIEW_REQUIRED / EXTERNAL_OUT_OF_SCOPE`.

Evidence rules:
- STRONG: explicit merged successor; later merged PR referencing old as superseded;
  continuity receipt naming later authority; engineering memory status SUPERSEDED;
  current-main containment of the same capability (blob-identical proof is strongest).
- WEAK (never sufficient alone): age, inactivity, behind main, merge conflicts,
  similar title, newer PR number, draft, failed/absent CI, no comments.
- Unique required delta present → `REVIEW_REQUIRED` (no-loss rule). Never close it.

## Salvage decision framework (for REVIEW_REQUIRED PRs)

1. Semantic decomposition — NOT line counts. Reduce the patch into capabilities:
   `{ capability, files, purpose, unique_vs_main, still_required, canonical_equivalent, decision }`.
   Ignore: formatting-only diffs, stale test fixtures, outdated doc wording, obsolete wiring,
   APIs replaced by stronger canonical abstractions, copied subtrees whose authoritative
   source exists elsewhere.
2. Canonical-equivalence check: can current main already answer the same questions or
   provide a stronger abstraction? (Example from #631: a static parallel creative inventory
   was rejected because stable-block registry + capability-manifest-builder + capability
   graph + existing conversation routing already provide that authority.)
3. Salvage value gate: score 0–2 per dimension
   `CURRENT_NEED, ARCHITECTURAL_FIT, SAFETY_VALUE, OPERATIONAL_VALUE, DUPLICATION_COST, MAINTENANCE_COST`.
   Salvage ONLY when `CURRENT_NEED >= 1` AND `ARCHITECTURAL_FIT >= 1` and benefit clearly
   outweighs duplication/maintenance cost. A technically valid but low-value stale feature
   may be intentionally discarded — document why.
4. Minimal salvage: find the exact extension point, implement only that piece, never port
   the whole historical PR, never create a second engine / second registry / second planner.

## Gap detection: wired-but-dead pipeline slot

The best salvage signal is a pipeline that CONSUMES data the producer never fills:
- `collectCleanupInventory` iterated `adapter.worktrees` expecting a `.temporary` flag, but
  `createVpsInventoryAdapter` returned `worktrees: []` hardcoded → the whole
  `temporary_git_worktrees` → `git:remove_clean_temporary_worktree` (HUMAN_GATE) category was
  dead code. Filling the slot = real, minimal salvage.
- Also grep for TODO stubs that return OK falsely (e.g. git-health.mjs worktree/untracked
  probes). A declared-but-empty probe is false assurance, not a working check.

## Stale subtree / duplicate-copy detection

For a subtree suspected of being a stale copy (Case 1 closure):
- Compare blob SHAs: `git rev-parse <branch>:<path>` vs `git rev-parse origin/<default>:<path>`
  AND vs the canonical merge commit's tree (`git rev-parse <merge_sha>:<path>`).
- Blob-identical at merge time + canonical evolved AHEAD → stale copy → `CLOSE_SUPERSEDED`.
- Check the direction of any difference (canonical richer? base richer?) with `diff`.
- Regenerable generated artifacts (token-budget.json, memory snapshots) are NOT code deltas;
  newer/wider base records prove the branch is historical.

## Closure workflow (never merge the old PR)

1. Lock original head SHAs at audit time (persist in evidence).
2. Immediately before each closure: re-fetch GitHub — require `state==OPEN`,
   `head_sha==locked_original`, no `mergedAt`/`mergeCommit`. Any mismatch →
   `BLOCKED_BY_STATE_DRIFT`, do not close.
3. Post a factual comment — SALVAGED template (names the replacement PR + merge SHA +
   salvaged capability) or NO_SALVAGE template (names the canonical authority). Never claim
   replacement unless proven.
4. Close WITHOUT merge (`gh pr close` — no merge flags).
5. Re-fetch post-close: require `state=CLOSED`, `merged=false`, `mergeCommit=null`.
   Never infer success from the mutation request alone.

## Pitfalls (all hit in real sessions)

- `gh pr view` has NO `merged` or `repository` field — use `mergedAt` / `mergeCommit`
  (`mergeCommit` may be `null` for unmerged), and the `--repo` flag for the repo.
  `gh pr list` uses `isDraft`, not `draft`.
- For old PRs whose base is ancient, use the 3-dot diff (`origin/main...HEAD`). The direct
  2-dot diff shows tens of thousands of deletions from the stale base (pure noise).
- Adapter `command()` helpers may ignore `config.cwd` — use `git -C <repo> ...` explicitly.
- Engineering Memory `finalize-mission-card.mjs` reads `tests` and `runtime_evidence` at the
  REPORT ROOT, not under `validation`. A SUCCESS card without them fails validation and
  blocks the index build.
- Test fixtures for collectors that guard with `existsSync(repo)` must use REAL tmp dirs
  (`mkdtempSync`); fake absolute paths are silently skipped and assertions never run.
- ci-governance dead on main (GitHub Actions billing: "job was not acquired by Runner of
  type hosted" / "spending limit") → document the pre-existing failure on main, then merge
  implementation PRs with `gh pr merge --admin --merge --match-head-commit <sha>`
  (LOCAL_CI_VERIFIED doctrine, established PRs #710/#711/#104).

## Result artifacts

Persist under `/home/deploy/evidence/<mission>/`:
- `initial-authority.json` — default-branch SHAs + full PR metadata (state, head SHA,
  commits, changed files, mergeability)
- `decision-matrix.json` — immutable per-PR decision + evidence + salvage value gate
- `mission-report.json` — feeds the engineering memory card
- continuity JSON in `docs/mcporter/` + engineering memory card via
  `finalize-mission-card.mjs` (validate + rebuild index)
